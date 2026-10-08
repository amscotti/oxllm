//! Cooldown tracking for failed deployments.

use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// A per-status failure count with a TTL window. Each counted failure
/// re-arms the window (the reference re-`set_cache`s the count with
/// `ttl=cooldown_time` on every tolerated failure,
/// `router_utils/cooldown_handlers.py:589`), so a sustained
/// slow-drip failure stream — one failure per sub-window interval — keeps
/// the count alive and monotonically growing until it crosses
/// `allowed_fails + 1`. A success clears the counts; a failure arriving
/// after the window elapses starts a fresh window at count 1.
#[derive(Debug, Clone, Copy)]
struct FailureEntry {
    count: u32,
    expires_at: Instant,
}

/// A timestamped outcome sample for the sliding-window success/failure
/// ratio used by percentage-based cooldown. Entries older than the window
/// are pruned on access.
#[derive(Debug, Clone, Copy)]
enum Outcome {
    Success,
    Failure,
}

/// One timestamped outcome in the sliding window.
#[derive(Debug, Clone, Copy)]
struct OutcomeSample {
    ts: Instant,
    outcome: Outcome,
}

/// The sliding window length (60 seconds) for percentage-based cooldown.
pub const PERCENTAGE_WINDOW: Duration = Duration::from_secs(60);

/// Upper bound on retained outcome samples per deployment key. The window
/// is also bounded by time ([`PERCENTAGE_WINDOW`]); the count cap keeps
/// both memory and per-record CPU O(cap) even under sustained high RPS, so
/// a hot deployment cannot pin a core scanning hundreds of thousands of
/// samples. At ~1k RPS the 60s window would otherwise grow to ~60k entries
/// scanned on every record and every count.
pub const MAX_OUTCOME_SAMPLES: usize = 1024;

/// Ceiling on any cooldown/expiry window applied to an `Instant`.
/// [`Instant::checked_add`] returns `None` for absurd durations (e.g. a
/// `cooldown_time_secs` config typo of `u64::MAX`), so durations are clamped
/// to one year instead of letting `Instant::now() + duration` panic. One year
/// is far beyond any realistic cooldown while keeping `now + window` well
/// within the `Instant` range.
const MAX_WINDOW: Duration = Duration::from_secs(86_400 * 365);

/// Clamp a cooldown/expiry window so `Instant::now() + window` cannot
/// overflow. Durations at or under the ceiling pass through unchanged.
fn clamp_window(duration: Duration) -> Duration {
    if duration > MAX_WINDOW {
        MAX_WINDOW
    } else {
        duration
    }
}

/// Tracks which deployments are in cooldown (temporarily excluded from routing).
///
/// When a deployment fails with a retryable error (5xx, 429, timeout, network),
/// it is marked as cooling for `cooldown_time_secs` seconds. During that window,
/// the router skips it when selecting deployments.
///
/// Also tracks per-status failure counts per deployment for the allowed-fails
/// policy: the host records each failed attempt with [`Self::record_failure`]
/// and cools the deployment once the count reaches the configured threshold.
/// Failure counts share the cooldown window as their TTL — a deployment that
/// stops failing sees its counts expire after the window elapses.
///
/// Additionally tracks a sliding 60-second window of success/failure outcomes
/// per deployment for percentage-based cooldown: once the failure ratio
/// exceeds the configured threshold (and the sample size is large enough),
/// the deployment is cooled.
///
/// This is an in-memory implementation using `std::sync::Mutex` (no tokio).
/// Future Redis-backed implementations will live in `oxllm-cache-redis`
///.
#[derive(Debug)]
pub struct CooldownStore {
    entries: Mutex<HashMap<String, Instant>>,
    failure_counts: Mutex<HashMap<String, HashMap<u16, FailureEntry>>>,
    /// Per-deployment sliding-window outcome samples for percentage-based
    /// cooldown. Bounded by both time and [`MAX_OUTCOME_SAMPLES`] entries;
    /// pruned on every access to bound memory and per-record CPU.
    outcomes: Mutex<HashMap<String, VecDeque<OutcomeSample>>>,
    /// Last-observed deployment count per model group, for the
    /// single-deployment safety of percentage-based cooldown. Updated by
    /// the router on each `pick_deployment` call.
    group_sizes: Mutex<HashMap<String, usize>>,
}

impl Default for CooldownStore {
    fn default() -> Self {
        Self::new()
    }
}

impl CooldownStore {
    /// Create a new empty cooldown store.
    pub fn new() -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
            failure_counts: Mutex::new(HashMap::new()),
            outcomes: Mutex::new(HashMap::new()),
            group_sizes: Mutex::new(HashMap::new()),
        }
    }

    /// Mark a deployment as cooling for the given duration.
    ///
    /// The duration is clamped to `MAX_WINDOW` so a pathological config
    /// value (`cooldown_time_secs: u64::MAX`) cannot overflow the `Instant`
    /// arithmetic and panic the caller's task.
    pub fn mark(&self, key: &str, duration: Duration) {
        let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        entries.insert(key.to_string(), Instant::now() + clamp_window(duration));
    }

    /// Check if a deployment is currently cooling.
    ///
    /// Expired entries are lazily removed on access.
    pub fn is_cooling(&self, key: &str) -> bool {
        let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        match entries.get(key).copied() {
            Some(expiry) => {
                if Instant::now() < expiry {
                    true
                } else {
                    entries.remove(key);
                    false
                }
            }
            None => false,
        }
    }

    /// Record a failed attempt for (`key`, `status`) and return the failure
    /// count within the current window. Each counted failure re-arms the
    /// window (`expires_at = now + window`, sliding TTL semantics — see
    /// `FailureEntry`); once the window elapses with no intervening
    /// failure, the next failure starts a fresh window at count 1.
    ///
    /// The count saturates at `u32::MAX` (matching the in-flight counter's
    /// `saturating_add` in `stores.rs`) rather than panicking on overflow — a
    /// cooldown window is bounded by the one-year `clamp_window` cap, so this
    /// ceiling is unreachable in practice, but saturating arithmetic keeps the
    /// no-panic contract for upstream-driven inputs.
    pub fn record_failure(&self, key: &str, status: u16, window: Duration) -> u32 {
        // A zero window would expire the entry instantly (`now >=
        // expires_at` on the very next call), so the count could never
        // accumulate past 1 and the allowed-fails policy would be
        // silently disabled — clamp it to a 1-second floor. (Non-zero
        // sub-second windows still accumulate correctly and are left
        // alone so tests can exercise expiry without real-time waits.)
        let window = if window.is_zero() {
            Duration::from_secs(1)
        } else {
            clamp_window(window)
        };
        let mut counts = self
            .failure_counts
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let per_status = counts.entry(key.to_string()).or_default();
        let now = Instant::now();
        let entry = per_status.entry(status).or_insert(FailureEntry {
            count: 0,
            expires_at: now + window,
        });
        if now >= entry.expires_at {
            *entry = FailureEntry {
                count: 0,
                expires_at: now + window,
            };
        }
        entry.count = entry.count.saturating_add(1);
        entry.expires_at = now + window;
        entry.count
    }

    /// Current failure count for (`key`, `status`) — 0 when absent or expired.
    ///
    /// Expired entries are lazily removed on access.
    pub fn failure_count(&self, key: &str, status: u16) -> u32 {
        let mut counts = self
            .failure_counts
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let Some(per_status) = counts.get_mut(key) else {
            return 0;
        };
        match per_status.get(&status) {
            Some(entry) if Instant::now() < entry.expires_at => entry.count,
            Some(_) => {
                per_status.remove(&status);
                if per_status.is_empty() {
                    counts.remove(key);
                }
                0
            }
            None => 0,
        }
    }

    /// Clear the failure count for (`key`, `status`).
    pub fn clear_failure_count(&self, key: &str, status: u16) {
        let mut counts = self
            .failure_counts
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if let Some(per_status) = counts.get_mut(key) {
            per_status.remove(&status);
            if per_status.is_empty() {
                counts.remove(key);
            }
        }
    }

    /// Clear all failure counts for a deployment key (called on success: a
    /// healthy response resets the deployment's failure history).
    pub fn clear_failure_counts(&self, key: &str) {
        let mut counts = self
            .failure_counts
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        counts.remove(key);
    }

    /// Record a success outcome for a deployment in the sliding window used
    /// by percentage-based cooldown. Successes dilute the failure ratio.
    pub fn record_success_outcome(&self, key: &str) {
        self.record_outcome(key, Outcome::Success);
    }

    /// Record a failure outcome for a deployment in the sliding window used
    /// by percentage-based cooldown, and return the current `(failures,
    /// total)` counts within the window so the caller can evaluate the
    /// threshold in one pass.
    pub fn record_failure_outcome(&self, key: &str) -> (u32, u32) {
        self.record_outcome(key, Outcome::Failure);
        self.outcome_counts(key)
    }

    fn record_outcome(&self, key: &str, outcome: Outcome) {
        let mut outcomes = self.outcomes.lock().unwrap_or_else(|e| e.into_inner());
        let samples = outcomes.entry(key.to_string()).or_default();
        samples.push_back(OutcomeSample {
            ts: Instant::now(),
            outcome,
        });
        Self::prune(samples);
    }

    /// Current `(failures, total)` counts for `key` within the sliding
    /// window. Both are 0 when no outcomes have been recorded.
    pub fn outcome_counts(&self, key: &str) -> (u32, u32) {
        let mut outcomes = self.outcomes.lock().unwrap_or_else(|e| e.into_inner());
        let Some(samples) = outcomes.get_mut(key) else {
            return (0, 0);
        };
        Self::prune(samples);
        if samples.is_empty() {
            // Opportunistic eviction: a key whose samples all aged out (e.g.
            // a deployment that stopped receiving traffic) is dropped so the
            // map does not accumulate empty windows across config reloads
            // that remove deployments.
            outcomes.remove(key);
            return (0, 0);
        }
        let total = samples.len() as u32;
        let failures = samples
            .iter()
            .filter(|s| matches!(s.outcome, Outcome::Failure))
            .count() as u32;
        (failures, total)
    }

    /// Drop samples older than the sliding window and trim to the count cap
    /// ([`MAX_OUTCOME_SAMPLES`]) by removing the oldest entries. Bounding by
    /// count keeps both memory and per-record CPU O(cap) regardless of RPS.
    fn prune(samples: &mut VecDeque<OutcomeSample>) {
        // Clock underflow (checked_sub returns None) is extremely rare; in
        // that case we keep all samples rather than dropping data.
        if let Some(c) = Instant::now().checked_sub(PERCENTAGE_WINDOW) {
            samples.retain(|s| s.ts >= c);
        }
        while samples.len() > MAX_OUTCOME_SAMPLES {
            samples.pop_front();
        }
    }

    /// Clear the sliding-window outcomes for a deployment key (called on
    /// full reset of failure history).
    pub fn clear_outcomes(&self, key: &str) {
        let mut outcomes = self.outcomes.lock().unwrap_or_else(|e| e.into_inner());
        outcomes.remove(key);
    }

    /// Record the observed deployment count for a model group. Used by the
    /// router's `pick_deployment` so percentage-based cooldown can apply the
    /// single-deployment safety (never cool a solo deployment).
    pub fn set_group_size(&self, model: &str, size: usize) {
        let mut sizes = self.group_sizes.lock().unwrap_or_else(|e| e.into_inner());
        sizes.insert(model.to_string(), size);
    }

    /// The last-observed deployment count for a model group. Returns `0`
    /// when unknown (treated as "no safety guarantee" — the percentage
    /// check applies).
    pub fn group_size_for(&self, model: &str) -> usize {
        let sizes = self.group_sizes.lock().unwrap_or_else(|e| e.into_inner());
        sizes.get(model).copied().unwrap_or(0)
    }

    /// Remove all cooldown entries and failure counts.
    pub fn clear(&self) {
        let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        entries.clear();
        let mut counts = self
            .failure_counts
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        counts.clear();
        let mut outcomes = self.outcomes.lock().unwrap_or_else(|e| e.into_inner());
        outcomes.clear();
        let mut sizes = self.group_sizes.lock().unwrap_or_else(|e| e.into_inner());
        sizes.clear();
    }

    /// Drop cooldown entries, per-status failure counts, and sliding-window
    /// outcomes for keys not in `known`. Called at config hot-reload (see
    /// [`crate::router::Router::prune_stale_store_keys`]) so deployments
    /// removed from the model list do not leak their cooldown state forever.
    pub fn retain_keys(&self, known: &std::collections::HashSet<String>) {
        let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        entries.retain(|key, _| known.contains(key));
        let mut counts = self
            .failure_counts
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        counts.retain(|key, _| known.contains(key));
        let mut outcomes = self.outcomes.lock().unwrap_or_else(|e| e.into_inner());
        outcomes.retain(|key, _| known.contains(key));
    }

    /// Drop last-observed model-group sizes for model names not in
    /// `known_models`. [`Self::set_group_size`] runs on every
    /// [`crate::router::Router::pick_deployment`], so entries for surviving
    /// models re-populate naturally; this sweep (paired with
    /// [`Self::retain_keys`] at config hot-reload) stops removed or renamed
    /// model names from leaking their size entries forever.
    pub fn retain_group_sizes(&self, known_models: &std::collections::HashSet<String>) {
        let mut sizes = self.group_sizes.lock().unwrap_or_else(|e| e.into_inner());
        sizes.retain(|model, _| known_models.contains(model));
    }

    /// Number of tracked entries (including not-yet-lazily-removed expired ones).
    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.entries.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// Whether there are no tracked entries.
    #[cfg(test)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of keys with a retained outcome window (test/inspection only).
    #[cfg(test)]
    pub fn outcome_key_count(&self) -> usize {
        self.outcomes
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .len()
    }

    /// Number of tracked model-group sizes (test/inspection only).
    #[cfg(test)]
    pub fn group_size_key_count(&self) -> usize {
        self.group_sizes
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mark_and_check() {
        let store = CooldownStore::new();
        store.mark("dep1", Duration::from_secs(60));
        assert!(store.is_cooling("dep1"));
        assert!(!store.is_cooling("dep2"));
    }

    #[test]
    fn zero_duration_expires_immediately() {
        let store = CooldownStore::new();
        store.mark("dep1", Duration::ZERO);
        assert!(!store.is_cooling("dep1"));
    }

    #[test]
    fn expired_entry_is_removed_on_access() {
        let store = CooldownStore::new();
        store.mark("dep1", Duration::from_millis(1));
        std::thread::sleep(Duration::from_millis(10));
        assert!(!store.is_cooling("dep1"));
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn clear_removes_all() {
        let store = CooldownStore::new();
        store.mark("dep1", Duration::from_secs(60));
        store.mark("dep2", Duration::from_secs(60));
        assert!(store.is_cooling("dep1"));
        store.clear();
        assert!(!store.is_cooling("dep1"));
        assert!(!store.is_cooling("dep2"));
    }

    #[test]
    fn re_mark_extends_cooldown() {
        let store = CooldownStore::new();
        store.mark("dep1", Duration::from_secs(1));
        store.mark("dep1", Duration::from_secs(60));
        std::thread::sleep(Duration::from_millis(10));
        assert!(store.is_cooling("dep1"));
    }

    #[test]
    fn record_failure_counts_per_status() {
        let store = CooldownStore::new();
        let window = Duration::from_secs(60);
        assert_eq!(store.record_failure("dep1", 429, window), 1);
        assert_eq!(store.record_failure("dep1", 429, window), 2);
        assert_eq!(store.record_failure("dep1", 500, window), 1);
        assert_eq!(store.failure_count("dep1", 429), 2);
        assert_eq!(store.failure_count("dep1", 500), 1);
        assert_eq!(store.failure_count("dep2", 429), 0);
    }

    #[test]
    fn record_failure_zero_window_still_accumulates() {
        // A zero window is clamped to the 1-second floor: without the
        // clamp the entry would expire instantly and the count could never
        // pass 1, silently disabling the allowed-fails policy.
        let store = CooldownStore::new();
        assert_eq!(store.record_failure("dep1", 500, Duration::ZERO), 1);
        assert_eq!(
            store.record_failure("dep1", 500, Duration::ZERO),
            2,
            "zero window must be clamped so failures accumulate"
        );
        assert_eq!(store.failure_count("dep1", 500), 2);
    }

    #[test]
    fn failure_count_unknown_key_is_zero() {
        let store = CooldownStore::new();
        assert_eq!(store.failure_count("nope", 500), 0);
    }

    #[test]
    fn failure_counts_expire_after_window() {
        let store = CooldownStore::new();
        let window = Duration::from_millis(1);
        assert_eq!(store.record_failure("dep1", 500, window), 1);
        std::thread::sleep(Duration::from_millis(10));
        assert_eq!(
            store.failure_count("dep1", 500),
            0,
            "expired failure counts must read as zero"
        );
        // After expiry the next failure starts a fresh window at count 1.
        assert_eq!(
            store.record_failure("dep1", 500, Duration::from_secs(60)),
            1
        );
    }

    #[test]
    fn record_failure_rearms_window_on_each_failure() {
        // Sliding-TTL parity (`cooldown_handlers.py:589`: every tolerated
        // failure re-`set_cache`s the count with `ttl=cooldown_time`): a
        // sustained slow-drip failure stream — one failure per sub-window
        // interval — must keep the count alive and growing instead of
        // resetting at the original window boundary. Failures spaced well
        // inside a 500ms window; the last one lands past the FIRST
        // failure's expiry, which the pre-parity fixed window would have
        // treated as a fresh window (count 1).
        let store = CooldownStore::new();
        let window = Duration::from_millis(500);
        let mut count = 0;
        for _ in 0..6 {
            count = store.record_failure("dep1", 500, window);
            std::thread::sleep(Duration::from_millis(100));
        }
        assert_eq!(
            count, 6,
            "sub-window failure spacing must re-arm the window (LiteLLM TTL semantics)"
        );
    }

    #[test]
    fn clear_failure_count_removes_only_that_status() {
        let store = CooldownStore::new();
        let window = Duration::from_secs(60);
        store.record_failure("dep1", 429, window);
        store.record_failure("dep1", 500, window);
        store.clear_failure_count("dep1", 429);
        assert_eq!(store.failure_count("dep1", 429), 0);
        assert_eq!(store.failure_count("dep1", 500), 1);
    }

    #[test]
    fn clear_failure_counts_removes_all_statuses() {
        let store = CooldownStore::new();
        let window = Duration::from_secs(60);
        store.record_failure("dep1", 429, window);
        store.record_failure("dep1", 500, window);
        store.clear_failure_counts("dep1");
        assert_eq!(store.failure_count("dep1", 429), 0);
        assert_eq!(store.failure_count("dep1", 500), 0);
    }

    #[test]
    fn clear_removes_failure_counts_too() {
        let store = CooldownStore::new();
        store.record_failure("dep1", 500, Duration::from_secs(60));
        store.clear();
        assert_eq!(store.failure_count("dep1", 500), 0);
    }

    #[test]
    fn outcome_window_is_bounded_by_count_cap() {
        // Under sustained high RPS the per-key window must stay bounded by
        // the count cap, not grow to RPS×60s entries. Recording far past the
        // cap must leave exactly MAX_OUTCOME_SAMPLES entries and complete
        // quickly: each record is O(cap) (retain over ≤cap + pop_front), so
        // the whole loop is O(N×cap) — whereas the pre-cap code was O(N²)
        // in RPS (the window grew unbounded within 60s). The budget below
        // is generous for debug builds yet an O(N²) implementation would
        // blow past it by orders of magnitude.
        let store = CooldownStore::new();
        let n = 50_000;
        let start = Instant::now();
        for _ in 0..n {
            let _ = store.record_failure_outcome("hot-dep");
        }
        let elapsed = start.elapsed();
        let (_, total) = store.outcome_counts("hot-dep");
        assert_eq!(
            total, MAX_OUTCOME_SAMPLES as u32,
            "window must be trimmed to the count cap"
        );
        assert!(
            elapsed.as_secs() < 15,
            "recording {n} outcomes must stay O(N×cap), took {elapsed:?}"
        );
    }

    #[test]
    fn outcome_window_ratio_stays_approximately_correct_under_cap() {
        // The count cap keeps the approximate failure ratio: a steady ~29%
        // failure stream must report a ratio near 2/7 once saturated, not
        // be distorted by the trim (oldest samples drop, ratio holds).
        let store = CooldownStore::new();
        for _ in 0..(MAX_OUTCOME_SAMPLES * 4) {
            let _ = store.record_failure_outcome("dep");
            store.record_success_outcome("dep");
            store.record_success_outcome("dep");
            let _ = store.record_failure_outcome("dep");
            store.record_success_outcome("dep");
            store.record_success_outcome("dep");
            store.record_success_outcome("dep");
        }
        let (failures, total) = store.outcome_counts("dep");
        assert_eq!(total, MAX_OUTCOME_SAMPLES as u32);
        let ratio = failures as f64 / total as f64;
        assert!(
            (0.20..=0.30).contains(&ratio),
            "trim must preserve the approximate ratio (~0.286), got {ratio}"
        );
    }

    #[test]
    fn outcome_clear_removes_window_key() {
        // clear_outcomes drops the key entirely (the opportunistic empty-
        // window eviction path runs on count, but exercising it requires
        // waiting out the 60s window; clear is the deterministic equivalent
        // and confirms the map does not retain emptied keys).
        let store = CooldownStore::new();
        let _ = store.record_failure_outcome("dep");
        assert_eq!(store.outcome_key_count(), 1);
        store.clear_outcomes("dep");
        assert_eq!(
            store.outcome_key_count(),
            0,
            "clear_outcomes must drop the key from the map"
        );
    }

    #[test]
    fn retain_keys_drops_state_for_removed_deployments() {
        let store = CooldownStore::new();
        let window = Duration::from_secs(60);
        store.mark("keep:0", window);
        store.mark("drop:1", window);
        store.record_failure("drop:1", 500, window);
        let _ = store.record_failure_outcome("drop:1");
        assert_eq!(store.len(), 2);
        assert_eq!(store.outcome_key_count(), 1);

        let known: std::collections::HashSet<String> = ["keep:0".to_string()].into_iter().collect();
        store.retain_keys(&known);

        assert!(store.is_cooling("keep:0"));
        assert!(!store.is_cooling("drop:1"));
        assert_eq!(store.failure_count("drop:1", 500), 0);
        assert_eq!(
            store.outcome_key_count(),
            0,
            "outcomes for dropped keys are swept"
        );
    }

    #[test]
    fn retain_group_sizes_drops_stale_model_names() {
        // `set_group_size` runs on every pick; the hot-reload sweep must drop
        // entries for model names removed from the list instead of leaking
        // them monotonically across rolling config updates.
        let store = CooldownStore::new();
        store.set_group_size("gpt-4o", 2);
        store.set_group_size("claude", 1);
        store.set_group_size("retired-model", 3);
        assert_eq!(store.group_size_key_count(), 3);

        let known_models: std::collections::HashSet<String> =
            ["gpt-4o".to_string(), "claude".to_string()]
                .into_iter()
                .collect();
        store.retain_group_sizes(&known_models);

        assert_eq!(store.group_size_for("gpt-4o"), 2);
        assert_eq!(store.group_size_for("claude"), 1);
        assert_eq!(
            store.group_size_for("retired-model"),
            0,
            "a removed model's size entry is swept"
        );
        assert_eq!(store.group_size_key_count(), 2);
    }

    #[test]
    fn mark_with_max_duration_does_not_panic() {
        // `Duration::from_secs(u64::MAX)` is accepted by config
        // deserialization; the resulting `Instant` arithmetic must clamp
        // rather than overflow-panic the caller's task.
        let store = CooldownStore::new();
        store.mark("dep1", Duration::from_secs(u64::MAX));
        assert!(store.is_cooling("dep1"));

        let window = Duration::from_secs(u64::MAX);
        assert_eq!(store.record_failure("dep1", 500, window), 1);
        assert_eq!(store.failure_count("dep1", 500), 1);
    }

    #[test]
    fn clamp_window_keeps_realistic_durations() {
        let window = Duration::from_secs(60);
        assert_eq!(clamp_window(window), window);
        assert_eq!(
            clamp_window(Duration::from_secs(86_400 * 365)),
            Duration::from_secs(86_400 * 365)
        );
        assert_eq!(clamp_window(Duration::from_secs(u64::MAX)), MAX_WINDOW);
    }
}
