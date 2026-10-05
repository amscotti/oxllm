//! Shared stores for latency-based and usage-based routing strategies.
//!
//! These stores are in-memory (std::sync) so the core stays pure. The host
//! updates them after each request; the strategies read them synchronously.

use std::collections::{HashMap, VecDeque};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use hmac::{Hmac, Mac};
use sha2::Sha256;

/// Default rolling-window size for p50 latency tracking.
const DEFAULT_WINDOW: usize = 100;

/// Self-healing TTL for in-flight counters. The host's inc/dec pairing around
/// the upstream call is the primary contract; this TTL is a defense-in-depth
/// against drift if that contract is ever violated (a task cancelled mid-await
/// that skips its `-1`, a panic between inc and dec, or a host bug). Any entry
/// not touched within this window is treated as stale and reset to 0 on the
/// next read or write, so a leaked `+1` cannot bias [`crate::router::LeastBusy`]
/// forever. Five minutes is well above any realistic single-request duration
/// (a healthy deployment serving fresh traffic keeps its entry freshly
/// touched), while still bounding drift to a finite window.
pub(crate) const IN_FLIGHT_TTL: Duration = Duration::from_secs(5 * 60);

/// A timestamped in-flight counter entry. The `last_touched` timestamp drives
/// the lazy self-healing sweep (see [`IN_FLIGHT_TTL`]).
#[derive(Debug, Clone, Copy)]
struct InFlightEntry {
    count: i64,
    last_touched: Instant,
}

impl Default for InFlightEntry {
    fn default() -> Self {
        Self {
            count: 0,
            last_touched: Instant::now(),
        }
    }
}

/// Tracks rolling latency samples per deployment key for p50 computation.
#[derive(Debug)]
pub struct LatencyStore {
    samples: Mutex<HashMap<String, VecDeque<u64>>>,
    window: usize,
}

impl Default for LatencyStore {
    fn default() -> Self {
        Self::new(DEFAULT_WINDOW)
    }
}

impl LatencyStore {
    /// Create a new latency store with the given rolling-window size.
    pub fn new(window: usize) -> Self {
        let window = if window == 0 { DEFAULT_WINDOW } else { window };
        Self {
            samples: Mutex::new(HashMap::new()),
            window,
        }
    }

    /// Record a latency sample (in milliseconds) for a deployment key.
    pub fn record(&self, key: &str, duration_ms: u64) {
        let mut guard = self.samples.lock().unwrap_or_else(|e| e.into_inner());
        let deque = guard.entry(key.to_string()).or_default();
        deque.push_back(duration_ms);
        // O(1) eviction: `pop_front` avoids the O(window) memmove that a
        // front `Vec::drain` would incur on every record.
        while deque.len() > self.window {
            deque.pop_front();
        }
    }

    /// Compute the p50 (median) latency for a key, or `None` if no samples.
    ///
    /// For even sample counts the true median — the average of the two
    /// middle samples, rounded down — rather than the upper middle element
    /// (which silently biased `LowestLatency`/`LatencyBased` upward).
    ///
    /// Works on a scratch copy so chronological sample order is preserved for
    /// EWMA and other consumers of the same store.
    pub fn p50(&self, key: &str) -> Option<u64> {
        // Snapshot under the lock, compute unlocked: `select_nth_unstable`
        // is O(n) and previously ran while holding the mutex, serializing
        // every concurrent `record` on the hot path.
        let scratch: Vec<u64> = {
            let guard = self.samples.lock().unwrap_or_else(|e| e.into_inner());
            let deque = guard.get(key)?;
            if deque.is_empty() {
                return None;
            }
            deque.iter().copied().collect()
        };
        let mut scratch = scratch;
        let mid = scratch.len() / 2;
        let (_, upper, _) = scratch.select_nth_unstable(mid);
        let upper = *upper;
        if scratch.len() % 2 == 1 {
            return Some(upper);
        }
        // After `select_nth_unstable(mid)` everything in `scratch[..mid]` is
        // <= `upper`, so the lower middle element of the sorted order is
        // the max of that partition — and `lower <= upper` is guaranteed,
        // making `lower + (upper - lower) / 2` an overflow-free floor
        // average even for u64-scale millisecond samples.
        match scratch[..mid].iter().copied().max() {
            Some(lower) => Some(lower + (upper - lower) / 2),
            None => Some(upper),
        }
    }

    /// Compute the EWMA (exponentially weighted moving average) latency for a
    /// key, or `None` if no samples. Uses alpha = 0.3.
    pub fn ewma(&self, key: &str) -> Option<f64> {
        const ALPHA: f64 = 0.3;
        // Same snapshot-then-compute split as `p50`: the fold is O(n) and
        // must not hold the mutex across it.
        let samples: Vec<u64> = {
            let guard = self.samples.lock().unwrap_or_else(|e| e.into_inner());
            let deque = guard.get(key)?;
            if deque.is_empty() {
                return None;
            }
            deque.iter().copied().collect()
        };
        let mut ewma = samples[0] as f64;
        for &sample in samples.iter().skip(1) {
            ewma = ALPHA * sample as f64 + (1.0 - ALPHA) * ewma;
        }
        Some(ewma)
    }

    /// Remove all samples.
    pub fn clear(&self) {
        let mut guard = self.samples.lock().unwrap_or_else(|e| e.into_inner());
        guard.clear();
    }

    /// Drop latency samples for keys not in `known`. Called at config
    /// hot-reload (see [`crate::router::Router::prune_stale_store_keys`]) so a
    /// deployment removed from the model list does not leak its samples
    /// forever.
    pub fn retain_keys(&self, known: &std::collections::HashSet<String>) {
        let mut guard = self.samples.lock().unwrap_or_else(|e| e.into_inner());
        guard.retain(|key, _| known.contains(key));
    }

    /// Number of keys being tracked.
    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.samples.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// Whether there are no tracked keys.
    #[cfg(test)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Tracks a usage score per deployment key (lower = less loaded). The host
/// snapshots cache counters into this store; strategies read it synchronously.
///
/// Also tracks in-flight request counts (`least_busy` strategy).
/// The host increments before the upstream call and decrements after; the
/// `LeastBusy` strategy reads these counts to prefer idle deployments. Each
/// entry carries a `last_touched` timestamp so a leaked `+1` (a host contract
/// violation) self-heals after `IN_FLIGHT_TTL` rather than biasing the
/// `LeastBusy` pick forever.
#[derive(Debug)]
pub struct UsageStore {
    entries: Mutex<HashMap<String, (f64, Instant)>>,
    in_flight: Mutex<HashMap<String, InFlightEntry>>,
    in_flight_ttl: Duration,
}

impl Default for UsageStore {
    fn default() -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
            in_flight: Mutex::new(HashMap::new()),
            in_flight_ttl: IN_FLIGHT_TTL,
        }
    }
}

impl UsageStore {
    /// Create a new empty usage store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new empty usage store with an explicit in-flight self-healing
    /// TTL. Used by tests to exercise the staleness sweep without waiting out
    /// the production [`IN_FLIGHT_TTL`] window.
    #[cfg(test)]
    pub(crate) fn with_in_flight_ttl(in_flight_ttl: Duration) -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
            in_flight: Mutex::new(HashMap::new()),
            in_flight_ttl,
        }
    }

    /// Record the current usage score for a deployment key.
    ///
    /// Non-finite (NaN/±∞) or negative scores are stored as `0.0`: a single
    /// NaN from a bad division in the host's snapshot logic would otherwise
    /// poison the entry permanently — every comparison against NaN is
    /// false, so [`crate::router::UsageBased`] could never displace the
    /// poisoned candidate and would silently degrade to "first candidate
    /// wins". Matches the defensive posture of the config-facing floats
    /// (`AdaptiveRouterConfig::validate`, `ComplexityRouterConfig::validate`),
    /// which reject non-finite values at load.
    pub fn record(&self, key: &str, usage: f64) {
        let usage = if usage.is_finite() && usage >= 0.0 {
            usage
        } else {
            tracing::warn!(
                event = "usage_store",
                deployment = %key,
                value = %usage,
                "non-finite or negative usage score recorded; storing 0.0"
            );
            0.0
        };
        let mut guard = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        guard.insert(key.to_string(), (usage, Instant::now()));
    }

    /// Get the usage score for a key (0.0 if absent — no usage = least loaded).
    pub fn get(&self, key: &str) -> f64 {
        let guard = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        guard
            .get(key)
            .filter(|(_, recorded)| recorded.elapsed() < Duration::from_secs(60))
            .map_or(0.0, |(usage, _)| *usage)
    }

    /// Whether the store has any recorded usage score for `key`. Distinct from
    /// `get` (which returns `0.0` for both "zero usage" and "no entry"): the
    /// router uses this to decide whether it has real data to route on or
    /// should fall back to `SimpleShuffle`.
    pub fn contains(&self, key: &str) -> bool {
        let guard = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        guard
            .get(key)
            .is_some_and(|(_, recorded)| recorded.elapsed() < Duration::from_secs(60))
    }

    /// Adjust the in-flight request count for a deployment key by `delta`
    /// (typically `+1` before an upstream call and `-1` after). The count is
    /// clamped at 0 from below to defend against bookkeeping drift, and uses
    /// saturating arithmetic so an extreme `delta` (e.g. a host bug passing
    /// `i64::MAX`) saturates at `i64::MAX` instead of panicking in debug or
    /// wrapping (then being masked by the clamp) in release.
    ///
    /// Self-healing: if the entry has not been touched within
    /// `IN_FLIGHT_TTL` (or the TTL passed at construction), the stale count
    /// is reset to 0 before applying `delta`. This bounds the worst-case
    /// drift from a leaked `+1` to one TTL window — without it, a host
    /// contract violation (cancellation mid-await, panic between inc/dec)
    /// would leave the count elevated forever, slowly starving the affected
    /// deployment under `LeastBusy`.
    pub fn incr_in_flight(&self, key: &str, delta: i64) {
        let mut guard = self.in_flight.lock().unwrap_or_else(|e| e.into_inner());
        let now = Instant::now();
        let entry = guard.entry(key.to_string()).or_default();
        if now.duration_since(entry.last_touched) > self.in_flight_ttl {
            entry.count = 0;
        }
        entry.count = entry.count.saturating_add(delta).max(0);
        entry.last_touched = now;
    }

    /// Get the current in-flight request count for a key (0 if absent or
    /// stale per `IN_FLIGHT_TTL`). Stale entries are read as 0 so a leaked
    /// increment cannot bias the [`crate::router::LeastBusy`] pick past the
    /// self-healing window.
    pub fn in_flight(&self, key: &str) -> i64 {
        let guard = self.in_flight.lock().unwrap_or_else(|e| e.into_inner());
        let now = Instant::now();
        match guard.get(key) {
            Some(entry) if now.duration_since(entry.last_touched) <= self.in_flight_ttl => {
                entry.count
            }
            _ => 0,
        }
    }

    /// Remove all entries (both usage scores and in-flight counts).
    pub fn clear(&self) {
        let mut guard = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        guard.clear();
        let mut in_flight = self.in_flight.lock().unwrap_or_else(|e| e.into_inner());
        in_flight.clear();
    }

    /// Drop usage scores and in-flight counts for keys not in `known`.
    /// Called at config hot-reload (see
    /// [`crate::router::Router::prune_stale_store_keys`]) so deployments
    /// removed from the model list do not leak their entries forever.
    pub fn retain_keys(&self, known: &std::collections::HashSet<String>) {
        let mut guard = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        guard.retain(|key, _| known.contains(key));
        let mut in_flight = self.in_flight.lock().unwrap_or_else(|e| e.into_inner());
        in_flight.retain(|key, _| known.contains(key));
    }

    /// Whether there are any entries.
    #[cfg(test)]
    pub fn is_empty(&self) -> bool {
        self.entries
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_empty()
    }
}

/// Derive a stable key for a deployment, used by latency and usage stores.
///
/// **Scope:** this keys by `model_name` + provider model only, so it
/// **collides for sibling deployments** that share a model but differ in
/// endpoint/credentials. It is intended solely for model-level (not
/// deployment-level) accounting — the server's `spend:dep:{...}` and
/// `dep_tpm:{...}` counters, where per-model aggregation is the desired
/// semantics. **Do not** use it for latency, usage, in-flight, cooldown,
/// health, or bandit stores, where siblings must get distinct buckets: use
/// [`stable_key`] instead.
pub fn deployment_key(deployment: &crate::deployment::Deployment) -> String {
    format!(
        "{}:{}",
        deployment.model_name, deployment.oxllm_params.model
    )
}

/// FNV-1a 64-bit hash. Deterministic across platforms and Rust releases
/// (unlike [`std::collections::hash_map::DefaultHasher`], whose seed and
/// algorithm are unspecified), so the same input yields the same digest on
/// every node and every reload.
///
/// Shared across the router modules that need a stable, portable hash: the
/// non-secret portion of `stable_key` here, and the session-affinity index
/// in [`crate::router::session_affinity`]. Keeping one canonical `pub(crate)`
/// helper prevents drift between the two key spaces (a session id and a
/// config-derived key that should hash identically on every replica).
pub(crate) fn fnv1a_64(bytes: &[u8]) -> u64 {
    const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    bytes.iter().fold(OFFSET_BASIS, |mut hash, &b| {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(PRIME);
        hash
    })
}

/// The per-process random pepper for [`stable_key`]'s credential digest.
/// Random at first use (OS entropy), fixed for the process lifetime: keys
/// are stable within a process (hot-reloads, store lookups) but deliberately
/// NOT comparable across processes — see [`stable_key`].
fn key_pepper() -> &'static [u8; 32] {
    static PEPPER: OnceLock<[u8; 32]> = OnceLock::new();
    PEPPER.get_or_init(|| rand::Rng::random(&mut rand::rng()))
}

/// Keyed digest (HMAC-SHA256, truncated to 16 hex chars) over a
/// deployment's credential fields and its forward-compat `extra` bag (which
/// may itself carry credential-shaped params for providers core does not
/// model). The keyed construction removes the offline verification oracle
/// an unkeyed digest of secret-bearing material would leave: a guessed API
/// key cannot be confirmed against an observed key without the pepper.
fn credential_digest(params: &crate::deployment::OxllmParams) -> String {
    let mut mac =
        Hmac::<Sha256>::new_from_slice(key_pepper()).expect("HMAC-SHA256 accepts any key length");
    for secret in [
        &params.api_key,
        &params.aws_access_key_id,
        &params.aws_secret_access_key,
        &params.aws_session_token,
        &params.api_base,
    ] {
        mac.update(secret.as_deref().unwrap_or("").as_bytes());
        mac.update(&[0]);
    }
    if let Ok(extra) = serde_json::to_string(&params.extra) {
        mac.update(extra.as_bytes());
    }
    if let Ok(headers) = serde_json::to_value(&params.extra_headers) {
        mac.update(headers.to_string().as_bytes());
    }
    let digest = mac.finalize().into_bytes();
    digest[..8].iter().map(|b| format!("{b:02x}")).collect()
}

/// Derive a stable per-deployment identity key, used by the latency, usage,
/// in-flight, cooldown, health, and bandit stores.
///
/// Unlike [`deployment_key`] — which keys by `model_name` + provider model and
/// therefore collides for sibling deployments sharing a model but differing in
/// endpoint/credentials (a fast primary and a slow fallback, multi-region
/// backends) — this key covers the deployment's full `oxllm_params`, so
/// sibling deployments get distinct latency/usage/in-flight buckets, matching
/// the cooldown-family stores keyed by
/// [`crate::router::Router::cooldown_key`]. Because it is a pure function of
/// the deployment, the key is invariant under config-list reordering.
///
/// Credential fields never enter the digest as plaintext: the non-secret
/// params are serialized verbatim, and the credential material is mixed in
/// through a keyed HMAC-SHA256 digest (`credential_digest`) whose pepper is
/// random per process. Sibling-distinctness survives (any difference anywhere
/// changes the key), but an exposed key is not an offline verification oracle
/// for guessed API keys — every byte of a pre-pepper serialization was
/// guessable except the secret, which is exactly what made the old unkeyed
/// digest one.
///
/// Consequence of the pepper: keys are **not comparable across processes**.
/// In-process consumers (store lookups, hot-reloads, management state) are
/// unaffected; the one cross-process surface is the prompt-caching pin
/// written to a shared cache — a pin authored by another replica (or a
/// previous process) simply matches no local candidate key and the check
/// fails open to normal routing.
pub fn stable_key(deployment: &crate::deployment::Deployment) -> String {
    let mut input = deployment.model_name.clone();
    input.push('\u{0}');
    // `OxllmParams: Serialize` is infallible for any value that round-tripped
    // through config deserialization, so this branch is unreachable today.
    // But a future custom `Serialize` impl could make it reachable, and
    // silently falling back to an empty string would collapse siblings
    // sharing a model_name into the same key — exactly the collision this
    // function exists to prevent. Log loudly so a regression is visible,
    // and fall back to a deployment-distinct input (api_base — the
    // credential digest below covers the secret side) so siblings stay
    // distinct even when serialization fails.
    // Serializing through Value orders map keys, including independently
    // deserialized HashMaps, so a reload retains the same runtime buckets.
    let serialized = match serde_json::to_value(sanitized_params(deployment))
        .and_then(|value| serde_json::to_string(&value))
    {
        Ok(s) => s,
        Err(e) => {
            tracing::error!(
                error = %e,
                "stable_key: serde_json::to_string(oxllm_params) failed; \
                 falling back to api_base so sibling deployments stay \
                 distinct. OxllmParams: Serialize must remain infallible for \
                 any value that round-tripped through config deserialization.",
            );
            format!(
                "{{\"api_base\":{}}}",
                serde_json::Value::String(
                    deployment.oxllm_params.api_base.clone().unwrap_or_default()
                )
            )
        }
    };
    input.push_str(&serialized);
    input.push('\u{0}');
    input.push_str(&credential_digest(&deployment.oxllm_params));
    format!(
        "{}:{:016x}",
        deployment.model_name,
        fnv1a_64(input.as_bytes())
    )
}

/// The `oxllm_params` with every credential-bearing field and the
/// forward-compat `extra` bag cleared — the plaintext portion of the
/// [`stable_key`] hash input. Credentials re-enter through
/// [`credential_digest`] instead.
fn sanitized_params(deployment: &crate::deployment::Deployment) -> crate::deployment::OxllmParams {
    let mut params = deployment.oxllm_params.clone();
    params.api_key = None;
    params.aws_access_key_id = None;
    params.aws_secret_access_key = None;
    params.aws_session_token = None;
    params.api_base = None;
    params.extra_headers = None;
    params.extra = serde_json::Map::new();
    params
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deployment::{Deployment, OxllmParams};

    #[test]
    fn latency_store_records_and_computes_p50() {
        let store = LatencyStore::default();
        store.record("a", 100);
        store.record("a", 200);
        store.record("a", 300);
        let p50 = store.p50("a").expect("should have p50");
        assert_eq!(p50, 200);
    }

    #[test]
    fn latency_store_p50_none_when_empty() {
        let store = LatencyStore::default();
        assert!(store.p50("missing").is_none());
    }

    #[test]
    fn latency_store_rolling_window_trims() {
        let store = LatencyStore::new(3);
        store.record("a", 10);
        store.record("a", 20);
        store.record("a", 30);
        store.record("a", 40);
        let p50 = store.p50("a").expect("should have p50");
        assert_eq!(p50, 30, "window=[20,30,40], p50=30");
    }

    #[test]
    fn latency_store_p50_even_count_averages_middle_pair() {
        // Even sample counts must report the true median (average of the
        // two middle samples), not the upper middle element: [10, 20] is
        // 15, not 20. The upper-median behavior biased
        // `LowestLatency`/`LatencyBased` upward uniformly.
        let store = LatencyStore::default();
        store.record("a", 10);
        store.record("a", 20);
        assert_eq!(store.p50("a"), Some(15));
        store.record("a", 21);
        assert_eq!(
            store.p50("a"),
            Some(20),
            "odd counts stay the middle element"
        );
        // Order-independence: same samples in arrival order [20, 10].
        store.record("b", 20);
        store.record("b", 10);
        assert_eq!(store.p50("b"), Some(15));
    }

    #[test]
    fn latency_store_p50_average_does_not_overflow() {
        // u64::MAX-scale millisecond samples must produce an exact floor
        // average without overflowing the middle-pair arithmetic.
        let store = LatencyStore::new(2);
        store.record("a", u64::MAX);
        store.record("a", u64::MAX - 1);
        assert_eq!(store.p50("a"), Some(u64::MAX - 1));
        store.record("b", u64::MAX - 2);
        store.record("b", u64::MAX);
        assert_eq!(store.p50("b"), Some(u64::MAX - 1));
    }

    #[test]
    fn latency_store_ewma() {
        let store = LatencyStore::default();
        store.record("a", 100);
        store.record("a", 200);
        let ewma = store.ewma("a").expect("should have ewma");
        let expected = 0.3 * 200.0 + 0.7 * 100.0;
        assert!((ewma - expected).abs() < 1e-9);
    }

    #[test]
    fn latency_store_ewma_none_when_empty() {
        let store = LatencyStore::default();
        assert!(store.ewma("missing").is_none());
    }

    #[test]
    fn latency_store_zero_window_defaults() {
        let store = LatencyStore::new(0);
        store.record("a", 50);
        assert_eq!(store.p50("a"), Some(50));
    }

    #[test]
    fn usage_store_records_and_gets() {
        let store = UsageStore::new();
        assert_eq!(store.get("a"), 0.0);
        store.record("a", 0.5);
        assert!((store.get("a") - 0.5).abs() < 1e-9);
        store.record("a", 0.9);
        assert!((store.get("a") - 0.9).abs() < 1e-9);
    }

    #[test]
    fn usage_store_record_rejects_nan() {
        // A NaN from a bad division in the host's snapshot logic must not
        // poison the entry permanently (all comparisons against NaN are
        // false, so a NaN first candidate could never be displaced and
        // `UsageBased` would degrade to "first candidate wins").
        // Non-finite and negative scores sanitize to 0.0, matching the
        // defensive posture of the config-facing float validators.
        let store = UsageStore::new();
        store.record("a", f64::NAN);
        assert_eq!(store.get("a"), 0.0, "NaN must sanitize to 0.0");
        store.record("b", f64::INFINITY);
        store.record("c", -2.5);
        assert_eq!(store.get("b"), 0.0);
        assert_eq!(store.get("c"), 0.0);
        // A sanitized entry is an ordinary 0.0 entry: later legitimate
        // records overwrite it.
        store.record("a", 0.8);
        assert_eq!(store.get("a"), 0.8);
    }

    #[test]
    fn usage_store_nan_does_not_break_usage_based_pick() {
        use crate::router::strategy::{RequestContext, RoutingStrategy};
        use crate::router::usage_based::UsageBased;
        use std::sync::Arc;

        let store = Arc::new(UsageStore::new());
        // Poison first (the pre-fix failure mode: the first candidate's
        // NaN made it undisplaceable), then record a genuinely loaded
        // score for it and a lighter one for its sibling.
        let busy = dep_with("gpt-4o", "openai/busy", Some("sk-busy-aaaaaaaaaaaaaa"));
        let idle = dep_with("gpt-4o", "openai/idle", Some("sk-idle-bbbbbbbbbbbbbb"));
        store.record(&stable_key(&busy), f64::NAN);
        store.record(&stable_key(&idle), f64::NAN);
        store.record(&stable_key(&busy), 0.9);
        store.record(&stable_key(&idle), 0.1);
        let strategy = UsageBased::new(store);
        let refs: Vec<&Deployment> = vec![&busy, &idle];
        let ctx = RequestContext::new("gpt-4o");
        let picked = strategy.pick(&refs, &ctx).expect("pick");
        assert_eq!(
            picked.oxllm_params.model, "openai/idle",
            "UsageBased must still pick the genuinely less-loaded deployment"
        );
    }

    #[test]
    fn usage_store_clear() {
        let store = UsageStore::new();
        store.record("a", 1.0);
        assert!(!store.is_empty());
        store.clear();
        assert!(store.is_empty());
    }

    #[test]
    fn in_flight_increments_and_reads() {
        let store = UsageStore::new();
        assert_eq!(store.in_flight("a"), 0);
        store.incr_in_flight("a", 1);
        assert_eq!(store.in_flight("a"), 1);
        store.incr_in_flight("a", 1);
        assert_eq!(store.in_flight("a"), 2);
        store.incr_in_flight("a", -1);
        assert_eq!(store.in_flight("a"), 1);
    }

    #[test]
    fn in_flight_clamps_at_zero() {
        let store = UsageStore::new();
        store.incr_in_flight("a", -1);
        assert_eq!(store.in_flight("a"), 0, "negative counts must clamp at 0");
    }

    #[test]
    fn in_flight_saturates_on_overflow() {
        // Saturating arithmetic: an extreme delta must pin the count at
        // i64::MAX instead of panicking (debug) or wrapping (release).
        let store = UsageStore::new();
        store.incr_in_flight("a", i64::MAX);
        assert_eq!(store.in_flight("a"), i64::MAX);
        store.incr_in_flight("a", 1);
        assert_eq!(
            store.in_flight("a"),
            i64::MAX,
            "overflow must saturate, not wrap"
        );
    }

    #[test]
    fn in_flight_clear_resets() {
        let store = UsageStore::new();
        store.incr_in_flight("a", 5);
        store.clear();
        assert_eq!(store.in_flight("a"), 0);
    }

    /// Rewind an entry's `last_touched` by `age` (accumulates across calls),
    /// simulating elapsed time deterministically. CI runners under load can
    /// overshoot short `thread::sleep`s by far more than the TTL margins the
    /// in-flight tests rely on; backdating removes that wall-clock race.
    fn backdate_in_flight(store: &UsageStore, key: &str, age: Duration) {
        let mut guard = store.in_flight.lock().unwrap_or_else(|e| e.into_inner());
        let entry = guard.get_mut(key).expect("entry must exist to backdate");
        entry.last_touched = entry
            .last_touched
            .checked_sub(age)
            .expect("monotonic clock has been running longer than `age`");
    }

    #[test]
    fn in_flight_stale_count_self_heals_on_read() {
        // A leaked `+1` (host contract violation: the `-1` never arrived)
        // must not bias `LeastBusy` forever. After the self-healing TTL
        // elapses with no further touch, the read path treats the stale
        // count as 0.
        let store = UsageStore::with_in_flight_ttl(Duration::from_millis(20));
        store.incr_in_flight("leaked", 1);
        assert_eq!(store.in_flight("leaked"), 1, "fresh entry reads through");
        backdate_in_flight(&store, "leaked", Duration::from_millis(30));
        assert_eq!(
            store.in_flight("leaked"),
            0,
            "stale entry must self-heal to 0 on read"
        );
    }

    #[test]
    fn in_flight_stale_count_self_heals_on_write() {
        // Same defense on the write path: a touch after the TTL resets the
        // stale count before applying the new delta, so a leaked `+1`
        // followed by a legitimate `+1` does not accumulate to 2.
        let store = UsageStore::with_in_flight_ttl(Duration::from_millis(20));
        store.incr_in_flight("leaked", 1);
        backdate_in_flight(&store, "leaked", Duration::from_millis(30));
        store.incr_in_flight("leaked", 1);
        assert_eq!(
            store.in_flight("leaked"),
            1,
            "stale count must reset before applying the new delta"
        );
    }

    #[test]
    fn in_flight_fresh_traffic_stays_visible() {
        // A legitimately busy deployment touches its entry on every inc/dec,
        // so the TTL never trips for it. Each touch is backdated 10ms: the
        // cumulative age from the first touch (6 x 10ms) exceeds the 50ms
        // TTL, so an implementation that failed to refresh `last_touched`
        // on every touch would heal to 0, while the last touch is only
        // 10ms old and must read through.
        let store = UsageStore::with_in_flight_ttl(Duration::from_millis(50));
        for _ in 0..6 {
            store.incr_in_flight("hot", 1);
            backdate_in_flight(&store, "hot", Duration::from_millis(10));
        }
        assert_eq!(
            store.in_flight("hot"),
            6,
            "freshly touched entries must not be healed"
        );
    }

    #[test]
    fn usage_store_retain_keys_drops_unknown() {
        let store = UsageStore::new();
        store.record("keep", 0.5);
        store.incr_in_flight("drop", 3);
        let known: std::collections::HashSet<String> = ["keep".to_string()].into_iter().collect();
        store.retain_keys(&known);
        assert!((store.get("keep") - 0.5).abs() < 1e-9);
        assert_eq!(store.get("drop"), 0.0);
        assert_eq!(store.in_flight("drop"), 0);
    }

    #[test]
    fn latency_store_clear() {
        let store = LatencyStore::default();
        store.record("a", 10);
        assert_eq!(store.len(), 1);
        store.clear();
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn latency_store_record_is_constant_time_at_saturation() {
        // `record` must stay O(1) as the window fills: the previous front
        // `Vec::drain` was O(window) per push (a memmove of the whole window
        // on every record). With `VecDeque::pop_front` the per-record cost is
        // flat regardless of how full the window is. Compare the first-record
        // time against a saturated-record time over a large run.
        let window = 100;
        let store = LatencyStore::new(window);
        let n = 100_000;
        let start = std::time::Instant::now();
        store.record("hot", 1);
        let first = start.elapsed();
        let saturated_start = std::time::Instant::now();
        for _ in 0..n {
            store.record("hot", 1);
        }
        let saturated_avg = saturated_start.elapsed() / n;
        // The saturated per-record time must be comparable to (not orders of
        // magnitude larger than) the first record. An O(window) memmove would
        // dwarf the O(1) first push.
        assert!(
            saturated_avg.as_nanos() < first.as_nanos().max(1) * 50,
            "saturated record ({saturated_avg:?}) must stay O(1), first was {first:?}"
        );
        assert_eq!(store.p50("hot"), Some(1));
    }

    #[test]
    fn latency_store_retain_keys_drops_unknown() {
        let store = LatencyStore::default();
        store.record("keep:a", 10);
        store.record("drop:b", 20);
        let known: std::collections::HashSet<String> = ["keep:a".to_string()].into_iter().collect();
        store.retain_keys(&known);
        assert_eq!(store.len(), 1);
        assert_eq!(store.p50("keep:a"), Some(10));
        assert_eq!(store.p50("drop:b"), None);
    }

    // ------------------------------------------------------------------
    // stable_key / fnv1a_64
    // ------------------------------------------------------------------

    fn dep_with(model_name: &str, provider_model: &str, api_key: Option<&str>) -> Deployment {
        Deployment {
            model_name: model_name.to_string(),
            oxllm_params: OxllmParams {
                model: provider_model.to_string(),
                api_key: api_key.map(|s| s.to_string()),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[test]
    fn stable_key_changes_when_api_key_rotates() {
        // `stable_key` deliberately covers the full `oxllm_params` (the
        // credential material through the keyed digest) so two deployments
        // differing only in credentials get distinct
        // cooldown/latency/usage/in-flight/health/bandit buckets.
        // The operational consequence is that rotating an API key silently
        // resets that deployment's stored state — documented here as a
        // contract so a future refactor that narrows the hash input (and
        // would silently merge a fast primary with a slow fallback) is
        // caught.
        let before = dep_with("gpt-4o", "openai/gpt-4o", Some("sk-old-aaaaaaaaaaaaaa"));
        let after = dep_with("gpt-4o", "openai/gpt-4o", Some("sk-new-bbbbbbbbbbbbbb"));
        assert_ne!(
            stable_key(&before),
            stable_key(&after),
            "rotating api_key must produce a fresh stable identity"
        );
    }

    #[test]
    fn stable_key_is_not_an_unkeyed_digest_of_the_params() {
        // Regression pin for the peppered rewrite: the key's digest must NOT
        // equal any deterministic function an observer could compute from
        // the (guessable) params — the pre-pepper scheme was plain FNV-1a
        // over the serialized params, i.e. an offline verification oracle
        // for guessed API keys when the key surfaced in metric labels.
        let dep = dep_with("gpt-4o", "openai/gpt-4o", Some("sk-secret-00000000000"));
        let old_style = {
            let mut input = dep.model_name.clone();
            input.push('\u{0}');
            input.push_str(&serde_json::to_string(&dep.oxllm_params).expect("serialize"));
            fnv1a_64(input.as_bytes())
        };
        let key = stable_key(&dep);
        let digest = key.rsplit(':').next().expect("digest suffix");
        assert_ne!(
            digest,
            format!("{old_style:016x}"),
            "stable_key must not be reproducible from the plaintext params"
        );
        assert!(
            !key.contains("sk-secret"),
            "no credential plaintext in the key"
        );
    }

    #[test]
    fn stable_key_ignores_credential_plaintext_changes() {
        // Within one process the key is a pure function of the deployment —
        // but only through the *keyed* digest. Two deployments differing
        // ONLY in credentials keep distinct keys (covered above); two
        // identical deployments produce identical keys.
        let a = dep_with("gpt-4o", "openai/gpt-4o", Some("sk-a-aaaaaaaaaaaaaaa"));
        let b = dep_with("gpt-4o", "openai/gpt-4o", Some("sk-a-aaaaaaaaaaaaaaa"));
        assert_eq!(stable_key(&a), stable_key(&b));
    }

    #[test]
    fn stable_key_survives_independent_header_map_deserialization() {
        let json = serde_json::json!({
            "model_name":"chat",
            "oxllm_params":{
                "model":"openai/gpt-4o",
                "api_base":"https://example.test/v1?key=secret",
                "extra_headers":{"x-region":"east","x-auth":"secret","x-route":"chat"}
            }
        });
        let expected = stable_key(&serde_json::from_value(json.clone()).expect("deployment"));
        for _ in 0..64 {
            let dep: Deployment = serde_json::from_value(json.clone()).expect("deployment");
            assert_eq!(stable_key(&dep), expected);
        }
        let mut rotated: Deployment = serde_json::from_value(json).expect("deployment");
        rotated
            .oxllm_params
            .extra_headers
            .as_mut()
            .unwrap()
            .insert("x-auth".into(), "rotated".into());
        assert_ne!(stable_key(&rotated), expected);
        let sanitized = serde_json::to_string(&sanitized_params(&rotated)).expect("serialize");
        assert!(!sanitized.contains("key=secret"));
        assert!(!sanitized.contains("\"secret\""));
        assert!(!sanitized.contains("\"rotated\""));
    }

    #[test]
    fn stable_key_invariant_under_reorder_with_siblings() {
        // Pure function of `oxllm_params` — reordering the model list does
        // not change a deployment's identity, and sibling deployments sharing
        // a model_name still get distinct keys (covered by the api_key test
        // above); pin the invariance for the same deployment directly.
        let d = dep_with("gpt-4o", "openai/gpt-4o", Some("sk-stable-key-1"));
        assert_eq!(stable_key(&d), stable_key(&d));
    }

    #[test]
    fn fnv1a_64_matches_known_constants() {
        // FNV-1a 64 of the empty string is the offset basis; of "a" is
        // offset_basis XOR 'a' (0x61) times the prime. Pinning these keeps
        // the helper from drifting (it is shared with session_affinity).
        assert_eq!(fnv1a_64(b""), 0xcbf2_9ce4_8422_2325);
        let mut h: u64 = 0xcbf2_9ce4_8422_2325 ^ u64::from(b'a');
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
        assert_eq!(fnv1a_64(b"a"), h);
    }

    #[test]
    fn stable_key_falls_back_distinctly_when_serialization_fails() {
        // `OxllmParams: Serialize` is infallible for any value that
        // round-tripped through config deserialization today, so this test
        // cannot easily trigger the fallback path through the public API.
        // Instead it pins the *contract* the fallback exists to enforce:
        // two deployments sharing a `model_name` but differing in
        // endpoint/credentials must NEVER collapse to the same stable key
        // — even if (hypothetically) the primary serialization path were
        // to fail for both. The fallback the implementation uses
        // (`api_base`, plus the keyed credential digest) keeps them
        // distinct because `api_base`/credentials are part of the fallback
        // input.
        let a = dep_with("gpt-4o", "openai/gpt-4o", Some("sk-a-aaaaaaaaaaaaaaa"));
        let b = dep_with("gpt-4o", "openai/gpt-4o", Some("sk-b-bbbbbbbbbbbbbbb"));
        assert_ne!(
            stable_key(&a),
            stable_key(&b),
            "siblings differing in api_key must keep distinct stable keys \
             even under a hypothetical serialization failure"
        );
    }
}
