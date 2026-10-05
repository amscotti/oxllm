//! Per-deployment connectivity health tracking.
//!
//! Health is distinct from cooldown: [`CooldownStore`](crate::router::CooldownStore)
//! is error-rate-driven (a deployment cooling after repeated retryable
//! errors), while [`HealthState`] tracks **connectivity** — whether the
//! deployment can be reached at all (network errors, DNS failures, TLS
//! errors). The host marks connection outcomes per attempt; the router
//! filters unhealthy deployments out of
//! [`Router::pick_deployment`](crate::router::Router::pick_deployment) when
//! `enable_health_check_routing` is set.
//!
//! Entries carry a timestamp and go **stale**: an entry older than the
//! staleness threshold reads as healthy again (fail-open), so a recovered
//! deployment automatically re-enters rotation without an explicit probe.
//! The safety net in [`HealthState::healthy_deployments`] guarantees
//! progress: when every candidate is unhealthy, all candidates are returned
//! rather than failing the request outright.
//!
//! This is an in-memory implementation using `std::sync::Mutex` (no tokio),
//! matching the other router stores.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Health record for one deployment key.
#[derive(Debug, Clone)]
pub struct HealthEntry {
    /// Whether the deployment is currently considered connectable.
    pub is_healthy: bool,
    /// When the entry's health state was last set (creation or transition
    /// to unhealthy). Drives staleness: entries older than the staleness
    /// threshold read as healthy again (fail-open) and are lazily removed.
    /// Deliberately NOT refreshed by below-threshold failures, so
    /// intermittently-failing entries can still age out.
    pub timestamp: Instant,
    /// Why the deployment was marked unhealthy (e.g. the connection error).
    pub reason: Option<String>,
    /// Consecutive connection failures since the last success.
    pub consecutive_failures: u32,
}

/// Tracks per-deployment connectivity health.
#[derive(Debug, Default)]
pub struct HealthState {
    entries: Mutex<HashMap<String, HealthEntry>>,
}

impl HealthState {
    /// Create a new empty health state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Mark a deployment as unhealthy with an optional reason.
    pub fn mark_unhealthy(&self, key: &str, reason: Option<String>) {
        let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        let consecutive_failures = entries
            .get(key)
            .map(|e| e.consecutive_failures)
            .unwrap_or(0);
        entries.insert(
            key.to_string(),
            HealthEntry {
                is_healthy: false,
                timestamp: Instant::now(),
                reason,
                consecutive_failures,
            },
        );
    }

    /// Mark a deployment as healthy, clearing any failure history. Called
    /// after a successful connection: one good attempt proves connectivity.
    pub fn mark_healthy(&self, key: &str) {
        let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        entries.remove(key);
    }

    /// Whether a deployment is currently considered healthy.
    ///
    /// Fail-open: unknown keys are healthy, and entries older than
    /// `staleness_threshold` are lazily removed and read as healthy so a
    /// recovered deployment re-enters rotation automatically.
    pub fn is_healthy(&self, key: &str, staleness_threshold: Duration) -> bool {
        let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        let Some(entry) = entries.get(key) else {
            return true;
        };
        if entry.timestamp.elapsed() > staleness_threshold {
            entries.remove(key);
            return true;
        }
        entry.is_healthy
    }

    /// The recorded unhealthy reason for a deployment key, when the store
    /// holds an entry for it. Staleness is NOT applied here — pair with
    /// [`HealthState::is_healthy`] for the effective state (a stale entry
    /// reads as healthy but may still carry its last recorded reason).
    /// `None` when the deployment has no tracked entry (never failed, or
    /// recovered) or none was recorded. Intended for state-inspection
    /// endpoints.
    pub fn reason(&self, key: &str) -> Option<String> {
        let entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        entries.get(key).and_then(|e| e.reason.clone())
    }

    /// Filter `candidates` down to the healthy ones. `key_fn` derives the
    /// health key for each candidate (the same key the host uses when
    /// marking outcomes). Returning `&str` (rather than `String`) avoids a
    /// per-candidate allocation on the routing hot path — health routing
    /// is the most likely pre-call check to be enabled in production.
    ///
    /// Safety net: when **every** candidate is unhealthy, the unfiltered
    /// candidate list is returned — health routing must never turn "all
    /// deployments degraded" into "no deployments at all".
    pub fn healthy_deployments<T>(
        &self,
        candidates: Vec<T>,
        key_fn: impl Fn(&T) -> &str,
        staleness_threshold: Duration,
    ) -> Vec<T> {
        let (healthy, unhealthy): (Vec<T>, Vec<T>) = candidates
            .into_iter()
            .partition(|c| self.is_healthy(key_fn(c), staleness_threshold));
        if healthy.is_empty() {
            unhealthy
        } else {
            healthy
        }
    }

    /// Record a connection failure for a deployment and report whether this
    /// call **newly** marked the deployment unhealthy: `true` only on the
    /// call that crosses `failure_threshold` (the healthy→unhealthy
    /// transition) — exactly once per unhealthy episode, so callers can
    /// fire transition side effects (alerts, state-change metrics) without
    /// deduplicating. Below the threshold the deployment stays healthy —
    /// transient blips must not shrink the routing pool — and subsequent
    /// failures while already unhealthy return `false`. Callers that need
    /// the effective current state can follow up with
    /// [`HealthState::is_healthy`].
    ///
    /// The entry's `timestamp` (the staleness clock) is only set at
    /// creation and on the healthy→unhealthy transition. Refreshing it on
    /// every failure would let a trickle of below-threshold failures keep
    /// the entry alive forever, defeating the lazy eviction of stale
    /// entries.
    pub fn record_failure(&self, key: &str, reason: &str, failure_threshold: u32) -> bool {
        let threshold = failure_threshold.max(1);
        let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        let entry = entries.entry(key.to_string()).or_insert(HealthEntry {
            is_healthy: true,
            timestamp: Instant::now(),
            reason: None,
            consecutive_failures: 0,
        });
        // Saturating like the other router counters (`CooldownStore::
        // record_failure`, `UsageStore::incr_in_flight`): 2^32 consecutive
        // failures without an intervening success is unrealistic, but the
        // no-panic contract covers provider-driven inputs and a plain `+=`
        // would panic in debug builds.
        entry.consecutive_failures = entry.consecutive_failures.saturating_add(1);
        let mut newly_unhealthy = false;
        if entry.consecutive_failures >= threshold {
            if entry.is_healthy {
                // Healthy→unhealthy transition: restart the staleness
                // clock so the fresh unhealthy state stays authoritative
                // for a full staleness window, and capture the cause that
                // actually crossed the threshold. The reason is NOT
                // refreshed by later below/above-threshold failures, so an
                // alert payload reports the transition cause rather than a
                // more recent (possibly different) connection error.
                entry.timestamp = Instant::now();
                entry.reason = Some(reason.to_string());
                newly_unhealthy = true;
            }
            entry.is_healthy = false;
        }
        newly_unhealthy
    }

    /// Remove all health entries.
    pub fn clear(&self) {
        let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        entries.clear();
    }

    /// Drop health entries for keys not in `known`. Called at config
    /// hot-reload (see [`crate::router::Router::prune_stale_store_keys`]) so
    /// deployments removed from the model list do not retain stale health
    /// entries (the lazy stale-eviction on query only runs for keys that are
    /// still consulted).
    pub fn retain_keys(&self, known: &std::collections::HashSet<String>) {
        let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        entries.retain(|key, _| known.contains(key));
    }

    /// Number of tracked entries (including not-yet-lazily-removed stale ones).
    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.entries.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// Whether there are no tracked entries.
    #[cfg(test)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const STALENESS: Duration = Duration::from_secs(60);

    #[test]
    fn unknown_key_is_healthy() {
        let state = HealthState::new();
        assert!(state.is_healthy("dep1", STALENESS));
    }

    #[test]
    fn mark_unhealthy_then_healthy_recovers() {
        let state = HealthState::new();
        state.mark_unhealthy("dep1", Some("connect refused".to_string()));
        assert!(!state.is_healthy("dep1", STALENESS));
        state.mark_healthy("dep1");
        assert!(state.is_healthy("dep1", STALENESS));
        assert!(state.is_empty(), "mark_healthy clears the entry");
    }

    #[test]
    fn stale_entry_fails_open() {
        let state = HealthState::new();
        state.mark_unhealthy("dep1", None);
        // A zero staleness threshold makes every entry instantly stale.
        assert!(
            state.is_healthy("dep1", Duration::ZERO),
            "stale entries must read as healthy (fail-open)"
        );
        assert!(state.is_empty(), "stale entry is lazily removed on access");
    }

    #[test]
    fn healthy_deployments_filters_unhealthy() {
        let state = HealthState::new();
        state.mark_unhealthy("b", Some("dns failure".to_string()));
        let out = state.healthy_deployments(vec!["a", "b", "c"], |s| *s, STALENESS);
        assert_eq!(out, vec!["a", "c"]);
    }

    #[test]
    fn healthy_deployments_all_unhealthy_returns_all() {
        // Safety net: marking every candidate unhealthy must not empty the
        // routing pool — degraded deployments are better than none.
        let state = HealthState::new();
        state.mark_unhealthy("a", None);
        state.mark_unhealthy("b", None);
        let out = state.healthy_deployments(vec!["a", "b"], |s| *s, STALENESS);
        assert_eq!(out, vec!["a", "b"]);
    }

    #[test]
    fn healthy_deployments_empty_input_stays_empty() {
        let state = HealthState::new();
        let out: Vec<&str> = state.healthy_deployments(Vec::new(), |s: &&str| *s, STALENESS);
        assert!(out.is_empty());
    }

    #[test]
    fn record_failure_marks_unhealthy_at_threshold() {
        let state = HealthState::new();
        assert!(
            !state.record_failure("dep1", "timeout", 3),
            "first failure must not mark unhealthy"
        );
        assert!(state.is_healthy("dep1", STALENESS));
        assert!(!state.record_failure("dep1", "timeout", 3));
        assert!(
            state.record_failure("dep1", "timeout", 3),
            "third consecutive failure reaches the threshold"
        );
        assert!(!state.is_healthy("dep1", STALENESS));
    }

    #[test]
    fn record_failure_reports_newly_unhealthy_only_on_transition() {
        let state = HealthState::new();
        assert!(!state.record_failure("dep1", "timeout", 2));
        assert!(
            state.record_failure("dep1", "timeout", 2),
            "the threshold-crossing call reports the transition"
        );
        assert!(
            !state.record_failure("dep1", "timeout", 2),
            "already-unhealthy failures are not newly unhealthy"
        );
        // Recovery resets the episode: the next crossing is newly unhealthy
        // again.
        state.mark_healthy("dep1");
        assert!(!state.record_failure("dep1", "timeout", 2));
        assert!(
            state.record_failure("dep1", "timeout", 2),
            "a new episode reports the transition again"
        );
    }

    #[test]
    fn reason_returns_recorded_reason_for_tracked_entries() {
        let state = HealthState::new();
        assert_eq!(state.reason("dep1"), None, "unknown key has no reason");
        state.record_failure("dep1", "tls error", 1);
        assert_eq!(state.reason("dep1").as_deref(), Some("tls error"));
        state.mark_healthy("dep1");
        assert_eq!(state.reason("dep1"), None, "recovered entry is removed");
    }

    #[test]
    fn reason_preserves_threshold_crossing_cause() {
        // The reason must capture the failure that crossed the threshold
        // (the one an operator wants for the alert), and must NOT be
        // overwritten by later below/above-threshold failures.
        let state = HealthState::new();
        // Below-threshold failures with different causes…
        assert!(!state.record_failure("dep1", "timeout", 3));
        assert!(!state.record_failure("dep1", "refused", 3));
        // …then the threshold-crossing failure with a third cause.
        assert!(state.record_failure("dep1", "dns", 3));
        assert_eq!(
            state.reason("dep1").as_deref(),
            Some("dns"),
            "reason must be the transition cause"
        );
        // A subsequent failure (already unhealthy) must not overwrite it.
        assert!(!state.record_failure("dep1", "tls", 3));
        assert_eq!(
            state.reason("dep1").as_deref(),
            Some("dns"),
            "later failures must not overwrite the transition reason"
        );
    }

    #[test]
    fn record_failure_threshold_one_marks_immediately() {
        let state = HealthState::new();
        assert!(state.record_failure("dep1", "tls error", 1));
        assert!(!state.is_healthy("dep1", STALENESS));
    }

    #[test]
    fn success_after_failures_resets_consecutive_count() {
        let state = HealthState::new();
        assert!(!state.record_failure("dep1", "timeout", 3));
        assert!(!state.record_failure("dep1", "timeout", 3));
        state.mark_healthy("dep1");
        // After a success the count restarts: two more failures are still
        // below the threshold of three.
        assert!(!state.record_failure("dep1", "timeout", 3));
        assert!(!state.record_failure("dep1", "timeout", 3));
        assert!(state.is_healthy("dep1", STALENESS));
    }

    #[test]
    fn failures_below_threshold_do_not_refresh_staleness() {
        // The staleness clock starts at entry creation and is NOT bumped
        // by below-threshold failures, so a trickle of failures cannot
        // keep the entry alive forever (slow memory leak).
        let state = HealthState::new();
        let staleness = Duration::from_millis(50);
        assert!(!state.record_failure("dep1", "timeout", 5));
        std::thread::sleep(Duration::from_millis(30));
        // Second failure ~30ms later, still below the threshold. If it
        // refreshed the timestamp, the entry would survive until ~80ms.
        assert!(!state.record_failure("dep1", "timeout", 5));
        std::thread::sleep(Duration::from_millis(30));
        // t≈60ms: stale from the entry's original creation time.
        assert!(
            state.is_healthy("dep1", staleness),
            "stale below-threshold entry must read as healthy (fail-open)"
        );
        assert_eq!(
            state.len(),
            0,
            "below-threshold failures must not extend an entry's staleness lifetime"
        );
    }

    #[test]
    fn health_flapping_cycles_between_healthy_and_unhealthy() {
        let state = HealthState::new();
        // Unknown key starts healthy.
        assert!(state.is_healthy("dep1", STALENESS));
        // Fail up to the threshold → unhealthy.
        assert!(!state.record_failure("dep1", "timeout", 2));
        assert!(state.record_failure("dep1", "timeout", 2));
        assert!(!state.is_healthy("dep1", STALENESS));
        // A success restores health and resets the consecutive count.
        state.mark_healthy("dep1");
        assert!(state.is_healthy("dep1", STALENESS));
        // Failing again: the count restarted, so one failure is below the
        // threshold of two.
        assert!(
            !state.record_failure("dep1", "timeout", 2),
            "consecutive_failures must reset on success"
        );
        assert!(state.is_healthy("dep1", STALENESS));
        {
            let entries = state.entries.lock().unwrap_or_else(|e| e.into_inner());
            assert_eq!(
                entries.get("dep1").map(|e| e.consecutive_failures),
                Some(1),
                "post-success failures count from zero again"
            );
        }
        // And the deployment can go unhealthy once more.
        assert!(state.record_failure("dep1", "timeout", 2));
        assert!(!state.is_healthy("dep1", STALENESS));
    }

    #[test]
    fn mark_unhealthy_preserves_failure_count() {
        let state = HealthState::new();
        state.record_failure("dep1", "timeout", 5);
        state.mark_unhealthy("dep1", Some("manual".to_string()));
        let entries = state.entries.lock().unwrap_or_else(|e| e.into_inner());
        assert_eq!(
            entries.get("dep1").map(|e| e.consecutive_failures),
            Some(1),
            "an explicit mark must not reset the consecutive-failure count"
        );
    }

    #[test]
    fn clear_removes_all() {
        let state = HealthState::new();
        state.mark_unhealthy("a", None);
        state.mark_unhealthy("b", None);
        state.clear();
        assert!(state.is_empty());
        assert!(state.is_healthy("a", STALENESS));
        assert!(state.is_healthy("b", STALENESS));
    }

    #[test]
    fn retain_keys_drops_unknown() {
        let state = HealthState::new();
        state.mark_unhealthy("keep:0", None);
        state.mark_unhealthy("drop:1", None);
        let known: std::collections::HashSet<String> = ["keep:0".to_string()].into_iter().collect();
        state.retain_keys(&known);
        assert!(!state.is_healthy("keep:0", STALENESS));
        assert!(
            state.is_healthy("drop:1", STALENESS),
            "dropped key reads healthy"
        );
        assert_eq!(state.len(), 1);
    }
}
