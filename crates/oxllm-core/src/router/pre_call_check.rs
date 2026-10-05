//! The pre-call check framework: pluggable candidate filters that run
//! before the routing strategy picks a deployment.
//!
//! A [`PreCallCheck`] inspects the candidate deployment list (already
//! filtered to the requested model and past cooldown) and returns the
//! deployments that should remain eligible. Checks run sequentially in the
//! order named by [`crate::router::RouterConfig::optional_pre_call_checks`];
//! each check sees the output of the previous one.
//!
//! The framework is synchronous and pure — the router never performs I/O.
//! Checks that need asynchronously-resolved data (cache lookups, spend
//! counters) read it from [`super::RequestContext::pre_call`], which the
//! host populates ahead of the pick.
//!
//! Pipeline semantics (enforced by the router, see
//! [`crate::router::Router`]):
//!
//! - **Ordering**: checks run in config order. The built-in health check
//!   runs at its listed position, or last when health-check routing is
//!   enabled without being listed.
//! - **Fail-open**: a check that returns `Err` is skipped — its input
//!   candidate list flows to the next check unchanged (routing availability
//!   beats check correctness).
//! - **Safety net**: a check that filters out *every* candidate is treated
//!   as a no-op — its input list is restored and passed on. A buggy or
//!   over-aggressive check must never turn "deployments exist" into "no
//!   deployment available".

use std::sync::Arc;
use std::time::Duration;

use crate::deployment::Deployment;
use crate::router::health::HealthState;
use crate::router::strategy::RequestContext;

/// A routing pre-call check: filters the candidate deployment list before
/// the strategy picks.
///
/// Implementations must be cheap and non-blocking; the router calls them
/// synchronously on every pick. `name()` is the identifier used in
/// `optional_pre_call_checks` config and in `pre_call_check` tracing events
/// (snake_case, e.g. `"prompt_caching"`, `"budget_limiter"`,
/// `"health_check"`).
pub trait PreCallCheck: Send + Sync + std::fmt::Debug {
    /// Filter `candidates` down to the deployments that should remain
    /// eligible for this request. Returning the input unchanged (or a
    /// superset reordering) is always legal.
    ///
    /// The router guarantees that
    /// [`RequestContext::pre_call.candidate_keys`][super::strategy::PreCallData::candidate_keys]
    /// is aligned by position with `candidates` at call time.
    fn filter_deployments<'a>(
        &self,
        candidates: &[&'a Deployment],
        ctx: &RequestContext,
    ) -> Result<Vec<&'a Deployment>, crate::Error>;

    /// The check's config/tracing name.
    fn name(&self) -> &'static str;
}

/// The built-in health-check routing filter.
///
/// Drops candidates whose connectivity is degraded (tracked by
/// [`HealthState`], keyed by the same cooldown-style keys the router uses).
/// This is the pre-call-check form of the router's health routing; the
/// router constructs and injects it automatically when
/// `enable_health_check_routing` is set.
///
/// Like [`HealthState::healthy_deployments`], the filter never empties the
/// pool on its own: when every candidate is unhealthy, all candidates pass
/// (a degraded deployment beats no deployment).
#[derive(Debug, Clone)]
pub struct HealthCheck {
    health: Arc<HealthState>,
    staleness: Duration,
}

impl HealthCheck {
    /// Create a health check over the shared health state. `staleness` is
    /// how long an entry stays authoritative before the deployment reads as
    /// healthy again (fail-open).
    pub fn new(health: Arc<HealthState>, staleness: Duration) -> Self {
        Self { health, staleness }
    }
}

impl PreCallCheck for HealthCheck {
    fn filter_deployments<'a>(
        &self,
        candidates: &[&'a Deployment],
        ctx: &RequestContext,
    ) -> Result<Vec<&'a Deployment>, crate::Error> {
        let keys = &ctx.pre_call.candidate_keys;
        if keys.len() != candidates.len() {
            // Keys are router-injected and must align; a mismatch means the
            // caller bypassed the router pipeline. Fail open.
            return Ok(candidates.to_vec());
        }
        let pairs: Vec<(&'a Deployment, &str)> = candidates
            .iter()
            .zip(keys.iter())
            .map(|(d, k)| (*d, k.as_str()))
            .collect();
        // The closure returns the existing `&str` slice from `keys` with no
        // per-candidate allocation (the key_fn signature is `&str`, not
        // `String`). On a 1000-RPS hot path with N candidates this saves N
        // allocations/sec purely for the health-key lookup.
        let kept = self
            .health
            .healthy_deployments(pairs, |(_, key)| *key, self.staleness);
        Ok(kept.into_iter().map(|(d, _)| d).collect())
    }

    fn name(&self) -> &'static str {
        "health_check"
    }
}

/// Normalize a configured check name for comparison: kebab-case and
/// snake_case spellings of the same check are equivalent
/// (`"health-check"` ≡ `"health_check"`).
pub fn normalize_check_name(name: &str) -> String {
    name.replace('-', "_")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deployment::OxllmParams;

    fn make_dep(name: &str) -> Deployment {
        Deployment {
            model_name: name.to_string(),
            oxllm_params: OxllmParams {
                model: format!("openai/{name}"),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn ctx_with_keys(keys: &[&str]) -> RequestContext {
        let mut ctx = RequestContext::new("gpt-4o");
        ctx.pre_call.candidate_keys = keys.iter().map(|s| s.to_string()).collect();
        ctx
    }

    const STALENESS: Duration = Duration::from_secs(60);

    #[test]
    fn health_check_filters_unhealthy_candidates() {
        let health = Arc::new(HealthState::new());
        health.mark_unhealthy("gpt-4o:1", Some("dns failure".to_string()));
        let check = HealthCheck::new(health, STALENESS);
        let deps = [make_dep("gpt-4o"), make_dep("gpt-4o"), make_dep("gpt-4o")];
        let candidates: Vec<&Deployment> = deps.iter().collect();
        let ctx = ctx_with_keys(&["gpt-4o:0", "gpt-4o:1", "gpt-4o:2"]);
        let out = check
            .filter_deployments(&candidates, &ctx)
            .expect("health check is infallible");
        assert_eq!(out.len(), 2);
        assert!(std::ptr::eq(out[0], &deps[0]));
        assert!(std::ptr::eq(out[1], &deps[2]));
    }

    #[test]
    fn health_check_all_unhealthy_returns_all() {
        // Safety net: an all-unhealthy pool must not empty the candidate
        // list — degraded deployments are better than none.
        let health = Arc::new(HealthState::new());
        health.mark_unhealthy("gpt-4o:0", None);
        health.mark_unhealthy("gpt-4o:1", None);
        let check = HealthCheck::new(health, STALENESS);
        let deps = [make_dep("gpt-4o"), make_dep("gpt-4o")];
        let candidates: Vec<&Deployment> = deps.iter().collect();
        let ctx = ctx_with_keys(&["gpt-4o:0", "gpt-4o:1"]);
        let out = check
            .filter_deployments(&candidates, &ctx)
            .expect("health check is infallible");
        assert_eq!(out.len(), 2, "all-unhealthy must return all candidates");
    }

    #[test]
    fn health_check_misaligned_keys_passes_through() {
        let health = Arc::new(HealthState::new());
        health.mark_unhealthy("gpt-4o:0", None);
        let check = HealthCheck::new(health, STALENESS);
        let deps = [make_dep("gpt-4o")];
        let candidates: Vec<&Deployment> = deps.iter().collect();
        // No candidate keys populated (e.g. a host calling the check
        // outside the router pipeline) → fail open.
        let ctx = RequestContext::new("gpt-4o");
        let out = check
            .filter_deployments(&candidates, &ctx)
            .expect("health check is infallible");
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn health_check_stale_entries_read_healthy() {
        let health = Arc::new(HealthState::new());
        health.mark_unhealthy("gpt-4o:0", None);
        // Zero staleness: every entry is instantly stale → fail-open.
        let check = HealthCheck::new(health, Duration::ZERO);
        let deps = [make_dep("gpt-4o")];
        let candidates: Vec<&Deployment> = deps.iter().collect();
        let ctx = ctx_with_keys(&["gpt-4o:0"]);
        let out = check
            .filter_deployments(&candidates, &ctx)
            .expect("health check is infallible");
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn health_check_name_is_stable() {
        let check = HealthCheck::new(Arc::new(HealthState::new()), STALENESS);
        assert_eq!(check.name(), "health_check");
    }

    #[test]
    fn normalize_check_name_accepts_kebab_and_snake() {
        assert_eq!(normalize_check_name("health-check"), "health_check");
        assert_eq!(normalize_check_name("health_check"), "health_check");
        assert_eq!(normalize_check_name("budget-limiter"), "budget_limiter");
    }
}
