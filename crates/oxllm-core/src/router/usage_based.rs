//! `UsageBased` strategy — picks the deployment with the lowest current usage.

use std::sync::Arc;

use crate::deployment::Deployment;
use crate::router::simple_shuffle::SimpleShuffle;
use crate::router::stores::UsageStore;
use crate::router::strategy::{RequestContext, RoutingStrategy, candidate_keys};

/// Picks the deployment with the lowest usage score from a [`UsageStore`].
///
/// The usage score (0.0–1.0+) represents how close the deployment is to its
/// RPM/TPM limit. A score of 0.0 means no usage. If the store has no entries
/// (no usage data), falls back to [`SimpleShuffle`].
#[derive(Debug)]
pub struct UsageBased {
    store: Arc<UsageStore>,
}

impl UsageBased {
    /// Create a new strategy backed by the given usage store.
    pub fn new(store: Arc<UsageStore>) -> Self {
        Self { store }
    }
}

impl RoutingStrategy for UsageBased {
    fn pick<'a>(
        &self,
        deployments: &[&'a Deployment],
        ctx: &RequestContext,
    ) -> Option<&'a Deployment> {
        if deployments.is_empty() {
            return None;
        }
        let mut any_data = false;
        let mut best: Option<(&Deployment, f64)> = None;
        let keys = candidate_keys(deployments, ctx);
        for (dep, key) in deployments.iter().zip(keys.iter()) {
            let raw = self.store.get(key);
            // Bug #37: when the deployment publishes an RPM or TPM limit,
            // normalise the raw counter to a utilization ratio
            // (`count / limit`). Without this, a high-RPM deployment at a
            // modest request count would lose to a low-RPM deployment at a
            // similar count, even though the former is barely loaded. If no
            // limit is published, fall back to the raw count (legacy).
            let usage = compute_utilization(dep, raw);
            // Distinguish "no data at all" from "zero usage". A recorded 0.0
            // is real signal (the deployment is freshly idle), so it counts
            // as data; an absent entry means the store has never heard of
            // this deployment, so we fall back to shuffle. The previous
            // `usage > 0.0` check conflated the two and dropped the 0.0 case.
            if self.store.contains(key) {
                any_data = true;
            }
            match &best {
                None => best = Some((dep, usage)),
                Some((_, bu)) if usage < *bu => best = Some((dep, usage)),
                _ => {}
            }
        }
        if !any_data {
            return SimpleShuffle.pick(deployments, ctx);
        }
        best.map(|(dep, _)| dep)
    }
}

/// Convert a raw usage counter to a utilization ratio when the deployment
/// publishes an RPM limit: the host snapshots the `dep_rpm:{...}` REQUEST
/// counter into the store, so `count / rpm` is the request-utilization
/// ratio. A `tpm` limit is deliberately NOT used as a divisor — the raw
/// counter is a request count, and `requests / tokens-per-minute` mixes
/// units by orders of magnitude, tying every tpm-only candidate at ≈0 and
/// pinning the pick to the first-listed deployment. TPM-only deployments
/// fall back to comparing raw counts (fewer recorded requests wins).
fn compute_utilization(dep: &Deployment, raw: f64) -> f64 {
    if let Some(rpm) = dep.oxllm_params.rpm {
        if rpm > 0 {
            return raw / rpm as f64;
        }
    }
    raw
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deployment::OxllmParams;
    use crate::router::stores::stable_key;

    fn make_dep(name: &str, model: &str) -> Deployment {
        Deployment {
            model_name: name.to_string(),
            oxllm_params: OxllmParams {
                model: model.to_string(),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn make_dep_with_rpm(name: &str, model: &str, rpm: u32) -> Deployment {
        let mut dep = make_dep(name, model);
        dep.oxllm_params.rpm = Some(rpm);
        dep
    }

    #[test]
    fn empty_returns_none() {
        let store = Arc::new(UsageStore::new());
        let strategy = UsageBased::new(store);
        let ctx = RequestContext::new("model");
        assert!(strategy.pick(&[], &ctx).is_none());
    }

    #[test]
    fn picks_least_loaded() {
        let store = Arc::new(UsageStore::new());
        let loaded = make_dep("model", "loaded");
        let idle = make_dep("model", "idle");
        store.record(&stable_key(&loaded), 0.9);
        store.record(&stable_key(&idle), 0.1);
        let strategy = UsageBased::new(store);
        let ctx = RequestContext::new("model");
        let deps = [loaded, idle];
        let refs: Vec<&Deployment> = deps.iter().collect();
        let picked = strategy.pick(&refs, &ctx).expect("should pick");
        assert_eq!(picked.oxllm_params.model, "idle");
    }

    #[test]
    fn no_usage_data_falls_back_to_shuffle() {
        let store = Arc::new(UsageStore::new());
        let strategy = UsageBased::new(store);
        let ctx = RequestContext::new("model");
        let deps = [make_dep("model", "a"), make_dep("model", "b")];
        let refs: Vec<&Deployment> = deps.iter().collect();
        assert!(strategy.pick(&refs, &ctx).is_some());
    }

    /// Bug #37: with `rpm` limits published, the strategy must normalise the
    /// raw counter to `count / limit`. A high-RPM deployment at a modest
    /// request count must beat a low-RPM deployment at the same count, since
    /// the former is far less utilized.
    #[test]
    fn utilization_prefers_lower_utilization_when_limits_set() {
        let store = Arc::new(UsageStore::new());
        // Both deployments have served 50 requests this minute.
        let small = make_dep_with_rpm("model", "small", 100);
        let large = make_dep_with_rpm("model", "large", 10_000);
        store.record(&stable_key(&small), 50.0);
        store.record(&stable_key(&large), 50.0);
        let strategy = UsageBased::new(store);
        let ctx = RequestContext::new("model");
        let deps = [small, large];
        let refs: Vec<&Deployment> = deps.iter().collect();
        let picked = strategy.pick(&refs, &ctx).expect("should pick");
        assert_eq!(
            picked.oxllm_params.model, "large",
            "rpm=10_000 at 50 req (0.5% utilized) must beat rpm=100 at 50 req (50% utilized)"
        );
    }

    /// Bug #37: when no RPM/TPM limit is published, the strategy must
    /// preserve the legacy raw-count comparison.
    #[test]
    fn utilization_without_limit_falls_back_to_raw_count() {
        let store = Arc::new(UsageStore::new());
        let busy = make_dep("model", "busy");
        let idle = make_dep("model", "idle");
        store.record(&stable_key(&busy), 100.0);
        store.record(&stable_key(&idle), 5.0);
        let strategy = UsageBased::new(store);
        let ctx = RequestContext::new("model");
        let deps = [busy, idle];
        let refs: Vec<&Deployment> = deps.iter().collect();
        let picked = strategy.pick(&refs, &ctx).expect("should pick");
        assert_eq!(picked.oxllm_params.model, "idle");
    }

    /// Bug #37: utilisation should still pick the lower-utilization
    /// deployment when the raw counts differ but ratios are inverted.
    #[test]
    fn utilization_uses_ratio_not_raw_count() {
        let store = Arc::new(UsageStore::new());
        // big: 40 req / 10_000 rpm = 0.4%
        // small: 40 req / 100 rpm = 40%
        let small = make_dep_with_rpm("model", "small", 100);
        let big = make_dep_with_rpm("model", "big", 10_000);
        store.record(&stable_key(&small), 40.0);
        store.record(&stable_key(&big), 40.0);
        let strategy = UsageBased::new(store);
        let ctx = RequestContext::new("model");
        let deps = [small, big];
        let refs: Vec<&Deployment> = deps.iter().collect();
        let picked = strategy.pick(&refs, &ctx).expect("should pick");
        assert_eq!(picked.oxllm_params.model, "big");
    }

    /// TPM-only deployments must not normalize the request counter by a
    /// token limit: `requests / tpm` mixes units by ~3 orders of magnitude,
    /// tying every tpm-only candidate at ≈0 and deterministically pinning
    /// the first-listed deployment. The raw counts must decide.
    #[test]
    fn tpm_only_deployments_compare_raw_counts() {
        let store = Arc::new(UsageStore::new());
        let mut tiny = make_dep("model", "tiny");
        tiny.oxllm_params.tpm = Some(10_000);
        let mut huge = make_dep("model", "huge");
        huge.oxllm_params.tpm = Some(1_000_000);
        // tiny has served 2 requests, huge 100 — under the old `raw / tpm`
        // normalization huge won (100/1M < 2/10k) despite 50x the load.
        store.record(&stable_key(&tiny), 2.0);
        store.record(&stable_key(&huge), 100.0);
        let strategy = UsageBased::new(store);
        let ctx = RequestContext::new("model");
        let deps = [tiny, huge];
        let refs: Vec<&Deployment> = deps.iter().collect();
        let picked = strategy.pick(&refs, &ctx).expect("should pick");
        assert_eq!(
            picked.oxllm_params.model, "tiny",
            "tpm-only deployments must compare the raw request counts"
        );
    }

    /// L3: `any_data` must distinguish "no entry" from "entry whose usage is
    /// exactly 0.0". A recorded 0.0 is real signal (the deployment is freshly
    /// idle and should win), so the strategy must NOT fall back to shuffle in
    /// that case. Previously `usage > 0.0` conflated the two, dropping the
    /// zero-usage signal and falling back to shuffle.
    #[test]
    fn recorded_zero_usage_is_treated_as_data_not_none() {
        let store = Arc::new(UsageStore::new());
        // `known` is explicitly recorded at 0.0; `unknown` has no entry.
        let known = make_dep("model", "known");
        store.record(&stable_key(&known), 0.0);
        let strategy = UsageBased::new(store.clone());
        let ctx = RequestContext::new("model");
        let deps = [make_dep("model", "known"), make_dep("model", "unknown")];
        let refs: Vec<&Deployment> = deps.iter().collect();

        // The store must agree `known` exists (distinct from `get` returning 0.0).
        assert!(store.contains(&stable_key(&known)));
        assert!(!store.contains(&stable_key(&make_dep("model", "unknown"))));

        // Because there IS data, the strategy must deterministically pick the
        // least-loaded deployment (`known`, at 0.0) rather than fall back to
        // shuffle. If `any_data` had stayed false, we would have shuffled.
        let picked = strategy.pick(&refs, &ctx).expect("should pick");
        assert_eq!(
            picked.oxllm_params.model, "known",
            "a recorded 0.0 is real data and must win, not trigger shuffle fallback"
        );
    }
}
