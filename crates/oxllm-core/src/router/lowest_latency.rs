//! `LowestLatency` strategy — picks the deployment with the lowest p50 latency.

use std::sync::Arc;

use crate::deployment::Deployment;
use crate::router::simple_shuffle::SimpleShuffle;
use crate::router::stores::LatencyStore;
use crate::router::strategy::{RequestContext, RoutingStrategy, candidate_keys};

/// Picks the deployment with the lowest p50 latency from a [`LatencyStore`].
///
/// Deployments without latency data are treated as having a latency of 0
/// (preferred, since they haven't been tried yet). Falls back to
/// [`SimpleShuffle`] if all candidates are unsampled.
#[derive(Debug)]
pub struct LowestLatency {
    store: Arc<LatencyStore>,
}

impl LowestLatency {
    /// Create a new strategy backed by the given latency store.
    pub fn new(store: Arc<LatencyStore>) -> Self {
        Self { store }
    }
}

impl RoutingStrategy for LowestLatency {
    fn pick<'a>(
        &self,
        deployments: &[&'a Deployment],
        ctx: &RequestContext,
    ) -> Option<&'a Deployment> {
        if deployments.is_empty() {
            return None;
        }
        let mut any_sampled = false;
        let mut best: Option<(&Deployment, u64)> = None;
        let keys = candidate_keys(deployments, ctx);
        for (dep, key) in deployments.iter().zip(keys.iter()) {
            // Unsampled deployments score as 0 so they are always preferred
            // over any sampled (warm) deployment, matching the strategy docs.
            let p50 = match self.store.p50(key) {
                Some(p) => {
                    any_sampled = true;
                    p
                }
                None => 0,
            };
            match &best {
                None => best = Some((dep, p50)),
                Some((_, bp)) if p50 < *bp => best = Some((dep, p50)),
                _ => {}
            }
        }
        if !any_sampled {
            return SimpleShuffle.pick(deployments, ctx);
        }
        best.map(|(dep, _)| dep)
    }
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

    #[test]
    fn empty_returns_none() {
        let store = Arc::new(LatencyStore::default());
        let strategy = LowestLatency::new(store);
        let ctx = RequestContext::new("model");
        assert!(strategy.pick(&[], &ctx).is_none());
    }

    #[test]
    fn picks_fastest_deployment() {
        let store = Arc::new(LatencyStore::default());
        let slow = make_dep("model", "slow");
        let fast = make_dep("model", "fast");
        store.record(&stable_key(&slow), 500);
        store.record(&stable_key(&fast), 50);
        let strategy = LowestLatency::new(store);
        let ctx = RequestContext::new("model");
        let deps = [slow, fast];
        let refs: Vec<&Deployment> = deps.iter().collect();
        let picked = strategy.pick(&refs, &ctx).expect("should pick");
        assert_eq!(picked.oxllm_params.model, "fast");
    }

    #[test]
    fn no_samples_falls_back_to_shuffle() {
        let store = Arc::new(LatencyStore::default());
        let strategy = LowestLatency::new(store);
        let ctx = RequestContext::new("model");
        let deps = [make_dep("model", "a"), make_dep("model", "b")];
        let refs: Vec<&Deployment> = deps.iter().collect();
        assert!(strategy.pick(&refs, &ctx).is_some());
    }

    #[test]
    fn prefers_unsampled_over_sampled() {
        let store = Arc::new(LatencyStore::default());
        let warm = make_dep("model", "warm");
        store.record(&stable_key(&warm), 10);
        let strategy = LowestLatency::new(store);
        let ctx = RequestContext::new("model");
        // Warm first, cold second — cold must still win.
        let deps = [make_dep("model", "warm"), make_dep("model", "cold")];
        let refs: Vec<&Deployment> = deps.iter().collect();
        let picked = strategy.pick(&refs, &ctx).expect("should pick");
        assert_eq!(picked.oxllm_params.model, "cold");
    }

    #[test]
    fn sibling_deployments_share_no_latency_bucket() {
        // Two deployments with the same `model_name` + provider model but
        // different `api_base` must NOT share a latency bucket (a fast primary
        // and a slow fallback differ in endpoint but would collide under the
        // old `deployment_key`). Recording a slow sample for one sibling must
        // not make the other sibling's pick slow.
        let store = Arc::new(LatencyStore::default());
        let primary = Deployment {
            model_name: "model".to_string(),
            oxllm_params: OxllmParams {
                model: "openai/gpt-4o".to_string(),
                api_base: Some("https://primary".to_string()),
                ..Default::default()
            },
            ..Default::default()
        };
        let fallback = Deployment {
            model_name: "model".to_string(),
            oxllm_params: OxllmParams {
                model: "openai/gpt-4o".to_string(),
                api_base: Some("https://fallback".to_string()),
                ..Default::default()
            },
            ..Default::default()
        };
        assert_ne!(
            stable_key(&primary),
            stable_key(&fallback),
            "sibling deployments must get distinct store keys"
        );
        store.record(&stable_key(&fallback), 900);
        let strategy = LowestLatency::new(store);
        let ctx = RequestContext::new("model");
        let deps = [primary, fallback];
        let refs: Vec<&Deployment> = deps.iter().collect();
        let picked = strategy.pick(&refs, &ctx).expect("should pick");
        assert_eq!(
            picked.oxllm_params.api_base.as_deref(),
            Some("https://primary"),
            "the unsampled primary must win over the sampled slow fallback"
        );
    }
}
