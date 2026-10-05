//! `LatencyBased` strategy — picks the deployment with the lowest EWMA latency.

use std::sync::Arc;

use crate::deployment::Deployment;
use crate::router::simple_shuffle::SimpleShuffle;
use crate::router::stores::LatencyStore;
use crate::router::strategy::{RequestContext, RoutingStrategy, candidate_keys};

/// Picks the deployment with the lowest EWMA (exponentially weighted moving
/// average) latency from a [`LatencyStore`].
///
/// Deployments without latency data are treated as having a latency of 0
/// (preferred). Falls back to [`SimpleShuffle`] if all candidates are unsampled.
#[derive(Debug)]
pub struct LatencyBased {
    store: Arc<LatencyStore>,
}

impl LatencyBased {
    /// Create a new strategy backed by the given latency store.
    pub fn new(store: Arc<LatencyStore>) -> Self {
        Self { store }
    }
}

impl RoutingStrategy for LatencyBased {
    fn pick<'a>(
        &self,
        deployments: &[&'a Deployment],
        ctx: &RequestContext,
    ) -> Option<&'a Deployment> {
        if deployments.is_empty() {
            return None;
        }
        let mut any_sampled = false;
        let mut best: Option<(&Deployment, f64)> = None;
        let keys = candidate_keys(deployments, ctx);
        for (dep, key) in deployments.iter().zip(keys.iter()) {
            // Unsampled deployments score as 0.0 so they are always preferred
            // over any sampled (warm) deployment, matching the strategy docs.
            let ewma = match self.store.ewma(key) {
                Some(e) => {
                    any_sampled = true;
                    e
                }
                None => 0.0,
            };
            match &best {
                None => best = Some((dep, ewma)),
                Some((_, be)) if ewma < *be => best = Some((dep, ewma)),
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
        let strategy = LatencyBased::new(store);
        let ctx = RequestContext::new("model");
        assert!(strategy.pick(&[], &ctx).is_none());
    }

    #[test]
    fn picks_lowest_ewma() {
        let store = Arc::new(LatencyStore::default());
        let slow = make_dep("model", "slow");
        let fast = make_dep("model", "fast");
        for _ in 0..10 {
            store.record(&stable_key(&slow), 500);
            store.record(&stable_key(&fast), 50);
        }
        let strategy = LatencyBased::new(store);
        let ctx = RequestContext::new("model");
        let deps = [slow, fast];
        let refs: Vec<&Deployment> = deps.iter().collect();
        let picked = strategy.pick(&refs, &ctx).expect("should pick");
        assert_eq!(picked.oxllm_params.model, "fast");
    }

    #[test]
    fn no_samples_falls_back_to_shuffle() {
        let store = Arc::new(LatencyStore::default());
        let strategy = LatencyBased::new(store);
        let ctx = RequestContext::new("model");
        let deps = [make_dep("model", "a")];
        let refs: Vec<&Deployment> = deps.iter().collect();
        assert!(strategy.pick(&refs, &ctx).is_some());
    }
}
