//! `LeastBusy` strategy — picks the deployment with the fewest in-flight requests.
//!
//! The host (chat handler) wraps the upstream call with inc/dec on the shared
//! [`UsageStore`]; this strategy reads those counts to prefer deployments that
//! are not currently serving other requests. When no candidate has any
//! in-flight traffic (start-up, cold cache), it falls back to
//! [`SimpleShuffle`] so the request still proceeds.

use std::sync::Arc;

use crate::deployment::Deployment;
use crate::router::simple_shuffle::SimpleShuffle;
use crate::router::stores::UsageStore;
use crate::router::strategy::{RequestContext, RoutingStrategy, candidate_keys};

/// Picks the deployment with the lowest in-flight request count from a
/// [`UsageStore`].
#[derive(Debug)]
pub struct LeastBusy {
    store: Arc<UsageStore>,
}

impl LeastBusy {
    /// Create a new strategy backed by the given usage store.
    pub fn new(store: Arc<UsageStore>) -> Self {
        Self { store }
    }
}

impl RoutingStrategy for LeastBusy {
    fn pick<'a>(
        &self,
        deployments: &[&'a Deployment],
        ctx: &RequestContext,
    ) -> Option<&'a Deployment> {
        if deployments.is_empty() {
            return None;
        }
        let mut any_busy = false;
        let mut best: Option<(&Deployment, i64)> = None;
        let keys = candidate_keys(deployments, ctx);
        for (dep, key) in deployments.iter().zip(keys.iter()) {
            let count = self.store.in_flight(key);
            if count > 0 {
                any_busy = true;
            }
            match &best {
                None => best = Some((dep, count)),
                Some((_, bc)) if count < *bc => best = Some((dep, count)),
                _ => {}
            }
        }
        if !any_busy {
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
        let store = Arc::new(UsageStore::new());
        let strategy = LeastBusy::new(store);
        let ctx = RequestContext::new("model");
        assert!(strategy.pick(&[], &ctx).is_none());
    }

    #[test]
    fn picks_idle_deployment() {
        let store = Arc::new(UsageStore::new());
        let busy = make_dep("model", "busy");
        let idle = make_dep("model", "idle");
        store.incr_in_flight(&stable_key(&busy), 5);
        store.incr_in_flight(&stable_key(&idle), 0);
        let strategy = LeastBusy::new(store);
        let ctx = RequestContext::new("model");
        let deps = [busy, idle];
        let refs: Vec<&Deployment> = deps.iter().collect();
        let picked = strategy.pick(&refs, &ctx).expect("should pick");
        assert_eq!(picked.oxllm_params.model, "idle");
    }

    #[test]
    fn no_in_flight_data_falls_back_to_shuffle() {
        let store = Arc::new(UsageStore::new());
        let strategy = LeastBusy::new(store);
        let ctx = RequestContext::new("model");
        let deps = [make_dep("model", "a"), make_dep("model", "b")];
        let refs: Vec<&Deployment> = deps.iter().collect();
        assert!(strategy.pick(&refs, &ctx).is_some());
    }

    #[test]
    fn picks_lowest_count_when_all_busy() {
        let store = Arc::new(UsageStore::new());
        let a = make_dep("model", "a");
        let b = make_dep("model", "b");
        let c = make_dep("model", "c");
        store.incr_in_flight(&stable_key(&a), 3);
        store.incr_in_flight(&stable_key(&b), 1);
        store.incr_in_flight(&stable_key(&c), 2);
        let strategy = LeastBusy::new(store);
        let ctx = RequestContext::new("model");
        let deps = [a, b, c];
        let refs: Vec<&Deployment> = deps.iter().collect();
        let picked = strategy.pick(&refs, &ctx).expect("should pick");
        assert_eq!(picked.oxllm_params.model, "b");
    }

    #[test]
    fn pick_uses_precomputed_candidate_keys_when_aligned() {
        // The router memoizes cooldown keys per pick
        // (`PreCallData::candidate_keys`, position-aligned); the strategy
        // must consume them instead of re-serializing each candidate's
        // params. The store lookups follow the provided keys even when they
        // disagree with `stable_key(deployment)` — recording in-flight
        // counts under the precomputed keys must steer the pick.
        let store = Arc::new(UsageStore::new());
        let a = make_dep("model", "a");
        let b = make_dep("model", "b");
        store.incr_in_flight("precomputed-a", 5);
        store.incr_in_flight("precomputed-b", 0);
        let strategy = LeastBusy::new(store);
        let mut ctx = RequestContext::new("model");
        ctx.pre_call.candidate_keys =
            vec!["precomputed-a".to_string(), "precomputed-b".to_string()];
        // Identity fingerprints must match the deployments for the keys to
        // be trusted (see `candidate_keys`); otherwise the strategy falls
        // back to fresh stable-key computation.
        ctx.pre_call.candidate_ids = vec!["model:a".to_string(), "model:b".to_string()];
        let deps = [a, b];
        let refs: Vec<&Deployment> = deps.iter().collect();
        let picked = strategy.pick(&refs, &ctx).expect("should pick");
        assert_eq!(
            picked.oxllm_params.model, "b",
            "aligned candidate_keys must drive the store lookups"
        );
    }
}
