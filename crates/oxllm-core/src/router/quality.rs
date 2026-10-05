//! `QualityRouter` strategy — picks the deployment with the highest configured
//! quality score.
//!
//! A simple, config-driven version: scores come from
//! `oxllm_settings.quality_scores` (a static map). A full LLM-judge-based
//! adaptive router is a future extension; the strategy hook is in place.
//!
//! Scores must be finite; a negative value is a config typo (the strategy
//! silently reads it as `0.0` and treats it as unscored — falling back to
//! `SimpleShuffle`), so the [`RouterConfig`] validation rejects non-finite
//! or negative entries at load. A configured score of exactly `0.0` is a
//! valid pin ("this deployment is the worst of the configured set, but the
//! operator did configure it"). It enables deterministic score selection;
//! it does not win ties against an unscored deployment's implicit `0.0`.
//! Ties keep the first candidate in declaration order.
//!
//! [`RouterConfig`]: crate::router::types::RouterConfig

use std::collections::HashMap;

use crate::deployment::Deployment;
use crate::router::simple_shuffle::SimpleShuffle;
use crate::router::strategy::{RequestContext, RoutingStrategy};

/// Picks the deployment with the highest quality score from a static map.
///
/// Scores range 0.0–1.0 (higher is better); a configured `0.0` counts as a
/// pinned score (distinct from "no score configured"). Deployments without
/// a configured score read as `0.0` but do NOT mark the candidate pool as
/// "scored". If no candidate carries a configured score, falls back to
/// [`SimpleShuffle`] so the router still returns a valid deployment.
/// Once the pool is scored, equal values keep the first candidate, including
/// a configured zero tied with an unscored candidate's implicit zero.
#[derive(Debug, Default)]
pub struct QualityRouter {
    scores: HashMap<String, f64>,
}

impl QualityRouter {
    /// Create a new quality router with the given score map.
    ///
    /// The map is keyed by the deployment's `oxllm_params.model` string.
    pub fn new(scores: HashMap<String, f64>) -> Self {
        Self { scores }
    }
}

impl RoutingStrategy for QualityRouter {
    fn pick<'a>(
        &self,
        deployments: &[&'a Deployment],
        ctx: &RequestContext,
    ) -> Option<&'a Deployment> {
        if deployments.is_empty() {
            return None;
        }
        let mut best: Option<(&Deployment, f64)> = None;
        let mut any_scored = false;
        for dep in deployments {
            // A configured entry (even `0.0`) marks the pool as scored; a
            // missing entry reads as 0.0 but does NOT set `any_scored`.
            // Non-finite and negative scores are skipped defensively
            // (validation rejects them at load; this guards runtime maps).
            let (score, configured) = match self.scores.get(&dep.oxllm_params.model) {
                Some(&v) if v.is_finite() && v >= 0.0 => (v, true),
                Some(_) => (0.0, false),
                None => (0.0, false),
            };
            if configured {
                any_scored = true;
            }
            match &best {
                None => best = Some((dep, score)),
                Some((_, bs)) if score > *bs => best = Some((dep, score)),
                _ => {}
            }
        }
        if !any_scored {
            return SimpleShuffle.pick(deployments, ctx);
        }
        best.map(|(dep, _)| dep)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deployment::OxllmParams;
    use serde_json::Map;

    fn make_dep(model: &str) -> Deployment {
        Deployment {
            model_name: "m".to_string(),
            oxllm_params: OxllmParams {
                model: model.to_string(),
                extra: Map::new(),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[test]
    fn empty_returns_none() {
        let strategy = QualityRouter::default();
        let ctx = RequestContext::new("m");
        assert!(strategy.pick(&[], &ctx).is_none());
    }

    #[test]
    fn picks_highest_score() {
        let mut scores = HashMap::new();
        scores.insert("openai/gpt-4o".to_string(), 0.9);
        scores.insert("openai/gpt-4o-mini".to_string(), 0.7);
        let strategy = QualityRouter::new(scores);
        let ctx = RequestContext::new("m");
        let deps = [make_dep("openai/gpt-4o-mini"), make_dep("openai/gpt-4o")];
        let refs: Vec<&Deployment> = deps.iter().collect();
        let picked = strategy.pick(&refs, &ctx).expect("should pick");
        assert_eq!(picked.oxllm_params.model, "openai/gpt-4o");
    }

    #[test]
    fn unscored_falls_back_to_shuffle() {
        let strategy = QualityRouter::default();
        let ctx = RequestContext::new("m");
        let deps = [make_dep("openai/unknown")];
        let refs: Vec<&Deployment> = deps.iter().collect();
        assert!(strategy.pick(&refs, &ctx).is_some());
    }

    #[test]
    fn mixed_picks_scored_over_unscored() {
        let mut scores = HashMap::new();
        scores.insert("openai/gpt-4o-mini".to_string(), 0.1);
        let strategy = QualityRouter::new(scores);
        let ctx = RequestContext::new("m");
        let deps = [make_dep("openai/unknown"), make_dep("openai/gpt-4o-mini")];
        let refs: Vec<&Deployment> = deps.iter().collect();
        let picked = strategy.pick(&refs, &ctx).expect("should pick");
        assert_eq!(picked.oxllm_params.model, "openai/gpt-4o-mini");
    }

    #[test]
    fn configured_zero_score_is_scored_not_unscored() {
        // A zero score alone must enable deterministic argmax. Weight the
        // unscored candidate at zero so shuffle would ALWAYS choose the
        // configured candidate instead: no random distribution assertion is
        // needed to distinguish the two behaviors.
        let strategy = QualityRouter::new(HashMap::from([("openai/scored-zero".to_string(), 0.0)]));
        let ctx = RequestContext::new("m");
        let mut unscored = make_dep("openai/unscored");
        unscored.oxllm_params.weight = Some(0);
        let scored = make_dep("openai/scored-zero");
        for refs in [[&unscored, &scored], [&scored, &unscored]] {
            let picked = strategy.pick(&refs, &ctx).expect("scored pool");
            assert!(
                std::ptr::eq(picked, refs[0]),
                "equal scores keep the first candidate"
            );
        }
        assert!(std::ptr::eq(
            QualityRouter::default()
                .pick(&[&unscored, &scored], &ctx)
                .expect("unscored pool"),
            &scored,
        ));
    }

    #[test]
    fn negative_score_reads_as_unscored() {
        // A negative score is a config typo; validation rejects it at load.
        // When a runtime-constructed map slips one through, the strategy
        // defensively skips the entry (reads as 0.0 and does NOT mark the
        // pool as scored), so this pool falls back to shuffle.
        let mut scores = HashMap::new();
        scores.insert("openai/negative".to_string(), -0.1);
        let strategy = QualityRouter::new(scores);
        let ctx = RequestContext::new("m");
        let mut other = make_dep("openai/other");
        other.oxllm_params.weight = Some(0);
        let deps = [make_dep("openai/negative"), other];
        let refs: Vec<&Deployment> = deps.iter().collect();
        assert_eq!(
            strategy
                .pick(&refs, &ctx)
                .expect("weighted shuffle")
                .oxllm_params
                .model,
            "openai/negative",
            "invalid scores must not activate argmax and select the implicit-zero sibling"
        );
    }

    #[test]
    fn non_finite_score_reads_as_unscored() {
        let mut scores = HashMap::new();
        scores.insert("openai/nan".to_string(), f64::NAN);
        scores.insert("openai/inf".to_string(), f64::INFINITY);
        let strategy = QualityRouter::new(scores);
        let ctx = RequestContext::new("m");
        let deps = [make_dep("openai/nan"), make_dep("openai/inf")];
        let refs: Vec<&Deployment> = deps.iter().collect();
        assert!(
            strategy.pick(&refs, &ctx).is_some(),
            "non-finite-score pool falls back to shuffle"
        );
    }
}
