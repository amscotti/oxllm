//! The default routing strategy: random shuffle.

use rand::Rng;
use rand::seq::IndexedRandom;

use crate::deployment::Deployment;
use crate::router::strategy::{RequestContext, RoutingStrategy};

/// The default routing strategy: picks a random deployment from the available
/// list. Uses `rand` for randomness.
///
/// When one or more deployments carry a capacity signal, the selection is
/// WEIGHTED random (`simple-shuffle` semantics): a deployment's weight
/// is `oxllm_params.weight`, falling back to `oxllm_params.rpm`, then to
/// `oxllm_params.tpm`, then to `1` — a deployment with weight 4 is picked
/// twice as often as one with weight 2. When no deployment carries a weight,
/// rpm, or tpm, behavior is uniform random.
///
/// Two deliberate divergences from the reference `simple_shuffle` are documented
/// here so they are not "fixed" by accident: (a) the metric is resolved
/// per-deployment (weight → rpm → tpm → 1), where the reference selects ONE metric
/// from the first candidate and applies it group-wide; and (b) a deployment
/// without `weight` falls back to its own `rpm`, then `tpm`, then `1` (per
/// `effective_weight`), instead of being excluded. The reference
/// `total_weight <= 0` path moves to the next metric group-wide and starves
/// any sibling missing it; oxllm's per-deployment fallback never starves an
/// otherwise-unweighted deployment, so operators porting a mixed-metric
/// existing config (e.g. one deployment with `weight=10`, another with only
/// `rpm=100`) will see the rpm-only deployment still receive traffic.
#[derive(Debug, Default)]
pub struct SimpleShuffle;

/// The effective selection weight of a deployment: the explicit `weight`
/// first, then the `rpm` capacity hint, then the `tpm` capacity hint, then
/// `1`.
///
/// Footgun: `weight: 0`, `rpm: 0`, or `tpm: 0` all mean "starve this
/// deployment" — an explicit `Some(0)` is *not* treated as "unset" (it
/// returns `0` from the `.unwrap_or(1)` fallback chain). A deployment with
/// `rpm: 0` under a `SimpleShuffle` group with any positive-weight sibling
/// receives no traffic as long as the sibling is up. This matches the reference
/// `total_weight <= 0` handling and is intentional for `weight: 0`; the
/// surprise is `rpm: 0`/`tpm: 0`, which an operator might set meaning "no
/// rate limit" or as a typo. `usage_based::compute_utilization` guards
/// division with `if rpm > 0`; this path does not (0 slots in the rotation
/// is the documented behavior). Pinned by
/// `rpm_zero_starves_deployment_under_simple_shuffle` and the round-robin
/// equivalent.
fn effective_weight(dep: &Deployment) -> u64 {
    dep.oxllm_params
        .weight
        .or(dep.oxllm_params.rpm)
        .or(dep.oxllm_params.tpm)
        .unwrap_or(1) as u64
}

impl RoutingStrategy for SimpleShuffle {
    fn pick<'a>(
        &self,
        deployments: &[&'a Deployment],
        _ctx: &RequestContext,
    ) -> Option<&'a Deployment> {
        if deployments.is_empty() {
            return None;
        }
        let mut rng = rand::rng();

        // Fast path: if no deployment carries a weight, rpm, or tpm, pick
        // uniformly. Avoids computing the weight total.
        if !deployments.iter().any(|d| {
            d.oxllm_params.weight.is_some()
                || d.oxllm_params.rpm.is_some()
                || d.oxllm_params.tpm.is_some()
        }) {
            return deployments.choose(&mut rng).copied();
        }

        // Weighted selection: compute total in a single pass (no Vec
        // allocation). Saturating add guards against a pathological config
        // of billions of weighted deployments: `sum` would panic on overflow
        // in debug and silently wrap in release (inverting the weighted pick).
        let total: u64 = deployments
            .iter()
            .map(|d| effective_weight(d))
            .fold(0u64, u64::saturating_add);

        if total == 0 {
            return deployments.choose(&mut rng).copied();
        }

        // Walk deployments without a Vec — iterate again, subtracting weights.
        let mut r: u64 = rng.random_range(0..total);
        for dep in deployments {
            let w = effective_weight(dep);
            if r < w {
                return Some(*dep);
            }
            r -= w;
        }
        // Defensive fallback for rounding drift; mathematically unreachable
        // when `total > 0`.
        deployments.choose(&mut rng).copied()
    }
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

    fn make_dep_with_weight(name: &str, weight: Option<u32>) -> Deployment {
        let mut d = make_dep(name);
        d.oxllm_params.weight = weight;
        d
    }

    fn make_dep_with_rpm(name: &str, rpm: Option<u32>) -> Deployment {
        let mut d = make_dep(name);
        d.oxllm_params.rpm = rpm;
        d
    }

    fn make_dep_with_tpm(name: &str, tpm: Option<u32>) -> Deployment {
        let mut d = make_dep(name);
        d.oxllm_params.tpm = tpm;
        d
    }

    #[test]
    fn empty_returns_none() {
        let strategy = SimpleShuffle;
        let ctx = RequestContext::new("gpt-4o");
        let result = strategy.pick(&[], &ctx);
        assert!(result.is_none());
    }

    #[test]
    fn single_deployment_always_picked() {
        let strategy = SimpleShuffle;
        let ctx = RequestContext::new("gpt-4o");
        let only = make_dep_with_weight("only", Some(7));
        let refs: Vec<&Deployment> = vec![&only];
        for _ in 0..50 {
            let picked = strategy.pick(&refs, &ctx).expect("should pick");
            assert_eq!(picked.model_name, "only");
        }
    }

    #[test]
    fn picks_from_available() {
        let strategy = SimpleShuffle;
        let ctx = RequestContext::new("gpt-4o");
        let deps = [make_dep("d1"), make_dep("d2"), make_dep("d3")];
        let refs: Vec<&Deployment> = deps.iter().collect();
        let picked = strategy.pick(&refs, &ctx);
        assert!(picked.is_some());
    }

    #[test]
    fn eventually_hits_different_deployments() {
        let strategy = SimpleShuffle;
        let ctx = RequestContext::new("gpt-4o");
        let deps = [
            make_dep("d1"),
            make_dep("d2"),
            make_dep("d3"),
            make_dep("d4"),
        ];
        let refs: Vec<&Deployment> = deps.iter().collect();

        let mut seen = std::collections::HashSet::new();
        for _ in 0..200 {
            if let Some(d) = strategy.pick(&refs, &ctx) {
                seen.insert(d.model_name.clone());
            }
        }
        assert!(
            seen.len() > 1,
            "shuffle should eventually pick different deployments, got: {seen:?}"
        );
    }

    #[test]
    fn weighted_always_picks_zero_weight_excluded_when_others_present() {
        let strategy = SimpleShuffle;
        let ctx = RequestContext::new("gpt-4o");
        let zero = make_dep_with_weight("zero", Some(0));
        let heavy = make_dep_with_weight("heavy", Some(10));
        let refs: Vec<&Deployment> = vec![&zero, &heavy];
        let mut saw_zero = false;
        for _ in 0..500 {
            let picked = strategy.pick(&refs, &ctx).expect("should pick");
            if picked.model_name == "zero" {
                saw_zero = true;
            }
        }
        assert!(
            !saw_zero,
            "weight=0 deployment should never be picked when total > 0"
        );
    }

    #[test]
    fn weighted_distributes_proportionally() {
        let strategy = SimpleShuffle;
        let ctx = RequestContext::new("gpt-4o");
        let big = make_dep_with_weight("big", Some(9));
        let small = make_dep_with_weight("small", Some(1));
        let refs: Vec<&Deployment> = vec![&big, &small];
        let mut big_hits = 0usize;
        let total = 10_000usize;
        for _ in 0..total {
            let picked = strategy.pick(&refs, &ctx).expect("should pick");
            if picked.model_name == "big" {
                big_hits += 1;
            }
        }
        // Expect ~9000/10000 = 90% big; allow ±5%.
        let ratio = big_hits as f64 / total as f64;
        assert!(
            (ratio - 0.9).abs() < 0.05,
            "expected ~0.9 big ratio, got {ratio:.3}"
        );
    }

    #[test]
    fn weighted_with_all_none_falls_back_to_uniform() {
        let strategy = SimpleShuffle;
        let ctx = RequestContext::new("gpt-4o");
        let a = make_dep("a");
        let b = make_dep("b");
        let refs: Vec<&Deployment> = vec![&a, &b];
        let mut a_hits = 0usize;
        let total = 2_000usize;
        for _ in 0..total {
            let picked = strategy.pick(&refs, &ctx).expect("should pick");
            if picked.model_name == "a" {
                a_hits += 1;
            }
        }
        let ratio = a_hits as f64 / total as f64;
        assert!(
            (ratio - 0.5).abs() < 0.1,
            "expected ~0.5 ratio for uniform, got {ratio:.3}"
        );
    }

    #[test]
    fn none_weight_counts_as_one_against_weighted_sibling() {
        // A deployment with weight=None (and rpm=None) counts as weight 1.
        let strategy = SimpleShuffle;
        let ctx = RequestContext::new("gpt-4o");
        let heavy = make_dep_with_weight("heavy", Some(3));
        let plain = make_dep("plain");
        let refs: Vec<&Deployment> = vec![&heavy, &plain];
        let mut plain_hits = 0usize;
        let total = 8_000usize;
        for _ in 0..total {
            let picked = strategy.pick(&refs, &ctx).expect("should pick");
            if picked.model_name == "plain" {
                plain_hits += 1;
            }
        }
        // Expected share for weight 1 out of total 4: 25%.
        let ratio = plain_hits as f64 / total as f64;
        assert!(
            (ratio - 0.25).abs() < 0.05,
            "None weight must count as 1 (~0.25 share), got {ratio:.3}"
        );
    }

    #[test]
    fn rpm_biases_selection_when_no_weight() {
        // Capacity-weight parity: with no explicit `weight`, `rpm` is the capacity
        // signal for the weighted pick.
        let strategy = SimpleShuffle;
        let ctx = RequestContext::new("gpt-4o");
        let big = make_dep_with_rpm("big", Some(900));
        let small = make_dep_with_rpm("small", Some(100));
        let refs: Vec<&Deployment> = vec![&big, &small];
        let mut big_hits = 0usize;
        let total = 10_000usize;
        for _ in 0..total {
            let picked = strategy.pick(&refs, &ctx).expect("should pick");
            if picked.model_name == "big" {
                big_hits += 1;
            }
        }
        let ratio = big_hits as f64 / total as f64;
        assert!(
            (ratio - 0.9).abs() < 0.05,
            "rpm must bias the pick (~0.9 big ratio), got {ratio:.3}"
        );
    }

    #[test]
    fn tpm_biases_selection_when_no_weight_or_rpm() {
        // Capacity-weight parity: with no explicit `weight` and no `rpm`, `tpm` is
        // the capacity signal for the weighted pick.
        let strategy = SimpleShuffle;
        let ctx = RequestContext::new("gpt-4o");
        let big = make_dep_with_tpm("big", Some(9_000));
        let small = make_dep_with_tpm("small", Some(1_000));
        let refs: Vec<&Deployment> = vec![&big, &small];
        let mut big_hits = 0usize;
        let total = 10_000usize;
        for _ in 0..total {
            let picked = strategy.pick(&refs, &ctx).expect("should pick");
            if picked.model_name == "big" {
                big_hits += 1;
            }
        }
        let ratio = big_hits as f64 / total as f64;
        assert!(
            (ratio - 0.9).abs() < 0.05,
            "tpm must bias the pick (~0.9 big ratio), got {ratio:.3}"
        );
    }

    #[test]
    fn rpm_wins_over_tpm() {
        // `rpm` takes precedence over `tpm`: rpm=100 with tpm=9000 must
        // behave as weight 100, not 9000.
        let strategy = SimpleShuffle;
        let ctx = RequestContext::new("gpt-4o");
        let mut capped = make_dep_with_tpm("capped", Some(9_000));
        capped.oxllm_params.rpm = Some(100);
        let heavy = make_dep_with_rpm("heavy", Some(900));
        let refs: Vec<&Deployment> = vec![&capped, &heavy];
        let mut capped_hits = 0usize;
        let total = 10_000usize;
        for _ in 0..total {
            let picked = strategy.pick(&refs, &ctx).expect("should pick");
            if picked.model_name == "capped" {
                capped_hits += 1;
            }
        }
        let ratio = capped_hits as f64 / total as f64;
        assert!(
            (ratio - 0.1).abs() < 0.05,
            "rpm must win over tpm (~0.1 capped share), got {ratio:.3}"
        );
    }

    #[test]
    fn explicit_weight_wins_over_rpm() {
        // `weight` takes precedence over `rpm`: weight=1 with rpm=1000 must
        // behave as weight 1, not 1000.
        let strategy = SimpleShuffle;
        let ctx = RequestContext::new("gpt-4o");
        let mut capped = make_dep_with_rpm("capped", Some(1000));
        capped.oxllm_params.weight = Some(1);
        let heavy = make_dep_with_weight("heavy", Some(3));
        let refs: Vec<&Deployment> = vec![&capped, &heavy];
        let mut capped_hits = 0usize;
        let total = 8_000usize;
        for _ in 0..total {
            let picked = strategy.pick(&refs, &ctx).expect("should pick");
            if picked.model_name == "capped" {
                capped_hits += 1;
            }
        }
        // If rpm leaked through, capped would win ~1000/1003 ≈ 99.7%;
        // with weight winning, its share is 1/4 = 25%.
        let ratio = capped_hits as f64 / total as f64;
        assert!(
            (ratio - 0.25).abs() < 0.05,
            "explicit weight must override rpm (~0.25 share), got {ratio:.3}"
        );
    }

    #[test]
    fn mixed_weight_and_rpm_does_not_starve_rpm_only_deployment() {
        // The reference selects one metric group-wide from the first candidate and
        // starves siblings missing it; oxllm resolves the metric per
        // deployment via `effective_weight` (weight → rpm → tpm → 1), so a
        // group with `dep-A: weight=10` and `dep-B: rpm=100` (no weight)
        // gives `dep-B` effective weight 100, not 1 and not 0. Pin this so
        // a "fix" that collapses to the reference group-wide metric does not
        // silently starve the rpm-only deployment in mixed configs.
        let strategy = SimpleShuffle;
        let ctx = RequestContext::new("gpt-4o");
        let weighted = make_dep_with_weight("weighted", Some(10));
        let rpm_only = make_dep_with_rpm("rpm-only", Some(100));
        let refs: Vec<&Deployment> = vec![&weighted, &rpm_only];
        let mut rpm_hits = 0usize;
        let total = 10_000usize;
        for _ in 0..total {
            let picked = strategy.pick(&refs, &ctx).expect("should pick");
            if picked.model_name == "rpm-only" {
                rpm_hits += 1;
            }
        }
        // Effective weights are weighted=10 and rpm-only=100, so rpm-only's
        // share is 100/110 ≈ 0.909. If the rpm-only deployment were starved
        // (the reference behavior for a missing group-wide metric) the hit
        // count would be 0; if its rpm were ignored in favour of weight=1,
        // the share would collapse to 1/11 ≈ 0.091.
        let ratio = rpm_hits as f64 / total as f64;
        assert!(
            (ratio - 0.909).abs() < 0.05,
            "rpm-only deployment must keep its rpm weight (~0.909 share), got {ratio:.3}"
        );
    }

    #[test]
    fn rpm_zero_starves_deployment_under_simple_shuffle() {
        // `rpm: Some(0)` is NOT treated as "unset" — `.or()` returns the
        // `Some(0)` value, and the `.unwrap_or(1)` fallback never fires, so
        // the deployment gets effective weight 0 and receives no traffic as
        // long as any sibling has positive weight. This is the documented
        // footgun (`weight: 0` / `rpm: 0` / `tpm: 0` all mean "starve").
        // An operator who sets `rpm: 0` meaning "no rate limit" or as a
        // typo gets a starved deployment. Pin this so a future change to
        // treat `Some(0)` as "unset" (matching `usage_based`'s `if rpm > 0`
        // guard) is caught.
        let strategy = SimpleShuffle;
        let ctx = RequestContext::new("gpt-4o");
        let mut starved = make_dep_with_rpm("starved", Some(0));
        starved.oxllm_params.weight = None;
        starved.oxllm_params.tpm = None;
        let heavy = make_dep_with_weight("heavy", Some(10));
        let refs: Vec<&Deployment> = vec![&starved, &heavy];
        let mut starved_hits = 0usize;
        for _ in 0..5_000 {
            let picked = strategy.pick(&refs, &ctx).expect("should pick");
            if picked.model_name == "starved" {
                starved_hits += 1;
            }
        }
        assert_eq!(
            starved_hits, 0,
            "rpm: Some(0) must starve the deployment (effective weight 0); \
             got {starved_hits}/5000 picks — pin the documented footgun"
        );
    }
}
