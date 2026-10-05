//! Round-robin routing strategy: strict rotation through the candidate list.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::deployment::Deployment;
use crate::router::strategy::{RequestContext, RoutingStrategy};

/// Round-robin routing: cycles through the candidate list with a monotonically
/// incrementing counter, so consecutive requests hit different deployments.
///
/// The counter is held **per model group** in a shared
/// `Arc<Mutex<HashMap<String, Arc<AtomicU64>>>>` constructed once per `Router`
/// (see [`Router::round_robin_counters_arc`]) and threaded through the
/// strategy via `StrategyDeps`. It survives
/// config hot-reloads (`Router::rebuild_from_stores`) exactly like the
/// cooldown/latency/usage/health/bandit stores — every other stateful
/// strategy already keeps its state in a shared `Arc`, and round-robin's
/// strict alternation is just as visible to clients, so a config reload
/// (canary add, credentials rotation) must NOT silently reset the rotation
/// position back to `deployments[0]`.
///
/// Each model group gets its **own** counter (lazily populated on the first
/// pick for that group), so a global `round_robin` config with multiple
/// groups strictly alternates within each group regardless of cross-group
/// interleaving. Sharing a single counter across all groups would let an
/// inter-group pick advance another group's index in lockstep: under a
/// deterministic A,B,A,B,… interleave over groups A=`[a1,a2]` and
/// B=`[b1,b2,b3]`, group A would advance its counter by 2 between its own
/// picks and pin to `a1` forever. The per-group counter keeps the
/// documented contract ("strict alternation", "load testing", "deterministic
/// debug traces") intact under the very workloads that contract calls out.
///
/// The group key is
/// [`PreCallData::model_group_key`](crate::router::strategy::PreCallData::model_group_key)
/// when the router set it (falling back to `RequestContext::model` for
/// standalone use): a wildcard deployment (`model_name: "*"`) serves every
/// distinct client-requested model string, so wildcard-served requests
/// aggregate under the single `"*"` counter instead of allocating one
/// counter per requested name — which would grow the map without bound off
/// client-controlled input.
///
/// When multiple oxllm instances sit behind a load balancer, each instance
/// keeps its own per-group counters (no cross-instance coordination), so
/// strict alternation holds within an instance and the long-run distribution
/// across instances averages out. Weighted deployments are supported via
/// weighted round-robin: each deployment occupies `weight` slots in the
/// rotation (default weight 1).
///
/// This is a true rotating round-robin, distinct from `SimpleShuffle` which
/// picks randomly. Use it when you need strict alternation (e.g. even
/// distribution for load testing, deterministic debug traces, or fair
/// request splitting across equally-priced deployments).
///
/// [`Router::rebuild_from_stores`]: crate::router::Router::rebuild_from_stores
/// [`Router::round_robin_counters_arc`]: crate::router::Router::round_robin_counters_arc
pub struct RoundRobin {
    counters: Arc<Mutex<HashMap<String, Arc<AtomicU64>>>>,
}

impl RoundRobin {
    /// Create a new round-robin strategy with a fresh per-group counter map.
    /// Construction from a shared map (the path `Router` uses so the
    /// counters survive hot-reloads) goes through [`Self::with_counters`].
    pub fn new() -> Self {
        Self {
            counters: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Create a round-robin strategy sharing an existing per-group counter
    /// map. Used by the strategy factory so the counters are preserved
    /// across `Router::rebuild_from_stores`.
    pub fn with_counters(counters: Arc<Mutex<HashMap<String, Arc<AtomicU64>>>>) -> Self {
        Self { counters }
    }

    /// Resolve (lazily creating) the counter for `model`. Per-group counters
    /// keep strict alternation within each group regardless of cross-group
    /// interleaving — see the type-level doc.
    fn counter_for(&self, model: &str) -> Arc<AtomicU64> {
        let mut guard = self.counters.lock().unwrap_or_else(|e| e.into_inner());
        guard
            .entry(model.to_string())
            .or_insert_with(|| Arc::new(AtomicU64::new(0)))
            .clone()
    }
}

impl Default for RoundRobin {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for RoundRobin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let guard = self.counters.lock().unwrap_or_else(|e| e.into_inner());
        let snapshot: Vec<(String, u64)> = guard
            .iter()
            .map(|(k, v)| (k.clone(), v.load(Ordering::Relaxed)))
            .collect();
        f.debug_struct("RoundRobin")
            .field("counters", &snapshot)
            .finish()
    }
}

/// The effective selection weight of a deployment: the explicit `weight`
/// first, then the `rpm` capacity hint, then the `tpm` capacity hint, then
/// `1`. See [`crate::router::simple_shuffle::effective_weight`] for the
/// shared footgun doc (`weight: 0` / `rpm: 0` / `tpm: 0` all mean
/// "starve", pinned by `rpm_zero_starves_deployment_under_round_robin`).
fn effective_weight(dep: &Deployment) -> u64 {
    dep.oxllm_params
        .weight
        .or(dep.oxllm_params.rpm)
        .or(dep.oxllm_params.tpm)
        .unwrap_or(1) as u64
}

impl RoutingStrategy for RoundRobin {
    fn pick<'a>(
        &self,
        deployments: &[&'a Deployment],
        ctx: &RequestContext,
    ) -> Option<&'a Deployment> {
        if deployments.is_empty() {
            return None;
        }
        if deployments.len() == 1 {
            return Some(deployments[0]);
        }

        let any_weighted = deployments.iter().any(|d| {
            d.oxllm_params.weight.is_some()
                || d.oxllm_params.rpm.is_some()
                || d.oxllm_params.tpm.is_some()
        });

        let counter = self.counter_for(
            ctx.pre_call
                .model_group_key
                .as_deref()
                .unwrap_or(&ctx.model),
        );
        if !any_weighted {
            // Reduce the modulo in `u64` before the `as usize` cast. On
            // 64-bit targets `usize == u64` so this is a no-op; on 32-bit
            // targets `usize == u32`, and `(fetch_add as usize) % len`
            // would silently drop the high 32 bits of the counter first,
            // producing a different rotation sequence than a 64-bit replica
            // after ~4.29 × 10⁹ requests. Mirrors the `session_affinity`
            // fix (and is regression-tested the same way). See the analogous
            // `session_affinity::pick_index_uses_u64_modulo_not_truncated`.
            let len = deployments.len() as u64;
            let idx = (counter.fetch_add(1, Ordering::Relaxed) % len) as usize;
            return Some(deployments[idx]);
        }

        let total: u64 = deployments
            .iter()
            .map(|d| effective_weight(d))
            .fold(0u64, u64::saturating_add);
        if total == 0 {
            // All-zero weights: fall back to uniform RR. Same u64-modulo
            // reduction as the non-weighted fast path (see above).
            let len = deployments.len() as u64;
            let idx = (counter.fetch_add(1, Ordering::Relaxed) % len) as usize;
            return Some(deployments[idx]);
        }

        let slot = counter.fetch_add(1, Ordering::Relaxed) % total;
        let mut acc: u64 = 0;
        for dep in deployments {
            acc = acc.saturating_add(effective_weight(dep));
            if slot < acc {
                return Some(*dep);
            }
        }
        // Defensive fallback.
        Some(deployments[0])
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

    fn make_dep_with_model(name: &str, model: &str) -> Deployment {
        let mut d = make_dep(name);
        d.oxllm_params.model = model.to_string();
        d
    }

    #[test]
    fn empty_returns_none() {
        let strategy = RoundRobin::new();
        let ctx = RequestContext::new("gpt-4o");
        assert!(strategy.pick(&[], &ctx).is_none());
    }

    #[test]
    fn single_deployment_always_picked() {
        let strategy = RoundRobin::new();
        let ctx = RequestContext::new("gpt-4o");
        let only = make_dep("only");
        let refs: Vec<&Deployment> = vec![&only];
        for _ in 0..10 {
            assert_eq!(strategy.pick(&refs, &ctx).unwrap().model_name, "only");
        }
    }

    #[test]
    fn alternates_strictly() {
        let strategy = RoundRobin::new();
        let ctx = RequestContext::new("gpt-4o");
        let deps = [make_dep("a"), make_dep("b"), make_dep("c")];
        let refs: Vec<&Deployment> = deps.iter().collect();
        let picks: Vec<&str> = (0..6)
            .map(|_| strategy.pick(&refs, &ctx).unwrap().model_name.as_str())
            .collect();
        assert_eq!(picks, vec!["a", "b", "c", "a", "b", "c"]);
    }

    #[test]
    fn weighted_round_robin_2_to_1() {
        let strategy = RoundRobin::new();
        let ctx = RequestContext::new("gpt-4o");
        let heavy = make_dep_with_weight("heavy", Some(2));
        let light = make_dep_with_weight("light", Some(1));
        let refs: Vec<&Deployment> = vec![&heavy, &light];
        let picks: Vec<&str> = (0..6)
            .map(|_| strategy.pick(&refs, &ctx).unwrap().model_name.as_str())
            .collect();
        // total=3: heavy has slots 0,1; light has slot 2.
        assert_eq!(
            picks,
            vec!["heavy", "heavy", "light", "heavy", "heavy", "light"]
        );
    }

    #[test]
    fn weighted_distribution_over_many_picks() {
        let strategy = RoundRobin::new();
        let ctx = RequestContext::new("gpt-4o");
        let big = make_dep_with_weight("big", Some(9));
        let small = make_dep_with_weight("small", Some(1));
        let refs: Vec<&Deployment> = vec![&big, &small];
        let mut big_count = 0;
        let total = 10_000;
        for _ in 0..total {
            if strategy.pick(&refs, &ctx).unwrap().model_name == "big" {
                big_count += 1;
            }
        }
        let ratio = big_count as f64 / total as f64;
        assert!(
            (ratio - 0.9).abs() < 0.01,
            "expected ~0.9 big ratio, got {ratio:.3}"
        );
    }

    #[test]
    fn wraps_around_correctly() {
        let strategy = RoundRobin::new();
        let ctx = RequestContext::new("gpt-4o");
        let deps: Vec<_> = (0..5).map(|i| make_dep(&format!("d{i}"))).collect();
        let refs: Vec<&Deployment> = deps.iter().collect();
        for round in 0..3 {
            for i in 0..5 {
                let expected = format!("d{i}");
                let got = strategy.pick(&refs, &ctx).unwrap().model_name.clone();
                assert_eq!(got, expected, "round {round} slot {i}");
            }
        }
    }

    #[test]
    fn pick_index_uses_u64_modulo_not_truncated() {
        // The non-weighted fast path must reduce the modulo in `u64` before
        // the `as usize` cast, so a 32-bit replica rotates to the same
        // deployment as a 64-bit one for a given counter value. Find a
        // discriminator counter value where `(value as u32 as usize) % len
        // != (value % len as u64) as usize`, pre-seed the per-group counter,
        // and assert the strategy picks the u64-modulo index.
        //
        // `0x1_0000_0003` over a 5-deployment group:
        // - u64 path: `0x1_0000_0003 == 4294967299`, `% 5 == 4`
        // - truncated u32 path: `0x0000_0003 == 3`, `% 5 == 3`
        let value: u64 = 0x1_0000_0003;
        let len: u64 = 5;
        let u64_idx = (value % len) as usize;
        let trunc_idx = (value as u32 as usize) % len as usize;
        assert_ne!(
            u64_idx, trunc_idx,
            "this test only exercises the bug if the two formulas disagree"
        );
        assert_eq!(u64_idx, 4);
        assert_eq!(trunc_idx, 3);

        let counters = Arc::new(Mutex::new(HashMap::from([(
            "gpt-4o".to_string(),
            Arc::new(AtomicU64::new(value)),
        )])));
        let strategy = RoundRobin::with_counters(counters);
        let ctx = RequestContext::new("gpt-4o");
        let deps: Vec<_> = (0..len).map(|i| make_dep(&format!("d{i}"))).collect();
        let refs: Vec<&Deployment> = deps.iter().collect();
        let picked = strategy.pick(&refs, &ctx).expect("pick");
        assert_eq!(
            picked.model_name,
            format!("d{u64_idx}"),
            "must pick the u64-modulo index even on 32-bit targets"
        );
    }

    #[test]
    fn counter_survives_rebuild() {
        // The rotation position must survive a config hot-reload rebuild.
        // Pre-seed a shared counter map, pick once, build a new strategy from
        // the SAME map (as the Router factory does on rebuild), and assert
        // the next pick continues the rotation rather than restarting from
        // index 0.
        use crate::router::Router;
        use crate::router::types::{RouterConfig, RoutingStrategyKind};

        let config = RouterConfig {
            routing_strategy: RoutingStrategyKind::RoundRobin,
            ..RouterConfig::default()
        };
        let router = Router::new(config.clone());
        // Deployments share model_name "gpt-4o" so `pick_deployment` can
        // resolve them; their distinct provider models keep their stable
        // keys distinct.
        let deps: Vec<_> = (0..3)
            .map(|i| make_dep_with_model("gpt-4o", &format!("openai/gpt-4o-{i}")))
            .collect();
        let ctx = RequestContext::new("gpt-4o");
        let first = router
            .pick_deployment(&deps, &ctx)
            .expect("pick")
            .oxllm_params
            .model
            .clone();
        assert_eq!(first, "openai/gpt-4o-0", "first pick must be index 0");

        let rebuilt = Router::rebuild_from_stores(
            config,
            router.cooldown_arc(),
            router.latency_store_arc(),
            router.usage_store_arc(),
            router.health_store_arc(),
            router.bandit_store_arc(),
            router.round_robin_counters_arc(),
        );
        let second = rebuilt
            .pick_deployment(&deps, &ctx)
            .expect("pick")
            .oxllm_params
            .model
            .clone();
        assert_eq!(
            second, "openai/gpt-4o-1",
            "rotation must continue (index 1) after the rebuild, not reset to index 0"
        );
    }

    #[test]
    fn global_round_robin_strictly_alternates_within_each_group_under_interleaved_traffic() {
        // C65 HIGH regression: a global `round_robin` config must give each
        // model group its own counter. Under a deterministic A,B,A,B,…
        // interleave over two 2-deployment groups (A=`[a1,a2]`,
        // B=`[b1,b2]`), the shared-single-counter bug advanced A's
        // effective index by 2 between its own picks (B's picks incremented
        // the same counter), pinning A to `a1` forever (even-counter % 2).
        // Per-group counters keep each group on its own strict rotation
        // regardless of cross-group interleaving.
        use crate::router::Router;
        use crate::router::types::{RouterConfig, RoutingStrategyKind};

        let config = RouterConfig {
            routing_strategy: RoutingStrategyKind::RoundRobin,
            ..RouterConfig::default()
        };
        let router = Router::new(config);

        // Group A — two deployments sharing model_name "gpt-4o".
        let group_a: Vec<_> = (0..2)
            .map(|i| make_dep_with_model("gpt-4o", &format!("openai/gpt-4o-{i}")))
            .collect();
        // Group B — two deployments sharing model_name "claude".
        let group_b: Vec<_> = (0..2)
            .map(|i| make_dep_with_model("claude", &format!("anthropic/claude-{i}")))
            .collect();
        // Combined deployment slice as a single Router would see them.
        let mut combined: Vec<Deployment> = Vec::new();
        combined.extend(group_a.iter().cloned());
        combined.extend(group_b.iter().cloned());

        let ctx_a = RequestContext::new("gpt-4o");
        let ctx_b = RequestContext::new("claude");

        // Drive an explicitly interleaved A,B,A,B,… sequence and assert each
        // group strictly alternates. With the single-shared-counter bug,
        // group A would advance by 2 between its picks and stick on `a1`.
        for round in 0..20 {
            let pick_a = router
                .pick_deployment(&combined, &ctx_a)
                .expect("pick A")
                .oxllm_params
                .model
                .clone();
            assert_eq!(
                pick_a,
                format!("openai/gpt-4o-{}", round % 2),
                "group A must strictly alternate (round {round}); \
                 the single-shared-counter bug would pin it to gpt-4o-0"
            );
            let pick_b = router
                .pick_deployment(&combined, &ctx_b)
                .expect("pick B")
                .oxllm_params
                .model
                .clone();
            assert_eq!(
                pick_b,
                format!("anthropic/claude-{}", round % 2),
                "group B must strictly alternate (round {round})"
            );
        }
    }

    #[test]
    fn global_round_robin_per_group_counters_independent_under_steady_concurrent_traffic() {
        // C65 HIGH regression, steady-concurrent variant: drive each group
        // in a tight burst (not strictly interleaved) and assert the within-
        // group rotation still steps 1:1 with that group's picks. With the
        // single-shared-counter bug, picks for group A followed by a burst
        // of picks for group B would leave A's next pick at an offset
        // influenced by B's traffic.
        use crate::router::Router;
        use crate::router::types::{RouterConfig, RoutingStrategyKind};

        let router = Router::new(RouterConfig {
            routing_strategy: RoutingStrategyKind::RoundRobin,
            ..RouterConfig::default()
        });

        let group_a: Vec<_> = (0..3)
            .map(|i| make_dep_with_model("gpt-4o", &format!("openai/gpt-4o-{i}")))
            .collect();
        let group_b: Vec<_> = (0..2)
            .map(|i| make_dep_with_model("claude", &format!("anthropic/claude-{i}")))
            .collect();
        let mut combined: Vec<Deployment> = Vec::new();
        combined.extend(group_a.iter().cloned());
        combined.extend(group_b.iter().cloned());

        let ctx_a = RequestContext::new("gpt-4o");
        let ctx_b = RequestContext::new("claude");

        // Burst of 5 picks for B (advances the shared counter by 5 under the
        // bug). Then 6 picks for A: with per-group counters A must rotate
        // 0,1,2,0,1,2 exactly.
        for _ in 0..5 {
            let _ = router.pick_deployment(&combined, &ctx_b).expect("pick B");
        }
        let a_picks: Vec<String> = (0..6)
            .map(|_| {
                router
                    .pick_deployment(&combined, &ctx_a)
                    .expect("pick A")
                    .oxllm_params
                    .model
                    .clone()
            })
            .collect();
        assert_eq!(
            a_picks,
            vec![
                "openai/gpt-4o-0",
                "openai/gpt-4o-1",
                "openai/gpt-4o-2",
                "openai/gpt-4o-0",
                "openai/gpt-4o-1",
                "openai/gpt-4o-2",
            ],
            "group A's rotation must be independent of group B's traffic"
        );
    }

    #[test]
    fn rpm_zero_starves_deployment_under_round_robin() {
        // `rpm: Some(0)` is NOT treated as "unset" — the deployment gets
        // zero slots in the weighted rotation and is starved as long as any
        // sibling has positive weight. Mirrors the `simple_shuffle` pin;
        // both share the same `effective_weight` footgun. An operator who
        // sets `rpm: 0` meaning "no rate limit" or as a typo gets a
        // starved deployment.
        let strategy = RoundRobin::new();
        let ctx = RequestContext::new("gpt-4o");
        let mut starved = make_dep_with_weight("starved", None);
        starved.oxllm_params.rpm = Some(0);
        starved.oxllm_params.tpm = None;
        let heavy = make_dep_with_weight("heavy", Some(2));
        let refs: Vec<&Deployment> = vec![&starved, &heavy];
        let mut starved_hits = 0usize;
        for _ in 0..100 {
            let picked = strategy.pick(&refs, &ctx).expect("should pick");
            if picked.model_name == "starved" {
                starved_hits += 1;
            }
        }
        assert_eq!(
            starved_hits, 0,
            "rpm: Some(0) must starve the deployment under RoundRobin too; \
             got {starved_hits}/100 picks — pin the documented footgun"
        );
    }
}
