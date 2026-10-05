use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use oxllm_core::deployment::{Deployment, ModelInfo, OxllmParams};
use oxllm_core::router::complexity::{Tier, classify};
use oxllm_core::router::stores::{LatencyStore, stable_key};
use oxllm_core::router::strategy::RequestContext;
use oxllm_core::router::{ComplexityRouterConfig, Router, RouterConfig, RoutingStrategyKind};

fn make_deployments(n: usize, model_name: &str) -> Vec<Deployment> {
    (0..n)
        .map(|i| Deployment {
            model_name: model_name.to_string(),
            oxllm_params: OxllmParams {
                model: format!("openai/gpt-4o-{i}"),
                api_key: Some(format!("key-{i}")),
                rpm: Some(480),
                tpm: Some(1_000_000),
                extra: Default::default(),
                ..Default::default()
            },
            ..Default::default()
        })
        .collect()
}

fn make_tagged_deployments(n: usize, model_name: &str, tag: &str) -> Vec<Deployment> {
    (0..n)
        .map(|i| Deployment {
            model_name: model_name.to_string(),
            oxllm_params: OxllmParams {
                model: format!("openai/gpt-4o-{i}"),
                api_key: Some(format!("key-{i}")),
                rpm: Some(480),
                tpm: Some(1_000_000),
                extra: Default::default(),
                ..Default::default()
            },
            model_info: Some(ModelInfo {
                // Leave half untagged so the benchmark actually narrows the pool.
                tags: if i % 2 == 0 {
                    vec![tag.to_string()]
                } else {
                    Vec::new()
                },
                ..Default::default()
            }),
        })
        .collect()
}

fn make_ctx(model: &str) -> RequestContext {
    RequestContext {
        api_key_hash: "test".to_string(),
        request_id: "bench".to_string(),
        ..RequestContext::new(model)
    }
}

fn deployments_for_model(c: &mut Criterion) {
    let router = Router::new(RouterConfig::default());
    let ctx = make_ctx("gpt-4o");
    let mut group = c.benchmark_group("deployments_for_model");
    for n in [1, 10, 100] {
        let deps = make_deployments(n, "gpt-4o");
        group.bench_with_input(BenchmarkId::from_parameter(n), &deps, |b, deps| {
            b.iter(|| router.deployments_for_model(deps, &ctx.model));
        });
    }
    group.finish();
}

fn deployments_for_model_wildcard(c: &mut Criterion) {
    let router = Router::new(RouterConfig::default());
    let ctx = make_ctx("gpt-4o");
    let mut deps = make_deployments(99, "other-model");
    deps.push(Deployment {
        model_name: "*".to_string(),
        oxllm_params: OxllmParams {
            model: "openai/*".to_string(),
            api_key: Some("wildcard-key".to_string()),
            ..Default::default()
        },
        ..Default::default()
    });
    c.bench_function("deployments_for_model_wildcard_100", |b| {
        b.iter(|| router.deployments_for_model(&deps, &ctx.model));
    });
}

fn pick_deployment(c: &mut Criterion) {
    let ctx = make_ctx("gpt-4o");

    let mut group = c.benchmark_group("pick_deployment");
    group.bench_with_input(
        BenchmarkId::new("simple_shuffle", 10),
        &make_deployments(10, "gpt-4o"),
        |b, deps| {
            let router = Router::new(RouterConfig {
                routing_strategy: RoutingStrategyKind::SimpleShuffle,
                ..Default::default()
            });
            b.iter(|| router.pick_deployment(deps, &ctx));
        },
    );
    group.bench_with_input(
        BenchmarkId::new("simple_shuffle", 100),
        &make_deployments(100, "gpt-4o"),
        |b, deps| {
            let router = Router::new(RouterConfig {
                routing_strategy: RoutingStrategyKind::SimpleShuffle,
                ..Default::default()
            });
            b.iter(|| router.pick_deployment(deps, &ctx));
        },
    );
    group.bench_with_input(
        BenchmarkId::new("lowest_cost", 10),
        &make_deployments(10, "gpt-4o"),
        |b, deps| {
            let router = Router::new(RouterConfig {
                routing_strategy: RoutingStrategyKind::LowestCost,
                ..Default::default()
            });
            b.iter(|| router.pick_deployment(deps, &ctx));
        },
    );
    group.bench_with_input(
        BenchmarkId::new("lowest_latency", 10),
        &make_deployments(10, "gpt-4o"),
        |b, deps| {
            let router = Router::new(RouterConfig {
                routing_strategy: RoutingStrategyKind::LowestLatency,
                ..Default::default()
            });
            for (i, dep) in deps.iter().enumerate() {
                let key = stable_key(dep);
                router.record_latency(&key, (i as u64) * 10 + 50);
            }
            let picked = router.pick_deployment(deps, &ctx).expect("should pick");
            assert_eq!(
                picked.oxllm_params.model, "openai/gpt-4o-0",
                "deployment 0 has the lowest latency (50 ms) and must win"
            );
            b.iter(|| router.pick_deployment(deps, &ctx));
        },
    );

    // Half-cooled: 100 deployments, 50 in cooldown.
    let deps_100 = make_deployments(100, "gpt-4o");
    group.bench_with_input(
        BenchmarkId::new("half_cooled", 100),
        &deps_100,
        |b, deps| {
            let router = Router::new(RouterConfig {
                routing_strategy: RoutingStrategyKind::SimpleShuffle,
                ..Default::default()
            });
            let cooled: std::collections::HashSet<String> =
                deps.iter().take(50).map(stable_key).collect();
            for dep in deps.iter().take(50) {
                let key = stable_key(dep);
                router.mark_cooldown(&key);
            }
            let picked = router.pick_deployment(deps, &ctx).expect("should pick");
            assert!(
                !cooled.contains(&stable_key(picked)),
                "must not pick a cooled deployment"
            );
            b.iter(|| router.pick_deployment(deps, &ctx));
        },
    );
    group.finish();
}

fn pick_deployment_more_strategies(c: &mut Criterion) {
    let ctx = make_ctx("gpt-4o");
    let deps_10 = make_deployments(10, "gpt-4o");
    let mut group = c.benchmark_group("pick_deployment");

    group.bench_with_input(BenchmarkId::new("usage_based", 10), &deps_10, |b, deps| {
        let router = Router::new(RouterConfig {
            routing_strategy: RoutingStrategyKind::UsageBased,
            ..Default::default()
        });
        for (i, dep) in deps.iter().enumerate() {
            router.record_usage(&stable_key(dep), (i as f64) * 0.1);
        }
        b.iter(|| router.pick_deployment(deps, &ctx));
    });

    group.bench_with_input(
        BenchmarkId::new("latency_based", 10),
        &deps_10,
        |b, deps| {
            let router = Router::new(RouterConfig {
                routing_strategy: RoutingStrategyKind::LatencyBased,
                ..Default::default()
            });
            for (i, dep) in deps.iter().enumerate() {
                router.record_latency(&stable_key(dep), (i as u64) * 10 + 50);
            }
            b.iter(|| router.pick_deployment(deps, &ctx));
        },
    );

    group.bench_with_input(BenchmarkId::new("least_busy", 10), &deps_10, |b, deps| {
        let router = Router::new(RouterConfig {
            routing_strategy: RoutingStrategyKind::LeastBusy,
            ..Default::default()
        });
        for (i, dep) in deps.iter().enumerate() {
            router.record_in_flight(&stable_key(dep), i as i64);
        }
        b.iter(|| router.pick_deployment(deps, &ctx));
    });

    group.bench_with_input(
        BenchmarkId::new("quality_router", 10),
        &deps_10,
        |b, deps| {
            let mut quality_scores = std::collections::HashMap::new();
            for (i, dep) in deps.iter().enumerate() {
                quality_scores.insert(dep.oxllm_params.model.clone(), 1.0 - (i as f64) * 0.1);
            }
            let router = Router::new(RouterConfig {
                routing_strategy: RoutingStrategyKind::QualityRouter,
                quality_scores,
                ..Default::default()
            });
            b.iter(|| router.pick_deployment(deps, &ctx));
        },
    );

    let tagged_deps = make_tagged_deployments(10, "gpt-4o", "canary");
    group.bench_with_input(
        BenchmarkId::new("tag_based", 10),
        &tagged_deps,
        |b, deps| {
            let router = Router::new(RouterConfig {
                routing_strategy: RoutingStrategyKind::TagBased,
                ..Default::default()
            });
            let ctx = RequestContext {
                tag: Some("canary".to_string()),
                tags: vec!["canary".to_string()],
                ..ctx.clone()
            };
            let picked = router
                .pick_deployment(deps, &ctx)
                .expect("tagged deployment");
            assert_eq!(
                picked.model_info.as_ref().expect("tag metadata").tags,
                ["canary"]
            );
            b.iter(|| router.pick_deployment(deps, &ctx));
        },
    );

    group.bench_with_input(
        BenchmarkId::new("session_affinity", 10),
        &deps_10,
        |b, deps| {
            let router = Router::new(RouterConfig {
                routing_strategy: RoutingStrategyKind::SessionAffinity,
                ..Default::default()
            });
            let ctx = RequestContext {
                session_id: Some("bench-session".to_string()),
                ..ctx.clone()
            };
            b.iter(|| router.pick_deployment(deps, &ctx));
        },
    );

    // Ten tier candidates plus a separate requested group: a missing config
    // must fail the preflight assertion instead of silently measuring fallback.
    let mut complexity_deps = make_deployments(10, "reasoning-pool");
    complexity_deps.extend(make_deployments(1, "gpt-4o"));
    group.bench_with_input(
        BenchmarkId::new("complexity_router", 10),
        &complexity_deps,
        |b, deps| {
            let complexity_config = ComplexityRouterConfig {
                tiers: std::collections::HashMap::from([(
                    "reasoning".to_string(),
                    vec!["reasoning-pool".to_string()],
                )]),
                ..Default::default()
            };
            let ctx = RequestContext {
                request_text: Some("Prove the theorem by induction, step by step.".to_string()),
                ..ctx.clone()
            };
            assert_eq!(classify(&ctx, &complexity_config).0, Tier::Reasoning);
            let router = Router::new(RouterConfig {
                routing_strategy: RoutingStrategyKind::ComplexityRouter,
                complexity_router_config: Some(complexity_config),
                ..Default::default()
            });
            let picked = router
                .pick_deployment(deps, &ctx)
                .expect("configured tier pool");
            assert_eq!(
                picked.model_name, "reasoning-pool",
                "must not use requested-group fallback"
            );
            b.iter(|| router.pick_deployment(deps, &ctx));
        },
    );

    group.bench_with_input(
        BenchmarkId::new("adaptive_router", 10),
        &deps_10,
        |b, deps| {
            let router = Router::new(RouterConfig {
                routing_strategy: RoutingStrategyKind::AdaptiveRouter,
                ..Default::default()
            });
            let ctx = RequestContext {
                request_text: Some(
                    "Explain the pros and cons of microservices versus a monolithic architecture."
                        .to_string(),
                ),
                ..ctx.clone()
            };
            b.iter(|| router.pick_deployment(deps, &ctx));
        },
    );

    group.bench_with_input(BenchmarkId::new("round_robin", 10), &deps_10, |b, deps| {
        let router = Router::new(RouterConfig {
            routing_strategy: RoutingStrategyKind::RoundRobin,
            ..Default::default()
        });
        b.iter(|| router.pick_deployment(deps, &ctx));
    });

    group.finish();
}

fn fallbacks_for_model(c: &mut Criterion) {
    let router = Router::new(RouterConfig {
        fallbacks: vec![oxllm_core::router::FallbackRule {
            primary: "gpt-4o".to_string(),
            fallbacks: vec!["claude-haiku".to_string(), "gemini-flash".to_string()],
        }],
        ..Default::default()
    });
    c.bench_function("fallbacks_for_model", |b| {
        b.iter(|| router.fallbacks_for("gpt-4o"));
    });
}

fn router_new(c: &mut Criterion) {
    c.bench_function("router_new", |b| {
        b.iter(|| Router::new(RouterConfig::default()));
    });
    c.bench_function("router_new_lowest_latency", |b| {
        b.iter(|| {
            Router::new(RouterConfig {
                routing_strategy: RoutingStrategyKind::LowestLatency,
                ..Default::default()
            })
        });
    });
}

fn latency_store_p50(c: &mut Criterion) {
    let store = LatencyStore::default();
    for i in 0..100u64 {
        store.record("k", (i + 1) * 7);
    }
    c.bench_function("latency_store_p50_full_window", |b| {
        b.iter(|| store.p50("k"));
    });
}

criterion_group!(
    benches,
    deployments_for_model,
    deployments_for_model_wildcard,
    pick_deployment,
    pick_deployment_more_strategies,
    fallbacks_for_model,
    router_new,
    latency_store_p50,
);
criterion_main!(benches);
