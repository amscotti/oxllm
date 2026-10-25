//! `LowestCost` strategy — picks the deployment with the cheapest per-token cost.

use crate::cost::{deployment_pricing_model, lookup_pricing_with_provider};
use crate::deployment::Deployment;
use crate::router::simple_shuffle::SimpleShuffle;
use crate::router::strategy::{RequestContext, RoutingStrategy};

/// Picks the deployment with the lowest per-token cost.
///
/// Cost is computed as `input_cost_per_token + output_cost_per_token` — the
/// per-deployment pricing override (`OxllmParams::pricing_override`) when
/// present, otherwise the price-table entry. The static sum is a documented
/// approximation of request cost: it does not weight the rates by the
/// request's actual input/output mix (a genuine divergence from the reference
/// `lowest_cost`, which accounts for measured token usage), so deployments
/// with inverted input/output rates are ranked by the sum.
///
/// Deployments with neither an override nor a price-table entry are skipped,
/// as are deployments whose summed per-token rate is non-finite (a NaN/inf
/// pricing override or table rate reads as "unpriced" rather than winning by
/// poisoning every comparison). If no deployment has usable pricing data,
/// falls back to [`SimpleShuffle`].
#[derive(Debug, Default)]
pub struct LowestCost;

impl RoutingStrategy for LowestCost {
    fn pick<'a>(
        &self,
        deployments: &[&'a Deployment],
        ctx: &RequestContext,
    ) -> Option<&'a Deployment> {
        if deployments.is_empty() {
            return None;
        }
        let fallback = SimpleShuffle;
        let mut best: Option<(&Deployment, f64)> = None;
        for dep in deployments {
            // A NaN cost must be skipped, not installed as `best`: the very
            // first NaN-priced candidate would otherwise win deterministically
            // (every later `cost < NaN` comparison is false). Mirrors the
            // adaptive strategy's non-finite-rate handling. A per-second-only
            // override has no per-token sum to rank by, so it falls through
            // to the table entry (ranking it 0.0 would read as "free").
            let cost = match dep.oxllm_params.pricing_override() {
                Some(dp) if !dp.is_per_second_priced() => {
                    let sum = dp.input_cost_per_token + dp.output_cost_per_token;
                    if !sum.is_finite() {
                        continue;
                    }
                    sum
                }
                _ => {
                    let pricing_model = deployment_pricing_model(dep, &ctx.model);
                    match lookup_pricing_with_provider(&pricing_model, dep.provider_slug()) {
                        Some(pricing) => {
                            if pricing.is_per_second_priced() {
                                continue;
                            }
                            let sum = pricing.input_cost_per_token + pricing.output_cost_per_token;
                            if !sum.is_finite() {
                                continue;
                            }
                            sum
                        }
                        None => continue,
                    }
                }
            };
            match &best {
                None => best = Some((dep, cost)),
                Some((_, bc)) if cost < *bc => best = Some((dep, cost)),
                _ => {}
            }
        }
        match best {
            Some((dep, _)) => Some(dep),
            None => fallback.pick(deployments, ctx),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deployment::OxllmParams;

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

    fn make_dep_with_rates(name: &str, model: &str, input: f64, output: f64) -> Deployment {
        Deployment {
            model_name: name.to_string(),
            oxllm_params: OxllmParams {
                model: model.to_string(),
                input_cost_per_token: Some(input),
                output_cost_per_token: Some(output),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[test]
    fn empty_returns_none() {
        let strategy = LowestCost;
        let ctx = RequestContext::new("gpt-4o");
        assert!(strategy.pick(&[], &ctx).is_none());
    }

    #[test]
    fn picks_cheapest_priced_deployment() {
        let strategy = LowestCost;
        let ctx = RequestContext::new("model");
        let deps = [
            make_dep("model", "openai/gpt-4o"),
            make_dep("model", "openai/gpt-4o-mini"),
        ];
        let refs: Vec<&Deployment> = deps.iter().collect();
        let picked = strategy.pick(&refs, &ctx).expect("should pick");
        assert!(
            picked.oxllm_params.model.contains("mini"),
            "should pick the cheaper model, got {}",
            picked.oxllm_params.model
        );
    }

    #[test]
    fn unpriced_falls_back_to_shuffle() {
        let strategy = LowestCost;
        let ctx = RequestContext::new("model");
        let deps = [make_dep("model", "totally-unknown-model-xyz")];
        let refs: Vec<&Deployment> = deps.iter().collect();
        let picked = strategy.pick(&refs, &ctx);
        assert!(picked.is_some());
    }

    #[test]
    fn mixed_priced_and_unpriced_picks_priced() {
        let strategy = LowestCost;
        let ctx = RequestContext::new("model");
        let deps = [
            make_dep("model", "unknown-model-xyz"),
            make_dep("model", "openai/gpt-4o-mini"),
        ];
        let refs: Vec<&Deployment> = deps.iter().collect();
        let picked = strategy.pick(&refs, &ctx).expect("should pick");
        assert!(picked.oxllm_params.model.contains("mini"));
    }

    #[test]
    fn base_model_prices_opaque_deployment_aliases() {
        for hint_in_model_info in [true, false] {
            let mut aliased = make_dep("aliased", "openai/private-alias");
            if hint_in_model_info {
                aliased.model_info = Some(crate::deployment::ModelInfo {
                    base_model: Some("openai/gpt-4o-mini".to_string()),
                    ..Default::default()
                });
            } else {
                aliased.oxllm_params.extra.insert(
                    "base_model".to_string(),
                    serde_json::json!("openai/gpt-4o-mini"),
                );
            }
            let deps = [make_dep("catalog", "openai/gpt-4o"), aliased];
            let refs: Vec<&Deployment> = deps.iter().collect();
            let picked = LowestCost
                .pick(&refs, &RequestContext::new("model"))
                .expect("base model should price the alias");
            assert_eq!(picked.model_name, "aliased");
        }
    }

    #[test]
    fn namespaced_base_model_uses_the_serving_provider_catalog() {
        for provider in ["groq", "deepinfra", "together_ai"] {
            let mut aliased = make_dep("aliased", &format!("{provider}/private-alias"));
            aliased.model_info = Some(crate::deployment::ModelInfo {
                base_model: Some("openai/gpt-oss-120b".to_string()),
                ..Default::default()
            });
            let expected = format!("{provider}/openai/gpt-oss-120b");
            assert_eq!(deployment_pricing_model(&aliased, "model"), expected);
            assert!(lookup_pricing_with_provider(&expected, provider).is_some());
        }
    }

    #[test]
    fn non_exact_namespaced_hint_returns_canonical_model_for_overrides() {
        let mut aliased = make_dep("aliased", "azure_ai/private-alias");
        aliased.model_info = Some(crate::deployment::ModelInfo {
            base_model: Some("anthropic/claude-sonnet-5-5".to_string()),
            ..Default::default()
        });
        aliased.oxllm_params.input_cost_per_token = Some(1e-6);
        let resolved = deployment_pricing_model(&aliased, "model");
        assert_eq!(resolved, "azure_ai/claude-sonnet-5-5");
        aliased.oxllm_params.model = resolved;
        let override_pricing = aliased.oxllm_params.pricing_override().unwrap();
        assert_eq!(override_pricing.cache_read_input_token_cost, Some(2e-7));
    }

    #[test]
    fn lowest_cost_prefers_pricing_override_over_table() {
        // A deployment carrying an explicit `input_cost_per_token` /
        // `output_cost_per_token` override is ranked by the override, not the
        // price table — matching the adaptive strategy's `normalized_costs`
        // and the cost/budget hooks, so the two cost-aware strategies agree
        // on the same deployment.
        let strategy = LowestCost;
        let ctx = RequestContext::new("model");
        let deps = [
            make_dep("table", "openai/gpt-4o"),
            make_dep_with_rates("override", "openai/gpt-4o", 0.0000001, 0.0000001),
        ];
        let refs: Vec<&Deployment> = deps.iter().collect();
        let picked = strategy.pick(&refs, &ctx).expect("should pick");
        assert_eq!(
            picked.model_name, "override",
            "the cheaper override must win over the equal table rate"
        );
    }

    #[test]
    fn per_second_override_ranks_by_table_not_zero() {
        // A per-second-only override has no per-token sum; it must fall
        // through to the table entry instead of ranking as a free (0.0)
        // deployment — otherwise a duration-priced override would
        // deterministically win every cost comparison.
        let strategy = LowestCost;
        let ctx = RequestContext::new("model");
        let deps = [
            Deployment {
                model_name: "per_second".to_string(),
                oxllm_params: OxllmParams {
                    model: "openai/gpt-4o".to_string(),
                    input_cost_per_second: Some(0.00042),
                    ..Default::default()
                },
                ..Default::default()
            },
            make_dep("cheaper_table", "openai/gpt-4o-mini"),
        ];
        let refs: Vec<&Deployment> = deps.iter().collect();
        let picked = strategy.pick(&refs, &ctx).expect("should pick");
        assert_eq!(
            picked.model_name, "cheaper_table",
            "a per-second override must not rank as a free deployment"
        );
    }

    #[test]
    fn lowest_cost_ranks_by_static_sum_not_request_mix() {
        // Both deployments have the same per-token SUM (0.000012) but
        // inverted input/output rates: A is cheap-input/expensive-output,
        // B is expensive-input/cheap-output. `LowestCost` ranks by the static
        // sum, so an output-heavy workload that would bill cheaper on B is
        // still routed by the sum — ties keep the first deployment. This
        // pins the documented approximation (the reference weights by measured
        // usage; oxllm deliberately does not).
        let strategy = LowestCost;
        let ctx = RequestContext::new("model");
        let deps = [
            make_dep_with_rates("A", "openai/a", 0.000002, 0.000010),
            make_dep_with_rates("B", "openai/b", 0.000010, 0.000002),
        ];
        let refs: Vec<&Deployment> = deps.iter().collect();
        let picked = strategy.pick(&refs, &ctx).expect("should pick");
        assert_eq!(
            picked.model_name, "A",
            "equal static sums must tie on the first deployment, not re-rank by request mix"
        );
    }

    #[test]
    fn lowest_cost_picks_finite_over_nan_override() {
        // A NaN-priced first candidate (a `pricing_override` carrying `.nan`
        // from YAML, or a NaN table rate) must NOT pin the pick: every later
        // finite candidate must win, because `cost < NaN` is always false.
        let strategy = LowestCost;
        let ctx = RequestContext::new("model");
        let deps = [
            make_dep_with_rates("nan", "openai/gpt-4o", f64::NAN, 0.00001),
            make_dep_with_rates("finite", "openai/gpt-4o-mini", 0.0000001, 0.0000001),
        ];
        let refs: Vec<&Deployment> = deps.iter().collect();
        let picked = strategy.pick(&refs, &ctx).expect("should pick");
        assert_eq!(
            picked.model_name, "finite",
            "the finite-priced candidate must win over the NaN-priced one"
        );
    }

    #[test]
    fn lowest_cost_skips_infinite_rates_as_unpriced() {
        // `inf`/`-inf` per-token rates read as unpriced, same as a missing
        // entry — the finite candidate wins instead of the inf one losing by
        // a poisoned comparison.
        let strategy = LowestCost;
        let ctx = RequestContext::new("model");
        let deps = [
            make_dep_with_rates("inf", "openai/gpt-4o", f64::INFINITY, 0.00001),
            make_dep_with_rates("finite", "openai/gpt-4o-mini", 0.0000001, 0.0000001),
        ];
        let refs: Vec<&Deployment> = deps.iter().collect();
        let picked = strategy.pick(&refs, &ctx).expect("should pick");
        assert_eq!(picked.model_name, "finite");
    }

    #[test]
    fn lowest_cost_all_nan_falls_back_to_shuffle() {
        // Every candidate NaN-priced → no usable pricing → shuffle (must not
        // deterministically pin the first deployment).
        let strategy = LowestCost;
        let ctx = RequestContext::new("model");
        let deps = [
            make_dep_with_rates("a", "openai/a", f64::NAN, f64::NAN),
            make_dep_with_rates("b", "openai/b", f64::NAN, f64::NAN),
        ];
        let refs: Vec<&Deployment> = deps.iter().collect();
        assert!(strategy.pick(&refs, &ctx).is_some());
    }
}
