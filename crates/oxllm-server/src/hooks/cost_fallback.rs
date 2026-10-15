//! Shared cost-fallback helper for telemetry/exporter hooks.
//!
//! Every observability backend (Langfuse, Langsmith, Helicone, Datadog, MLflow,
//! W&B, S3, generic callback) embeds a per-request cost in its payload. When
//! the pipeline pre-computed the cost (`ResponseSummary::cost`) that value is
//! authoritative; otherwise the hook re-derives one from the price table.
//!
//! Historically each hook called the provider-agnostic
//! [`oxllm_core::calculate_cost_with_override`], which silently falls back to a
//! different provider's bare alias when the response model name is bare (e.g.
//! `gemini-2.5-pro` echoed by an OpenAI-compat shim). The primary billing path
//! ([`crate::hooks::cost::CostTrackingHook`]) uses the provider-aware variant
//! specifically to avoid this skew. This helper routes every telemetry fallback
//! through the same provider-aware lookup so the cost figure shown in
//! Langfuse / Datadog / W&B / S3 / etc. agrees with `spend_logs`.
//!
//! See `hooks/cost.rs` and `crates/oxllm-core/src/cost.rs` for the underlying
//! resolution semantics.

use oxllm_core::{CallContext, ResponseSummary, UsageSummary, calculate_cost_for_service_tier};
use serde_json::Value;

/// Canonical response-cost resolution used by headers, budgets, spend logs and
/// exporters. Public aliases retain their wire identity while pricing can fall
/// back to the concrete model selected by the router.
pub(crate) fn resolve_cost(
    ctx: &CallContext,
    model: &str,
    usage: Option<&UsageSummary>,
    body: Option<&Value>,
    body_cost: Option<f64>,
    provider: &str,
) -> Option<f64> {
    let valid_cost = |cost: f64| cost.is_finite() && cost >= 0.0;
    if let Some(cost) = ctx.cost_override.filter(|cost| valid_cost(*cost)) {
        return Some(cost);
    }
    if usage.is_none() && ctx.deployment_pricing.as_ref().is_some_and(pricing_is_free) {
        // A receipt without token counters is not evidence of a charge under
        // a completely free override. Preserve absent usage rather than invent
        // zero token counts just to resolve the operator's known zero price.
        return Some(0.0);
    }
    let tier = ctx.billable_service_tier(body);
    let models = [
        Some(model),
        ctx.pricing_model.as_deref(),
        Some(ctx.model.as_str()),
        ctx.deployment_model.as_deref(),
    ];
    let table_cost = usage.and_then(|usage| {
        models.iter().flatten().find_map(|model| {
            calculate_cost_for_service_tier(
                model,
                usage,
                ctx.deployment_pricing.as_ref(),
                provider,
                tier,
            )
            .map(|cost| (cost, *model))
        })
    });
    let reported_cost = || {
        body_cost
            .filter(|cost| valid_cost(*cost))
            .or_else(|| body.and_then(crate::pipeline::extract_body_usage_cost))
    };
    // Aggregator receipts include routing/search fees absent from token counts.
    // Keep explicit operator prices authoritative; otherwise use a measured
    // receipt consistently for headers, budgets, the ledger and exporters.
    if matches!(provider, "perplexity" | "openrouter") && ctx.deployment_pricing.is_none() {
        if let Some(cost) = reported_cost() {
            return Some(cost);
        }
    }
    if let Some((cost, priced_model)) = table_cost {
        if crate::pipeline::is_model_router_fee_entry(priced_model)
            && let Some((usage, rate)) = usage.zip(ctx.model_router_input_rate)
        {
            return Some(usage.prompt_tokens as f64 * rate);
        }
        let fee = usage
            .zip(ctx.model_router_input_rate)
            .filter(|_| !crate::pipeline::is_model_router_fee_entry(priced_model))
            .map_or(0.0, |(usage, rate)| usage.prompt_tokens as f64 * rate);
        return Some(cost + fee);
    }
    if let Some((usage, rate)) = usage.zip(ctx.model_router_input_rate) {
        return Some(usage.prompt_tokens as f64 * rate);
    }
    let duration_override = ctx
        .deployment_pricing
        .as_ref()
        .filter(|pricing| pricing.is_per_second_priced());
    let duration_table = if ctx.deployment_pricing.is_none() {
        models.iter().flatten().find_map(|model| {
            oxllm_core::lookup_pricing_with_provider(model, provider)
                .filter(|pricing| pricing.is_per_second_priced())
        })
    } else {
        None
    };
    if duration_override.is_none() && duration_table.is_none() {
        // Normal unpriced requests (and token-priced requests lacking usage)
        // do not require an upstream duration observation.
        return reported_cost();
    }
    let Some(start) = ctx.upstream_start_time else {
        tracing::warn!(
            request_id = %ctx.request_id,
            "duration pricing skipped: upstream timing is unavailable"
        );
        // Admission time includes local work, backoff and previous attempts;
        // it is never evidence of the duration of the serving upstream call.
        return reported_cost();
    };
    let elapsed = start.elapsed().as_millis() as f64;
    duration_override
        .and_then(|pricing| pricing.cost_per_second(elapsed))
        .or_else(|| duration_table.and_then(|pricing| pricing.cost_per_second(elapsed)))
        .or_else(reported_cost)
}

/// Prove a zero charge across every explicit bucket and context tier. Duration
/// pricing follows its existing canonical/input/output precedence; otherwise
/// token pricing wins, even when an inactive duration rate is configured.
pub(crate) fn pricing_is_free(pricing: &oxllm_core::DeploymentPricing) -> bool {
    // Match the core billing rule: negative and non-finite operator rates
    // collapse to zero, including when counters or timing are unavailable.
    let zero = |rate: f64| !rate.is_finite() || rate <= 0.0;
    if pricing.is_per_second_priced() {
        return pricing
            .cost_per_second_rate
            .or(pricing.input_cost_per_second)
            .or(pricing.output_cost_per_second)
            .is_some_and(zero);
    }
    let zero = |rate: Option<f64>| rate.is_none_or(zero);
    [
        Some(pricing.input_cost_per_token),
        Some(pricing.output_cost_per_token),
        pricing.cache_read_input_token_cost,
        pricing.cache_creation_input_token_cost,
        pricing.output_cost_per_reasoning_token,
        pricing.cache_creation_input_token_cost_above_1hr,
    ]
    .into_iter()
    .all(zero)
        && pricing.tiers.iter().all(|tier| {
            [
                tier.input_cost_per_token,
                tier.output_cost_per_token,
                tier.cache_read_input_token_cost,
                tier.cache_creation_input_token_cost,
                tier.cache_creation_input_token_cost_above_1hr,
            ]
            .into_iter()
            .all(zero)
        })
}

/// Resolve the cost to embed in a telemetry/exporter payload for `resp`.
///
/// Returns `Some(cost)` for a pre-computed charge, applicable rate card,
/// measured receipt, timed duration, or proven free deployment. Returns `None`
/// when no supported source establishes a cost.
///
/// Mirrors [`CostTrackingHook`](crate::hooks::cost::CostTrackingHook)'s
/// fallback exactly: provider-aware lookup on the response model first, then a
/// second attempt against the *requested* model (`ctx.model`) when it differs
/// from the response model. The provider argument is always `resp.provider`
/// (the provider that actually served the request), never empty.
///
/// Defence in depth: returns `None` for cache hits. Cached responses carry no
/// fresh upstream observation and must not be billed or reported as a cost by
/// telemetry (matching `spend_logs`, which skips cache hits). Every current
/// caller already guards on `ctx.cache_hit` before calling, so this is a
/// no-op for them — the central check exists so a future telemetry hook that
/// forgets the guard cannot emit an inconsistent non-zero cost.
pub fn fallback_cost(ctx: &CallContext, resp: &ResponseSummary) -> Option<f64> {
    if ctx.cache_hit {
        return None;
    }
    if let Some(c) = resp.cost {
        return Some(c);
    }
    resolve_cost(
        ctx,
        &resp.model,
        resp.usage.as_ref(),
        resp.body.as_ref(),
        None,
        &resp.provider,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxllm_core::{CallContext, ResponseSummary, UsageSummary};

    fn usage(prompt: u64, completion: u64) -> UsageSummary {
        UsageSummary {
            prompt_tokens: prompt,
            completion_tokens: completion,
            total_tokens: prompt + completion,
            cache_read_input_tokens: None,
            cache_creation_input_tokens: None,
            audio_input_tokens: None,
            reasoning_tokens: None,
            citation_tokens: None,
            web_search_requests: None,
            cache_creation_1h_input_tokens: None,
        }
    }

    #[test]
    fn aggregator_receipts_override_fallback_cards_but_not_operator_prices() {
        let model = "openrouter/openai/gpt-4o";
        let mut ctx = CallContext::for_test(model);
        let measured = usage(10, 5);
        for receipt in [0.0, 0.017, 0.001] {
            let body = serde_json::json!({"usage":{"cost":receipt}});
            assert_eq!(
                resolve_cost(
                    &ctx,
                    model,
                    Some(&measured),
                    Some(&body),
                    None,
                    "openrouter"
                ),
                Some(receipt)
            );
            assert_eq!(
                resolve_cost(
                    &ctx,
                    model,
                    Some(&measured),
                    None,
                    Some(receipt),
                    "openrouter"
                ),
                Some(receipt)
            );
        }
        let fallback =
            oxllm_core::calculate_cost_with_provider(model, &measured, "openrouter").unwrap();
        for receipt in [None, Some(-1.0), Some(f64::NAN), Some(f64::INFINITY)] {
            assert_eq!(
                resolve_cost(&ctx, model, Some(&measured), None, receipt, "openrouter"),
                Some(fallback)
            );
        }
        ctx.deployment_pricing = Some(oxllm_core::DeploymentPricing {
            input_cost_per_token: 0.002,
            output_cost_per_token: 0.005,
            ..Default::default()
        });
        let expected = oxllm_core::calculate_cost_with_override_for_provider(
            model,
            &measured,
            ctx.deployment_pricing.as_ref(),
            "openrouter",
        )
        .unwrap();
        assert_eq!(
            resolve_cost(
                &ctx,
                model,
                Some(&measured),
                None,
                Some(0.017),
                "openrouter"
            ),
            Some(expected)
        );
        ctx.deployment_pricing = Some(oxllm_core::DeploymentPricing {
            input_cost_per_token: 0.0,
            output_cost_per_token: 0.0,
            ..Default::default()
        });
        assert_eq!(
            resolve_cost(
                &ctx,
                model,
                Some(&measured),
                None,
                Some(0.017),
                "openrouter"
            ),
            Some(0.0)
        );
        assert_eq!(
            resolve_cost(&ctx, model, None, None, Some(0.017), "openrouter"),
            Some(0.0)
        );
        // Cache-only or tier-only prices cannot be declared free when the
        // serving provider omitted the counters needed to price those buckets.
        ctx.deployment_pricing = Some(oxllm_core::DeploymentPricing {
            cache_read_input_token_cost: Some(1e-6),
            ..Default::default()
        });
        assert_eq!(
            resolve_cost(&ctx, model, None, None, Some(0.017), "openrouter"),
            Some(0.017)
        );
        ctx.deployment_pricing = Some(oxllm_core::DeploymentPricing {
            tiers: vec![oxllm_core::cost::CostTier {
                threshold_tokens: 1,
                input_cost_per_token: Some(1e-6),
                output_cost_per_token: None,
                cache_read_input_token_cost: None,
                cache_creation_input_token_cost: None,
                cache_creation_input_token_cost_above_1hr: None,
            }],
            ..Default::default()
        });
        assert_eq!(
            resolve_cost(&ctx, model, None, None, Some(0.017), "openrouter"),
            Some(0.017)
        );
        ctx.deployment_pricing = Some(oxllm_core::DeploymentPricing {
            cost_per_second_rate: Some(0.0),
            input_cost_per_second: Some(1.0),
            ..Default::default()
        });
        assert_eq!(
            resolve_cost(&ctx, model, None, None, Some(0.017), "openrouter"),
            Some(0.0)
        );
        ctx.deployment_pricing = Some(oxllm_core::DeploymentPricing {
            cache_read_input_token_cost: Some(0.0),
            cost_per_second_rate: Some(1.0),
            ..Default::default()
        });
        assert_eq!(
            resolve_cost(&ctx, model, None, None, Some(0.017), "openrouter"),
            Some(0.0)
        );
    }

    #[test]
    fn concrete_deployment_remains_a_last_resort_for_unpriced_response_aliases() {
        let mut ctx = CallContext::for_test("public-alias");
        ctx.deployment_model = Some("openrouter/openai/gpt-4o".into());
        let measured = usage(5, 1);
        assert_eq!(
            resolve_cost(
                &ctx,
                "unpriced-response-alias",
                Some(&measured),
                None,
                None,
                "openrouter"
            ),
            Some(5.0 * 2.5e-6 + 10.0e-6)
        );
        assert_eq!(
            resolve_cost(
                &ctx,
                "openai/gpt-4o-mini",
                Some(&measured),
                None,
                None,
                "openrouter"
            ),
            Some(5.0 * 0.15e-6 + 0.60e-6)
        );
    }

    #[test]
    fn receipt_only_free_pricing_matches_rate_sanitization() {
        let model = "private-model";
        let mut ctx = CallContext::for_test(model);
        for rate in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            ctx.deployment_pricing = Some(oxllm_core::DeploymentPricing {
                input_cost_per_token: rate,
                output_cost_per_token: rate,
                cache_read_input_token_cost: Some(rate),
                cache_creation_input_token_cost: Some(rate),
                output_cost_per_reasoning_token: Some(rate),
                cache_creation_input_token_cost_above_1hr: Some(rate),
                ..Default::default()
            });
            let measured = usage(5, 1);
            assert_eq!(
                resolve_cost(
                    &ctx,
                    model,
                    Some(&measured),
                    None,
                    Some(0.017),
                    "openrouter"
                ),
                Some(0.0)
            );
            assert_eq!(
                resolve_cost(&ctx, model, None, None, Some(0.017), "openrouter"),
                Some(0.0),
                "missing counters must not turn sanitized zero rates into a receipt charge"
            );
            ctx.deployment_pricing = Some(oxllm_core::DeploymentPricing {
                cost_per_second_rate: Some(rate),
                ..Default::default()
            });
            assert_eq!(
                ctx.deployment_pricing
                    .as_ref()
                    .unwrap()
                    .cost_per_second(1000.0),
                Some(0.0)
            );
            assert_eq!(
                resolve_cost(&ctx, model, None, None, Some(0.017), "openrouter"),
                Some(0.0)
            );
        }
        ctx.deployment_pricing = Some(oxllm_core::DeploymentPricing {
            input_cost_per_token: 1e-6,
            output_cost_per_token: f64::NAN,
            ..Default::default()
        });
        assert_eq!(
            resolve_cost(&ctx, model, None, None, Some(0.017), "openrouter"),
            Some(0.017),
            "a positive effective bucket still needs usage and retains receipt fallback"
        );
    }

    /// HIGH #1 (parity with `cost.rs::cost_hook_fallback_uses_serving_provider_for_tiered_gemini`):
    /// a bare response model name (Gemini's OpenAI-compat shape) priced via
    /// the bare alias would resolve to a different provider's row with
    /// potentially different tier rates; the provider-aware lookup must pick
    /// the serving provider's canonical row (`gemini/gemini-2.5-pro`) so the
    /// cost figure sent to telemetry matches `spend_logs`.
    #[test]
    fn fallback_uses_serving_provider_for_tiered_gemini() {
        let ctx = CallContext::for_test("gemini-2.5-pro");
        // >200k prompt so the context tier fires for `gemini-2.5-pro`.
        let prompt = 250_000u64;
        let completion = 1_000u64;
        let resp = ResponseSummary {
            model: "gemini-2.5-pro".to_string(),
            cost: None,
            usage: Some(usage(prompt, completion)),
            body: None,
            provider: "gemini".to_string(),
        };
        let cost = fallback_cost(&ctx, &resp).expect("provider-aware fallback resolves");

        let expected = oxllm_core::calculate_cost_with_provider(
            "gemini-2.5-pro",
            resp.usage.as_ref().unwrap(),
            "gemini",
        )
        .expect("provider-aware tier cost resolves");
        assert!(
            (cost - expected).abs() < 1e-9,
            "fallback must use the serving provider's tiered row: got {cost}, expected {expected}"
        );
        assert_eq!(expected, 250_000.0 * 2.5e-6 + 1_000.0 * 1.5e-5);
    }

    /// Pre-computed `resp.cost` is authoritative and returned verbatim — the
    /// pipeline already did the work, the helper must not recompute.
    #[test]
    fn fallback_prefers_response_cost_when_present() {
        let ctx = CallContext::for_test("gpt-4o");
        let resp = ResponseSummary {
            model: "gpt-4o".to_string(),
            cost: Some(0.1234),
            usage: Some(usage(10, 5)),
            body: None,
            provider: "openai".to_string(),
        };
        assert_eq!(fallback_cost(&ctx, &resp), Some(0.1234));
    }

    /// No usage → no cost can be derived.
    #[test]
    fn fallback_returns_none_without_usage() {
        let ctx = CallContext::for_test("gpt-4o");
        let resp = ResponseSummary {
            model: "gpt-4o".to_string(),
            cost: None,
            usage: None,
            body: None,
            provider: "openai".to_string(),
        };
        assert!(fallback_cost(&ctx, &resp).is_none());
    }

    /// Defence in depth (MEDIUM): even with usage + a known model, a cache-hit
    /// context must return `None` so a future telemetry hook that forgets the
    /// `cache_hit` guard cannot emit an inconsistent non-zero cost (disagreeing
    /// with `spend_logs`, which skips cache hits). Today every caller guards
    /// before calling, so this is a no-op for them.
    #[test]
    fn fallback_cost_returns_none_on_cache_hit() {
        let mut ctx = CallContext::for_test("gpt-4o");
        ctx.cache_hit = true;
        let resp = ResponseSummary {
            model: "gpt-4o".to_string(),
            cost: None,
            usage: Some(usage(10, 5)),
            body: None,
            provider: "openai".to_string(),
        };
        assert_eq!(
            fallback_cost(&ctx, &resp),
            None,
            "cache-hit context must yield no cost"
        );
    }

    /// Even when the pipeline pre-computed `resp.cost`, a cache-hit context
    /// must still return `None` — the cached response was already billed at
    /// store time and must not be re-reported.
    #[test]
    fn fallback_cost_returns_none_on_cache_hit_even_with_response_cost() {
        let mut ctx = CallContext::for_test("gpt-4o");
        ctx.cache_hit = true;
        let resp = ResponseSummary {
            model: "gpt-4o".to_string(),
            cost: Some(0.5),
            usage: Some(usage(10, 5)),
            body: None,
            provider: "openai".to_string(),
        };
        assert_eq!(fallback_cost(&ctx, &resp), None);
    }

    /// Coverage gap: when a `deployment_pricing` override is configured, the
    /// fallback must use the override's rates (not the price-table rates),
    /// matching the billing path's `calculate_cost_with_override_for_provider`.
    #[test]
    fn fallback_uses_deployment_pricing_override() {
        use oxllm_core::DeploymentPricing;

        let mut ctx = CallContext::for_test("gpt-4o");
        ctx.deployment_pricing = Some(DeploymentPricing {
            input_cost_per_token: 1e-6,
            output_cost_per_token: 2e-6,
            cache_read_input_token_cost: None,
            cache_creation_input_token_cost: None,
            output_cost_per_reasoning_token: None,
            ..Default::default()
        });
        let resp = ResponseSummary {
            model: "gpt-4o".to_string(),
            cost: None,
            usage: Some(usage(100, 50)),
            body: None,
            provider: "openai".to_string(),
        };
        let cost = fallback_cost(&ctx, &resp).expect("override resolves a cost");
        assert!(
            (cost - (100.0 * 1e-6 + 50.0 * 2e-6)).abs() < 1e-12,
            "override rates must be used: got {cost}"
        );
    }

    /// An unpriced bare model name with no alias resolution yields `None`
    /// (no panic), mirroring `cost.rs`'s `(Some(usage), None)` branch.
    #[test]
    fn fallback_returns_none_for_unknown_model() {
        let ctx = CallContext::for_test("totally-unknown-requested-model");
        let resp = ResponseSummary {
            model: "totally-unknown-model".to_string(),
            cost: None,
            usage: Some(usage(10, 5)),
            body: None,
            provider: "openai".to_string(),
        };
        assert!(fallback_cost(&ctx, &resp).is_none());
    }

    #[test]
    fn model_router_fee_is_added_once_to_served_model_cost_and_canonical_fee_rows() {
        let rate = oxllm_core::cost::price_table()["azure_ai/model-router"].input_cost_per_token;
        let tokens = UsageSummary {
            cache_read_input_tokens: Some(40),
            ..usage(100, 20)
        };
        for provider in ["azure", "azure_ai"] {
            let mut ctx = CallContext::for_test("public-router");
            ctx.deployment_model = Some(format!("{provider}/model_router/production"));
            ctx.pricing_model = Some(format!("{provider}/model-router"));
            ctx.model_router_input_rate = Some(rate);
            let base =
                oxllm_core::calculate_cost_with_provider("gpt-6.1-sol", &tokens, provider).unwrap();
            let cost =
                resolve_cost(&ctx, "gpt-6.1-sol", Some(&tokens), None, None, provider).unwrap();
            assert!((cost - (base + 100.0 * rate)).abs() < 1e-12);
            let cost =
                resolve_cost(&ctx, "model-router", Some(&tokens), None, None, provider).unwrap();
            assert!(
                (cost - 100.0 * rate).abs() < 1e-12,
                "fee-only entries must not lose cached tokens or double charge"
            );
            ctx.cost_override = Some(0.0);
            assert_eq!(
                resolve_cost(&ctx, "gpt-6.1-sol", Some(&tokens), None, None, provider),
                Some(0.0)
            );
        }
    }

    #[test]
    fn model_router_detection_requires_explicit_selector_identity() {
        for model in [
            "model-router",
            "model_router",
            "model_router/production",
            "model-router/production",
            "azure-model-router",
            "azure_model_router",
            "azure/model-router",
            "azure_ai/model_router/production",
            "AZURE_AI/MODEL-ROUTER/production",
        ] {
            assert!(crate::pipeline::is_model_router_name(model), "{model}");
        }
        for model in [
            "not-model-router",
            "ordinary-model-router-wrapper",
            "ordinary_model_router_wrapper",
            "model-routerish",
            "model_router_backup",
            "model-router/",
            "azure_ai/custom/model-router",
            "openai/model-router",
        ] {
            assert!(!crate::pipeline::is_model_router_name(model), "{model}");
        }
        for model in [
            "model-router",
            "model_router",
            "azure/model-router",
            "azure_ai/model_router",
            "AZURE_AI/MODEL-ROUTER",
        ] {
            assert!(crate::pipeline::is_model_router_fee_entry(model), "{model}");
        }
        for model in [
            "azure-model-router",
            "model_router/production",
            "ordinary-model-router-wrapper",
        ] {
            assert!(
                !crate::pipeline::is_model_router_fee_entry(model),
                "{model}"
            );
        }
    }

    #[tokio::test]
    async fn model_router_fee_follows_deployment_identity_and_base_model_hints() {
        let config = serde_json::from_value(serde_json::json!({"model_list":[]})).unwrap();
        let state =
            crate::state::AppState::with_registry(config, oxllm_providers::default_registry())
                .unwrap();
        let rate = oxllm_core::cost::price_table()["azure_ai/model-router"].input_cost_per_token;
        let tokens = UsageSummary {
            cache_read_input_tokens: Some(40),
            ..usage(100, 20)
        };
        for provider in ["azure", "azure_ai", "openai"] {
            let base =
                oxllm_core::calculate_cost_with_provider("gpt-6.1-sol", &tokens, provider).unwrap();
            for (native, base_model, router) in [
                ("model-router", None, true),
                ("model_router", None, true),
                ("model_router/production", None, true),
                ("model-router/production", None, true),
                ("azure-model-router", None, true),
                ("azure_model_router", None, true),
                ("ordinary-model-router-wrapper", None, false),
                ("ordinary_model_router_wrapper", None, false),
                ("not-model-router", None, false),
                ("gpt-6.1-sol", None, false),
                ("opaque-deployment", Some("model-router"), true),
                ("my-model-router-deployment", Some("model-router"), true),
                ("prod-model_router", Some("model-router"), true),
                ("opaque-deployment", Some("gpt-6.1-sol"), false),
            ] {
                let deployment = serde_json::from_value(serde_json::json!({
                    "model_name":"public-model-router-alias",
                    "oxllm_params":{"model":format!("{provider}/{native}")},
                    "model_info":{"base_model":base_model}
                }))
                .unwrap();
                let mut ctx = CallContext::for_test("public-model-router-alias");
                // A later ordinary deployment must also clear a prior attempt's fee.
                ctx.model_router_input_rate = Some(rate);
                crate::pipeline::prepare_deployment_pricing(
                    &state,
                    &mut ctx,
                    &deployment,
                    &serde_json::json!({"model":"public-model-router-alias"}),
                )
                .await
                .unwrap();
                let routed = router && matches!(provider, "azure" | "azure_ai");
                assert_eq!(
                    ctx.model_router_input_rate,
                    routed.then_some(rate),
                    "{provider}/{native}"
                );
                let cost =
                    resolve_cost(&ctx, "gpt-6.1-sol", Some(&tokens), None, None, provider).unwrap();
                let expected = base + if routed { 100.0 * rate } else { 0.0 };
                assert!(
                    (cost - expected).abs() < 1e-12,
                    "{provider}/{native}: got {cost}, expected {expected}"
                );
            }
        }
    }

    #[tokio::test]
    async fn model_router_explicit_deployment_pricing_is_an_all_in_override() {
        let config = serde_json::from_value(serde_json::json!({"model_list":[]})).unwrap();
        let state =
            crate::state::AppState::with_registry(config, oxllm_providers::default_registry())
                .unwrap();
        let tokens = UsageSummary {
            cache_read_input_tokens: Some(40),
            ..usage(100, 20)
        };
        for provider in ["azure", "azure_ai"] {
            for input in [2e-6, 0.0] {
                let deployment = serde_json::from_value(serde_json::json!({
                    "model_name":"public-router",
                    "oxllm_params":{
                        "model":format!("{provider}/model_router/production"),
                        "input_cost_per_token":input,
                        "output_cost_per_token":input * 1.5,
                        "cache_read_input_token_cost":input * 0.5,
                        "cache_creation_input_token_cost":0.0
                    }
                }))
                .unwrap();
                let mut ctx = CallContext::for_test("public-router");
                ctx.model_router_input_rate = Some(
                    oxllm_core::cost::price_table()["azure_ai/model-router"].input_cost_per_token,
                );
                crate::pipeline::prepare_deployment_pricing(
                    &state,
                    &mut ctx,
                    &deployment,
                    &serde_json::json!({"model":"public-router"}),
                )
                .await
                .unwrap();
                assert!(ctx.deployment_pricing.is_some());
                assert_eq!(ctx.model_router_input_rate, None);
                let estimate = crate::hooks::budget::estimate_max_request_cost(&ctx).await;
                let prompt = *ctx.token_estimate.get().unwrap() as f64;
                let expected_estimate = prompt * input + 16_384.0 * input * 1.5;
                assert!((estimate - expected_estimate).abs() < 1e-12);
                for served_model in ["gpt-6.1-sol", "model-router"] {
                    let cost =
                        resolve_cost(&ctx, served_model, Some(&tokens), None, None, provider)
                            .unwrap();
                    let expected = 60.0 * input + 40.0 * input * 0.5 + 20.0 * input * 1.5;
                    assert!(
                        (cost - expected).abs() < 1e-12,
                        "explicit deployment rates must cover the complete charge"
                    );
                }
            }
        }
    }

    #[test]
    fn model_router_fee_overrides_include_cached_tokens_and_cache_hits_are_free() {
        let catalog_rate =
            oxllm_core::cost::price_table()["azure_ai/model-router"].input_cost_per_token;
        let tokens = UsageSummary {
            cache_read_input_tokens: Some(40),
            ..usage(100, 20)
        };
        for rate in [catalog_rate * 2.0, 0.0] {
            let mut ctx = CallContext::for_test("public-router");
            ctx.pricing_model = Some("azure_ai/model-router".into());
            ctx.model_router_input_rate = Some(rate);
            for model in ["gpt-6.1-sol", "model-router", "model_router"] {
                let mut response = ResponseSummary {
                    model: model.into(),
                    cost: None,
                    usage: Some(tokens.clone()),
                    body: None,
                    provider: "azure_ai".into(),
                };
                let base = if model == "gpt-6.1-sol" {
                    oxllm_core::calculate_cost_with_provider(model, &tokens, "azure_ai").unwrap()
                } else {
                    0.0
                };
                let expected = base + 100.0 * rate;
                let cost = fallback_cost(&ctx, &response).unwrap();
                assert!((cost - expected).abs() < 1e-12, "{model}/{rate}");
                ctx.cache_hit = true;
                assert_eq!(fallback_cost(&ctx, &response), None);
                response.cost = Some(expected);
                assert_eq!(fallback_cost(&ctx, &response), None);
                ctx.cache_hit = false;
                assert_eq!(fallback_cost(&ctx, &response), Some(expected));
            }
        }
    }

    #[test]
    fn concrete_serving_model_prices_unrecognized_public_alias() {
        let mut ctx = CallContext::for_test("public-alias");
        ctx.pricing_model = Some("openai/gpt-4o".into());
        let cost = resolve_cost(
            &ctx,
            "provider-public-alias",
            Some(&usage(10, 5)),
            None,
            None,
            "openai",
        )
        .expect("serving rate card");
        assert!((cost - 0.000075).abs() < 1e-12);
    }

    #[test]
    fn duration_prices_exclude_admission_and_prior_attempt_time() {
        let mut ctx = CallContext::for_test("private-model");
        ctx.start_time = Some(std::time::Instant::now() - std::time::Duration::from_secs(3600));
        ctx.upstream_start_time =
            Some(std::time::Instant::now() - std::time::Duration::from_secs(1));
        ctx.deployment_pricing = Some(oxllm_core::DeploymentPricing {
            cost_per_second_rate: Some(1.0),
            ..Default::default()
        });
        let cost =
            resolve_cost(&ctx, "private-model", None, None, None, "openai").expect("duration cost");
        assert!(
            (1.0..2.0).contains(&cost),
            "only the current attempt is billed: {cost}"
        );
    }

    #[test]
    fn r36_missing_upstream_time_never_bills_old_admission_time() {
        let mut ctx = CallContext::for_test("private-duration-model");
        ctx.request_id = "r36-missing-upstream-time".into();
        ctx.start_time = Some(std::time::Instant::now() - std::time::Duration::from_secs(3600));
        ctx.deployment_pricing = Some(oxllm_core::DeploymentPricing {
            cost_per_second_rate: Some(1.0),
            ..Default::default()
        });
        let (cost, log) = crate::security::capture_test_logs(|| {
            resolve_cost(&ctx, "private-duration-model", None, None, None, "openai")
        });
        assert_eq!(
            cost, None,
            "admission latency must never become invented spend"
        );
        assert!(log.contains("WARN"));
        assert!(log.contains("r36-missing-upstream-time"));
        assert!(log.contains("duration pricing skipped: upstream timing is unavailable"));
        assert!(
            !log.contains("private-duration-model"),
            "diagnostic should be generic"
        );
    }

    #[test]
    fn r36_table_duration_prices_also_require_upstream_time() {
        let model = "bedrock/ap-northeast-1/1-month-commitment/anthropic.claude-instant-v1";
        assert!(
            oxllm_core::lookup_pricing_with_provider(model, "bedrock")
                .expect("duration-priced row")
                .is_per_second_priced()
        );
        let mut ctx = CallContext::for_test(model);
        ctx.start_time = Some(std::time::Instant::now() - std::time::Duration::from_secs(3600));
        let (cost, log) = crate::security::capture_test_logs(|| {
            resolve_cost(&ctx, model, Some(&usage(100, 50)), None, None, "bedrock")
        });
        assert_eq!(cost, None);
        assert!(log.contains("upstream timing is unavailable"));
        assert!(log.contains(&ctx.request_id));
    }

    #[test]
    fn r36_reported_costs_survive_unknown_duration() {
        let mut ctx = CallContext::for_test("private-duration-model");
        ctx.start_time = Some(std::time::Instant::now() - std::time::Duration::from_secs(3600));
        ctx.deployment_pricing = Some(oxllm_core::DeploymentPricing {
            cost_per_second_rate: Some(1.0),
            ..Default::default()
        });
        let body = serde_json::json!({"usage": {"cost": 0.25}});
        let (costs, _) = crate::security::capture_test_logs(|| {
            (
                resolve_cost(
                    &ctx,
                    "private-duration-model",
                    None,
                    None,
                    Some(0.0),
                    "openai",
                ),
                resolve_cost(
                    &ctx,
                    "private-duration-model",
                    None,
                    Some(&body),
                    None,
                    "openai",
                ),
            )
        });
        assert_eq!(costs, (Some(0.0), Some(0.25)));
        ctx.cost_override = Some(0.5);
        let (cost, log) = crate::security::capture_test_logs(|| {
            resolve_cost(&ctx, "private-duration-model", None, None, None, "openai")
        });
        assert_eq!(cost, Some(0.5));
        assert!(!log.contains("duration pricing skipped"));
        ctx.cost_override = None;
        let response = ResponseSummary {
            model: ctx.model.clone(),
            provider: "openai".into(),
            cost: Some(0.75),
            usage: None,
            body: None,
        };
        let (cost, log) = crate::security::capture_test_logs(|| fallback_cost(&ctx, &response));
        assert_eq!(cost, Some(0.75));
        assert!(!log.contains("duration pricing skipped"));
    }

    #[test]
    fn r36_token_pricing_and_unpriced_requests_do_not_warn_about_duration() {
        let mut token_ctx = CallContext::for_test("openai/gpt-4o");
        token_ctx.start_time =
            Some(std::time::Instant::now() - std::time::Duration::from_secs(3600));
        let mut unknown_ctx = CallContext::for_test("unpriced-private-model");
        unknown_ctx.start_time = token_ctx.start_time;
        let (costs, log) = crate::security::capture_test_logs(|| {
            (
                resolve_cost(
                    &token_ctx,
                    "openai/gpt-4o",
                    Some(&usage(10, 5)),
                    None,
                    None,
                    "openai",
                ),
                resolve_cost(&token_ctx, "openai/gpt-4o", None, None, None, "openai"),
                resolve_cost(
                    &unknown_ctx,
                    "unpriced-private-model",
                    None,
                    None,
                    None,
                    "openai",
                ),
            )
        });
        let token_cost = costs.0.expect("token-based cost");
        assert!((token_cost - 0.000075).abs() < 1e-12);
        assert_eq!((costs.1, costs.2), (None, None));
        assert!(!log.contains("duration pricing skipped"));
    }

    /// When the response model is unknown but the *requested* model is priced,
    /// the second-attempt fallback resolves the cost via the requested name —
    /// matching the billing path's two-step lookup.
    #[test]
    fn fallback_falls_back_to_requested_model_when_response_model_unpriced() {
        let ctx = CallContext::for_test("gpt-4o");
        let resp = ResponseSummary {
            model: "gpt-4o-alias-blob".to_string(),
            cost: None,
            usage: Some(usage(10, 5)),
            body: None,
            provider: "openai".to_string(),
        };
        let cost = fallback_cost(&ctx, &resp);
        assert!(
            cost.is_some(),
            "second-attempt fallback against ctx.model must resolve when response model is bare"
        );
    }
}
