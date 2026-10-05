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
    let tier = ctx.billable_service_tier(body);
    let models = [
        Some(model),
        ctx.pricing_model.as_deref(),
        Some(ctx.model.as_str()),
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
        })
    });
    let reported_cost = || {
        body_cost
            .filter(|cost| valid_cost(*cost))
            .or_else(|| body.and_then(crate::pipeline::extract_body_usage_cost))
    };
    // Perplexity reports a complete receipt, including dynamic search/request
    // fees absent from token counts. Keep explicit operator pricing in control,
    // but otherwise use the receipt for the header, budgets, ledger and logs.
    if provider == "perplexity" && ctx.deployment_pricing.is_none() {
        if let Some(cost) = reported_cost() {
            return Some(cost);
        }
    }
    if table_cost.is_some() {
        return table_cost;
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

/// Resolve the cost to embed in a telemetry/exporter payload for `resp`.
///
/// Returns `Some(cost)` when the response already carried a pre-computed cost
/// or when the price table resolves a row for the serving model + provider.
/// Returns `None` when no usage was reported or no price-row match exists.
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
