//! `MetricsHook` — records Prometheus metrics for each request.

use async_trait::async_trait;
use oxllm_core::{CallContext, Hook, ResponseSummary};

use crate::metrics;

/// Routing decisions by effective strategy and requested model. Stamped at
/// pick time onto [`CallContext::router_strategy`]; counted once per
/// completed request (success or failure) when a pick happened.
pub const ROUTER_DECISIONS_TOTAL: &str = "oxllm_router_decisions_total";

/// Count the request's routing decision, when the router made a pick.
/// `model` is the requested model group (consistent across the success and
/// failure paths, unlike the serving model on a fallback). Routed through
/// `metrics::record_router_decision` so the client-supplied model shares the
/// length and lifetime-cardinality bounds used by the other metric helpers.
fn record_router_decision(ctx: &CallContext) {
    if let Some(strategy) = ctx.router_strategy {
        metrics::record_router_decision(strategy.as_str(), &ctx.model);
    }
}

/// A hook that records request/token/cost metrics.
pub struct MetricsHook;

impl MetricsHook {
    pub fn new() -> Self {
        Self
    }
}

impl Default for MetricsHook {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Hook for MetricsHook {
    async fn post_success(&self, ctx: &CallContext, resp: &ResponseSummary) {
        // A cache hit is still a completed request. Only upstream usage/cost
        // and routing decisions are absent on that short-circuit path.
        metrics::record_request(&resp.model, &resp.provider, "success");
        if ctx.cache_hit {
            return;
        }
        record_router_decision(ctx);

        if let Some(usage) = &resp.usage {
            metrics::record_tokens(&resp.model, "prompt", usage.prompt_tokens);
            metrics::record_tokens(&resp.model, "completion", usage.completion_tokens);
        }

        if let Some(cost) = resp.cost {
            metrics::record_cost(&resp.model, cost);
        }
    }

    async fn post_failure(&self, ctx: &CallContext, err: &oxllm_core::Error) {
        record_router_decision(ctx);
        let error_type = match err {
            oxllm_core::Error::Upstream { .. } => "upstream",
            oxllm_core::Error::Network(_) => "network",
            oxllm_core::Error::RateLimited(_) => "rate_limited",
            oxllm_core::Error::BudgetExceeded(_) => "budget_exceeded",
            oxllm_core::Error::Unauthorized => "unauthorized",
            oxllm_core::Error::BadRequest(_) => "bad_request",
            oxllm_core::Error::ModelNotFound(_) => "model_not_found",
            _ => "internal",
        };
        metrics::record_error(error_type);
        // Record the request under the *requested* model so per-model error
        // rates are observable. Provider is unknown on the failure path
        // (the request may have failed before provider resolution), so the
        // label is left empty rather than guessed.
        //
        // Unknown-model rejections are the exception: the requested name
        // matched no deployment, so stamping it verbatim would let arbitrary
        // client input grow the registry one series per distinct name
        // (`bounded_label` caps length, not cardinality). Count those under
        // the fixed `unknown` placeholder instead.
        let model = match err {
            oxllm_core::Error::ModelNotFound(_) => "unknown",
            _ => &ctx.model,
        };
        metrics::record_request(model, "", "error");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxllm_core::UsageSummary;

    fn make_ctx() -> CallContext {
        CallContext::for_test("gpt-4o")
    }

    fn make_response(model: &str, prompt: u64, completion: u64) -> ResponseSummary {
        ResponseSummary {
            model: model.to_string(),
            cost: None,
            usage: Some(UsageSummary {
                prompt_tokens: prompt,
                completion_tokens: completion,
                total_tokens: prompt + completion,
                cache_read_input_tokens: None,
                cache_creation_input_tokens: None,
                audio_input_tokens: None,
                reasoning_tokens: None,
                citation_tokens: None,
                web_search_requests: None,
                browser_open_requests: None,
                cache_creation_1h_input_tokens: None,
            }),
            body: None,
            provider: "openai".to_string(),
        }
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn post_success_does_not_panic() {
        let hook = MetricsHook::new();
        let ctx = make_ctx();
        let resp = make_response("gpt-4o", 100, 50);
        hook.post_success(&ctx, &resp).await;
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn cache_hit_counts_request_without_usage_cost_or_routing() {
        let _ = metrics::render();
        let hook = MetricsHook::new();
        let mut ctx = CallContext::for_test("cache-hit-metric-model");
        ctx.cache_hit = true;
        let mut resp = make_response("cache-hit-metric-model", 100, 50);
        resp.cost = Some(0.5);
        hook.post_success(&ctx, &resp).await;
        let rendered = metrics::render();
        assert!(
            rendered.contains(
                "oxllm_requests_total{model=\"cache-hit-metric-model\",provider=\"openai\",status=\"success\"} 1"
            ),
            "cache-hit path must count the completed request, got:\n{rendered}"
        );
        for name in [
            metrics::TOKENS_TOTAL,
            metrics::COST_TOTAL,
            ROUTER_DECISIONS_TOTAL,
        ] {
            assert!(
                !rendered.lines().any(|line| line.starts_with(name)
                    && line.contains("model=\"cache-hit-metric-model\"")),
                "cache-hit path must not increment {name}, got:\n{rendered}"
            );
        }
    }

    /// The router-decision counter increments once per stamped request on
    /// both post_success and post_failure, labelled by strategy + requested
    /// model; an unstamped request (no pick) records nothing.
    #[tokio::test]
    #[serial_test::serial]
    async fn router_decision_counter_increments_on_success_and_failure() {
        // Install the Prometheus recorder BEFORE writing — macro writes go
        // to a no-op recorder until the first `render()`/`install()`.
        let _ = metrics::render();
        let hook = MetricsHook::new();
        let mut ctx = CallContext::for_test("router-metric-model");
        ctx.router_strategy = Some(oxllm_core::RoutingStrategyKind::SimpleShuffle);
        let resp = make_response("router-metric-model", 10, 5);
        hook.post_success(&ctx, &resp).await;
        hook.post_failure(&ctx, &oxllm_core::Error::Network("boom".into()))
            .await;
        let rendered = metrics::render();
        assert!(
            rendered.contains(
                "oxllm_router_decisions_total{strategy=\"simple_shuffle\",model=\"router-metric-model\"} 2"
            ),
            "one success + one failure must count 2 decisions, got:\n{rendered}"
        );

        // Unstamped (no pick — e.g. a pre-routing rejection): no router
        // decision increment. (The request itself is still recorded under
        // `oxllm_requests_total{...,status="error"}` — only the router
        // decision counter must stay absent for this model.) Walk the
        // rendered exposition line by line so the check is scoped to the
        // router-decisions metric, not every metric that mentions the model.
        let hook = MetricsHook::new();
        let ctx = CallContext::for_test("router-metric-unstamped");
        hook.post_failure(&ctx, &oxllm_core::Error::Network("boom".into()))
            .await;
        let rendered = metrics::render();
        let unstamped_decision = rendered.lines().any(|line| {
            line.starts_with("oxllm_router_decisions_total{")
                && line.contains("model=\"router-metric-unstamped\"")
        });
        assert!(
            !unstamped_decision,
            "an unstamped request must not record a router decision, got:\n{rendered}"
        );
    }

    /// MEDIUM #4: the requested-model label on `oxllm_router_decisions_total`
    /// must be length-capped as well as cardinality-bounded, matching every
    /// other model-labelled metric.
    #[tokio::test]
    #[serial_test::serial]
    async fn router_decision_truncates_long_requested_model_label() {
        let _ = metrics::render();
        let hook = MetricsHook::new();
        let long_model = "r".repeat(64 * 3 + 8);
        let mut ctx = CallContext::for_test(&long_model);
        ctx.router_strategy = Some(oxllm_core::RoutingStrategyKind::SimpleShuffle);
        hook.post_success(&ctx, &make_response(&long_model, 1, 1))
            .await;
        let rendered = metrics::render();
        assert!(
            !rendered.contains(&format!("model=\"{long_model}\"")),
            "untruncated requested model leaked into router_decisions label"
        );
        assert!(
            rendered.contains(&format!("model=\"{}\"", "r".repeat(64))),
            "expected a model label truncated to 64 chars in:\n{rendered}"
        );
    }

    /// An unknown-model rejection must count under the fixed `unknown`
    /// placeholder, never under the raw requested name — otherwise arbitrary
    /// client input grows the registry one series per distinct name.
    #[tokio::test]
    #[serial_test::serial]
    async fn post_failure_records_unknown_model_under_placeholder() {
        let _ = metrics::render();
        let hook = MetricsHook::new();
        let ctx = CallContext::for_test("no-such-model-xyz");
        hook.post_failure(
            &ctx,
            &oxllm_core::Error::ModelNotFound("no-such-model-xyz".into()),
        )
        .await;
        let rendered = metrics::render();
        assert!(
            rendered
                .contains("oxllm_requests_total{model=\"unknown\",provider=\"\",status=\"error\"}"),
            "unknown-model rejection must count under the placeholder, got:\n{rendered}"
        );
        assert!(
            !rendered.contains("no-such-model-xyz"),
            "raw unknown model name must not leak into any metric, got:\n{rendered}"
        );
    }

    /// MEDIUM #5: the failure path must record the *requested* model so
    /// per-model error rates (the most common alerting use case) are
    /// observable. Provider is unknown on the failure path and left empty.
    #[tokio::test]
    #[serial_test::serial]
    async fn post_failure_records_model_label() {
        let _ = metrics::render();
        let hook = MetricsHook::new();
        let ctx = CallContext::for_test("gpt-4o");
        hook.post_failure(&ctx, &oxllm_core::Error::Network("boom".into()))
            .await;
        let rendered = metrics::render();
        assert!(
            rendered
                .contains("oxllm_requests_total{model=\"gpt-4o\",provider=\"\",status=\"error\"}"),
            "post_failure must record the requested model under the error series, got:\n{rendered}"
        );
    }

    /// The bounded model label applies in both directions: an over-long
    /// serving model on the success path and an over-long requested model on
    /// the failure path must both be truncated, never rendered raw.
    #[tokio::test]
    #[serial_test::serial]
    async fn success_and_failure_request_labels_are_length_bounded() {
        let _ = metrics::render();
        let hook = MetricsHook::new();
        let long_success = "h".repeat(64 * 3 + 8);
        hook.post_success(
            &CallContext::for_test("ctx-model"),
            &make_response(&long_success, 1, 1),
        )
        .await;

        let long_failure = "g".repeat(64 * 3 + 8);
        let ctx = CallContext::for_test(&long_failure);
        hook.post_failure(&ctx, &oxllm_core::Error::Network("boom".into()))
            .await;

        let rendered = metrics::render();
        assert!(
            !rendered.contains(&long_success) && !rendered.contains(&long_failure),
            "raw over-long models must not appear in any metric, got:\n{rendered}"
        );
        assert!(
            rendered.contains(&format!("model=\"{}\"", "h".repeat(64))),
            "success-path model label must be truncated to 64 chars, got:\n{rendered}"
        );
        assert!(
            rendered.contains(&format!("model=\"{}\"", "g".repeat(64))),
            "failure-path model label must be truncated to 64 chars, got:\n{rendered}"
        );
    }
}
