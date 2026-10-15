//! Chat orchestration: retry loops, fallback chains, streaming, and cache
//! short-circuit for `POST /v1/chat/completions`.
//!
//! The handler in [`crate::handlers::chat`] is responsible only for parsing
//! the request, checking JSON depth, building the [`EndpointPipeline`], and
//! mapping the final [`Response`] to an HTTP response. Everything else
//! cache-hit short-circuit, the retry/fallback loop, the upstream HTTP call,
//! the SSE pump, and error classification — lives in this module.
//!
//! Layout:
//! - `outcomes`: `ModelOutcome`, `UpstreamError`, error classifiers.
//! - `usage`: token-usage extraction shared by stream/non-stream/cache paths.
//! - `upstream`: `attempt_upstream` (the outbound HTTP call) + `drop_params`.
//! - `stream`: `handle_stream_success` (the SSE pump).
//! - `chat`: `execute_request_chain` (retry + fallback orchestration).

pub(crate) mod budget;
pub(crate) mod chat;
mod lifecycle;
pub(crate) mod outcomes;
pub(crate) mod request_controls;
pub(crate) mod stream;
mod stream_start;
pub(crate) mod unary;
pub(crate) mod upstream;
pub(crate) mod usage;

use std::sync::{Arc, OnceLock};

use axum::Json;
use axum::response::{IntoResponse, Response};
use oxllm_core::CallContext;
use oxllm_core::deployment::Deployment;
use serde_json::Value;

use crate::auth::AuthUser;
use crate::error::ApiError;
use crate::orchestration::chat::execute_request_chain;
use crate::orchestration::upstream::{attempt_upstream, resolve_provider};
use crate::orchestration::usage::extract_usage;
use crate::pipeline::EndpointPipeline;
use crate::pipeline::{
    CACHE_HIT_HEADER, apply_cost_header, apply_ratelimit_headers, set_header,
    stamp_ratelimit_headers, stamp_request_id_header,
};
use crate::state::AppState;

/// Default request timeout if none configured.
pub(crate) const DEFAULT_TIMEOUT_SECS: u64 = 600;

const SILENT_MODEL_CONCURRENCY: usize = 16;
static SILENT_MODEL_SLOTS: OnceLock<Arc<tokio::sync::Semaphore>> = OnceLock::new();

/// Maximum bytes accumulated in the SSE stream buffer before a best-effort
/// flush. Guards against unbounded growth if the upstream sends a very long
/// line without an `\n\n` event boundary.
pub const MAX_STREAM_BUFFER_BYTES: usize = 4 * 1024 * 1024;

/// Default cooldown header name used when a fallback model serves the request.
pub(crate) const FALLBACK_HEADER: &str = "x-oxllm-fallback-used";

/// Header carrying the routing strategy that picked the serving deployment
/// (`RoutingStrategyKind::as_str`, e.g. `"simple_shuffle"`).
pub(crate) const ROUTER_STRATEGY_HEADER: &str = "x-oxllm-router-strategy";

/// Header carrying how the serving deployment was reached: `"picked"`
/// (strategy pick), `"fallback"` (fallback chain served),
/// `"default_fallback"` (unknown-model reroute), or `"prompt_cache_pin"`
/// (prompt-caching pre-call check pinned the pick).
pub(crate) const ROUTER_DECISION_HEADER: &str = "x-oxllm-router-decision";

/// Stamp the router observability headers from the pick-time stashes on the
/// call context. No-ops when no pick happened (e.g. a cache short-circuit
/// or a pre-routing rejection never engaged the router).
///
/// # Coverage
///
/// Emitted on every response that returns `Ok` from
/// [`ChatOrchestrator::run`] — streaming and non-streaming alike (stream
/// headers are set at stream start), including fallback-exhausted
/// responses whose body carries an upstream error but which the fallback
/// chain returned as an `Ok` error-response. Responses on the hard-`Err`
/// path (the chain returns `Err` and `run` bails before stamping, e.g. a
/// terminal upstream failure with no fallbacks configured, or a
/// pre-routing rejection) do NOT carry these headers. Consumers must treat
/// header absence as "no routing decision completed", never as a routing
/// outcome in itself.
pub(crate) fn apply_router_headers(response: &mut Response, ctx: &CallContext) {
    crate::pipeline::stamp_deployment_headers(response, ctx);
    if let Some(strategy) = ctx.router_strategy {
        set_header(response, ROUTER_STRATEGY_HEADER, strategy.as_str());
    }
    if let Some(decision) = ctx.router_decision.as_deref() {
        set_header(response, ROUTER_DECISION_HEADER, decision);
    }
}

/// H3: Traffic mirroring (shadow traffic). After the primary upstream
/// completes successfully, fork a fire-and-forget `tokio::spawn` task that
/// sends the same request body to the deployment's `silent_model`. The
/// silent model's response is logged but never returned to the client, and
/// its spend is **not** charged to the caller's budget (the spawned task
/// bypasses the post-success hook chain — no `post_success` runs, so
/// `BudgetHook`/`CostTrackingHook` never debit).
///
/// Failures (provider resolution, upstream error) are logged at `warn`
/// level and swallowed. The task is never awaited by the caller.
/// At most 16 shadow tasks may be outstanding per process (also across
/// config reloads). Excess work is skipped before cloning the request and
/// counted by `oxllm_silent_model_skipped_total`; it is never queued.
/// Shadow calls use non-streaming bodies and retain their slot until the
/// response is drained, within the normal response-size/request-time bounds.
///
/// No-op when `deployment.oxllm_params.silent_model` is `None`.
pub(crate) fn spawn_silent_model_call(
    state: &AppState,
    request_body: &Value,
    deployment: &Deployment,
    traceparent: Option<&str>,
) {
    if deployment.oxllm_params.silent_model.is_none() {
        return;
    }
    let slots = SILENT_MODEL_SLOTS
        .get_or_init(|| Arc::new(tokio::sync::Semaphore::new(SILENT_MODEL_CONCURRENCY)))
        .clone();
    let _ = spawn_silent_model_call_limited(state, request_body, deployment, traceparent, slots);
}

fn spawn_silent_model_call_limited(
    state: &AppState,
    request_body: &Value,
    deployment: &Deployment,
    traceparent: Option<&str>,
    slots: Arc<tokio::sync::Semaphore>,
) -> Option<tokio::task::JoinHandle<()>> {
    let silent_model = deployment.oxllm_params.silent_model.as_ref()?;
    let permit = match slots.try_acquire_owned() {
        Ok(permit) => permit,
        Err(_) => {
            crate::metrics::record_silent_model_skipped();
            return None;
        }
    };
    let silent_model = silent_model.clone();
    // Build a synthetic deployment for the silent model. It reuses the
    // primary deployment's credential/base/timeout so the operator only
    // has to point at a model string (e.g. `"openai/gpt-4o"`) to mirror
    // traffic on the same account. Other fields are left at their
    // defaults — the silent call must not consume the primary's RPM/TPM
    // budget or carry its pricing override (spend is intentionally
    // untracked).
    let silent_dep = Deployment {
        model_name: silent_model.clone(),
        oxllm_params: oxllm_core::deployment::OxllmParams {
            model: silent_model.clone(),
            api_key: deployment.oxllm_params.api_key.clone(),
            api_base: deployment.oxllm_params.api_base.clone(),
            api_version: deployment.oxllm_params.api_version.clone(),
            aws_access_key_id: deployment.oxllm_params.aws_access_key_id.clone(),
            aws_secret_access_key: deployment.oxllm_params.aws_secret_access_key.clone(),
            aws_session_token: deployment.oxllm_params.aws_session_token.clone(),
            timeout: deployment.oxllm_params.timeout,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut body = request_body.clone();
    if let Some(m) = body.as_object_mut() {
        m.insert("model".to_string(), Value::String(silent_model.clone()));
        // `is_stream=false` controls the HTTP path, not the provider payload.
        // Leaving the caller's true flag here would release the shadow slot
        // at SSE headers while the upstream was still generating output.
        m.insert("stream".to_string(), Value::Bool(false));
        m.remove("stream_options");
    }
    let state = state.clone();
    let traceparent = traceparent.map(str::to_string);
    // Capture the primary deployment's provider so the spawned task can guard
    // against a misconfigured `silent_model` that resolves to a DIFFERENT
    // provider. The synthetic deployment reuses the primary's credentials; a
    // mismatched provider would send those credentials to an unrelated vendor
    // (a silent credential exfiltration and an inevitable 401). The doc
    // contract is "point at a model string to mirror on the same account", so
    // a provider mismatch is enforced as a hard abort here rather than a warn
    // — closing the leak. Cross-provider mirroring would require the synthetic
    // deployment to carry its own credentials (a separate deployment entry or
    // an explicit `silent_api_key`), never the primary's.
    let primary_provider = deployment.provider_slug().to_string();
    Some(tokio::spawn(async move {
        let _permit = permit;
        let provider = match resolve_provider(&state, &silent_dep) {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!(
                    error = ?e,
                    silent_model = %silent_model,
                    "silent model shadow traffic: provider resolution failed"
                );
                return;
            }
        };
        if provider.id() != primary_provider {
            tracing::warn!(
                silent_model = %silent_model,
                silent_provider = %provider.id(),
                primary_provider = %primary_provider,
                "silent model resolves to a different provider than the primary; \
                 aborting shadow traffic to avoid sending the primary deployment's \
                 credentials to an unrelated upstream (likely a misconfiguration)"
            );
            return;
        }
        match attempt_upstream(
            &state,
            &silent_dep,
            &body,
            false,
            &provider,
            traceparent.as_deref(),
        )
        .await
        {
            Ok(resp) => {
                let status = resp.status().as_u16();
                match discard_silent_model_body(resp).await {
                    Ok(()) => tracing::debug!(
                        silent_model = %silent_model,
                        status,
                        "silent model shadow traffic completed"
                    ),
                    Err(e) => tracing::warn!(
                        error = ?e,
                        silent_model = %silent_model,
                        "silent model shadow response failed"
                    ),
                }
            }
            Err(e) => {
                tracing::warn!(
                    error = ?e,
                    silent_model = %silent_model,
                    "silent model shadow traffic failed"
                );
            }
        }
    }))
}

/// Consume the optional response without retaining its contents. Reqwest's
/// non-streaming request deadline remains active during body reads. Checking
/// both Content-Length and accumulated chunks bounds dishonest/chunked peers.
async fn discard_silent_model_body(mut response: reqwest::Response) -> Result<(), ApiError> {
    let cap = crate::security::MAX_RESPONSE_BODY_BYTES;
    crate::security::check_content_length(response.headers(), cap)?;
    let mut received = 0usize;
    while let Some(chunk) = response.chunk().await.map_err(|error| {
        tracing::warn!(%error, "failed to drain silent model response");
        ApiError::internal("failed to drain silent model response")
    })? {
        received = received.saturating_add(chunk.len());
        if received > cap {
            return Err(ApiError::internal(
                "silent model response exceeds size limit",
            ));
        }
    }
    Ok(())
}

/// The chat orchestrator: ties together cache lookup, the retry/fallback
/// chain, and the streaming/non-streaming response handlers.
///
/// Constructed by [`crate::handlers::chat`] once the request body has been
/// parsed and the [`EndpointPipeline`] built. [`ChatOrchestrator::run`] then
/// drives the full request lifecycle and returns the final [`Response`].
pub(crate) struct ChatOrchestrator<'a> {
    state: &'a Arc<AppState>,
    pipeline: EndpointPipeline<'a>,
    auth: &'a AuthUser,
}

impl<'a> ChatOrchestrator<'a> {
    /// Build an orchestrator over a pre-constructed pipeline.
    pub(crate) fn new(
        state: &'a Arc<AppState>,
        pipeline: EndpointPipeline<'a>,
        auth: &'a AuthUser,
    ) -> Self {
        Self {
            state,
            pipeline,
            auth,
        }
    }

    /// Run the full request lifecycle: cache lookup (short-circuit),
    /// retry/fallback chain, and request-id stamping.
    ///
    /// `session_id` is the normalized sticky-routing identity from headers or
    /// body metadata (`routing_strategy: session_affinity`). `tag` is the
    /// normalized primary request tag (`routing_strategy: tag_based`).
    pub(crate) async fn run(
        mut self,
        model: &str,
        request_body: &Value,
        is_stream: bool,
        traceparent: Option<&str>,
        session_id: Option<String>,
        tag: Option<String>,
    ) -> Result<Response, ApiError> {
        if let Some(cached) = self.pipeline.pre_call().await? {
            tracing::info!(request_id = %self.pipeline.call_ctx.request_id, "response cache hit");
            self.pipeline.call_ctx.cache_hit = true;
            let request_id = self.pipeline.request_id().to_string();
            let mut resp = build_cache_hit_response(
                &cached,
                model,
                &request_id,
                is_stream,
                &self.pipeline.call_ctx,
            );
            let cached_model = cached
                .get("model")
                .and_then(|v| v.as_str())
                .unwrap_or(model)
                .to_string();
            let usage = extract_usage(&cached);
            // Thread the cached response's provider prefix (when present) into
            // the post-success chain so the cost hooks and spend log attribute
            // spend to the serving provider instead of the empty default —
            // matching the non-cache path. An empty prefix preserves the
            // legacy provider-agnostic behaviour.
            let cached_provider = provider_prefix(&cached_model);
            self.pipeline
                .post_success(
                    &cached_model,
                    usage.as_ref(),
                    Some(&cached),
                    cached_provider,
                )
                .await;
            crate::pipeline::stamp_key_spend_header(&mut resp, self.state, &self.pipeline.call_ctx)
                .await;
            return Ok(resp);
        }

        let request_id = self.pipeline.request_id().to_string();
        let mut ctx = oxllm_core::RequestContext::new(model);
        ctx.session_id = session_id;
        ctx.tag = tag.or_else(|| self.pipeline.call_ctx.request_tags.first().cloned());
        ctx.tags = self.pipeline.call_ctx.request_tags.clone();
        // Request-text extraction is cheap (concatenated message contents,
        // capped) but skipped entirely unless a text-classifying strategy
        // is in play for this model. Both text classifiers run HERE, once
        // per request: the complexity tier is stamped onto
        // `PreCallData::complexity_classification` (read by the tier metric
        // and reused by the core strategy pick), and for the adaptive
        // router the request type is stamped because the host needs it to
        // stamp the bandit cell for the learning hook after each pick —
        // the router then reuses it via `PreCallData::adaptive_request_type`
        // instead of classifying a second time.
        match self.state.router.strategy_kind_for(model) {
            oxllm_core::RoutingStrategyKind::ComplexityRouter => {
                ctx.request_text = Some(oxllm_core::router::complexity::extract_request_text(
                    request_body,
                ));
                // Classify ONCE per request and stamp the context: the
                // tier metric in `execute_request_chain` reads the stamp,
                // and the core's `ComplexityRouter::pick` reuses
                // `pre_call.complexity_classification` when present
                // (batch-3 mechanism) instead of classifying again.
                // Fallback legs inherit the stamp through the cloned
                // request contexts.
                if let Some(config) = self.state.router.config().complexity_router_config.as_ref() {
                    let classification = oxllm_core::router::complexity::classify(&ctx, config);
                    ctx.pre_call.complexity_classification = Some(classification);
                }
            }
            oxllm_core::RoutingStrategyKind::AdaptiveRouter => {
                let text = oxllm_core::router::complexity::extract_request_text(request_body);
                ctx.pre_call.adaptive_request_type =
                    Some(oxllm_core::router::adaptive::classify_request_type(&text));
                ctx.request_text = Some(text);
            }
            _ => {}
        }
        // Fallback legs may resolve to an adaptive routing group even when
        // the primary model's strategy is not adaptive (adaptive configured
        // for ANY group). Classify once here — the legs inherit this
        // classification through the cloned request contexts instead of
        // re-classifying per leg.
        if ctx.pre_call.adaptive_request_type.is_none() && self.state.router.adaptive_active() {
            let text = ctx.request_text.clone().unwrap_or_else(|| {
                oxllm_core::router::complexity::extract_request_text(request_body)
            });
            ctx.pre_call.adaptive_request_type =
                Some(oxllm_core::router::adaptive::classify_request_type(&text));
            ctx.request_text = Some(text);
        }
        let mut response = match execute_request_chain(
            self.state,
            &mut self.pipeline.call_ctx,
            self.auth,
            model,
            request_body,
            &ctx,
            is_stream,
            traceparent,
        )
        .await
        {
            Ok(response) => response,
            Err(err) => {
                self.pipeline
                    .post_failure(&crate::pipeline::core_error_from_api(&err))
                    .await;
                return Err(err);
            }
        };

        stamp_request_id_header(&mut response, &request_id);
        // Stamp the full `x-ratelimit-*` header set (limit + remaining + reset)
        // from the live per-key cache counters after pre_call reserved the
        // slots. No-op for master-key requests and cache hits.
        stamp_ratelimit_headers(&mut response, self.pipeline.state, &self.pipeline.call_ctx).await;
        // Router observability: strategy + decision stashed at pick time.
        // Stamped here (after the chain returns) so streaming responses
        // carry them at stream start exactly like non-streaming ones.
        apply_router_headers(&mut response, &self.pipeline.call_ctx);
        if !is_stream {
            crate::pipeline::stamp_key_spend_header(
                &mut response,
                self.state,
                &self.pipeline.call_ctx,
            )
            .await;
        }

        Ok(response)
    }
}

/// Run the post-failure hook chain for an orchestrator [`ApiError`]. All
/// orchestrators (chat chain legs, unary executor) funnel through here so
/// the `ApiError → oxllm_core::Error` mapping the rate-limit hook keys off
/// (release vs keep RPM/TPM) is resolved in exactly one place
/// ([`crate::pipeline::core_error_from_api`]).
pub(crate) async fn pipeline_failure(
    state: &AppState,
    call_ctx: &oxllm_core::CallContext,
    err: &ApiError,
) {
    crate::pipeline::post_failure(state, call_ctx, &crate::pipeline::core_error_from_api(err))
        .await;
}

/// Build the response for a cache short-circuit. When `stream` is true the
/// cached body is wrapped in the SSE shape (`data: <body>\n\n` followed by
/// `data: [DONE]\n\n`) so clients expecting `text/event-stream` receive the
/// correct content type. Otherwise a plain JSON response is returned.
pub(crate) fn build_cache_hit_response(
    cached: &Value,
    default_model: &str,
    request_id: &str,
    stream: bool,
    ctx: &CallContext,
) -> Response {
    let mut public = cached.clone();
    crate::pipeline::normalize_client_response(
        &mut public,
        crate::pipeline::public_response_model(ctx),
    );
    let mut resp = if stream {
        let body = format!("data: {public}\n\ndata: [DONE]\n\n");
        axum::response::Response::builder()
            .status(axum::http::StatusCode::OK)
            .header(axum::http::header::CONTENT_TYPE, "text/event-stream")
            .body(axum::body::Body::from(body))
            .unwrap_or_else(|_| ApiError::internal("failed to build sse body").into_response())
    } else {
        Json(public).into_response()
    };
    set_cost_header(&mut resp, cached, default_model, ctx);
    set_header(&mut resp, CACHE_HIT_HEADER, "true");
    stamp_request_id_header(&mut resp, request_id);
    apply_ratelimit_headers(&mut resp, &ctx.ratelimit_info, ctx.is_master);
    apply_router_headers(&mut resp, ctx);
    resp
}

/// Derive a provider slug from a model string that carries a `provider/`
/// prefix (mirroring [`Deployment::provider_slug`](oxllm_core::deployment::Deployment::provider_slug)),
/// returning an empty string when no prefix is present. The cache short-circuit
/// runs before any deployment has been picked, so the cached body's `model`
/// field is the only available hint about which provider originally served the
/// response. An empty return preserves the legacy provider-agnostic cost lookup
/// rather than guessing a vendor for a bare model name.
fn provider_prefix(model: &str) -> &str {
    match model.split_once('/') {
        Some((provider, _)) if !provider.is_empty() => provider,
        _ => "",
    }
}

/// Compute cost from a response body and stamp the cost header. Uses the
/// provider-aware cost lookup when `provider` is non-empty so a cached response
/// whose `model` carries a `provider/` prefix resolves to the same price row as
/// the non-cache path (consistency with [`pipeline::post_success`]).
fn set_cost_header(response: &mut Response, body: &Value, default_model: &str, ctx: &CallContext) {
    let model = body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or(default_model);
    let provider = provider_prefix(model);
    let usage = extract_usage(body);
    let cost = ctx.cached_response_cost.or_else(|| {
        crate::hooks::cost_fallback::resolve_cost(
            ctx,
            model,
            usage.as_ref(),
            Some(body),
            None,
            provider,
        )
    });
    apply_cost_header(response, cost);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::credentials::CredentialResolver;
    use crate::pipeline::{check_deployment_limits, record_deployment_tpm};
    use oxllm_core::deployment::Deployment;
    use serde_json::json;

    #[test]
    fn build_cache_hit_response_non_stream_returns_json() {
        let cached = json!({"id": "x", "choices": []});
        let resp = build_cache_hit_response(&cached, "gpt-4o", "req-1", false, &dummy_ctx());
        assert_ne!(
            resp.headers()
                .get(axum::http::header::CONTENT_TYPE)
                .map(|v| v.to_str().unwrap_or("")),
            Some("text/event-stream")
        );
        assert_eq!(
            resp.headers()
                .get(CACHE_HIT_HEADER)
                .and_then(|v| v.to_str().ok()),
            Some("true")
        );
    }

    #[test]
    fn build_cache_hit_response_stream_returns_sse() {
        let cached = json!({"id": "x", "choices": []});
        let resp = build_cache_hit_response(&cached, "gpt-4o", "req-1", true, &dummy_ctx());
        assert_eq!(
            resp.headers()
                .get(axum::http::header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok()),
            Some("text/event-stream")
        );
        assert_eq!(
            resp.headers()
                .get(CACHE_HIT_HEADER)
                .and_then(|v| v.to_str().ok()),
            Some("true")
        );
    }

    #[test]
    fn provider_prefix_extracts_prefix_segment() {
        assert_eq!(provider_prefix("anthropic/claude-3-5-sonnet"), "anthropic");
        assert_eq!(provider_prefix("openai/gpt-4o"), "openai");
        // Bare model → empty (preserve provider-agnostic fallback).
        assert_eq!(provider_prefix("gpt-4o"), "");
        // Leading slash → empty prefix segment.
        assert_eq!(provider_prefix("/gpt-4o"), "");
    }

    /// MEDIUM (C11) follow-up (LOW): the single `ApiError → Error` mapping
    /// (`pipeline::core_error_from_api`, used by every orchestrator via
    /// `pipeline_failure`) must send a gateway-synthesized 429 — e.g. the
    /// pick-loop-exhaustion error `all deployments are rate-limited or
    /// unavailable` — to `Error::RateLimited` so the rate-limit hook
    /// RELEASES the key's RPM/TPM reservations (no provider was contacted);
    /// a provider-served 429 (`upstream_rate_limited` code) stays
    /// `Error::Upstream` so the reservation is kept.
    #[tokio::test]
    async fn pick_loop_exhaustion_429_releases_key_rpm_reservation() {
        let state = state_with_in_memory_cache();
        let ctx = CallContext {
            model: "gpt-4o".to_string(),
            api_key_hash: "keyhash".to_string(),
            request_id: "r".to_string(),
            rpm: Some(10),
            is_master: false,
            ..Default::default()
        };
        // Simulate the pre_call RPM reservation on the key counter.
        let rpm_key = "rpm:keyhash".to_string();
        let _ = state
            .cache
            .incr(&rpm_key, 1.0, Some(std::time::Duration::from_secs(60)))
            .await;

        // The exact error `try_model_with_retries` synthesizes when the pick
        // loop finds no reachable deployment.
        let err =
            ApiError::rate_limited("all deployments for 'gpt-4o' are rate-limited or unavailable");
        pipeline_failure(&state, &ctx, &err).await;

        let rpm = state.cache.counter_value(&rpm_key).await.unwrap_or(0.0);
        assert!(
            rpm.abs() < 1e-9,
            "the synthesized pick-loop 429 never reached a provider; the key's RPM reservation must be released, got {rpm}"
        );

        // Contrast: a provider-served 429 keeps the reservation (the
        // provider processed the request; an immediate local retry could
        // otherwise exceed the provider's RPM ceiling).
        let _ = state
            .cache
            .incr(&rpm_key, 1.0, Some(std::time::Duration::from_secs(60)))
            .await;
        let provider_429 = ApiError::upstream(429, "provider throttled us");
        pipeline_failure(&state, &ctx, &provider_429).await;
        let rpm = state.cache.counter_value(&rpm_key).await.unwrap_or(0.0);
        assert!(
            (rpm - 1.0).abs() < 1e-9,
            "provider-served 429 must keep the reservation, got {rpm}"
        );
    }

    /// MEDIUM (C11): a cache-hit response whose cached body carries a
    /// `provider/`-prefixed model must compute its cost header through the
    /// provider-aware lookup (matching the non-cache path) rather than the
    /// legacy provider-agnostic one. Asserts the header is present and equals
    /// `calculate_cost_with_override_for_provider(.., provider)`.
    #[test]
    fn build_cache_hit_response_cost_header_uses_provider_prefix() {
        use crate::pipeline::COST_HEADER;
        let cached = json!({
            "id": "x",
            "model": "openai/gpt-4o",
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        });
        let ctx = dummy_ctx();
        let resp = build_cache_hit_response(&cached, "gpt-4o", "req-1", false, &ctx);
        let expected = extract_usage(&cached).and_then(|u| {
            oxllm_core::calculate_cost_with_override_for_provider(
                "openai/gpt-4o",
                &u,
                ctx.deployment_pricing.as_ref(),
                "openai",
            )
        });
        let actual = resp
            .headers()
            .get(COST_HEADER)
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<f64>().ok());
        assert!(
            expected.unwrap_or(0.0) > 0.0,
            "fixture should resolve a price"
        );
        assert!(
            (actual.unwrap_or(-1.0) - expected.unwrap_or(0.0)).abs() < 1e-12,
            "cost header must match provider-aware computation"
        );
    }

    fn dummy_ctx() -> CallContext {
        CallContext {
            model: "gpt-4o".to_string(),
            api_key_hash: "h".to_string(),
            request_id: "r".to_string(),
            ..Default::default()
        }
    }

    fn deployment_with_limits(rpm: Option<u32>, tpm: Option<u32>) -> Deployment {
        Deployment {
            model_name: "gpt-4o".to_string(),
            oxllm_params: oxllm_core::deployment::OxllmParams {
                model: "openai/gpt-4o".to_string(),
                api_key: Some("k".to_string()),
                rpm,
                tpm,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn state_with_in_memory_cache() -> AppState {
        let config = crate::config::types::ProxyConfig {
            model_list: vec![],
            oxllm_settings: crate::config::types::OxllmSettings::default(),
            general_settings: crate::config::types::GeneralSettings::default(),
            router_settings: Default::default(),
            extra: serde_json::Map::new(),
        };
        AppState::with_registry(config, oxllm_providers::default_registry()).expect("state built")
    }

    #[tokio::test]
    async fn check_deployment_limits_no_limits_passes() {
        let state = state_with_in_memory_cache();
        let dep = deployment_with_limits(None, None);
        let result = check_deployment_limits(&state, &dep, "gpt-4o:openai/gpt-4o").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn check_deployment_limits_rpm_under_limit_passes() {
        let state = state_with_in_memory_cache();
        let dep = deployment_with_limits(Some(10), None);
        let _ = state
            .cache
            .incr("dep_rpm:gpt-4o:openai/gpt-4o", 5.0, None)
            .await;
        let result = check_deployment_limits(&state, &dep, "gpt-4o:openai/gpt-4o").await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn check_deployment_limits_rpm_at_limit_rejects() {
        let state = state_with_in_memory_cache();
        let dep = deployment_with_limits(Some(10), None);
        let _ = state
            .cache
            .incr("dep_rpm:gpt-4o:openai/gpt-4o", 10.0, None)
            .await;
        let result = check_deployment_limits(&state, &dep, "gpt-4o:openai/gpt-4o").await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.status, axum::http::StatusCode::TOO_MANY_REQUESTS);
    }

    #[tokio::test]
    async fn check_deployment_limits_tpm_at_limit_rejects() {
        let state = state_with_in_memory_cache();
        let dep = deployment_with_limits(None, Some(1000));
        let _ = state
            .cache
            .incr("dep_tpm:gpt-4o:openai/gpt-4o", 1000.0, None)
            .await;
        let result = check_deployment_limits(&state, &dep, "gpt-4o:openai/gpt-4o").await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.status, axum::http::StatusCode::TOO_MANY_REQUESTS);
    }

    #[tokio::test]
    async fn record_deployment_tpm_increments_counter() {
        let state = state_with_in_memory_cache();
        record_deployment_tpm(&state, "gpt-4o:openai/gpt-4o", 500).await;
        let val = state
            .cache
            .get("dep_tpm:gpt-4o:openai/gpt-4o")
            .await
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(0.0);
        assert!((val - 500.0).abs() < 1e-9);

        record_deployment_tpm(&state, "gpt-4o:openai/gpt-4o", 100).await;
        let val = state
            .cache
            .get("dep_tpm:gpt-4o:openai/gpt-4o")
            .await
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(0.0);
        assert!((val - 600.0).abs() < 1e-9);
    }

    #[tokio::test]
    async fn record_deployment_tpm_zero_is_noop() {
        let state = state_with_in_memory_cache();
        record_deployment_tpm(&state, "gpt-4o:openai/gpt-4o", 0).await;
        let present = state.cache.get("dep_tpm:gpt-4o:openai/gpt-4o").await;
        assert!(present.is_none());
    }

    #[tokio::test]
    async fn static_key_resolver_returns_resolved_key() {
        let dep = Deployment {
            model_name: "claude".to_string(),
            oxllm_params: oxllm_core::deployment::OxllmParams {
                model: "anthropic/claude-3-5-sonnet".to_string(),
                api_key: Some("k".to_string()),
                ..Default::default()
            },
            ..Default::default()
        };
        let resolver = crate::credentials::StaticKeyResolver;
        let creds = resolver.resolve(&dep).await.expect("ok");
        match creds {
            oxllm_core::provider::ResolvedCredentials::ApiKey { bearer } => {
                assert_eq!(bearer, "k")
            }
            other => panic!("expected ApiKey, got {other:?}"),
        }
        let _ = oxllm_core::provider::ResolvedCredentials::from_key("x");
    }

    // ---------------------------------------------------------------
    // H3: traffic mirroring / silent model
    // ---------------------------------------------------------------

    #[tokio::test]
    #[serial_test::serial]
    async fn silent_model_limit_skips_without_queueing_and_releases_after_completion() {
        let _ = crate::metrics::install();
        let server = httpmock::MockServer::start();
        let upstream = server.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/chat/completions")
                .json_body_partial(r#"{"stream":false}"#);
            then.status(200).json_body(json!({"choices": []}));
        });
        let config = crate::config::types::ProxyConfig {
            model_list: vec![],
            oxllm_settings: Default::default(),
            general_settings: crate::config::types::GeneralSettings {
                allow_private_api_base: Some(true),
                ..Default::default()
            },
            router_settings: Default::default(),
            extra: serde_json::Map::new(),
        };
        let state = AppState::with_registry(config, oxllm_providers::default_registry()).unwrap();
        let dep = Deployment {
            model_name: "shadow-limit-probe".into(),
            oxllm_params: oxllm_core::deployment::OxllmParams {
                model: "openai/gpt-4o".into(),
                api_key: Some("sk-test".into()),
                api_base: Some(server.base_url()),
                silent_model: Some("openai/gpt-4o-mini".into()),
                timeout: Some(2.0),
                ..Default::default()
            },
            ..Default::default()
        };
        let slots = Arc::new(tokio::sync::Semaphore::new(1));
        let held = slots.clone().try_acquire_owned().unwrap();
        let body = json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true,
            "stream_options": {"include_usage": true}
        });
        let skipped_before = shadow_skip_count();
        assert!(
            spawn_silent_model_call_limited(&state, &body, &dep, None, slots.clone()).is_none()
        );
        assert_eq!(shadow_skip_count(), skipped_before + 1);
        upstream.assert_hits(0);

        drop(held);
        let job = spawn_silent_model_call_limited(&state, &body, &dep, None, slots.clone())
            .expect("a free slot must admit the shadow call");
        tokio::time::timeout(std::time::Duration::from_secs(5), job)
            .await
            .expect("shadow request must complete")
            .unwrap();
        upstream.assert_hits(1);
        assert_eq!(slots.available_permits(), 1);
    }

    fn shadow_skip_count() -> u64 {
        crate::metrics::render()
            .lines()
            .find(|line| line.starts_with("oxllm_silent_model_skipped_total{"))
            .and_then(|line| line.split_whitespace().last())
            .map(|value| value.parse().unwrap())
            .unwrap_or(0)
    }

    /// When `silent_model` is unset the helper is a pure no-op: no task is
    /// spawned, nothing happens. This is the default-deployment path.
    #[tokio::test]
    async fn silent_model_call_noop_when_unset() {
        let state = state_with_in_memory_cache();
        let dep = Deployment {
            model_name: "gpt-4o".to_string(),
            oxllm_params: oxllm_core::deployment::OxllmParams {
                model: "openai/gpt-4o".to_string(),
                ..Default::default()
            },
            ..Default::default()
        };
        // Must not panic and must return immediately.
        spawn_silent_model_call(&state, &json!({"model": "gpt-4o"}), &dep, None);
    }

    /// When `silent_model` is set, a task is spawned. With no real upstream
    /// the task logs a warn and exits — the test just verifies no panic and
    /// that the function returns without blocking (fire-and-forget).
    #[tokio::test]
    async fn silent_model_call_spawns_without_blocking_when_set() {
        let state = state_with_in_memory_cache();
        let dep = Deployment {
            model_name: "gpt-4o".to_string(),
            oxllm_params: oxllm_core::deployment::OxllmParams {
                model: "openai/gpt-4o".to_string(),
                api_key: Some("sk-test".to_string()),
                silent_model: Some("openai/gpt-4o-mini".to_string()),
                ..Default::default()
            },
            ..Default::default()
        };
        let start = std::time::Instant::now();
        spawn_silent_model_call(&state, &json!({"model": "gpt-4o"}), &dep, None);
        let elapsed = start.elapsed();
        // Fire-and-forget: the function must return near-instantly (the
        // spawned task does its work in the background). Allow a generous
        // 500ms ceiling for slow CI.
        assert!(
            elapsed.as_millis() < 500,
            "spawn_silent_model_call must not block; took {elapsed:?}"
        );
        // Give the spawned task a moment to complete (it will fail at
        // network/credential resolution and log a warn — that's fine).
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }

    /// HIGH (C11): when the `silent_model` resolves to a DIFFERENT provider
    /// than the primary, the primary's API key must NOT be sent to that
    /// provider — the shadow traffic aborts instead. The synthetic deployment
    /// reuses the primary's credentials, so a cross-provider typo (e.g. an
    /// Anthropic `silent_model` on an OpenAI deployment) would silently
    /// exfiltrate the OpenAI key to Anthropic's endpoint. We stand up a mock
    /// upstream at the primary's `api_base`; with the fix the mock is never
    /// hit (the task aborts on the provider mismatch). Without the fix the
    /// synthetic call would reach the mock carrying `sk-primary`.
    #[tokio::test]
    async fn silent_model_call_aborts_when_provider_mismatches() {
        use crate::config::types::{ConfigModel, GeneralSettings, ProxyConfig};
        use httpmock::MockServer;

        let server = MockServer::start();
        let mock = server.mock(|_when, _then| {
            // No `when` constraints: match ANY request to this upstream. The
            // fix must prevent the shadow traffic from arriving at all, so the
            // mock must count hits regardless of method/path/body.
        });

        let model_list = vec![ConfigModel {
            model_name: "gpt-4o".to_string(),
            oxllm_params: oxllm_core::deployment::OxllmParams {
                model: "openai/gpt-4o".to_string(),
                api_key: Some("sk-primary".to_string()),
                api_base: Some(server.base_url()),
                silent_model: Some("anthropic/claude-3-5-sonnet".to_string()),
                ..Default::default()
            },
            ..Default::default()
        }];
        let config = ProxyConfig {
            model_list,
            oxllm_settings: crate::config::types::OxllmSettings::default(),
            general_settings: GeneralSettings {
                allow_private_api_base: Some(true),
                ..Default::default()
            },
            router_settings: Default::default(),
            extra: serde_json::Map::new(),
        };
        let state = AppState::with_registry(config, oxllm_providers::default_registry())
            .expect("state built");

        let dep = Deployment {
            model_name: "gpt-4o".to_string(),
            oxllm_params: oxllm_core::deployment::OxllmParams {
                model: "openai/gpt-4o".to_string(),
                api_key: Some("sk-primary".to_string()),
                api_base: Some(server.base_url()),
                silent_model: Some("anthropic/claude-3-5-sonnet".to_string()),
                ..Default::default()
            },
            ..Default::default()
        };
        let job = spawn_silent_model_call_limited(
            &state,
            &json!({"model": "gpt-4o"}),
            &dep,
            None,
            Arc::new(tokio::sync::Semaphore::new(1)),
        )
        .expect("shadow check must actually run");
        tokio::time::timeout(std::time::Duration::from_secs(5), job)
            .await
            .expect("provider mismatch must abort promptly")
            .unwrap();
        mock.assert_hits(0);
    }
}
