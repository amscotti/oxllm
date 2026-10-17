//! `POST /v1/embeddings` — the embeddings endpoint.

use std::collections::HashMap;

use axum::Extension;
use axum::Json;
use axum::body::Bytes;
use axum::extract::State;
use axum::response::{IntoResponse, Response};
use oxllm_core::UsageSummary;
use oxllm_core::provider::{
    EmbeddingProvider, EmbeddingRequest, EmbeddingRequestPlan, ResolvedCredentials,
    is_base64_data_uri,
};
use serde_json::Value;

use crate::auth::AuthUser;
use crate::error::ApiError;
use crate::orchestration::unary::{UnarySuccess, UnaryUpstreamResponse, execute_unary};
use crate::pipeline::{
    CACHE_HIT_HEADER, EndpointPipeline, apply_cost_header, sanitize_timeout_secs, set_header,
    stamp_ratelimit_headers, stamp_request_id_header,
};
use crate::state::SharedState;

/// Default embeddings request timeout (seconds).
const DEFAULT_TIMEOUT_SECS: u64 = 60;

mod batches;

/// Embeddings handler.
///
/// Flow (mirrors the chat handler so embeddings participate in the same hook
/// chain — budget, rate-limit, cache, cost-tracking):
/// 1. Parse the OpenAI-shaped embeddings request (`model`, `input`).
/// 2. Run pre-call hooks (rate-limit, budget, cache lookup).
/// 3. Resolve a deployment via the router.
/// 4. Resolve the provider via the registry → `EmbeddingProvider`.
/// 5. Transform request, POST to upstream, transform response.
/// 6. Run post-call hooks (debit spend, increment RPM/TPM, cache store, cost).
///
/// Guardrail scope: only in-place PII masking is applied to the `input` field.
/// The full chat guardrail chain (Presidio, injection, AIM, Lakera, Bedrock,
/// Model Armor, custom, semantic — see `chat::apply_request_guardrails`) is
/// **not** invoked. This is an intentional Rust-idiomatic divergence: those
/// guardrails are chat-shaped and inspect `messages`, while PII is the most
/// relevant guardrail for embedding inputs (which otherwise leak into vector
/// stores). An operator who configures an injection guardrail expecting it to
/// cover all text-ingressing endpoints will find embeddings are uncovered.
pub async fn embeddings(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    headers: axum::http::HeaderMap,
    body: Bytes,
) -> Result<Response, ApiError> {
    let state = shared.load_full();
    embeddings_inner_with_headers(state, &auth, body, &headers).await
}

/// Header-aware entrypoint for sibling HTTP handlers. Retains all repeated /
/// comma-separated tags from the request headers.
pub(crate) async fn embeddings_inner_with_headers(
    state: std::sync::Arc<crate::state::AppState>,
    auth: &AuthUser,
    body: Bytes,
    headers: &axum::http::HeaderMap,
) -> Result<Response, ApiError> {
    let mut request_body: Value = serde_json::from_slice(&body)
        .map_err(|e| ApiError::bad_request(format!("invalid JSON body: {e}")))?;

    crate::security::check_json_depth(&request_body, crate::security::MAX_JSON_DEPTH)?;
    crate::orchestration::request_controls::validate(&request_body)?;

    let model = request_body
        .get("model")
        .and_then(|v| v.as_str())
        .filter(|model| !model.trim().is_empty())
        .ok_or_else(|| ApiError::bad_request("missing required field: 'model'"))?
        .to_string();

    crate::pipeline::check_request_model_acl(&state, auth, &model, &request_body, headers).await?;
    if let Some(pii) = state.pii_guardrail.as_ref() {
        mask_embeddings_input(pii, &mut request_body);
    }
    let tags = crate::pipeline::extract_tags(headers, Some(&request_body));

    let mut pipeline = EndpointPipeline::for_request_with_headers(
        &state,
        auth,
        &model,
        request_body.clone(),
        headers,
    )
    .await?;
    let session_id = pipeline.call_ctx.session_id.clone();
    pipeline.call_ctx.cache_namespace = Some("embeddings".into());
    pipeline.apply_request_tags(&tags).await?;

    if let Some(cached) = pipeline.pre_call().await? {
        tracing::info!(request_id = %pipeline.call_ctx.request_id, "embeddings response cache hit");
        pipeline.call_ctx.cache_hit = true;
        let usage = extract_embedding_usage(&cached);
        let cached_model = cached
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or(&model)
            .to_string();
        // No deployment is resolved on the cache short-circuit (the request
        // never reaches the router), so derive the provider prefix from the
        // cached response's `model` field (mirroring the chat orchestrator's
        // `provider_prefix`) so spend logs and observability hooks attribute
        // the (skipped) cost to the serving provider instead of the empty
        // default. An empty prefix preserves the legacy provider-agnostic
        // behaviour for bare model names.
        let cached_provider = provider_prefix(&cached_model);
        // The cost header reflects the cached response's would-have-been value
        // (cost computation has no cache-hit guard); the spend-debit skip
        // happens later inside `CostTrackingHook::post_success`, which returns
        // early when `ctx.cache_hit` is set.
        let cost = pipeline
            .post_success(
                &cached_model,
                usage.as_ref(),
                Some(&cached),
                cached_provider,
            )
            .await;
        let mut public = cached;
        crate::pipeline::normalize_client_response(
            &mut public,
            crate::pipeline::public_response_model(&pipeline.call_ctx),
        );
        let mut resp = Json(public).into_response();
        apply_cost_header(&mut resp, cost);
        set_header(&mut resp, CACHE_HIT_HEADER, "true");
        stamp_request_id_header(&mut resp, pipeline.request_id());
        crate::orchestration::apply_router_headers(&mut resp, &pipeline.call_ctx);
        crate::pipeline::stamp_key_spend_header(&mut resp, &state, &pipeline.call_ctx).await;
        return Ok(resp);
    }

    let request_id = pipeline.request_id().to_string();

    // Delegate the retry/failover loop, budget reservation/settle, and
    // post-success/post-failure hooks to the shared unary executor. The
    // closure is the only embeddings-specific piece: provider resolution,
    // request/response transform, and the upstream HTTP call.
    let result = execute_unary(
        &state,
        &mut pipeline.call_ctx,
        auth,
        &model,
        &request_body,
        session_id,
        tags.first().cloned(),
        |deployment, body| {
            let state = state.clone();
            let deployment = deployment.clone();
            let body = body.clone();
            async move { call_upstream(&state, &deployment, body).await }
        },
    )
    .await;

    match result {
        Ok(UnarySuccess {
            body: transformed,
            cost,
        }) => {
            let mut resp = Json(transformed).into_response();
            apply_cost_header(&mut resp, cost);
            stamp_request_id_header(&mut resp, &request_id);
            // Stamp the full `x-ratelimit-*` set (limit + remaining + reset)
            // from the live per-key counters, matching the chat orchestrator.
            // No-op for master-key requests and cache hits (the latter never
            // reach this branch).
            stamp_ratelimit_headers(&mut resp, pipeline.state, &pipeline.call_ctx).await;
            crate::orchestration::apply_router_headers(&mut resp, &pipeline.call_ctx);
            crate::pipeline::stamp_key_spend_header(&mut resp, &state, &pipeline.call_ctx).await;
            Ok(resp)
        }
        Err(err) => Err(err.for_inference()),
    }
}

/// Perform the upstream embeddings call against one deployment: provider/
/// credential resolution, request transform, POST, and response transform.
///
/// Returns the transformed response body and extracted token usage. This is
/// the closure injected into [`execute_unary`] — everything else (limits,
/// budget, cooldown, hooks) is handled by the shared executor.
async fn call_upstream(
    state: &crate::state::AppState,
    deployment: &oxllm_core::Deployment,
    request_body: Value,
) -> Result<UnaryUpstreamResponse, ApiError> {
    let timeout_secs = crate::orchestration::request_controls::seconds(&request_body, "timeout")
        .or(deployment.oxllm_params.timeout)
        .unwrap_or_else(|| {
            state
                .config
                .oxllm_settings
                .request_timeout
                .map(|t| t as f64)
                .unwrap_or(DEFAULT_TIMEOUT_SECS as f64)
        });
    let deadline = tokio::time::Instant::now() + sanitize_timeout_secs(timeout_secs);
    tokio::time::timeout_at(
        deadline,
        call_upstream_inner(state, deployment, request_body, deadline),
    )
    .await
    .map_err(|_| ApiError::gateway_timeout())?
}

struct EmbeddingCall<'a> {
    state: &'a crate::state::AppState,
    deployment: &'a oxllm_core::Deployment,
    embeddings: &'a dyn EmbeddingProvider,
    creds: ResolvedCredentials,
    response_model: String,
    estimated_input_tokens: u64,
    deadline: tokio::time::Instant,
}

async fn call_upstream_inner(
    state: &crate::state::AppState,
    deployment: &oxllm_core::Deployment,
    mut request_body: Value,
    deadline: tokio::time::Instant,
) -> Result<UnaryUpstreamResponse, ApiError> {
    let provider = state
        .registry
        .resolve_for_deployment(deployment)
        .ok_or_else(|| {
            ApiError::upstream(
                500,
                format!("unsupported provider: '{}'", deployment.provider_slug()),
            )
        })?;
    let embeddings = provider.embeddings().ok_or_else(|| {
        ApiError::upstream(
            500,
            format!(
                "provider '{}' does not support embeddings",
                deployment.provider_slug()
            ),
        )
    })?;
    let response_model = if deployment.oxllm_params.model.contains('*') {
        let prefix = format!("{}/", deployment.provider_slug());
        let requested = request_body
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or(&deployment.oxllm_params.model);
        format!(
            "{prefix}{}",
            requested.strip_prefix(&prefix).unwrap_or(requested)
        )
    } else {
        deployment.oxllm_params.model.clone()
    };
    let estimated_input_tokens = if deployment.provider_slug() == "bedrock" {
        match request_body.get("input") {
            Some(Value::String(text)) => oxllm_core::tokens::estimate_tokens(text, &response_model),
            Some(Value::Array(items)) => {
                items
                    .iter()
                    .filter_map(Value::as_str)
                    .fold(0u64, |total, text| {
                        total.saturating_add(oxllm_core::tokens::estimate_tokens(
                            text,
                            &response_model,
                        ))
                    })
            }
            _ => 0,
        }
    } else {
        0
    };
    let creds = state
        .credential_resolver
        .resolve_for(deployment)
        .resolve(deployment)
        .await
        .map_err(ApiError::from)?;
    let endpoint_deployment =
        crate::orchestration::upstream::deployment_with_resolved_region(deployment, &creds);
    crate::orchestration::request_controls::strip_before_transform(&mut request_body);
    crate::orchestration::upstream::remove_additional_params(
        &mut request_body,
        deployment.oxllm_params.additional_drop_params.as_deref(),
    );
    let input_count = match request_body.get("input") {
        Some(Value::Array(input)) => input.len(),
        Some(Value::String(_)) => 1,
        _ => 0,
    };
    let plan = embeddings
        .prepare_requests(request_body, &endpoint_deployment)
        .map_err(transform_error_to_bad_request)?;
    let call = EmbeddingCall {
        state,
        deployment,
        embeddings,
        creds,
        response_model,
        estimated_input_tokens,
        deadline,
    };
    match plan {
        EmbeddingRequestPlan::Single(request) => {
            Ok(
                call_upstream_batch(&call, request, crate::security::MAX_RESPONSE_BODY_BYTES)
                    .await?
                    .0,
            )
        }
        EmbeddingRequestPlan::Batched(batches) => {
            batches::validate_plan(&batches, input_count)?;
            let mut remaining = crate::security::MAX_RESPONSE_BODY_BYTES;
            let mut responses = Vec::with_capacity(batches.len());
            // Sequential execution avoids spending the image leg's quota after
            // a text failure. Both legs share one deadline and response budget.
            for batch in batches {
                let (response, bytes) =
                    call_upstream_batch(&call, batch.upstream, remaining).await?;
                remaining = remaining.saturating_sub(bytes);
                responses.push((batch.input_indices, response));
            }
            batches::combine(responses, input_count)
        }
    }
}

/// Send one provider-selected batch through the ordinary bounded/authenticated
/// transport. Retry, reservation and cache hooks still see one logical request.
async fn call_upstream_batch(
    call: &EmbeddingCall<'_>,
    request: EmbeddingRequest,
    response_cap: usize,
) -> Result<(UnaryUpstreamResponse, usize), ApiError> {
    let state = call.state;
    let deployment = call.deployment;
    let embeddings = call.embeddings;
    let creds = &call.creds;
    let EmbeddingRequest {
        url,
        body: mut transformed,
        params,
        allowed_fields,
    } = request;
    crate::orchestration::upstream::restore_allowed_params(
        &mut transformed,
        &params,
        deployment.oxllm_params.allowed_openai_params.as_deref(),
        Some(&["model", "input", "encoding_format", "dimensions"]),
        None,
    );
    // Deployment overrides are applied before the final drop pass.
    crate::orchestration::upstream::shape_outbound_body(
        &mut transformed,
        deployment.oxllm_params.additional_drop_params.as_deref(),
        deployment.oxllm_params.extra_body.as_ref(),
    );
    drop(params);
    if let Some(allowed) = allowed_fields
        && let Some(body) = transformed.as_object_mut()
    {
        body.retain(|field, _| allowed.contains(&field.as_str()));
    }

    let mut headers = HashMap::new();
    embeddings.apply_auth(deployment, creds, &mut headers);
    oxllm_core::provider::apply_auth_overrides(deployment, creds, &mut headers);
    crate::orchestration::upstream::merge_extra_headers(&mut headers, deployment);

    let body_bytes = serde_json::to_vec(&transformed)
        .map_err(|e| ApiError::internal(format!("failed to serialize request body: {e}")))?;
    drop(transformed);
    // Bedrock signs the exact body bytes. Serialize once, then send those
    // bytes, so the signature matches the payload on the wire.
    if let oxllm_core::provider::ResolvedCredentials::AwsSigV4 {
        access_key,
        secret_key,
        session_token,
        region,
    } = creds
    {
        let mut header_pairs: Vec<(String, String)> = headers
            .iter()
            .map(|(k, v)| (k.to_ascii_lowercase(), v.clone()))
            .collect();
        if let Err(e) = crate::bedrock_sigv4::sign_request(
            "POST",
            &url,
            &mut header_pairs,
            &body_bytes,
            access_key,
            secret_key,
            region,
            "bedrock",
            session_token.as_deref(),
        ) {
            tracing::error!(
                error.kind = "config",
                detail = %e,
                "SigV4 request signing failed"
            );
            return Err(ApiError::internal("upstream request signing failed"));
        }
        headers = header_pairs.into_iter().collect();
    }

    let mut req_builder = state
        .http_client
        .post(&url)
        .header("content-type", "application/json")
        .timeout(
            call.deadline
                .saturating_duration_since(tokio::time::Instant::now()),
        )
        .body(body_bytes);

    for (key, value) in &headers {
        req_builder = req_builder.header(key, value);
    }

    let response = req_builder.send().await.map_err(|e| {
        if e.is_timeout() {
            ApiError::gateway_timeout()
        } else {
            // Transport failure: redact the raw reqwest detail (which embeds
            // the resolved upstream URL) and render a generic 502, matching
            // the chat path's redaction contract.
            ApiError::from(oxllm_core::Error::Network(e.to_string()))
        }
    })?;

    let status = response.status();
    if !status.is_success() {
        // Capture upstream headers BEFORE reading the body so the client-facing
        // error can forward the debugging allowlist (`x-request-id`,
        // `retry-after`, `x-ratelimit-*`) — mirroring the chat error path.
        // Without this, provider support correlation and the vendor's backoff
        // signal are silently dropped.
        let headers = response.headers().clone();
        let status_u16 = status.as_u16();
        // Truncate oversized error bodies, retaining the actionable provider
        // status and retry headers rather than replacing them with a 500.
        let bytes = crate::security::read_error_body_capped(
            response,
            crate::security::MAX_ERROR_BODY_BYTES,
        )
        .await;
        let body_text = String::from_utf8_lossy(&bytes).to_string();
        let body: Value = serde_json::from_str(&body_text).unwrap_or(Value::Null);
        let message = body
            .get("error")
            .and_then(|e| e.get("message"))
            .and_then(|m| m.as_str())
            .unwrap_or("upstream error")
            .to_string();
        return Err(ApiError::upstream(status_u16, message).with_headers(Some(headers)));
    }

    let response_headers = response.headers().clone();

    // Stream the success body into a buffer capped at `MAX_RESPONSE_BODY_BYTES`
    // (Content-Length pre-check + mid-stream abort), then deserialize. A
    // malicious/compromised upstream returning a 200 with a multi-GB JSON
    // document is rejected after the cap rather than OOMing the proxy — this
    // matches the chat path's `sanitize_upstream_response` defence.
    let reported_input_tokens = response
        .headers()
        .get("x-amzn-bedrock-input-token-count")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok());
    let foundry_input_tokens = response
        .headers()
        .get("llm_provider-num_tokens")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|tokens| *tokens > 0);
    let bytes = read_embedding_response(response, response_cap, &response_headers).await?;
    let response_bytes = bytes.len();
    let response_body: Value = serde_json::from_slice(&bytes).map_err(|e| {
        tracing::error!(
            error.kind = "serialization",
            detail = %e,
            "failed to parse embeddings response as JSON"
        );
        ApiError::internal("failed to parse embeddings response as JSON")
    })?;

    let mut transformed = embeddings
        .transform_response(&response_body, &call.response_model)
        .map_err(transform_error_to_bad_request)?;

    // Foundry's Cohere responses may meter through Azure ML headers.
    // The reported count is authoritative even when the body has usage.
    if let Some(tokens) = foundry_input_tokens {
        transformed["usage"] = serde_json::json!({"prompt_tokens": tokens, "total_tokens": tokens});
    }

    if deployment.provider_slug() == "bedrock" && extract_embedding_usage(&transformed).is_none() {
        let tokens = reported_input_tokens.unwrap_or(call.estimated_input_tokens);
        transformed["usage"] = serde_json::json!({"prompt_tokens": tokens, "total_tokens": tokens});
    }

    let usage = extract_embedding_usage(&transformed);
    Ok(((transformed, usage, response_headers), response_bytes))
}

async fn read_embedding_response(
    response: reqwest::Response,
    cap: usize,
    headers: &axum::http::HeaderMap,
) -> Result<Vec<u8>, ApiError> {
    crate::security::read_bounded_bytes(response, cap)
        .await
        .map_err(|error| ApiError::upstream(502, error.message).with_headers(Some(headers.clone())))
}

/// Explicit bad requests remain terminal. Adapter/configuration failures may
/// be specific to one deployment in a mixed-provider group.
fn transform_error_to_bad_request(e: oxllm_core::Error) -> ApiError {
    match e {
        oxllm_core::Error::Transform(msg) => {
            ApiError::upstream(500, format!("transform error: {msg}"))
        }
        _ => ApiError::from(e),
    }
}

/// Derive a provider slug from a model string that carries a `provider/`
/// prefix, returning an empty string when no prefix is present. Used on the
/// cache short-circuit (before any deployment is picked) so the post-success
/// chain attributes the (skipped) cost to the serving provider — matching the
/// chat orchestrator's `provider_prefix`.
fn provider_prefix(model: &str) -> &str {
    match model.split_once('/') {
        Some((provider, _)) if !provider.is_empty() => provider,
        _ => "",
    }
}

/// Extract token usage from an embeddings response body. Embeddings responses
/// carry `usage.prompt_tokens` and `usage.total_tokens` (no completion tokens).
///
/// Returns `None` when there is no `usage` object, and also when both resolved
/// token counts are zero: a present-but-empty usage block (e.g. `{"usage": {}}`
/// or `{"usage": {"prompt_tokens": 0}}`) carries no real metering signal, so
/// surfacing it as `Some(zeros)` would mask "provider returned a usage object
/// with no numbers" from observability hooks that treat `usage.is_some()` as
/// "metered".
///
/// Embeddings produce no completion tokens, so `total_tokens == prompt_tokens`
/// on a well-formed response. Some providers/aggregators return only
/// `total_tokens` (no `prompt_tokens`); the cost calculator bills off
/// `prompt_tokens`, so when only `total_tokens` is present it is treated as the
/// prompt token count — otherwise such a request would be metered at $0.
fn extract_embedding_usage(value: &Value) -> Option<UsageSummary> {
    let usage = value.get("usage")?;
    let prompt_raw = usage
        .get("prompt_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let total = usage
        .get("total_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(prompt_raw);
    let prompt = prompt_raw.max(total);
    if prompt == 0 && total == 0 {
        return None;
    }
    Some(UsageSummary {
        prompt_tokens: prompt,
        completion_tokens: 0,
        total_tokens: prompt,
        cache_read_input_tokens: None,
        cache_creation_input_tokens: None,
        audio_input_tokens: None,
        reasoning_tokens: None,
        citation_tokens: None,
        web_search_requests: None,
        browser_open_requests: None,
        cache_creation_1h_input_tokens: None,
    })
}

/// Mask PII in the embeddings `input` field in place. Handles the string
/// shape (`"input": "hi"`) and the array-of-strings shape
/// (`"input": ["a", "b"]`). Non-string / non-array inputs are left untouched.
///
/// The reference additionally recognises a token-array shape
/// (`"input": [[101, 102], ...]`) and, for providers that do not support
/// token-array input, decodes the tokens back to text before forwarding so
/// PII can be re-masked. oxllm forwards `input` verbatim to the provider
/// transform and does not decode token arrays — this is an intentional scope
/// omission. As a consequence, a token-array `input` falls through the `_ => {}`
/// arm here: pre-tokenized PII cannot be regex-matched anyway, so the no-op is
/// safe, but operators should be aware the masking does not cover that shape.
fn mask_embeddings_input(pii: &crate::guardrails::PiiGuardrail, body: &mut Value) {
    let Some(input) = body.get_mut("input") else {
        return;
    };
    match input {
        Value::String(s) => {
            mask_embedding_text(pii, s);
        }
        Value::Array(items) => {
            for item in items.iter_mut() {
                if let Value::String(s) = item {
                    mask_embedding_text(pii, s);
                }
            }
        }
        _ => {}
    }
}

fn mask_embedding_text(pii: &crate::guardrails::PiiGuardrail, text: &mut String) {
    let opaque_image = text.split_once(',').is_some_and(|(header, _)| {
        header.starts_with("data:image/")
            && header
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || b":/;=+.-".contains(&byte))
    }) && is_base64_data_uri(text);
    if !opaque_image {
        *text = pii.mask_text(text);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_embedding_usage_reads_prompt_and_total() {
        let v = serde_json::json!({
        "usage": {"prompt_tokens": 5, "total_tokens": 5}
        });
        let u = extract_embedding_usage(&v).expect("present");
        assert_eq!(u.prompt_tokens, 5);
        assert_eq!(u.total_tokens, 5);
        assert_eq!(u.completion_tokens, 0);
    }

    #[test]
    fn extract_embedding_usage_returns_none_when_missing() {
        assert!(extract_embedding_usage(&serde_json::json!({"data": []})).is_none());
    }

    #[test]
    fn pii_masking_preserves_opaque_images_without_exempting_data_like_prose() {
        let pii = crate::guardrails::PiiGuardrail::default();
        let image = "data:image/png;base64,/4111111111111111/AA";
        assert_ne!(
            pii.mask_text(image),
            image,
            "the regression must exercise a real false positive"
        );
        let mut body = serde_json::json!({"input":[image, "contact john@example.com", "data:image/png;base64,john@example.com", "data:john@example.com,YWJj"]});
        mask_embeddings_input(&pii, &mut body);
        assert_eq!(body["input"][0], image);
        for index in 1..4 {
            assert!(body["input"][index].as_str().unwrap().contains("[EMAIL]"));
        }
    }

    #[test]
    fn mask_embeddings_input_masks_string_input() {
        let pii = crate::guardrails::PiiGuardrail::default();
        let mut body = serde_json::json!({
            "model": "text-embedding-3-small",
            "input": "contact john@example.com"
        });
        mask_embeddings_input(&pii, &mut body);
        assert_eq!(body["input"], "contact [EMAIL]");
    }

    #[test]
    fn mask_embeddings_input_masks_array_input() {
        let pii = crate::guardrails::PiiGuardrail::default();
        let mut body = serde_json::json!({
            "model": "text-embedding-3-small",
            "input": ["john@example.com", "no pii here", "call 555-123-4567"]
        });
        mask_embeddings_input(&pii, &mut body);
        assert_eq!(body["input"][0], "[EMAIL]");
        assert_eq!(body["input"][1], "no pii here");
        assert_eq!(body["input"][2], "call [PHONE]");
    }

    #[test]
    fn mask_embeddings_input_noop_when_missing() {
        let pii = crate::guardrails::PiiGuardrail::default();
        let mut body = serde_json::json!({"model": "text-embedding-3-small"});
        mask_embeddings_input(&pii, &mut body);
        assert_eq!(body["model"], "text-embedding-3-small");
    }

    #[test]
    fn adapter_errors_are_retryable_and_explicit_bad_requests_remain_terminal() {
        let err =
            transform_error_to_bad_request(oxllm_core::Error::Transform("bad shape".to_string()));
        assert_eq!(err.status, axum::http::StatusCode::INTERNAL_SERVER_ERROR);
        let err = transform_error_to_bad_request(oxllm_core::Error::BadRequest("bad input".into()));
        assert_eq!(err.status, axum::http::StatusCode::BAD_REQUEST);
    }

    #[test]
    fn provider_prefix_extracts_prefix_segment() {
        // Cache-hit attribution: the cached response's `model` field carries a
        // `provider/` prefix that must surface as the ResponseSummary provider.
        assert_eq!(provider_prefix("openai/text-embedding-3-small"), "openai");
        assert_eq!(provider_prefix("cohere/embed-english-v3"), "cohere");
        // Bare model names (no deployment resolved → no prefix) stay empty.
        assert_eq!(provider_prefix("text-embedding-3-small"), "");
        assert_eq!(provider_prefix("/leading-slash"), "");
    }

    #[test]
    fn non_transform_core_error_keeps_normal_status() {
        let err = transform_error_to_bad_request(oxllm_core::Error::Internal);
        assert_eq!(
            err.status,
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            "non-transform errors must not be remapped"
        );
    }

    /// Success reads reject oversized bodies; error reads instead truncate at
    /// the smaller error cap while preserving the provider's status.
    #[test]
    fn upstream_success_body_is_bounded() {
        // A body just under the limit is accepted.
        let ok = vec![b' '; crate::security::MAX_RESPONSE_BODY_BYTES];
        assert!(
            crate::security::check_bytes_limit(&ok).is_ok(),
            "body at the limit must be accepted"
        );
        // Success bodies one byte over the limit must be rejected.
        let mut big = ok.clone();
        big.push(b' ');
        assert!(
            crate::security::check_bytes_limit(&big).is_err(),
            "body over the limit must be rejected so the embeddings read cannot be unbounded"
        );
    }

    #[test]
    fn extract_embedding_usage_returns_none_for_empty_usage() {
        assert!(extract_embedding_usage(&serde_json::json!({"usage": {}})).is_none());
    }

    #[test]
    fn extract_embedding_usage_returns_none_when_all_tokens_zero() {
        assert!(
            extract_embedding_usage(&serde_json::json!({
                "usage": {"prompt_tokens": 0, "total_tokens": 0}
            }))
            .is_none()
        );
        // A single zero field with the other defaulting to it is also empty.
        assert!(
            extract_embedding_usage(&serde_json::json!({
                "usage": {"prompt_tokens": 0}
            }))
            .is_none()
        );
    }

    /// A provider/transform that returns only `total_tokens` (no
    /// `prompt_tokens`) must surface `prompt_tokens == total_tokens` so the
    /// cost calculator (which bills off `prompt_tokens`) does not silently
    /// zero-cost the request. Embeddings carry no completion tokens, so
    /// `total == prompt` on a well-formed response.
    #[test]
    fn extract_embedding_usage_promotes_total_when_prompt_absent() {
        let u = extract_embedding_usage(&serde_json::json!({
            "usage": {"total_tokens": 42}
        }))
        .expect("present");
        assert_eq!(u.prompt_tokens, 42);
        assert_eq!(u.total_tokens, 42);
        assert_eq!(u.completion_tokens, 0);
    }

    /// When a provider returns `prompt_tokens: 0` explicitly with a non-zero
    /// `total_tokens`, the same promotion applies — embeddings have no
    /// completion tokens, so `total_tokens` is the prompt count.
    #[test]
    fn extract_embedding_usage_promotes_total_when_prompt_zero() {
        let u = extract_embedding_usage(&serde_json::json!({
            "usage": {"prompt_tokens": 0, "total_tokens": 7}
        }))
        .expect("present");
        assert_eq!(u.prompt_tokens, 7);
        assert_eq!(u.total_tokens, 7);
    }

    /// `read_bounded_bytes` (used by the embeddings success path)
    /// must abort mid-stream as soon as the body exceeds the cap, rather than
    /// buffering the whole payload first.
    #[tokio::test]
    async fn read_bounded_bytes_aborts_oversized_stream() {
        let mock = httpmock::MockServer::start();
        let cap = 1024;
        mock.mock(|when, then| {
            when.method(httpmock::Method::GET).path("/big");
            then.status(200).body("x".repeat(cap * 4));
        });
        let resp = reqwest::get(format!("{}/big", mock.base_url()))
            .await
            .expect("request");
        let err = crate::security::read_bounded_bytes(resp, cap)
            .await
            .expect_err("oversized body must be rejected");
        assert_eq!(
            err.status,
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            "streaming abort must surface a bounded error, not hang or OOM"
        );
    }

    #[tokio::test]
    async fn oversized_embedding_success_is_a_bounded_upstream_502_not_an_internal_500() {
        let server = httpmock::MockServer::start_async().await;
        let wire = server
            .mock_async(|when, then| {
                when.method(httpmock::Method::GET).path("/embedding");
                then.status(200).body("x".repeat(128));
            })
            .await;
        let response = reqwest::get(server.url("/embedding")).await.unwrap();
        let headers = response.headers().clone();
        let error = read_embedding_response(response, 64, &headers)
            .await
            .unwrap_err();
        assert_eq!(error.status, axum::http::StatusCode::BAD_GATEWAY);
        assert!(error.message.contains("64"));
        assert!(error.headers.is_some());
        wire.assert_hits_async(1).await;
    }

    /// `read_bounded_bytes` must return the full body intact when it stays
    /// under the cap (the common path for well-formed embeddings responses).
    #[tokio::test]
    async fn read_bounded_bytes_returns_body_under_cap() {
        let mock = httpmock::MockServer::start();
        let payload = "hello embeddings";
        mock.mock(|when, then| {
            when.method(httpmock::Method::GET).path("/small");
            then.status(200).body(payload);
        });
        let resp = reqwest::get(format!("{}/small", mock.base_url()))
            .await
            .expect("request");
        let bytes = crate::security::read_bounded_bytes(resp, 4096)
            .await
            .expect("under-cap body");
        assert_eq!(bytes, payload.as_bytes());
    }

    /// A provider 429/5xx on the embeddings path must forward the debugging
    /// allowlist (`x-request-id`, `retry-after`) to the client — same parity
    /// requirement as the chat error path (`upstream.rs` header capture) — and
    /// must NOT leak non-allowlisted upstream headers (`server`, internals).
    #[tokio::test]
    async fn embeddings_upstream_error_forwards_debug_headers() {
        use crate::auth::AuthUser;
        use crate::config::types::{ConfigModel, GeneralSettings, OxllmSettings, ProxyConfig};
        use crate::state::AppState;
        use axum::http::StatusCode;
        use oxllm_core::deployment::OxllmParams;
        use std::sync::Arc;

        let server = httpmock::MockServer::start();
        server.mock(|when, then| {
            when.method("POST").path("/embeddings");
            then.status(429)
                .header("content-type", "application/json")
                .header("x-request-id", "req_emb_xyz")
                .header("retry-after", "7")
                .header("server", "vendor/1.0")
                .header("x-vendor-internal", "secret")
                .body(r#"{"error":{"message":"rate limited"}}"#);
        });

        let config = ProxyConfig {
            model_list: vec![ConfigModel {
                model_name: "text-embedding-3-small".to_string(),
                oxllm_params: OxllmParams {
                    model: "openai/text-embedding-3-small".to_string(),
                    api_key: Some("sk-test".to_string()),
                    api_base: Some(server.base_url()),
                    ..Default::default()
                },
                ..Default::default()
            }],
            oxllm_settings: OxllmSettings::default(),
            general_settings: GeneralSettings {
                allow_private_api_base: Some(true),
                ..Default::default()
            },
            router_settings: Default::default(),
            extra: serde_json::Map::new(),
        };
        let state = Arc::new(
            AppState::with_registry(config, oxllm_providers::default_registry())
                .expect("state built"),
        );

        let body = serde_json::json!({
            "model": "text-embedding-3-small",
            "input": "hello"
        });
        let result = embeddings_inner_with_headers(
            state,
            &AuthUser::open(),
            Bytes::from(serde_json::to_vec(&body).expect("serialize")),
            &axum::http::HeaderMap::new(),
        )
        .await;
        let err = result.expect_err("provider 429 must surface an error");
        assert_eq!(err.status, StatusCode::TOO_MANY_REQUESTS);
        let resp = err.into_response();
        assert_eq!(
            resp.headers()
                .get("x-request-id")
                .and_then(|v| v.to_str().ok()),
            Some("req_emb_xyz"),
            "provider x-request-id must be forwarded"
        );
        assert_eq!(
            resp.headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok()),
            Some("7"),
            "provider retry-after must be forwarded"
        );
        assert!(
            resp.headers().get("server").is_none(),
            "non-allowlisted upstream headers must not leak"
        );
        assert!(
            resp.headers().get("x-vendor-internal").is_none(),
            "vendor-internal headers must not leak"
        );
    }

    /// MEDIUM (C20): an upstream embeddings response that returns only
    /// `total_tokens` (no `prompt_tokens`) — a non-conformant but plausible
    /// shape from some aggregator wrappers — must still be billed. The cost
    /// calculator bills off `prompt_tokens`, so `extract_embedding_usage`
    /// promotes `total_tokens` to `prompt_tokens` for embeddings (which carry
    /// no completion tokens). This test asserts the cost header on the happy
    /// path is present and non-zero, closing the "silently zero-cost" gap and
    /// the handler-level cost-metering coverage gap.
    #[tokio::test]
    async fn embeddings_total_only_usage_produces_nonzero_cost_header() {
        use crate::auth::AuthUser;
        use crate::config::types::{ConfigModel, GeneralSettings, OxllmSettings, ProxyConfig};
        use crate::state::AppState;
        use oxllm_core::deployment::OxllmParams;
        use std::sync::Arc;

        let server = httpmock::MockServer::start();
        server.mock(|when, then| {
            when.method("POST").path("/embeddings");
            then.status(200)
                .header("content-type", "application/json")
                .body(
                    r#"{"object":"list","data":[{"object":"embedding","index":0,"embedding":[0.1,0.2,0.3]}],"model":"text-embedding-3-small","usage":{"total_tokens":42}}"#,
                );
        });

        let config = ProxyConfig {
            model_list: vec![ConfigModel {
                model_name: "text-embedding-3-small".to_string(),
                oxllm_params: OxllmParams {
                    model: "openai/text-embedding-3-small".to_string(),
                    api_key: Some("sk-test".to_string()),
                    api_base: Some(server.base_url()),
                    ..Default::default()
                },
                ..Default::default()
            }],
            oxllm_settings: OxllmSettings::default(),
            general_settings: GeneralSettings {
                allow_private_api_base: Some(true),
                ..Default::default()
            },
            router_settings: Default::default(),
            extra: serde_json::Map::new(),
        };
        let state = Arc::new(
            AppState::with_registry(config, oxllm_providers::default_registry())
                .expect("state built"),
        );

        let body = serde_json::json!({
            "model": "text-embedding-3-small",
            "input": "hello"
        });
        let resp = embeddings_inner_with_headers(
            state,
            &AuthUser::open(),
            Bytes::from(serde_json::to_vec(&body).expect("serialize")),
            &axum::http::HeaderMap::new(),
        )
        .await
        .expect("happy-path embeddings call");

        let cost = resp
            .headers()
            .get("x-oxllm-response-cost")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<f64>().ok())
            .expect("cost header must be present");
        assert!(
            cost > 0.0,
            "total-only usage must be billed (prompt_tokens promoted from total_tokens), got {cost}"
        );
    }
}
