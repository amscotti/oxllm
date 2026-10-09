//! `POST /v1/messages/count_tokens` — Anthropic Count Tokens API.
//!
//! Anthropic exposes `/v1/messages/count_tokens` to return the exact token
//! count for a given message set without generating a completion. The request
//! and response are already Anthropic-shaped, so — unlike `/v1/messages` —
//! system/tools fields remain native, image parts are normalized, generation
//! fields are omitted, and the upstream response is returned as-is.
//!
//! The handler is a thin wrapper around
//! [`crate::pipeline::forward_with_mode`] with [`PassthroughMode::HookedRaw`],
//! so the request still runs the full pre/post hook chain (auth via the proxy
//! middleware, rate-limit, budget, spend tracking, deployment RPM/TPM) and
//! resolves a deployment through the router (`pick_deployment_index`) exactly
//! like the other LLM endpoints.

use axum::Extension;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::Response;

use crate::auth::AuthUser;
use crate::error::ApiError;
use crate::handlers::anthropic_common::extract_request_model;
use crate::pipeline::{
    PassthroughMode, PassthroughRequest, forward_with_mode, vetted_client_headers,
};
use crate::state::SharedState;

/// Anthropic Count Tokens handler.
///
/// Flow:
/// 1. Parse the Anthropic-format body and extract the `model` field (required
///    — without it the upstream count_tokens call is meaningless).
/// 2. Delegate to [`forward_with_mode`] in [`PassthroughMode::HookedRaw`]:
///    pre-call hooks (rate-limit, budget, cache lookup) → router picks a
///    deployment → deployment RPM/TPM limits → forward the normalized body to
///    `{api_base}/messages/count_tokens` with provider-native auth →
///    post-success/post-failure hooks (spend tracking, metrics).
/// 3. Return the upstream Anthropic response directly (no conversion).
///
/// Usage metering on this route is effectively a no-op for *settled* spend:
/// count_tokens is free on Anthropic, and even though the passthrough tier
/// conservatively reserves a prompt-based budget estimate for the duration of
/// the upstream call (`BudgetHook::pre_call`), the free utility operation
/// releases the generation estimate at settlement. The transient
/// reservation can briefly appear to consume budget under heavy concurrency,
/// but no generation charge accrues. Native Messages inference instead uses
/// token-usage settlement on JSON and SSE responses.
pub async fn count_tokens(
    State(state): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, ApiError> {
    let model = extract_request_model(&body)?;
    let request: serde_json::Value = serde_json::from_slice(&body)
        .map_err(|_| ApiError::bad_request("invalid token-counting JSON"))?;
    let mut native = oxllm_providers::anthropic::count_tokens::normalize_request(&request, &model)
        .map_err(ApiError::from)?;
    if let Some(object) = native.as_object_mut() {
        crate::orchestration::request_controls::copy(&request, object);
    }
    let body = Bytes::from(native.to_string());
    forward_with_mode(
        &state.load_full(),
        &auth,
        PassthroughMode::HookedRaw {
            upstream_path: "messages/count_tokens".to_string(),
        },
        PassthroughRequest::new(
            axum::http::Method::POST,
            Some(&model),
            None,
            body,
            Some("application/json"),
        )
        .with_request_headers(&headers)
        .with_forward_headers(vetted_client_headers(&headers)),
    )
    .await
}

#[cfg(feature = "platform-apis")]
pub(crate) async fn count_at_provider(
    state: &crate::state::AppState,
    auth: &AuthUser,
    request: &serde_json::Value,
    model: &str,
) -> Result<Option<(u64, &'static str)>, ApiError> {
    use oxllm_core::provider::ChatProvider;
    use serde_json::{Value, json};
    crate::orchestration::request_controls::validate(request)?;
    let Some(deployment) = state
        .router
        .deployments_for_model(&state.deployments, model)
        .into_iter()
        .next()
    else {
        return Ok(None);
    };
    let tokenizer = match deployment.provider_slug() {
        "vertex_ai" if deployment.oxllm_params.model.contains("claude") => {
            "vertex_ai_partner_models"
        }
        "anthropic" => "anthropic",
        "azure_ai" if oxllm_providers::azure_ai::AzureAiProvider::is_anthropic(deployment) => {
            "anthropic"
        }
        _ => return Ok(None),
    };
    oxllm_providers::anthropic::count_tokens::validate_tools(request).map_err(ApiError::from)?;
    let mut native_deployment = deployment.clone();
    native_deployment.oxllm_params.model = deployment
        .oxllm_params
        .model
        .split_once('/')
        .map_or(deployment.oxllm_params.model.as_str(), |(_, model)| model)
        .to_string();
    let mut input = request.clone();
    if input.get("messages").and_then(Value::as_array).is_none() {
        let Some(prompt) = input.get("prompt").and_then(Value::as_str) else {
            return Ok(None);
        };
        input["messages"] = json!([{"role":"user","content":prompt}]);
    }
    let mut native = oxllm_providers::anthropic::AnthropicProvider
        .transform_request(&input, &native_deployment)
        .map_err(ApiError::from)?;
    if let Some(system) = request.get("system").filter(|value| !value.is_null()) {
        native["system"] = system.clone();
    }
    if let Some(tools) = request.get("tools").and_then(Value::as_array)
        && tools.iter().all(|tool| tool.get("function").is_none())
    {
        native["tools"] = json!(tools);
    }
    native["model"] = json!(model);
    if let Some(object) = native.as_object_mut() {
        crate::orchestration::request_controls::copy(request, object);
    }
    let response = forward_with_mode(
        state,
        auth,
        PassthroughMode::HookedRaw {
            upstream_path: "messages/count_tokens".into(),
        },
        PassthroughRequest::new(
            axum::http::Method::POST,
            Some(model),
            None,
            Bytes::from(native.to_string()),
            Some("application/json"),
        ),
    )
    .await?;
    if !response.status().is_success() {
        return Err(ApiError::upstream(
            response.status().as_u16(),
            "provider token counting failed",
        ));
    }
    let bytes = axum::body::to_bytes(
        response.into_body(),
        crate::security::MAX_RESPONSE_BODY_BYTES,
    )
    .await
    .map_err(|_| ApiError::internal("provider token-count response could not be read"))?;
    let reply: Value = serde_json::from_slice(&bytes)
        .map_err(|_| ApiError::internal("invalid provider token-count response"))?;
    let tokens = reply
        .get("input_tokens")
        .and_then(Value::as_u64)
        .ok_or_else(|| {
            ApiError::internal("provider token-count response is missing input_tokens")
        })?;
    Ok(Some((tokens, tokenizer)))
}
