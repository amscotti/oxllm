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
    if let Some(system) = oxllm_providers::anthropic::count_tokens::merge_system(
        request.get("system"),
        native.get("system"),
    )
    .map_err(ApiError::from)?
    {
        native["system"] = system;
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

#[cfg(test)]
mod tests {
    use super::*;
    use httpmock::MockServer;
    use serde_json::{Value, json};

    const HOSTED_COUNTERS: [(&str, &str, &str, &str, &str, &str); 3] = [
        (
            "anthropic/claude-sonnet-5-5",
            "/v1",
            "/v1/messages/count_tokens",
            "claude-sonnet-5-5",
            "x-api-key",
            "synthetic-key",
        ),
        (
            "azure_ai/claude-sonnet-5-5",
            "/anthropic/v1",
            "/anthropic/v1/messages/count_tokens",
            "claude-sonnet-5-5",
            "x-api-key",
            "synthetic-key",
        ),
        (
            "vertex_ai/claude-haiku-5-5@default",
            "",
            "/v1/projects/project/locations/us-east5/publishers/anthropic/models/count-tokens:rawPredict",
            "claude-haiku-5-5",
            "authorization",
            "Bearer synthetic-key",
        ),
    ];

    fn state(model: &str, api_base: &str) -> SharedState {
        let config = crate::config::parse_config_str(&format!(
            r#"
model_list:
  - model_name: counted
    oxllm_params:
      model: {model}
      api_base: {api_base}
      api_key: synthetic-key
      vertex_project: project
      vertex_location: global
      vertex_count_tokens_location: us-east5
general_settings:
  allow_private_api_base: true
router_settings:
  num_retries: 0
"#,
        ))
        .unwrap();
        let state =
            crate::state::AppState::with_registry(config, oxllm_providers::default_registry())
                .unwrap();
        std::sync::Arc::new(arc_swap::ArcSwap::from_pointee(state))
    }

    fn request(system: Value) -> Value {
        json!({
            "model": "counted",
            "system": system,
            "messages": [
                {"role": "system", "content": "Lift 1", "cache_control": {"type": "ephemeral", "ttl": "1h"}},
                {"role": "system", "content": [{"type": "text", "text": "Lift 2", "cache_control": {"type": "ephemeral"}}]},
                {"role": "user", "content": "hello"},
            ],
            "tools": [{"name": "lookup", "input_schema": {"type": "object"}, "cache_control": {"type": "ephemeral"}}],
        })
    }

    fn expected(request: &Value, model: &str) -> Value {
        let mut system = match &request["system"] {
            Value::String(text) if !text.is_empty() => vec![json!({"type": "text", "text": text})],
            Value::Array(blocks) => blocks.clone(),
            _ => Vec::new(),
        };
        system.extend([
            json!({"type": "text", "text": "Lift 1", "cache_control": {"type": "ephemeral", "ttl": "1h"}}),
            json!({"type": "text", "text": "Lift 2", "cache_control": {"type": "ephemeral"}}),
        ]);
        json!({
            "model": model,
            "system": system,
            "messages": [{"role": "user", "content": "hello"}],
            "tools": request["tools"],
        })
    }

    #[tokio::test]
    async fn native_count_tokens_lifts_system_and_preserves_hosted_urls_and_auth() {
        for (model, base_path, path, wire_model, header, credential) in HOSTED_COUNTERS {
            let upstream = MockServer::start_async().await;
            let state = state(model, &upstream.url(base_path));
            let input = request(json!([{
                "type": "text", "text": "Caller prefix",
                "cache_control": {"type": "ephemeral"},
            }]));
            let expected = expected(&input, wire_model);
            let mock = upstream
                .mock_async(|when, then| {
                    when.method(httpmock::Method::POST)
                        .path(path)
                        .header(header, credential)
                        .json_body_obj(&expected);
                    then.status(200).json_body(json!({"input_tokens": 73}));
                })
                .await;
            let response = count_tokens(
                State(state),
                Extension(AuthUser::open()),
                HeaderMap::new(),
                Bytes::from(input.to_string()),
            )
            .await
            .unwrap();
            assert_eq!(response.status(), axum::http::StatusCode::OK, "{model}");
            let bytes = axum::body::to_bytes(response.into_body(), 1024)
                .await
                .unwrap();
            let body: Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(body["input_tokens"], 73);
            mock.assert_hits_async(1).await;
        }
    }

    #[cfg(feature = "platform-apis")]
    #[tokio::test]
    async fn utility_count_tokens_merges_caller_system_before_extracted_content_on_all_hosts() {
        for (model, base_path, path, wire_model, header, credential) in HOSTED_COUNTERS {
            let upstream = MockServer::start_async().await;
            let state = state(model, &upstream.url(base_path));
            for caller in [
                json!("Caller prefix"),
                json!([{"type": "text", "text": "Caller prefix", "cache_control": {"type": "ephemeral"}}]),
                Value::Null,
            ] {
                let input = request(caller);
                let original = input.clone();
                let expected = expected(&input, wire_model);
                let mock = upstream
                    .mock_async(|when, then| {
                        when.method(httpmock::Method::POST)
                            .path(path)
                            .header(header, credential)
                            .json_body_obj(&expected);
                        then.status(200).json_body(json!({"input_tokens": 73}));
                    })
                    .await;
                let result =
                    count_at_provider(&state.load_full(), &AuthUser::open(), &input, "counted")
                        .await
                        .unwrap();
                let tokenizer = if model.starts_with("vertex_ai/") {
                    "vertex_ai_partner_models"
                } else {
                    "anthropic"
                };
                assert_eq!(result, Some((73, tokenizer)), "{model}: {input}");
                assert_eq!(input, original);
                mock.assert_hits_async(1).await;
                mock.delete_async().await;
            }
        }
    }
}
