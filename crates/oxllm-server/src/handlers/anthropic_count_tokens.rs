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
use crate::handlers::anthropic_common::{
    anthropic_error_response, extract_request_model, status_to_anthropic_error_type,
};
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
    use axum::response::IntoResponse;
    match count_tokens_inner(state, auth, headers, body).await {
        Ok(response) => Ok(response),
        Err(error) => {
            let status = error.status;
            let message = error.message.clone();
            let preserved =
                crate::pipeline::preserved_response_headers(error.into_response().headers());
            Ok(anthropic_error_response(
                status,
                status_to_anthropic_error_type(status),
                &message,
                &preserved,
            ))
        }
    }
}

async fn count_tokens_inner(
    state: SharedState,
    auth: AuthUser,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, ApiError> {
    let model = extract_request_model(&body)?;
    let request: serde_json::Value = serde_json::from_slice(&body)
        .map_err(|_| ApiError::bad_request("invalid token-counting JSON"))?;
    let app = state.load_full();
    let drop_params = app
        .router
        .deployments_for_model(&app.deployments, &model)
        .into_iter()
        .next()
        .and_then(|deployment| deployment.oxllm_params.drop_params)
        .or(app.config.oxllm_settings.drop_params)
        .unwrap_or(false)
        || request
            .get("drop_params")
            .and_then(serde_json::Value::as_bool)
            == Some(true);
    let mut native = oxllm_providers::anthropic::count_tokens::normalize_request_with_drop_params(
        &request,
        &model,
        drop_params,
    )
    .map_err(ApiError::from)?;
    if let Some(object) = native.as_object_mut() {
        crate::orchestration::request_controls::copy(&request, object);
    }
    let body = Bytes::from(native.to_string());
    forward_with_mode(
        &app,
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
    if deployment.provider_slug() == "bedrock" {
        return count_at_bedrock(state, auth, request, model).await;
    }
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
    if !ensure_count_messages(&mut input) {
        return Ok(None);
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
    let reply = forward_count_tokens(state, auth, model, native).await?;
    let tokens = reply
        .get("input_tokens")
        .and_then(Value::as_u64)
        .ok_or_else(|| {
            ApiError::internal("provider token-count response is missing input_tokens")
        })?;
    Ok(Some((tokens, tokenizer)))
}

/// Guarantee a non-empty `messages` array for hosted counting. The
/// reference's hosted counters treat an EMPTY messages array like no
/// messages (`if not messages: return None`) — counting falls back to the
/// local estimate instead of offering the provider a zero-message body it
/// must reject. A prompt is wrapped into a user turn only when TRUTHY,
/// matching `bool(prompt)`. Returns `false` when neither is available.
#[cfg(feature = "platform-apis")]
fn ensure_count_messages(input: &mut serde_json::Value) -> bool {
    use serde_json::{Value, json};
    let has_empty_messages = input
        .get("messages")
        .and_then(Value::as_array)
        .is_some_and(|messages| messages.is_empty());
    if input.get("messages").and_then(Value::as_array).is_none() || has_empty_messages {
        let Some(prompt) = input
            .get("prompt")
            .and_then(Value::as_str)
            .filter(|prompt| !prompt.is_empty())
            .map(str::to_string)
        else {
            return false;
        };
        input["messages"] = json!([{"role":"user","content":prompt}]);
    }
    true
}

/// Forward a prepared count body through the hooked passthrough pipeline
/// and return the parsed provider reply. Provider errors surface with
/// their own (bounded, sanitized) detail and status, never a silent
/// local fallback — a caller who asked for endpoint counting needs to
/// know the endpoint failed.
#[cfg(feature = "platform-apis")]
async fn forward_count_tokens(
    state: &crate::state::AppState,
    auth: &AuthUser,
    model: &str,
    body: serde_json::Value,
) -> Result<serde_json::Value, ApiError> {
    use crate::handlers::anthropic_common::sanitise_forwarded_error_message;
    use serde_json::Value;
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
            Bytes::from(body.to_string()),
            Some("application/json"),
        ),
    )
    .await?;
    if !response.status().is_success() {
        // Preserve the provider's own error detail (bounded, sanitized) so
        // the caller can tell a malformed request from a bad credential —
        // the reference surfaces the upstream message too.
        let status = response.status().as_u16();
        let headers = response.headers().clone();
        let bytes =
            axum::body::to_bytes(response.into_body(), crate::security::MAX_ERROR_BODY_BYTES)
                .await
                .unwrap_or_default();
        let upstream_message = serde_json::from_slice::<Value>(&bytes)
            .ok()
            .and_then(|body| {
                // AWS JSON errors use capital `Message`; Anthropic-style
                // envelopes use error.message / message.
                body.pointer("/error/message")
                    .or_else(|| body.pointer("/message"))
                    .or_else(|| body.pointer("/Message"))
                    .and_then(Value::as_str)
                    .map(str::to_string)
            })
            .unwrap_or_default();
        let message = if upstream_message.trim().is_empty() {
            "provider token counting failed".to_string()
        } else {
            format!(
                "provider token counting failed: {}",
                sanitise_forwarded_error_message(&upstream_message)
            )
        };
        // ApiError forwards only its vetted headers and clamps retry-after.
        return Err(ApiError::upstream(status, message).with_headers(Some(headers)));
    }
    let bytes = axum::body::to_bytes(
        response.into_body(),
        crate::security::MAX_RESPONSE_BODY_BYTES,
    )
    .await
    .map_err(|_| ApiError::internal("provider token-count response could not be read"))?;
    serde_json::from_slice(&bytes)
        .map_err(|_| ApiError::internal("invalid provider token-count response"))
}

/// Bedrock hosted counting via the runtime CountTokens API. The body
/// forwarded here stays chat-shaped; the passthrough pipeline rewrites it
/// to the CountTokens Converse/InvokeModel input and routes it to
/// `/model/{id}/count-tokens` with SigV4 (see
/// `oxllm_providers::bedrock::count_tokens`). Failures surface like the
/// Anthropic counters' — a bad credential must not masquerade as a
/// successful local estimate.
#[cfg(feature = "platform-apis")]
async fn count_at_bedrock(
    state: &crate::state::AppState,
    auth: &AuthUser,
    request: &serde_json::Value,
    model: &str,
) -> Result<Option<(u64, &'static str)>, ApiError> {
    use serde_json::json;
    let mut input = request.clone();
    if !ensure_count_messages(&mut input) {
        return Ok(None);
    }
    input["model"] = json!(model);
    if let Some(object) = input.as_object_mut() {
        crate::orchestration::request_controls::copy(request, object);
    }
    let reply = forward_count_tokens(state, auth, model, input).await?;
    // The passthrough rewrites Bedrock counting replies to the Anthropic
    // {input_tokens} shape for native clients; accept either spelling.
    let tokens = reply
        .get("input_tokens")
        .and_then(serde_json::Value::as_u64)
        .or_else(|| oxllm_providers::bedrock::count_tokens::parse_count_tokens_response(&reply))
        .ok_or_else(|| {
            ApiError::internal("provider token-count response is missing inputTokens")
        })?;
    Ok(Some((
        tokens,
        oxllm_providers::bedrock::count_tokens::TOKENIZER_TYPE,
    )))
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

    #[tokio::test]
    async fn native_count_tokens_bedrock_profile_uses_foundation_url_and_anthropic_reply() {
        // CountTokens requires the foundation id even when the deployment uses
        // an inference profile. Its {"inputTokens": n} reply must still be
        // relayed in the Anthropic {"input_tokens": n} shape SDKs parse.
        let upstream = MockServer::start_async().await;
        let state = bedrock_state(&upstream.url(""));
        let mock = upstream
            .mock_async(|when, then| {
                when.method(httpmock::Method::POST)
                    .path("/model/anthropic.claude-sonnet-4-20250514-v1:0/count-tokens");
                then.status(200).json_body(json!({"inputTokens": 7}));
            })
            .await;
        let input = json!({"model": "counted", "messages": [{"role": "user", "content": "hello"}]});
        let response = count_tokens(
            State(state),
            Extension(AuthUser::open()),
            HeaderMap::new(),
            Bytes::from(input.to_string()),
        )
        .await
        .unwrap();
        assert_eq!(response.status().as_u16(), 200);
        let bytes = axum::body::to_bytes(response.into_body(), 1024)
            .await
            .unwrap();
        let reply: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(reply, json!({"input_tokens": 7}));
        mock.assert_hits_async(1).await;
    }

    #[tokio::test]
    async fn native_count_tokens_preserves_provider_error_body_status_and_headers() {
        let body = r#"{ "type": "error", "error": { "type": "api_error", "message": "invalid sk-synthetic-native-key Bearer synthetic-native-token at https://synthetic.invalid/count" }, "request_id": "synthetic-provider-id" }
"#;
        for (model, base_path, path, _, _, _) in HOSTED_COUNTERS {
            for status in [401, 429, 503] {
                let upstream = MockServer::start_async().await;
                let state = state(model, &upstream.url(base_path));
                let mock = upstream
                    .mock_async(|when, then| {
                        when.method(httpmock::Method::POST).path(path);
                        then.status(status)
                            .header("content-type", "application/json")
                            .header("retry-after", "7")
                            .header("x-ratelimit-remaining-tokens", "19")
                            .header("x-request-id", "synthetic-provider-id")
                            .body(body);
                    })
                    .await;
                let input =
                    json!({"model": "counted", "messages": [{"role": "user", "content": "hello"}]});
                let response = crate::pipeline::HTTP_REQUEST_ID
                    .scope(
                        "synthetic-native-count".to_string(),
                        count_tokens(
                            State(state),
                            Extension(AuthUser::open()),
                            HeaderMap::new(),
                            Bytes::from(input.to_string()),
                        ),
                    )
                    .await
                    .unwrap();

                assert_eq!(response.status().as_u16(), status, "{model}");
                assert_eq!(
                    response.headers().get("content-type").unwrap(),
                    "application/json"
                );
                assert_eq!(response.headers().get("retry-after").unwrap(), "7");
                assert_eq!(
                    response
                        .headers()
                        .get("x-ratelimit-remaining-tokens")
                        .unwrap(),
                    "19"
                );
                for name in [
                    crate::pipeline::REQUEST_ID_HEADER,
                    crate::pipeline::STANDARD_REQUEST_ID_HEADER,
                    crate::pipeline::LITELLM_CALL_ID_HEADER,
                ] {
                    assert_eq!(
                        response.headers().get(name).unwrap(),
                        "synthetic-native-count"
                    );
                }
                let bytes = axum::body::to_bytes(response.into_body(), 1024)
                    .await
                    .unwrap();
                assert_eq!(bytes.as_ref(), body.as_bytes(), "{model}: {status}");
                mock.assert_hits_async(1).await;
            }
        }
    }

    fn bedrock_state(api_base: &str) -> SharedState {
        let config = crate::config::parse_config_str(&format!(
            r#"
model_list:
  - model_name: counted
    oxllm_params:
      model: bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0
      api_base: {api_base}
      aws_access_key_id: AKIATESTKEY
      aws_secret_access_key: synthetic-secret
      aws_region: us-east-1
      use_converse_api: true
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

    #[cfg(feature = "platform-apis")]
    #[tokio::test]
    async fn utility_count_tokens_uses_bedrock_count_tokens_api() {
        let upstream = MockServer::start_async().await;
        let state = bedrock_state(&upstream.url(""));
        let mock = upstream
            .mock_async(|when, then| {
                when.method(httpmock::Method::POST)
                    .path("/model/anthropic.claude-sonnet-4-20250514-v1:0/count-tokens")
                    .header_exists("authorization")
                    .json_body_partial(
                        r#"{"input":{"converse":{"messages":[{"role":"user","content":[{"text":"hello"}]}]}}}"#,
                    );
                then.status(200).json_body(json!({"inputTokens": 42}));
            })
            .await;
        let input = json!({"model":"counted","messages":[{"role":"user","content":"hello"}]});
        let counted = count_at_provider(&state.load_full(), &AuthUser::open(), &input, "counted")
            .await
            .unwrap();
        assert_eq!(counted, Some((42, "bedrock_api")));
        mock.assert_hits_async(1).await;
    }

    #[cfg(feature = "platform-apis")]
    #[tokio::test]
    async fn utility_count_tokens_bedrock_error_surfaces_instead_of_falling_back() {
        // The bug this wiring fixes: a Bedrock deployment used to skip
        // hosted counting entirely, so bad credentials silently produced a
        // successful local estimate.
        let upstream = MockServer::start_async().await;
        let state = bedrock_state(&upstream.url(""));
        let mock = upstream
            .mock_async(|when, then| {
                when.method(httpmock::Method::POST)
                    .path("/model/anthropic.claude-sonnet-4-20250514-v1:0/count-tokens");
                then.status(403).json_body(
                    json!({"message": "The security token included in the request is invalid"}),
                );
            })
            .await;
        let input = json!({"model":"counted","messages":[{"role":"user","content":"hello"}]});
        let error = count_at_provider(&state.load_full(), &AuthUser::open(), &input, "counted")
            .await
            .expect_err("bedrock 403 must surface");
        assert_eq!(error.status.as_u16(), 403);
        assert!(
            error.message.contains("security token"),
            "provider detail preserved: {}",
            error.message
        );
        mock.assert_hits_async(1).await;
    }

    #[cfg(feature = "platform-apis")]
    #[tokio::test]
    async fn utility_count_tokens_prompt_handling_matches_reference_truthiness() {
        // The provider is offered a counted body only when there is something
        // to count: an empty prompt or an empty messages array falls back to
        // the local estimate (the reference's hosted counters return None for
        // falsy messages and wrap only `bool(prompt)` truths), while a truthy
        // prompt is wrapped into a user turn and actually sent.
        for (model, base_path, path, _, _, _) in HOSTED_COUNTERS {
            for (input, expected_calls) in [
                (json!({"model": "counted", "prompt": ""}), 0),
                (json!({"model": "counted"}), 0),
                (json!({"model": "counted", "messages": []}), 0),
                (json!({"model": "counted", "messages": [], "prompt": ""}), 0),
                (json!({"model": "counted", "prompt": "hello"}), 1),
                (
                    json!({"model": "counted", "messages": [], "prompt": "hello"}),
                    1,
                ),
            ] {
                let upstream = MockServer::start_async().await;
                let state = state(model, &upstream.url(base_path));
                let mock = upstream
                    .mock_async(|when, then| {
                        when.method(httpmock::Method::POST).path(path);
                        then.status(200).json_body(json!({"input_tokens": 7}));
                    })
                    .await;

                let result =
                    count_at_provider(&state.load_full(), &AuthUser::open(), &input, "counted")
                        .await
                        .unwrap_or_else(|error| panic!("{model} {input}: {error:?}"));

                if expected_calls == 0 {
                    assert_eq!(
                        result, None,
                        "{model} {input}: nothing countable must not call the provider"
                    );
                } else {
                    let (tokens, _) = result.unwrap_or_else(|| {
                        panic!("{model} {input}: a truthy prompt must be counted upstream")
                    });
                    assert_eq!(tokens, 7, "{model} {input}");
                }
                mock.assert_hits_async(expected_calls).await;
            }
        }
    }

    #[cfg(feature = "platform-apis")]
    #[tokio::test]
    async fn utility_count_tokens_redacts_provider_error_detail() {
        let detail = "invalid credential sk-synthetic-count-key Bearer \t \tsynthetic-count-token at https://synthetic.invalid/count?key=synthetic";
        for (model, base_path, path, _, _, _) in HOSTED_COUNTERS {
            for body in [
                json!({"error": {"message": detail}}),
                json!({"message": detail}),
            ] {
                let upstream = MockServer::start_async().await;
                let state = state(model, &upstream.url(base_path));
                let mock = upstream
                    .mock_async(|when, then| {
                        when.method(httpmock::Method::POST).path(path);
                        then.status(401).json_body(body);
                    })
                    .await;

                let error = count_at_provider(
                    &state.load_full(),
                    &AuthUser::open(),
                    &json!({"model": "counted", "messages": [{"role": "user", "content": "hello"}]}),
                    "counted",
                )
                .await
                .expect_err("provider rejection must reach the utility caller");

                assert_eq!(
                    error.status,
                    axum::http::StatusCode::UNAUTHORIZED,
                    "{model}"
                );
                assert_eq!(
                    error.message,
                    "provider token counting failed: invalid credential [redacted] bearer [redacted] at [redacted-url]",
                    "{model}"
                );
                mock.assert_hits_async(1).await;
            }
        }
    }

    #[cfg(feature = "platform-apis")]
    #[tokio::test]
    async fn utility_count_tokens_preserves_provider_error_status_and_vetted_headers() {
        use axum::response::IntoResponse;

        for (status, error_type, code, retry_after, expected_retry_after) in [
            (400, "invalid_request_error", None, "7", "7"),
            (
                401,
                "invalid_request_error",
                Some("upstream_authentication_error"),
                "7",
                "7",
            ),
            (
                429,
                "rate_limit_error",
                Some("upstream_rate_limited"),
                "7",
                "7",
            ),
            (
                429,
                "rate_limit_error",
                Some("upstream_rate_limited"),
                "18446744073709551616",
                "3600",
            ),
            (
                429,
                "rate_limit_error",
                Some("upstream_rate_limited"),
                "Wed, 21 Oct 2026 07:28:00 GMT",
                "Wed, 21 Oct 2026 07:28:00 GMT",
            ),
            (503, "api_error", Some("upstream_error"), "99999999", "3600"),
            (529, "api_error", Some("upstream_error"), "7", "7"),
        ] {
            let upstream = MockServer::start_async().await;
            let state = state("anthropic/claude-sonnet-5-5", &upstream.url("/v1"));
            let mock = upstream
                .mock_async(|when, then| {
                    when.method(httpmock::Method::POST)
                        .path("/v1/messages/count_tokens");
                    then.status(status)
                        .header("retry-after", retry_after)
                        .header("x-ratelimit-remaining-requests", "0")
                        .header("x-request-id", "synthetic-provider-id")
                        .header("x-vendor-internal", "synthetic-private-detail")
                        .header("authorization", "Bearer synthetic-header-token")
                        .header("x-api-key", "sk-synthetic-header-key")
                        .header("set-cookie", "synthetic-cookie=value")
                        .json_body(json!({"error": {"message": "model is overloaded"}}));
                })
                .await;
            let input =
                json!({"model": "counted", "messages": [{"role": "user", "content": "hello"}]});
            let error = crate::pipeline::HTTP_REQUEST_ID
                .scope(
                    "synthetic-count-request".to_string(),
                    count_at_provider(&state.load_full(), &AuthUser::open(), &input, "counted"),
                )
                .await
                .expect_err("provider rejection must reach the utility caller");

            assert_eq!(
                error.message,
                "provider token counting failed: model is overloaded"
            );
            let response = error.into_response();
            assert_eq!(response.status().as_u16(), status);
            assert_eq!(
                response.headers().get("retry-after").unwrap(),
                expected_retry_after
            );
            assert_eq!(
                response
                    .headers()
                    .get("x-ratelimit-remaining-requests")
                    .unwrap(),
                "0"
            );
            assert_eq!(
                response.headers().get("x-request-id").unwrap(),
                "synthetic-count-request"
            );
            for name in [
                "x-vendor-internal",
                "authorization",
                "x-api-key",
                "set-cookie",
            ] {
                assert!(response.headers().get(name).is_none(), "{name}");
            }
            let bytes = axum::body::to_bytes(response.into_body(), 1024)
                .await
                .unwrap();
            let body: Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(
                body,
                json!({"error": {
                    "message": "provider token counting failed: model is overloaded",
                    "type": error_type, "param": null, "code": code,
                }})
            );
            mock.assert_hits_async(1).await;
        }
    }

    #[cfg(feature = "platform-apis")]
    #[tokio::test]
    async fn utility_count_tokens_falls_back_for_oversized_or_truncated_provider_errors() {
        use axum::response::IntoResponse;

        for length in [600, crate::security::MAX_ERROR_BODY_BYTES * 2] {
            let upstream = MockServer::start_async().await;
            let state = state("anthropic/claude-sonnet-5-5", &upstream.url("/v1"));
            let body = json!({"error": {"message": format!(
                "sk-synthetic-long-key Bearer synthetic-long-token https://synthetic.invalid/count {}",
                "é".repeat(length),
            )}}).to_string();
            let mock = upstream
                .mock_async(|when, then| {
                    when.method(httpmock::Method::POST)
                        .path("/v1/messages/count_tokens");
                    then.status(503)
                        .header("retry-after", "9")
                        .header("content-type", "application/json")
                        .body(body);
                })
                .await;
            let error = count_at_provider(
                &state.load_full(),
                &AuthUser::open(),
                &json!({"model": "counted", "messages": [{"role": "user", "content": "hello"}]}),
                "counted",
            )
            .await
            .expect_err("truncated provider JSON must not become a local estimate");

            // The existing passthrough limits truncate error JSON before this
            // handler parses it. Never echo that partial body as a fallback.
            assert_eq!(error.message, "provider token counting failed", "{length}");
            let response = error.into_response();
            assert_eq!(
                response.status(),
                axum::http::StatusCode::SERVICE_UNAVAILABLE
            );
            assert_eq!(response.headers().get("retry-after").unwrap(), "9");
            mock.assert_hits_async(1).await;
        }
    }

    #[cfg(feature = "platform-apis")]
    #[tokio::test]
    async fn utility_count_tokens_falls_back_for_malformed_or_non_json_provider_errors() {
        for body in [
            "not JSON: sk-synthetic-text-key Bearer synthetic-text-token https://synthetic.invalid/count",
            r#"{"error":{"message":"sk-synthetic-truncated-key"#,
            r#"{"error":{"detail":"sk-synthetic-unrecognized-key"}}"#,
            r#"{"error":{"message":42}}"#,
            r#"{"message":" \t\n"}"#,
            "",
        ] {
            let upstream = MockServer::start_async().await;
            let state = state("anthropic/claude-sonnet-5-5", &upstream.url("/v1"));
            let mock = upstream
                .mock_async(|when, then| {
                    when.method(httpmock::Method::POST)
                        .path("/v1/messages/count_tokens");
                    then.status(403).body(body);
                })
                .await;
            let error = count_at_provider(
                &state.load_full(),
                &AuthUser::open(),
                &json!({"model": "counted", "messages": [{"role": "user", "content": "hello"}]}),
                "counted",
            )
            .await
            .expect_err("unusable provider detail must use a safe generic error");

            assert_eq!(error.status, axum::http::StatusCode::FORBIDDEN);
            assert_eq!(error.message, "provider token counting failed");
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
