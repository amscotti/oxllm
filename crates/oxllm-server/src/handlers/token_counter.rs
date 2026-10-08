//! `POST /utils/token_counter` — server-only token estimation.
//!
//! Computes the prompt token count for the supplied model + messages locally
//! using `crate::tokens::estimate_token_counter_tokens_async` (which dispatches to
//! `oxllm_core::estimate_request_tokens`, which in turn uses `tiktoken-rs` for
//! OpenAI models and falls back to `cl100k_base` for everything else — the
//! same default encoding the reference uses for unrecognised models; only if
//! `cl100k_base` itself fails to load does it degrade to a `bytes/4`
//! heuristic). A bounded number of utility jobs use the blocking pool so the
//! async runtime is not stalled by BPE encoding. No upstream call is made.

use axum::Extension;
use axum::Json;
use axum::body::Bytes;
use axum::extract::State;
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};

use crate::auth::AuthUser;
use crate::error::ApiError;
use crate::pipeline::check_model_acl;
use crate::state::SharedState;

/// Token counter handler.
///
/// Request body shape (matches the reference token-count request):
/// ```json
/// { "model": "gpt-4o", "messages": [{"role": "user", "content": "Hello"}] }
/// ```
/// A top-level `prompt` string is accepted as an alternative to `messages`
/// (the reference `text` argument). At least one of `prompt`, `messages`, or
/// `contents` must be present, otherwise the request is rejected with 400 —
/// matching the reference token-counter validation.
///
/// Response shape (reference token-count response parity):
/// ```json
/// {
/// "total_tokens": 15,
/// "prompt_tokens": 15,
/// "completion_tokens": 0,
/// "request_model": "gpt-4o",
/// "model_used": "gpt-4o",
/// "tokenizer_type": "openai_tokenizer"
/// }
/// ```
///
/// `model_used` resolves the request model's serving deployment and uses the
/// upstream model id (alias resolution, matching the reference) so the tokenizer and
/// reported `tokenizer_type` reflect the model that would actually serve the
/// request. `tokenizer_type` mirrors the reference
/// `str(_select_tokenizer(...)["type"])`, i.e. `"openai_tokenizer"` — oxllm
/// counts every model with tiktoken BPE, so it never reports
/// `"huggingface_tokenizer"` even for models the reference would route to an HF
/// tokenizer.
///
/// This endpoint performs no upstream call and therefore no spend, so it
/// intentionally does **not** run the full `EndpointPipeline` (budget
/// reservation / cache lookup would be meaningless for a free local estimate).
/// It still honours the model ACL ([`check_model_acl`]) for consistency with
/// every other LLM endpoint. Two process-wide utility slots bound CPU work;
/// saturation returns 429 immediately rather than building a blocking-work
/// queue. A slot remains held until its encoder finishes, even if the client
/// disconnects. The global `RequestBodyLimitLayer` bounds each payload. These
/// are local resource limits, independent of paid-call RPM/TPM accounting.
pub async fn token_counter(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    body: Bytes,
) -> Result<Response, ApiError> {
    let permit = crate::tokens::try_token_counter_slot()
        .map_err(|_| ApiError::rate_limited("token counter is busy; try again later"))?;
    let request: Value = serde_json::from_slice(&body)
        .map_err(|e| ApiError::bad_request(format!("invalid JSON body: {e}")))?;

    crate::security::check_json_depth(&request, crate::security::MAX_JSON_DEPTH)?;

    let model = request
        .get("model")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ApiError::bad_request("missing required field: 'model'"))?
        .to_string();

    check_model_acl(
        &auth,
        &model,
        auth.key_models.as_deref(),
        auth.team_models.as_deref(),
        auth.user_models.as_deref(),
    )?;

    // The reference rejects with 400 when none of prompt/messages/contents is present
    // (proxy_server.py:10955-10956). `contents` (Gemini shape) satisfies the
    // contract; `estimate_request_tokens` counts `parts[*].text` best-effort,
    // and the reference additionally routes through Google's `/countTokens` API when
    // `call_endpoint=True` (not wired here — local counting only).
    let has_prompt = request.get("prompt").and_then(|v| v.as_str()).is_some();
    let has_messages = request.get("messages").and_then(|v| v.as_array()).is_some();
    let has_contents = request.get("contents").and_then(|v| v.as_array()).is_some();
    if !has_prompt && !has_messages && !has_contents {
        return Err(ApiError::bad_request(
            "prompt, messages, or contents must be provided",
        ));
    }

    // Resolve the deployment's upstream model id (alias resolution, matching
    // the reference): an alias mapping to `openai/gpt-4o` is counted with `gpt-4o`'
    // tokenizer. Falls back to the request model when no deployment exists.
    let state = shared.load_full();
    let model_used: String = state
        .router
        .deployments_for_model(&state.deployments, &model)
        .into_iter()
        .next()
        .map(|d| {
            d.oxllm_params
                .model
                .split_once('/')
                .map(|(_, rest)| rest.to_string())
                .unwrap_or_else(|| d.oxllm_params.model.clone())
        })
        .unwrap_or_else(|| model.clone());
    let tokenizer_type = oxllm_core::tokenizer_type_for(&model_used);

    // Route the estimator through the resolved model so the count uses the same
    // tokenizer `model_used` / `tokenizer_type` report. The body is only used
    // for this local estimate, so mutating its `model` field is safe.
    let mut count_body = request;
    if model_used != model {
        count_body["model"] = json!(model_used);
    }
    // The reference local fallback calls `token_counter(model, text, messages,
    // custom_tokenizer)` — `tools`/`tool_choice` are only counted when
    // `call_endpoint=True` routes to a provider counting API
    // (proxy_server.py:11698-11704). Strip them so this free informational
    // endpoint reports the same total an existing client expects. The billing
    // pre-call estimate (TPM/budget hooks) keeps counting tools — there the
    // over-estimate is the safe direction.
    if let Some(obj) = count_body.as_object_mut() {
        obj.remove("tools");
        obj.remove("tool_choice");
    }
    let prompt_tokens =
        crate::tokens::estimate_token_counter_tokens_async(count_body, permit).await;

    let body = json!({
    "total_tokens": prompt_tokens,
    "prompt_tokens": prompt_tokens,
    "completion_tokens": 0,
    "request_model": model,
    "model_used": model_used,
    "tokenizer_type": tokenizer_type,
    });

    Ok(Json(body).into_response())
}

#[cfg(test)]
mod tests {
    use axum::Extension;
    use axum::body::Bytes;
    use axum::extract::State;
    use axum::http::StatusCode;
    use serde_json::json;

    use super::token_counter;
    use crate::auth::AuthUser;
    use crate::state::SharedState;

    fn shared_state() -> SharedState {
        shared_state_with("gpt-4o", "openai/gpt-4o")
    }

    fn shared_state_with(model_name: &str, upstream: &str) -> SharedState {
        let config = crate::config::types::ProxyConfig {
            model_list: vec![crate::config::types::ConfigModel {
                model_name: model_name.to_string(),
                oxllm_params: oxllm_core::deployment::OxllmParams {
                    model: upstream.to_string(),
                    api_key: Some("k".to_string()),
                    ..Default::default()
                },
                ..Default::default()
            }],
            oxllm_settings: crate::config::types::OxllmSettings::default(),
            general_settings: crate::config::types::GeneralSettings::default(),
            router_settings: Default::default(),
            extra: serde_json::Map::new(),
        };
        let state =
            crate::state::AppState::with_registry(config, oxllm_providers::default_registry())
                .expect("state built");
        std::sync::Arc::new(arc_swap::ArcSwap::from_pointee(state))
    }

    #[test]
    fn token_counter_counts_simple_messages() {
        let req = json!({
        "model": "gpt-4o",
        "messages": [
        {"role": "user", "content": "Hello world"}
        ]
        });
        let n = oxllm_core::estimate_request_tokens(&req);
        assert!(n > 0, "expected nonzero token estimate, got {n}");
    }

    #[test]
    fn token_counter_empty_messages_returns_reply_priming() {
        let req = json!({"model": "gpt-4o", "messages": []});
        // Token-count parity: even an empty messages list takes the `messages`
        // branch, so `_count_extra`'s unconditional `+3` reply priming applies
        // (`_count_messages` itself returns 0 for an empty list).
        assert_eq!(oxllm_core::estimate_request_tokens(&req), 3);
    }

    #[tokio::test]
    #[serial_test::serial(token_counter)]
    async fn token_counter_enforces_model_acl() {
        // A virtual key restricted to gpt-4o must NOT estimate tokens for a
        // disallowed model — the endpoint honours the same ACL as every other
        // LLM route, even though it performs no upstream call.
        let shared = shared_state();
        let auth = AuthUser {
            key_models: Some(vec!["gpt-4o".to_string()]),
            ..AuthUser::open()
        };
        let body = Bytes::from(r#"{"model":"claude-3","messages":[]}"#);
        let err = token_counter(State(shared), Extension(auth), body)
            .await
            .expect_err("disallowed model must be rejected");
        assert_eq!(err.status, StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    #[serial_test::serial(token_counter)]
    async fn token_counter_allows_model_on_acl() {
        // The positive counterpart: a key restricted to gpt-4o estimating
        // gpt-4o succeeds (the ACL does not over-block the local estimator).
        let shared = shared_state();
        let auth = AuthUser {
            key_models: Some(vec!["gpt-4o".to_string()]),
            ..AuthUser::open()
        };
        let body = Bytes::from(r#"{"model":"gpt-4o","messages":[]}"#);
        let resp = token_counter(State(shared), Extension(auth), body)
            .await
            .expect("allowed model must succeed");
        assert_eq!(resp.status(), StatusCode::OK);
    }

    async fn read_json(resp: axum::response::Response) -> serde_json::Value {
        let bytes = axum::body::to_bytes(resp.into_body(), 65536)
            .await
            .expect("body");
        serde_json::from_slice(&bytes).expect("json")
    }

    // A compatible client sending only a top-level `prompt` string must
    // receive a non-zero count (previously the prompt field was ignored and the
    // endpoint returned 0 silently).
    #[tokio::test]
    #[serial_test::serial(token_counter)]
    async fn token_counter_counts_prompt_only_field() {
        let shared = shared_state();
        let body = Bytes::from(r#"{"model":"gpt-4o","prompt":"hello world this is a test"}"#);
        let resp = token_counter(State(shared), Extension(AuthUser::open()), body)
            .await
            .expect("prompt-only request must succeed");
        assert_eq!(resp.status(), StatusCode::OK);
        let v = read_json(resp).await;
        let total = v["total_tokens"].as_u64().expect("total_tokens");
        assert!(total > 0, "prompt-only body must count tokens, got {total}");
        assert_eq!(v["request_model"], "gpt-4o");
        assert_eq!(v["model_used"], "gpt-4o");
        assert_eq!(v["tokenizer_type"], "openai_tokenizer");
    }

    // Matching the reference, a body with none of prompt/messages/contents is rejected
    // with 400 instead of silently returning total_tokens: 0.
    #[tokio::test]
    #[serial_test::serial(token_counter)]
    async fn token_counter_rejects_body_without_prompt_or_messages() {
        let shared = shared_state();
        let body = Bytes::from(r#"{"model":"gpt-4o"}"#);
        let err = token_counter(State(shared), Extension(AuthUser::open()), body)
            .await
            .expect_err("missing prompt/messages must be rejected");
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
    }

    // `contents` satisfies the "at least one field" contract, so it must
    // not be rejected with 400. The handler now counts `parts[*].text`
    // best-effort (matching the local-count arms for `input`/`prompt`), so a
    // contents-only body with text parts reports a non-zero `total_tokens`.
    #[tokio::test]
    #[serial_test::serial(token_counter)]
    async fn token_counter_accepts_contents_only_body() {
        let shared = shared_state();
        let body = Bytes::from(
            r#"{"model":"gpt-4o","contents":[{"role":"user","parts":[{"text":"hello world"}]}]}"#,
        );
        let resp = token_counter(State(shared), Extension(AuthUser::open()), body)
            .await
            .expect("contents-only body must be accepted");
        assert_eq!(resp.status(), StatusCode::OK);
        let v = read_json(resp).await;
        let total = v["total_tokens"].as_u64().expect("total_tokens");
        assert!(
            total > 0,
            "contents text parts must be counted locally, got {total}"
        );
    }

    // A `contents` block carrying only non-text parts (e.g. inline_data) is
    // accepted (the contract is satisfied) but estimates to 0 locally because
    // binary parts do not map to BPE tokens. The reference routes such bodies through
    // Google's `/countTokens` endpoint when `call_endpoint=True`; oxllm's local
    // estimator intentionally skips non-text parts. This pins the documented
    // behaviour so a future change is caught.
    #[tokio::test]
    #[serial_test::serial(token_counter)]
    async fn token_counter_contents_with_no_text_parts_estimates_zero() {
        let shared = shared_state();
        let body = Bytes::from(
            r#"{"model":"gpt-4o","contents":[{"role":"user","parts":[{"inline_data":{"mime_type":"image/png","data":"base64"}}]}]}"#,
        );
        let resp = token_counter(State(shared), Extension(AuthUser::open()), body)
            .await
            .expect("contents-only body must be accepted");
        assert_eq!(resp.status(), StatusCode::OK);
        let v = read_json(resp).await;
        assert_eq!(
            v["total_tokens"].as_u64(),
            Some(0),
            "non-text parts must not contribute to the local estimate"
        );
    }

    // Alias resolution (reference parity): a client alias mapping to
    // `openai/gpt-4o` reports `model_used` = upstream `gpt-4o` and the
    // `o200k_base` tokenizer, while `request_model` stays the alias.
    #[tokio::test]
    #[serial_test::serial(token_counter)]
    async fn token_counter_resolves_alias_for_model_used() {
        let shared = shared_state_with("fast", "openai/gpt-4o");
        let body =
            Bytes::from(r#"{"model":"fast","prompt":"hello world this is a test sentence"}"#);
        let resp = token_counter(State(shared), Extension(AuthUser::open()), body)
            .await
            .expect("alias request must succeed");
        assert_eq!(resp.status(), StatusCode::OK);
        let v = read_json(resp).await;
        assert_eq!(v["request_model"], "fast", "request_model is the alias");
        assert_eq!(
            v["model_used"], "gpt-4o",
            "model_used resolves to the stripped upstream model id"
        );
        assert_eq!(v["tokenizer_type"], "openai_tokenizer");
        assert!(
            v["total_tokens"].as_u64().unwrap_or(0) > 0,
            "alias must be counted with the upstream tokenizer"
        );
    }

    // A non-array `messages` value must not silently zero the count when a
    // `prompt` is also present: the handler accepts the body (has_prompt) and
    // the estimator must fall through to the prompt arm instead of gating on
    // mere `messages` key presence.
    #[tokio::test]
    #[serial_test::serial(token_counter)]
    async fn token_counter_malformed_messages_still_counts_prompt() {
        let shared = shared_state();
        let body = Bytes::from(
            r#"{"model":"gpt-4o","messages":"not an array","prompt":"hello world this is a test"}"#,
        );
        let resp = token_counter(State(shared), Extension(AuthUser::open()), body)
            .await
            .expect("prompt-bearing body must succeed");
        assert_eq!(resp.status(), StatusCode::OK);
        let v = read_json(resp).await;
        let total = v["total_tokens"].as_u64().expect("total_tokens");
        assert!(
            total > 0,
            "malformed messages must not suppress the prompt count, got {total}"
        );
    }

    // The reference local token-count fallback ignores `tools`/`tool_choice`
    // (they are only counted on the `call_endpoint=True` provider path), so
    // this endpoint must report the same total with and without a `tools`
    // array. The billing pre-call estimate deliberately still counts tools.
    #[tokio::test]
    #[serial_test::serial(token_counter)]
    async fn token_counter_ignores_tools_like_litellm_local_fallback() {
        let shared = shared_state();
        let without = Bytes::from(
            r#"{"model":"gpt-4o","messages":[{"role":"user","content":"weather in paris?"}]}"#,
        );
        let with = Bytes::from(
            r#"{"model":"gpt-4o","messages":[{"role":"user","content":"weather in paris?"}],"tools":[{"type":"function","function":{"name":"get_weather","description":"Get the current weather for a city.","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],"tool_choice":"none"}"#,
        );
        let a = read_json(
            token_counter(State(shared.clone()), Extension(AuthUser::open()), without)
                .await
                .expect("plain body must succeed"),
        )
        .await;
        let b = read_json(
            token_counter(State(shared), Extension(AuthUser::open()), with)
                .await
                .expect("tool-bearing body must succeed"),
        )
        .await;
        assert_eq!(
            a["total_tokens"], b["total_tokens"],
            "tools/tool_choice must not change the local count (LiteLLM parity)"
        );
    }

    // A non-object JSON body (array/string/number) has no `model` field and
    // must be rejected with 400 rather than panicking or estimating.
    #[tokio::test]
    #[serial_test::serial(token_counter)]
    async fn token_counter_rejects_non_object_bodies() {
        for body in ["[1,2,3]", "\"just a string\"", "42"] {
            let err = token_counter(
                State(shared_state()),
                Extension(AuthUser::open()),
                Bytes::from(body),
            )
            .await
            .expect_err("non-object body must be rejected");
            assert_eq!(
                err.status,
                StatusCode::BAD_REQUEST,
                "body {body} must 400 via the missing-model check"
            );
        }
    }

    #[tokio::test]
    #[serial_test::serial(token_counter)]
    async fn token_counter_rejects_overload_and_recovers_after_release() {
        let mut held = Vec::new();
        while let Ok(permit) = crate::tokens::try_token_counter_slot() {
            held.push(permit);
        }
        assert!(!held.is_empty(), "test must occupy the utility work slots");
        let shared = shared_state();
        let body = Bytes::from(r#"{"model":"gpt-4o","prompt":"hello"}"#);
        let err = token_counter(
            State(shared.clone()),
            Extension(AuthUser::open()),
            body.clone(),
        )
        .await
        .expect_err("a saturated utility pool must reject rather than queue work");
        assert_eq!(err.status, StatusCode::TOO_MANY_REQUESTS);

        drop(held);
        let resp = token_counter(State(shared), Extension(AuthUser::open()), body)
            .await
            .expect("released slots must allow new estimates");
        assert!(read_json(resp).await["prompt_tokens"].as_u64().unwrap() > 0);
    }
}
