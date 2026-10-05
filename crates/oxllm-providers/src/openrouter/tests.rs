use super::*;
use oxllm_core::Endpoint;
use oxllm_core::deployment::{Deployment, OxllmParams};
use oxllm_core::provider::{ChatProvider, EmbeddingProvider, Provider, ResolvedCredentials};
use serde_json::json;

fn deployment() -> Deployment {
    Deployment {
        model_name: "openai/gpt-4o".to_string(),
        oxllm_params: OxllmParams {
            model: "openrouter/openai/gpt-4o".to_string(),
            api_key: Some("openrouter-key".to_string()),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn deployment_with_extra(referer: Option<&str>, title: Option<&str>) -> Deployment {
    let mut dep = deployment();
    if let Some(r) = referer {
        dep.oxllm_params
            .extra
            .insert("http_referer".to_string(), json!(r));
    }
    if let Some(t) = title {
        dep.oxllm_params
            .extra
            .insert("x_title".to_string(), json!(t));
    }
    dep
}

#[test]
fn endpoint_url_uses_default_openrouter_base() {
    let p = OpenRouterProvider;
    assert_eq!(
        p.endpoint_url(&deployment()),
        "https://openrouter.ai/api/v1/chat/completions"
    );
}

#[test]
fn embedding_url_uses_default_openrouter_base() {
    let p = OpenRouterProvider;
    assert_eq!(
        p.embedding_url(&deployment()),
        "https://openrouter.ai/api/v1/embeddings"
    );
}

#[test]
fn endpoint_url_respects_custom_base() {
    let p = OpenRouterProvider;
    let mut dep = deployment();
    dep.oxllm_params.api_base = Some("https://custom.example.com/v1/".to_string());
    assert_eq!(
        p.endpoint_url(&dep),
        "https://custom.example.com/v1/chat/completions"
    );
}

#[test]
fn endpoint_url_does_not_double_append_chat_completions() {
    // MEDIUM: a caller who configures `api_base` as the full chat endpoint
    // URL must not get `/chat/completions` appended twice (mirrors the
    // `openai_compat` sibling's guard and LiteLLM's `get_complete_url`).
    let p = OpenRouterProvider;
    let mut dep = deployment();
    dep.oxllm_params.api_base = Some("https://example.org/api/v1/chat/completions".to_string());
    assert_eq!(
        p.endpoint_url(&dep),
        "https://example.org/api/v1/chat/completions"
    );
}

#[test]
fn embedding_url_does_not_double_append_embeddings() {
    // MEDIUM: a caller who configures `api_base` as the full embeddings
    // endpoint URL must not get `/embeddings` appended twice.
    let p = OpenRouterProvider;
    let mut dep = deployment();
    dep.oxllm_params.api_base = Some("https://example.org/api/v1/embeddings".to_string());
    assert_eq!(
        p.embedding_url(&dep),
        "https://example.org/api/v1/embeddings"
    );
}

#[test]
fn apply_auth_uses_bearer_token() {
    let p = OpenRouterProvider;
    let creds = ResolvedCredentials::from_key("openrouter-secret");
    let mut headers = std::collections::HashMap::new();
    ChatProvider::apply_auth(&p, &deployment(), &creds, &mut headers);
    assert_eq!(
        headers.get("authorization").map(String::as_str),
        Some("Bearer openrouter-secret")
    );
    assert!(
        !headers.contains_key("http-referer"),
        "no referer should be set when extra is empty"
    );
    assert!(
        !headers.contains_key("x-title"),
        "no x-title should be set when extra is empty"
    );
}

#[test]
fn apply_auth_forwards_referer_and_title_from_extra() {
    let p = OpenRouterProvider;
    let dep = deployment_with_extra(Some("https://myapp.example.com"), Some("My App"));
    let creds = ResolvedCredentials::from_key("openrouter-secret");
    let mut headers = std::collections::HashMap::new();
    ChatProvider::apply_auth(&p, &dep, &creds, &mut headers);
    assert_eq!(
        headers.get("http-referer").map(String::as_str),
        Some("https://myapp.example.com")
    );
    assert_eq!(headers.get("x-title").map(String::as_str), Some("My App"));
}

#[test]
fn apply_auth_forwards_only_referer_when_only_referer_present() {
    let p = OpenRouterProvider;
    let dep = deployment_with_extra(Some("https://myapp.example.com"), None);
    let creds = ResolvedCredentials::from_key("openrouter-secret");
    let mut headers = std::collections::HashMap::new();
    ChatProvider::apply_auth(&p, &dep, &creds, &mut headers);
    assert_eq!(
        headers.get("http-referer").map(String::as_str),
        Some("https://myapp.example.com")
    );
    assert!(!headers.contains_key("x-title"));
}

#[test]
fn request_strips_only_first_segment_for_openrouter_model() {
    let p = OpenRouterProvider;
    let req = json!({
    "model": "openai/gpt-4o",
    "messages": [{"role": "user", "content": "Hi"}],
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["model"], "openai/gpt-4o");
    assert_eq!(out["messages"][0]["content"], "Hi");
}

#[test]
fn request_preserves_stream_flag() {
    let p = OpenRouterProvider;
    let req = json!({
    "model": "openai/gpt-4o",
    "messages": [{"role": "user", "content": "Hi"}],
    "stream": true,
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["stream"], true);
}

#[test]
fn embedding_request_strips_only_first_segment() {
    let p = OpenRouterProvider;
    let req = json!({
    "model": "openai/gpt-4o",
    "input": "hello world",
    });
    let out = EmbeddingProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["model"], "openai/gpt-4o");
    assert_eq!(out["input"], "hello world");
}

#[test]
fn response_is_identity() {
    let p = OpenRouterProvider;
    let resp = json!({
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    });
    let out = ChatProvider::transform_response(&p, &resp, "openai/gpt-4o").expect("ok");
    assert_eq!(out, resp);
}

#[test]
fn openrouter_normalizes_non_streaming_message_reasoning() {
    // MEDIUM: the non-streaming transform must rename `message.reasoning` to
    // `message.reasoning_content`, mirroring the streaming path.
    let p = OpenRouterProvider;
    let resp = json!({
        "id": "chatcmpl-1",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Hi", "reasoning": "thinking"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    });
    let out = ChatProvider::transform_response(&p, &resp, "openai/o1").expect("ok");
    let message = &out["choices"][0]["message"];
    assert!(
        message.get("reasoning").is_none(),
        "raw `reasoning` must be removed"
    );
    assert_eq!(message["reasoning_content"], "thinking");
}

#[test]
fn openrouter_non_streaming_preserves_existing_reasoning_content() {
    // Complement: an existing non-null `reasoning_content` wins over a stray
    // `reasoning`, and the stray is dropped.
    let p = OpenRouterProvider;
    let resp = json!({
        "choices": [{
            "message": {"reasoning": "raw", "reasoning_content": "kept"}
        }]
    });
    let out = ChatProvider::transform_response(&p, &resp, "openai/o1").expect("ok");
    let message = &out["choices"][0]["message"];
    assert!(message.get("reasoning").is_none());
    assert_eq!(message["reasoning_content"], "kept");
}

#[test]
fn openrouter_non_streaming_overwrites_null_reasoning_content() {
    // The null-aware guard: an explicit `reasoning_content: null` must not
    // swallow the removed `reasoning` payload (non-streaming side).
    let p = OpenRouterProvider;
    let resp = json!({
        "choices": [{
            "message": {"reasoning": "raw", "reasoning_content": null}
        }]
    });
    let out = ChatProvider::transform_response(&p, &resp, "openai/o1").expect("ok");
    let message = &out["choices"][0]["message"];
    assert!(message.get("reasoning").is_none());
    assert_eq!(message["reasoning_content"], "raw");
}

#[test]
fn openrouter_response_null_tokens_sanitized_to_zero() {
    // LOW: chat + embeddings `transform_response` must rewrite null token
    // counts to `0`, matching every other OpenAI-compat provider.
    let p = OpenRouterProvider;
    let resp = json!({
        "id": "x",
        "usage": {"prompt_tokens": null, "completion_tokens": null, "total_tokens": null},
    });
    let out = ChatProvider::transform_response(&p, &resp, "openai/gpt-4o").expect("ok");
    assert_eq!(out["usage"]["prompt_tokens"], 0);
    assert_eq!(out["usage"]["completion_tokens"], 0);
    assert_eq!(out["usage"]["total_tokens"], 0);
}

#[test]
fn openrouter_embedding_response_null_tokens_sanitized_to_zero() {
    let p = OpenRouterProvider;
    let resp = json!({
        "data": [],
        "usage": {"prompt_tokens": null, "total_tokens": null},
    });
    let out = EmbeddingProvider::transform_response(&p, &resp, "openai/text-embedding-3-small")
        .expect("ok");
    assert_eq!(out["usage"]["prompt_tokens"], 0);
    assert_eq!(out["usage"]["total_tokens"], 0);
}

#[test]
fn openrouter_response_preserves_usage_cost() {
    // LOW: OpenRouter's authoritative reported cost (`usage.cost`) must
    // survive `transform_response` unchanged. The provider injects
    // `usage:{include:true}` specifically to obtain this number, so a future
    // transform must not strip or sanitize it (`sanitize_null_tokens` only
    // touches token counts, never `cost`).
    let p = OpenRouterProvider;
    let resp = json!({
        "id": "chatcmpl-1",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2, "cost": 0.0023},
    });
    let out = ChatProvider::transform_response(&p, &resp, "openai/gpt-4o").expect("ok");
    assert_eq!(out["usage"]["cost"], json!(0.0023));
    assert_eq!(out["usage"]["prompt_tokens"], 1);
}

#[test]
fn openrouter_stream_chunk_preserves_usage_cost() {
    // LOW: the final usage-only stream chunk carries OpenRouter's reported
    // cost; `transform_stream_chunk` must pass `usage.cost` through unchanged.
    let p = OpenRouterProvider;
    let raw = r#"data: {"id":"x","choices":[],"usage":{"prompt_tokens":1,"cost":0.0023}}"#;
    let out = p
        .transform_stream_chunk(raw)
        .expect("ok")
        .expect("should produce value");
    assert_eq!(out["usage"]["cost"], json!(0.0023));
    assert_eq!(out["usage"]["prompt_tokens"], 1);
}

#[test]
fn openrouter_overwrites_null_reasoning_content() {
    // LOW (streaming): an explicit `reasoning_content: null` must not cause
    // the removed `reasoning` payload to be dropped.
    let p = OpenRouterProvider;
    let raw = r#"data: {"id":"x","choices":[{"index":0,"delta":{"reasoning":"raw","reasoning_content":null}}]}"#;
    let out = p
        .transform_stream_chunk(raw)
        .expect("ok")
        .expect("should produce value");
    let delta = &out["choices"][0]["delta"];
    assert!(delta.get("reasoning").is_none());
    assert_eq!(delta["reasoning_content"], "raw");
}

#[test]
fn openrouter_request_drops_null_messages() {
    // LOW: `messages: null` must be dropped, matching the null-guard applied
    // to every other forwarded field.
    let p = OpenRouterProvider;
    let req = json!({
        "model": "openai/gpt-4o",
        "messages": null,
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert!(out.get("messages").is_none());
}

#[test]
fn stream_chunk_parses_openai_format() {
    let p = OpenRouterProvider;
    let raw = "data: {\"id\":\"x\",\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}";
    let out = p
        .transform_stream_chunk(raw)
        .expect("ok")
        .expect("should produce value");
    assert_eq!(out["id"], "x");
    assert_eq!(out["choices"][0]["delta"]["content"], "hi");
}

#[test]
fn stream_done_returns_none() {
    let p = OpenRouterProvider;
    assert!(
        p.transform_stream_chunk("data: [DONE]")
            .expect("ok")
            .is_none()
    );
}

#[test]
fn stream_in_body_error_is_surfaced_as_err() {
    // OpenRouter occasionally emits mid-stream error chunks shaped like
    // `{"error":{"message":..., "code":...}}`. These must surface as `Err`,
    // not as `Ok(Some(...))`, so the server can terminate the stream.
    let p = OpenRouterProvider;
    let raw = r#"data: {"error":{"message":"overloaded","code":429}}"#;
    let result = p.transform_stream_chunk(raw);
    assert!(result.is_err(), "error chunk must surface as Err");
    let err = result.expect_err("must surface provider error");
    assert!(
        matches!(
            &err,
            oxllm_core::Error::Upstream { status: 429, message }
                if message == "overloaded"
        ),
        "expected Upstream {{ status: 429 }}, got {err:?}"
    );
}

#[test]
fn provider_id_and_supports() {
    let p = OpenRouterProvider;
    assert_eq!(p.id(), "openrouter");
    assert!(p.supports(Endpoint::ChatCompletions));
    assert!(p.supports(Endpoint::Embeddings));
    assert!(p.embeddings().is_some());
}

#[test]
fn strip_openrouter_prefix_handles_unknown() {
    assert_eq!(
        strip_openrouter_prefix("openrouter/openai/gpt-4o"),
        "openai/gpt-4o"
    );
    assert_eq!(strip_openrouter_prefix("openai/gpt-4o"), "openai/gpt-4o");
}

#[test]
fn supported_params_published() {
    let p = OpenRouterProvider;
    let params = ChatProvider::supported_params(&p).expect("should publish params");
    assert!(params.contains(&"temperature"));
    assert!(params.contains(&"tools"));
}

#[test]
fn supported_params_aligns_with_openai_compat() {
    let p = OpenRouterProvider;
    let params = ChatProvider::supported_params(&p).expect("should publish params");
    assert!(params.contains(&"max_completion_tokens"));
    assert!(params.contains(&"stream_options"));
    assert!(params.contains(&"logprobs"));
    assert!(params.contains(&"parallel_tool_calls"));
}

#[test]
fn supported_params_includes_openrouter_routing_fields() {
    // MEDIUM / LOW: the OpenRouter-only top-level fields must be advertised so
    // the host's drop_params machinery does not strip them and warn_unsupported
    // does not emit spurious warnings.
    let p = OpenRouterProvider;
    let params = ChatProvider::supported_params(&p).expect("should publish params");
    assert!(
        params.contains(&"transforms"),
        "transforms must be supported"
    );
    assert!(params.contains(&"models"), "models must be supported");
    assert!(params.contains(&"route"), "route must be supported");
    assert!(params.contains(&"usage"), "usage must be supported");
}

const CANONICAL_GOLDEN_REQUEST: &str =
    include_str!("../openai_compat/goldens/chat_request_001.json");
const CANONICAL_GOLDEN_RESPONSE: &str =
    include_str!("../openai_compat/goldens/chat_response_001.json");

fn normalize(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut sorted = serde_json::Map::new();
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            for key in keys {
                let v = &map[key];
                if v.is_null() {
                    continue;
                }
                sorted.insert(key.clone(), normalize(v));
            }
            Value::Object(sorted)
        }
        Value::Array(items) => Value::Array(items.iter().map(normalize).collect()),
        other => other.clone(),
    }
}

fn assert_semantically_equal(a: &Value, b: &Value) {
    assert_eq!(normalize(a), normalize(b));
}

#[test]
fn test_golden_file_parity() {
    let provider = OpenRouterProvider;
    let deployment = deployment();
    let model = "openai/gpt-4o";

    let mut request: Value = serde_json::from_str(CANONICAL_GOLDEN_REQUEST)
        .expect("canonical golden request should be valid JSON");
    request["model"] = json!(model);
    // OpenRouter's transform always injects `usage: {include: true}` (M5), so
    // the golden request must carry it for the round-trip assertion to hold.
    request["usage"] = json!({"include": true});
    let out = ChatProvider::transform_request(&provider, &request, &deployment)
        .expect("golden request transform should succeed");
    assert_semantically_equal(&out, &request);

    let mut response: Value = serde_json::from_str(CANONICAL_GOLDEN_RESPONSE)
        .expect("canonical golden response should be valid JSON");
    response["model"] = json!(model);
    let out = ChatProvider::transform_response(&provider, &response, model)
        .expect("golden response transform should succeed");
    assert_semantically_equal(&out, &response);
}

fn deployment_with_model(model: &str) -> Deployment {
    let mut dep = deployment();
    dep.oxllm_params.model = model.to_string();
    dep
}

#[test]
fn chat_transform_request_wildcard_resolves_from_body() {
    let p = OpenRouterProvider;
    let dep = deployment_with_model("openrouter/*");
    let req = json!({
    "model": "openrouter/anthropic/claude-3.5-sonnet",
    "messages": [{"role": "user", "content": "Hi"}],
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    // Only the first `openrouter/` segment is stripped from the resolved model.
    assert_eq!(out["model"], "anthropic/claude-3.5-sonnet");
    assert_eq!(out["messages"][0]["content"], "Hi");
}

#[test]
fn chat_transform_request_bare_wildcard_resolves_from_body() {
    let p = OpenRouterProvider;
    let dep = deployment_with_model("*");
    let req = json!({
    "model": "openrouter/openai/gpt-4o-mini",
    "messages": [],
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(out["model"], "openai/gpt-4o-mini");
}

#[test]
fn chat_transform_request_non_wildcard_ignores_body_model() {
    let p = OpenRouterProvider;
    let dep = deployment_with_model("openrouter/openai/gpt-4o");
    let req = json!({
    "model": "anything-else-the-client-sent",
    "messages": [],
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(out["model"], "openai/gpt-4o");
}

#[test]
fn embedding_transform_request_wildcard_resolves_from_body() {
    let p = OpenRouterProvider;
    let dep = deployment_with_model("openrouter/*");
    let req = json!({
    "model": "openrouter/openai/text-embedding-3-small",
    "input": "hello",
    });
    let out = EmbeddingProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(out["model"], "openai/text-embedding-3-small");
    assert_eq!(out["input"], "hello");
}

/// Model the host's `strip_unsupported_params` inline: drop any top-level key
/// not present in the provider's `supported_params` list. Mirrors
/// `orchestration/upstream.rs::strip_unsupported_params` without pulling the
/// server crate into a provider unit test.
fn strip_like_host(provider: &OpenRouterProvider, request: &Value) -> Value {
    let supported = ChatProvider::supported_params(provider).expect("openrouter publishes params");
    let Some(obj) = request.as_object() else {
        return request.clone();
    };
    let mut out = serde_json::Map::new();
    for (key, value) in obj {
        if supported.contains(&key.as_str()) {
            out.insert(key.clone(), value.clone());
        }
    }
    Value::Object(out)
}

#[test]
fn transforms_models_route_survive_drop_params_then_transform() {
    // MEDIUM: with `drop_params = true` the host strips keys not in
    // `supported_params` *before* calling `transform_request`. The OpenRouter
    // re-injection loop reads from the (stripped) request, so the routing
    // fields must be in `supported_params` to survive. Simulate the host path
    // end-to-end and assert the fields reach the upstream body.
    let p = OpenRouterProvider;
    let req = json!({
        "model": "openai/gpt-4o",
        "messages": [{"role": "user", "content": "Hi"}],
        "transforms": ["middle-out"],
        "models": ["openai/gpt-4o", "anthropic/claude-3.5-sonnet"],
        "route": "fallback",
        "usage": {"include": false},
    });
    let stripped = strip_like_host(&p, &req);
    let out = ChatProvider::transform_request(&p, &stripped, &deployment()).expect("ok");
    assert_eq!(out["transforms"][0], "middle-out");
    assert_eq!(out["models"][0], "openai/gpt-4o");
    assert_eq!(out["models"][1], "anthropic/claude-3.5-sonnet");
    assert_eq!(out["route"], "fallback");
    // `usage` is always overwritten by the provider's `usage: {include: true}`.
    assert_eq!(out["usage"]["include"], true);
}

// --- apply_auth: no bearer credential emits a warning, leaves no auth header ---

#[test]
fn apply_auth_inserts_nothing_when_credential_has_no_bearer() {
    // LOW: an `AwsSigV4` credential has no bearer token. `apply_auth` must not
    // insert any auth header. The `tracing::warn!` is verified by code
    // inspection (capturing it would require a tracing dev-dependency).
    let p = OpenRouterProvider;
    let creds = ResolvedCredentials::AwsSigV4 {
        access_key: "AKIAEXAMPLE".to_string(),
        secret_key: "secret".to_string(),
        session_token: None,
        region: "us-east-1".to_string(),
    };
    let mut headers = std::collections::HashMap::new();
    ChatProvider::apply_auth(&p, &deployment(), &creds, &mut headers);
    assert!(
        !headers.contains_key("authorization"),
        "AwsSigV4 must not produce an authorization header: {headers:?}"
    );
    assert!(
        headers.is_empty(),
        "no auth/ranking headers should be set without a bearer: {headers:?}"
    );
}
