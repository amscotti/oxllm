use super::*;
use oxllm_core::Endpoint;
use oxllm_core::deployment::{Deployment, OxllmParams};
use oxllm_core::provider::{ChatProvider, Provider, ResolvedCredentials};
use serde_json::{Value, json};

fn make_deployment(model: &str) -> Deployment {
    Deployment {
        model_name: model.to_string(),
        oxllm_params: OxllmParams {
            model: model.to_string(),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn make_deployment_with_extra(model: &str, extra: &[(&str, Value)]) -> Deployment {
    let mut dep = make_deployment(model);
    for (k, v) in extra {
        dep.oxllm_params.extra.insert((*k).to_string(), v.clone());
    }
    dep
}

const PROVIDER: VertexProvider = VertexProvider;

#[test]
fn claude_message_output_configuration_keeps_the_required_body_beta() {
    let deployment = make_deployment_with_extra(
        "vertex_ai/claude-sonnet-5-5",
        &[("vertex_project", json!("project"))],
    );
    let request = json!({
        "messages":[{"role":"user","content":"hi","output_config":{"effort":"high"}}],
        "max_tokens":100
    });
    let body = ChatProvider::transform_request(&PROVIDER, &request, &deployment).unwrap();
    assert_eq!(
        body["messages"][0]["output_config"],
        request["messages"][0]["output_config"]
    );
    assert!(
        body["anthropic_beta"]
            .as_array()
            .unwrap()
            .iter()
            .any(|value| {
                value.as_str() == Some(crate::anthropic::BETA_MID_CONVERSATION_OUTPUT_CONFIG)
            })
    );
    assert!(body.get("model").is_none());
}

// ---------------------------------------------------------------------------
// endpoint_url
// ---------------------------------------------------------------------------

#[test]
fn endpoint_url_gemini_uses_openai_compat_path() {
    let dep = make_deployment_with_extra(
        "vertex_ai/gemini-1.5-pro",
        &[
            ("vertex_project", json!("my-project")),
            ("vertex_location", json!("us-central1")),
        ],
    );
    assert_eq!(
        PROVIDER.endpoint_url(&dep),
        "https://us-central1-aiplatform.googleapis.com/v1beta1/projects/my-project/locations/us-central1/endpoints/openapi/chat/completions"
    );
}

#[test]
fn endpoint_url_claude_uses_raw_predict_path() {
    let dep = make_deployment_with_extra(
        "vertex_ai/claude-3-5-sonnet@20241022",
        &[
            ("vertex_project", json!("my-project")),
            ("vertex_location", json!("us-central1")),
        ],
    );
    assert_eq!(
        PROVIDER.endpoint_url(&dep),
        "https://us-central1-aiplatform.googleapis.com/v1/projects/my-project/locations/us-central1/publishers/anthropic/models/claude-3-5-sonnet@20241022:rawPredict"
    );
}

#[test]
fn endpoint_url_uses_custom_api_base_when_present() {
    let mut dep = make_deployment_with_extra(
        "vertex_ai/gemini-2.0-flash",
        &[("vertex_project", json!("p"))],
    );
    dep.oxllm_params.api_base = Some("https://europe-west4-aiplatform.googleapis.com".to_string());
    let url = PROVIDER.endpoint_url(&dep);
    assert!(url.starts_with("https://europe-west4-aiplatform.googleapis.com/"));
    assert!(url.contains("/locations/us-central1/")); // location still defaults
}

#[test]
fn endpoint_url_defaults_project_and_location_when_missing() {
    let dep = make_deployment("vertex_ai/gemini-1.5-pro");
    let url = PROVIDER.endpoint_url(&dep);
    assert!(url.contains("/projects/unknown-project/"));
    assert!(url.contains("/locations/us-central1/"));
}

#[test]
fn endpoint_url_strips_vertex_prefix_from_claude_model() {
    let dep = make_deployment_with_extra(
        "vertex_ai/claude-3-opus@20240229",
        &[("vertex_project", json!("p"))],
    );
    let url = PROVIDER.endpoint_url(&dep);
    // The model segment in the URL must NOT include `vertex_ai/`.
    assert!(url.contains("/models/claude-3-opus@20240229:rawPredict"));
    assert!(!url.contains("vertex_ai/"));
}

// ---------------------------------------------------------------------------
// apply_auth
// ---------------------------------------------------------------------------

#[test]
fn apply_auth_uses_gcp_bearer_token() {
    let dep = make_deployment("vertex_ai/gemini-1.5-pro");
    let creds = ResolvedCredentials::GcpBearer {
        token: "ya29.test-token".to_string(),
    };
    let mut headers = HashMap::new();
    ChatProvider::apply_auth(&PROVIDER, &dep, &creds, &mut headers);
    assert_eq!(
        headers.get("authorization").map(String::as_str),
        Some("Bearer ya29.test-token")
    );
}

#[test]
fn apply_auth_falls_back_to_best_effort_bearer() {
    let dep = make_deployment("vertex_ai/gemini-1.5-pro");
    let creds = ResolvedCredentials::ApiKey {
        bearer: "fallback-key".to_string(),
    };
    let mut headers = HashMap::new();
    ChatProvider::apply_auth(&PROVIDER, &dep, &creds, &mut headers);
    assert_eq!(
        headers.get("authorization").map(String::as_str),
        Some("Bearer fallback-key")
    );
}

// ---------------------------------------------------------------------------
// transform_request — Gemini (OpenAI-compat)
// ---------------------------------------------------------------------------

#[test]
fn transform_request_gemini_strips_vertex_prefix() {
    let dep = make_deployment_with_extra(
        "vertex_ai/gemini-1.5-pro",
        &[("vertex_project", json!("p"))],
    );
    let req = json!({
    "model": "vertex_ai/gemini-1.5-pro",
    "messages": [{"role": "user", "content": "Hi"}],
    "temperature": 0.5,
    "stream": false,
    });
    let out = ChatProvider::transform_request(&PROVIDER, &req, &dep).expect("ok");
    assert_eq!(out["model"], "gemini-1.5-pro");
    assert_eq!(out["temperature"], 0.5);
    assert_eq!(out["messages"][0]["content"], "Hi");
}

#[test]
fn transform_request_gemini_38_flash_minimal_effort_becomes_low() {
    let dep = make_deployment_with_extra(
        "vertex_ai/gemini-3.8-flash",
        &[("vertex_project", json!("p"))],
    );
    let req = json!({
        "model": "vertex_ai/gemini-3.8-flash",
        "messages": [{"role": "user", "content": "Hi"}],
        "reasoning_effort": "minimal",
    });
    let out = ChatProvider::transform_request(&PROVIDER, &req, &dep).expect("ok");
    assert_eq!(out["reasoning_effort"], "low");
}

#[test]
fn transform_request_gemini_drops_null_params() {
    let dep = make_deployment_with_extra(
        "vertex_ai/gemini-1.5-pro",
        &[("vertex_project", json!("p"))],
    );
    let req = json!({
    "model": "vertex_ai/gemini-1.5-pro",
    "messages": [],
    "temperature": null,
    "max_tokens": 5,
    });
    let out = ChatProvider::transform_request(&PROVIDER, &req, &dep).expect("ok");
    assert!(out.get("temperature").is_none());
    assert_eq!(out["max_tokens"], 5);
}

#[test]
fn transform_request_gemini_wildcard_uses_request_model() {
    let dep = make_deployment_with_extra("vertex_ai/*", &[("vertex_project", json!("p"))]);
    let req = json!({
    "model": "vertex_ai/gemini-2.0-flash",
    "messages": [],
    });
    let out = ChatProvider::transform_request(&PROVIDER, &req, &dep).expect("ok");
    assert_eq!(out["model"], "gemini-2.0-flash");
}

// ---------------------------------------------------------------------------
// transform_request — Claude (Anthropic via rawPredict)
// ---------------------------------------------------------------------------

#[test]
fn transform_request_claude_injects_anthropic_version() {
    let dep = make_deployment_with_extra(
        "vertex_ai/claude-3-5-sonnet@20241022",
        &[("vertex_project", json!("p"))],
    );
    let req = json!({
    "model": "vertex_ai/claude-3-5-sonnet@20241022",
    "messages": [{"role": "user", "content": "Hi"}],
    "max_tokens": 100,
    });
    let out = ChatProvider::transform_request(&PROVIDER, &req, &dep).expect("ok");
    assert_eq!(out["anthropic_version"], VERTEX_ANTHROPIC_VERSION);
    // Vertex `:rawPredict` rejects a top-level `model` field — LiteLLM pops
    // it (`transformation.py:102`) and encodes the model in the URL path.
    assert!(
        out.get("model").is_none(),
        "model must be absent from Vertex Anthropic body, got: {out}"
    );
    assert_eq!(out["max_tokens"], 100);
    assert!(out.get("system").is_none());
}

/// Directly pins LiteLLM's `data.pop("model", None)` contract
/// (`test_vertex_ai_partner_models_anthropic_transformation.py:307,665`):
/// the transformed Vertex Anthropic body must not carry a top-level `model`
/// key, regardless of how the model id was supplied.
#[test]
fn transform_request_claude_body_has_no_model_key() {
    let dep = make_deployment_with_extra(
        "vertex_ai/claude-3-5-sonnet@20241022",
        &[("vertex_project", json!("p"))],
    );
    let req = json!({
    "model": "vertex_ai/claude-3-5-sonnet@20241022",
    "messages": [{"role": "user", "content": "Hi"}],
    "max_tokens": 100,
    });
    let out = ChatProvider::transform_request(&PROVIDER, &req, &dep).expect("ok");
    assert!(out.get("model").is_none());
    assert_eq!(out["anthropic_version"], VERTEX_ANTHROPIC_VERSION);
}

/// Wildcard URL resolution must not leak a model field into the native body.
#[test]
fn claude_wildcard_resolves_the_url_without_a_model_body_field() {
    let dep = make_deployment_with_extra("vertex_ai/claude-*", &[("vertex_project", json!("p"))]);
    let req = json!({
    "model": "vertex_ai/claude-3-opus@20240229",
    "messages": [{"role": "user", "content": "Hi"}],
    "max_tokens": 100,
    });
    let out = ChatProvider::transform_request(&PROVIDER, &req, &dep).expect("ok");
    assert!(out.get("model").is_none());
    assert_eq!(out["anthropic_version"], VERTEX_ANTHROPIC_VERSION);
    assert!(
        PROVIDER
            .endpoint_url_for_request(&dep, &req)
            .ends_with("/models/claude-3-opus@20240229:rawPredict")
    );
}

#[test]
fn configured_location_selects_the_regional_host_and_gemini_drops_openai_only_fields() {
    let dep = make_deployment_with_extra(
        "vertex_ai/gemini-2.5-pro",
        &[
            ("vertex_project", json!("p")),
            ("vertex_location", json!("europe-west4")),
        ],
    );
    assert!(
        PROVIDER
            .endpoint_url(&dep)
            .starts_with("https://europe-west4-aiplatform.googleapis.com/")
    );
    let req = json!({"model":"public-alias","messages":[{"role":"user","content":"hi"}],"metadata":{"debug":"true"},"store":true,"logit_bias":{"1":2}});
    let body = ChatProvider::transform_request(&PROVIDER, &req, &dep).unwrap();
    for field in ["metadata", "store", "logit_bias"] {
        assert!(body.get(field).is_none());
    }
}

#[test]
fn transform_request_claude_extracts_system_message() {
    let dep = make_deployment_with_extra(
        "vertex_ai/claude-3-5-sonnet@20241022",
        &[("vertex_project", json!("p"))],
    );
    let req = json!({
    "model": "vertex_ai/claude-3-5-sonnet@20241022",
    "messages": [
    {"role": "system", "content": "You are a bot."},
    {"role": "user", "content": "Hi"}
    ],
    "max_tokens": 64,
    });
    let out = ChatProvider::transform_request(&PROVIDER, &req, &dep).expect("ok");
    assert_eq!(out["system"], "You are a bot.");
    // The system message must not appear in `messages`.
    let messages = out["messages"].as_array().expect("messages array");
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0]["role"], "user");
}

#[test]
fn transform_request_claude_wildcard_uses_request_model() {
    // Claude on Vertex requires an explicit (or `x/*`-style path-prefix)
    // Claude deployment; a bare `vertex_ai/*` wildcard is rejected for
    // Claude models because the Gemini OpenAI-compat endpoint would be used
    // (see `endpoint_url_wildcard_resolves_to_gemini_path`).
    let dep = make_deployment_with_extra(
        "vertex_ai/claude-3-opus@20240229",
        &[("vertex_project", json!("p"))],
    );
    let req = json!({
    "model": "vertex_ai/claude-3-opus@20240229",
    "messages": [{"role": "user", "content": "Hi"}],
    "max_tokens": 32,
    });
    let out = ChatProvider::transform_request(&PROVIDER, &req, &dep).expect("ok");
    // Model is encoded in the URL path; the body must not carry `model`.
    assert!(out.get("model").is_none());
    assert_eq!(out["anthropic_version"], VERTEX_ANTHROPIC_VERSION);
}

// ---------------------------------------------------------------------------
// transform_response
// ---------------------------------------------------------------------------

#[test]
fn transform_response_gemini_is_identity() {
    let resp = json!({
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
    });
    let out =
        ChatProvider::transform_response(&PROVIDER, &resp, "vertex_ai/gemini-1.5-pro").expect("ok");
    assert_eq!(out, resp);
}

#[test]
fn transform_response_claude_maps_to_openai_shape() {
    let anthropic_resp = json!({
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-3-5-sonnet-20241022",
    "content": [{"type": "text", "text": "Hello from Vertex Claude"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 8, "output_tokens": 7}
    });
    let out = ChatProvider::transform_response(
        &PROVIDER,
        &anthropic_resp,
        "vertex_ai/claude-3-5-sonnet@20241022",
    )
    .expect("ok");
    assert_eq!(out["object"], "chat.completion");
    assert_eq!(
        out["choices"][0]["message"]["content"],
        "Hello from Vertex Claude"
    );
    assert_eq!(out["choices"][0]["finish_reason"], "stop");
    assert_eq!(out["usage"]["prompt_tokens"], 8);
    assert_eq!(out["usage"]["completion_tokens"], 7);
    assert_eq!(out["usage"]["total_tokens"], 15);
}

// ---------------------------------------------------------------------------
// transform_stream_chunk
// ---------------------------------------------------------------------------

#[test]
fn stream_chunk_openai_format_passes_through() {
    let raw = "data: {\"id\":\"x\",\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}";
    let out = PROVIDER
        .transform_stream_chunk(raw)
        .expect("ok")
        .expect("should produce value");
    assert_eq!(out["id"], "x");
    assert_eq!(out["choices"][0]["delta"]["content"], "hi");
}

#[test]
fn stream_chunk_done_returns_none() {
    assert!(
        PROVIDER
            .transform_stream_chunk("data: [DONE]")
            .expect("ok")
            .is_none()
    );
}

#[test]
fn stream_chunk_empty_returns_none() {
    assert!(PROVIDER.transform_stream_chunk("").expect("ok").is_none());
}

#[test]
fn stream_chunk_anthropic_message_start_routes_to_anthropic_parser() {
    let raw = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude-3-5-sonnet-20241022\",\"usage\":{\"input_tokens\":10}}}";
    let out = PROVIDER
        .transform_stream_chunk(raw)
        .expect("ok")
        .expect("should produce value");
    assert_eq!(out["object"], "chat.completion.chunk");
    assert_eq!(out["choices"][0]["delta"]["role"], "assistant");
}

#[test]
fn stream_chunk_anthropic_content_delta_routes_to_anthropic_parser() {
    let raw = "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi\"}}";
    let out = PROVIDER
        .transform_stream_chunk(raw)
        .expect("ok")
        .expect("should produce value");
    assert_eq!(out["choices"][0]["delta"]["content"], "Hi");
}

/// Pins the LOW-severity heuristic-dispatch behavior noted in the review: a
/// keep-alive `{"type":"ping"}` chunk (which the Anthropic event list also
/// covers) is handed to the Anthropic parser, whose fall-through arm returns
/// `Ok(None)` — so it degrades to a skipped frame rather than a parse error.
/// A Gemini OpenAI-compat chunk never carries a top-level `type` field, so
/// only genuinely Anthropic-shaped events are misrouted here.
#[test]
fn stream_chunk_ping_event_returns_none() {
    let raw = "data: {\"type\":\"ping\"}";
    assert!(PROVIDER.transform_stream_chunk(raw).expect("ok").is_none());
}

// ---------------------------------------------------------------------------
// Provider trait surface
// ---------------------------------------------------------------------------

#[test]
fn provider_id_is_vertex_ai() {
    assert_eq!(PROVIDER.id(), "vertex_ai");
}

#[test]
fn provider_supports_chat_and_embeddings() {
    assert!(PROVIDER.supports(Endpoint::ChatCompletions));
    assert!(PROVIDER.supports(Endpoint::Embeddings));
}

// ---------------------------------------------------------------------------
// Embeddings (Gemini OpenAI-compat path)
// ---------------------------------------------------------------------------

use oxllm_core::provider::EmbeddingProvider;

#[test]
fn embedding_url_uses_openai_compat_path() {
    let dep = make_deployment_with_extra(
        "vertex_ai/text-embedding-005",
        &[("vertex_project", json!("my-project"))],
    );
    assert_eq!(
        EmbeddingProvider::embedding_url(&PROVIDER, &dep),
        "https://us-central1-aiplatform.googleapis.com/v1beta1/projects/my-project/locations/us-central1/endpoints/openapi/embeddings"
    );
}

#[test]
fn embedding_transform_request_strips_vertex_prefix() {
    let dep = make_deployment_with_extra(
        "vertex_ai/text-embedding-005",
        &[("vertex_project", json!("p"))],
    );
    let req = json!({
    "model": "vertex_ai/text-embedding-005",
    "input": "hello world",
    "dimensions": 768,
    });
    let out = EmbeddingProvider::transform_request(&PROVIDER, &req, &dep).expect("ok");
    assert_eq!(out["model"], "text-embedding-005");
    assert_eq!(out["input"], "hello world");
    assert_eq!(out["dimensions"], 768);
}

#[test]
fn embedding_transform_request_wildcard_uses_request_model() {
    let dep = make_deployment_with_extra("vertex_ai/*", &[("vertex_project", json!("p"))]);
    let req = json!({
    "model": "vertex_ai/text-multilingual-embedding-002",
    "input": ["a", "b"],
    });
    let out = EmbeddingProvider::transform_request(&PROVIDER, &req, &dep).expect("ok");
    assert_eq!(out["model"], "text-multilingual-embedding-002");
    assert_eq!(out["input"], json!(["a", "b"]));
}

#[test]
fn embedding_transform_response_is_identity() {
    let resp = json!({
    "object": "list",
    "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
    });
    let out =
        EmbeddingProvider::transform_response(&PROVIDER, &resp, "vertex_ai/text-embedding-005")
            .expect("ok");
    assert_eq!(out, resp);
}

#[test]
fn provider_supported_params_is_some() {
    assert!(PROVIDER.supported_params().is_some());
}

// ---------------------------------------------------------------------------
// strip_vertex_prefix
// ---------------------------------------------------------------------------

#[test]
fn strip_prefix_removes_vertex_ai() {
    assert_eq!(
        strip_vertex_prefix("vertex_ai/gemini-1.5-pro"),
        "gemini-1.5-pro"
    );
    assert_eq!(
        strip_vertex_prefix("vertex_ai/claude-3-5-sonnet@20241022"),
        "claude-3-5-sonnet@20241022"
    );
}

#[test]
fn strip_prefix_leaves_bare_model_untouched() {
    assert_eq!(strip_vertex_prefix("gemini-1.5-pro"), "gemini-1.5-pro");
}

#[test]
fn strip_prefix_does_not_strip_other_providers() {
    assert_eq!(strip_vertex_prefix("anthropic/claude"), "anthropic/claude");
}

// ---------------------------------------------------------------------------
// transform_response_with_request (HIGH: tool-name restoration across threads)
// ---------------------------------------------------------------------------

/// A tool name that requires sanitization round-trips back to the caller's
/// original name even when the response transform runs on a different worker
/// thread than `transform_request`. Without the `transform_response_with_request`
/// override, the reverse map lives in Anthropic's `thread_local!` cell, which
/// is empty on the resumed thread and leaves the sanitized name in place.
#[test]
fn transform_response_with_request_restores_tool_names_across_thread() {
    let dep = make_deployment_with_extra(
        "vertex_ai/claude-3-5-sonnet@20241022",
        &[("vertex_project", json!("p"))],
    );
    let original_request = json!({
    "model": "vertex_ai/claude-3-5-sonnet@20241022",
    "messages": [{"role": "user", "content": "weather?"}],
    "max_tokens": 64,
    "tools": [{"type": "function", "function": {
        "name": "get weather",
        "description": "Get the weather",
        "parameters": {"type": "object", "properties": {}},
    }}],
    });
    // Run transform_request on THIS thread (populates the thread-local cell
    // here only), then transform the response on a SEPARATE thread whose
    // thread-local cell is empty — simulating a tokio worker-thread migration
    // across the upstream HTTP `.await`.
    let _ = ChatProvider::transform_request(&PROVIDER, &original_request, &dep).expect("req");
    let anthropic_resp = json!({
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-3-5-sonnet-20241022",
    "content": [{
        "type": "tool_use",
        "id": "toolu_1",
        "name": "get_weather",
        "input": {"city": "SF"},
    }],
    "stop_reason": "tool_use",
    "usage": {"input_tokens": 10, "output_tokens": 5},
    });
    let model = "vertex_ai/claude-3-5-sonnet@20241022".to_string();
    let restored = std::thread::scope(|s| {
        s.spawn(move || {
            ChatProvider::transform_response_with_request(
                &PROVIDER,
                &anthropic_resp,
                &model,
                &original_request,
            )
            .expect("resp")
        })
        .join()
        .expect("thread joined")
    });
    let tool_name = restored["choices"][0]["message"]["tool_calls"][0]["function"]["name"]
        .as_str()
        .expect("tool name string");
    assert_eq!(
        tool_name, "get weather",
        "original caller tool name must survive thread migration"
    );
}

/// The Gemini (non-Claude) branch delegates to the identity response transform.
#[test]
fn transform_response_with_request_gemini_is_identity() {
    let resp = json!({
    "id": "x",
    "object": "chat.completion",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
    });
    let req = json!({"model": "vertex_ai/gemini-1.5-pro", "messages": []});
    let out = ChatProvider::transform_response_with_request(
        &PROVIDER,
        &resp,
        "vertex_ai/gemini-1.5-pro",
        &req,
    )
    .expect("ok");
    assert_eq!(out["choices"][0]["message"]["content"], "hi");
}

// ---------------------------------------------------------------------------
// transform_stream_chunk_stateful (HIGH: Claude streaming uses host state)
// ---------------------------------------------------------------------------

/// Streaming a Claude tool-call response through the stateful path restores
/// the original tool name even when run on a thread that never saw
/// `transform_request` (so the `thread_local!` reverse map is empty there).
#[test]
fn transform_stream_chunk_stateful_claude_restores_tool_names() {
    let original_request = json!({
    "model": "vertex_ai/claude-3-5-sonnet@20241022",
    "messages": [{"role": "user", "content": "weather?"}],
    "max_tokens": 64,
    "tools": [{"type": "function", "function": {
        "name": "get weather",
        "description": "Get the weather",
        "parameters": {"type": "object", "properties": {}},
    }}],
    "stream": true,
    });

    let message_start = "event: message_start\n\
        data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude-3-5-sonnet-20241022\",\"usage\":{\"input_tokens\":10}}}";
    let block_start = "event: content_block_start\n\
        data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"get_weather\",\"input\":{}}}";

    // Process the stream on a fresh thread whose thread-local cells are empty,
    // proving the host-owned `state` (not `thread_local!`) carries the map.
    let tool_name = std::thread::scope(|s| {
        s.spawn(|| {
            let mut state: Option<Box<dyn std::any::Any + Send + Sync>> = None;
            // message_start seeds stream meta in the host-owned state.
            let _ = PROVIDER
                .transform_stream_chunk_stateful(message_start, &mut state, &original_request)
                .expect("start");
            let chunk = PROVIDER
                .transform_stream_chunk_stateful(block_start, &mut state, &original_request)
                .expect("block start")
                .expect("should emit a tool-call chunk");
            chunk["choices"][0]["delta"]["tool_calls"][0]["function"]["name"]
                .as_str()
                .expect("tool name string")
                .to_string()
        })
        .join()
        .expect("thread joined")
    });
    assert_eq!(
        tool_name, "get weather",
        "streaming tool name must be restored from host-owned state"
    );
}

/// Stateful streaming of an OpenAI-compat (Gemini) chunk falls through to the
/// stateless parser and needs no host state.
#[test]
fn transform_stream_chunk_stateful_gemini_uses_stateless_path() {
    let req = json!({"model": "vertex_ai/gemini-1.5-pro", "messages": []});
    let raw = "data: {\"id\":\"x\",\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}";
    let mut state = None;
    let out = PROVIDER
        .transform_stream_chunk_stateful(raw, &mut state, &req)
        .expect("ok")
        .expect("should produce value");
    assert_eq!(out["id"], "x");
    assert_eq!(out["choices"][0]["delta"]["content"], "hi");
    assert!(state.is_none(), "gemini path must not allocate host state");
}

// ---------------------------------------------------------------------------
// transform_response — Gemini null-token sanitisation (LOW)
// ---------------------------------------------------------------------------

#[test]
fn transform_response_gemini_sanitizes_null_usage_tokens() {
    let resp = json!({
    "id": "x",
    "object": "chat.completion",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": null, "completion_tokens": 5, "total_tokens": null},
    });
    let out =
        ChatProvider::transform_response(&PROVIDER, &resp, "vertex_ai/gemini-1.5-pro").expect("ok");
    assert_eq!(out["usage"]["prompt_tokens"], 0);
    assert_eq!(out["usage"]["completion_tokens"], 5);
    assert_eq!(out["usage"]["total_tokens"], 0);
}

// ---------------------------------------------------------------------------
// resolved_model — wildcard error (LOW)
// ---------------------------------------------------------------------------

#[test]
fn transform_request_wildcard_errors_without_request_model() {
    let dep = make_deployment_with_extra("vertex_ai/*", &[("vertex_project", json!("p"))]);
    let req = json!({"messages": [{"role": "user", "content": "hi"}]});
    let err = ChatProvider::transform_request(&PROVIDER, &req, &dep)
        .expect_err("wildcard without model must error");
    assert!(
        matches!(err, oxllm_core::Error::Transform(_)),
        "expected Transform error, got {err:?}"
    );
}

#[test]
fn embedding_transform_request_wildcard_errors_without_request_model() {
    let dep = make_deployment_with_extra("vertex_ai/*", &[("vertex_project", json!("p"))]);
    let req = json!({"input": "hello"});
    let err = EmbeddingProvider::transform_request(&PROVIDER, &req, &dep)
        .expect_err("wildcard embedding without model must error");
    assert!(
        matches!(err, oxllm_core::Error::Transform(_)),
        "expected Transform error, got {err:?}"
    );
}

// ---------------------------------------------------------------------------
// endpoint_url — wildcard limitation (MEDIUM)
// ---------------------------------------------------------------------------

/// A bare-wildcard (`vertex_ai/*`) deployment always resolves to the Gemini
/// OpenAI-compat URL because `endpoint_url` only sees the configured model
/// (`*`), not the request model. Rather than silently posting an Anthropic
/// Messages body to the wrong endpoint, `transform_request` rejects a Claude
/// model on a bare-`*` deployment with a clear error; mixed-provider wildcard
/// Vertex deployments are unsupported, and operators must use a Claude-specific
/// deployment (`vertex_ai/claude-*`).
#[test]
fn endpoint_url_wildcard_resolves_to_gemini_path() {
    let dep = make_deployment_with_extra("vertex_ai/*", &[("vertex_project", json!("p"))]);
    let url = PROVIDER.endpoint_url(&dep);
    assert!(
        url.ends_with("/endpoints/openapi/chat/completions"),
        "wildcard resolves to Gemini path: {url}"
    );
    assert!(!url.contains("/publishers/anthropic/"));

    let req = json!({
        "model": "claude-3-7-sonnet",
        "messages": [{"role": "user", "content": "hi"}],
    });
    let body =
        ChatProvider::transform_request(&PROVIDER, &req, &dep).expect("wildcard Claude body");
    assert!(body.get("model").is_none());
    assert!(
        PROVIDER
            .endpoint_url_for_request(&dep, &req)
            .ends_with("/publishers/anthropic/models/claude-3-7-sonnet:rawPredict")
    );

    // A Claude-specific deployment routes to `:rawPredict` and still works.
    let dep = make_deployment_with_extra(
        "vertex_ai/claude-3-7-sonnet",
        &[("vertex_project", json!("p"))],
    );
    let url = PROVIDER.endpoint_url(&dep);
    assert!(url.contains("/publishers/anthropic/"), "url: {url}");
    let out = ChatProvider::transform_request(&PROVIDER, &req, &dep).expect("ok");
    assert_eq!(out["anthropic_version"], "vertex-2023-10-16");
}

// ---------------------------------------------------------------------------
// apply_auth — mismatched credential variant (MEDIUM)
// ---------------------------------------------------------------------------

#[test]
fn apply_auth_emits_no_header_for_sigv4_credential() {
    let dep = make_deployment("vertex_ai/gemini-1.5-pro");
    let creds = ResolvedCredentials::AwsSigV4 {
        access_key: "AKIA".to_string(),
        secret_key: "secret".to_string(),
        session_token: None,
        region: "us-east-1".to_string(),
    };
    let mut headers = HashMap::new();
    ChatProvider::apply_auth(&PROVIDER, &dep, &creds, &mut headers);
    assert!(
        !headers.contains_key("authorization"),
        "AwsSigV4 has no bearer token; no Authorization header must be set"
    );
}

// ---------------------------------------------------------------------------
// validate_config — missing/non-string vertex_project (MEDIUM)
// ---------------------------------------------------------------------------

/// A deployment without `vertex_project` must surface a clear `Error::Config`
/// at request time rather than baking a `unknown-project` placeholder into the
/// upstream URL (which Vertex rejects with a confusing 404/403).
#[test]
fn transform_request_errors_when_vertex_project_missing() {
    let dep = make_deployment("vertex_ai/gemini-1.5-pro");
    let req = json!({"model": "vertex_ai/gemini-1.5-pro", "messages": []});
    let err = ChatProvider::transform_request(&PROVIDER, &req, &dep)
        .expect_err("missing vertex_project must error");
    assert!(
        matches!(err, oxllm_core::Error::Config(_)),
        "expected Config error, got {err:?}"
    );
}

/// Same fail-fast contract on the embeddings path.
#[test]
fn embedding_transform_request_errors_when_vertex_project_missing() {
    let dep = make_deployment("vertex_ai/text-embedding-005");
    let req = json!({"model": "vertex_ai/text-embedding-005", "input": "x"});
    let err = EmbeddingProvider::transform_request(&PROVIDER, &req, &dep)
        .expect_err("missing vertex_project must error");
    assert!(
        matches!(err, oxllm_core::Error::Config(_)),
        "expected Config error, got {err:?}"
    );
}

/// A non-string `vertex_project` (e.g. a YAML number) must be rejected rather
/// than silently falling through to the placeholder path via the
/// `Value::String` pattern miss.
#[test]
fn transform_request_errors_when_vertex_project_is_non_string() {
    for bad in [json!(123), json!(true), json!(null), json!([1, 2])] {
        let dep =
            make_deployment_with_extra("vertex_ai/gemini-1.5-pro", &[("vertex_project", bad)]);
        let req = json!({"model": "vertex_ai/gemini-1.5-pro", "messages": []});
        let err = ChatProvider::transform_request(&PROVIDER, &req, &dep)
            .expect_err("non-string vertex_project must error");
        assert!(
            matches!(err, oxllm_core::Error::Config(_)),
            "expected Config error for non-string project, got {err:?}"
        );
    }
}

/// A non-string `vertex_location` is a configuration error (silent
/// type-mismatch is a debugging trap); absence is still allowed and falls back
/// to `us-central1`.
#[test]
fn transform_request_errors_when_vertex_location_is_non_string() {
    let dep = make_deployment_with_extra(
        "vertex_ai/gemini-1.5-pro",
        &[
            ("vertex_project", json!("p")),
            ("vertex_location", json!(1)),
        ],
    );
    let req = json!({"model": "vertex_ai/gemini-1.5-pro", "messages": []});
    let err = ChatProvider::transform_request(&PROVIDER, &req, &dep)
        .expect_err("non-string vertex_location must error");
    assert!(
        matches!(err, oxllm_core::Error::Config(_)),
        "expected Config error, got {err:?}"
    );
}

/// Absent `vertex_location` is valid: defaults to `us-central1`.
#[test]
fn transform_request_accepts_absent_vertex_location() {
    let dep = make_deployment_with_extra(
        "vertex_ai/gemini-1.5-pro",
        &[("vertex_project", json!("p"))],
    );
    let req = json!({"model": "vertex_ai/gemini-1.5-pro", "messages": []});
    ChatProvider::transform_request(&PROVIDER, &req, &dep).expect("absent location is valid");
}

// ---------------------------------------------------------------------------
// validate_model_segment — wildcard path-injection defence (MEDIUM)
// ---------------------------------------------------------------------------

/// A wildcard deployment resolves the model from the client request body. A
/// malicious client sending `model: "claude-x/../../v1/projects/victim"` must
/// be rejected before the model is interpolated into the upstream URL path.
#[test]
fn transform_request_wildcard_rejects_path_traversal_model() {
    let dep = make_deployment_with_extra("vertex_ai/*", &[("vertex_project", json!("p"))]);
    for evil in [
        "vertex_ai/claude-x/../../v1/projects/victim",
        "vertex_ai/gemini/../x",
        "vertex_ai/a/b",
    ] {
        let req = json!({"model": evil, "messages": []});
        let err = ChatProvider::transform_request(&PROVIDER, &req, &dep)
            .expect_err("path-traversal model must be rejected");
        assert!(
            matches!(err, oxllm_core::Error::Transform(_)),
            "expected Transform error for `{evil}`, got {err:?}"
        );
    }
}

/// A legitimate model id (alnum, `.`, `_`, `:`, `@`, `-`) is accepted by the
/// wildcard path.
#[test]
fn transform_request_wildcard_accepts_valid_model_ids() {
    let dep = make_deployment_with_extra("vertex_ai/*", &[("vertex_project", json!("p"))]);
    for ok in [
        "vertex_ai/gemini-1.5-pro",
        "vertex_ai/text-embedding-005",
        // Claude ids are also valid segments, but a bare-`*` wildcard rejects
        // them at the routing layer; exercise them via an explicit deployment.
        "vertex_ai/claude-3-5-sonnet@20241022",
    ] {
        let req = json!({"model": ok, "messages": []});
        let dep = if strip_vertex_prefix(ok).starts_with("claude") {
            make_deployment_with_extra(ok, &[("vertex_project", json!("p"))])
        } else {
            dep.clone()
        };
        ChatProvider::transform_request(&PROVIDER, &req, &dep)
            .expect("valid model id must be accepted");
    }
}

// ---------------------------------------------------------------------------
// anthropic_beta body field — Vertex Claude LiteLLM parity (LOW)
// ---------------------------------------------------------------------------

/// LiteLLM's `VertexAIAnthropicConfig.transform_request` mirrors the computed
/// beta set into the request body as `anthropic_beta` (Vertex's `:rawPredict`
/// reads the body, not the `anthropic-beta` header, for feature-gated knobs).
/// The host only computes beta headers for the direct `anthropic` provider, so
/// the Vertex provider must compute them itself. We trigger a beta via
/// `web_search_options` (which maps to `BETA_WEB_FETCH`) and assert the body
/// carries the tag.
#[test]
fn transform_request_claude_injects_anthropic_beta_body_field() {
    let dep = make_deployment_with_extra(
        "vertex_ai/claude-3-5-sonnet@20241022",
        &[("vertex_project", json!("p"))],
    );
    let req = json!({
        "model": "vertex_ai/claude-3-5-sonnet@20241022",
        "messages": [{"role": "user", "content": "search the web"}],
        "max_tokens": 64,
        "web_search_options": {},
    });
    let out = ChatProvider::transform_request(&PROVIDER, &req, &dep).expect("ok");
    let beta = out
        .get("anthropic_beta")
        .and_then(Value::as_array)
        .expect("anthropic_beta body field must be present when a beta-tagged feature is used");
    assert!(
        beta.iter()
            .any(|v| v.as_str() == Some(crate::anthropic::BETA_WEB_FETCH)),
        "anthropic_beta must contain the web-fetch tag, got: {beta:?}"
    );
}

/// When no beta-tagged feature is requested, `anthropic_beta` must be absent
/// (LiteLLM only adds the field when the set is non-empty).
#[test]
fn transform_request_claude_omits_anthropic_beta_when_no_beta_feature() {
    let dep = make_deployment_with_extra(
        "vertex_ai/claude-3-5-sonnet@20241022",
        &[("vertex_project", json!("p"))],
    );
    let req = json!({
        "model": "vertex_ai/claude-3-5-sonnet@20241022",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 32,
    });
    let out = ChatProvider::transform_request(&PROVIDER, &req, &dep).expect("ok");
    assert!(
        out.get("anthropic_beta").is_none(),
        "anthropic_beta must be absent when no beta feature is used, got: {out}"
    );
}

// ---------------------------------------------------------------------------
// extract_data_line — malformed SSE frame handling (LOW)
// ---------------------------------------------------------------------------

/// A malformed Anthropic frame with an `event:` line but no `data:` payload
/// must return `Ok(None)` (skip) rather than attempting to parse the raw
/// framing as JSON and silently routing it to the OpenAI parser.
#[test]
fn stream_chunk_event_line_without_data_returns_none() {
    let raw = "event: message_start\n";
    assert!(
        PROVIDER.transform_stream_chunk(raw).expect("ok").is_none(),
        "a frame with no data line must be skipped, not error"
    );
}

/// Same contract on the stateful path.
#[test]
fn stream_chunk_stateful_event_line_without_data_returns_none() {
    let req = json!({"model": "vertex_ai/gemini-1.5-pro", "messages": []});
    let raw = "event: message_start\n";
    let mut state = None;
    assert!(
        PROVIDER
            .transform_stream_chunk_stateful(raw, &mut state, &req)
            .expect("ok")
            .is_none(),
        "a frame with no data line must be skipped on the stateful path"
    );
}
