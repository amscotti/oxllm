use super::*;
use oxllm_core::Endpoint;
use oxllm_core::deployment::{Deployment, OxllmParams};
use oxllm_core::provider::{ChatProvider, EmbeddingProvider, Provider, ResolvedCredentials};
use serde_json::json;

fn deployment() -> Deployment {
    Deployment {
        model_name: "meta-llama/Llama-3.3-70B-Instruct-Turbo".to_string(),
        oxllm_params: OxllmParams {
            model: "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo".to_string(),
            api_key: Some("together-key".to_string()),
            ..Default::default()
        },
        ..Default::default()
    }
}

#[test]
fn endpoint_url_uses_default_together_base() {
    let p = TogetherAiProvider;
    assert_eq!(
        p.endpoint_url(&deployment()),
        "https://api.together.xyz/v1/chat/completions"
    );
}

#[test]
fn embedding_url_uses_default_together_base() {
    let p = TogetherAiProvider;
    assert_eq!(
        p.embedding_url(&deployment()),
        "https://api.together.xyz/v1/embeddings"
    );
}

#[test]
fn endpoint_url_respects_custom_base() {
    let p = TogetherAiProvider;
    let mut dep = deployment();
    dep.oxllm_params.api_base = Some("https://custom.example.com/v1/".to_string());
    assert_eq!(
        p.endpoint_url(&dep),
        "https://custom.example.com/v1/chat/completions"
    );
}

#[test]
fn apply_auth_uses_bearer_token() {
    let p = TogetherAiProvider;
    let creds = ResolvedCredentials::from_key("together-secret");
    let mut headers = std::collections::HashMap::new();
    ChatProvider::apply_auth(&p, &deployment(), &creds, &mut headers);
    assert_eq!(
        headers.get("authorization").map(String::as_str),
        Some("Bearer together-secret")
    );
}

#[test]
fn request_strips_together_prefix_from_model() {
    let p = TogetherAiProvider;
    let req = json!({
    "model": "anything-in-request",
    "messages": [{"role": "user", "content": "Hi"}],
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["model"], "meta-llama/Llama-3.3-70B-Instruct-Turbo");
    assert_eq!(out["messages"][0]["content"], "Hi");
}

#[test]
fn request_preserves_stream_flag() {
    let p = TogetherAiProvider;
    let req = json!({
    "model": "llama-v3-70b",
    "messages": [{"role": "user", "content": "Hi"}],
    "stream": true,
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["stream"], true);
}

#[test]
fn embedding_request_strips_together_prefix_from_model() {
    let p = TogetherAiProvider;
    let req = json!({
    "model": "together-bert",
    "input": "hello world",
    });
    let out = EmbeddingProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["model"], "meta-llama/Llama-3.3-70B-Instruct-Turbo");
    assert_eq!(out["input"], "hello world");
}

#[test]
fn response_is_identity() {
    let p = TogetherAiProvider;
    let resp = json!({
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    });
    let out =
        ChatProvider::transform_response(&p, &resp, "meta-llama/Llama-3.3-70B-Instruct-Turbo")
            .expect("ok");
    assert_eq!(out, resp);
}

#[test]
fn stream_chunk_parses_openai_format() {
    let p = TogetherAiProvider;
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
    let p = TogetherAiProvider;
    assert!(
        p.transform_stream_chunk("data: [DONE]")
            .expect("ok")
            .is_none()
    );
}

#[test]
fn provider_id_and_supports() {
    let p = TogetherAiProvider;
    assert_eq!(p.id(), "together_ai");
    assert!(p.supports(Endpoint::ChatCompletions));
    assert!(p.supports(Endpoint::Embeddings));
    assert!(p.embeddings().is_some());
}

#[test]
fn strip_together_prefix_handles_unknown() {
    assert_eq!(
        strip_together_prefix("together_ai/llama-v3-70b"),
        "llama-v3-70b"
    );
    assert_eq!(strip_together_prefix("llama-v3-70b"), "llama-v3-70b");
}

#[test]
fn supported_params_published() {
    let p = TogetherAiProvider;
    let params = ChatProvider::supported_params(&p).expect("should publish params");
    assert!(params.contains(&"temperature"));
    assert!(params.contains(&"tools"));
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
    let provider = TogetherAiProvider;
    let deployment = deployment();
    let model = "meta-llama/Llama-3.3-70B-Instruct-Turbo";

    let mut request: Value = serde_json::from_str(CANONICAL_GOLDEN_REQUEST)
        .expect("canonical golden request should be valid JSON");
    request["model"] = json!(model);
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

// --- Together-specific transforms ---

#[test]
fn request_drops_text_response_format() {
    let p = TogetherAiProvider;
    let req = json!({
        "model": "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "messages": [{"role": "user", "content": "Hi"}],
        "response_format": {"type": "text"},
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert!(out.get("response_format").is_none());
}

#[test]
fn request_preserves_json_object_response_format() {
    let p = TogetherAiProvider;
    let req = json!({
        "model": "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "messages": [{"role": "user", "content": "Hi"}],
        "response_format": {"type": "json_object"},
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["response_format"]["type"], "json_object");
}

#[test]
fn request_strips_response_only_fields_from_replayed_messages() {
    // The endpoint rejects unknown message fields; a reply carrying the
    // gateway's thinking blocks or provider extension fields cannot be
    // replayed verbatim (reference strip_litellm_response_only_fields).
    let req = json!({
        "model": "account/model",
        "messages": [{
            "role": "assistant",
            "content": "earlier",
            "reasoning_items": [{"id": "rs_1"}],
            "thinking_blocks": [{"type": "thinking", "thinking": "x", "signature": "s"}],
            "provider_specific_fields": {"a": 1},
            "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "extract", "arguments": "{}", "provider_specific_fields": {"c": 3}},
                "provider_specific_fields": {"b": 2}
            }]
        }, {
            "role": "user", "content": "next"
        }],
    });
    let out = ChatProvider::transform_request(&PROVIDER, &req, &deployment()).expect("ok");
    let message = &out["messages"][0];
    assert!(message.get("reasoning_items").is_none());
    assert!(message.get("thinking_blocks").is_none());
    assert!(message.get("provider_specific_fields").is_none());
    assert!(
        message["tool_calls"][0]
            .get("provider_specific_fields")
            .is_none()
    );
    assert_eq!(message["tool_calls"][0]["function"]["name"], "extract");
    assert_eq!(out["messages"][1]["content"], "next");
}
