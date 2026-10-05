use super::*;
use oxllm_core::Endpoint;
use oxllm_core::deployment::{Deployment, OxllmParams};
use oxllm_core::provider::{ChatProvider, Provider, ResolvedCredentials};
use serde_json::json;

/// Replicate model ids use an `owner/model-name` structure (e.g.
/// `meta/llama-3.1-8b-instruct`). The `replicate/` provider prefix is layered
/// on top of that, so a fully-qualified id is `replicate/meta/llama-3.1-8b-instruct`.
fn deployment() -> Deployment {
    Deployment {
        model_name: "meta/llama-3.1-8b-instruct".to_string(),
        oxllm_params: OxllmParams {
            model: "replicate/meta/llama-3.1-8b-instruct".to_string(),
            api_key: Some("replicate-key".to_string()),
            ..Default::default()
        },
        ..Default::default()
    }
}

#[test]
fn endpoint_url_uses_default_replicate_base() {
    let p = ReplicateProvider;
    assert_eq!(
        p.endpoint_url(&deployment()),
        "https://api.replicate.com/v1/chat/completions"
    );
}

#[test]
fn endpoint_url_respects_custom_base() {
    let p = ReplicateProvider;
    let mut dep = deployment();
    dep.oxllm_params.api_base = Some("https://custom.example.com/v1/".to_string());
    assert_eq!(
        p.endpoint_url(&dep),
        "https://custom.example.com/v1/chat/completions"
    );
}

#[test]
fn apply_auth_uses_bearer_token() {
    let p = ReplicateProvider;
    let creds = ResolvedCredentials::from_key("replicate-secret");
    let mut headers = std::collections::HashMap::new();
    p.apply_auth(&deployment(), &creds, &mut headers);
    assert_eq!(
        headers.get("authorization").map(String::as_str),
        Some("Bearer replicate-secret")
    );
}

#[test]
fn request_strips_replicate_prefix_preserving_owner_model() {
    // The defining peculiarity: only the leading `replicate/` segment is
    // dropped; the `owner/model-name` path must survive intact.
    let p = ReplicateProvider;
    let req = json!({
    "model": "meta/llama-3.1-8b-instruct",
    "messages": [{"role": "user", "content": "Hi"}],
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["model"], "meta/llama-3.1-8b-instruct");
    assert_eq!(out["messages"][0]["content"], "Hi");
}

#[test]
fn wildcard_deployment_preserves_requested_model() {
    let p = ReplicateProvider;
    let mut dep = deployment();
    dep.oxllm_params.model = "replicate/*".to_string();
    let req = json!({
        "model": "meta/llama-3.1-8b-instruct",
        "messages": [{"role": "user", "content": "Hi"}]
    });
    let out = p.transform_request(&req, &dep).expect("ok");
    assert_eq!(out["model"], "meta/llama-3.1-8b-instruct");
}

#[test]
fn wildcard_deployment_preserves_owner_model_from_body() {
    // Even when the client sends a fully-qualified `replicate/owner/model`
    // string against a wildcard deployment, only the `replicate/` segment is
    // stripped.
    let p = ReplicateProvider;
    let mut dep = deployment();
    dep.oxllm_params.model = "replicate/*".to_string();
    let req = json!({
        "model": "replicate/meta/llama-3.1-8b-instruct",
        "messages": [{"role": "user", "content": "Hi"}]
    });
    let out = p.transform_request(&req, &dep).expect("ok");
    assert_eq!(out["model"], "meta/llama-3.1-8b-instruct");
}

#[test]
fn provider_wildcard_with_unmatched_prefix_preserves_requested_model() {
    // The documented `provider/*` pattern must work even when the configured
    // model's prefix doesn't match this provider's strip_prefix.
    let p = ReplicateProvider;
    let mut dep = deployment();
    dep.oxllm_params.model = "openai/*".to_string();
    let req = json!({
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "Hi"}]
    });
    let out = p.transform_request(&req, &dep).expect("ok");
    assert_eq!(out["model"], "gpt-4o-mini");
}

#[test]
fn provider_wildcard_preserves_full_path_when_unmatched() {
    let p = ReplicateProvider;
    let mut dep = deployment();
    dep.oxllm_params.model = "openai/*".to_string();
    let req = json!({
        "model": "openai/gpt-4o",
        "messages": [{"role": "user", "content": "Hi"}]
    });
    let out = p.transform_request(&req, &dep).expect("ok");
    // strip_model only strips `replicate/`, so `openai/gpt-4o` survives intact.
    assert_eq!(out["model"], "openai/gpt-4o");
}

#[test]
fn request_preserves_stream_flag() {
    let p = ReplicateProvider;
    let req = json!({
    "model": "meta/llama-3.1-8b-instruct",
    "messages": [{"role": "user", "content": "Hi"}],
    "stream": true,
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["stream"], true);
}

#[test]
fn request_passes_known_params_and_drops_unknown() {
    let p = ReplicateProvider;
    let req = json!({
    "model": "meta/llama-3.1-8b-instruct",
    "messages": [{"role": "user", "content": "Hi"}],
    "temperature": 0.5,
    "max_tokens": 100,
    "stream": false,
    "n": 2,
    "some_unknown_param": "ignored",
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["temperature"], 0.5);
    assert_eq!(out["max_tokens"], 100);
    assert_eq!(out["n"], 2);
    assert!(out.get("some_unknown_param").is_none());
}

#[test]
fn response_is_identity() {
    let p = ReplicateProvider;
    let resp = json!({
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    });
    let out = p
        .transform_response(&resp, "meta/llama-3.1-8b-instruct")
        .expect("ok");
    assert_eq!(out, resp);
}

#[test]
fn stream_chunk_parses_openai_format() {
    let p = ReplicateProvider;
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
    let p = ReplicateProvider;
    assert!(
        p.transform_stream_chunk("data: [DONE]")
            .expect("ok")
            .is_none()
    );
}

#[test]
fn provider_id_and_supports() {
    let p = ReplicateProvider;
    assert_eq!(p.id(), "replicate");
    assert!(p.supports(Endpoint::ChatCompletions));
    assert!(!p.supports(Endpoint::Embeddings));
}

#[test]
fn strip_replicate_prefix_handles_unknown() {
    assert_eq!(
        strip_replicate_prefix("replicate/meta/llama-3.1-8b-instruct"),
        "meta/llama-3.1-8b-instruct"
    );
    assert_eq!(
        strip_replicate_prefix("meta/llama-3.1-8b-instruct"),
        "meta/llama-3.1-8b-instruct"
    );
}

#[test]
fn supported_params_published() {
    let p = ReplicateProvider;
    let params = p.supported_params().expect("should publish params");
    assert!(params.contains(&"temperature"));
    assert!(params.contains(&"tools"));
}

#[test]
fn supported_params_includes_newer_openai_params() {
    // `drop_params: true` strips anything not in supported_params BEFORE the
    // transform runs. The newer OpenAI params must be listed so they survive
    // the strip and reach upstreams that accept them.
    let p = ReplicateProvider;
    let params = p.supported_params().expect("should publish params");
    for key in [
        "logit_bias",
        "store",
        "metadata",
        "modalities",
        "audio",
        "service_tier",
    ] {
        assert!(
            params.contains(&key),
            "`{key}` must be in SUPPORTED_CHAT_PARAMS so drop_params keeps it"
        );
    }
}

#[test]
fn transform_request_preserves_newer_openai_params() {
    // End-to-end check that the build_chat_request path actually forwards the
    // newly-added params to the upstream body.
    let p = ReplicateProvider;
    let req = json!({
    "model": "replicate/meta/llama-3.1-8b-instruct",
    "messages": [{"role": "user", "content": "Hi"}],
    "logit_bias": {"50256": -100},
    "store": true,
    "metadata": {"user_id": "u_123"},
    "modalities": ["text"],
    "audio": {"voice": "alloy", "format": "wav"},
    "service_tier": "default",
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["logit_bias"]["50256"], -100);
    assert_eq!(out["store"], true);
    assert_eq!(out["metadata"]["user_id"], "u_123");
    assert_eq!(out["modalities"][0], "text");
    assert_eq!(out["audio"]["voice"], "alloy");
    assert_eq!(out["service_tier"], "default");
}

const GOLDEN_REQUEST_001: &str = include_str!("goldens/chat_request_001.json");
const GOLDEN_RESPONSE_001: &str = include_str!("goldens/chat_response_001.json");

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
    let provider = ReplicateProvider;
    let deployment = deployment();

    let cases: &[(&str, &str, bool)] = &[
        ("chat_request_001", GOLDEN_REQUEST_001, true),
        ("chat_response_001", GOLDEN_RESPONSE_001, false),
    ];

    for (name, golden, is_request) in cases {
        let input: Value = serde_json::from_str(golden)
            .unwrap_or_else(|e| panic!("golden {name} should be valid JSON: {e}"));
        let output = if *is_request {
            ChatProvider::transform_request(&provider, &input, &deployment)
                .unwrap_or_else(|e| panic!("golden {name} request transform failed: {e}"))
        } else {
            ChatProvider::transform_response(&provider, &input, "meta/llama-3.1-8b-instruct")
                .unwrap_or_else(|e| panic!("golden {name} response transform failed: {e}"))
        };
        assert_semantically_equal(&output, &input);
    }
}

// --- Additional request parameter tests (ported from LiteLLM test_replicate.py) ---

#[test]
fn request_preserves_response_format_json_object() {
    let p = ReplicateProvider;
    let req = json!({
        "model": "replicate/meta/llama-3.1-8b-instruct",
        "messages": [{"role": "user", "content": "Hi"}],
        "response_format": {"type": "json_object"},
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["response_format"]["type"], "json_object");
}

#[test]
fn request_preserves_response_format_json_schema() {
    let p = ReplicateProvider;
    let req = json!({
        "model": "replicate/meta/llama-3.1-8b-instruct",
        "messages": [{"role": "user", "content": "Hi"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "weather",
                "schema": {
                    "type": "object",
                    "properties": {"temp": {"type": "number"}},
                },
            },
        },
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["response_format"]["type"], "json_schema");
    assert_eq!(out["response_format"]["json_schema"]["name"], "weather");
}

#[test]
fn request_preserves_seed() {
    let p = ReplicateProvider;
    let req = json!({
        "model": "replicate/meta/llama-3.1-8b-instruct",
        "messages": [{"role": "user", "content": "Hi"}],
        "seed": 42,
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["seed"], 42);
}

#[test]
fn request_preserves_user() {
    let p = ReplicateProvider;
    let req = json!({
        "model": "replicate/meta/llama-3.1-8b-instruct",
        "messages": [{"role": "user", "content": "Hi"}],
        "user": "test_user_123",
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["user"], "test_user_123");
}

#[test]
fn request_preserves_stream_options() {
    let p = ReplicateProvider;
    let req = json!({
        "model": "replicate/meta/llama-3.1-8b-instruct",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": true,
        "stream_options": {"include_usage": true},
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["stream_options"]["include_usage"], true);
}

#[test]
fn request_preserves_reasoning_effort() {
    let p = ReplicateProvider;
    let req = json!({
        "model": "replicate/meta/llama-3.1-8b-instruct",
        "messages": [{"role": "user", "content": "Hi"}],
        "reasoning_effort": "high",
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["reasoning_effort"], "high");
}

#[test]
fn request_preserves_max_completion_tokens() {
    let p = ReplicateProvider;
    let req = json!({
        "model": "replicate/meta/llama-3.1-8b-instruct",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_completion_tokens": 2048,
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["max_completion_tokens"], 2048);
}

#[test]
fn strip_replicate_prefix_with_various_inputs() {
    assert_eq!(
        strip_replicate_prefix("replicate/meta/llama-3.1-8b-instruct"),
        "meta/llama-3.1-8b-instruct"
    );
    assert_eq!(
        strip_replicate_prefix("replicate/deepseek-ai/deepseek-v3"),
        "deepseek-ai/deepseek-v3"
    );
    assert_eq!(
        strip_replicate_prefix("replicate/meta/llama-2-70b-chat"),
        "meta/llama-2-70b-chat"
    );
    assert_eq!(
        strip_replicate_prefix("meta/llama-3.1-8b-instruct"),
        "meta/llama-3.1-8b-instruct"
    );
}

#[test]
fn request_deployment_with_custom_model_respects_strip() {
    let p = ReplicateProvider;
    let mut dep = deployment();
    dep.oxllm_params.model = "replicate/meta/custom-model-v1".to_string();
    dep.model_name = "meta/custom-model-v1".to_string();
    let req = json!({
        "model": "replicate/meta/custom-model-v1",
        "messages": [{"role": "user", "content": "Hi"}],
    });
    let out = p.transform_request(&req, &dep).expect("ok");
    assert_eq!(out["model"], "meta/custom-model-v1");
}

// --- LiteLLM output-format tests (ported from test_replicate.py) ---
//
// LiteLLM's legacy Replicate integration uses the /predictions endpoint and
// returns `output` as either a list of token chunks or a single string. The
// OpenAI-compatible surface normalizes both into a chat-completion `content`
// string. These tests assert that our identity transform forwards whatever
// content the upstream produced, mirroring LiteLLM's string-join behavior.

#[test]
fn transform_response_preserves_list_joined_content() {
    // LiteLLM joins `["Hello", " ", "world"]` into `"Hello world"`. On the
    // OpenAI-compatible surface the upstream already joins it into
    // `message.content`, so the identity transform preserves it verbatim.
    let p = ReplicateProvider;
    let resp = json!({
    "id": "chatcmpl-list",
    "object": "chat.completion",
    "choices": [{
        "index": 0,
        "message": {"role": "assistant", "content": "Hello world"},
        "finish_reason": "stop"
    }],
    });
    let out = p
        .transform_response(&resp, "meta/llama-2-70b-chat")
        .expect("ok");
    assert_eq!(out["choices"][0]["message"]["content"], "Hello world");
}

#[test]
fn transform_response_preserves_string_content() {
    // DeepSeek-style string output, as exercised by LiteLLM's
    // `test_transform_response_string_output`.
    let p = ReplicateProvider;
    let resp = json!({
    "id": "chatcmpl-str",
    "object": "chat.completion",
    "choices": [{
        "index": 0,
        "message": {"role": "assistant", "content": "Hello from DeepSeek"},
        "finish_reason": "stop"
    }],
    });
    let out = p
        .transform_response(&resp, "deepseek-ai/deepseek-v3")
        .expect("ok");
    assert_eq!(
        out["choices"][0]["message"]["content"],
        "Hello from DeepSeek"
    );
}
