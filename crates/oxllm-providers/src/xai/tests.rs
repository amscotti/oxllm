use super::*;
use oxllm_core::Endpoint;
use oxllm_core::deployment::{Deployment, OxllmParams};
use oxllm_core::provider::{ChatProvider, Provider, ResolvedCredentials};
use serde_json::json;

fn deployment() -> Deployment {
    Deployment {
        model_name: "grok-2".to_string(),
        oxllm_params: OxllmParams {
            model: "xai/grok-2".to_string(),
            api_key: Some("xai-key".to_string()),
            ..Default::default()
        },
        ..Default::default()
    }
}

#[test]
fn endpoint_url_uses_default_xai_base() {
    let p = XaiProvider;
    assert_eq!(
        p.endpoint_url(&deployment()),
        "https://api.x.ai/v1/chat/completions"
    );
}

#[test]
fn endpoint_url_respects_custom_base() {
    let p = XaiProvider;
    let mut dep = deployment();
    dep.oxllm_params.api_base = Some("https://custom.example.com/v1/".to_string());
    assert_eq!(
        p.endpoint_url(&dep),
        "https://custom.example.com/v1/chat/completions"
    );
}

#[test]
fn apply_auth_uses_bearer_token() {
    let p = XaiProvider;
    let creds = ResolvedCredentials::from_key("xai-secret");
    let mut headers = std::collections::HashMap::new();
    p.apply_auth(&deployment(), &creds, &mut headers);
    assert_eq!(
        headers.get("authorization").map(String::as_str),
        Some("Bearer xai-secret")
    );
}

#[test]
fn request_strips_xai_prefix_from_model() {
    let p = XaiProvider;
    let req = json!({
    "model": "grok-2",
    "messages": [{"role": "user", "content": "Hi"}],
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["model"], "grok-2");
    assert_eq!(out["messages"][0]["content"], "Hi");
}

#[test]
fn request_preserves_stream_flag() {
    let p = XaiProvider;
    let req = json!({
    "model": "grok-2",
    "messages": [{"role": "user", "content": "Hi"}],
    "stream": true,
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["stream"], true);
}

#[test]
fn response_is_identity() {
    let p = XaiProvider;
    let resp = json!({
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    });
    let out = p.transform_response(&resp, "grok-2").expect("ok");
    assert_eq!(out, resp);
}

#[test]
fn stream_chunk_parses_openai_format() {
    let p = XaiProvider;
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
    let p = XaiProvider;
    assert!(
        p.transform_stream_chunk("data: [DONE]")
            .expect("ok")
            .is_none()
    );
}

#[test]
fn provider_id_and_supports() {
    let p = XaiProvider;
    assert_eq!(p.id(), "xai");
    assert!(p.supports(Endpoint::ChatCompletions));
    assert!(!p.supports(Endpoint::Embeddings));
}

#[test]
fn strip_xai_prefix_handles_unknown() {
    assert_eq!(strip_xai_prefix("xai/grok-2"), "grok-2");
    assert_eq!(strip_xai_prefix("grok-2"), "grok-2");
}

#[test]
fn supported_params_published() {
    let p = XaiProvider;
    let params = p.supported_params().expect("should publish params");
    assert!(params.contains(&"temperature"));
    assert!(params.contains(&"tools"));
}

#[test]
fn request_preserves_newer_openai_params() {
    let p = XaiProvider;
    let req = json!({
        "model": "xai/grok-2",
        "messages": [{"role": "user", "content": "Hi"}],
        "logit_bias": {"50256": -100},
        "store": true,
        "metadata": {"user_id": "u_123"},
        "modalities": ["text"],
        "audio": {"voice": "alloy", "format": "wav"},
        "service_tier": "default",
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["logit_bias"]["50256"], -100);
    assert_eq!(out["store"], true);
    assert_eq!(out["metadata"]["user_id"], "u_123");
    assert_eq!(out["modalities"][0], "text");
    assert_eq!(out["audio"]["voice"], "alloy");
    assert_eq!(out["service_tier"], "default");
}

#[test]
fn request_preserves_stream_options() {
    let p = XaiProvider;
    let req = json!({
        "model": "xai/grok-2",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": true,
        "stream_options": {"include_usage": true},
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["stream_options"]["include_usage"], true);
}

#[test]
fn request_preserves_seed_and_user() {
    let p = XaiProvider;
    let req = json!({
        "model": "xai/grok-2",
        "messages": [{"role": "user", "content": "Hi"}],
        "seed": 42,
        "user": "test_user",
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["seed"], 42);
    assert_eq!(out["user"], "test_user");
}

#[test]
fn request_preserves_response_format() {
    let p = XaiProvider;
    let req = json!({
        "model": "xai/grok-2",
        "messages": [{"role": "user", "content": "Hi"}],
        "response_format": {"type": "json_object"},
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["response_format"]["type"], "json_object");
}

#[test]
fn request_preserves_max_completion_tokens() {
    let p = XaiProvider;
    let req = json!({
        "model": "xai/grok-2",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_completion_tokens": 4096,
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["max_completion_tokens"], 4096);
}

#[test]
fn request_preserves_reasoning_effort() {
    let p = XaiProvider;
    let req = json!({
        "model": "xai/grok-2",
        "messages": [{"role": "user", "content": "Hi"}],
        "reasoning_effort": "high",
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["reasoning_effort"], "high");
}

#[test]
fn request_preserves_parallel_tool_calls() {
    let p = XaiProvider;
    let req = json!({
        "model": "xai/grok-2",
        "messages": [{"role": "user", "content": "Hi"}],
        "parallel_tool_calls": true,
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["parallel_tool_calls"], true);
}

#[test]
fn request_drops_unknown_params() {
    let p = XaiProvider;
    let req = json!({
        "model": "xai/grok-2",
        "messages": [{"role": "user", "content": "Hi"}],
        "temperature": 0.5,
        "some_unknown_param": "should_be_dropped",
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["temperature"], 0.5);
    assert!(out.get("some_unknown_param").is_none());
}

#[test]
fn strip_xai_prefix_with_various_inputs() {
    assert_eq!(strip_xai_prefix("xai/grok-2"), "grok-2");
    assert_eq!(strip_xai_prefix("xai/grok-3"), "grok-3");
    assert_eq!(strip_xai_prefix("xai/grok-3-mini"), "grok-3-mini");
    assert_eq!(strip_xai_prefix("grok-2"), "grok-2");
    assert_eq!(strip_xai_prefix("openai/gpt-4o"), "openai/gpt-4o");
}

#[test]
fn supported_params_includes_reasoning_effort() {
    let p = XaiProvider;
    let params = p.supported_params().expect("should publish params");
    assert!(params.contains(&"reasoning_effort"));
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
    let provider = XaiProvider;
    let deployment = deployment();
    let model = "grok-2";

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

// --- Wildcards ---

#[test]
fn wildcard_deployment_preserves_requested_model() {
    let p = XaiProvider;
    let mut dep = deployment();
    dep.oxllm_params.model = "xai/*".to_string();
    let req = json!({
        "model": "xai/grok-4",
        "messages": [{"role": "user", "content": "Hi"}],
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(out["model"], "grok-4");
}

#[test]
fn provider_wildcard_with_unmatched_prefix_preserves_requested_model() {
    let p = XaiProvider;
    let mut dep = deployment();
    dep.oxllm_params.model = "openai/*".to_string();
    let req = json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hi"}],
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(out["model"], "gpt-4o");
}

#[test]
fn provider_wildcard_preserves_full_path_when_unmatched() {
    let p = XaiProvider;
    let mut dep = deployment();
    dep.oxllm_params.model = "openai/*".to_string();
    let req = json!({
        "model": "openai/gpt-4o",
        "messages": [{"role": "user", "content": "Hi"}],
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    // strip_model only strips `xai/`, so the full `openai/gpt-4o` path survives.
    assert_eq!(out["model"], "openai/gpt-4o");
}

#[test]
fn request_deployment_with_custom_model_respects_strip() {
    let p = XaiProvider;
    let mut dep = deployment();
    dep.oxllm_params.model = "xai/custom-model".to_string();
    let req = json!({
        "model": "ignored-request-model",
        "messages": [{"role": "user", "content": "Hi"}],
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(out["model"], "custom-model");
}

// --- Request params ---

#[test]
fn request_preserves_tools_and_tool_choice() {
    let p = XaiProvider;
    let req = json!({
        "model": "xai/grok-4",
        "messages": [{"role": "user", "content": "What's the weather?"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                    },
                },
            }
        ],
        "tool_choice": "auto",
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["tool_choice"], "auto");
    assert_eq!(out["tools"][0]["function"]["name"], "get_weather");
}

#[test]
fn request_preserves_response_format_json_schema() {
    let p = XaiProvider;
    let req = json!({
        "model": "xai/grok-4",
        "messages": [{"role": "user", "content": "Hi"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "x",
                "strict": true,
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                },
            },
        },
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["response_format"]["type"], "json_schema");
    assert_eq!(out["response_format"]["json_schema"]["name"], "x");
    assert_eq!(out["response_format"]["json_schema"]["strict"], true);
    assert_eq!(
        out["response_format"]["json_schema"]["schema"]["properties"]["answer"]["type"],
        "string"
    );
}

#[test]
fn request_preserves_stop_sequence_array() {
    let p = XaiProvider;
    let req = json!({
        "model": "xai/grok-4",
        "messages": [{"role": "user", "content": "Hi"}],
        "stop": ["stop1", "stop2"],
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["stop"][0], "stop1");
    assert_eq!(out["stop"][1], "stop2");
}

#[test]
fn request_preserves_stop_sequence_string() {
    let p = XaiProvider;
    let req = json!({
        "model": "xai/grok-4",
        "messages": [{"role": "user", "content": "Hi"}],
        "stop": "\n",
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["stop"], "\n");
}

#[test]
fn request_preserves_sampling_params() {
    let p = XaiProvider;
    let req = json!({
        "model": "xai/grok-4",
        "messages": [{"role": "user", "content": "Hi"}],
        "temperature": 0.7,
        "top_p": 0.9,
        "presence_penalty": 0.5,
        "frequency_penalty": 0.2,
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["temperature"], 0.7);
    assert_eq!(out["top_p"], 0.9);
    assert_eq!(out["presence_penalty"], 0.5);
    assert_eq!(out["frequency_penalty"], 0.2);
}

#[test]
fn request_preserves_logprobs() {
    let p = XaiProvider;
    let req = json!({
        "model": "xai/grok-4",
        "messages": [{"role": "user", "content": "Hi"}],
        "logprobs": true,
        "top_logprobs": 3,
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["logprobs"], true);
    assert_eq!(out["top_logprobs"], 3);
}

#[test]
fn request_preserves_n() {
    let p = XaiProvider;
    let req = json!({
        "model": "xai/grok-4",
        "messages": [{"role": "user", "content": "Hi"}],
        "n": 3,
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["n"], 3);
}

#[test]
fn request_drops_null_params() {
    let p = XaiProvider;
    let req = json!({
        "model": "xai/grok-4",
        "messages": [{"role": "user", "content": "Hi"}],
        "temperature": null,
        "max_tokens": 5,
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert!(out.get("temperature").is_none());
    assert_eq!(out["max_tokens"], 5);
}

#[test]
fn transform_request_preserves_newer_openai_params() {
    let p = XaiProvider;
    let req = json!({
        "model": "xai/grok-4",
        "messages": [{"role": "user", "content": "Hi"}],
        "logit_bias": {"50256": -100},
        "store": true,
        "metadata": {"user_id": "u_123"},
        "modalities": ["text", "audio"],
        "audio": {"voice": "alloy", "format": "wav"},
        "service_tier": "default",
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["logit_bias"]["50256"], -100);
    assert_eq!(out["store"], true);
    assert_eq!(out["metadata"]["user_id"], "u_123");
    assert_eq!(out["modalities"][0], "text");
    assert_eq!(out["modalities"][1], "audio");
    assert_eq!(out["audio"]["voice"], "alloy");
    assert_eq!(out["service_tier"], "default");
}

#[test]
fn request_preserves_max_tokens() {
    let p = XaiProvider;
    let req = json!({
        "model": "xai/grok-4",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 500,
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["max_tokens"], 500);
}

#[test]
fn request_preserves_frequency_penalty_only() {
    let p = XaiProvider;
    let req = json!({
        "model": "xai/grok-4",
        "messages": [{"role": "user", "content": "Hi"}],
        "frequency_penalty": 0.5,
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["frequency_penalty"], 0.5);
    assert!(out.get("presence_penalty").is_none());
}

// --- Response ---

#[test]
fn transform_response_preserves_tool_calls() {
    let p = XaiProvider;
    let resp = json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{\"location\":\"Paris\"}"},
                }],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    });
    let out = ChatProvider::transform_response(&p, &resp, "grok-4").expect("ok");
    assert_eq!(
        out["choices"][0]["message"]["tool_calls"][0]["id"],
        "call_1"
    );
    assert_eq!(
        out["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
        "get_weather"
    );
    assert_eq!(out["choices"][0]["finish_reason"], "tool_calls");
}

#[test]
fn transform_response_normalizes_null_tokens() {
    let p = XaiProvider;
    let resp = json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": null, "completion_tokens": null, "total_tokens": null},
    });
    let out = ChatProvider::transform_response(&p, &resp, "grok-4").expect("ok");
    assert_eq!(out["usage"]["prompt_tokens"], 0);
    assert_eq!(out["usage"]["completion_tokens"], 0);
    assert_eq!(out["usage"]["total_tokens"], 0);
}

#[test]
fn transform_response_normalizes_null_token_details() {
    let p = XaiProvider;
    let resp = json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 2,
            "total_tokens": 7,
            "completion_tokens_details": {"reasoning_tokens": null},
        },
    });
    let out = ChatProvider::transform_response(&p, &resp, "grok-4").expect("ok");
    assert_eq!(
        out["usage"]["completion_tokens_details"]["reasoning_tokens"],
        0
    );
}

#[test]
fn transform_response_preserves_full_structure() {
    let p = XaiProvider;
    let resp = json!({
        "id": "chatcmpl-abc",
        "object": "chat.completion",
        "created": 1729459200,
        "model": "grok-4",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "The capital of France is Paris."},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
    });
    let out = ChatProvider::transform_response(&p, &resp, "grok-4").expect("ok");
    assert_eq!(out, resp);
}

#[test]
fn transform_response_with_empty_choices() {
    let p = XaiProvider;
    let resp = json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    });
    let out = ChatProvider::transform_response(&p, &resp, "grok-4").expect("ok");
    assert_eq!(out["choices"], json!([]));
    assert_eq!(out["usage"]["total_tokens"], 7);
}

// --- Stream ---

#[test]
fn stream_chunk_with_finish_reason() {
    let p = XaiProvider;
    let raw = "data: {\"id\":\"x\",\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}]}";
    let out = p
        .transform_stream_chunk(raw)
        .expect("ok")
        .expect("should produce value");
    assert_eq!(out["choices"][0]["delta"]["content"], "hi");
    assert_eq!(out["choices"][0]["finish_reason"], "stop");
}

#[test]
fn stream_chunk_with_usage() {
    let p = XaiProvider;
    let raw = "data: {\"id\":\"x\",\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":2,\"total_tokens\":7}}";
    let out = p
        .transform_stream_chunk(raw)
        .expect("ok")
        .expect("should produce value");
    assert_eq!(out["id"], "x");
    assert_eq!(out["usage"]["prompt_tokens"], 5);
    assert_eq!(out["usage"]["completion_tokens"], 2);
    assert_eq!(out["usage"]["total_tokens"], 7);
}

#[test]
fn stream_chunk_with_malformed_json() {
    let p = XaiProvider;
    let raw = "data: {not valid json";
    assert!(p.transform_stream_chunk(raw).is_err());
}

// --- Prefix edge cases ---

#[test]
fn strip_xai_prefix_edge_cases() {
    assert_eq!(strip_xai_prefix(""), "");
    assert_eq!(strip_xai_prefix("xai"), "xai");
    assert_eq!(strip_xai_prefix("/"), "/");
    assert_eq!(
        strip_xai_prefix("xai/sub/provider/model"),
        "sub/provider/model"
    );
}

#[test]
fn strip_xai_prefix_mixed_case() {
    assert_eq!(strip_xai_prefix("XAI/grok-4"), "XAI/grok-4");
}

// --- Provider metadata ---

#[test]
fn supported_params_includes_newer_openai_params() {
    let p = XaiProvider;
    let params = p.supported_params().expect("should publish params");
    for key in [
        "logit_bias",
        "store",
        "metadata",
        "modalities",
        "audio",
        "service_tier",
    ] {
        assert!(params.contains(&key), "`{key}` must be in supported params");
    }
}

// --- xAI-specific transforms ---

#[test]
fn request_strips_name_from_non_user_messages() {
    let p = XaiProvider;
    let req = json!({
        "model": "xai/grok-4",
        "messages": [
            {"role": "user", "name": "Alice", "content": "Hi"},
            {"role": "assistant", "name": "Bob", "content": "Hello"},
            {"role": "tool", "name": "Toolie", "content": "result"},
        ],
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    // user keeps name; assistant and tool lose it.
    assert_eq!(out["messages"][0]["name"], "Alice");
    assert_eq!(out["messages"][0]["content"], "Hi");
    assert!(out["messages"][1].get("name").is_none());
    assert_eq!(out["messages"][1]["content"], "Hello");
    assert!(out["messages"][2].get("name").is_none());
}

#[test]
fn request_strips_strict_from_tool_function_definitions() {
    let p = XaiProvider;
    let req = json!({
        "model": "xai/grok-4",
        "messages": [{"role": "user", "content": "Hi"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "strict": true,
                    "parameters": {"type": "object"},
                },
            },
            {
                "type": "function",
                "function": {"name": "f2", "strict": false},
            },
        ],
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert!(out["tools"][0]["function"].get("strict").is_none());
    assert_eq!(out["tools"][0]["function"]["name"], "get_weather");
    assert!(out["tools"][1]["function"].get("strict").is_none());
}

#[test]
fn transform_response_fixes_empty_finish_reason_for_tool_calls() {
    let p = XaiProvider;
    let resp = json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{}"},
                }],
            },
            "finish_reason": "",
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    });
    let out = ChatProvider::transform_response(&p, &resp, "grok-4").expect("ok");
    assert_eq!(out["choices"][0]["finish_reason"], "tool_calls");
}

#[test]
fn transform_response_leaves_finish_reason_when_no_tool_calls() {
    let p = XaiProvider;
    let resp = json!({
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Hi"},
            "finish_reason": "",
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    });
    let out = ChatProvider::transform_response(&p, &resp, "grok-4").expect("ok");
    // No tool_calls → finish_reason is left as-is (empty), not forced.
    assert_eq!(out["choices"][0]["finish_reason"], "");
}

#[test]
fn transform_response_folds_reasoning_tokens_into_completion() {
    let p = XaiProvider;
    // xAI additive accounting: total = prompt + completion + reasoning.
    let resp = json!({
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Hi"},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 100,
            "total_tokens": 250,
            "completion_tokens_details": {"reasoning_tokens": 50},
        },
    });
    let out = ChatProvider::transform_response(&p, &resp, "grok-4").expect("ok");
    assert_eq!(out["usage"]["completion_tokens"], 150);
    assert_eq!(out["usage"]["total_tokens"], 250);
    assert_eq!(out["usage"]["prompt_tokens"], 100);
}

#[test]
fn transform_response_does_not_fold_when_already_subset() {
    let p = XaiProvider;
    // OpenAI convention: reasoning already inside completion.
    let resp = json!({
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Hi"},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 150,
            "total_tokens": 250,
            "completion_tokens_details": {"reasoning_tokens": 50},
        },
    });
    let out = ChatProvider::transform_response(&p, &resp, "grok-4").expect("ok");
    assert_eq!(out["usage"]["completion_tokens"], 150);
    assert_eq!(out["usage"]["total_tokens"], 250);
}

#[test]
fn stream_chunk_preserves_empty_choices_on_usage_only_chunk() {
    let p = XaiProvider;
    let raw = "data: {\"id\":\"x\",\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":2,\"total_tokens\":7}}";
    let out = p
        .transform_stream_chunk(raw)
        .expect("ok")
        .expect("should produce value");
    assert_eq!(out["choices"], serde_json::json!([]));
    assert_eq!(out["usage"]["prompt_tokens"], 5);
}

#[test]
fn stream_chunk_folds_reasoning_tokens_in_usage() {
    let p = XaiProvider;
    let raw = "data: {\"id\":\"x\",\"choices\":[],\"usage\":{\"prompt_tokens\":100,\"completion_tokens\":100,\"total_tokens\":250,\"completion_tokens_details\":{\"reasoning_tokens\":50}}}";
    let out = p
        .transform_stream_chunk(raw)
        .expect("ok")
        .expect("should produce value");
    assert_eq!(out["usage"]["completion_tokens"], 150);
    assert_eq!(out["usage"]["total_tokens"], 250);
}

#[test]
fn stream_chunk_with_choices_and_no_usage_is_unchanged() {
    let p = XaiProvider;
    let raw = "data: {\"id\":\"x\",\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}";
    let out = p
        .transform_stream_chunk(raw)
        .expect("ok")
        .expect("should produce value");
    assert_eq!(out["choices"][0]["delta"]["content"], "hi");
}
