use super::*;
use oxllm_core::Endpoint;
use oxllm_core::deployment::{Deployment, OxllmParams};
use oxllm_core::provider::{ChatProvider, Provider, ResolvedCredentials};
use serde_json::json;

fn deployment() -> Deployment {
    Deployment {
        model_name: "llama-3.1-70b".to_string(),
        oxllm_params: OxllmParams {
            model: "groq/llama-3.1-70b".to_string(),
            api_key: Some("groq-key".to_string()),
            ..Default::default()
        },
        ..Default::default()
    }
}

#[test]
fn endpoint_url_uses_default_groq_base() {
    let p = GroqProvider;
    assert_eq!(
        p.endpoint_url(&deployment()),
        "https://api.groq.com/openai/v1/chat/completions"
    );
}

#[test]
fn endpoint_url_respects_custom_base() {
    let p = GroqProvider;
    let mut dep = deployment();
    dep.oxllm_params.api_base = Some("https://custom.example.com/v1/".to_string());
    assert_eq!(
        p.endpoint_url(&dep),
        "https://custom.example.com/v1/chat/completions"
    );
}

#[test]
fn apply_auth_uses_bearer_token() {
    let p = GroqProvider;
    let creds = ResolvedCredentials::from_key("groq-secret");
    let mut headers = std::collections::HashMap::new();
    p.apply_auth(&deployment(), &creds, &mut headers);
    assert_eq!(
        headers.get("authorization").map(String::as_str),
        Some("Bearer groq-secret")
    );
}

#[test]
fn request_strips_groq_prefix_from_model() {
    let p = GroqProvider;
    let req = json!({
    "model": "llama-3.1-70b",
    "messages": [{"role": "user", "content": "Hi"}],
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["model"], "llama-3.1-70b");
    assert_eq!(out["messages"][0]["content"], "Hi");
}

#[test]
fn wildcard_deployment_preserves_requested_model() {
    let p = GroqProvider;
    let mut dep = deployment();
    dep.oxllm_params.model = "groq/*".to_string();
    let req = json!({
        "model": "groq/llama-4-scout",
        "messages": [{"role": "user", "content": "Hi"}]
    });
    let out = p.transform_request(&req, &dep).expect("ok");
    assert_eq!(out["model"], "llama-4-scout");
}

#[test]
fn provider_wildcard_with_unmatched_prefix_preserves_requested_model() {
    // The documented `provider/*` pattern must work even when the configured
    // model's prefix doesn't match this provider's strip_prefix. Before the
    // fix, a configured model like `openai/*` on a Groq deployment was routed
    // as the literal string `openai/*`; now the body's `model` field wins.
    let p = GroqProvider;
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
    // When the client sends a model whose prefix doesn't match the provider's
    // strip_prefix either, the wildcard still kicks in (the whole client
    // string is forwarded after the configured `/*` is treated as wildcard).
    let p = GroqProvider;
    let mut dep = deployment();
    dep.oxllm_params.model = "openai/*".to_string();
    let req = json!({
        "model": "openai/gpt-4o",
        "messages": [{"role": "user", "content": "Hi"}]
    });
    let out = p.transform_request(&req, &dep).expect("ok");
    // strip_model only strips `groq/`, so `openai/gpt-4o` survives intact.
    assert_eq!(out["model"], "openai/gpt-4o");
}

#[test]
fn request_preserves_stream_flag() {
    let p = GroqProvider;
    let req = json!({
    "model": "llama-3.1-70b",
    "messages": [{"role": "user", "content": "Hi"}],
    "stream": true,
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["stream"], true);
}

#[test]
fn request_passes_known_params_and_drops_unknown() {
    let p = GroqProvider;
    let req = json!({
    "model": "llama-3.1-70b",
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
    let p = GroqProvider;
    let resp = json!({
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    });
    let out = p.transform_response(&resp, "llama-3.1-70b").expect("ok");
    assert_eq!(out, resp);
}

#[test]
fn stream_chunk_parses_openai_format() {
    let p = GroqProvider;
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
    let p = GroqProvider;
    assert!(
        p.transform_stream_chunk("data: [DONE]")
            .expect("ok")
            .is_none()
    );
}

#[test]
fn stream_reasoning_normalization_preserves_existing_text_and_ignores_empty_alternates() {
    for existing in [
        None,
        Some(Value::Null),
        Some(json!("")),
        Some(json!("original")),
    ] {
        for alternate in [
            Value::Null,
            json!(""),
            json!("alternate"),
            json!(" "),
            json!(false),
        ] {
            let mut delta = json!({"content": "answer", "reasoning": alternate});
            if let Some(value) = &existing {
                delta["reasoning_content"] = value.clone();
            }
            let raw = format!("data: {}", json!({"choices": [{"delta": delta}]}));
            let body = GroqProvider.transform_stream_chunk(&raw).unwrap().unwrap();
            let normalized = &body["choices"][0]["delta"];
            let expected = if existing
                .as_ref()
                .is_some_and(|value| value.as_str() == Some("original"))
            {
                existing.clone()
            } else if alternate.as_str().is_some_and(|text| !text.is_empty()) {
                Some(alternate.clone())
            } else {
                existing.clone()
            };
            assert_eq!(normalized.get("reasoning_content"), expected.as_ref());
            assert!(normalized.get("reasoning").is_none());
            assert_eq!(normalized["content"], "answer");
        }
    }
}

#[test]
fn provider_id_and_supports() {
    let p = GroqProvider;
    assert_eq!(p.id(), "groq");
    assert!(p.supports(Endpoint::ChatCompletions));
    assert!(p.supports(Endpoint::AudioTranscriptions));
    assert!(p.supports(Endpoint::AudioSpeech));
    assert!(!p.supports(Endpoint::Embeddings));
}

#[test]
fn strip_groq_prefix_handles_unknown() {
    assert_eq!(strip_groq_prefix("groq/llama-3.1-70b"), "llama-3.1-70b");
    assert_eq!(strip_groq_prefix("llama-3.1-70b"), "llama-3.1-70b");
}

#[test]
fn supported_params_published() {
    let p = GroqProvider;
    let params = p.supported_params().expect("should publish params");
    assert!(params.contains(&"temperature"));
    assert!(params.contains(&"tools"));
}

#[test]
fn supported_params_includes_reasoning_effort() {
    let p = GroqProvider;
    let params = p.supported_params().expect("should publish params");
    assert!(params.contains(&"reasoning_effort"));
}

#[test]
fn supported_params_includes_newer_openai_params() {
    // `drop_params: true` strips anything not in supported_params BEFORE the
    // transform runs. The newer OpenAI params must be listed so they survive
    // the strip and reach upstreams that accept them.
    let p = GroqProvider;
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
    let p = GroqProvider;
    let req = json!({
    "model": "groq/llama-3.1-70b",
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
    let provider = GroqProvider;
    let deployment = deployment();
    let model = "llama-3.1-70b";

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

// --- Additional request parameter tests (ported from the reference optional-param tests) ---

#[test]
fn request_preserves_response_format_json_object() {
    let p = GroqProvider;
    let req = json!({
        "model": "groq/llama-3.1-70b",
        "messages": [{"role": "user", "content": "Hi"}],
        "response_format": {"type": "json_object"},
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["response_format"]["type"], "json_object");
}

#[test]
fn request_preserves_response_format_json_schema() {
    let p = GroqProvider;
    let req = json!({
        "model": "groq/llama-3.1-70b",
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
    let p = GroqProvider;
    let req = json!({
        "model": "groq/llama-3.1-70b",
        "messages": [{"role": "user", "content": "Hi"}],
        "seed": 42,
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["seed"], 42);
}

#[test]
fn request_preserves_user() {
    let p = GroqProvider;
    let req = json!({
        "model": "groq/llama-3.1-70b",
        "messages": [{"role": "user", "content": "Hi"}],
        "user": "test_user_123",
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["user"], "test_user_123");
}

#[test]
fn request_preserves_stream_options() {
    let p = GroqProvider;
    let req = json!({
        "model": "groq/llama-3.1-70b",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": true,
        "stream_options": {"include_usage": true},
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["stream_options"]["include_usage"], true);
}

#[test]
fn request_preserves_reasoning_effort() {
    let p = GroqProvider;
    let req = json!({
        "model": "groq/llama-3.1-70b",
        "messages": [{"role": "user", "content": "Hi"}],
        "reasoning_effort": "high",
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["reasoning_effort"], "high");
}

#[test]
fn request_preserves_max_completion_tokens() {
    let p = GroqProvider;
    let req = json!({
        "model": "groq/llama-3.1-70b",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_completion_tokens": 2048,
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["max_completion_tokens"], 2048);
}

#[test]
fn strip_groq_prefix_with_various_inputs() {
    assert_eq!(
        strip_groq_prefix("groq/llama-3.3-70b-versatile"),
        "llama-3.3-70b-versatile"
    );
    assert_eq!(
        strip_groq_prefix("groq/deepseek-r1-distill-llama-70b"),
        "deepseek-r1-distill-llama-70b"
    );
    assert_eq!(strip_groq_prefix("groq/qwen/qwen3-32b"), "qwen/qwen3-32b");
    assert_eq!(
        strip_groq_prefix("llama-3.3-70b-versatile"),
        "llama-3.3-70b-versatile"
    );
}

#[test]
fn request_deployment_with_custom_model_respects_strip() {
    let p = GroqProvider;
    let mut dep = deployment();
    dep.oxllm_params.model = "groq/custom-model-v1".to_string();
    dep.model_name = "custom-model-v1".to_string();
    let req = json!({
        "model": "groq/custom-model-v1",
        "messages": [{"role": "user", "content": "Hi"}],
    });
    let out = p.transform_request(&req, &dep).expect("ok");
    assert_eq!(out["model"], "custom-model-v1");
}

#[test]
fn request_preserves_tools_and_tool_choice() {
    let p = GroqProvider;
    let req = json!({
        "model": "groq/llama-3.1-70b",
        "messages": [{"role": "user", "content": "Hi"}],
        "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}],
        "tool_choice": "auto",
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["tools"][0]["function"]["name"], "get_weather");
    assert_eq!(out["tool_choice"], "auto");
}

#[test]
fn transform_response_preserves_tool_calls() {
    let p = GroqProvider;
    let resp = json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}]
            },
            "finish_reason": "stop"
        }],
    });
    let out = p.transform_response(&resp, "llama-3.1-70b").expect("ok");
    assert_eq!(
        out["choices"][0]["message"]["tool_calls"][0]["id"],
        "call_1"
    );
    assert_eq!(
        out["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
        "get_weather"
    );
}

#[test]
fn stream_chunk_with_finish_reason() {
    let p = GroqProvider;
    let raw = "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"},\"finish_reason\":\"stop\"}]}";
    let out = p
        .transform_stream_chunk(raw)
        .expect("ok")
        .expect("should produce value");
    assert_eq!(out["choices"][0]["delta"]["content"], "Hello");
    assert_eq!(out["choices"][0]["finish_reason"], "stop");
}

#[test]
fn stream_chunk_with_usage() {
    let p = GroqProvider;
    let raw = "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":3,\"total_tokens\":8}}";
    let out = p
        .transform_stream_chunk(raw)
        .expect("ok")
        .expect("should produce value");
    assert_eq!(out["usage"]["prompt_tokens"], 5);
    assert_eq!(out["usage"]["completion_tokens"], 3);
    assert_eq!(out["usage"]["total_tokens"], 8);
}

#[test]
fn request_preserves_stop_sequence_array() {
    let p = GroqProvider;
    let req = json!({
        "model": "groq/llama-3.1-70b",
        "messages": [{"role": "user", "content": "Hi"}],
        "stop": ["stop1", "stop2"],
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["stop"][0], "stop1");
    assert_eq!(out["stop"][1], "stop2");
}

#[test]
fn request_preserves_stop_sequence_string() {
    let p = GroqProvider;
    let req = json!({
        "model": "groq/llama-3.1-70b",
        "messages": [{"role": "user", "content": "Hi"}],
        "stop": "\n",
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["stop"], "\n");
}

#[test]
fn request_preserves_sampling_params() {
    let p = GroqProvider;
    let req = json!({
        "model": "groq/llama-3.1-70b",
        "messages": [{"role": "user", "content": "Hi"}],
        "temperature": 0.7,
        "top_p": 0.9,
        "presence_penalty": 0.5,
        "frequency_penalty": 0.2,
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["temperature"], 0.7);
    assert_eq!(out["top_p"], 0.9);
    assert_eq!(out["presence_penalty"], 0.5);
    assert_eq!(out["frequency_penalty"], 0.2);
}

#[test]
fn request_preserves_logprobs() {
    let p = GroqProvider;
    let req = json!({
        "model": "groq/llama-3.1-70b",
        "messages": [{"role": "user", "content": "Hi"}],
        "logprobs": true,
        "top_logprobs": 3,
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["logprobs"], true);
    assert_eq!(out["top_logprobs"], 3);
}

#[test]
fn request_preserves_n() {
    let p = GroqProvider;
    let req = json!({
        "model": "groq/llama-3.1-70b",
        "messages": [{"role": "user", "content": "Hi"}],
        "n": 3,
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["n"], 3);
}

#[test]
fn request_preserves_parallel_tool_calls() {
    let p = GroqProvider;
    let req = json!({
        "model": "groq/llama-3.1-70b",
        "messages": [{"role": "user", "content": "Hi"}],
        "parallel_tool_calls": true,
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["parallel_tool_calls"], true);
}

#[test]
fn request_drops_null_params() {
    let p = GroqProvider;
    let req = json!({
        "model": "groq/llama-3.1-70b",
        "messages": [{"role": "user", "content": "Hi"}],
        "temperature": null,
        "max_tokens": 5,
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert!(
        out.get("temperature").is_none(),
        "null temperature should be dropped"
    );
    assert_eq!(out["max_tokens"], 5);
}

#[test]
fn transform_response_normalizes_null_tokens() {
    let p = GroqProvider;
    let resp = json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": null, "completion_tokens": null, "total_tokens": null},
    });
    let out = p.transform_response(&resp, "llama-3.1-70b").expect("ok");
    assert_eq!(out["usage"]["prompt_tokens"], 0);
    assert_eq!(out["usage"]["completion_tokens"], 0);
    assert_eq!(out["usage"]["total_tokens"], 0);
}

#[test]
fn transform_response_normalizes_null_token_details() {
    let p = GroqProvider;
    let resp = json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8, "completion_tokens_details": {"reasoning_tokens": null}},
    });
    let out = p.transform_response(&resp, "llama-3.1-70b").expect("ok");
    assert_eq!(
        out["usage"]["completion_tokens_details"]["reasoning_tokens"],
        0
    );
}

#[test]
fn supports_embeddings_returns_false() {
    let p = GroqProvider;
    assert!(!p.supports(Endpoint::Embeddings));
    assert!(p.embeddings().is_none());
}

#[test]
fn strip_groq_prefix_edge_cases() {
    assert_eq!(strip_groq_prefix(""), "");
    assert_eq!(strip_groq_prefix("groq"), "groq");
    assert_eq!(strip_groq_prefix("/"), "/");
    assert_eq!(
        strip_groq_prefix("groq/sub/provider/model"),
        "sub/provider/model"
    );
}

// --- Groq-specific transforms ---

#[test]
fn stream_chunk_renames_delta_reasoning_to_reasoning_content() {
    let p = GroqProvider;
    let raw = "data: {\"id\":\"x\",\"choices\":[{\"delta\":{\"reasoning\":\"thinking...\"}}]}";
    let out = p
        .transform_stream_chunk(raw)
        .expect("ok")
        .expect("should produce value");
    assert_eq!(
        out["choices"][0]["delta"]["reasoning_content"],
        "thinking..."
    );
    assert!(out["choices"][0]["delta"].get("reasoning").is_none());
}

#[test]
fn stream_chunk_leaves_reasoning_content_unchanged_when_absent() {
    let p = GroqProvider;
    let raw = "data: {\"id\":\"x\",\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}";
    let out = p
        .transform_stream_chunk(raw)
        .expect("ok")
        .expect("should produce value");
    assert_eq!(out["choices"][0]["delta"]["content"], "hi");
    assert!(
        out["choices"][0]["delta"]
            .get("reasoning_content")
            .is_none()
    );
}

#[test]
fn request_strips_null_fields_from_assistant_messages() {
    let p = GroqProvider;
    let req = json!({
        "model": "groq/llama-3.1-70b",
        "messages": [
            {"role": "assistant", "content": "Hi", "function_call": null},
            {"role": "user", "content": "Hello"},
        ],
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert!(out["messages"][0].get("function_call").is_none());
    assert_eq!(out["messages"][0]["content"], "Hi");
    assert_eq!(out["messages"][0]["role"], "assistant");
    // Non-assistant messages are untouched.
    assert_eq!(out["messages"][1]["content"], "Hello");
}

#[test]
fn request_preserves_non_null_assistant_fields() {
    let p = GroqProvider;
    let req = json!({
        "model": "groq/llama-3.1-70b",
        "messages": [
            {"role": "assistant", "content": "Hi", "tool_calls": [{"id": "call_1"}]},
        ],
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["messages"][0]["tool_calls"][0]["id"], "call_1");
    assert_eq!(out["messages"][0]["content"], "Hi");
}
