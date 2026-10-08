use super::*;
use oxllm_core::Endpoint;
use oxllm_core::deployment::{Deployment, OxllmParams};
use oxllm_core::provider::{ChatProvider, EmbeddingProvider, Provider, ResolvedCredentials};
use serde_json::{Value, json};

fn deployment() -> Deployment {
    Deployment {
        model_name: "accounts/fireworks/models/llama-v3p1-405b-instruct".to_string(),
        oxllm_params: OxllmParams {
            model: "fireworks_ai/accounts/fireworks/models/llama-v3p1-405b-instruct".to_string(),
            api_key: Some("fireworks-key".to_string()),
            ..Default::default()
        },
        ..Default::default()
    }
}

#[test]
fn endpoint_url_uses_default_fireworks_base() {
    let p = FireworksAiProvider;
    assert_eq!(
        p.endpoint_url(&deployment()),
        "https://api.fireworks.ai/inference/v1/chat/completions"
    );
}

#[test]
fn embedding_url_uses_default_fireworks_base() {
    let p = FireworksAiProvider;
    assert_eq!(
        p.embedding_url(&deployment()),
        "https://api.fireworks.ai/inference/v1/embeddings"
    );
}

#[test]
fn endpoint_url_respects_custom_base() {
    let p = FireworksAiProvider;
    let mut dep = deployment();
    dep.oxllm_params.api_base = Some("https://custom.example.com/v1/".to_string());
    assert_eq!(
        p.endpoint_url(&dep),
        "https://custom.example.com/v1/chat/completions"
    );
}

#[test]
fn apply_auth_uses_bearer_token() {
    let p = FireworksAiProvider;
    let creds = ResolvedCredentials::from_key("fireworks-secret");
    let mut headers = std::collections::HashMap::new();
    ChatProvider::apply_auth(&p, &deployment(), &creds, &mut headers);
    assert_eq!(
        headers.get("authorization").map(String::as_str),
        Some("Bearer fireworks-secret")
    );
}

#[test]
fn request_strips_fireworks_prefix_from_model() {
    let p = FireworksAiProvider;
    let req = json!({
    "model": "anything-in-request",
    "messages": [{"role": "user", "content": "Hi"}],
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(
        out["model"],
        "accounts/fireworks/models/llama-v3p1-405b-instruct"
    );
    assert_eq!(out["messages"][0]["content"], "Hi");
}

#[test]
fn request_preserves_stream_flag() {
    let p = FireworksAiProvider;
    let req = json!({
    "model": "llama-v3p1-405b-instruct",
    "messages": [{"role": "user", "content": "Hi"}],
    "stream": true,
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["stream"], true);
}

#[test]
fn embedding_request_strips_fireworks_prefix_from_model() {
    let p = FireworksAiProvider;
    let req = json!({
    "model": "fireworks-bert",
    "input": "hello world",
    });
    let out = EmbeddingProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(
        out["model"],
        "accounts/fireworks/models/llama-v3p1-405b-instruct"
    );
    assert_eq!(out["input"], "hello world");
}

#[test]
fn embedding_request_prefixes_short_model_name() {
    let p = FireworksAiProvider;
    let dep = Deployment {
        model_name: "qwen3-embedding-8b".to_string(),
        oxllm_params: OxllmParams {
            model: "fireworks_ai/qwen3-embedding-8b".to_string(),
            api_key: Some("fireworks-key".to_string()),
            ..Default::default()
        },
        ..Default::default()
    };
    let req = json!({
        "model": "qwen3-embedding-8b",
        "input": "hello",
    });
    let out = EmbeddingProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(out["model"], "accounts/fireworks/models/qwen3-embedding-8b");
}

#[test]
fn response_is_identity() {
    let p = FireworksAiProvider;
    let resp = json!({
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    });
    let out = ChatProvider::transform_response(
        &p,
        &resp,
        "accounts/fireworks/models/llama-v3p1-405b-instruct",
    )
    .expect("ok");
    assert_eq!(out, resp);
}

#[test]
fn stream_chunk_parses_openai_format() {
    let p = FireworksAiProvider;
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
    let p = FireworksAiProvider;
    assert!(
        p.transform_stream_chunk("data: [DONE]")
            .expect("ok")
            .is_none()
    );
}

#[test]
fn provider_id_and_supports() {
    let p = FireworksAiProvider;
    assert_eq!(p.id(), "fireworks_ai");
    assert!(p.supports(Endpoint::ChatCompletions));
    assert!(p.supports(Endpoint::Embeddings));
    assert!(p.embeddings().is_some());
}

#[test]
fn strip_fireworks_prefix_handles_unknown() {
    assert_eq!(
        strip_fireworks_ai_prefix("fireworks_ai/accounts/fireworks/models/minimax-m3"),
        "accounts/fireworks/models/minimax-m3"
    );
    assert_eq!(
        strip_fireworks_ai_prefix("accounts/fireworks/models/minimax-m3"),
        "accounts/fireworks/models/minimax-m3"
    );
}

#[test]
fn supported_params_published() {
    let p = FireworksAiProvider;
    let params = ChatProvider::supported_params(&p).expect("should publish params");
    assert!(params.contains(&"temperature"));
    assert!(params.contains(&"tools"));
}

// --- Tests ported from the reference test_fireworks_ai_translation.py ---

#[test]
fn request_passes_tool_choice_through() {
    // The reference maps tool_choice "required" -> "any" for Fireworks, but the
    // oxllm compat provider is near-identity: it forwards tool_choice
    // unchanged (including "auto" and "required").
    let p = FireworksAiProvider;
    let req = json!({
    "model": "fireworks_ai/accounts/fireworks/models/llama-v3p1-405b-instruct",
    "messages": [{"role": "user", "content": "Hi"}],
    "tool_choice": "auto",
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["tool_choice"], "auto");
}

#[test]
fn request_preserves_response_format_json_schema() {
    // Ported from test_map_response_format: json_schema response_format is
    // passed through to Fireworks unchanged (Fireworks accepts the OpenAI
    // strict json_schema shape natively).
    let p = FireworksAiProvider;
    let response_format = json!({
        "type": "json_schema",
        "json_schema": {
            "schema": {
                "properties": {"result": {"type": "boolean"}},
                "required": ["result"],
                "type": "object",
            },
            "name": "BooleanResponse",
            "strict": true,
        },
    });
    let req = json!({
    "model": "fireworks_ai/accounts/fireworks/models/llama-v3p1-405b-instruct",
    "messages": [{"role": "user", "content": "Hi"}],
    "response_format": response_format,
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["response_format"], response_format);
    assert_eq!(out["response_format"]["json_schema"]["strict"], true);
    assert_eq!(
        out["response_format"]["json_schema"]["name"],
        "BooleanResponse"
    );
}

#[test]
fn request_passes_known_params_and_drops_unknown() {
    let p = FireworksAiProvider;
    let req = json!({
    "model": "accounts/fireworks/models/llama-v3p1-405b-instruct",
    "messages": [{"role": "user", "content": "Hi"}],
    "temperature": 0.5,
    "max_tokens": 100,
    "stream": false,
    "n": 2,
    "some_unknown_param": "ignored",
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["temperature"], 0.5);
    assert_eq!(out["max_tokens"], 100);
    assert_eq!(out["n"], 2);
    assert!(out.get("some_unknown_param").is_none());
}

fn deployment_with_model(model: &str) -> Deployment {
    let mut dep = deployment();
    dep.oxllm_params.model = model.to_string();
    dep
}

// --- Wildcards ---

#[test]
fn wildcard_deployment_preserves_requested_model() {
    let p = FireworksAiProvider;
    let dep = deployment_with_model("fireworks_ai/*");
    let req = json!({
        "model": "fireworks_ai/accounts/fireworks/models/llama-4",
        "messages": [{"role": "user", "content": "Hi"}],
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(out["model"], "accounts/fireworks/models/llama-4");
}

#[test]
fn provider_wildcard_with_unmatched_prefix_preserves_requested_model() {
    let p = FireworksAiProvider;
    let dep = deployment_with_model("openai/*");
    let req = json!({
        "model": "openai/gpt-4o",
        "messages": [{"role": "user", "content": "Hi"}],
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(out["model"], "openai/gpt-4o");
}

#[test]
fn provider_wildcard_preserves_full_path_when_unmatched() {
    let p = FireworksAiProvider;
    let dep = deployment_with_model("openai/*");
    let req = json!({
        "model": "openai/gpt-4o",
        "messages": [{"role": "user", "content": "Hi"}],
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(out["model"], "openai/gpt-4o");
}

#[test]
fn request_deployment_with_custom_model_respects_strip() {
    let p = FireworksAiProvider;
    let dep = deployment_with_model("*");
    let req = json!({
        "model": "fireworks_ai/custom-model",
        "messages": [{"role": "user", "content": "Hi"}],
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    // Strip resolves `fireworks_ai/custom-model` → `custom-model`; the
    // Fireworks prefix transform then promotes the bare name to the
    // `accounts/fireworks/models/` form the upstream API requires.
    assert_eq!(out["model"], "accounts/fireworks/models/custom-model");
}

// --- Supported params ---

#[test]
fn supported_params_includes_reasoning_effort() {
    let p = FireworksAiProvider;
    let params = ChatProvider::supported_params(&p).expect("should publish params");
    assert!(params.contains(&"reasoning_effort"));
}

#[test]
fn supported_params_includes_newer_openai_params() {
    let p = FireworksAiProvider;
    let params = ChatProvider::supported_params(&p).expect("should publish params");
    for key in &[
        "logit_bias",
        "store",
        "metadata",
        "modalities",
        "audio",
        "service_tier",
    ] {
        assert!(params.contains(key), "`{key}` must be in supported_params");
    }
}

// --- Request params ---

#[test]
fn request_preserves_response_format_json_object() {
    let p = FireworksAiProvider;
    let req = json!({
        "model": "llama-4",
        "messages": [{"role": "user", "content": "Hi"}],
        "response_format": {"type": "json_object"},
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["response_format"]["type"], "json_object");
}

#[test]
fn request_preserves_seed() {
    let p = FireworksAiProvider;
    let req = json!({
        "model": "llama-4",
        "messages": [{"role": "user", "content": "Hi"}],
        "seed": 42,
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["seed"], 42);
}

#[test]
fn request_preserves_user() {
    let p = FireworksAiProvider;
    let req = json!({
        "model": "llama-4",
        "messages": [{"role": "user", "content": "Hi"}],
        "user": "test_user",
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["user"], "test_user");
}

#[test]
fn request_preserves_stream_options() {
    let p = FireworksAiProvider;
    let req = json!({
        "model": "llama-4",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": true,
        "stream_options": {"include_usage": true},
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["stream"], true);
    assert_eq!(out["stream_options"]["include_usage"], true);
}

#[test]
fn request_preserves_reasoning_effort() {
    let p = FireworksAiProvider;
    let req = json!({
        "model": "llama-4",
        "messages": [{"role": "user", "content": "Hi"}],
        "reasoning_effort": "high",
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["reasoning_effort"], "high");
}

#[test]
fn request_preserves_max_completion_tokens() {
    let p = FireworksAiProvider;
    let req = json!({
        "model": "llama-4",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_completion_tokens": 2048,
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["max_completion_tokens"], 2048);
}

#[test]
fn request_preserves_tools_and_tool_choice() {
    let p = FireworksAiProvider;
    let req = json!({
        "model": "llama-4",
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
fn request_preserves_stop_sequence_array() {
    let p = FireworksAiProvider;
    let req = json!({
        "model": "llama-4",
        "messages": [{"role": "user", "content": "Hi"}],
        "stop": ["stop1", "stop2"],
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["stop"][0], "stop1");
    assert_eq!(out["stop"][1], "stop2");
}

#[test]
fn request_preserves_stop_sequence_string() {
    let p = FireworksAiProvider;
    let req = json!({
        "model": "llama-4",
        "messages": [{"role": "user", "content": "Hi"}],
        "stop": "\n",
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["stop"], "\n");
}

#[test]
fn request_preserves_sampling_params() {
    let p = FireworksAiProvider;
    let req = json!({
        "model": "llama-4",
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
    let p = FireworksAiProvider;
    let req = json!({
        "model": "llama-4",
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
    let p = FireworksAiProvider;
    let req = json!({
        "model": "llama-4",
        "messages": [{"role": "user", "content": "Hi"}],
        "n": 3,
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["n"], 3);
}

#[test]
fn request_preserves_parallel_tool_calls() {
    let p = FireworksAiProvider;
    let req = json!({
        "model": "llama-4",
        "messages": [{"role": "user", "content": "Hi"}],
        "parallel_tool_calls": true,
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["parallel_tool_calls"], true);
}

#[test]
fn request_drops_null_params() {
    let p = FireworksAiProvider;
    let req = json!({
        "model": "llama-4",
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
    let p = FireworksAiProvider;
    let req = json!({
        "model": "llama-4",
        "messages": [{"role": "user", "content": "Hi"}],
        "logit_bias": {"123": 1},
        "store": true,
        "metadata": {"key": "value"},
        "modalities": ["text"],
        "audio": {"voice": "alloy", "format": "wav"},
        "service_tier": "auto",
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["logit_bias"]["123"], 1);
    assert_eq!(out["store"], true);
    assert_eq!(out["metadata"]["key"], "value");
    assert_eq!(out["modalities"][0], "text");
    assert_eq!(out["audio"]["voice"], "alloy");
    assert_eq!(out["service_tier"], "auto");
}

// --- Response transforms ---

#[test]
fn transform_response_preserves_tool_calls() {
    let p = FireworksAiProvider;
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
                    "function": {"name": "get_weather", "arguments": "{\"loc\":\"SF\"}"}
                }]
            },
            "finish_reason": "tool_calls"
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    });
    let out = ChatProvider::transform_response(
        &p,
        &resp,
        "accounts/fireworks/models/llama-v3p1-405b-instruct",
    )
    .expect("ok");
    assert_eq!(
        out["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
        "get_weather"
    );
}

#[test]
fn transform_response_normalizes_null_tokens() {
    let p = FireworksAiProvider;
    let resp = json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": null, "completion_tokens": null, "total_tokens": null},
    });
    let out = ChatProvider::transform_response(
        &p,
        &resp,
        "accounts/fireworks/models/llama-v3p1-405b-instruct",
    )
    .expect("ok");
    assert_eq!(out["usage"]["prompt_tokens"], 0);
    assert_eq!(out["usage"]["completion_tokens"], 0);
    assert_eq!(out["usage"]["total_tokens"], 0);
}

#[test]
fn transform_response_normalizes_null_token_details() {
    let p = FireworksAiProvider;
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
    let out = ChatProvider::transform_response(
        &p,
        &resp,
        "accounts/fireworks/models/llama-v3p1-405b-instruct",
    )
    .expect("ok");
    assert_eq!(
        out["usage"]["completion_tokens_details"]["reasoning_tokens"],
        0
    );
}

// --- Stream chunks ---

#[test]
fn stream_chunk_with_finish_reason() {
    let p = FireworksAiProvider;
    let raw = "data: {\"id\":\"x\",\"choices\":[{\"delta\":{\"content\":\"hello\"},\"finish_reason\":\"stop\"}]}";
    let out = p
        .transform_stream_chunk(raw)
        .expect("ok")
        .expect("should produce value");
    assert_eq!(out["id"], "x");
    assert_eq!(out["choices"][0]["finish_reason"], "stop");
    assert_eq!(out["choices"][0]["delta"]["content"], "hello");
}

#[test]
fn stream_chunk_with_usage() {
    let p = FireworksAiProvider;
    let raw = "data: {\"id\":\"x\",\"choices\":[{\"delta\":{}}],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":2,\"total_tokens\":7}}";
    let out = p
        .transform_stream_chunk(raw)
        .expect("ok")
        .expect("should produce value");
    assert_eq!(out["usage"]["prompt_tokens"], 5);
    assert_eq!(out["usage"]["completion_tokens"], 2);
    assert_eq!(out["usage"]["total_tokens"], 7);
}

// --- Prefix stripping ---

#[test]
fn strip_fireworks_ai_prefix_with_various_inputs() {
    assert_eq!(
        strip_fireworks_ai_prefix("fireworks_ai/accounts/fireworks/models/llama-v3p1-70b-instruct"),
        "accounts/fireworks/models/llama-v3p1-70b-instruct"
    );
    assert_eq!(
        strip_fireworks_ai_prefix("fireworks_ai/accounts/fireworks/models/deepseek-r1"),
        "accounts/fireworks/models/deepseek-r1"
    );
    assert_eq!(
        strip_fireworks_ai_prefix(
            "fireworks_ai/accounts/fireworks/models/llama-v3p1-405b-instruct"
        ),
        "accounts/fireworks/models/llama-v3p1-405b-instruct"
    );
}

#[test]
fn strip_fireworks_ai_prefix_edge_cases() {
    assert_eq!(strip_fireworks_ai_prefix(""), "");
    assert_eq!(strip_fireworks_ai_prefix("fireworks_ai"), "fireworks_ai");
    assert_eq!(strip_fireworks_ai_prefix("a/b/c"), "a/b/c");
    assert_eq!(strip_fireworks_ai_prefix("other/model"), "other/model");
}

// --- Embeddings ---

#[test]
fn embedding_response_is_identity() {
    let p = FireworksAiProvider;
    let resp = json!({
        "object": "list",
        "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}],
    });
    let out = EmbeddingProvider::transform_response(
        &p,
        &resp,
        "accounts/fireworks/models/llama-v3p1-405b-instruct",
    )
    .expect("ok");
    assert_eq!(out, resp);
}

#[test]
fn embedding_request_with_user_param() {
    let p = FireworksAiProvider;
    let req = json!({
        "model": "fireworks-bert",
        "input": "hello world",
        "user": "test_user",
    });
    let out = EmbeddingProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(
        out["model"],
        "accounts/fireworks/models/llama-v3p1-405b-instruct"
    );
    assert_eq!(out["input"], "hello world");
    assert_eq!(out["user"], "test_user");
}

#[test]
fn embedding_request_with_encoding_format() {
    let p = FireworksAiProvider;
    let req = json!({
        "model": "fireworks-bert",
        "input": "hello world",
        "encoding_format": "float",
    });
    let out = EmbeddingProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["encoding_format"], "float");
}

#[test]
fn embedding_request_with_dimensions() {
    let p = FireworksAiProvider;
    let req = json!({
        "model": "fireworks-bert",
        "input": "hello world",
        "dimensions": 256,
    });
    let out = EmbeddingProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["dimensions"], 256);
}

#[test]
fn embedding_request_drops_unknown_params() {
    let p = FireworksAiProvider;
    let req = json!({
        "model": "fireworks-bert",
        "input": "hello world",
        "unknown_param": "should_be_dropped",
        "temperature": 0.5,
    });
    let out = EmbeddingProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert!(out.get("unknown_param").is_none());
    assert!(out.get("temperature").is_none());
}

// --- Golden file ---

#[test]
fn test_golden_file_parity() {
    let p = FireworksAiProvider;
    let request: Value = serde_json::from_str(include_str!(
        "../openai_compat/goldens/chat_request_001.json"
    ))
    .expect("valid golden request");
    let response: Value = serde_json::from_str(include_str!(
        "../openai_compat/goldens/chat_response_001.json"
    ))
    .expect("valid golden response");

    let transformed_req =
        ChatProvider::transform_request(&p, &request, &deployment()).expect("transform request");
    assert_eq!(
        transformed_req["model"],
        "accounts/fireworks/models/llama-v3p1-405b-instruct"
    );
    assert_eq!(transformed_req["messages"], request["messages"]);
    assert_eq!(transformed_req["temperature"], request["temperature"]);
    assert_eq!(transformed_req["max_tokens"], request["max_tokens"]);
    assert_eq!(transformed_req["stream"], request["stream"]);

    let transformed_resp = ChatProvider::transform_response(
        &p,
        &response,
        "accounts/fireworks/models/llama-v3p1-405b-instruct",
    )
    .expect("transform response");
    assert_eq!(transformed_resp["id"], response["id"]);
    assert_eq!(
        transformed_resp["choices"][0]["message"]["content"],
        response["choices"][0]["message"]["content"]
    );
}

// --- Fireworks-specific transforms ---

#[test]
fn request_prefixes_bare_model_name() {
    let p = FireworksAiProvider;
    let dep = deployment_with_model("fireworks_ai/llama-v3p1-405b-instruct");
    let req = json!({
        "model": "ignored",
        "messages": [{"role": "user", "content": "Hi"}],
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(
        out["model"],
        "accounts/fireworks/models/llama-v3p1-405b-instruct"
    );
}

#[test]
fn request_prefixes_fast_model_with_routers() {
    let p = FireworksAiProvider;
    let dep = deployment_with_model("fireworks_ai/foo-fast");
    let req = json!({
        "model": "ignored",
        "messages": [{"role": "user", "content": "Hi"}],
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(out["model"], "accounts/fireworks/routers/foo-fast");
}

#[test]
fn request_leaves_pre_prefixed_model_unchanged() {
    let p = FireworksAiProvider;
    let dep = deployment_with_model("fireworks_ai/accounts/fireworks/models/bar");
    let req = json!({
        "model": "ignored",
        "messages": [{"role": "user", "content": "Hi"}],
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(out["model"], "accounts/fireworks/models/bar");
}

#[test]
fn request_leaves_cross_provider_wildcard_model_unchanged() {
    let p = FireworksAiProvider;
    let dep = deployment_with_model("openai/*");
    let req = json!({
        "model": "openai/gpt-4o",
        "messages": [{"role": "user", "content": "Hi"}],
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    // Cross-provider path contains `/` and must not be mangled by the
    // Fireworks prefix transform.
    assert_eq!(out["model"], "openai/gpt-4o");
}

#[test]
fn request_maps_tool_choice_required_to_any() {
    let p = FireworksAiProvider;
    let req = json!({
        "model": "accounts/fireworks/models/llama-v3p1-405b-instruct",
        "messages": [{"role": "user", "content": "Hi"}],
        "tool_choice": "required",
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["tool_choice"], "any");
}

#[test]
fn request_leaves_tool_choice_auto_unchanged() {
    let p = FireworksAiProvider;
    let req = json!({
        "model": "accounts/fireworks/models/llama-v3p1-405b-instruct",
        "messages": [{"role": "user", "content": "Hi"}],
        "tool_choice": "auto",
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["tool_choice"], "auto");
}

#[test]
fn request_injects_stream_options_include_usage_when_streaming() {
    let p = FireworksAiProvider;
    let req = json!({
        "model": "llama-4",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": true,
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["stream"], true);
    assert_eq!(out["stream_options"]["include_usage"], true);
}

#[test]
fn request_does_not_override_explicit_include_usage_true() {
    let p = FireworksAiProvider;
    let req = json!({
        "model": "llama-4",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": true,
        "stream_options": {"include_usage": true},
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["stream_options"]["include_usage"], true);
}

#[test]
fn request_respects_explicit_include_usage_false() {
    let p = FireworksAiProvider;
    let req = json!({
        "model": "llama-4",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": true,
        "stream_options": {"include_usage": false},
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    // Explicit false is respected (reference parity).
    assert_eq!(out["stream_options"]["include_usage"], false);
}

#[test]
fn request_does_not_inject_stream_options_when_not_streaming() {
    let p = FireworksAiProvider;
    let req = json!({
        "model": "llama-4",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": false,
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert!(out.get("stream_options").is_none());
}

#[test]
fn request_strips_strict_from_tool_function_definitions() {
    let p = FireworksAiProvider;
    let req = json!({
        "model": "accounts/fireworks/models/llama-v3p1-405b-instruct",
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
        ],
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert!(out["tools"][0]["function"].get("strict").is_none());
    assert_eq!(out["tools"][0]["function"]["name"], "get_weather");
}
