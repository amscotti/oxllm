//! Twin tests for the shared [`OpenAiCompatProvider`].
//!
//! Ported from `/Users/ascotti/code/oxllm/crates/oxllm-providers/src/openai_compatible.rs`
//! (twin tests for the OpenAI-compatible provider). The source tests exercise
//! full HTTP round-trips through wiremock; these adapted tests verify the
//! pure transform/URL/auth surface against the same LiteLLM Python contract.

use super::*;
use oxllm_core::Endpoint;
use oxllm_core::deployment::{Deployment, OxllmParams};
use oxllm_core::provider::{ChatProvider, EmbeddingProvider, Provider, ResolvedCredentials};
use serde_json::json;

/// Construct a test `OpenAiCompatProvider` that requires auth, publishes
/// supported params, and supports embeddings (like most real providers).
fn test_provider() -> OpenAiCompatProvider {
    OpenAiCompatProvider::new(
        "testcompat",
        "https://api.test.example.com/v1",
        "testcompat",
        true,
        true,
        true,
    )
}

fn deployment() -> Deployment {
    Deployment {
        model_name: "llama-3.1-8b".to_string(),
        oxllm_params: OxllmParams {
            model: "testcompat/llama-3.1-8b".to_string(),
            api_key: Some("test-key".to_string()),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn deployment_with_base(base: &str) -> Deployment {
    let mut dep = deployment();
    dep.oxllm_params.api_base = Some(base.to_string());
    dep
}

fn deployment_with_model(model: &str) -> Deployment {
    let mut dep = deployment();
    dep.oxllm_params.model = model.to_string();
    dep
}

// --- endpoint_url ---

#[test]
fn endpoint_url_uses_default_base() {
    let p = test_provider();
    assert_eq!(
        p.endpoint_url(&deployment()),
        "https://api.test.example.com/v1/chat/completions"
    );
}

#[test]
fn endpoint_url_respects_custom_base() {
    let p = test_provider();
    let dep = deployment_with_base("https://custom.example.com/v1/");
    assert_eq!(
        p.endpoint_url(&dep),
        "https://custom.example.com/v1/chat/completions"
    );
}

#[test]
fn endpoint_url_trims_trailing_slash() {
    let p = test_provider();
    let dep = deployment_with_base("https://api.test.example.com/v1/");
    assert_eq!(
        p.endpoint_url(&dep),
        "https://api.test.example.com/v1/chat/completions"
    );
}

// --- apply_auth ---

#[test]
fn apply_auth_sets_bearer_token() {
    let p = test_provider();
    let creds = ResolvedCredentials::from_key("test-secret");
    let mut headers = std::collections::HashMap::new();
    ChatProvider::apply_auth(&p, &deployment(), &creds, &mut headers);
    assert_eq!(
        headers.get("authorization").map(String::as_str),
        Some("Bearer test-secret")
    );
}

#[test]
fn apply_auth_without_usable_bearer_sets_no_header() {
    // A credential kind that cannot yield a bearer token (AWS SigV4 material
    // meant for Bedrock) must not produce an Authorization header even when
    // the provider requires auth — `apply_bearer_auth` degrades to no header
    // instead of inserting a bogus value.
    let p = test_provider();
    let creds = ResolvedCredentials::AwsSigV4 {
        access_key: "AKIATEST".to_string(),
        secret_key: "secret".to_string(),
        session_token: None,
        region: "us-east-1".to_string(),
    };
    let mut headers = std::collections::HashMap::new();
    ChatProvider::apply_auth(&p, &deployment(), &creds, &mut headers);
    assert!(headers.is_empty());
}

// --- transform_request ---

#[test]
fn request_strips_provider_prefix_from_model() {
    let p = test_provider();
    // The request model deliberately differs from the deployment's stripped
    // model: the output can only be `llama-3.1-8b` if the provider took the
    // deployment's `testcompat/llama-3.1-8b` AND stripped the prefix. A
    // request-model passthrough would yield `unrelated-request-model`; a
    // missing prefix-strip would yield `testcompat/llama-3.1-8b`.
    let req = json!({
        "model": "unrelated-request-model",
        "messages": [{"role": "user", "content": "Hi"}],
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["model"], "llama-3.1-8b");
    assert_eq!(out["messages"][0]["content"], "Hi");
}

#[test]
fn request_overrides_model_with_deployment_model() {
    let p = test_provider();
    let req = json!({
        "model": "caller-supplied-ignored",
        "messages": [{"role": "user", "content": "Hi"}],
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["model"], "llama-3.1-8b");
}

#[test]
fn request_preserves_stream_flag() {
    let p = test_provider();
    let req = json!({
        "model": "llama-3.1-8b",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": true,
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["stream"], true);
}

#[test]
fn request_passes_known_params_and_drops_unknown() {
    let p = test_provider();
    let req = json!({
        "model": "llama-3.1-8b",
        "messages": [{"role": "user", "content": "Hi"}],
        "temperature": 0.5,
        "max_tokens": 100,
        "n": 2,
        "stream": false,
        "some_unknown_param": "ignored",
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["temperature"], 0.5);
    assert_eq!(out["max_tokens"], 100);
    assert_eq!(out["n"], 2);
    assert!(out.get("some_unknown_param").is_none());
}

#[test]
fn request_drops_null_params() {
    let p = test_provider();
    let req = json!({
        "model": "llama-3.1-8b",
        "messages": [{"role": "user", "content": "Hi"}],
        "temperature": null,
        "max_tokens": 5,
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert!(out.get("temperature").is_none());
    assert_eq!(out["max_tokens"], 5);
}

#[test]
fn request_wildcard_uses_request_model() {
    let p = test_provider();
    let dep = deployment_with_model("testcompat/*");
    let req = json!({
        "model": "testcompat/llama-4-scout",
        "messages": [{"role": "user", "content": "Hi"}],
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(out["model"], "llama-4-scout");
}

#[test]
fn request_bare_wildcard_uses_request_model() {
    let p = test_provider();
    let dep = deployment_with_model("*");
    let req = json!({
        "model": "testcompat/llama-4-scout",
        "messages": [{"role": "user", "content": "Hi"}],
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(out["model"], "llama-4-scout");
}

#[test]
fn request_wildcard_preserves_other_provider_prefix() {
    let p = test_provider();
    let dep = deployment_with_model("openai/*");
    let req = json!({
        "model": "openai/gpt-4o-mini",
        "messages": [{"role": "user", "content": "Hi"}],
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    // strip_model only strips `testcompat/`, so `openai/gpt-4o-mini` survives.
    assert_eq!(out["model"], "openai/gpt-4o-mini");
}

#[test]
fn request_drops_null_messages() {
    // A `"messages": null` from the client must not be forwarded upstream;
    // the null guard mirrors the per-param `is_null` filter in the main loop.
    let p = test_provider();
    let req = json!({
        "model": "llama-3.1-8b",
        "messages": null,
        "temperature": 0.5,
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert!(out.get("messages").is_none());
    assert_eq!(out["temperature"], 0.5);
}

// --- transform_response (identity passthrough) ---

#[test]
fn response_is_identity() {
    let p = test_provider();
    let resp = json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    });
    let out = ChatProvider::transform_response(&p, &resp, "llama-3.1-8b").expect("ok");
    assert_eq!(out, resp);
}

#[test]
fn response_maps_generic_error_finish_reason_to_stop() {
    let p = test_provider();
    let resp = json!({
        "id": "chatcmpl-1",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": ""}, "finish_reason": "error"}],
    });
    let out = ChatProvider::transform_response(&p, &resp, "llama-3.1-8b").expect("ok");
    assert_eq!(out["choices"][0]["finish_reason"], "stop");
}

#[test]
fn stream_chunk_maps_generic_error_finish_reason_to_stop() {
    let p = test_provider();
    let raw = r#"data: {"id":"x","choices":[{"delta":{},"finish_reason":"error"}]}"#;
    let out = p
        .transform_stream_chunk(raw)
        .expect("ok")
        .expect("should produce value");
    assert_eq!(out["choices"][0]["finish_reason"], "stop");
}

// --- transform_stream_chunk ---

#[test]
fn stream_chunk_parses_openai_format() {
    let p = test_provider();
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
    let p = test_provider();
    assert!(
        p.transform_stream_chunk("data: [DONE]")
            .expect("ok")
            .is_none()
    );
}

#[test]
fn stream_empty_chunk_returns_none() {
    let p = test_provider();
    assert!(p.transform_stream_chunk("").expect("ok").is_none());
}

// --- Embedding provider ---

#[test]
fn embedding_url_uses_default_base() {
    let p = test_provider();
    assert_eq!(
        EmbeddingProvider::embedding_url(&p, &deployment()),
        "https://api.test.example.com/v1/embeddings"
    );
}

#[test]
fn embedding_url_does_not_double_append_embeddings() {
    // Mirrors the chat `endpoint_url` guard: an operator who configures
    // `api_base` as the full embeddings endpoint must not get `/embeddings`
    // appended twice.
    let p = test_provider();
    let dep = deployment_with_base("https://custom.example.com/v1/embeddings");
    assert_eq!(
        EmbeddingProvider::embedding_url(&p, &dep),
        "https://custom.example.com/v1/embeddings"
    );
}

#[test]
fn embedding_request_strips_provider_prefix() {
    let p = test_provider();
    let req = json!({
        "model": "embed-v3",
        "input": "hello world",
    });
    let out = EmbeddingProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["model"], "llama-3.1-8b");
    assert_eq!(out["input"], "hello world");
}

#[test]
fn embedding_request_wildcard_uses_request_model() {
    // A `testcompat/*` wildcard embedding deployment must defer to the model
    // in the request body (prefix-stripped), not forward the literal `*`.
    let p = test_provider();
    let dep = deployment_with_model("testcompat/*");
    let req = json!({
        "model": "testcompat/bge-large",
        "input": "hello world",
    });
    let out = EmbeddingProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(out["model"], "bge-large");
    assert_eq!(out["input"], "hello world");
}

#[test]
fn embedding_request_bare_wildcard_uses_request_model() {
    let p = test_provider();
    let dep = deployment_with_model("*");
    let req = json!({
        "model": "testcompat/bge-large",
        "input": "hello world",
    });
    let out = EmbeddingProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(out["model"], "bge-large");
    assert_eq!(out["input"], "hello world");
}

#[test]
fn embedding_request_drops_null_input() {
    // A `"input": null` from the client must not be forwarded upstream.
    let p = test_provider();
    let req = json!({
        "model": "embed-v3",
        "input": null,
        "user": "test_user",
    });
    let out = EmbeddingProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert!(out.get("input").is_none());
    assert_eq!(out["user"], "test_user");
}

#[test]
fn embedding_response_is_identity() {
    let p = test_provider();
    let resp = json!({
        "object": "list",
        "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
    });
    let out = EmbeddingProvider::transform_response(&p, &resp, "llama-3.1-8b").expect("ok");
    assert_eq!(out, resp);
}

// --- Provider trait ---

#[test]
fn provider_id_and_supports() {
    let p = OpenAiCompatProvider::new(
        "test_slug",
        "https://api.example.com/v1",
        "test_slug",
        true,
        true,
        true,
    );
    assert_eq!(p.id(), "test_slug");
    assert!(p.supports(Endpoint::ChatCompletions));
    assert!(p.supports(Endpoint::Embeddings));
    assert!(!p.supports(Endpoint::Rerank));
    assert!(p.chat().is_some());
    assert!(p.embeddings().is_some());
}

#[test]
fn provider_requires_auth_config() {
    let with_auth = OpenAiCompatProvider::new("a", "https://a.com/v1", "a", false, false, true);
    let no_auth = OpenAiCompatProvider::new("b", "https://b.com/v1", "b", false, false, false);

    let creds = ResolvedCredentials::from_key("secret");
    let mut h1 = std::collections::HashMap::new();
    let mut h2 = std::collections::HashMap::new();
    ChatProvider::apply_auth(&with_auth, &deployment(), &creds, &mut h1);
    ChatProvider::apply_auth(&no_auth, &deployment(), &creds, &mut h2);
    assert!(h1.contains_key("authorization"));
    assert!(h2.is_empty());
}

#[test]
fn supported_params_published() {
    let p = OpenAiCompatProvider::new("p", "https://p.com/v1", "p", false, true, true);
    let params = ChatProvider::supported_params(&p).expect("should publish");
    assert!(params.contains(&"temperature"));
    assert!(params.contains(&"tools"));
    assert!(params.contains(&"max_tokens"));
}

#[test]
fn supported_params_none_when_disabled() {
    let p = OpenAiCompatProvider::new("p", "https://p.com/v1", "p", false, false, true);
    assert!(ChatProvider::supported_params(&p).is_none());
}

#[test]
fn strip_one_segment_handles_various_inputs() {
    assert_eq!(
        OpenAiCompatProvider::strip_one_segment("testcompat/llama-3.1-8b", "testcompat"),
        "llama-3.1-8b"
    );
    assert_eq!(
        OpenAiCompatProvider::strip_one_segment("llama-3.1-8b", "testcompat"),
        "llama-3.1-8b"
    );
    assert_eq!(
        OpenAiCompatProvider::strip_one_segment("other/llama", "testcompat"),
        "other/llama"
    );
}

// --- SUPPORTED_CHAT_PARAMS integrity ---

#[test]
fn supported_chat_params_contains_newer_params() {
    for key in [
        "logit_bias",
        "store",
        "metadata",
        "modalities",
        "audio",
        "service_tier",
        "max_completion_tokens",
        "stream_options",
        "parallel_tool_calls",
        "reasoning_effort",
    ] {
        assert!(
            SUPPORTED_CHAT_PARAMS.contains(&key),
            "`{key}` must be in SUPPORTED_CHAT_PARAMS so drop_params keeps it"
        );
    }
}

// --- Additional parameter passthrough tests (ported from LiteLLM test_optional_params.py) ---

#[test]
fn request_preserves_stream_options() {
    let p = test_provider();
    let req = json!({
        "model": "llama-3.1-8b",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": true,
        "stream_options": {"include_usage": true},
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["stream"], true);
    assert_eq!(out["stream_options"]["include_usage"], true);
}

#[test]
fn request_preserves_response_format_json_schema() {
    let p = test_provider();
    let req = json!({
        "model": "llama-3.1-8b",
        "messages": [{"role": "user", "content": "Hi"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "test",
                "schema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
        },
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["response_format"]["type"], "json_schema");
    assert_eq!(out["response_format"]["json_schema"]["name"], "test");
}

#[test]
fn request_preserves_response_format_json_object() {
    let p = test_provider();
    let req = json!({
        "model": "llama-3.1-8b",
        "messages": [{"role": "user", "content": "Hi"}],
        "response_format": {"type": "json_object"},
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["response_format"]["type"], "json_object");
}

#[test]
fn request_preserves_stop_sequences() {
    let p = test_provider();
    let req = json!({
        "model": "llama-3.1-8b",
        "messages": [{"role": "user", "content": "Hi"}],
        "stop": ["stop1", "stop2", "stop3"],
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["stop"][0], "stop1");
    assert_eq!(out["stop"][1], "stop2");
    assert_eq!(out["stop"][2], "stop3");
}

#[test]
fn request_preserves_stop_as_string() {
    let p = test_provider();
    let req = json!({
        "model": "llama-3.1-8b",
        "messages": [{"role": "user", "content": "Hi"}],
        "stop": "\n",
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["stop"], "\n");
}

#[test]
fn request_preserves_tool_choice() {
    let p = test_provider();
    let req = json!({
        "model": "llama-3.1-8b",
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
fn request_preserves_tool_choice_required() {
    let p = test_provider();
    let req = json!({
        "model": "llama-3.1-8b",
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
        "tool_choice": "required",
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["tool_choice"], "required");
}

#[test]
fn request_preserves_max_completion_tokens() {
    let p = test_provider();
    let req = json!({
        "model": "llama-3.1-8b",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_completion_tokens": 1024,
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["max_completion_tokens"], 1024);
}

#[test]
fn request_preserves_seed() {
    let p = test_provider();
    let req = json!({
        "model": "llama-3.1-8b",
        "messages": [{"role": "user", "content": "Hi"}],
        "seed": 42,
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["seed"], 42);
}

#[test]
fn request_preserves_user() {
    let p = test_provider();
    let req = json!({
        "model": "llama-3.1-8b",
        "messages": [{"role": "user", "content": "Hi"}],
        "user": "test_user_123",
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["user"], "test_user_123");
}

#[test]
fn request_preserves_logprobs_and_top_logprobs() {
    let p = test_provider();
    let req = json!({
        "model": "llama-3.1-8b",
        "messages": [{"role": "user", "content": "Hi"}],
        "logprobs": true,
        "top_logprobs": 3,
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["logprobs"], true);
    assert_eq!(out["top_logprobs"], 3);
}

#[test]
fn request_preserves_presence_and_frequency_penalty() {
    let p = test_provider();
    let req = json!({
        "model": "llama-3.1-8b",
        "messages": [{"role": "user", "content": "Hi"}],
        "presence_penalty": 0.5,
        "frequency_penalty": 0.2,
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["presence_penalty"], 0.5);
    assert_eq!(out["frequency_penalty"], 0.2);
}

#[test]
fn request_preserves_parallel_tool_calls() {
    let p = test_provider();
    let req = json!({
        "model": "llama-3.1-8b",
        "messages": [{"role": "user", "content": "Hi"}],
        "parallel_tool_calls": true,
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["parallel_tool_calls"], true);
}

#[test]
fn request_preserves_reasoning_effort() {
    let p = test_provider();
    let req = json!({
        "model": "llama-3.1-8b",
        "messages": [{"role": "user", "content": "Hi"}],
        "reasoning_effort": "medium",
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["reasoning_effort"], "medium");
}

// --- Embedding parameter tests ---

#[test]
fn embedding_request_with_user_param() {
    let p = test_provider();
    let req = json!({
        "model": "text-embedding-3-small",
        "input": "hello world",
        "user": "test_user",
    });
    let out = EmbeddingProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["model"], "llama-3.1-8b");
    assert_eq!(out["input"], "hello world");
    assert_eq!(out["user"], "test_user");
}

#[test]
fn embedding_request_with_encoding_format() {
    let p = test_provider();
    let req = json!({
        "model": "text-embedding-3-small",
        "input": "hello world",
        "encoding_format": "float",
    });
    let out = EmbeddingProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["encoding_format"], "float");
}

#[test]
fn embedding_request_with_dimensions() {
    let p = test_provider();
    let req = json!({
        "model": "text-embedding-3-small",
        "input": "hello world",
        "dimensions": 256,
    });
    let out = EmbeddingProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["dimensions"], 256);
}

#[test]
fn embedding_request_drops_unknown_params() {
    let p = test_provider();
    let req = json!({
        "model": "text-embedding-3-small",
        "input": "hello world",
        "unknown_param": "should_be_dropped",
        "temperature": 0.5,
    });
    let out = EmbeddingProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert!(out.get("unknown_param").is_none());
    assert!(out.get("temperature").is_none());
}

// --- Provider trait configuration ---

#[test]
fn provider_without_embeddings_returns_none() {
    let p = OpenAiCompatProvider::new(
        "chat_only",
        "https://chat.com/v1",
        "chat_only",
        false,
        true,
        true,
    );
    assert!(p.embeddings().is_none());
    assert!(p.chat().is_some());
}

#[test]
fn provider_default_base_no_trailing_slash_works() {
    let p = OpenAiCompatProvider::new("test", "https://api.test.com", "test", false, true, true);
    let mut dep = deployment();
    dep.oxllm_params.model = "test/model".to_string();
    // base has no /v1 suffix; chat_completions appends /chat/completions directly
    assert_eq!(
        p.endpoint_url(&dep),
        "https://api.test.com/chat/completions"
    );
}
