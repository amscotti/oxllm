use super::*;
use oxllm_core::Endpoint;
use oxllm_core::deployment::{Deployment, OxllmParams};
use oxllm_core::provider::{ChatProvider, EmbeddingProvider, Provider, ResolvedCredentials};
use serde_json::json;

fn deployment() -> Deployment {
    Deployment {
        model_name: "databricks/dbrx-instruct".to_string(),
        oxllm_params: OxllmParams {
            model: "nvidia_nim/databricks/dbrx-instruct".to_string(),
            api_key: Some("nvidia-key".to_string()),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn embedding_deployment() -> Deployment {
    Deployment {
        model_name: "nvidia/nv-embedqa-e5-v5".to_string(),
        oxllm_params: OxllmParams {
            model: "nvidia_nim/nvidia/nv-embedqa-e5-v5".to_string(),
            api_key: Some("nvidia-key".to_string()),
            ..Default::default()
        },
        ..Default::default()
    }
}

#[test]
fn endpoint_url_uses_default_nvidia_base() {
    let p = NvidiaNimProvider;
    assert_eq!(
        p.endpoint_url(&deployment()),
        "https://integrate.api.nvidia.com/v1/chat/completions"
    );
}

#[test]
fn embedding_url_uses_default_nvidia_base() {
    let p = NvidiaNimProvider;
    assert_eq!(
        p.embedding_url(&embedding_deployment()),
        "https://integrate.api.nvidia.com/v1/embeddings"
    );
}

#[test]
fn endpoint_url_respects_custom_base() {
    let p = NvidiaNimProvider;
    let mut dep = deployment();
    dep.oxllm_params.api_base = Some("https://custom.example.com/v1/".to_string());
    assert_eq!(
        p.endpoint_url(&dep),
        "https://custom.example.com/v1/chat/completions"
    );
}

#[test]
fn apply_auth_uses_bearer_token() {
    let p = NvidiaNimProvider;
    let creds = ResolvedCredentials::from_key("nvidia-secret");
    let mut headers = std::collections::HashMap::new();
    ChatProvider::apply_auth(&p, &deployment(), &creds, &mut headers);
    assert_eq!(
        headers.get("authorization").map(String::as_str),
        Some("Bearer nvidia-secret")
    );
}

#[test]
fn request_strips_nvidia_prefix_from_model() {
    let p = NvidiaNimProvider;
    let req = json!({
    "model": "anything-in-request",
    "messages": [{"role": "user", "content": "What's the weather like in Boston today in Fahrenheit?"}],
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["model"], "databricks/dbrx-instruct");
    assert_eq!(
        out["messages"][0]["content"],
        "What's the weather like in Boston today in Fahrenheit?"
    );
}

#[test]
fn request_preserves_stream_flag() {
    let p = NvidiaNimProvider;
    let req = json!({
    "model": "dbrx-instruct",
    "messages": [{"role": "user", "content": "Hi"}],
    "stream": true,
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["stream"], true);
}

#[test]
fn response_is_identity() {
    let p = NvidiaNimProvider;
    let resp = json!({
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    });
    let out = ChatProvider::transform_response(&p, &resp, "databricks/dbrx-instruct").expect("ok");
    assert_eq!(out, resp);
}

#[test]
fn stream_chunk_parses_openai_format() {
    let p = NvidiaNimProvider;
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
    let p = NvidiaNimProvider;
    assert!(
        p.transform_stream_chunk("data: [DONE]")
            .expect("ok")
            .is_none()
    );
}

#[test]
fn provider_id_and_supports() {
    let p = NvidiaNimProvider;
    assert_eq!(p.id(), "nvidia_nim");
    assert!(p.supports(Endpoint::ChatCompletions));
    assert!(p.supports(Endpoint::Embeddings));
    assert!(p.embeddings().is_some());
}

#[test]
fn strip_nvidia_prefix_handles_unknown() {
    assert_eq!(
        strip_nvidia_nim_prefix("nvidia_nim/databricks/dbrx-instruct"),
        "databricks/dbrx-instruct"
    );
    assert_eq!(
        strip_nvidia_nim_prefix("databricks/dbrx-instruct"),
        "databricks/dbrx-instruct"
    );
}

#[test]
fn supported_params_published() {
    let p = NvidiaNimProvider;
    let params = ChatProvider::supported_params(&p).expect("should publish params");
    assert!(params.contains(&"temperature"));
    assert!(params.contains(&"tools"));
}

// --- Tests ported from LiteLLM test_nvidia_nim.py ---

#[test]
fn request_preserves_presence_and_frequency_penalty() {
    // Ported from test_completion_nvidia_nim: presence_penalty and
    // frequency_penalty are forwarded to the upstream body.
    let p = NvidiaNimProvider;
    let req = json!({
    "model": "nvidia_nim/databricks/dbrx-instruct",
    "messages": [{"role": "user", "content": "What's the weather like in Boston today in Fahrenheit?"}],
    "presence_penalty": 0.5,
    "frequency_penalty": 0.1,
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["model"], "databricks/dbrx-instruct");
    assert_eq!(out["frequency_penalty"], 0.1);
    assert_eq!(out["presence_penalty"], 0.5);
}

#[test]
fn embedding_request_strips_nvidia_prefix_and_preserves_dimensions() {
    // Ported from test_embedding_nvidia_nim: model prefix stripped, input and
    // dimensions forwarded.
    let p = NvidiaNimProvider;
    let req = json!({
    "model": "nvidia/nv-embedqa-e5-v5",
    "input": "What is the meaning of life?",
    "dimensions": 1024,
    });
    let out = EmbeddingProvider::transform_request(&p, &req, &embedding_deployment()).expect("ok");
    assert_eq!(out["model"], "nvidia/nv-embedqa-e5-v5");
    assert_eq!(out["input"], "What is the meaning of life?");
    assert_eq!(out["dimensions"], 1024);
}

#[test]
fn chat_completion_preserves_tools_and_params() {
    // Ported from test_chat_completion_nvidia_nim_with_tools: tools,
    // tool_choice, parallel_tool_calls, and temperature forwarded.
    let p = NvidiaNimProvider;
    let tools = json!([
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                    },
                    "required": ["location"],
                },
            },
        }
    ]);
    let req = json!({
    "model": "nvidia_nim/meta/llama3-70b-instruct",
    "messages": [{"role": "user", "content": "What's the weather like in Boston today?"}],
    "tools": tools,
    "tool_choice": "auto",
    "parallel_tool_calls": true,
    "temperature": 0.7,
    });
    let mut dep = deployment();
    dep.oxllm_params.model = "nvidia_nim/meta/llama3-70b-instruct".to_string();
    dep.model_name = "meta/llama3-70b-instruct".to_string();
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(out["model"], "meta/llama3-70b-instruct");
    assert_eq!(out["temperature"], 0.7);
    assert_eq!(out["tools"], tools);
    assert_eq!(out["tool_choice"], "auto");
    assert_eq!(out["parallel_tool_calls"], true);
}
