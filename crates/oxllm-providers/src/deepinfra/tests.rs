use super::*;
use oxllm_core::Endpoint;
use oxllm_core::deployment::{Deployment, OxllmParams};
use oxllm_core::provider::{ChatProvider, EmbeddingProvider, Provider, ResolvedCredentials};
use serde_json::json;

fn deployment() -> Deployment {
    Deployment {
        model_name: "meta-llama/Llama-3.3-70B-Instruct".to_string(),
        oxllm_params: OxllmParams {
            model: "deepinfra/meta-llama/Llama-3.3-70B-Instruct".to_string(),
            api_key: Some("deepinfra-key".to_string()),
            ..Default::default()
        },
        ..Default::default()
    }
}

#[test]
fn endpoint_url_uses_default_deepinfra_base() {
    let p = DeepInfraProvider;
    assert_eq!(
        p.endpoint_url(&deployment()),
        "https://api.deepinfra.com/v1/openai/chat/completions"
    );
}

#[test]
fn endpoint_url_respects_custom_base() {
    let p = DeepInfraProvider;
    let mut dep = deployment();
    dep.oxllm_params.api_base = Some("https://custom.example.com/v1/".to_string());
    assert_eq!(
        p.endpoint_url(&dep),
        "https://custom.example.com/v1/chat/completions"
    );
}

#[test]
fn apply_auth_uses_bearer_token() {
    let p = DeepInfraProvider;
    let creds = ResolvedCredentials::from_key("deepinfra-secret");
    let mut headers = std::collections::HashMap::new();
    ChatProvider::apply_auth(&p, &deployment(), &creds, &mut headers);
    assert_eq!(
        headers.get("authorization").map(String::as_str),
        Some("Bearer deepinfra-secret")
    );
}

#[test]
fn request_strips_deepinfra_prefix_from_model() {
    let p = DeepInfraProvider;
    let req = json!({
    "model": "anything-in-request",
    "messages": [{"role": "user", "content": "Hi"}],
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["model"], "meta-llama/Llama-3.3-70B-Instruct");
    assert_eq!(out["messages"][0]["content"], "Hi");
}

#[test]
fn request_preserves_stream_flag() {
    let p = DeepInfraProvider;
    let req = json!({
    "model": "llama-v3-70b",
    "messages": [{"role": "user", "content": "Hi"}],
    "stream": true,
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["stream"], true);
}

#[test]
fn request_passes_known_params_and_drops_unknown() {
    let p = DeepInfraProvider;
    let req = json!({
    "model": "meta-llama/Llama-3.3-70B-Instruct",
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

#[test]
fn response_is_identity() {
    let p = DeepInfraProvider;
    let resp = json!({
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    });
    let out = ChatProvider::transform_response(&p, &resp, "meta-llama/Llama-3.3-70B-Instruct")
        .expect("ok");
    assert_eq!(out, resp);
}

#[test]
fn stream_chunk_parses_openai_format() {
    let p = DeepInfraProvider;
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
    let p = DeepInfraProvider;
    assert!(
        p.transform_stream_chunk("data: [DONE]")
            .expect("ok")
            .is_none()
    );
}

#[test]
fn provider_id_and_supports() {
    let p = DeepInfraProvider;
    assert_eq!(p.id(), "deepinfra");
    assert!(p.supports(Endpoint::ChatCompletions));
    assert!(p.supports(Endpoint::Embeddings));
    assert!(p.embeddings().is_some());
}

#[test]
fn embedding_url_uses_default_deepinfra_base() {
    let p = DeepInfraProvider;
    assert_eq!(
        p.embedding_url(&deployment()),
        "https://api.deepinfra.com/v1/openai/embeddings"
    );
}

#[test]
fn embedding_request_strips_deepinfra_prefix_from_model() {
    let p = DeepInfraProvider;
    let req = json!({
    "model": "deepinfra-bert",
    "input": "hello world",
    });
    let out = EmbeddingProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["model"], "meta-llama/Llama-3.3-70B-Instruct");
    assert_eq!(out["input"], "hello world");
}

#[test]
fn strip_deepinfra_prefix_handles_unknown() {
    assert_eq!(
        strip_deepinfra_prefix("deepinfra/meta-llama/Llama-3.3-70B-Instruct"),
        "meta-llama/Llama-3.3-70B-Instruct"
    );
    assert_eq!(
        strip_deepinfra_prefix("meta-llama/Llama-3.3-70B-Instruct"),
        "meta-llama/Llama-3.3-70B-Instruct"
    );
}

#[test]
fn supported_params_published() {
    let p = DeepInfraProvider;
    let params = ChatProvider::supported_params(&p).expect("should publish params");
    assert!(params.contains(&"temperature"));
    assert!(params.contains(&"tools"));
}

#[test]
fn request_preserves_response_format_json_schema() {
    let p = DeepInfraProvider;
    let req = json!({
    "model": "deepinfra/meta-llama/Llama-3.3-70B-Instruct",
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
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["response_format"]["type"], "json_schema");
    assert_eq!(out["response_format"]["json_schema"]["name"], "weather");
}
