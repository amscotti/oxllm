use super::*;
use oxllm_core::Endpoint;
use oxllm_core::deployment::{Deployment, OxllmParams};
use oxllm_core::provider::{ChatProvider, EmbeddingProvider, Provider, ResolvedCredentials};
use serde_json::json;

fn deployment() -> Deployment {
    Deployment {
        model_name: "gpt-oss-120b".to_string(),
        oxllm_params: OxllmParams {
            model: "sambanova/gpt-oss-120b".to_string(),
            api_key: Some("sambanova-key".to_string()),
            ..Default::default()
        },
        ..Default::default()
    }
}

#[test]
fn endpoint_url_uses_default_sambanova_base() {
    let p = SambanovaProvider;
    assert_eq!(
        p.endpoint_url(&deployment()),
        "https://api.sambanova.ai/v1/chat/completions"
    );
}

#[test]
fn embedding_url_uses_default_sambanova_base() {
    let p = SambanovaProvider;
    assert_eq!(
        p.embedding_url(&deployment()),
        "https://api.sambanova.ai/v1/embeddings"
    );
}

#[test]
fn endpoint_url_respects_custom_base() {
    let p = SambanovaProvider;
    let mut dep = deployment();
    dep.oxllm_params.api_base = Some("https://custom.example.com/v1/".to_string());
    assert_eq!(
        p.endpoint_url(&dep),
        "https://custom.example.com/v1/chat/completions"
    );
}

#[test]
fn apply_auth_uses_bearer_token() {
    let p = SambanovaProvider;
    let creds = ResolvedCredentials::from_key("sambanova-secret");
    let mut headers = std::collections::HashMap::new();
    ChatProvider::apply_auth(&p, &deployment(), &creds, &mut headers);
    assert_eq!(
        headers.get("authorization").map(String::as_str),
        Some("Bearer sambanova-secret")
    );
}

#[test]
fn request_strips_sambanova_prefix_from_model() {
    let p = SambanovaProvider;
    let req = json!({
        "model": "gpt-oss-120b",
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["model"], "gpt-oss-120b");
    assert_eq!(out["messages"][0]["content"], "Hello, how are you?");
}

#[test]
fn request_preserves_string_content_unchanged() {
    // SambaNova accepts string content; the compat provider forwards it as-is.
    let p = SambanovaProvider;
    let req = json!({
        "model": "sambanova/gpt-oss-120b",
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["messages"][0]["content"], "Hello, how are you?");
}

#[test]
fn embedding_request_strips_sambanova_prefix_from_model() {
    let p = SambanovaProvider;
    let mut dep = deployment();
    dep.oxllm_params.model = "sambanova/bge-large-en-v1.5".to_string();
    let req = json!({
        "model": "bge-large-en-v1.5",
        "input": ["good morning"],
    });
    let out = EmbeddingProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(out["model"], "bge-large-en-v1.5");
    assert_eq!(out["input"][0], "good morning");
}

#[test]
fn request_preserves_stream_flag() {
    let p = SambanovaProvider;
    let req = json!({
        "model": "gpt-oss-120b",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": true,
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["stream"], true);
}

#[test]
fn response_is_identity() {
    let p = SambanovaProvider;
    let resp = json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": "gpt-oss-120b",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    });
    let out = ChatProvider::transform_response(&p, &resp, "gpt-oss-120b").expect("ok");
    assert_eq!(out, resp);
}

#[test]
fn stream_chunk_parses_openai_format() {
    let p = SambanovaProvider;
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
    let p = SambanovaProvider;
    assert!(
        p.transform_stream_chunk("data: [DONE]")
            .expect("ok")
            .is_none()
    );
}

#[test]
fn provider_id_and_supports() {
    let p = SambanovaProvider;
    assert_eq!(p.id(), "sambanova");
    assert!(p.supports(Endpoint::ChatCompletions));
    assert!(p.supports(Endpoint::Embeddings));
}

#[test]
fn strip_sambanova_prefix_handles_unknown() {
    assert_eq!(
        strip_sambanova_prefix("sambanova/gpt-oss-120b"),
        "gpt-oss-120b"
    );
    assert_eq!(strip_sambanova_prefix("gpt-oss-120b"), "gpt-oss-120b");
}

#[test]
fn supported_params_published() {
    let p = SambanovaProvider;
    let params = p.supported_params().expect("should publish params");
    assert!(params.contains(&"temperature"));
    assert!(params.contains(&"tools"));
}
