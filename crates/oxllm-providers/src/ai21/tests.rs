use super::*;
use oxllm_core::Endpoint;
use oxllm_core::deployment::{Deployment, OxllmParams};
use oxllm_core::provider::{ChatProvider, Provider, ResolvedCredentials};
use serde_json::json;

fn deployment() -> Deployment {
    Deployment {
        model_name: "jamba-large-1.6".to_string(),
        oxllm_params: OxllmParams {
            model: "ai21/jamba-large-1.6".to_string(),
            api_key: Some("ai21-key".to_string()),
            ..Default::default()
        },
        ..Default::default()
    }
}

#[test]
fn endpoint_url_uses_default_ai21_base() {
    let p = AI21Provider;
    assert_eq!(
        p.endpoint_url(&deployment()),
        "https://api.ai21.com/studio/v1/chat/completions"
    );
}

#[test]
fn endpoint_url_respects_custom_base() {
    let p = AI21Provider;
    let mut dep = deployment();
    dep.oxllm_params.api_base = Some("https://custom.example.com/v1/".to_string());
    assert_eq!(
        p.endpoint_url(&dep),
        "https://custom.example.com/v1/chat/completions"
    );
}

#[test]
fn apply_auth_uses_bearer_token() {
    let p = AI21Provider;
    let creds = ResolvedCredentials::from_key("ai21-secret");
    let mut headers = std::collections::HashMap::new();
    p.apply_auth(&deployment(), &creds, &mut headers);
    assert_eq!(
        headers.get("authorization").map(String::as_str),
        Some("Bearer ai21-secret")
    );
}

#[test]
fn request_strips_ai21_prefix_from_model() {
    let p = AI21Provider;
    let req = json!({
        "model": "jamba-large-1.6",
        "messages": [{"role": "user", "content": "Hi"}],
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["model"], "jamba-large-1.6");
    assert_eq!(out["messages"][0]["content"], "Hi");
}

#[test]
fn request_preserves_stream_flag() {
    let p = AI21Provider;
    let req = json!({
        "model": "jamba-large-1.6",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": true,
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["stream"], true);
}

#[test]
fn response_is_identity() {
    let p = AI21Provider;
    let resp = json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    });
    let out = p.transform_response(&resp, "jamba-large-1.6").expect("ok");
    assert_eq!(out, resp);
}

#[test]
fn stream_chunk_parses_openai_format() {
    let p = AI21Provider;
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
    let p = AI21Provider;
    assert!(
        p.transform_stream_chunk("data: [DONE]")
            .expect("ok")
            .is_none()
    );
}

#[test]
fn provider_id_and_supports() {
    let p = AI21Provider;
    assert_eq!(p.id(), "ai21");
    assert!(p.supports(Endpoint::ChatCompletions));
    assert!(!p.supports(Endpoint::Embeddings));
}

#[test]
fn strip_ai21_prefix_handles_unknown() {
    assert_eq!(strip_ai21_prefix("ai21/jamba-large-1.6"), "jamba-large-1.6");
    assert_eq!(strip_ai21_prefix("jamba-large-1.6"), "jamba-large-1.6");
}

#[test]
fn supported_params_published() {
    let p = AI21Provider;
    let params = p.supported_params().expect("should publish params");
    assert!(params.contains(&"temperature"));
    assert!(params.contains(&"tools"));
}

#[test]
fn request_passes_known_params_and_drops_unknown() {
    let p = AI21Provider;
    let req = json!({
        "model": "jamba-large-1.6",
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
fn wildcard_deployment_preserves_requested_model() {
    let p = AI21Provider;
    let mut dep = deployment();
    dep.oxllm_params.model = "ai21/*".to_string();
    let req = json!({
        "model": "ai21/jamba-large-1.6",
        "messages": [{"role": "user", "content": "Hi"}]
    });
    let out = p.transform_request(&req, &dep).expect("ok");
    assert_eq!(out["model"], "jamba-large-1.6");
}

#[test]
fn strip_ai21_prefix_with_various_inputs() {
    assert_eq!(strip_ai21_prefix("ai21/jamba-large-1.6"), "jamba-large-1.6");
    assert_eq!(strip_ai21_prefix("ai21/jamba-1.5-mini"), "jamba-1.5-mini");
    assert_eq!(strip_ai21_prefix("jamba-large-1.6"), "jamba-large-1.6");
}
