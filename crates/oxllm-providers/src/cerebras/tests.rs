use super::*;
use oxllm_core::Endpoint;
use oxllm_core::deployment::{Deployment, OxllmParams};
use oxllm_core::provider::{ChatProvider, Provider, ResolvedCredentials};
use serde_json::json;

fn deployment() -> Deployment {
    Deployment {
        model_name: "llama-3.1-8b".to_string(),
        oxllm_params: OxllmParams {
            model: "cerebras/llama-3.1-8b".to_string(),
            api_key: Some("cerebras-key".to_string()),
            ..Default::default()
        },
        ..Default::default()
    }
}

#[test]
fn endpoint_url_uses_default_cerebras_base() {
    let p = CerebrasProvider;
    assert_eq!(
        p.endpoint_url(&deployment()),
        "https://api.cerebras.ai/v1/chat/completions"
    );
}

#[test]
fn endpoint_url_respects_custom_base() {
    let p = CerebrasProvider;
    let mut dep = deployment();
    dep.oxllm_params.api_base = Some("https://custom.example.com/v1/".to_string());
    assert_eq!(
        p.endpoint_url(&dep),
        "https://custom.example.com/v1/chat/completions"
    );
}

#[test]
fn apply_auth_uses_bearer_token() {
    let p = CerebrasProvider;
    let creds = ResolvedCredentials::from_key("cerebras-secret");
    let mut headers = std::collections::HashMap::new();
    p.apply_auth(&deployment(), &creds, &mut headers);
    assert_eq!(
        headers.get("authorization").map(String::as_str),
        Some("Bearer cerebras-secret")
    );
}

#[test]
fn request_strips_cerebras_prefix_from_model() {
    let p = CerebrasProvider;
    let req = json!({
        "model": "llama-3.1-8b",
        "messages": [{"role": "user", "content": "Hi"}],
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["model"], "llama-3.1-8b");
    assert_eq!(out["messages"][0]["content"], "Hi");
}

#[test]
fn request_preserves_stream_flag() {
    let p = CerebrasProvider;
    let req = json!({
        "model": "llama-3.1-8b",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": true,
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["stream"], true);
}

#[test]
fn response_is_identity() {
    let p = CerebrasProvider;
    let resp = json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    });
    let out = p.transform_response(&resp, "llama-3.1-8b").expect("ok");
    assert_eq!(out, resp);
}

#[test]
fn stream_chunk_parses_openai_format() {
    let p = CerebrasProvider;
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
    let p = CerebrasProvider;
    assert!(
        p.transform_stream_chunk("data: [DONE]")
            .expect("ok")
            .is_none()
    );
}

#[test]
fn provider_id_and_supports() {
    let p = CerebrasProvider;
    assert_eq!(p.id(), "cerebras");
    assert!(p.supports(Endpoint::ChatCompletions));
    assert!(!p.supports(Endpoint::Embeddings));
}

#[test]
fn strip_cerebras_prefix_handles_unknown() {
    assert_eq!(
        strip_cerebras_prefix("cerebras/llama-3.1-8b"),
        "llama-3.1-8b"
    );
    assert_eq!(strip_cerebras_prefix("llama-3.1-8b"), "llama-3.1-8b");
}

#[test]
fn supported_params_published() {
    let p = CerebrasProvider;
    let params = p.supported_params().expect("should publish params");
    assert!(params.contains(&"temperature"));
    assert!(params.contains(&"tools"));
}

#[test]
fn request_passes_known_params_and_drops_unknown() {
    let p = CerebrasProvider;
    let req = json!({
        "model": "llama-3.1-8b",
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
    let p = CerebrasProvider;
    let mut dep = deployment();
    dep.oxllm_params.model = "cerebras/*".to_string();
    let req = json!({
        "model": "cerebras/llama-3.1-8b",
        "messages": [{"role": "user", "content": "Hi"}]
    });
    let out = p.transform_request(&req, &dep).expect("ok");
    assert_eq!(out["model"], "llama-3.1-8b");
}

#[test]
fn strip_cerebras_prefix_with_various_inputs() {
    assert_eq!(
        strip_cerebras_prefix("cerebras/llama-3.1-8b"),
        "llama-3.1-8b"
    );
    assert_eq!(
        strip_cerebras_prefix("cerebras/llama3.1-70b"),
        "llama3.1-70b"
    );
    assert_eq!(strip_cerebras_prefix("llama-3.1-8b"), "llama-3.1-8b");
}

#[test]
fn parsed_reasoning_is_normalized_for_response_and_stream_without_losing_native_data() {
    let message = json!({"role": "assistant", "content": "42", "reasoning": "Compute the answer."});
    let response = json!({"choices": [{"message": message, "finish_reason": "stop"}], "usage": {"completion_tokens": 5}});
    let out = CerebrasProvider
        .transform_response(&response, "gpt-oss-120b")
        .unwrap();
    assert_eq!(
        out["choices"][0]["message"]["reasoning_content"],
        "Compute the answer."
    );
    assert_eq!(
        out["choices"][0]["message"]["reasoning"],
        "Compute the answer."
    );
    assert_eq!(out["choices"][0]["message"]["content"], "42");
    assert_eq!(out["usage"], response["usage"]);

    let chunk = r#"data: {"choices":[{"delta":{"reasoning":"Compute "},"finish_reason":null}]}"#;
    let out = CerebrasProvider
        .transform_stream_chunk(chunk)
        .unwrap()
        .unwrap();
    assert_eq!(out["choices"][0]["delta"]["reasoning_content"], "Compute ");
    assert!(out["choices"][0]["delta"].get("content").is_none());
}

#[test]
fn reasoning_normalization_does_not_parse_raw_content_or_overwrite_explicit_normalized_field() {
    for message in [
        json!({"content": "<think>raw</think>answer"}),
        json!({"content": "answer", "reasoning": {"unexpected": true}}),
        json!({"reasoning": "native", "reasoning_content": "explicit"}),
    ] {
        let response = json!({"choices": [{"message": message}]});
        assert_eq!(
            CerebrasProvider
                .transform_response(&response, "model")
                .unwrap(),
            response
        );
    }
}

#[test]
fn replayed_reasoning_uses_native_request_field_and_preserves_format() {
    let request = json!({"messages": [
        {"role": "assistant", "content": "42", "reasoning_content": "computed"},
        {"role": "assistant", "content": "43", "reasoning": "native", "reasoning_content": "alias"},
        {"role": "user", "content": "why?"}
    ], "reasoning_format": "parsed"});
    let out = CerebrasProvider
        .transform_request(&request, &deployment())
        .unwrap();
    assert_eq!(out["reasoning_format"], "parsed");
    assert_eq!(out["messages"][0]["reasoning"], "computed");
    assert_eq!(out["messages"][1]["reasoning"], "native");
    assert!(out["messages"][0].get("reasoning_content").is_none());
    assert!(
        CerebrasProvider
            .supported_params()
            .unwrap()
            .contains(&"reasoning_format")
    );
}
