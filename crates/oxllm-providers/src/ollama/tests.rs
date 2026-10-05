use super::*;
use oxllm_core::Endpoint;
use oxllm_core::deployment::{Deployment, OxllmParams};
use oxllm_core::provider::{ChatProvider, Provider, ResolvedCredentials};
use serde_json::json;

fn deployment() -> Deployment {
    Deployment {
        model_name: "llama3".to_string(),
        oxllm_params: OxllmParams {
            model: "ollama/llama3".to_string(),
            ..Default::default()
        },
        ..Default::default()
    }
}

#[test]
fn endpoint_url_uses_default_ollama_base() {
    let p = OllamaProvider;
    assert_eq!(
        p.endpoint_url(&deployment()),
        "http://localhost:11434/v1/chat/completions"
    );
}

#[test]
fn endpoint_url_respects_custom_base() {
    let p = OllamaProvider;
    let mut dep = deployment();
    dep.oxllm_params.api_base = Some("http://ollama-host:11434/v1/".to_string());
    assert_eq!(
        p.endpoint_url(&dep),
        "http://ollama-host:11434/v1/chat/completions"
    );
}

#[test]
fn apply_auth_does_not_set_anything() {
    let p = OllamaProvider;
    let creds = ResolvedCredentials::from_key("");
    let mut headers = std::collections::HashMap::new();
    p.apply_auth(&deployment(), &creds, &mut headers);
    assert!(headers.is_empty());
}

#[test]
fn request_strips_ollama_prefix() {
    let p = OllamaProvider;
    let req = json!({
    "model": "llama3",
    "messages": [{"role": "user", "content": "Hi"}],
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["model"], "llama3");
}

#[test]
fn request_passes_known_params() {
    let p = OllamaProvider;
    let req = json!({
    "model": "llama3",
    "messages": [{"role": "user", "content": "Hi"}],
    "temperature": 0.7,
    "max_tokens": 50,
    "stream": false,
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["temperature"], 0.7);
    assert_eq!(out["max_tokens"], 50);
    assert_eq!(out["messages"][0]["content"], "Hi");
}

#[test]
fn response_is_identity() {
    let p = OllamaProvider;
    let resp = json!({
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
    });
    let out = p.transform_response(&resp, "llama3").expect("ok");
    assert_eq!(out, resp);
}

#[test]
fn stream_chunk_parses_openai_format() {
    let p = OllamaProvider;
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
    let p = OllamaProvider;
    assert!(
        p.transform_stream_chunk("data: [DONE]")
            .expect("ok")
            .is_none()
    );
}

#[test]
fn stream_empty_chunk_returns_none() {
    let p = OllamaProvider;
    assert!(p.transform_stream_chunk("").expect("ok").is_none());
}

#[test]
fn stream_invalid_json_errors() {
    let p = OllamaProvider;
    assert!(p.transform_stream_chunk("data: {bad json").is_err());
}

#[test]
fn provider_id_and_supports() {
    let p = OllamaProvider;
    assert_eq!(p.id(), "ollama");
    assert!(p.supports(Endpoint::ChatCompletions));
    assert!(p.supports(Endpoint::Embeddings));
}

#[test]
fn strip_ollama_prefix_handles_unknown() {
    assert_eq!(strip_ollama_prefix("ollama/llama3"), "llama3");
    assert_eq!(strip_ollama_prefix("llama3"), "llama3");
}

const GOLDEN_REQUEST_001: &str = include_str!("goldens/chat_request_001.json");
const GOLDEN_REQUEST_002: &str = include_str!("goldens/chat_request_002.json");
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
    let provider = OllamaProvider;
    let deployment = deployment();

    let cases: &[(&str, &str, bool)] = &[
        ("chat_request_001", GOLDEN_REQUEST_001, true),
        ("chat_request_002", GOLDEN_REQUEST_002, true),
        ("chat_response_001", GOLDEN_RESPONSE_001, false),
    ];

    for (name, golden, is_request) in cases {
        let input: Value = serde_json::from_str(golden)
            .unwrap_or_else(|e| panic!("golden {name} should be valid JSON: {e}"));
        let output = if *is_request {
            ChatProvider::transform_request(&provider, &input, &deployment)
                .unwrap_or_else(|e| panic!("golden {name} request transform failed: {e}"))
        } else {
            ChatProvider::transform_response(&provider, &input, "llama3")
                .unwrap_or_else(|e| panic!("golden {name} response transform failed: {e}"))
        };
        assert_semantically_equal(&output, &input);
    }
}
