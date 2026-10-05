use super::*;
use oxllm_core::Endpoint;
use oxllm_core::deployment::{Deployment, OxllmParams};
use oxllm_core::provider::{ChatProvider, Provider, ResolvedCredentials};
use serde_json::json;

fn deployment() -> Deployment {
    Deployment {
        model_name: "openai.gpt-oss-120b".to_string(),
        oxllm_params: OxllmParams {
            model: "bedrock_mantle/openai.gpt-oss-120b".to_string(),
            api_key: Some("bedrock-api-key-secret".to_string()),
            ..Default::default()
        },
        ..Default::default()
    }
}

#[test]
fn provider_id_and_supports() {
    let p = BedrockMantleProvider;
    assert_eq!(p.id(), "bedrock_mantle");
    assert!(p.supports(Endpoint::ChatCompletions));
    assert!(!p.supports(Endpoint::Embeddings));
}

#[test]
fn default_base_url() {
    let p = BedrockMantleProvider;
    assert_eq!(
        p.endpoint_url(&deployment()),
        "https://bedrock-mantle.us-east-1.api.aws/v1/chat/completions"
    );
}

#[test]
fn modern_models_and_wildcards_use_the_model_specific_api_base() {
    let provider = BedrockMantleProvider;
    let mut deployment = deployment();
    for model in [
        "openai.gpt-6.1-sol",
        "openai.gpt-6-astra",
        "google.gemma-4-31b",
    ] {
        deployment.oxllm_params.model = format!("bedrock_mantle/{model}");
        assert_eq!(
            provider.endpoint_url(&deployment),
            "https://bedrock-mantle.us-east-1.api.aws/openai/v1/chat/completions"
        );
    }
    deployment.oxllm_params.model = "bedrock_mantle/*".into();
    assert_eq!(
        provider.endpoint_url_for_request(&deployment, &json!({"model":"openai.gpt-6.1-sol"})),
        "https://bedrock-mantle.us-east-1.api.aws/openai/v1/chat/completions"
    );
    assert_eq!(
        provider.endpoint_url_for_request(&deployment, &json!({"model":"openai.gpt-oss-120b"})),
        "https://bedrock-mantle.us-east-1.api.aws/v1/chat/completions"
    );
    deployment.oxllm_params.api_base = Some("https://private.example/custom".into());
    assert_eq!(
        provider.endpoint_url_for_request(&deployment, &json!({"model":"openai.gpt-6.1-sol"})),
        "https://private.example/custom/chat/completions"
    );
}

#[test]
fn endpoint_url_respects_custom_region_base() {
    let p = BedrockMantleProvider;
    let mut dep = deployment();
    dep.oxllm_params.api_base = Some("https://bedrock-mantle.eu-west-1.api.aws/v1".to_string());
    assert_eq!(
        p.endpoint_url(&dep),
        "https://bedrock-mantle.eu-west-1.api.aws/v1/chat/completions"
    );
}

#[test]
fn strip_bedrock_mantle_prefix_cases() {
    assert_eq!(
        strip_bedrock_mantle_prefix("bedrock_mantle/openai.gpt-oss-120b"),
        "openai.gpt-oss-120b"
    );
    assert_eq!(
        strip_bedrock_mantle_prefix("bedrock_mantle/openai.gpt-oss-20b"),
        "openai.gpt-oss-20b"
    );
    assert_eq!(
        strip_bedrock_mantle_prefix("openai.gpt-oss-120b"),
        "openai.gpt-oss-120b"
    );
    assert_eq!(
        strip_bedrock_mantle_prefix("openai.gpt-5.5"),
        "openai.gpt-5.5"
    );
}

#[test]
fn auth_uses_bearer_token() {
    let p = BedrockMantleProvider;
    let creds = ResolvedCredentials::from_key("bedrock-api-key-secret");
    let mut headers = std::collections::HashMap::new();
    p.apply_auth(&deployment(), &creds, &mut headers);
    assert_eq!(
        headers.get("authorization").map(String::as_str),
        Some("Bearer bedrock-api-key-secret")
    );
}

#[test]
fn registered_in_default_registry() {
    use crate::default_registry;
    let registry = default_registry();
    let provider = registry
        .get("bedrock_mantle")
        .expect("bedrock_mantle should be registered in the default registry");
    assert_eq!(provider.id(), "bedrock_mantle");
    assert!(provider.supports(Endpoint::ChatCompletions));
}

#[test]
fn request_strips_bedrock_mantle_prefix_from_model() {
    let p = BedrockMantleProvider;
    let req = json!({
        "model": "openai.gpt-oss-120b",
        "messages": [{"role": "user", "content": "Hello, world!"}],
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["model"], "openai.gpt-oss-120b");
    assert_eq!(out["messages"][0]["content"], "Hello, world!");
}

#[test]
fn response_is_identity() {
    let p = BedrockMantleProvider;
    let resp = json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    });
    let out = p
        .transform_response(&resp, "openai.gpt-oss-120b")
        .expect("ok");
    assert_eq!(out, resp);
}

#[test]
fn stream_chunk_parses_openai_format() {
    let p = BedrockMantleProvider;
    let raw = "data: {\"id\":\"x\",\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}";
    let out = p
        .transform_stream_chunk(raw)
        .expect("ok")
        .expect("should produce value");
    assert_eq!(out["id"], "x");
    assert_eq!(out["choices"][0]["delta"]["content"], "hi");
}

#[test]
fn gpt6_request_maps_max_tokens_and_drops_temperature() {
    let p = BedrockMantleProvider;
    let mut dep = deployment();
    dep.oxllm_params.model = "bedrock_mantle/openai.gpt-6-astra".to_string();
    let req = json!({
        "model": "openai.gpt-6-astra",
        "messages": [{"role": "user", "content": "Hi"}],
        "temperature": 0.3,
        "max_tokens": 20,
        "top_p": 0.8,
    });
    let out = p.transform_request(&req, &dep).expect("ok");
    assert_eq!(out["model"], "openai.gpt-6-astra");
    assert!(out.get("temperature").is_none());
    assert!(out.get("top_p").is_none());
    assert_eq!(out["max_completion_tokens"], 20);
}
