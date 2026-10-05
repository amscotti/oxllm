use super::*;
use oxllm_core::OxllmParams;
use serde_json::json;

fn deployment(model: &str) -> Deployment {
    Deployment {
        model_name: "fast".into(),
        oxllm_params: OxllmParams {
            model: model.into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

#[test]
fn chat_contract_preserves_model_namespace_tools_and_the_modern_token_limit() {
    let model = "baseten/deepseek-ai/DeepSeek-V4.1-Flash-Fast";
    let request = json!({
        "model":"fast","messages":[{"role":"user","content":"hi"}],
        "max_tokens":10,"max_completion_tokens":20,
        "tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object"}}}],
        "tool_choice":"auto","stream":true,"stream_options":{"include_usage":true},
        "unsupported_control":true
    });
    let body = PROVIDER
        .transform_request(&request, &deployment(model))
        .unwrap();
    assert_eq!(body["model"], "deepseek-ai/DeepSeek-V4.1-Flash-Fast");
    assert_eq!(body["max_tokens"], 20);
    assert!(body.get("max_completion_tokens").is_none());
    assert!(body.get("unsupported_control").is_none());
    assert_eq!(body["tools"], request["tools"]);
    assert_eq!(body["stream_options"]["include_usage"], true);
    let wildcard = deployment("baseten/*");
    assert_eq!(
        PROVIDER
            .transform_request(&json!({"model":model,"messages":[]}), &wildcard)
            .unwrap()["model"],
        "deepseek-ai/DeepSeek-V4.1-Flash-Fast"
    );
}

#[test]
fn auth_custom_base_and_stream_usage_follow_the_compatible_contract() {
    let mut deployment = deployment("baseten/deepseek-ai/DeepSeek-V4.1-Flash-Fast");
    assert_eq!(
        PROVIDER.endpoint_url(&deployment),
        "https://inference.baseten.co/v1/chat/completions"
    );
    deployment.oxllm_params.api_base = Some("https://private.example/v1/chat/completions".into());
    assert_eq!(
        PROVIDER.endpoint_url(&deployment),
        "https://private.example/v1/chat/completions"
    );
    let mut headers = HashMap::new();
    PROVIDER.apply_auth(
        &deployment,
        &ResolvedCredentials::from_key("test-key"),
        &mut headers,
    );
    assert_eq!(
        headers.get("authorization").map(String::as_str),
        Some("Bearer test-key")
    );
    let chunk = PROVIDER.transform_stream_chunk("data: {\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5,\"total_tokens\":15}}").unwrap().unwrap();
    assert_eq!(chunk["usage"]["total_tokens"], 15);
    assert!(
        PROVIDER
            .transform_stream_chunk("data: [DONE]")
            .unwrap()
            .is_none()
    );
}
