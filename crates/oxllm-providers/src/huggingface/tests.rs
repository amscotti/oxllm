use super::*;
use oxllm_core::Endpoint;
use oxllm_core::deployment::{Deployment, OxllmParams};
use oxllm_core::provider::{ChatProvider, Provider, ResolvedCredentials};
use serde_json::json;

fn deployment() -> Deployment {
    Deployment {
        model_name: "meta-llama/Meta-Llama-3-8B-Instruct".to_string(),
        oxllm_params: OxllmParams {
            model: "huggingface/meta-llama/Meta-Llama-3-8B-Instruct".to_string(),
            api_key: Some("hf-key".to_string()),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn deployment_with_model(model: &str) -> Deployment {
    let mut dep = deployment();
    dep.oxllm_params.model = model.to_string();
    dep
}

// --- URL routing: ported from LiteLLM `test_get_complete_url` ---

#[test]
fn endpoint_url_default_for_plain_model() {
    let p = PROVIDER;
    let dep = deployment_with_model("huggingface/meta-llama/Llama-3-8B-Instruct");
    assert_eq!(
        p.endpoint_url(&dep),
        "https://router.huggingface.co/v1/chat/completions"
    );
}

#[test]
fn endpoint_url_routes_together_provider() {
    let p = PROVIDER;
    let dep = deployment_with_model("huggingface/together/meta-llama/Llama-3-8B-Instruct");
    assert_eq!(
        p.endpoint_url(&dep),
        "https://router.huggingface.co/together/v1/chat/completions"
    );
}

#[test]
fn endpoint_url_routes_novita_provider() {
    let p = PROVIDER;
    let dep = deployment_with_model("huggingface/novita/meta-llama/Llama-3-8B-Instruct");
    assert_eq!(
        p.endpoint_url(&dep),
        "https://router.huggingface.co/novita/v3/openai/chat/completions"
    );
}

#[test]
fn endpoint_url_routes_fireworks_provider() {
    let p = PROVIDER;
    let dep = deployment_with_model("huggingface/fireworks-ai/meta-llama/Llama-3-8B-Instruct");
    assert_eq!(
        p.endpoint_url(&dep),
        "https://router.huggingface.co/fireworks-ai/inference/v1/chat/completions"
    );
}

#[test]
fn endpoint_url_passes_through_full_url_model() {
    let p = PROVIDER;
    let dep = deployment_with_model("http://custom-endpoint.com/v1/chat/completions");
    assert_eq!(
        p.endpoint_url(&dep),
        "http://custom-endpoint.com/v1/chat/completions"
    );
}

#[test]
fn endpoint_url_routes_hf_inference_provider() {
    let p = PROVIDER;
    let dep = deployment_with_model("huggingface/hf-inference/meta-llama/Meta-Llama-3-8B-Instruct");
    assert_eq!(
        p.endpoint_url(&dep),
        "https://router.huggingface.co/hf-inference/models/meta-llama/Meta-Llama-3-8B-Instruct/v1/chat/completions"
    );
}

// --- URL routing with api_base: ported from LiteLLM `test_get_complete_url_inference_endpoints` ---

fn deployment_with_api_base(api_base: &str) -> Deployment {
    let mut dep = deployment_with_model("huggingface/tgi");
    dep.oxllm_params.api_base = Some(api_base.to_string());
    dep
}

#[test]
fn endpoint_url_with_api_base_cases() {
    let p = PROVIDER;
    let cases: &[(&str, &str)] = &[
        (
            "https://abcd123.us-east-1.aws.endpoints.huggingface.cloud",
            "https://abcd123.us-east-1.aws.endpoints.huggingface.cloud/v1/chat/completions",
        ),
        (
            "https://abcd123.us-east-1.aws.endpoints.huggingface.cloud/",
            "https://abcd123.us-east-1.aws.endpoints.huggingface.cloud/v1/chat/completions",
        ),
        (
            "https://abcd123.us-east-1.aws.endpoints.huggingface.cloud/v1/chat/completions",
            "https://abcd123.us-east-1.aws.endpoints.huggingface.cloud/v1/chat/completions",
        ),
        (
            "https://example.com/custom/path",
            "https://example.com/custom/path/v1/chat/completions",
        ),
        (
            "https://example.com/custom/path/v1/chat/completions",
            "https://example.com/custom/path/v1/chat/completions",
        ),
        (
            "https://example.com/v1",
            "https://example.com/v1/chat/completions",
        ),
    ];

    for (api_base, expected) in cases {
        let dep = deployment_with_api_base(api_base);
        assert_eq!(p.endpoint_url(&dep), *expected, "api_base: {api_base}");
    }
}

// --- URL helper: ported from LiteLLM `test_build_chat_completion_url_function` ---

#[test]
fn build_chat_completion_url_appends_paths() {
    let cases: &[(&str, &str)] = &[
        (
            "https://example.com",
            "https://example.com/v1/chat/completions",
        ),
        (
            "https://example.com/",
            "https://example.com/v1/chat/completions",
        ),
        (
            "https://example.com/v1",
            "https://example.com/v1/chat/completions",
        ),
        (
            "https://example.com/v1/",
            "https://example.com/v1/chat/completions",
        ),
        (
            "https://example.com/v1/chat/completions",
            "https://example.com/v1/chat/completions",
        ),
        (
            "https://example.com/custom/path",
            "https://example.com/custom/path/v1/chat/completions",
        ),
        (
            "https://example.com/custom/path/",
            "https://example.com/custom/path/v1/chat/completions",
        ),
    ];
    for (input, expected) in cases {
        assert_eq!(
            build_chat_completion_url(input),
            *expected,
            "input: {input}"
        );
    }
}

// --- Auth: ported from LiteLLM `test_validate_environment` ---

#[test]
fn apply_auth_uses_bearer_token() {
    let p = PROVIDER;
    let creds = ResolvedCredentials::from_key("test_api_key");
    let mut headers = std::collections::HashMap::new();
    p.apply_auth(&deployment(), &creds, &mut headers);
    assert_eq!(
        headers.get("authorization").map(String::as_str),
        Some("Bearer test_api_key")
    );
}

// --- Request transform: ported from LiteLLM `test_transform_request` (no-provider case) ---

#[test]
fn transform_request_preserves_model_without_provider_prefix() {
    // LiteLLM `test_transform_request` case where the model has no provider
    // prefix: the model id is forwarded unchanged. (The provider-prefix case
    // requires a network provider-mapping lookup, which is out of scope for a
    // pure transform.)
    let p = PROVIDER;
    let dep = deployment_with_model("huggingface/meta-llama/Meta-Llama-3-8B-Instruct");
    let req = json!({
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "messages": [{"role": "user", "content": "Hello"}],
    });
    let out = p.transform_request(&req, &dep).expect("ok");
    assert_eq!(out["model"], "meta-llama/Meta-Llama-3-8B-Instruct");
    assert_eq!(out["messages"][0]["content"], "Hello");
}

#[test]
fn request_strips_huggingface_prefix_from_model() {
    let p = PROVIDER;
    let req = json!({
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "messages": [{"role": "user", "content": "Hi"}],
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["model"], "meta-llama/Meta-Llama-3-8B-Instruct");
    assert_eq!(out["messages"][0]["content"], "Hi");
}

#[test]
fn request_preserves_stream_flag() {
    let p = PROVIDER;
    let req = json!({
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": true,
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["stream"], true);
}

#[test]
fn request_passes_known_params_and_drops_unknown() {
    let p = PROVIDER;
    let req = json!({
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "messages": [{"role": "user", "content": "Hi"}],
        "temperature": 0.5,
        "max_tokens": 100,
        "stream": false,
        "some_unknown_param": "ignored",
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["temperature"], 0.5);
    assert_eq!(out["max_tokens"], 100);
    assert!(out.get("some_unknown_param").is_none());
}

#[test]
fn wildcard_deployment_preserves_requested_model() {
    let p = PROVIDER;
    let dep = deployment_with_model("huggingface/*");
    let req = json!({
        "model": "huggingface/meta-llama/Llama-3-8B-Instruct",
        "messages": [{"role": "user", "content": "Hi"}]
    });
    let out = p.transform_request(&req, &dep).expect("ok");
    assert_eq!(out["model"], "meta-llama/Llama-3-8B-Instruct");
}

// --- Response / stream: OpenAI-compat identity, ported from the mock shapes ---

#[test]
fn response_is_identity() {
    let p = PROVIDER;
    let resp = json!({
        "id": "9115d3daeab10608",
        "object": "chat.completion",
        "created": 11111,
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a test response from the mocked HuggingFace API."
                }
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    });
    let out = p
        .transform_response(&resp, "meta-llama/Meta-Llama-3-8B-Instruct")
        .expect("ok");
    assert_eq!(out, resp);
}

#[test]
fn stream_chunk_parses_openai_format() {
    let p = PROVIDER;
    let raw = r#"data: {"id":"id1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":"Deep"}}]}"#;
    let out = p
        .transform_stream_chunk(raw)
        .expect("ok")
        .expect("should produce value");
    assert_eq!(out["id"], "id1");
    assert_eq!(out["choices"][0]["delta"]["content"], "Deep");
}

#[test]
fn stream_done_returns_none() {
    let p = PROVIDER;
    assert!(
        p.transform_stream_chunk("data: [DONE]")
            .expect("ok")
            .is_none()
    );
}

// --- Provider metadata ---

#[test]
fn supported_params_published() {
    let p = PROVIDER;
    let params = p.supported_params().expect("should publish params");
    assert!(params.contains(&"temperature"));
    assert!(params.contains(&"tools"));
    assert!(params.contains(&"max_tokens"));
}

#[test]
fn provider_id_and_supports() {
    let p = PROVIDER;
    assert_eq!(p.id(), "huggingface");
    assert!(p.supports(Endpoint::ChatCompletions));
    assert!(!p.supports(Endpoint::Embeddings));
    assert!(!p.supports(Endpoint::Rerank));
}

#[test]
fn provider_exposes_chat() {
    let p = PROVIDER;
    assert!(Provider::chat(&p).is_some());
    assert!(Provider::embeddings(&p).is_none());
}

// --- Prefix stripping helper ---

#[test]
fn strip_huggingface_prefix_handles_known_and_unknown() {
    assert_eq!(
        strip_huggingface_prefix("huggingface/meta-llama/Meta-Llama-3-8B-Instruct"),
        "meta-llama/Meta-Llama-3-8B-Instruct"
    );
    assert_eq!(
        strip_huggingface_prefix("meta-llama/Meta-Llama-3-8B-Instruct"),
        "meta-llama/Meta-Llama-3-8B-Instruct"
    );
    assert_eq!(
        strip_huggingface_prefix("huggingface/together/meta-llama/Llama-3-8B-Instruct"),
        "together/meta-llama/Llama-3-8B-Instruct"
    );
}
