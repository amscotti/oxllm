use super::*;
use oxllm_core::Endpoint;
use oxllm_core::deployment::{Deployment, OxllmParams};
use oxllm_core::provider::{ChatProvider, Provider, ResolvedCredentials};
use serde_json::json;

const ACCOUNT_BASE: &str = "https://api.cloudflare.com/client/v4/accounts/fake-acct/ai/v1";

fn deployment() -> Deployment {
    Deployment {
        model_name: "@cf/meta/llama-2-7b-chat-int8".to_string(),
        oxllm_params: OxllmParams {
            model: "cloudflare/@cf/meta/llama-2-7b-chat-int8".to_string(),
            api_key: Some("fake-cf-api-key".to_string()),
            api_base: Some(ACCOUNT_BASE.to_string()),
            ..Default::default()
        },
        ..Default::default()
    }
}

#[test]
fn endpoint_url_uses_account_scoped_base() {
    let p = CloudflareProvider;
    assert_eq!(
        p.endpoint_url(&deployment()),
        "https://api.cloudflare.com/client/v4/accounts/fake-acct/ai/v1/chat/completions"
    );
}

#[test]
fn endpoint_url_strips_trailing_slash_from_account_base() {
    let p = CloudflareProvider;
    let mut dep = deployment();
    dep.oxllm_params.api_base = Some(format!("{ACCOUNT_BASE}/"));
    assert_eq!(
        p.endpoint_url(&dep),
        "https://api.cloudflare.com/client/v4/accounts/fake-acct/ai/v1/chat/completions"
    );
}

#[test]
fn endpoint_url_targets_chat_completions_not_ai_run() {
    // Cloudflare's OpenAI-compatible surface lives under /ai/v1/chat/completions,
    // not the per-model /ai/run/ path.
    let p = CloudflareProvider;
    let url = p.endpoint_url(&deployment());
    assert!(url.ends_with("/ai/v1/chat/completions"));
    assert!(!url.contains("/ai/run/"));
}

#[test]
fn apply_auth_uses_bearer_token() {
    let p = CloudflareProvider;
    let creds = ResolvedCredentials::from_key("cf-secret");
    let mut headers = std::collections::HashMap::new();
    p.apply_auth(&deployment(), &creds, &mut headers);
    assert_eq!(
        headers.get("authorization").map(String::as_str),
        Some("Bearer cf-secret")
    );
}

#[test]
fn request_strips_cloudflare_prefix_from_model() {
    let p = CloudflareProvider;
    let req = json!({
        "model": "@cf/meta/llama-2-7b-chat-int8",
        "messages": [{"role": "user", "content": "what llm are you"}],
        "max_tokens": 15,
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["model"], "@cf/meta/llama-2-7b-chat-int8");
    assert_eq!(out["messages"][0]["content"], "what llm are you");
    assert_eq!(out["max_tokens"], 15);
}

#[test]
fn request_preserves_tools_and_tool_choice() {
    let p = CloudflareProvider;
    let tools = json!([{
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }
        }
    }]);
    let req = json!({
        "model": "cloudflare/@cf/meta/llama-2-7b-chat-int8",
        "messages": [{"role": "user", "content": "weather in New York?"}],
        "tools": tools,
        "tool_choice": "auto",
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["tools"], tools);
    assert_eq!(out["tool_choice"], "auto");
}

#[test]
fn request_preserves_stream_flag() {
    let p = CloudflareProvider;
    let req = json!({
        "model": "@cf/meta/llama-2-7b-chat-int8",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": true,
        "max_tokens": 15,
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["stream"], true);
}

#[test]
fn response_is_identity() {
    let p = CloudflareProvider;
    let resp = json!({
        "id": "chatcmpl-cf",
        "object": "chat.completion",
        "model": "@cf/meta/llama-2-7b-chat-int8",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "I am a large language model."},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 8, "completion_tokens": 11, "total_tokens": 19},
    });
    let out = p
        .transform_response(&resp, "@cf/meta/llama-2-7b-chat-int8")
        .expect("ok");
    assert_eq!(out, resp);
}

#[test]
fn stream_chunk_parses_openai_format() {
    let p = CloudflareProvider;
    let raw = "data: {\"id\":\"chatcmpl-cf\",\"choices\":[{\"delta\":{\"content\":\"I am\"}}]}";
    let out = p
        .transform_stream_chunk(raw)
        .expect("ok")
        .expect("should produce value");
    assert_eq!(out["id"], "chatcmpl-cf");
    assert_eq!(out["choices"][0]["delta"]["content"], "I am");
}

#[test]
fn stream_done_returns_none() {
    let p = CloudflareProvider;
    assert!(
        p.transform_stream_chunk("data: [DONE]")
            .expect("ok")
            .is_none()
    );
}

#[test]
fn provider_id_and_supports() {
    let p = CloudflareProvider;
    assert_eq!(p.id(), "cloudflare");
    assert!(p.supports(Endpoint::ChatCompletions));
    assert!(!p.supports(Endpoint::Embeddings));
}

#[test]
fn strip_cloudflare_prefix_preserves_nested_model_path() {
    // Only the leading `cloudflare/` segment is dropped; the `@cf/meta/...`
    // path survives intact.
    assert_eq!(
        strip_cloudflare_prefix("cloudflare/@cf/meta/llama-2-7b-chat-int8"),
        "@cf/meta/llama-2-7b-chat-int8"
    );
    assert_eq!(
        strip_cloudflare_prefix("@cf/meta/llama-2-7b-chat-int8"),
        "@cf/meta/llama-2-7b-chat-int8"
    );
}

#[test]
fn supported_params_published() {
    let p = CloudflareProvider;
    let params = p.supported_params().expect("should publish params");
    assert!(params.contains(&"temperature"));
    assert!(params.contains(&"tools"));
}

/// Documented footgun: when a Cloudflare deployment forgets to set `api_base`,
/// the default template (with its literal `{account_id}` placeholder) leaks
/// into the request URL. The URL is unusable and Cloudflare 404s with a
/// confusing error. This test pins the behaviour so the footgun is visible in
/// code (and so a future change that swaps in a real default is noticed).
#[test]
fn endpoint_url_without_api_base_leaks_account_id_placeholder() {
    let p = CloudflareProvider;
    let mut dep = deployment();
    dep.oxllm_params.api_base = None;
    let url = p.endpoint_url(&dep);
    assert!(
        url.contains("{account_id}"),
        "default base carries a literal placeholder that callers must replace: {url}"
    );
}
