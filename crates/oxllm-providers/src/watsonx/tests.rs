use super::*;
use oxllm_core::Endpoint;
use oxllm_core::deployment::{Deployment, OxllmParams};
use oxllm_core::provider::{ChatProvider, Provider, ResolvedCredentials};
use serde_json::json;

fn deployment() -> Deployment {
    Deployment {
        model_name: "my-test-model".to_string(),
        oxllm_params: OxllmParams {
            model: "watsonx/my-test-model".to_string(),
            api_key: Some("test_api_key".to_string()),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn deployment_with_project_id(project_id: &str) -> Deployment {
    let mut dep = deployment();
    dep.oxllm_params.extra =
        serde_json::from_str(&format!(r#"{{"project_id":"{project_id}"}}"#)).expect("valid json");
    dep
}

fn deployment_with_space_id(space_id: &str) -> Deployment {
    let mut dep = deployment();
    dep.oxllm_params.extra =
        serde_json::from_str(&format!(r#"{{"space_id":"{space_id}"}}"#)).expect("valid json");
    dep
}

#[test]
fn wildcard_model_survives_supported_parameter_filtering() {
    let params = WatsonXProvider
        .supported_params()
        .expect("supported parameters");
    for (configured, requested) in [
        ("watsonx/*", "watsonx/ibm/granite-3-8b-instruct"),
        ("watsonx_text/*", "watsonx_text/ibm/granite-3-8b-instruct"),
    ] {
        let mut dep = deployment_with_project_id("project-1");
        dep.oxllm_params.model = configured.into();
        let mut request = json!({
            "model": requested,
            "messages": [{"role": "user", "content": "hello"}],
            "unsupported": true,
        });
        request
            .as_object_mut()
            .unwrap()
            .retain(|key, _| params.contains(&key.as_str()));
        let body = WatsonXProvider.transform_request(&request, &dep).unwrap();
        assert_eq!(body["model_id"], "ibm/granite-3-8b-instruct");
        assert_eq!(body["project_id"], "project-1");
        assert!(body.get("model").is_none());
        assert!(body.get("unsupported").is_none());
    }
}

// =============================================================================
// Ported from the reference `tests/llm_translation/test_watsonx.py`.
//
// The reference WatsonX suite patches the HTTP layer and IAM token exchange and
// asserts on the outgoing request (URL, headers, body). Those behaviors map
// directly onto the oxllm transform surface: `endpoint_url`, `apply_auth`, and
// `transform_request`. The ports below cover the cases that exercise the
// transform logic (auth header selection, payload project_id/space_id, and
// deployment-model handling).
// =============================================================================

/// Ported from `test_watsonx_custom_auth_header` (custom header case): when a
/// caller-supplied bearer token is resolved, it is forwarded verbatim in the
/// `Authorization` header.
#[test]
fn apply_auth_uses_resolved_bearer_token() {
    let p = WatsonXProvider;
    let creds = ResolvedCredentials::from_key("my-custom-auth-header");
    let mut headers = std::collections::HashMap::new();
    p.apply_auth(&deployment(), &creds, &mut headers);
    assert_eq!(
        headers.get("authorization").map(String::as_str),
        Some("Bearer my-custom-auth-header")
    );
}

/// Ported from `test_watsonx_custom_auth_header` (default case): the IAM token
/// resolved by the host is forwarded as a bearer token.
#[test]
fn apply_auth_uses_iam_bearer_token() {
    let p = WatsonXProvider;
    let creds = ResolvedCredentials::from_key("mock_access_token");
    let mut headers = std::collections::HashMap::new();
    p.apply_auth(&deployment(), &creds, &mut headers);
    assert_eq!(
        headers.get("authorization").map(String::as_str),
        Some("Bearer mock_access_token")
    );
}

/// Ported from `test_watsonx_token_in_env_var` (`WATSONX_TOKEN`): a token
/// resolved by the host is forwarded as a bearer token.
#[test]
fn apply_auth_uses_watsonx_token() {
    let p = WatsonXProvider;
    let creds = ResolvedCredentials::from_key("my-custom-token");
    let mut headers = std::collections::HashMap::new();
    p.apply_auth(&deployment(), &creds, &mut headers);
    assert_eq!(
        headers.get("authorization").map(String::as_str),
        Some("Bearer my-custom-token")
    );
}

/// Ported from `test_watsonx_chat_completions_endpoint`: the chat URL for a
/// regular model targets `/ml/v1/text/chat` and must not mention `deployment`.
#[test]
fn endpoint_url_targets_text_chat_and_not_deployment() {
    let p = WatsonXProvider;
    let mut dep = deployment();
    dep.oxllm_params.model = "watsonx/another-model".to_string();
    let url = p.endpoint_url(&dep);
    assert!(url.contains("/ml/v1/text/chat"), "url={url}");
    assert!(!url.contains("deployment"), "url={url}");
    assert!(
        url.contains("version="),
        "url must carry api version: {url}"
    );
}

/// Ported from `test_watsonx_chat_completions_endpoint_space_id`: when a
/// `space_id` is configured (and no `project_id`), the request payload carries
/// `space_id` and not `project_id`.
#[test]
fn request_carries_space_id_when_no_project_id() {
    let p = WatsonXProvider;
    let mut dep = deployment_with_space_id("xxx-xxx-xxx-xxx-xxx");
    dep.oxllm_params.model = "watsonx/another-model".to_string();
    let req = json!({
        "model": "watsonx/another-model",
        "messages": [{"role": "user", "content": "Test message"}],
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(out["space_id"], "xxx-xxx-xxx-xxx-xxx");
    assert!(
        out.get("project_id").is_none(),
        "project_id must be absent when only space_id is set"
    );
    assert_eq!(out["model_id"], "another-model");
    assert!(
        out.get("model").is_none(),
        "model must be renamed to model_id"
    );
}

/// Ported from `test_watsonx_deployment_space_id` / `test_watsonx_deployment`:
/// a `deployment/<id>` model must not carry `project_id` or `space_id` in the
/// payload, and the URL targets the deployments endpoint.
#[test]
fn deployment_model_omits_project_and_space_id() {
    let p = WatsonXProvider;
    let mut dep = deployment_with_space_id("xxx-xxx-xxx-xxx-xxx");
    dep.oxllm_params.model = "watsonx/deployment/<xxxx.xxx.xxx.xxxx>".to_string();
    let req = json!({
        "model": "watsonx/deployment/<xxxx.xxx.xxx.xxxx>",
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert!(
        out.get("project_id").is_none() && out.get("space_id").is_none(),
        "deployment models must not carry project_id/space_id"
    );
    assert!(
        out.get("model_id").is_none(),
        "deployment models do not set model_id in the body"
    );
    assert!(out.get("model").is_none());
    let url = p.endpoint_url(&dep);
    assert!(url.contains("/deployments/"), "url={url}");
    assert!(url.contains("/text/chat"), "url={url}");
}

/// Ported from `test_watsonx_deployment` (`watsonx_text/` prefix): the
/// `watsonx_text/` provider prefix is stripped just like `watsonx/`.
#[test]
fn deployment_model_strips_watsonx_text_prefix() {
    let p = WatsonXProvider;
    let mut dep = deployment();
    dep.oxllm_params.model = "watsonx_text/deployment/<xxxx>".to_string();
    let url = p.endpoint_url(&dep);
    assert!(url.contains("/deployments/<xxxx>/text/chat"), "url={url}");
}

// =============================================================================
// Core transform tests (URL default base, request shape, response, streaming).
// =============================================================================

#[test]
fn endpoint_url_uses_default_watsonx_base() {
    let p = WatsonXProvider;
    let url = p.endpoint_url(&deployment_with_project_id("proj-1"));
    assert_eq!(
        url,
        "https://us-south.ml.cloud.ibm.com/ml/v1/text/chat?version=2024-03-13"
    );
}

#[test]
fn endpoint_url_respects_custom_base() {
    let p = WatsonXProvider;
    let mut dep = deployment_with_project_id("proj-1");
    dep.oxllm_params.api_base = Some("https://eu-de.ml.cloud.ibm.com".to_string());
    let url = p.endpoint_url(&dep);
    assert_eq!(
        url,
        "https://eu-de.ml.cloud.ibm.com/ml/v1/text/chat?version=2024-03-13"
    );
}

#[test]
fn endpoint_url_honors_custom_api_version() {
    let p = WatsonXProvider;
    let mut dep = deployment_with_project_id("proj-1");
    dep.oxllm_params.api_version = Some("2025-01-01".to_string());
    let url = p.endpoint_url(&dep);
    assert!(url.contains("version=2025-01-01"), "url={url}");
}

/// A custom `api_version` containing URL-special characters (`&`, `#`, `=`,
/// spaces) must be percent-encoded so it cannot malform the query string.
/// The default (`2024-03-13`) is a date pattern and encodes to itself.
#[test]
fn endpoint_url_percent_encodes_special_chars_in_api_version() {
    let p = WatsonXProvider;
    let mut dep = deployment_with_project_id("proj-1");
    dep.oxllm_params.api_version = Some("2024-03-13&injected=evil".to_string());
    let url = p.endpoint_url(&dep);
    assert!(
        url.contains("version=2024-03-13%26injected%3Devil"),
        "url={url}"
    );
    assert!(
        !url.contains("&injected="),
        "raw `&` must not leak into the URL: {url}"
    );
}

/// A `#` in `api_version` must be encoded so it cannot truncate the URL into
/// a fragment.
#[test]
fn endpoint_url_encodes_hash_in_api_version() {
    let p = WatsonXProvider;
    let mut dep = deployment_with_project_id("proj-1");
    dep.oxllm_params.api_version = Some("bad#version".to_string());
    let url = p.endpoint_url(&dep);
    assert!(url.contains("version=bad%23version"), "url={url}");
    assert!(!url.contains('#'), "raw `#` must not appear: {url}");
}

#[test]
fn request_renames_model_to_model_id() {
    let p = WatsonXProvider;
    let dep = deployment_with_project_id("proj-1");
    let req = json!({
        "model": "watsonx/my-test-model",
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(out["model_id"], "my-test-model");
    assert!(
        out.get("model").is_none(),
        "model must be renamed to model_id"
    );
    assert_eq!(out["project_id"], "proj-1");
}

#[test]
fn request_prefers_project_id_over_space_id() {
    let p = WatsonXProvider;
    let mut dep = deployment_with_project_id("proj-1");
    dep.oxllm_params
        .extra
        .insert("space_id".to_string(), json!("space-1"));
    let req = json!({
        "messages": [{"role": "user", "content": "Hi"}],
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(out["project_id"], "proj-1");
    assert!(
        out.get("space_id").is_none(),
        "project_id wins over space_id"
    );
}

#[test]
fn request_preserves_openai_params() {
    let p = WatsonXProvider;
    let dep = deployment_with_project_id("proj-1");
    let req = json!({
        "model": "watsonx/my-test-model",
        "messages": [{"role": "user", "content": "Hi"}],
        "temperature": 0.5,
        "max_tokens": 100,
        "top_p": 0.9,
        "stream": false,
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(out["temperature"], 0.5);
    assert_eq!(out["max_tokens"], 100);
    assert_eq!(out["top_p"], 0.9);
    assert_eq!(out["stream"], false);
    assert_eq!(out["messages"][0]["content"], "Hi");
}

#[test]
fn request_strips_watsonx_text_prefix_for_model_id() {
    let p = WatsonXProvider;
    let mut dep = deployment_with_project_id("proj-1");
    dep.oxllm_params.model = "watsonx_text/ibm/granite-3-8b-instruct".to_string();
    let req = json!({
        "model": "watsonx_text/ibm/granite-3-8b-instruct",
        "messages": [{"role": "user", "content": "Hi"}],
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(out["model_id"], "ibm/granite-3-8b-instruct");
}

#[test]
fn response_passes_through_openai_shape() {
    let p = WatsonXProvider;
    let resp = json!({
        "id": "chat-1",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from WatsonX"},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    });
    let out = ChatProvider::transform_response(&p, &resp, "my-test-model").expect("ok");
    assert_eq!(out, resp);
}

#[test]
fn stream_chunk_parses_openai_format() {
    let p = WatsonXProvider;
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
    let p = WatsonXProvider;
    assert!(
        p.transform_stream_chunk("data: [DONE]")
            .expect("ok")
            .is_none()
    );
    assert!(p.transform_stream_chunk("").expect("ok").is_none());
}

#[test]
fn stream_comment_line_returns_none() {
    let p = WatsonXProvider;
    assert!(
        p.transform_stream_chunk(": keep-alive")
            .expect("ok")
            .is_none()
    );
    assert!(
        p.transform_stream_chunk("event: ping")
            .expect("ok")
            .is_none()
    );
}

/// A multi-line SSE event block (`event:` followed by `data:` in the same
/// chunk) must dispatch the JSON payload, not drop it as metadata. This was
/// lost when the bespoke parser rejected the whole chunk on any `event:` line.
#[test]
fn stream_multiline_event_block_dispatches_data() {
    let p = WatsonXProvider;
    let raw =
        "event: ping\ndata: {\"id\":\"x\",\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n";
    let out = p
        .transform_stream_chunk(raw)
        .expect("ok")
        .expect("should produce value");
    assert_eq!(out["id"], "x");
    assert_eq!(out["choices"][0]["delta"]["content"], "hi");
}

/// An empty `data:` line (a legitimate keep-alive) must not terminate the
/// stream with a parse error. The shared parser skips it cleanly.
#[test]
fn stream_empty_data_line_returns_none() {
    let p = WatsonXProvider;
    assert!(p.transform_stream_chunk("data:").expect("ok").is_none());
    assert!(
        p.transform_stream_chunk("data:   \n\n")
            .expect("ok")
            .is_none()
    );
}

/// An inline upstream error inside the SSE stream must surface as
/// `Error::Upstream` so the gateway applies its retry/cooldown logic. The
/// bespoke parser used to pass it through as a normal delta.
#[test]
fn stream_inline_error_surfaces_as_upstream() {
    let p = WatsonXProvider;
    let raw = r#"data: {"error":{"code":429,"message":"slow down"}}"#;
    let err = p.transform_stream_chunk(raw).expect_err("must surface");
    assert!(matches!(
        err,
        oxllm_core::Error::Upstream { status: 429, message }
            if message == "slow down"
    ));
}

/// `[DONE]` without the canonical space (`data:[DONE]`) is also terminal.
#[test]
fn stream_done_without_space_returns_none() {
    let p = WatsonXProvider;
    assert!(
        p.transform_stream_chunk("data:[DONE]")
            .expect("ok")
            .is_none()
    );
}

/// A wildcard deployment (`watsonx/*`) must take the model id from the request
/// body (prefix-stripped) rather than forwarding the literal `*` upstream,
/// mirroring `OpenAiCompatProvider::resolve_model`.
#[test]
fn wildcard_deployment_uses_request_model() {
    let p = WatsonXProvider;
    let mut dep = deployment_with_project_id("proj-1");
    dep.oxllm_params.model = "watsonx/*".to_string();
    let req = json!({
        "model": "watsonx/ibm/granite-3-8b-instruct",
        "messages": [{"role": "user", "content": "Hi"}],
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(out["model_id"], "ibm/granite-3-8b-instruct");
    assert!(out.get("model").is_none());
    assert_eq!(out["project_id"], "proj-1");
}

/// A wildcard deployment whose request carries no `model` falls back to the
/// wildcard literal — matching the shared compat layer's behaviour.
#[test]
fn wildcard_deployment_without_request_model_falls_back_to_literal() {
    let p = WatsonXProvider;
    let mut dep = deployment_with_project_id("proj-1");
    dep.oxllm_params.model = "watsonx/*".to_string();
    let req = json!({
        "messages": [{"role": "user", "content": "Hi"}],
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(out["model_id"], "*");
}

/// A `<provider>/*`-shaped wildcard (not just bare `*`) is also recognised.
#[test]
fn wildcard_deployment_with_provider_segment_is_resolved() {
    let p = WatsonXProvider;
    let mut dep = deployment_with_project_id("proj-1");
    dep.oxllm_params.model = "watsonx/granite/*".to_string();
    let req = json!({
        "model": "watsonx/granite/granite-3-8b",
        "messages": [{"role": "user", "content": "Hi"}],
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(out["model_id"], "granite/granite-3-8b");
}

#[test]
fn provider_id_and_supports() {
    let p = WatsonXProvider;
    assert_eq!(p.id(), "watsonx");
    assert!(p.supports(Endpoint::ChatCompletions));
    assert!(!p.supports(Endpoint::Embeddings));
    assert!(!p.supports(Endpoint::Rerank));
    assert!(p.chat().is_some());
    assert!(p.embeddings().is_none());
}

#[test]
fn supported_params_published() {
    let p = WatsonXProvider;
    let params = ChatProvider::supported_params(&p).expect("should publish params");
    assert!(params.contains(&"temperature"));
    assert!(params.contains(&"tools"));
    assert!(params.contains(&"max_tokens"));
    assert!(params.contains(&"reasoning_effort"));
}

#[test]
fn strip_watsonx_prefix_handles_variants() {
    assert_eq!(
        strip_watsonx_prefix("watsonx/my-test-model"),
        "my-test-model"
    );
    assert_eq!(
        strip_watsonx_prefix("watsonx_text/my-test-model"),
        "my-test-model"
    );
    assert_eq!(strip_watsonx_prefix("my-test-model"), "my-test-model");
}
