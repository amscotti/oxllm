use super::*;
use oxllm_core::Endpoint;
use oxllm_core::deployment::{Deployment, OxllmParams};
use oxllm_core::provider::{
    ChatProvider, EmbeddingProvider, Provider, RerankProvider, ResolvedCredentials,
};
use serde_json::json;

fn deployment() -> Deployment {
    Deployment {
        model_name: "qwen-plus".to_string(),
        oxllm_params: OxllmParams {
            model: "dashscope/qwen-plus".to_string(),
            api_key: Some("dashscope-key".to_string()),
            ..Default::default()
        },
        ..Default::default()
    }
}

#[test]
fn endpoint_url_uses_default_dashscope_base() {
    let p = DashScopeProvider;
    assert_eq!(
        p.endpoint_url(&deployment()),
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    );
}

#[test]
fn embedding_url_uses_default_dashscope_base() {
    let p = DashScopeProvider;
    assert_eq!(
        p.embedding_url(&deployment()),
        "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
    );
}

#[test]
fn rerank_url_uses_compatible_api_default() {
    // DashScope serves rerank under `compatible-api/v1`, not the chat/embed
    // `compatible-mode/v1` route. A default-configured deployment (no
    // `api_base`) must target `compatible-api`.
    let p = DashScopeProvider;
    assert_eq!(
        p.rerank_url(&deployment()),
        "https://dashscope.aliyuncs.com/compatible-api/v1/reranks"
    );
}

#[test]
fn endpoint_url_respects_custom_base() {
    let p = DashScopeProvider;
    let mut dep = deployment();
    dep.oxllm_params.api_base =
        Some("https://dashscope-intl.aliyuncs.com/compatible-mode/v1/".to_string());
    assert_eq!(
        p.endpoint_url(&dep),
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions"
    );
}

#[test]
fn apply_auth_uses_bearer_token() {
    let p = DashScopeProvider;
    let creds = ResolvedCredentials::from_key("dashscope-secret");
    let mut headers = std::collections::HashMap::new();
    ChatProvider::apply_auth(&p, &deployment(), &creds, &mut headers);
    assert_eq!(
        headers.get("authorization").map(String::as_str),
        Some("Bearer dashscope-secret")
    );
}

#[test]
fn request_strips_dashscope_prefix_from_model() {
    let p = DashScopeProvider;
    let req = json!({
        "model": "anything-in-request",
        "messages": [{"role": "user", "content": "Hi"}],
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["model"], "qwen-plus");
    assert_eq!(out["messages"][0]["content"], "Hi");
}

#[test]
fn request_preserves_stream_flag() {
    let p = DashScopeProvider;
    let req = json!({
        "model": "qwen-plus",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": true,
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["stream"], true);
}

#[test]
fn response_is_identity() {
    let p = DashScopeProvider;
    let resp = json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    });
    let out = ChatProvider::transform_response(&p, &resp, "qwen-plus").expect("ok");
    assert_eq!(out, resp);
}

#[test]
fn stream_chunk_parses_openai_format() {
    let p = DashScopeProvider;
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
    let p = DashScopeProvider;
    assert!(
        p.transform_stream_chunk("data: [DONE]")
            .expect("ok")
            .is_none()
    );
}

#[test]
fn provider_id_and_supports() {
    let p = DashScopeProvider;
    assert_eq!(p.id(), "dashscope");
    assert!(p.supports(Endpoint::ChatCompletions));
    assert!(p.supports(Endpoint::Embeddings));
    assert!(p.supports(Endpoint::Rerank));
    assert!(!p.supports(Endpoint::ImagesGenerations));
    assert!(p.chat().is_some());
    assert!(p.embeddings().is_some());
    assert!(p.rerank().is_some());
}

#[test]
fn strip_dashscope_prefix_handles_unknown() {
    assert_eq!(strip_dashscope_prefix("dashscope/qwen-plus"), "qwen-plus");
    assert_eq!(strip_dashscope_prefix("qwen-turbo"), "qwen-turbo");
    assert_eq!(strip_dashscope_prefix("dashscope/qwen-max"), "qwen-max");
}

#[test]
fn supported_params_published() {
    let p = DashScopeProvider;
    let params = p.supported_params().expect("should publish params");
    assert!(params.contains(&"temperature"));
    assert!(params.contains(&"tools"));
}

#[test]
fn wildcard_deployment_preserves_requested_model() {
    let p = DashScopeProvider;
    let mut dep = deployment();
    dep.oxllm_params.model = "dashscope/*".to_string();
    let req = json!({
        "model": "dashscope/qwen-turbo",
        "messages": [{"role": "user", "content": "Hi"}]
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(out["model"], "qwen-turbo");
}

#[test]
fn embedding_request_strips_prefix() {
    let p = DashScopeProvider;
    let mut dep = deployment();
    dep.oxllm_params.model = "dashscope/text-embedding-v3".to_string();
    let req = json!({
        "model": "text-embedding-v3",
        "input": "hello",
    });
    let out = EmbeddingProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(out["model"], "text-embedding-v3");
    assert_eq!(out["input"], "hello");
}

#[test]
fn rerank_request_forwards_fields() {
    let p = DashScopeProvider;
    let mut dep = deployment();
    dep.oxllm_params.model = "dashscope/gte-rerank".to_string();
    let req = json!({
        "model": "gte-rerank",
        "query": "what is rerank?",
        "documents": ["doc a", "doc b"],
        "top_n": 2,
        "return_documents": true,
    });
    let out = RerankProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(out["model"], "gte-rerank");
    assert_eq!(out["query"], "what is rerank?");
    assert_eq!(out["top_n"], 2);
    assert_eq!(out["return_documents"], true);
}

/// A wildcard rerank deployment (`dashscope/*`) must take the model from the
/// request body (prefix-stripped) rather than sending the literal `*`
/// upstream, mirroring `OpenAiCompatProvider::resolve_model`.
#[test]
fn rerank_request_with_wildcard_deployment_uses_request_model() {
    let p = DashScopeProvider;
    let mut dep = deployment();
    dep.oxllm_params.model = "dashscope/*".to_string();
    let req = json!({
        "model": "gte-rerank",
        "query": "q",
        "documents": ["a"],
    });
    let out = RerankProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(out["model"], "gte-rerank");
}

/// A wildcard rerank deployment whose request carries no `model` falls back
/// to the wildcard literal.
#[test]
fn rerank_request_with_wildcard_deployment_without_model_falls_back() {
    let p = DashScopeProvider;
    let mut dep = deployment();
    dep.oxllm_params.model = "dashscope/*".to_string();
    let req = json!({"query": "q", "documents": ["a"]});
    let out = RerankProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(out["model"], "*");
}

#[test]
fn rerank_response_passthrough_compatible_shape() {
    let p = DashScopeProvider;
    let resp = json!({
        "id": "r-1",
        "results": [
            {"index": 0, "relevance_score": 0.9},
            {"index": 1, "relevance_score": 0.1},
        ],
        "usage": {"total_tokens": 10},
    });
    let out = RerankProvider::transform_response(&p, &resp, "gte-rerank").expect("ok");
    assert_eq!(out, resp);
}

#[test]
fn rerank_response_unwraps_native_output_wrapper() {
    let p = DashScopeProvider;
    let resp = json!({
        "request_id": "req-1",
        "output": {
            "results": [
                {"index": 0, "relevance_score": 0.95, "document": {"text": "a"}},
            ]
        },
        "usage": {"total_tokens": 12},
    });
    let out = RerankProvider::transform_response(&p, &resp, "gte-rerank").expect("ok");
    assert_eq!(out["id"], "req-1");
    assert_eq!(out["results"][0]["index"], 0);
    assert_eq!(out["usage"]["total_tokens"], 12);
}

#[test]
fn rerank_url_passes_through_full_endpoint() {
    let p = DashScopeProvider;
    let mut dep = deployment();
    dep.oxllm_params.api_base = Some("https://custom.example.com/v1/reranks".to_string());
    assert_eq!(p.rerank_url(&dep), "https://custom.example.com/v1/reranks");
}

#[test]
fn rerank_url_custom_base_appends_reranks() {
    // A custom `api_base` (no `/reranks` suffix) is honored verbatim with
    // `/reranks` appended — distinct default does not override an explicit base.
    let p = DashScopeProvider;
    let mut dep = deployment();
    dep.oxllm_params.api_base = Some("https://custom.example.com/v1".to_string());
    assert_eq!(p.rerank_url(&dep), "https://custom.example.com/v1/reranks");
}

#[test]
fn rerank_url_rewrites_configured_chat_base_for_each_region_and_proxy() {
    let p = DashScopeProvider;
    for root in [
        "https://dashscope.aliyuncs.com",
        "https://dashscope-intl.aliyuncs.com",
        "https://dashscope-us.aliyuncs.com",
        "https://proxy.example.com/alibaba",
    ] {
        let mut dep = deployment();
        dep.oxllm_params.api_base = Some(format!("{root}/compatible-mode/v1/"));
        assert_eq!(
            p.rerank_url(&dep),
            format!("{root}/compatible-api/v1/reranks")
        );
        assert_eq!(
            p.endpoint_url(&dep),
            format!("{root}/compatible-mode/v1/chat/completions")
        );
    }
}

#[test]
fn rerank_url_preserves_explicit_route_and_compatible_api_base() {
    for base in [
        "https://proxy.example.com/compatible-api/v1",
        "https://proxy.example.com/compatible-api/v1/reranks/",
        "https://proxy.example.com/compatible-mode/v1/reranks",
        "https://proxy.example.com/custom/rerank",
    ] {
        let mut dep = deployment();
        dep.oxllm_params.api_base = Some(base.into());
        let trimmed = base.trim_end_matches('/');
        let expected = if trimmed.ends_with("/v1") {
            format!("{trimmed}/reranks")
        } else {
            trimmed.into()
        };
        assert_eq!(DashScopeProvider.rerank_url(&dep), expected);
    }
}

#[test]
fn rerank_response_error_envelope_surfaces_upstream_error() {
    // A 200-status body carrying a DashScope error envelope must surface as
    // an error (not be forwarded as a fake-empty rerank result).
    let p = DashScopeProvider;
    let resp = json!({
        "code": "InvalidApiKey",
        "message": "Invalid api-key",
        "request_id": "req-2",
    });
    let err =
        RerankProvider::transform_response(&p, &resp, "gte-rerank").expect_err("should error");
    match err {
        oxllm_core::Error::Upstream { message, .. } => {
            assert!(message.contains("Invalid api-key"), "message: {message}");
            assert!(message.contains("InvalidApiKey"), "message: {message}");
        }
        other => panic!("expected Upstream, got {other:?}"),
    }
}

/// DashScope wraps errors inside HTTP 200 responses. Known error codes must
/// be mapped to HTTP statuses the host's retry/cooldown logic can act on
/// (e.g. `InvalidApiKey` → 401, `Throttling.*` → 429), rather than surfacing
/// as `status: 200` which bypasses retry entirely.
#[test]
fn rerank_response_error_envelope_maps_known_codes_to_http_statuses() {
    let p = DashScopeProvider;
    for (code, expected_status) in [
        ("InvalidApiKey", 401u16),
        ("Throttling.RateQuota", 429),
        ("Throttling.AllocationQuota", 429),
        ("AccessForbidden", 403),
        ("ModelNotFound", 404),
        ("BadRequest.MissingParameter", 400),
        ("DataInspection", 400),
        ("InternalError.Timeout", 500),
        ("SomeUnknownCode", 500),
    ] {
        let resp = json!({"code": code, "message": "err"});
        let err =
            RerankProvider::transform_response(&p, &resp, "gte-rerank").expect_err("should error");
        match err {
            oxllm_core::Error::Upstream { status, .. } => {
                assert_eq!(
                    status, expected_status,
                    "code {code} should map to {expected_status}, got {status}"
                );
            }
            other => panic!("expected Upstream, got {other:?}"),
        }
    }
}

#[test]
fn rerank_response_missing_results_surfaces_transform_error() {
    let p = DashScopeProvider;
    let resp = json!({"foo": "bar"});
    let err =
        RerankProvider::transform_response(&p, &resp, "gte-rerank").expect_err("should error");
    assert!(matches!(err, oxllm_core::Error::Transform(_)));
}
