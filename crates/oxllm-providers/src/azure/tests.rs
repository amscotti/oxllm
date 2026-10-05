use super::*;

#[test]
fn route_suffix_matching_is_safe_for_unicode_and_exact_prefixes() {
    for path in ["/éééééé", "/路径/模型", "/日本語openai"] {
        assert_eq!(strip_route_suffix(path, "/openai"), path);
    }
    assert_eq!(strip_route_suffix("/openai", "/openai"), "");
    assert_eq!(strip_route_suffix("/路径/openai", "/openai"), "/路径");
}
use oxllm_core::Endpoint;
use oxllm_core::deployment::{Deployment, OxllmParams};
use oxllm_core::provider::{ChatProvider, EmbeddingProvider, Provider, ResolvedCredentials};
use serde_json::json;

fn deployment() -> Deployment {
    Deployment {
        model_name: "my-gpt4-deployment".to_string(),
        oxllm_params: OxllmParams {
            model: "azure/my-gpt4-deployment".to_string(),
            api_key: Some("azure-key".to_string()),
            api_base: Some("https://myresource.openai.azure.com".to_string()),
            api_version: Some("2024-02-15-preview".to_string()),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn deployment_no_version() -> Deployment {
    let mut dep = deployment();
    dep.oxllm_params.api_version = None;
    dep
}

#[test]
fn endpoint_url_uses_deployment_path() {
    let p = AzureProvider;
    assert_eq!(
        p.endpoint_url(&deployment()),
        "https://myresource.openai.azure.com/openai/deployments/my-gpt4-deployment/chat/completions?api-version=2024-02-15-preview"
    );
}

#[test]
fn embedding_url_uses_deployment_path() {
    let p = AzureProvider;
    assert_eq!(
        p.embedding_url(&deployment()),
        "https://myresource.openai.azure.com/openai/deployments/my-gpt4-deployment/embeddings?api-version=2024-02-15-preview"
    );
}

#[test]
fn endpoint_url_uses_default_api_version_when_missing() {
    let p = AzureProvider;
    assert_eq!(
        p.endpoint_url(&deployment_no_version()),
        "https://myresource.openai.azure.com/openai/deployments/my-gpt4-deployment/chat/completions?api-version=2025-02-01-preview"
    );
}

#[test]
fn endpoint_url_respects_custom_api_version() {
    let p = AzureProvider;
    let mut dep = deployment();
    dep.oxllm_params.api_version = Some("2024-10-21".to_string());
    assert_eq!(
        p.endpoint_url(&dep),
        "https://myresource.openai.azure.com/openai/deployments/my-gpt4-deployment/chat/completions?api-version=2024-10-21"
    );
}

#[test]
fn endpoint_url_trims_trailing_slash() {
    let p = AzureProvider;
    let mut dep = deployment();
    dep.oxllm_params.api_base = Some("https://myresource.openai.azure.com/".to_string());
    assert_eq!(
        p.endpoint_url(&dep),
        "https://myresource.openai.azure.com/openai/deployments/my-gpt4-deployment/chat/completions?api-version=2024-02-15-preview"
    );
}

#[test]
fn endpoint_url_handles_no_azure_prefix() {
    let p = AzureProvider;
    let mut dep = deployment();
    dep.oxllm_params.model = "my-deployment".to_string();
    assert_eq!(
        p.endpoint_url(&dep),
        "https://myresource.openai.azure.com/openai/deployments/my-deployment/chat/completions?api-version=2024-02-15-preview"
    );
}

#[test]
fn endpoint_url_returns_sentinel_when_api_base_missing() {
    // Missing `api_base` must NOT panic (which would kill the request task).
    // The provider returns a sentinel URL on the reserved `.invalid` TLD so
    // the upstream HTTP call fails fast with a clear DNS-resolution error.
    let p = AzureProvider;
    let mut dep = deployment();
    dep.oxllm_params.api_base = None;
    let url = p.endpoint_url(&dep);
    assert!(
        url.starts_with(AZURE_API_BASE_NOT_CONFIGURED),
        "missing api_base must yield the sentinel URL, got: {url}"
    );
    assert!(
        url.contains("/openai/deployments/my-gpt4-deployment/chat/completions"),
        "sentinel URL must still carry the deployment path: {url}"
    );
}

#[test]
fn embedding_url_returns_sentinel_when_api_base_missing() {
    let p = AzureProvider;
    let mut dep = deployment();
    dep.oxllm_params.api_base = None;
    let url = p.embedding_url(&dep);
    assert!(
        url.starts_with(AZURE_API_BASE_NOT_CONFIGURED),
        "missing api_base must yield the sentinel URL, got: {url}"
    );
    assert!(
        url.contains("/openai/deployments/my-gpt4-deployment/embeddings"),
        "sentinel URL must still carry the deployment path: {url}"
    );
}

#[test]
fn sentinel_url_uses_reserved_invalid_tld() {
    // RFC 2606 reserves `.invalid`: DNS resolution is guaranteed to fail
    // quickly, so requests to the sentinel never reach a real host.
    assert!(AZURE_API_BASE_NOT_CONFIGURED.ends_with(".invalid"));
    assert!(AZURE_API_BASE_NOT_CONFIGURED.starts_with("https://"));
}

#[test]
fn default_api_version_is_modern_2025_release() {
    // The old default `2024-02-15-preview` predates `max_completion_tokens`,
    // `stream_options`, and o-series models. The `2024-10-21` GA release added
    // those, and `2025-02-01-preview` is LiteLLM's current
    // `AZURE_DEFAULT_API_VERSION`, carrying reasoning-effort / service-tier
    // fields used by GPT-5 and later. The default must be at least
    // `2025-02-01-preview` so these features work out of the box.
    let parsed: Vec<u32> = DEFAULT_AZURE_API_VERSION
        .split('-')
        .filter_map(|s| s.parse().ok())
        .collect();
    assert_eq!(parsed.len(), 3, "default version should be YYYY-MM-DD");
    let [year, month, day] = [parsed[0], parsed[1], parsed[2]];
    // Strictly newer than 2024-10-21.
    assert!(
        (year, month, day) > (2024, 10, 21),
        "default API version {DEFAULT_AZURE_API_VERSION} must be newer than 2024-10-21"
    );
    // Pinned to LiteLLM's AZURE_DEFAULT_API_VERSION.
    assert_eq!(
        DEFAULT_AZURE_API_VERSION, "2025-02-01-preview",
        "default must track LiteLLM's AZURE_DEFAULT_API_VERSION (2025-02-01-preview)"
    );
}

#[test]
fn deployment_name_reads_extra_azure_deployment_when_no_prefix() {
    let p = AzureProvider;
    let mut dep = deployment();
    dep.oxllm_params.model = "gpt-4o".to_string();
    dep.oxllm_params
        .extra
        .insert("azure_deployment".to_string(), json!("extra-deployment"));
    let url = p.endpoint_url(&dep);
    assert!(url.contains("/openai/deployments/extra-deployment/chat/completions"));
}

#[test]
fn deployment_name_azure_prefix_takes_precedence_over_extra() {
    let p = AzureProvider;
    let mut dep = deployment();
    dep.oxllm_params.model = "azure/prefix-deployment".to_string();
    dep.oxllm_params
        .extra
        .insert("azure_deployment".to_string(), json!("extra-deployment"));
    let url = p.endpoint_url(&dep);
    assert!(url.contains("/openai/deployments/prefix-deployment/chat/completions"));
}

#[test]
fn apply_auth_uses_api_key_header_for_azure_creds() {
    let p = AzureProvider;
    let creds = ResolvedCredentials::Azure {
        api_key: "azure-secret".to_string(),
    };
    let mut headers = std::collections::HashMap::new();
    ChatProvider::apply_auth(&p, &deployment(), &creds, &mut headers);
    assert_eq!(
        headers.get("api-key").map(String::as_str),
        Some("azure-secret")
    );
    assert!(
        !headers.contains_key("authorization"),
        "Azure must NOT use Authorization header"
    );
}

#[test]
fn apply_auth_falls_back_to_bearer_from_other_creds() {
    let p = AzureProvider;
    let creds = ResolvedCredentials::from_key("azure-fallback");
    let mut headers = std::collections::HashMap::new();
    ChatProvider::apply_auth(&p, &deployment(), &creds, &mut headers);
    assert_eq!(
        headers.get("api-key").map(String::as_str),
        Some("azure-fallback")
    );
}

#[test]
fn request_is_identity() {
    let p = AzureProvider;
    let req = json!({
    "model": "my-gpt4-deployment",
    "messages": [{"role": "user", "content": "Hi"}],
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out, req);
}

#[test]
fn response_is_identity() {
    let p = AzureProvider;
    let resp = json!({
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    });
    let out = ChatProvider::transform_response(&p, &resp, "my-gpt4-deployment").expect("ok");
    assert_eq!(out, resp);
}

#[test]
fn stream_chunk_parses_openai_format() {
    let p = AzureProvider;
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
    let p = AzureProvider;
    assert!(
        p.transform_stream_chunk("data: [DONE]")
            .expect("ok")
            .is_none()
    );
}

#[test]
fn provider_id_and_supports() {
    let p = AzureProvider;
    assert_eq!(p.id(), "azure");
    assert!(p.supports(Endpoint::ChatCompletions));
    assert!(p.supports(Endpoint::Embeddings));
    assert!(p.supports(Endpoint::ImagesGenerations));
    assert!(p.supports(Endpoint::AudioTranscriptions));
    assert!(p.supports(Endpoint::AudioSpeech));
    assert!(!p.supports(Endpoint::Rerank));
    // Azure has no /moderations endpoint in any official spec; moderation is
    // a separate service (Azure AI Content Safety).
    assert!(!p.supports(Endpoint::Moderations));
    assert!(p.chat().is_some());
    assert!(p.embeddings().is_some());
    assert!(p.rerank().is_none());
}

#[test]
fn supported_params_is_none() {
    let p = AzureProvider;
    assert!(ChatProvider::supported_params(&p).is_none());
}

const CANONICAL_GOLDEN_REQUEST: &str =
    include_str!("../openai_compat/goldens/chat_request_001.json");
const CANONICAL_GOLDEN_RESPONSE: &str =
    include_str!("../openai_compat/goldens/chat_response_001.json");

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
    let provider = AzureProvider;
    let deployment = deployment();
    let model = "my-gpt4-deployment";

    let mut request: Value = serde_json::from_str(CANONICAL_GOLDEN_REQUEST)
        .expect("canonical golden request should be valid JSON");
    request["model"] = json!(model);
    let out = ChatProvider::transform_request(&provider, &request, &deployment)
        .expect("golden request transform should succeed");
    assert_semantically_equal(&out, &request);

    let mut response: Value = serde_json::from_str(CANONICAL_GOLDEN_RESPONSE)
        .expect("canonical golden response should be valid JSON");
    response["model"] = json!(model);
    let out = ChatProvider::transform_response(&provider, &response, model)
        .expect("golden response transform should succeed");
    assert_semantically_equal(&out, &response);
}

// --- Ported twin: request is identity (model stays as-is) ---
//
// Azure uses the deployment name on the URL path, not the body's `model`
// field, for routing. The request model is forwarded unchanged.
// Source `twin_azure_provider_success` adapted.

#[test]
fn transform_request_preserves_request_model() {
    let p = AzureProvider;
    let mut dep = deployment();
    dep.oxllm_params.model = "azure/my-custom-deployment".to_string();
    let req = json!({
        "model": "caller-supplied-ignored",
        "messages": [{"role": "user", "content": "Hi"}],
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    // Azure does NOT rewrite the body's model — it uses the URL path for routing.
    assert_eq!(out["model"], "caller-supplied-ignored");
    assert_eq!(out["messages"][0]["content"], "Hi");
}

// --- Ported from LiteLLM test_azure_openai.py ---
//
// These tests verify that the Azure OpenAI provider correctly passes through
// all API parameters (identity transform for request body), handles streaming
// chunks with modern fields, and constructs correct endpoint URLs for image
// generation and deployment-path scenarios.

// Litellm test: test_azure_gpt_4o_with_tool_call_and_response_format
// Azure identity transform preserves all openai params including tools + response_format
#[test]
fn request_preserves_all_openai_params() {
    let p = AzureProvider;
    let req = json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hello"}],
        "prediction": {"type": "content", "content": "some code"},
        "service_tier": "priority",
        "reasoning_effort": "medium",
        "tools": [{"type": "function", "function": {"name": "test", "parameters": {}}}],
        "response_format": {"type": "json_object"},
        "max_completion_tokens": 500,
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out, req);
}

// Litellm test: test_openai_chat_completion_streaming_handler_reasoning_content
// (Azure uses the same SSE parser as OpenAI)
#[test]
fn stream_chunk_with_reasoning_content() {
    let p = AzureProvider;
    let chunk = concat!(
        "data: {\"id\":\"x\",\"choices\":[{\"index\":0,",
        "\"delta\":{\"reasoning_content\":\"thinking step by step\"}}]}\n\n"
    );
    let out = p
        .transform_stream_chunk(chunk)
        .expect("ok")
        .expect("should produce value");
    assert_eq!(
        out["choices"][0]["delta"]["reasoning_content"],
        "thinking step by step"
    );
}

// Litellm test: test_streaming_tool_calls_with_n_greater_than_1 (chunk parsing)
#[test]
fn stream_chunk_with_tool_calls() {
    let p = AzureProvider;
    let chunk = concat!(
        "data: {\"id\":\"x\",\"choices\":[{\"index\":0,",
        "\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",",
        "\"function\":{\"name\":\"get_weather\",\"arguments\":\"{}\"}}]}}]}\n\n"
    );
    let out = p
        .transform_stream_chunk(chunk)
        .expect("ok")
        .expect("should produce value");
    assert_eq!(
        out["choices"][0]["delta"]["tool_calls"][0]["function"]["name"],
        "get_weather"
    );
    assert_eq!(out["choices"][0]["delta"]["tool_calls"][0]["index"], 0);
}

// Litellm test: test_process_azure_endpoint_url (deployment-path URL pattern)
// Azure routes the model via the deployment name in the URL path. This test
// validates the chat-completions URL construction when the model carries the
// `azure/` prefix.
#[test]
fn endpoint_url_with_model_that_has_azure_prefix() {
    let p = AzureProvider;
    let mut dep = deployment();
    dep.oxllm_params.model = "azure/dall-e-3-test".to_string();
    dep.oxllm_params.api_version = Some("2023-12-01-preview".to_string());
    let url = p.endpoint_url(&dep);
    assert_eq!(
        url,
        "https://myresource.openai.azure.com/openai/deployments/dall-e-3-test/chat/completions?api-version=2023-12-01-preview"
    );
}

// Litellm test: test_azure_extra_headers (auth header passthrough)
// Azure applies auth via `api-key` header. This test verifies the header
// is set correctly and that the Authorization header is NOT used.
#[test]
fn apply_auth_prefers_api_key_over_bearer() {
    let p = AzureProvider;
    let creds = ResolvedCredentials::Azure {
        api_key: "my-azure-key".to_string(),
    };
    let mut headers = std::collections::HashMap::new();
    ChatProvider::apply_auth(&p, &deployment(), &creds, &mut headers);
    let key = headers.get("api-key").map(String::as_str);
    assert_eq!(key, Some("my-azure-key"));
    assert!(
        !headers.contains_key("authorization"),
        "Azure must NOT set Authorization when api-key is available"
    );
}

// --- Coverage for the silent-auth-fallback (MEDIUM) ---
//
// When a credential with no bearer (e.g. `AwsSigV4`) is handed to `apply_auth`,
// the function must NOT insert any auth header — the request is left
// unauthenticated and a `tracing::warn!` is emitted (the warn is verified by
// code inspection; capturing it would require a tracing-capture dev-dependency,
// deliberately avoided to keep the dep footprint minimal).
#[test]
fn apply_auth_inserts_nothing_when_credential_has_no_bearer() {
    let p = AzureProvider;
    let creds = ResolvedCredentials::AwsSigV4 {
        access_key: "AKIAEXAMPLE".to_string(),
        secret_key: "secret".to_string(),
        session_token: None,
        region: "us-east-1".to_string(),
    };
    let mut headers = std::collections::HashMap::new();
    ChatProvider::apply_auth(&p, &deployment(), &creds, &mut headers);
    assert!(
        !headers.contains_key("api-key"),
        "AwsSigV4 must not produce an api-key header: got {headers:?}"
    );
    assert!(
        !headers.contains_key("authorization"),
        "AwsSigV4 must not produce an authorization header: got {headers:?}"
    );
    assert!(
        headers.is_empty(),
        "no auth headers should be set: {headers:?}"
    );
}

// --- Coverage for the percent-encoding gap (LOW) ---

// A deployment name derived from `"azure/foo/bar"` contains a `/`. It must be
// percent-encoded as `foo%2Fbar` in the path so it is not silently split into
// two path segments that route the request to the wrong deployment.
#[test]
fn endpoint_url_percent_encodes_deployment_name_with_slash() {
    let p = AzureProvider;
    let mut dep = deployment();
    dep.oxllm_params.model = "azure/foo/bar".to_string();
    let url = p.endpoint_url(&dep);
    assert!(
        url.contains("/openai/deployments/foo%2Fbar/chat/completions"),
        "deployment name with `/` must be percent-encoded, got: {url}"
    );
    assert!(
        !url.contains("/openai/deployments/foo/bar/"),
        "deployment name must NOT be injected as raw path segments, got: {url}"
    );
}

// An `api_version` containing `&` or `#` must be encoded so it cannot inject
// extra query parameters or a fragment into the upstream URL.
#[test]
fn endpoint_url_percent_encodes_api_version_with_ampersand() {
    let p = AzureProvider;
    let mut dep = deployment();
    dep.oxllm_params.api_version = Some("2024&evil=1#frag".to_string());
    let url = p.endpoint_url(&dep);
    assert!(
        url.contains("api-version=2024%26evil%3D1%23frag"),
        "api_version with `&`/`#`/`=` must be percent-encoded, got: {url}"
    );
    assert!(
        !url.contains("&evil=") && !url.contains("#frag"),
        "api_version must NOT inject raw query params / fragment, got: {url}"
    );
}

// The embeddings URL builder must percent-encode the same way as chat.
#[test]
fn embedding_url_percent_encodes_deployment_name_with_slash() {
    let p = AzureProvider;
    let mut dep = deployment();
    dep.oxllm_params.model = "azure/foo/bar".to_string();
    let url = p.embedding_url(&dep);
    assert!(
        url.contains("/openai/deployments/foo%2Fbar/embeddings"),
        "embedding deployment name with `/` must be percent-encoded, got: {url}"
    );
}

// A plain-alphanumeric deployment name and a standard `YYYY-MM-DD-preview`
// api_version must pass through UNencoded so existing happy-path behavior
// (and the URL format Azure expects) is preserved.
#[test]
fn endpoint_url_does_not_over_encode_normal_values() {
    let p = AzureProvider;
    let url = p.endpoint_url(&deployment());
    assert_eq!(
        url,
        "https://myresource.openai.azure.com/openai/deployments/my-gpt4-deployment/chat/completions?api-version=2024-02-15-preview"
    );
}

// --- Coverage for the http:// api_base scheme (LOW) ---
//
// An `http://` api_base is still forwarded (required by the httpmock-based
// integration tests), but `api_base()` emits a `tracing::warn!` about the
// cleartext api-key. The warn is verified by code inspection; this test pins
// the behavior that the URL is still built and the scheme is not clobbered.
#[test]
fn api_base_http_scheme_still_builds_url() {
    let p = AzureProvider;
    let mut dep = deployment();
    dep.oxllm_params.api_base = Some("http://local-mock.upstream".to_string());
    let url = p.endpoint_url(&dep);
    assert!(
        url.starts_with("http://local-mock.upstream/openai/deployments/"),
        "http:// api_base must still be used for the URL, got: {url}"
    );
    assert!(url.contains("/openai/deployments/my-gpt4-deployment/chat/completions"));
}

// --- Coverage for the empty-deployment-name gap (MEDIUM) ---
//
// `model = "azure/"` (trailing slash, empty rest) must NOT produce an empty
// deployment name and a `/openai/deployments//chat/completions` URL (which
// Azure answers with a confusing 404). It should fall through to
// `extra.azure_deployment` when present, and otherwise at least emit a
// non-empty path segment.
#[test]
fn deployment_name_with_empty_azure_prefix_uses_extra() {
    let p = AzureProvider;
    let mut dep = deployment();
    dep.oxllm_params.model = "azure/".to_string();
    dep.oxllm_params
        .extra
        .insert("azure_deployment".to_string(), json!("fallback-deployment"));
    let url = p.endpoint_url(&dep);
    assert!(
        url.contains("/openai/deployments/fallback-deployment/chat/completions"),
        "empty `azure/` prefix should fall through to extra.azure_deployment, got: {url}"
    );
    assert!(
        !url.contains("/deployments//chat/completions"),
        "empty deployment name must not produce a double slash, got: {url}"
    );
}

#[test]
fn deployment_name_with_empty_azure_prefix_and_no_extra_is_nonempty() {
    let p = AzureProvider;
    let mut dep = deployment();
    dep.oxllm_params.model = "azure/".to_string();
    dep.oxllm_params.extra.clear();
    let url = p.endpoint_url(&dep);
    assert!(
        !url.contains("/deployments//chat/completions"),
        "deployment name must not be empty in the URL, got: {url}"
    );
    // The literal `azure/` prefix must not be percent-encoded into the path
    // (a guaranteed upstream 404); the sentinel `.invalid` URL is returned
    // instead so the request fails fast with a clear DNS error.
    assert!(
        url.starts_with(crate::azure::AZURE_DEPLOYMENT_NAME_NOT_CONFIGURED),
        "bare `azure/` prefix with no azure_deployment extra must hit the sentinel URL, got: {url}"
    );
}

// --- Coverage for the GcpBearer mis-route gap (MEDIUM) ---
//
// A `GcpBearer` token must NOT be silently inserted into the `api-key`
// header — Azure rejects AAD-style bearer tokens there with a 401. A
// dedicated `AzureAd` variant exists for Entra ID tokens (routed to
// `Authorization: Bearer`); a GCP bearer is a different cloud's credential
// and must stay unauthenticated (matching `AwsSigV4` / `PreSignedHeaders`).
#[test]
fn apply_auth_does_not_misroute_gcp_bearer_into_api_key() {
    let p = AzureProvider;
    let creds = ResolvedCredentials::GcpBearer {
        token: "ya29.gcp-token".to_string(),
    };
    let mut headers = std::collections::HashMap::new();
    ChatProvider::apply_auth(&p, &deployment(), &creds, &mut headers);
    assert!(
        !headers.contains_key("api-key"),
        "GcpBearer must not be routed into the `api-key` header: {headers:?}"
    );
    assert!(
        !headers.contains_key("authorization"),
        "GcpBearer must not be routed into the `Authorization` header: {headers:?}"
    );
    assert!(
        headers.is_empty(),
        "no auth headers should be set for GcpBearer: {headers:?}"
    );
}

#[test]
fn apply_auth_falls_back_to_api_key_for_generic_apikey() {
    // A generic `ApiKey` is a plausible Azure shared key when the resolver
    // did not tag it as `Azure`; the fallback to `api-key` is intentional.
    let p = AzureProvider;
    let creds = ResolvedCredentials::from_key("generic-azure-key");
    let mut headers = std::collections::HashMap::new();
    ChatProvider::apply_auth(&p, &deployment(), &creds, &mut headers);
    assert_eq!(
        headers.get("api-key").map(String::as_str),
        Some("generic-azure-key")
    );
}

// --- Coverage for the Azure message-normalization gap (LOW) ---
//
// Azure is stricter than OpenAI: a bare-string `image_url` (must be an object
// `{"url": ...}`) and a null `function_call.arguments` (must be a string) are
// both rejected by Azure but accepted by OpenAI. LiteLLM's
// `convert_to_azure_openai_messages` fixes these up before sending.
#[test]
fn transform_request_normalizes_bare_string_image_url() {
    let p = AzureProvider;
    let req = json!({
        "model": "my-gpt4-deployment",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this?"},
                {"type": "image_url", "image_url": "https://example.com/cat.png"}
            ]
        }],
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(
        out["messages"][0]["content"][1]["image_url"]["url"], "https://example.com/cat.png",
        "bare-string image_url must be wrapped as {{\"url\": ...}}: {out}"
    );
    assert_eq!(
        out["messages"][0]["content"][1]["image_url"]["url"],
        req["messages"][0]["content"][1]["image_url"]
    );
    assert_eq!(
        out["messages"][0]["content"][0]["text"], "what is this?",
        "unrelated content parts must be untouched"
    );
}

#[test]
fn transform_request_normalizes_object_image_url_unchanged() {
    // The already-Azure-compatible object form must pass through verbatim.
    let p = AzureProvider;
    let req = json!({
        "model": "my-gpt4-deployment",
        "messages": [{
            "role": "user",
            "content": [{
                "type": "image_url",
                "image_url": {"url": "https://example.com/cat.png", "detail": "high"}
            }]
        }],
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out, req, "object image_url must be unchanged");
}

#[test]
fn transform_request_normalizes_null_function_call_arguments() {
    let p = AzureProvider;
    let req = json!({
        "model": "my-gpt4-deployment",
        "messages": [{
            "role": "assistant",
            "content": null,
            "function_call": {"name": "get_weather", "arguments": null}
        }],
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(
        out["messages"][0]["function_call"]["arguments"], "",
        "null function_call.arguments must be coerced to empty string: {out}"
    );
    assert_eq!(
        out["messages"][0]["function_call"]["name"], "get_weather",
        "function_call.name must be preserved"
    );
}

#[test]
fn transform_request_leaves_string_arguments_unchanged() {
    // A non-null `arguments` (already a string) must not be touched.
    let p = AzureProvider;
    let req = json!({
        "model": "my-gpt4-deployment",
        "messages": [{
            "role": "assistant",
            "function_call": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}
        }],
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out, req, "non-null arguments must be unchanged");
}

// --- Coverage for the missing-arguments coercion gap (LOW) ---
//
// LiteLLM's `_azure_tool_call_invoke_helper` coerces `arguments` when it is
// missing OR null (`get(...) is None`). oxllm previously only coerced the
// null case; a `function_call` with only `name` (no `arguments` key) was
// passed through unchanged and Azure may reject it with a 400.
#[test]
fn transform_request_normalizes_missing_function_call_arguments() {
    let p = AzureProvider;
    let req = json!({
        "model": "my-gpt4-deployment",
        "messages": [{
            "role": "assistant",
            "function_call": {"name": "get_weather"}
        }],
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(
        out["messages"][0]["function_call"]["arguments"], "",
        "missing function_call.arguments must be coerced to empty string: {out}"
    );
    assert_eq!(
        out["messages"][0]["function_call"]["name"], "get_weather",
        "function_call.name must be preserved"
    );
}

// --- Coverage for the Azure AD / Entra ID auth path (MEDIUM) ---
//
// An `AzureAd` token is a bearer token, NOT a shared key. Azure rejects it
// from the `api-key` header with a 401, so it must go on
// `Authorization: Bearer` — matching LiteLLM's `_base_validate_azure_environment`
// AAD fallback. This is the provider-side half of AAD support; the host-side
// resolver (Entra ID / managed identity token acquisition with refresh) is a
// separate component.
#[test]
fn apply_auth_uses_authorization_bearer_for_azure_ad_token() {
    let p = AzureProvider;
    let creds = ResolvedCredentials::AzureAd {
        token: "eyJ0eXAi.aad-token.example".to_string(),
    };
    let mut headers = std::collections::HashMap::new();
    ChatProvider::apply_auth(&p, &deployment(), &creds, &mut headers);
    assert_eq!(
        headers.get("authorization").map(String::as_str),
        Some("Bearer eyJ0eXAi.aad-token.example"),
        "AzureAd token must be sent as Authorization: Bearer: {headers:?}"
    );
    assert!(
        !headers.contains_key("api-key"),
        "AzureAd token must NOT be routed into the api-key header: {headers:?}"
    );
}

#[test]
fn apply_auth_embedding_uses_authorization_bearer_for_azure_ad_token() {
    let p = AzureProvider;
    let creds = ResolvedCredentials::AzureAd {
        token: "aad-token".to_string(),
    };
    let mut headers = std::collections::HashMap::new();
    EmbeddingProvider::apply_auth(&p, &deployment(), &creds, &mut headers);
    assert_eq!(
        headers.get("authorization").map(String::as_str),
        Some("Bearer aad-token"),
    );
}

/// Entra ID access tokens pasted into the key field must ride
/// `Authorization: Bearer` (Azure rejects a JWT on `api-key`), while a real
/// hex shared key keeps using `api-key`.
#[test]
fn apply_auth_routes_jwt_shaped_key_to_bearer_and_hex_key_to_api_key() {
    let p = AzureProvider;
    let jwt = "eyJhbGciOiJub25lIn0.eyJzdWIiOiJ4In0.sig";

    let mut headers = std::collections::HashMap::new();
    ChatProvider::apply_auth(
        &p,
        &deployment(),
        &ResolvedCredentials::Azure {
            api_key: jwt.to_string(),
        },
        &mut headers,
    );
    assert_eq!(
        headers.get("authorization").map(String::as_str),
        Some(format!("Bearer {jwt}").as_str()),
    );
    assert!(
        !headers.contains_key("api-key"),
        "JWT must not be sent as api-key: {headers:?}"
    );

    let hex = "0123456789abcdef0123456789abcdef";
    let mut headers = std::collections::HashMap::new();
    ChatProvider::apply_auth(
        &p,
        &deployment(),
        &ResolvedCredentials::Azure {
            api_key: hex.to_string(),
        },
        &mut headers,
    );
    assert_eq!(headers.get("api-key").map(String::as_str), Some(hex));
    assert!(!headers.contains_key("authorization"));
}

// --- Coverage for the Azure v1 API routing (LOW) ---
//
// LiteLLM's `_is_azure_v1_api_version` detects `api_version` values
// `preview` / `latest` / `v1` and routes to `{base}/openai/v1/{endpoint}`
// (model in body) instead of the deployment-path URL. oxllm previously
// always built the deployment-path URL.
#[test]
fn endpoint_url_uses_v1_route_for_preview_api_version() {
    let p = AzureProvider;
    let mut dep = deployment();
    dep.oxllm_params.api_version = Some("preview".to_string());
    assert_eq!(
        p.endpoint_url(&dep),
        "https://myresource.openai.azure.com/openai/v1/chat/completions?api-version=preview"
    );
    assert_eq!(
        p.embedding_url(&dep),
        "https://myresource.openai.azure.com/openai/v1/embeddings?api-version=preview"
    );
}

#[test]
fn endpoint_url_uses_v1_route_for_latest_and_v1_api_versions() {
    let p = AzureProvider;
    for v in ["latest", "v1"] {
        let mut dep = deployment();
        dep.oxllm_params.api_version = Some(v.to_string());
        let url = p.endpoint_url(&dep);
        assert!(
            url.contains("/openai/v1/chat/completions"),
            "api_version `{v}` must route to /openai/v1/, got: {url}"
        );
        assert!(
            !url.contains("/openai/deployments/"),
            "api_version `{v}` must NOT use the deployment path, got: {url}"
        );
    }
}

#[test]
fn endpoint_url_uses_deployment_path_for_dated_api_version() {
    // A dated api_version (the common case) must still use the deployment path.
    let p = AzureProvider;
    let mut dep = deployment();
    dep.oxllm_params.api_version = Some("2024-10-21".to_string());
    let url = p.endpoint_url(&dep);
    assert!(
        url.contains("/openai/deployments/my-gpt4-deployment/chat/completions"),
        "dated api_version must use the deployment path, got: {url}"
    );
}

/// Azure documents the v1 base URL as
/// `https://{resource}.openai.azure.com/openai/v1/` (and the unified
/// `https://{resource}.services.ai.azure.com/openai/v1/` host). Operators
/// paste that full base into `api_base`; the provider must not append the
/// route a second time (`/openai/v1/openai/v1/...`), which is a known
/// LiteLLM bug.
#[test]
fn endpoint_url_does_not_double_v1_base_prefix() {
    let p = AzureProvider;
    for (base, expected) in [
        (
            "https://myresource.openai.azure.com/openai/v1",
            "https://myresource.openai.azure.com/openai/v1/chat/completions?api-version=v1",
        ),
        (
            "https://myresource.openai.azure.com/openai/v1/",
            "https://myresource.openai.azure.com/openai/v1/chat/completions?api-version=v1",
        ),
        (
            "https://myresource.services.ai.azure.com/openai/v1",
            "https://myresource.services.ai.azure.com/openai/v1/chat/completions?api-version=v1",
        ),
    ] {
        let mut dep = deployment();
        dep.oxllm_params.api_base = Some(base.to_string());
        dep.oxllm_params.api_version = Some("v1".to_string());
        assert_eq!(p.endpoint_url(&dep), expected, "base={base}");
    }
}

/// A classic client base that already includes `/openai` must not double it
/// either.
#[test]
fn endpoint_url_does_not_double_classic_openai_prefix() {
    let p = AzureProvider;
    let mut dep = deployment();
    dep.oxllm_params.api_base = Some("https://myresource.openai.azure.com/openai".to_string());
    dep.oxllm_params.api_version = Some("2024-10-21".to_string());
    assert_eq!(
        p.endpoint_url(&dep),
        "https://myresource.openai.azure.com/openai/deployments/my-gpt4-deployment/chat/completions?api-version=2024-10-21"
    );
}

/// LiteLLM accepts `api_version: latest` as an alias for the v1 surface, but
/// Azure's v1 REST reference documents only `v1` and `preview` as valid
/// `api-version` query values, so the alias must emit `v1` on the wire.
#[test]
fn latest_api_version_alias_emits_documented_v1_value() {
    let p = AzureProvider;
    let mut dep = deployment();
    dep.oxllm_params.api_version = Some("latest".to_string());
    assert_eq!(
        p.endpoint_url(&dep),
        "https://myresource.openai.azure.com/openai/v1/chat/completions?api-version=v1"
    );
}

// --- Coverage for the api_base query-string handling (LOW) ---
//
// If `api_base` carries a query string (e.g. `…?api-version=old`), the URL
// must not be malformed by a naive `format!` that appends the deployment
// path after the query. Existing params are preserved and `api-version` is
// appended.
#[test]
fn endpoint_url_with_api_base_containing_query_params() {
    let p = AzureProvider;
    let mut dep = deployment();
    dep.oxllm_params.api_base = Some("https://myresource.openai.azure.com?existing=1".to_string());
    let url = p.endpoint_url(&dep);
    assert!(
        url.contains("https://myresource.openai.azure.com/openai/deployments/my-gpt4-deployment/chat/completions"),
        "path must be appended before the query, got: {url}"
    );
    assert!(
        url.contains("existing=1"),
        "existing query params must be preserved, got: {url}"
    );
    assert!(
        url.contains("api-version=2024-02-15-preview"),
        "api-version must be appended, got: {url}"
    );
    assert_eq!(
        url,
        "https://myresource.openai.azure.com/openai/deployments/my-gpt4-deployment/chat/completions?existing=1&api-version=2024-02-15-preview"
    );
}

#[test]
fn endpoint_url_with_api_base_containing_trailing_slash_and_query() {
    let p = AzureProvider;
    let mut dep = deployment();
    dep.oxllm_params.api_base = Some("https://myresource.openai.azure.com/?existing=1".to_string());
    let url = p.endpoint_url(&dep);
    assert!(
        !url.contains(".com//openai"),
        "trailing slash before query must not double the slash, got: {url}"
    );
    assert!(
        url.contains("existing=1&api-version="),
        "query params must be merged, got: {url}"
    );
}

#[test]
fn gpt6_request_drops_rejected_params_and_orphan_tool_choice() {
    let p = AzureProvider;
    let mut dep = deployment();
    dep.oxllm_params.model = "azure/gpt-6-luna".to_string();
    dep.model_info = Some(oxllm_core::deployment::ModelInfo {
        base_model: Some("openai/gpt-6-luna".into()),
        ..Default::default()
    });
    let req = json!({
        "model": "my-deployment",
        "messages": [{"role": "user", "content": "Hi"}],
        "temperature": 0.2,
        "max_tokens": 32,
        "tool_choice": "auto",
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert!(out.get("temperature").is_none());
    assert!(out.get("tool_choice").is_none());
    assert_eq!(out["max_completion_tokens"], 32);
    assert!(out.get("max_tokens").is_none());
}
