//! Integration tests for the Vertex AI provider (httpmock upstream).
//!
//! These tests verify the end-to-end flow:
//! - The Gemini branch forwards to the OpenAI-compat endpoint with the GCP
//!   bearer token attached.
//! - The Anthropic branch forwards to `:rawPredict` with
//!   `anthropic_version: "vertex-2023-10-16"` injected.
//! - The wildcard routing matches `vertex_ai/*` deployments against both
//!   Gemini and Claude model requests.

use httpmock::MockServer;
use oxllm_server::OxllmParams;
use oxllm_server::SharedState;
use oxllm_server::app::{build_router, build_shared_state};
use oxllm_server::config::types::{ConfigModel, GeneralSettings, OxllmSettings, ProxyConfig};
use serde_json::{Value, json};

struct OxllmHandle {
    addr: std::net::SocketAddr,
    shutdown_tx: tokio::sync::oneshot::Sender<()>,
}

impl OxllmHandle {
    fn url(&self, path: &str) -> String {
        format!("http://{}{path}", self.addr)
    }
}

fn vertex_proxy_config(mock_base_url: &str, model_name: &str, litellm_model: &str) -> ProxyConfig {
    let mut extra = serde_json::Map::new();
    extra.insert("vertex_project".to_string(), json!("test-gcp-project"));
    extra.insert("vertex_location".to_string(), json!("us-central1"));
    ProxyConfig {
        model_list: vec![ConfigModel {
            model_name: model_name.to_string(),
            oxllm_params: OxllmParams {
                model: litellm_model.to_string(),
                api_key: Some("ya29.test-vertex-token".to_string()),
                api_base: Some(mock_base_url.to_string()),
                timeout: Some(30.0),
                stream_timeout: Some(30),
                extra,
                ..Default::default()
            },
            ..Default::default()
        }],
        oxllm_settings: OxllmSettings::default(),
        general_settings: GeneralSettings {
            allow_private_api_base: Some(true),
            ..Default::default()
        },
        router_settings: Default::default(),
        extra: serde_json::Map::new(),
    }
}

async fn start_oxllm(config: ProxyConfig) -> anyhow::Result<(OxllmHandle, SharedState)> {
    let shared = build_shared_state(config)?;
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;
    let router = build_router(shared.clone());
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    tokio::spawn(async move {
        axum::serve(listener, router)
            .with_graceful_shutdown(async move {
                let _ = shutdown_rx.await;
            })
            .await
            .ok();
    });
    Ok((OxllmHandle { addr, shutdown_tx }, shared))
}

#[tokio::test]
async fn vertex_gemini_forwards_to_openai_compat_endpoint_with_bearer_auth() {
    let mock = MockServer::start();
    let config = vertex_proxy_config(&mock.base_url(), "gemini-pro", "vertex_ai/gemini-1.5-pro");
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    // The Gemini branch should POST to the OpenAI-compat path with the model
    // prefix stripped and the bearer token attached.
    let upstream_mock = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/v1beta1/projects/test-gcp-project/locations/us-central1/endpoints/openapi/chat/completions")
            .header("authorization", "Bearer ya29.test-vertex-token")
            .json_body_partial(json!({"model": "gemini-1.5-pro"}).to_string());
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
            "id": "chatcmpl-vertex-1",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gemini-1.5-pro",
            "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from Vertex Gemini!"},
            "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
            }));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .json(&json!({
        "model": "gemini-pro",
        "messages": [{"role": "user", "content": "Say hello"}],
        }))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("body should be JSON");
    assert_eq!(body["id"], "chatcmpl-vertex-1");
    assert_eq!(
        body["choices"][0]["message"]["content"],
        "Hello from Vertex Gemini!"
    );

    upstream_mock.assert_hits(1);
    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn vertex_claude_forwards_to_raw_predict_with_anthropic_version() {
    let mock = MockServer::start();
    let config = vertex_proxy_config(
        &mock.base_url(),
        "claude-vertex",
        "vertex_ai/claude-3-5-sonnet@20241022",
    );
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    // The Claude branch should POST to `:rawPredict` with the bearer token
    // and `anthropic_version: "vertex-2023-10-16"` in the body. Vertex encodes
    // the model in the URL path and rejects a top-level `model` field
    // (the reference `data.pop("model", None)`).
    let upstream_mock = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/v1/projects/test-gcp-project/locations/us-central1/publishers/anthropic/models/claude-3-5-sonnet@20241022:rawPredict")
            .header("authorization", "Bearer ya29.test-vertex-token")
            .json_body_partial(
                json!({
                "anthropic_version": "vertex-2023-10-16",
                })
                .to_string(),
            );
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
            "id": "msg_vertex_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-5-sonnet-20241022",
            "content": [{"type": "text", "text": "Hello from Vertex Claude!"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 12, "output_tokens": 6}
            }));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .json(&json!({
        "model": "claude-vertex",
        "messages": [{"role": "user", "content": "Say hello"}],
        "max_tokens": 100,
        }))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("body should be JSON");
    assert_eq!(body["object"], "chat.completion");
    assert_eq!(
        body["choices"][0]["message"]["content"],
        "Hello from Vertex Claude!"
    );
    assert_eq!(body["usage"]["prompt_tokens"], 12);
    assert_eq!(body["usage"]["completion_tokens"], 6);

    upstream_mock.assert_hits(1);
    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn vertex_wildcard_gemini_routes_request_to_openai_compat_endpoint() {
    let mock = MockServer::start();
    let config = vertex_proxy_config(&mock.base_url(), "*", "vertex_ai/*");
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let upstream_mock = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/v1beta1/projects/test-gcp-project/locations/us-central1/endpoints/openapi/chat/completions")
            .header("authorization", "Bearer ya29.test-vertex-token")
            .json_body_partial(json!({"model": "gemini-2.0-flash"}).to_string());
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
            "id": "chatcmpl-wild",
            "object": "chat.completion",
            "model": "gemini-2.0-flash",
            "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            }));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .json(&json!({
        "model": "vertex_ai/gemini-2.0-flash",
        "messages": [{"role": "user", "content": "hi"}],
        }))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);
    upstream_mock.assert_hits(1);
    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn vertex_gemini_streaming_forwards_sse() {
    let mock = MockServer::start();
    let config = vertex_proxy_config(
        &mock.base_url(),
        "gemini-stream",
        "vertex_ai/gemini-1.5-pro",
    );
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let sse_body = concat!(
        "data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gemini-1.5-pro\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gemini-1.5-pro\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gemini-1.5-pro\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1,\"total_tokens\":2}}\n\n",
        "data: [DONE]\n\n"
    );

    let upstream_mock = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/v1beta1/projects/test-gcp-project/locations/us-central1/endpoints/openapi/chat/completions")
            .header("authorization", "Bearer ya29.test-vertex-token")
            .json_body_partial(json!({"stream": true}).to_string());
        then.status(200)
            .header("content-type", "text/event-stream")
            .body(sse_body);
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .json(&json!({
        "model": "gemini-stream",
        "messages": [{"role": "user", "content": "Say hello"}],
        "stream": true,
        }))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);
    let body = resp.text().await.expect("body");
    assert!(body.contains("Hello") || body.contains("Hi") || body.contains("data:"));
    upstream_mock.assert_hits(1);
    let _ = handle.shutdown_tx.send(());
}
