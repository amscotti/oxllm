//! Integration tests for the AWS Bedrock provider.
//!
//! Verifies end-to-end flow: an OpenAI-shaped client request is routed to a
//! Bedrock deployment, transformed to the Anthropic Messages body shape,
//! signed with AWS SigV4, forwarded to the mock upstream, and the native
//! Anthropic response is transformed back into the OpenAI chat-completion
//! shape.

use httpmock::MockServer;
use oxllm_server::OxllmParams;
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

async fn start_oxllm(config: ProxyConfig) -> anyhow::Result<OxllmHandle> {
    let shared = build_shared_state(config)?;

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;
    let router = build_router(shared);

    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    tokio::spawn(async move {
        axum::serve(listener, router)
            .with_graceful_shutdown(async move {
                let _ = shutdown_rx.await;
            })
            .await
            .ok();
    });

    Ok(OxllmHandle { addr, shutdown_tx })
}

fn bedrock_config(mock_base_url: &str, model_name: &str, configured_model: &str) -> ProxyConfig {
    let mut extra = serde_json::Map::new();
    extra.insert(
        "aws_access_key_id".to_string(),
        serde_json::Value::String("AKIATESTKEY".to_string()),
    );
    extra.insert(
        "aws_secret_access_key".to_string(),
        serde_json::Value::String("supersecretkey".to_string()),
    );
    extra.insert(
        "aws_region".to_string(),
        serde_json::Value::String("us-east-1".to_string()),
    );

    ProxyConfig {
        model_list: vec![ConfigModel {
            model_name: model_name.to_string(),
            oxllm_params: OxllmParams {
                model: configured_model.to_string(),
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

#[tokio::test]
async fn bedrock_anthropic_non_streaming_signs_and_routes_correctly() {
    let mock = MockServer::start();
    let config = bedrock_config(
        &mock.base_url(),
        "bedrock-claude",
        "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
    );
    let handle = start_oxllm(config).await.expect("server start");

    let _m = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path_matches(
                regex::Regex::new(r"^/model/anthropic\.claude-3-5-sonnet-20241022-v2:0/invoke$")
                    .unwrap(),
            )
            .header_exists("authorization")
            .header_exists("x-amz-date")
            .header_exists("x-amz-content-sha256")
            .json_body_partial(
                json!({
                "anthropic_version": "bedrock-2023-05-31",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 100,
                })
                .to_string(),
            );
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
            "id": "msg_bedrock_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-5-sonnet-20241022-v2:0",
            "content": [{"type": "text", "text": "Hello from Bedrock!"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 4},
            }));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .json(&json!({
        "model": "bedrock-claude",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 100,
        }))
        .send()
        .await
        .expect("request");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("json");
    assert_eq!(body["object"], "chat.completion");
    assert_eq!(body["id"], "msg_bedrock_1");
    assert_eq!(body["model"], "bedrock-claude");
    assert_eq!(
        body["choices"][0]["message"]["content"],
        "Hello from Bedrock!"
    );
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
    assert_eq!(body["usage"]["prompt_tokens"], 5);
    assert_eq!(body["usage"]["completion_tokens"], 4);
    assert_eq!(body["usage"]["total_tokens"], 9);

    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn bedrock_authorization_header_is_sigv4_shaped() {
    let mock = MockServer::start();
    let config = bedrock_config(
        &mock.base_url(),
        "bedrock-claude",
        "bedrock/anthropic.claude-3-haiku",
    );
    let handle = start_oxllm(config).await.expect("server start");

    let _m = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path_contains("/model/anthropic.claude-3-haiku/invoke")
            .header_exists("authorization")
            .header_exists("x-amz-date")
            .header_exists("x-amz-content-sha256")
            .header_exists("host");
        then.status(200).json_body(json!({
            "id": "msg", "model": "claude-3-haiku",
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .json(&json!({
        "model": "bedrock-claude",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 5,
        }))
        .send()
        .await
        .expect("request");

    assert_eq!(resp.status(), 200);
    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn bedrock_strips_bedrock_prefix_in_url() {
    let mock = MockServer::start();
    let config = bedrock_config(
        &mock.base_url(),
        "bedrock-claude",
        "bedrock/anthropic.claude-3-5-sonnet",
    );
    let handle = start_oxllm(config).await.expect("server start");

    let _m = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            // Must NOT contain a "bedrock/" segment in the URL path.
            .path("/model/anthropic.claude-3-5-sonnet/invoke");
        then.status(200).json_body(json!({
            "id": "msg", "model": "claude",
            "content": [{"type": "text", "text": "x"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .json(&json!({
        "model": "bedrock-claude",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 1,
        }))
        .send()
        .await
        .expect("request");

    assert_eq!(resp.status(), 200);
    let _ = handle.shutdown_tx.send(());
}
