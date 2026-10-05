//! Integration tests: `session_affinity` routing and `x-oxllm-session-id`.

use httpmock::{Mock, MockServer};
use oxllm_server::OxllmParams;
use oxllm_server::app::{build_router, build_shared_state};
use oxllm_server::config::types::{
    ConfigModel, GeneralSettings, OxllmSettings, ProxyConfig, RoutingKnobs,
};
use serde_json::{Value, json};
use std::collections::HashSet;

struct OxllmHandle {
    addr: std::net::SocketAddr,
    shutdown_tx: tokio::sync::oneshot::Sender<()>,
}

impl OxllmHandle {
    fn url(&self, path: &str) -> String {
        format!("http://{}{path}", self.addr)
    }
}

fn params(litellm_model: &str, base: &str, key: &str) -> OxllmParams {
    OxllmParams {
        model: litellm_model.to_string(),
        api_key: Some(key.to_string()),
        api_base: Some(base.to_string()),
        timeout: Some(30.0),
        stream_timeout: Some(30),
        ..Default::default()
    }
}

fn model_entry(name: &str, litellm_model: &str, base: &str, key: &str) -> ConfigModel {
    ConfigModel {
        model_name: name.to_string(),
        oxllm_params: params(litellm_model, base, key),
        ..Default::default()
    }
}

fn success_body(content: &str) -> Value {
    json!({
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "created": 1,
    "model": "gpt-4o",
    "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": content},
    "finish_reason": "stop"
    }],
    "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10}
    })
}

fn setup_ok_mock<'a>(mock: &'a MockServer, content: &str, key: &str) -> Mock<'a> {
    mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .header("authorization", format!("Bearer {key}"));
        then.status(200)
            .header("content-type", "application/json")
            .json_body(success_body(content));
    })
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

fn affinity_config(model_list: Vec<ConfigModel>) -> ProxyConfig {
    ProxyConfig {
        model_list,
        oxllm_settings: OxllmSettings {
            routing: RoutingKnobs {
                routing_strategy: Some("session_affinity".to_string()),
                num_retries: Some(0),
                ..Default::default()
            },
            ..Default::default()
        },
        general_settings: GeneralSettings {
            allow_private_api_base: Some(true),
            ..Default::default()
        },
        router_settings: Default::default(),
        extra: serde_json::Map::new(),
    }
}

async fn send_chat(
    handle: &OxllmHandle,
    model: &str,
    session_id: Option<&str>,
) -> reqwest::Response {
    let mut req = reqwest::Client::new()
        .post(handle.url("/v1/chat/completions"))
        .json(&json!({
        "model": model,
        "messages": [{"role": "user", "content": "hi"}],
        }));
    if let Some(sid) = session_id {
        req = req.header("x-oxllm-session-id", sid);
    }
    req.send().await.expect("request should succeed")
}

async fn content_of(resp: reqwest::Response) -> String {
    assert_eq!(resp.status(), 200, "chat should succeed");
    let body: Value = resp.json().await.expect("json");
    body["choices"][0]["message"]["content"]
        .as_str()
        .expect("content")
        .to_string()
}

#[tokio::test]
async fn session_affinity_same_id_sticks_to_one_deployment() {
    let mock_a = MockServer::start();
    let mock_b = MockServer::start();
    let mock_c = MockServer::start();

    let _a = setup_ok_mock(&mock_a, "from-a", "key-a");
    let _b = setup_ok_mock(&mock_b, "from-b", "key-b");
    let _c = setup_ok_mock(&mock_c, "from-c", "key-c");

    let handle = start_oxllm(affinity_config(vec![
        model_entry("chat", "openai/gpt-4o", &mock_a.base_url(), "key-a"),
        model_entry("chat", "xai/grok-4", &mock_b.base_url(), "key-b"),
        model_entry(
            "chat",
            "openrouter/openai/gpt-4o",
            &mock_c.base_url(),
            "key-c",
        ),
    ]))
    .await
    .expect("server starts");

    let mut contents = HashSet::new();
    for _ in 0..20 {
        let resp = send_chat(&handle, "chat", Some("thread-sticky-1")).await;
        contents.insert(content_of(resp).await);
    }
    assert_eq!(
        contents.len(),
        1,
        "same session id must stick to one deployment, got {contents:?}"
    );

    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn session_affinity_is_stable_across_many_requests() {
    let mock_a = MockServer::start();
    let mock_b = MockServer::start();

    let hits_a = setup_ok_mock(&mock_a, "from-a", "key-a");
    let hits_b = setup_ok_mock(&mock_b, "from-b", "key-b");

    let handle = start_oxllm(affinity_config(vec![
        model_entry("chat", "openai/gpt-4o", &mock_a.base_url(), "key-a"),
        model_entry("chat", "xai/grok-4", &mock_b.base_url(), "key-b"),
    ]))
    .await
    .expect("server starts");

    let first = content_of(send_chat(&handle, "chat", Some("stable-session")).await).await;
    for _ in 0..15 {
        let again = content_of(send_chat(&handle, "chat", Some("stable-session")).await).await;
        assert_eq!(first, again, "session affinity must not drift");
    }

    // Exactly one of the two upstreams should have been hit.
    let total = hits_a.hits() + hits_b.hits();
    assert_eq!(total, 16);
    assert!(
        hits_a.hits() == 16 || hits_b.hits() == 16,
        "all hits should land on one mock (a={}, b={})",
        hits_a.hits(),
        hits_b.hits()
    );

    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn session_affinity_different_ids_can_spread() {
    let mock_a = MockServer::start();
    let mock_b = MockServer::start();
    let mock_c = MockServer::start();

    let _a = setup_ok_mock(&mock_a, "from-a", "key-a");
    let _b = setup_ok_mock(&mock_b, "from-b", "key-b");
    let _c = setup_ok_mock(&mock_c, "from-c", "key-c");

    let handle = start_oxllm(affinity_config(vec![
        model_entry("chat", "openai/gpt-4o", &mock_a.base_url(), "key-a"),
        model_entry("chat", "xai/grok-4", &mock_b.base_url(), "key-b"),
        model_entry(
            "chat",
            "openrouter/openai/gpt-4o",
            &mock_c.base_url(),
            "key-c",
        ),
    ]))
    .await
    .expect("server starts");

    let mut contents = HashSet::new();
    for i in 0..48 {
        let sid = format!("session-{i}");
        let resp = send_chat(&handle, "chat", Some(&sid)).await;
        contents.insert(content_of(resp).await);
    }
    assert!(
        contents.len() > 1,
        "different session ids should spread across deployments, got {contents:?}"
    );

    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn session_affinity_missing_header_falls_back_to_shuffle() {
    let mock_a = MockServer::start();
    let mock_b = MockServer::start();

    let _a = setup_ok_mock(&mock_a, "from-a", "key-a");
    let _b = setup_ok_mock(&mock_b, "from-b", "key-b");

    let handle = start_oxllm(affinity_config(vec![
        model_entry("chat", "openai/gpt-4o", &mock_a.base_url(), "key-a"),
        model_entry("chat", "xai/grok-4", &mock_b.base_url(), "key-b"),
    ]))
    .await
    .expect("server starts");

    let mut contents = HashSet::new();
    for _ in 0..40 {
        let resp = send_chat(&handle, "chat", None).await;
        contents.insert(content_of(resp).await);
    }
    assert!(
        contents.len() > 1,
        "without x-oxllm-session-id, shuffle fallback should spread, got {contents:?}"
    );

    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn session_affinity_empty_header_falls_back_to_shuffle() {
    let mock_a = MockServer::start();
    let mock_b = MockServer::start();

    let _a = setup_ok_mock(&mock_a, "from-a", "key-a");
    let _b = setup_ok_mock(&mock_b, "from-b", "key-b");

    let handle = start_oxllm(affinity_config(vec![
        model_entry("chat", "openai/gpt-4o", &mock_a.base_url(), "key-a"),
        model_entry("chat", "xai/grok-4", &mock_b.base_url(), "key-b"),
    ]))
    .await
    .expect("server starts");

    let mut contents = HashSet::new();
    for _ in 0..40 {
        // Whitespace-only is treated as empty by extract_session_id.
        let resp = send_chat(&handle, "chat", Some(" ")).await;
        contents.insert(content_of(resp).await);
    }
    assert!(
        contents.len() > 1,
        "blank session id should fall back to shuffle, got {contents:?}"
    );

    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn session_affinity_embeddings_honors_header() {
    let mock_a = MockServer::start();
    let mock_b = MockServer::start();

    let emb_body = json!({
    "object": "list",
    "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
    "model": "text-embedding-3-small",
    "usage": {"prompt_tokens": 2, "total_tokens": 2}
    });

    let hits_a = mock_a.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/embeddings")
            .header("authorization", "Bearer key-a");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(emb_body.clone());
    });
    let hits_b = mock_b.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/embeddings")
            .header("authorization", "Bearer key-b");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(emb_body);
    });

    let handle = start_oxllm(affinity_config(vec![
        model_entry(
            "embed",
            "openai/text-embedding-3-small",
            &mock_a.base_url(),
            "key-a",
        ),
        model_entry(
            "embed",
            "openai/text-embedding-3-small-b",
            &mock_b.base_url(),
            "key-b",
        ),
    ]))
    .await
    .expect("server starts");

    for _ in 0..12 {
        let resp = reqwest::Client::new()
            .post(handle.url("/v1/embeddings"))
            .header("x-oxllm-session-id", "embed-session-1")
            .json(&json!({
            "model": "embed",
            "input": "hello",
            }))
            .send()
            .await
            .expect("embeddings request");
        assert_eq!(resp.status(), 200);
    }

    let total = hits_a.hits() + hits_b.hits();
    assert_eq!(total, 12);
    assert!(
        hits_a.hits() == 12 || hits_b.hits() == 12,
        "embeddings session affinity should stick (a={}, b={})",
        hits_a.hits(),
        hits_b.hits()
    );

    let _ = handle.shutdown_tx.send(());
}
