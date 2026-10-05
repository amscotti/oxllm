//! Integration tests: Langfuse observability hook end-to-end.

use std::sync::Arc;
use std::time::Duration;

use httpmock::MockServer;
use oxllm_server::app::{build_router, build_shared_state_with_db};
use oxllm_server::config::types::{
    ConfigModel, GeneralSettings, LangfuseConfig, OxllmSettings, ProxyConfig,
};
use oxllm_server::spend_writer;
use oxllm_server::{Database, OxllmParams, SharedState};
use serde_json::json;

const MASTER_KEY: &str = "sk-master-langfuse";
const VIRTUAL_KEY: &str = "sk-virtual-langfuse";

struct OxllmHandle {
    addr: std::net::SocketAddr,
    shutdown_tx: tokio::sync::oneshot::Sender<()>,
    _db: Arc<Database>,
}

impl OxllmHandle {
    fn url(&self, path: &str) -> String {
        format!("http://{}{path}", self.addr)
    }
}

fn base_params(mock_base_url: &str) -> OxllmParams {
    OxllmParams {
        model: "openai/gpt-4o".to_string(),
        api_key: Some("sk-upstream".to_string()),
        api_base: Some(mock_base_url.to_string()),
        timeout: Some(30.0),
        stream_timeout: Some(30),
        ..Default::default()
    }
}

fn proxy_config_with_langfuse(
    mock_base_url: &str,
    langfuse_host: &str,
    max_batch_size: usize,
) -> ProxyConfig {
    ProxyConfig {
        model_list: vec![ConfigModel {
            model_name: "gpt-4o".to_string(),
            oxllm_params: base_params(mock_base_url),
            ..Default::default()
        }],
        oxllm_settings: OxllmSettings {
            langfuse_config: Some(LangfuseConfig {
                host: langfuse_host.to_string(),
                public_key: "pk-lf-test".to_string(),
                secret_key: "sk-lf-test".to_string(),
                flush_interval_secs: Some(60),
                max_batch_size: Some(max_batch_size),
                log_inputs: Some(true),
                log_outputs: Some(true),
            }),
            ..Default::default()
        },
        general_settings: GeneralSettings {
            master_key: Some(MASTER_KEY.to_string()),
            allow_private_api_base: Some(true),
            ..Default::default()
        },
        router_settings: Default::default(),
        extra: serde_json::Map::new(),
    }
}

fn proxy_config_with_langfuse_no_outputs(mock_base_url: &str, langfuse_host: &str) -> ProxyConfig {
    ProxyConfig {
        model_list: vec![ConfigModel {
            model_name: "gpt-4o".to_string(),
            oxllm_params: base_params(mock_base_url),
            ..Default::default()
        }],
        oxllm_settings: OxllmSettings {
            langfuse_config: Some(LangfuseConfig {
                host: langfuse_host.to_string(),
                public_key: "pk-lf-test".to_string(),
                secret_key: "sk-lf-test".to_string(),
                flush_interval_secs: Some(60),
                max_batch_size: Some(2),
                log_inputs: Some(false),
                log_outputs: Some(false),
            }),
            ..Default::default()
        },
        general_settings: GeneralSettings {
            master_key: Some(MASTER_KEY.to_string()),
            allow_private_api_base: Some(true),
            ..Default::default()
        },
        router_settings: Default::default(),
        extra: serde_json::Map::new(),
    }
}

async fn fresh_db() -> Arc<Database> {
    let db = Database::connect("sqlite::memory:")
        .await
        .expect("connect sqlite memory");
    db.migrate().await.expect("migrate");
    db
}

async fn start_oxllm(config: ProxyConfig) -> OxllmHandle {
    let db = fresh_db().await;
    // Create a virtual key so we have a non-master principal to exercise the
    // happy path (master-key requests are skipped by the Langfuse hook).
    create_virtual_key(&db).await;
    let writer = spend_writer::spawn(db.clone(), 1);
    let shared: SharedState =
        build_shared_state_with_db(config, Some(writer.sender()), Some(db.clone()))
            .expect("build state");
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let addr = listener.local_addr().expect("addr");
    let router = build_router(shared);
    let make_service = router.into_make_service_with_connect_info::<std::net::SocketAddr>();
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    tokio::spawn(async move {
        axum::serve(listener, make_service)
            .with_graceful_shutdown(async move {
                let _ = shutdown_rx.await;
            })
            .await
            .ok();
    });
    OxllmHandle {
        addr,
        shutdown_tx,
        _db: db,
    }
}

async fn create_virtual_key(db: &Arc<Database>) {
    use oxllm_server::KeyRecord;
    let now = oxllm_server::now_iso();
    db.create_team(&oxllm_server::TeamRecord {
        team_id: "team-lf-1".into(),
        team_name: "Langfuse test team".into(),
        max_budget: None,
        budget_duration: None,
        rpm: None,
        tpm: None,
        created_at: now.clone(),
        updated_at: now.clone(),
        models: None,
        blocked: false,
    })
    .await
    .expect("create virtual key team");
    let record = KeyRecord {
        key_id: "k-lf-1".to_string(),
        key_hash: oxllm_server::auth::hash_key_hex(VIRTUAL_KEY.as_bytes()),
        key_alias: Some("langfuse-test".to_string()),
        team_id: Some("team-lf-1".to_string()),
        user_id: None,
        org_id: None,
        project_id: None,
        max_budget: None,
        budget_duration: None,
        rpm: None,
        tpm: None,
        created_at: now.clone(),
        updated_at: now,
        expires_at: None,
        is_active: true,
        models: None,
        allowed_cache_controls: None,
    };
    db.create_key(&record).await.expect("create virtual key");
}

async fn shutdown(handle: OxllmHandle) {
    let _ = handle.shutdown_tx.send(());
    tokio::time::sleep(Duration::from_millis(50)).await;
}

fn mock_chat(mock: &MockServer) -> httpmock::Mock<'_> {
    mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .header("authorization", "Bearer sk-upstream");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
            "id": "chatcmpl-7",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-4o",
            "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "hi"},
            "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
            }));
    })
}

#[tokio::test]
async fn langfuse_receives_ingestion_batch_on_chat_completion() {
    let upstream = MockServer::start();
    let langfuse = MockServer::start();

    // max_batch_size = 2 → a single request (trace-create + generation-create)
    // triggers an immediate flush.
    let ingestion = langfuse.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/api/public/ingestion")
            .body_contains("\"type\":\"trace-create\"")
            .body_contains("\"type\":\"generation-create\"")
            .body_contains("\"model\":\"gpt-4o\"")
            .body_contains("\"promptTokens\":10");
        then.status(200).json_body(json!({"success": true}));
    });

    let handle = start_oxllm(proxy_config_with_langfuse(
        &upstream.base_url(),
        &langfuse.base_url(),
        2,
    ))
    .await;
    let _m = mock_chat(&upstream);

    let client = reqwest::Client::new();
    let started = std::time::Instant::now();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth(VIRTUAL_KEY)
        .json(&json!({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .expect("chat");
    let elapsed = started.elapsed();
    assert_eq!(resp.status(), 200, "chat should succeed");
    assert!(
        elapsed < Duration::from_secs(3),
        "chat should not block on the callback (took {elapsed:?})"
    );

    for _ in 0..60 {
        if ingestion.hits() >= 1 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert!(
        ingestion.hits() >= 1,
        "langfuse ingestion endpoint should have received a batch"
    );

    shutdown(handle).await;
}

#[tokio::test]
async fn langfuse_uses_basic_auth_header() {
    let upstream = MockServer::start();
    let langfuse = MockServer::start();

    // base64("pk-lf-test:sk-lf-test") = "cGstbGYtdGVzdDpzay1sZi10ZXN0"
    let expected = "Basic cGstbGYtdGVzdDpzay1sZi10ZXN0";
    let ingestion = langfuse.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/api/public/ingestion")
            .header("Authorization", expected);
        then.status(200);
    });

    let handle = start_oxllm(proxy_config_with_langfuse(
        &upstream.base_url(),
        &langfuse.base_url(),
        2,
    ))
    .await;
    let _m = mock_chat(&upstream);

    let client = reqwest::Client::new();
    let _ = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth(VIRTUAL_KEY)
        .json(&json!({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .expect("chat");

    for _ in 0..60 {
        if ingestion.hits() >= 1 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert_eq!(ingestion.hits(), 1, "Basic auth header must be present");

    shutdown(handle).await;
}

#[tokio::test]
async fn langfuse_carries_metadata() {
    let upstream = MockServer::start();
    let langfuse = MockServer::start();

    let ingestion = langfuse.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/api/public/ingestion")
            .body_contains("\"provider\":\"openai\"")
            .body_contains("\"key_id\":\"k-lf-1\"")
            .body_contains("\"team_id\":\"team-lf-1\"");
        then.status(200);
    });

    let handle = start_oxllm(proxy_config_with_langfuse(
        &upstream.base_url(),
        &langfuse.base_url(),
        2,
    ))
    .await;
    let _m = mock_chat(&upstream);

    let client = reqwest::Client::new();
    let _ = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth(VIRTUAL_KEY)
        .json(&json!({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .expect("chat");

    for _ in 0..60 {
        if ingestion.hits() >= 1 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert!(
        ingestion.hits() >= 1,
        "ingestion should carry cost / identity metadata"
    );

    shutdown(handle).await;
}

#[tokio::test]
async fn langfuse_respects_input_output_masking() {
    let upstream = MockServer::start();
    let langfuse = MockServer::start();

    // With log_inputs=false / log_outputs=false, both fields are null.
    let ingestion = langfuse.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/api/public/ingestion")
            .body_contains("\"input\":null")
            .body_contains("\"output\":null");
        then.status(200);
    });

    let handle = start_oxllm(proxy_config_with_langfuse_no_outputs(
        &upstream.base_url(),
        &langfuse.base_url(),
    ))
    .await;
    let _m = mock_chat(&upstream);

    let client = reqwest::Client::new();
    let _ = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth(VIRTUAL_KEY)
        .json(&json!({"model": "gpt-4o", "messages": [{"role": "user", "content": "secret"}]}))
        .send()
        .await
        .expect("chat");

    for _ in 0..60 {
        if ingestion.hits() >= 1 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    assert_eq!(
        ingestion.hits(),
        1,
        "input/output must be null when log_inputs/log_outputs are false"
    );

    shutdown(handle).await;
}

#[tokio::test]
async fn langfuse_skips_master_key_requests() {
    let upstream = MockServer::start();
    let langfuse = MockServer::start();

    let ingestion = langfuse.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/api/public/ingestion");
        then.status(200);
    });

    let handle = start_oxllm(proxy_config_with_langfuse(
        &upstream.base_url(),
        &langfuse.base_url(),
        2,
    ))
    .await;
    let _m = mock_chat(&upstream);

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth(MASTER_KEY)
        .json(&json!({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .expect("chat");
    assert_eq!(resp.status(), 200);

    tokio::time::sleep(Duration::from_millis(300)).await;
    assert_eq!(
        ingestion.hits(),
        0,
        "master-key requests must not be logged to Langfuse"
    );

    shutdown(handle).await;
}

#[tokio::test]
async fn langfuse_does_not_break_chat_on_unreachable_host() {
    let upstream = MockServer::start();
    let _m = mock_chat(&upstream);

    let handle = start_oxllm(proxy_config_with_langfuse(
        &upstream.base_url(),
        "http://127.0.0.1:1",
        2,
    ))
    .await;

    let client = reqwest::Client::new();
    let started = std::time::Instant::now();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth(VIRTUAL_KEY)
        .json(&json!({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .expect("chat");
    assert_eq!(resp.status(), 200);
    assert!(
        started.elapsed() < Duration::from_secs(3),
        "request must not block on the unreachable langfuse host"
    );

    shutdown(handle).await;
}
