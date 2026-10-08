//! Integration tests: platform-style APIs (batches, files, fine-tuning, assistants) and token counter.

use httpmock::MockServer;
use oxllm_server::OxllmParams;
use oxllm_server::app::{build_router, build_shared_state};
use oxllm_server::config::types::{ConfigModel, GeneralSettings, OxllmSettings, ProxyConfig};
use serde_json::{Value, json};

const MASTER_KEY: &str = "sk-master-platform";
const UPSTREAM_KEY: &str = "sk-upstream-platform";

struct OxllmHandle {
    addr: std::net::SocketAddr,
    shutdown_tx: tokio::sync::oneshot::Sender<()>,
}

impl OxllmHandle {
    fn url(&self, path: &str) -> String {
        format!("http://{}{path}", self.addr)
    }
}

fn proxy_config(mock_base_url: &str) -> ProxyConfig {
    ProxyConfig {
        model_list: vec![ConfigModel {
            model_name: "gpt-4o".to_string(),
            oxllm_params: OxllmParams {
                model: "openai/gpt-4o".to_string(),
                api_key: Some(UPSTREAM_KEY.to_string()),
                api_base: Some(mock_base_url.to_string()),
                timeout: Some(30.0),
                ..Default::default()
            },
            ..Default::default()
        }],
        oxllm_settings: OxllmSettings::default(),
        general_settings: GeneralSettings {
            master_key: Some(MASTER_KEY.to_string()),
            allow_private_api_base: Some(true),
            ..Default::default()
        },
        router_settings: Default::default(),
        extra: serde_json::Map::new(),
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

// ---------------------------------------------------------------------------
// Token counter (server-only)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn token_counter_returns_estimate_for_known_text() {
    let mock = MockServer::start();
    let config = proxy_config(&mock.base_url());
    let handle = start_oxllm(config).await.expect("server starts");

    let resp = reqwest::Client::new()
        .post(handle.url("/utils/token_counter"))
        .bearer_auth(MASTER_KEY)
        .json(&json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hello world"}]
        }))
        .send()
        .await
        .expect("request succeeds");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("json body");
    assert_eq!(body["request_model"], "gpt-4o");
    assert_eq!(body["completion_tokens"], 0);
    let prompt = body["prompt_tokens"].as_u64().expect("prompt_tokens u64");
    assert!(
        prompt > 0,
        "expected nonzero prompt token count, got {prompt}"
    );
    assert_eq!(body["total_tokens"], body["prompt_tokens"]);

    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn token_counter_rejects_missing_model() {
    let mock = MockServer::start();
    let config = proxy_config(&mock.base_url());
    let handle = start_oxllm(config).await.expect("server starts");

    let resp = reqwest::Client::new()
        .post(handle.url("/utils/token_counter"))
        .bearer_auth(MASTER_KEY)
        .json(&json!({"messages": []}))
        .send()
        .await
        .expect("request succeeds");

    assert_eq!(resp.status(), 400);
    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn token_counter_requires_auth() {
    let mock = MockServer::start();
    let config = proxy_config(&mock.base_url());
    let handle = start_oxllm(config).await.expect("server starts");

    let resp = reqwest::Client::new()
        .post(handle.url("/utils/token_counter"))
        .json(&json!({"model": "gpt-4o", "messages": []}))
        .send()
        .await
        .expect("request succeeds");

    assert_eq!(resp.status(), 401);
    let _ = handle.shutdown_tx.send(());
}

// ---------------------------------------------------------------------------
// Batches
// ---------------------------------------------------------------------------

#[tokio::test]
async fn batches_create_forwards_to_upstream() {
    let mock = MockServer::start();
    let _m = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/batches")
            .header("authorization", format!("Bearer {UPSTREAM_KEY}"))
            .json_body(json!({"model": "gpt-4o", "input_file_id": "file-abc"}));
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({"id": "batch_1", "object": "batch", "status": "validating"}));
    });

    let config = proxy_config(&mock.base_url());
    let handle = start_oxllm(config).await.expect("server starts");

    let resp = reqwest::Client::new()
        .post(handle.url("/v1/batches"))
        .bearer_auth(MASTER_KEY)
        .json(&json!({"model": "gpt-4o", "input_file_id": "file-abc"}))
        .send()
        .await
        .expect("request succeeds");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("json body");
    assert_eq!(body["id"], "batch_1");
    assert_eq!(_m.hits(), 1);
    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn batches_list_forwards_query_string() {
    let mock = MockServer::start();
    let _m = mock.mock(|when, then| {
        when.method(httpmock::Method::GET)
            .path("/batches")
            .query_param("limit", "10")
            .header("authorization", format!("Bearer {UPSTREAM_KEY}"));
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({"data": [], "object": "list"}));
    });

    let config = proxy_config(&mock.base_url());
    let handle = start_oxllm(config).await.expect("server starts");

    let resp = reqwest::Client::new()
        .get(handle.url("/v1/batches?limit=10"))
        .bearer_auth(MASTER_KEY)
        .send()
        .await
        .expect("request succeeds");

    assert_eq!(resp.status(), 200);
    assert_eq!(_m.hits(), 1);
    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn batches_retrieve_includes_id_in_path() {
    let mock = MockServer::start();
    let _m = mock.mock(|when, then| {
        when.method(httpmock::Method::GET)
            .path("/batches/batch_42")
            .header("authorization", format!("Bearer {UPSTREAM_KEY}"));
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({"id": "batch_42", "object": "batch"}));
    });

    let config = proxy_config(&mock.base_url());
    let handle = start_oxllm(config).await.expect("server starts");

    let resp = reqwest::Client::new()
        .get(handle.url("/v1/batches/batch_42"))
        .bearer_auth(MASTER_KEY)
        .send()
        .await
        .expect("request succeeds");

    assert_eq!(resp.status(), 200);
    assert_eq!(_m.hits(), 1);
    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn batches_cancel_posts_to_cancel_path() {
    let mock = MockServer::start();
    let _m = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/batches/batch_99/cancel")
            .header("authorization", format!("Bearer {UPSTREAM_KEY}"));
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({"id": "batch_99", "status": "cancelling"}));
    });

    let config = proxy_config(&mock.base_url());
    let handle = start_oxllm(config).await.expect("server starts");

    let resp = reqwest::Client::new()
        .post(handle.url("/v1/batches/batch_99/cancel"))
        .bearer_auth(MASTER_KEY)
        .send()
        .await
        .expect("request succeeds");

    assert_eq!(resp.status(), 200);
    assert_eq!(_m.hits(), 1);
    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn batches_requires_auth() {
    let mock = MockServer::start();
    let config = proxy_config(&mock.base_url());
    let handle = start_oxllm(config).await.expect("server starts");

    let resp = reqwest::Client::new()
        .post(handle.url("/v1/batches"))
        .json(&json!({"model": "gpt-4o"}))
        .send()
        .await
        .expect("request succeeds");

    assert_eq!(resp.status(), 401);
    let _ = handle.shutdown_tx.send(());
}

/// The `?model=` query param is an oxllm routing input, not a provider API
/// parameter — it must be stripped before forwarding upstream so the
/// internal alias is not leaked. Other query params (e.g. `limit`) must
/// pass through.
#[tokio::test]
async fn batches_list_strips_model_query_param_from_upstream() {
    let mock = MockServer::start();
    let _m = mock.mock(|when, then| {
        when.method(httpmock::Method::GET)
            .path("/batches")
            .query_param("limit", "5")
            .header("authorization", format!("Bearer {UPSTREAM_KEY}"));
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({"data": [], "object": "list"}));
    });
    // A separate mock that would match if the model param leaked through.
    let leak = mock.mock(|when, then| {
        when.method(httpmock::Method::GET)
            .path("/batches")
            .query_param("model", "gpt-4o");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({"leaked": true}));
    });

    let config = proxy_config(&mock.base_url());
    let handle = start_oxllm(config).await.expect("server starts");

    let resp = reqwest::Client::new()
        .get(handle.url("/v1/batches?model=gpt-4o&limit=5"))
        .bearer_auth(MASTER_KEY)
        .send()
        .await
        .expect("request succeeds");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("json body");
    assert!(
        body.get("leaked").is_none(),
        "model param must not reach upstream"
    );
    assert_eq!(_m.hits(), 1, "limit param must reach upstream");
    assert_eq!(leak.hits(), 0, "model param must be stripped");
    let _ = handle.shutdown_tx.send(());
}

// ---------------------------------------------------------------------------
// Files
// ---------------------------------------------------------------------------

#[tokio::test]
async fn files_upload_forwards_multipart_body() {
    let mock = MockServer::start();
    let _m = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/files")
            .header("authorization", format!("Bearer {UPSTREAM_KEY}"))
            .header("content-type", "multipart/form-data; boundary=xyz");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({"id": "file-1", "object": "file"}));
    });

    let config = proxy_config(&mock.base_url());
    let handle = start_oxllm(config).await.expect("server starts");

    let resp = reqwest::Client::new()
        .post(handle.url("/v1/files"))
        .bearer_auth(MASTER_KEY)
        .header("content-type", "multipart/form-data; boundary=xyz")
        .body("--xyz--\r\n")
        .send()
        .await
        .expect("request succeeds");

    assert_eq!(resp.status(), 200);
    assert_eq!(_m.hits(), 1);
    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn files_list_forwards_to_files_path() {
    let mock = MockServer::start();
    let _m = mock.mock(|when, then| {
        when.method(httpmock::Method::GET)
            .path("/files")
            .header("authorization", format!("Bearer {UPSTREAM_KEY}"));
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({"data": [], "object": "list"}));
    });

    let config = proxy_config(&mock.base_url());
    let handle = start_oxllm(config).await.expect("server starts");

    let resp = reqwest::Client::new()
        .get(handle.url("/v1/files"))
        .bearer_auth(MASTER_KEY)
        .send()
        .await
        .expect("request succeeds");

    assert_eq!(resp.status(), 200);
    assert_eq!(_m.hits(), 1);
    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn files_retrieve_includes_id_in_path() {
    let mock = MockServer::start();
    let _m = mock.mock(|when, then| {
        when.method(httpmock::Method::GET)
            .path("/files/file-7")
            .header("authorization", format!("Bearer {UPSTREAM_KEY}"));
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({"id": "file-7", "object": "file"}));
    });

    let config = proxy_config(&mock.base_url());
    let handle = start_oxllm(config).await.expect("server starts");

    let resp = reqwest::Client::new()
        .get(handle.url("/v1/files/file-7"))
        .bearer_auth(MASTER_KEY)
        .send()
        .await
        .expect("request succeeds");

    assert_eq!(resp.status(), 200);
    assert_eq!(_m.hits(), 1);
    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn files_delete_uses_delete_method() {
    let mock = MockServer::start();
    let _m = mock.mock(|when, then| {
        when.method(httpmock::Method::DELETE)
            .path("/files/file-9")
            .header("authorization", format!("Bearer {UPSTREAM_KEY}"));
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({"id": "file-9", "deleted": true, "object": "file"}));
    });

    let config = proxy_config(&mock.base_url());
    let handle = start_oxllm(config).await.expect("server starts");

    let resp = reqwest::Client::new()
        .delete(handle.url("/v1/files/file-9"))
        .bearer_auth(MASTER_KEY)
        .send()
        .await
        .expect("request succeeds");

    assert_eq!(resp.status(), 200);
    assert_eq!(_m.hits(), 1);
    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn files_content_returns_binary_body() {
    let mock = MockServer::start();
    let _m = mock.mock(|when, then| {
        when.method(httpmock::Method::GET)
            .path("/files/file-bin/content")
            .header("authorization", format!("Bearer {UPSTREAM_KEY}"));
        then.status(200)
            .header("content-type", "application/octet-stream")
            .body(b"raw-bytes-here");
    });

    let config = proxy_config(&mock.base_url());
    let handle = start_oxllm(config).await.expect("server starts");

    let resp = reqwest::Client::new()
        .get(handle.url("/v1/files/file-bin/content"))
        .bearer_auth(MASTER_KEY)
        .send()
        .await
        .expect("request succeeds");

    assert_eq!(resp.status(), 200);
    let bytes = resp.bytes().await.expect("body");
    assert_eq!(&bytes[..], b"raw-bytes-here");
    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn files_requires_auth() {
    let mock = MockServer::start();
    let config = proxy_config(&mock.base_url());
    let handle = start_oxllm(config).await.expect("server starts");

    let resp = reqwest::Client::new()
        .get(handle.url("/v1/files"))
        .send()
        .await
        .expect("request succeeds");

    assert_eq!(resp.status(), 401);
    let _ = handle.shutdown_tx.send(());
}

// ---------------------------------------------------------------------------
// Fine-tuning
// ---------------------------------------------------------------------------

#[tokio::test]
async fn fine_tuning_create_forwards_to_jobs_path() {
    let mock = MockServer::start();
    let _m = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/fine_tuning/jobs")
            .header("authorization", format!("Bearer {UPSTREAM_KEY}"))
            .json_body(json!({"model": "gpt-4o", "training_file": "file-1"}));
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({"id": "ftjob-1", "object": "fine_tuning.job"}));
    });

    let config = proxy_config(&mock.base_url());
    let handle = start_oxllm(config).await.expect("server starts");

    let resp = reqwest::Client::new()
        .post(handle.url("/v1/fine_tuning/jobs"))
        .bearer_auth(MASTER_KEY)
        .json(&json!({"model": "gpt-4o", "training_file": "file-1"}))
        .send()
        .await
        .expect("request succeeds");

    assert_eq!(resp.status(), 200);
    assert_eq!(_m.hits(), 1);
    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn fine_tuning_list_forwards_query() {
    let mock = MockServer::start();
    let _m = mock.mock(|when, then| {
        when.method(httpmock::Method::GET)
            .path("/fine_tuning/jobs")
            .query_param("limit", "5")
            .header("authorization", format!("Bearer {UPSTREAM_KEY}"));
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({"data": [], "object": "list"}));
    });

    let config = proxy_config(&mock.base_url());
    let handle = start_oxllm(config).await.expect("server starts");

    let resp = reqwest::Client::new()
        .get(handle.url("/v1/fine_tuning/jobs?limit=5"))
        .bearer_auth(MASTER_KEY)
        .send()
        .await
        .expect("request succeeds");

    assert_eq!(resp.status(), 200);
    assert_eq!(_m.hits(), 1);
    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn fine_tuning_retrieve_includes_id_in_path() {
    let mock = MockServer::start();
    let _m = mock.mock(|when, then| {
        when.method(httpmock::Method::GET)
            .path("/fine_tuning/jobs/ftjob-42")
            .header("authorization", format!("Bearer {UPSTREAM_KEY}"));
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({"id": "ftjob-42", "object": "fine_tuning.job"}));
    });

    let config = proxy_config(&mock.base_url());
    let handle = start_oxllm(config).await.expect("server starts");

    let resp = reqwest::Client::new()
        .get(handle.url("/v1/fine_tuning/jobs/ftjob-42"))
        .bearer_auth(MASTER_KEY)
        .send()
        .await
        .expect("request succeeds");

    assert_eq!(resp.status(), 200);
    assert_eq!(_m.hits(), 1);
    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn fine_tuning_cancel_posts_to_cancel_path() {
    let mock = MockServer::start();
    let _m = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/fine_tuning/jobs/ftjob-99/cancel")
            .header("authorization", format!("Bearer {UPSTREAM_KEY}"));
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({"id": "ftjob-99", "status": "cancelled"}));
    });

    let config = proxy_config(&mock.base_url());
    let handle = start_oxllm(config).await.expect("server starts");

    let resp = reqwest::Client::new()
        .post(handle.url("/v1/fine_tuning/jobs/ftjob-99/cancel"))
        .bearer_auth(MASTER_KEY)
        .send()
        .await
        .expect("request succeeds");

    assert_eq!(resp.status(), 200);
    assert_eq!(_m.hits(), 1);
    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn fine_tuning_requires_auth() {
    let mock = MockServer::start();
    let config = proxy_config(&mock.base_url());
    let handle = start_oxllm(config).await.expect("server starts");

    let resp = reqwest::Client::new()
        .get(handle.url("/v1/fine_tuning/jobs"))
        .send()
        .await
        .expect("request succeeds");

    assert_eq!(resp.status(), 401);
    let _ = handle.shutdown_tx.send(());
}

// ---------------------------------------------------------------------------
// Assistants
// ---------------------------------------------------------------------------

#[tokio::test]
async fn assistants_create_forwards_to_assistants_path() {
    let mock = MockServer::start();
    let _m = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/assistants")
            .header("authorization", format!("Bearer {UPSTREAM_KEY}"))
            .json_body(json!({"model": "gpt-4o", "name": "Helper"}));
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({"id": "asst-1", "object": "assistant"}));
    });

    let config = proxy_config(&mock.base_url());
    let handle = start_oxllm(config).await.expect("server starts");

    let resp = reqwest::Client::new()
        .post(handle.url("/v1/assistants"))
        .bearer_auth(MASTER_KEY)
        .json(&json!({"model": "gpt-4o", "name": "Helper"}))
        .send()
        .await
        .expect("request succeeds");

    assert_eq!(resp.status(), 200);
    assert_eq!(_m.hits(), 1);
    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn assistants_list_forwards_query() {
    let mock = MockServer::start();
    let _m = mock.mock(|when, then| {
        when.method(httpmock::Method::GET)
            .path("/assistants")
            .query_param("limit", "20")
            .header("authorization", format!("Bearer {UPSTREAM_KEY}"));
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({"data": [], "object": "list"}));
    });

    let config = proxy_config(&mock.base_url());
    let handle = start_oxllm(config).await.expect("server starts");

    let resp = reqwest::Client::new()
        .get(handle.url("/v1/assistants?limit=20"))
        .bearer_auth(MASTER_KEY)
        .send()
        .await
        .expect("request succeeds");

    assert_eq!(resp.status(), 200);
    assert_eq!(_m.hits(), 1);
    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn assistants_retrieve_includes_id_in_path() {
    let mock = MockServer::start();
    let _m = mock.mock(|when, then| {
        when.method(httpmock::Method::GET)
            .path("/assistants/asst-42")
            .header("authorization", format!("Bearer {UPSTREAM_KEY}"));
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({"id": "asst-42", "object": "assistant"}));
    });

    let config = proxy_config(&mock.base_url());
    let handle = start_oxllm(config).await.expect("server starts");

    let resp = reqwest::Client::new()
        .get(handle.url("/v1/assistants/asst-42"))
        .bearer_auth(MASTER_KEY)
        .send()
        .await
        .expect("request succeeds");

    assert_eq!(resp.status(), 200);
    assert_eq!(_m.hits(), 1);
    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn assistants_modify_posts_to_assistant_path() {
    let mock = MockServer::start();
    let _m = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/assistants/asst-7")
            .header("authorization", format!("Bearer {UPSTREAM_KEY}"));
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({"id": "asst-7", "object": "assistant"}));
    });

    let config = proxy_config(&mock.base_url());
    let handle = start_oxllm(config).await.expect("server starts");

    let resp = reqwest::Client::new()
        .post(handle.url("/v1/assistants/asst-7"))
        .bearer_auth(MASTER_KEY)
        .json(&json!({"name": "Updated"}))
        .send()
        .await
        .expect("request succeeds");

    assert_eq!(resp.status(), 200);
    assert_eq!(_m.hits(), 1);
    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn assistants_delete_uses_delete_method() {
    let mock = MockServer::start();
    let _m = mock.mock(|when, then| {
        when.method(httpmock::Method::DELETE)
            .path("/assistants/asst-9")
            .header("authorization", format!("Bearer {UPSTREAM_KEY}"));
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({"id": "asst-9", "deleted": true, "object": "assistant.deleted"}));
    });

    let config = proxy_config(&mock.base_url());
    let handle = start_oxllm(config).await.expect("server starts");

    let resp = reqwest::Client::new()
        .delete(handle.url("/v1/assistants/asst-9"))
        .bearer_auth(MASTER_KEY)
        .send()
        .await
        .expect("request succeeds");

    assert_eq!(resp.status(), 200);
    assert_eq!(_m.hits(), 1);
    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn assistants_requires_auth() {
    let mock = MockServer::start();
    let config = proxy_config(&mock.base_url());
    let handle = start_oxllm(config).await.expect("server starts");

    let resp = reqwest::Client::new()
        .get(handle.url("/v1/assistants"))
        .send()
        .await
        .expect("request succeeds");

    assert_eq!(resp.status(), 401);
    let _ = handle.shutdown_tx.send(());
}

// ---------------------------------------------------------------------------
// Threads
// ---------------------------------------------------------------------------

#[tokio::test]
async fn threads_create_forwards_to_threads_path() {
    let mock = MockServer::start();
    let _m = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/threads")
            .header("authorization", format!("Bearer {UPSTREAM_KEY}"));
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({"id": "thread_1", "object": "thread"}));
    });

    let config = proxy_config(&mock.base_url());
    let handle = start_oxllm(config).await.expect("server starts");

    let resp = reqwest::Client::new()
        .post(handle.url("/v1/threads"))
        .bearer_auth(MASTER_KEY)
        .json(&json!({"messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .expect("request succeeds");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("json body");
    assert_eq!(body["id"], "thread_1");
    assert_eq!(_m.hits(), 1);
    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn threads_retrieve_includes_id_in_path() {
    let mock = MockServer::start();
    let _m = mock.mock(|when, then| {
        when.method(httpmock::Method::GET)
            .path("/threads/thread_42")
            .header("authorization", format!("Bearer {UPSTREAM_KEY}"));
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({"id": "thread_42", "object": "thread"}));
    });

    let config = proxy_config(&mock.base_url());
    let handle = start_oxllm(config).await.expect("server starts");

    let resp = reqwest::Client::new()
        .get(handle.url("/v1/threads/thread_42"))
        .bearer_auth(MASTER_KEY)
        .send()
        .await
        .expect("request succeeds");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("json body");
    assert_eq!(body["id"], "thread_42");
    assert_eq!(_m.hits(), 1);
    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn threads_modify_posts_to_thread_path() {
    let mock = MockServer::start();
    let _m = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/threads/thread_7")
            .header("authorization", format!("Bearer {UPSTREAM_KEY}"))
            .json_body(json!({"metadata": {"foo": "bar"}}));
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({"id": "thread_7", "object": "thread"}));
    });

    let config = proxy_config(&mock.base_url());
    let handle = start_oxllm(config).await.expect("server starts");

    let resp = reqwest::Client::new()
        .post(handle.url("/v1/threads/thread_7"))
        .bearer_auth(MASTER_KEY)
        .json(&json!({"metadata": {"foo": "bar"}}))
        .send()
        .await
        .expect("request succeeds");

    assert_eq!(resp.status(), 200);
    assert_eq!(_m.hits(), 1);
    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn threads_delete_uses_delete_method() {
    let mock = MockServer::start();
    let _m = mock.mock(|when, then| {
        when.method(httpmock::Method::DELETE)
            .path("/threads/thread_9")
            .header("authorization", format!("Bearer {UPSTREAM_KEY}"));
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({"id": "thread_9", "deleted": true, "object": "thread.deleted"}));
    });

    let config = proxy_config(&mock.base_url());
    let handle = start_oxllm(config).await.expect("server starts");

    let resp = reqwest::Client::new()
        .delete(handle.url("/v1/threads/thread_9"))
        .bearer_auth(MASTER_KEY)
        .send()
        .await
        .expect("request succeeds");

    assert_eq!(resp.status(), 200);
    assert_eq!(_m.hits(), 1);
    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn threads_create_message_forwards_to_messages_path() {
    let mock = MockServer::start();
    let _m = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/threads/thread_1/messages")
            .header("authorization", format!("Bearer {UPSTREAM_KEY}"))
            .json_body(json!({"role": "user", "content": "hello"}));
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({"id": "msg_1", "object": "thread.message"}));
    });

    let config = proxy_config(&mock.base_url());
    let handle = start_oxllm(config).await.expect("server starts");

    let resp = reqwest::Client::new()
        .post(handle.url("/v1/threads/thread_1/messages"))
        .bearer_auth(MASTER_KEY)
        .json(&json!({"role": "user", "content": "hello"}))
        .send()
        .await
        .expect("request succeeds");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("json body");
    assert_eq!(body["id"], "msg_1");
    assert_eq!(_m.hits(), 1);
    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn threads_list_messages_forwards_query_string() {
    let mock = MockServer::start();
    let _m = mock.mock(|when, then| {
        when.method(httpmock::Method::GET)
            .path("/threads/thread_1/messages")
            .query_param("limit", "5")
            .header("authorization", format!("Bearer {UPSTREAM_KEY}"));
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({"data": [], "object": "list"}));
    });

    let config = proxy_config(&mock.base_url());
    let handle = start_oxllm(config).await.expect("server starts");

    let resp = reqwest::Client::new()
        .get(handle.url("/v1/threads/thread_1/messages?limit=5"))
        .bearer_auth(MASTER_KEY)
        .send()
        .await
        .expect("request succeeds");

    assert_eq!(resp.status(), 200);
    assert_eq!(_m.hits(), 1);
    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn threads_retrieve_message_includes_ids_in_path() {
    let mock = MockServer::start();
    let _m = mock.mock(|when, then| {
        when.method(httpmock::Method::GET)
            .path("/threads/thread_1/messages/msg_42")
            .header("authorization", format!("Bearer {UPSTREAM_KEY}"));
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({"id": "msg_42", "object": "thread.message"}));
    });

    let config = proxy_config(&mock.base_url());
    let handle = start_oxllm(config).await.expect("server starts");

    let resp = reqwest::Client::new()
        .get(handle.url("/v1/threads/thread_1/messages/msg_42"))
        .bearer_auth(MASTER_KEY)
        .send()
        .await
        .expect("request succeeds");

    assert_eq!(resp.status(), 200);
    assert_eq!(_m.hits(), 1);
    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn threads_create_run_forwards_to_runs_path_with_model() {
    let mock = MockServer::start();
    let _m = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/threads/thread_1/runs")
            .header("authorization", format!("Bearer {UPSTREAM_KEY}"))
            .json_body(json!({"assistant_id": "asst_1", "model": "gpt-4o"}));
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({"id": "run_1", "object": "thread.run", "status": "queued"}));
    });

    let config = proxy_config(&mock.base_url());
    let handle = start_oxllm(config).await.expect("server starts");

    let resp = reqwest::Client::new()
        .post(handle.url("/v1/threads/thread_1/runs"))
        .bearer_auth(MASTER_KEY)
        .json(&json!({"assistant_id": "asst_1", "model": "gpt-4o"}))
        .send()
        .await
        .expect("request succeeds");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("json body");
    assert_eq!(body["id"], "run_1");
    assert_eq!(_m.hits(), 1);
    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn threads_create_run_v2_forwards_to_threads_runs_path() {
    let mock = MockServer::start();
    let _m = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/threads/runs")
            .header("authorization", format!("Bearer {UPSTREAM_KEY}"))
            .json_body(json!({"assistant_id": "asst_1", "thread_id": "thread_1"}));
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({"id": "run_1", "object": "thread.run"}));
    });

    let config = proxy_config(&mock.base_url());
    let handle = start_oxllm(config).await.expect("server starts");

    let resp = reqwest::Client::new()
        .post(handle.url("/v1/threads/runs"))
        .bearer_auth(MASTER_KEY)
        .json(&json!({"assistant_id": "asst_1", "thread_id": "thread_1"}))
        .send()
        .await
        .expect("request succeeds");

    assert_eq!(resp.status(), 200);
    assert_eq!(_m.hits(), 1);
    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn threads_cancel_run_posts_to_cancel_path() {
    let mock = MockServer::start();
    let _m = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/threads/thread_1/runs/run_9/cancel")
            .header("authorization", format!("Bearer {UPSTREAM_KEY}"));
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({"id": "run_9", "status": "cancelling"}));
    });

    let config = proxy_config(&mock.base_url());
    let handle = start_oxllm(config).await.expect("server starts");

    let resp = reqwest::Client::new()
        .post(handle.url("/v1/threads/thread_1/runs/run_9/cancel"))
        .bearer_auth(MASTER_KEY)
        .json(&json!({}))
        .send()
        .await
        .expect("request succeeds");

    assert_eq!(resp.status(), 200);
    assert_eq!(_m.hits(), 1);
    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn threads_requires_auth() {
    let mock = MockServer::start();
    let config = proxy_config(&mock.base_url());
    let handle = start_oxllm(config).await.expect("server starts");

    let resp = reqwest::Client::new()
        .post(handle.url("/v1/threads"))
        .json(&json!({}))
        .send()
        .await
        .expect("request succeeds");

    assert_eq!(resp.status(), 401);
    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn threads_run_stream_forwards_to_stream_path() {
    let mock = MockServer::start();
    let sse = "event: thread.created\ndata: {\"id\":\"thread_1\"}\n\n";
    let _m = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/threads/runs/stream")
            .header("authorization", format!("Bearer {UPSTREAM_KEY}"))
            .header("content-type", "application/json");
        then.status(200)
            .header("content-type", "text/event-stream")
            .body(sse);
    });

    let config = proxy_config(&mock.base_url());
    let handle = start_oxllm(config).await.expect("server starts");

    let resp = reqwest::Client::new()
        .post(handle.url("/v1/threads/runs/stream"))
        .bearer_auth(MASTER_KEY)
        .header("content-type", "application/json")
        .body("{\"assistant_id\":\"asst_1\",\"thread_id\":\"thread_1\"}")
        .send()
        .await
        .expect("request succeeds");

    assert_eq!(resp.status(), 200);
    let body = resp.text().await.expect("body");
    assert!(body.contains("thread.created"));
    assert_eq!(_m.hits(), 1);
    let _ = handle.shutdown_tx.send(());
}

// ---------------------------------------------------------------------------
// CRITICAL: path-traversal / SSRF rejection on `{id}`-carrying routes
// ---------------------------------------------------------------------------

/// Every `{id}`-carrying platform route must reject a client-controlled path
/// segment that would escape the deployment's base prefix once Axum decodes
/// `%2F` to `/`. Asserts a 400 (handler allowlist) and that the pooled
/// credential never reaches an arbitrary upstream path.
#[tokio::test]
async fn platform_routes_reject_path_traversal_in_id_segment() {
    let mock = MockServer::start();
    // Catch-all that would record a hit if any traversal payload escaped past
    // the validator and reached the upstream host root.
    let escape = mock.mock(|when, then| {
        when.method(httpmock::Method::GET)
            .path_matches(regex::Regex::new(r"^/users(/.*)?$|organizations").unwrap());
        then.status(200).json_body(json!({"escaped": true}));
    });
    let escape_post = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path_matches(regex::Regex::new(r"^/users(/.*)?$").unwrap());
        then.status(200).json_body(json!({"escaped": true}));
    });
    let escape_delete = mock.mock(|when, then| {
        when.method(httpmock::Method::DELETE)
            .path_matches(regex::Regex::new(r"^/users(/.*)?$").unwrap());
        then.status(200).json_body(json!({"escaped": true}));
    });

    let config = proxy_config(&mock.base_url());
    let handle = start_oxllm(config).await.expect("server starts");
    let client = reqwest::Client::new();

    // (method, path with an encoded-traversal `{id}`, optional POST body)
    let cases: &[(reqwest::Method, &str, Option<&str>)] = &[
        (reqwest::Method::GET, "/v1/batches/..%2f..%2fusers", None),
        (
            reqwest::Method::POST,
            "/v1/batches/..%2f..%2fusers/cancel",
            Some("{}"),
        ),
        (
            reqwest::Method::GET,
            "/v1/files/..%2f..%2forganizations",
            None,
        ),
        (reqwest::Method::DELETE, "/v1/files/..%2f..%2fusers", None),
        (
            reqwest::Method::GET,
            "/v1/files/..%2f..%2fusers/content",
            None,
        ),
        (
            reqwest::Method::GET,
            "/v1/fine_tuning/jobs/..%2f..%2fusers",
            None,
        ),
        (
            reqwest::Method::POST,
            "/v1/fine_tuning/jobs/..%2f..%2fusers/cancel",
            Some("{}"),
        ),
        (reqwest::Method::GET, "/v1/assistants/..%2f..%2fusers", None),
        (
            reqwest::Method::POST,
            "/v1/assistants/..%2f..%2fusers",
            Some("{}"),
        ),
        (
            reqwest::Method::DELETE,
            "/v1/assistants/..%2f..%2fusers",
            None,
        ),
        (reqwest::Method::GET, "/v1/threads/..%2f..%2fusers", None),
        (
            reqwest::Method::GET,
            "/anthropic/v1/files/..%2f..%2fusers",
            None,
        ),
        (
            reqwest::Method::GET,
            "/v1/messages/batches/..%2f..%2fusers",
            None,
        ),
    ];

    for (method, path, body) in cases {
        let mut req = client
            .request(method.clone(), handle.url(path))
            .bearer_auth(MASTER_KEY);
        if let Some(b) = body {
            req = req.header("content-type", "application/json").body(*b);
        }
        let resp = req.send().await.expect("request succeeds");
        let status = resp.status();
        assert_eq!(
            status,
            reqwest::StatusCode::BAD_REQUEST,
            "expected 400 for traversal on {method} {path}, got {status}"
        );
    }

    assert_eq!(
        escape.hits(),
        0,
        "no GET traversal payload reached upstream"
    );
    assert_eq!(
        escape_post.hits(),
        0,
        "no POST traversal payload reached upstream"
    );
    assert_eq!(
        escape_delete.hits(),
        0,
        "no DELETE traversal payload reached upstream"
    );
    let _ = handle.shutdown_tx.send(());
}

// A literal dot-segment (`..`) cannot be delivered over HTTP by a
// spec-compliant client: the URL parser normalizes `/v1/batches/..` to
// `/v1/` before the request is sent, so the route layer never observes it
// (the request 404s as an unknown path). The protocol-level guarantee is
// the encoded form (`%2e%2e`), which IS observable and is covered by
// `platform_routes_reject_path_traversal_in_id_segment` above. Keeping a test
// here that asserts 400 would be pinning behavior no client can trigger.

/// Threads multi-segment routes must reject a decoded `%2f` in a path
/// parameter (`foo%2fbar` → `foo/bar`) before it is formatted into the
/// upstream path. Before the fix the per-segment split-after-format approach
/// allowed each split segment to pass the allowlist individually, injecting
/// an extra path segment. Now each raw `Path<String>` is validated before
/// formatting, matching every sibling handler.
#[tokio::test]
async fn threads_routes_reject_decoded_slash_in_message_id() {
    let mock = MockServer::start();
    let escape = mock.mock(|when, then| {
        when.method(httpmock::Method::GET)
            .path_matches(regex::Regex::new(r"^/threads/.+/messages/.+").unwrap());
        then.status(200).json_body(json!({"escaped": true}));
    });

    let config = proxy_config(&mock.base_url());
    let handle = start_oxllm(config).await.expect("server starts");

    let resp = reqwest::Client::new()
        .get(handle.url("/v1/threads/valid_thread/messages/foo%2fbar"))
        .bearer_auth(MASTER_KEY)
        .send()
        .await
        .expect("request succeeds");

    assert_eq!(
        resp.status(),
        reqwest::StatusCode::BAD_REQUEST,
        "decoded %2f in message_id must be rejected by validate_path_param"
    );
    assert_eq!(
        escape.hits(),
        0,
        "no decoded-slash payload must reach upstream"
    );
    let _ = handle.shutdown_tx.send(());
}

/// Threads multi-segment runs routes must also reject decoded `%2f` in
/// `run_id`.
#[tokio::test]
async fn threads_routes_reject_decoded_slash_in_run_id() {
    let mock = MockServer::start();
    let escape = mock.mock(|when, then| {
        when.method(httpmock::Method::GET);
        then.status(200).json_body(json!({"escaped": true}));
    });

    let config = proxy_config(&mock.base_url());
    let handle = start_oxllm(config).await.expect("server starts");

    let resp = reqwest::Client::new()
        .get(handle.url("/v1/threads/valid_thread/runs/foo%2fbar"))
        .bearer_auth(MASTER_KEY)
        .send()
        .await
        .expect("request succeeds");

    assert_eq!(
        resp.status(),
        reqwest::StatusCode::BAD_REQUEST,
        "decoded %2f in run_id must be rejected by validate_path_param"
    );
    assert_eq!(escape.hits(), 0);
    let _ = handle.shutdown_tx.send(());
}

// ---------------------------------------------------------------------------
// MEDIUM: Anthropic platform endpoints forward `anthropic-beta`
// ---------------------------------------------------------------------------

#[tokio::test]
async fn anthropic_batches_forwards_vetted_beta_header() {
    let mock = MockServer::start();
    let _m = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/messages/batches")
            .header("anthropic-beta", "prompt-caching-2024-07-31");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({"id": "msgbatch_1", "type": "message_batch"}));
    });

    // The endpoint requires an Anthropic deployment (model-less platform
    // request with an `anthropic` provider hint must not fall back to the
    // OpenAI deployment in a multi-provider config).
    let config = multi_provider_config(
        &mock.base_url(),
        UPSTREAM_KEY,
        &mock.base_url(),
        UPSTREAM_KEY,
    );
    let handle = start_oxllm(config).await.expect("server starts");

    let resp = reqwest::Client::new()
        .post(handle.url("/v1/messages/batches"))
        .bearer_auth(MASTER_KEY)
        .header("anthropic-beta", "prompt-caching-2024-07-31")
        .header("content-type", "application/json")
        .body(r#"{"requests":[]}"#)
        .send()
        .await
        .expect("request succeeds");

    assert_eq!(resp.status(), 200);
    assert_eq!(_m.hits(), 1, "anthropic-beta must be forwarded upstream");
    let _ = handle.shutdown_tx.send(());
}

/// LOW (C25): the Anthropic Message Batches API has a fifth endpoint —
/// `GET /v1/messages/batches/{batch_id}/results` — that streams batch results
/// as Anthropic JSONL. The reference supports it (`llms/anthropic/files/
/// handler.py:94`). oxllm previously registered create/list/retrieve/cancel
/// but not results; a client requesting the results URL fell through to the
/// catch-all. This test verifies the route is now registered and relays the
/// JSONL body with the upstream content-type.
#[tokio::test]
async fn anthropic_batches_results_relays_jsonl() {
    let mock = MockServer::start();
    let jsonl = "{\"type\":\"custom\",\"batch_id\":\"msgbatch_1\"}\n\
                 {\"type\":\"succeeded\",\"result\":{\"type\":\"message\"}}\n";
    let _m = mock.mock(|when, then| {
        when.method(httpmock::Method::GET)
            .path("/messages/batches/msgbatch_1/results")
            .header("x-api-key", UPSTREAM_KEY);
        then.status(200)
            .header("content-type", "application/x-jsonlines")
            .body(jsonl);
    });

    let config = multi_provider_config(
        &mock.base_url(),
        UPSTREAM_KEY,
        &mock.base_url(),
        UPSTREAM_KEY,
    );
    let handle = start_oxllm(config).await.expect("server starts");

    let resp = reqwest::Client::new()
        .get(handle.url("/v1/messages/batches/msgbatch_1/results"))
        .bearer_auth(MASTER_KEY)
        .send()
        .await
        .expect("request succeeds");

    assert_eq!(resp.status(), 200);
    assert_eq!(
        resp.headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or(""),
        "application/x-jsonlines",
        "results content-type must be relayed from upstream"
    );
    let body = resp.text().await.expect("body");
    assert_eq!(body, jsonl, "JSONL body must be relayed verbatim");
    assert_eq!(_m.hits(), 1, "upstream /results must be hit exactly once");
    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn anthropic_files_forwards_vetted_beta_header() {
    let mock = MockServer::start();
    let _m = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/files")
            .header("anthropic-beta", "prompt-caching-2024-07-31");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({"id": "file_1", "type": "file"}));
    });

    let config = multi_provider_config(
        &mock.base_url(),
        UPSTREAM_KEY,
        &mock.base_url(),
        UPSTREAM_KEY,
    );
    let handle = start_oxllm(config).await.expect("server starts");

    let resp = reqwest::Client::new()
        .post(handle.url("/anthropic/v1/files"))
        .bearer_auth(MASTER_KEY)
        .header("anthropic-beta", "prompt-caching-2024-07-31")
        .header("content-type", "multipart/form-data; boundary=xyz")
        .body("--xyz--\r\n")
        .send()
        .await
        .expect("request succeeds");

    assert_eq!(resp.status(), 200);
    assert_eq!(_m.hits(), 1, "anthropic-beta must be forwarded upstream");
    let _ = handle.shutdown_tx.send(());
}

// ---------------------------------------------------------------------------
// MEDIUM: multi-provider config — model-less requests must route to a
// deployment of the endpoint's intended provider, not `deployments.first()`.
// ---------------------------------------------------------------------------

fn multi_provider_config(
    anthropic_base: &str,
    anthropic_key: &str,
    openai_base: &str,
    openai_key: &str,
) -> ProxyConfig {
    let anthropic = ConfigModel {
        model_name: "claude".to_string(),
        oxllm_params: OxllmParams {
            model: "anthropic/claude-3-5-sonnet".to_string(),
            api_key: Some(anthropic_key.to_string()),
            api_base: Some(anthropic_base.to_string()),
            timeout: Some(30.0),
            ..Default::default()
        },
        ..Default::default()
    };
    let openai = ConfigModel {
        model_name: "gpt-4o".to_string(),
        oxllm_params: OxllmParams {
            model: "openai/gpt-4o".to_string(),
            api_key: Some(openai_key.to_string()),
            api_base: Some(openai_base.to_string()),
            timeout: Some(30.0),
            ..Default::default()
        },
        ..Default::default()
    };
    // Anthropic first on purpose — the previous `deployments.first()` fallback
    // would route every model-less OpenAI platform request to the Anthropic
    // upstream (and vice-versa for `/anthropic/v1/files`).
    ProxyConfig {
        model_list: vec![anthropic, openai],
        oxllm_settings: OxllmSettings::default(),
        general_settings: GeneralSettings {
            master_key: Some(MASTER_KEY.to_string()),
            allow_private_api_base: Some(true),
            ..Default::default()
        },
        router_settings: Default::default(),
        extra: serde_json::Map::new(),
    }
}

/// `GET /v1/batches` with no `?model=` must hit the OpenAI upstream even when
/// an Anthropic deployment is listed first in the config. Before the fix the
/// handler resolved the model to `claude` (the first deployment's
/// `model_name`), causing the request to be sent to the Anthropic host with
/// Anthropic credentials.
#[tokio::test]
async fn openai_batches_routes_to_openai_in_multi_provider_config() {
    let anthropic_mock = MockServer::start();
    let openai_mock = MockServer::start();

    // Sentinel upstreams: each answers only when called with its own key, so
    // a hit on the wrong upstream is detectable as a 401 / unmatched mock.
    let openai_upstream = openai_mock.mock(|when, then| {
        when.method(httpmock::Method::GET)
            .path("/batches")
            .header("authorization", "Bearer sk-openai-key");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({"data": [], "object": "list", "routed_to": "openai"}));
    });
    let anthropic_upstream = anthropic_mock.mock(|when, then| {
        when.method(httpmock::Method::GET).path("/batches");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({"data": [], "object": "list", "routed_to": "anthropic"}));
    });

    let config = multi_provider_config(
        &anthropic_mock.base_url(),
        "sk-anthropic-key",
        &openai_mock.base_url(),
        "sk-openai-key",
    );
    let handle = start_oxllm(config).await.expect("server starts");

    let resp = reqwest::Client::new()
        .get(handle.url("/v1/batches"))
        .bearer_auth(MASTER_KEY)
        .send()
        .await
        .expect("request succeeds");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("json body");
    assert_eq!(
        body["routed_to"], "openai",
        "model-less GET /v1/batches must route to the OpenAI upstream, not Anthropic"
    );
    assert_eq!(openai_upstream.hits(), 1);
    assert_eq!(
        anthropic_upstream.hits(),
        0,
        "Anthropic upstream must not be hit by an OpenAI batches call"
    );
    let _ = handle.shutdown_tx.send(());
}

/// `POST /anthropic/v1/files` (multipart upload, no body model) must hit the
/// Anthropic upstream even when an OpenAI deployment is listed first. Before
/// the fix the handler resolved the model to `gpt-4o` and routed the file
/// bytes to `api.openai.com` with OpenAI credentials — a confidentiality
/// concern.
#[tokio::test]
async fn anthropic_files_routes_to_anthropic_in_multi_provider_config() {
    let anthropic_mock = MockServer::start();
    let openai_mock = MockServer::start();

    let anthropic_upstream = anthropic_mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/files")
            .header("x-api-key", "sk-anthropic-key");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({"id": "file_1", "type": "file", "routed_to": "anthropic"}));
    });
    let openai_upstream = openai_mock.mock(|when, then| {
        when.method(httpmock::Method::POST).path("/files");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({"id": "file_1", "object": "file", "routed_to": "openai"}));
    });

    // OpenAI first on purpose — proves the `anthropic` provider hint overrides
    // the deployment order for the `/anthropic/*` surface.
    let openai = ConfigModel {
        model_name: "gpt-4o".to_string(),
        oxllm_params: OxllmParams {
            model: "openai/gpt-4o".to_string(),
            api_key: Some("sk-openai-key".to_string()),
            api_base: Some(openai_mock.base_url()),
            timeout: Some(30.0),
            ..Default::default()
        },
        ..Default::default()
    };
    let anthropic = ConfigModel {
        model_name: "claude".to_string(),
        oxllm_params: OxllmParams {
            model: "anthropic/claude-3-5-sonnet".to_string(),
            api_key: Some("sk-anthropic-key".to_string()),
            api_base: Some(anthropic_mock.base_url()),
            timeout: Some(30.0),
            ..Default::default()
        },
        ..Default::default()
    };
    let config = ProxyConfig {
        model_list: vec![openai, anthropic],
        oxllm_settings: OxllmSettings::default(),
        general_settings: GeneralSettings {
            master_key: Some(MASTER_KEY.to_string()),
            allow_private_api_base: Some(true),
            ..Default::default()
        },
        router_settings: Default::default(),
        extra: serde_json::Map::new(),
    };
    let handle = start_oxllm(config).await.expect("server starts");

    let resp = reqwest::Client::new()
        .post(handle.url("/anthropic/v1/files"))
        .bearer_auth(MASTER_KEY)
        .header("content-type", "multipart/form-data; boundary=xyz")
        .body("--xyz--\r\n")
        .send()
        .await
        .expect("request succeeds");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("json body");
    assert_eq!(
        body["routed_to"], "anthropic",
        "model-less POST /anthropic/v1/files must route to the Anthropic upstream, not OpenAI"
    );
    assert_eq!(anthropic_upstream.hits(), 1);
    assert_eq!(
        openai_upstream.hits(),
        0,
        "OpenAI upstream must not be hit by an Anthropic files call"
    );
    let _ = handle.shutdown_tx.send(());
}
