//! Integration tests: DB persistence, virtual keys, and management REST API.

mod common;

use std::sync::Arc;

use common::OxllmHandle;
use httpmock::MockServer;
use oxllm_server::app::{build_router, build_shared_state_with_db};
use oxllm_server::config::types::{ConfigModel, GeneralSettings, OxllmSettings, ProxyConfig};
use oxllm_server::spend_writer;
use oxllm_server::{Database, ModelConfigRecord, OxllmParams, SharedState, SpendFilter};
use serde_json::{Value, json};

const MASTER_KEY: &str = "sk-master-test-key";

fn proxy_config(mock_base_url: &str) -> ProxyConfig {
    ProxyConfig {
        model_list: vec![
            ConfigModel {
                model_name: "gpt-4o".to_string(),
                oxllm_params: OxllmParams {
                    model: "openai/gpt-4o".to_string(),
                    api_key: Some("sk-upstream".to_string()),
                    api_base: Some(mock_base_url.to_string()),
                    timeout: Some(30.0),
                    stream_timeout: Some(30),
                    ..Default::default()
                },
                ..Default::default()
            },
            ConfigModel {
                model_name: "claude-3".to_string(),
                oxllm_params: OxllmParams {
                    model: "anthropic/claude-3".to_string(),
                    api_key: Some("sk-upstream".to_string()),
                    api_base: Some(mock_base_url.to_string()),
                    timeout: Some(30.0),
                    stream_timeout: Some(30),
                    ..Default::default()
                },
                ..Default::default()
            },
        ],
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

async fn fresh_db() -> Arc<Database> {
    let db = Database::connect("sqlite::memory:")
        .await
        .expect("connect sqlite memory");
    db.migrate().await.expect("migrate");
    db
}

#[tokio::test]
async fn key_scope_references_must_exist_before_a_key_is_created() {
    let mock = MockServer::start();
    let (handle, db) = start_oxllm(proxy_config(&mock.base_url())).await;
    for scope in ["team_id", "user_id", "org_id", "project_id"] {
        let response = common::http_client()
            .post(handle.url("/key/generate"))
            .bearer_auth(MASTER_KEY)
            .json(&json!({(scope):"missing-scope"}))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 400, "{scope}");
        assert!(db.list_keys(None).await.unwrap().is_empty());
    }
    shutdown(handle).await;
}

#[tokio::test]
async fn parallel_limits_reject_invalid_values_and_zero_blocks_admission() {
    let mock = MockServer::start();
    let upstream = mock_chat(&mock);
    let (handle, db) = start_oxllm(proxy_config(&mock.base_url())).await;
    let client = common::http_client();
    for value in [-1_i64, i64::from(u32::MAX) + 1] {
        for (path, body) in [
            (
                "/organization/add",
                json!({"org_name":"invalid","max_parallel_requests":value}),
            ),
            (
                "/organization/update",
                json!({"org_id":"missing","max_parallel_requests":value}),
            ),
            ("/budget/new", json!({"max_parallel_requests":value})),
            (
                "/budget/update",
                json!({"budget_id":"missing","max_parallel_requests":value}),
            ),
        ] {
            let response = client
                .post(handle.url(path))
                .bearer_auth(MASTER_KEY)
                .json(&body)
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), 400, "{path}: {value}");
            assert!(
                response
                    .text()
                    .await
                    .unwrap()
                    .contains("max_parallel_requests")
            );
        }
    }
    assert!(db.list_organizations(None).await.unwrap().is_empty());
    let response = client
        .post(handle.url("/organization/add"))
        .bearer_auth(MASTER_KEY)
        .json(&json!({"org_id":"paused-org","org_name":"Paused","max_parallel_requests":0}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let response = client
        .post(handle.url("/key/generate"))
        .bearer_auth(MASTER_KEY)
        .json(&json!({"org_id":"paused-org"}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let key: Value = response.json().await.unwrap();
    let response = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth(key["key"].as_str().unwrap())
        .json(&json!({"model":"gpt-4o","messages":[{"role":"user","content":"hello"}]}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 429);
    upstream.assert_hits(0);
    shutdown(handle).await;
}

#[tokio::test]
async fn model_updates_reject_unloadable_and_shadowing_changes_without_mutating_storage() {
    let mock = MockServer::start();
    let upstream = mock_chat(&mock);
    let (handle, db) = start_oxllm(proxy_config(&mock.base_url())).await;
    let response = common::http_client()
        .post(handle.url("/model/new"))
        .bearer_auth(MASTER_KEY)
        .json(&json!({"model_name":"dynamic-model","oxllm_params":{
            "model":"openai/gpt-4o","api_key":"sk-upstream","api_base":mock.base_url()
        }}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let id = response.json::<Value>().await.unwrap()["model_id"]
        .as_str()
        .unwrap()
        .to_string();
    let original = db.get_model(&id).await.unwrap().unwrap();
    for patch in [
        json!({"model_id":id,"oxllm_params":{"timeout":"invalid"}}),
        json!({"model_id":id,"model_name":"gpt-4o"}),
    ] {
        let response = common::http_client()
            .post(handle.url("/model/update"))
            .bearer_auth(MASTER_KEY)
            .json(&patch)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 400);
        let current = db.get_model(&id).await.unwrap().unwrap();
        assert_eq!(current.model_name, original.model_name);
        assert_eq!(current.oxllm_params, original.oxllm_params);
    }
    let response = common::http_client()
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth(MASTER_KEY)
        .json(&json!({"model":"dynamic-model","messages":[{"role":"user","content":"hi"}]}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    upstream.assert_hits(1);
    shutdown(handle).await;
}

/// DB-backed variant of [`common::start_oxllm`]: boots the proxy with a
/// fresh in-memory SQLite database and the spend writer task. Returns the
/// handle plus the database for direct inspection.
/// Test-owned proxy handle: the [`OxllmHandle`] (address + Axum shutdown
/// signal) plus the serve task's `JoinHandle` and the spend writer's
/// [`spend_writer::SpendWriterHandle`], so [`shutdown`] can wait for
/// graceful request drain and a deterministic final spend flush instead of
/// relying on flush-ticker timing.
struct Handle {
    inner: OxllmHandle,
    server: tokio::task::JoinHandle<()>,
    writer: spend_writer::SpendWriterHandle,
    shared: SharedState,
}

impl Handle {
    fn url(&self, path: &str) -> String {
        self.inner.url(path)
    }
}

async fn start_oxllm(config: ProxyConfig) -> (Handle, Arc<Database>) {
    let db = fresh_db().await;
    let (writer, writer_handle) = spend_writer::spawn_with_shutdown(db.clone(), 1, 0);
    let shared: SharedState =
        build_shared_state_with_db(config, Some(writer.sender()), Some(db.clone()))
            .expect("build state");
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let addr = listener.local_addr().expect("addr");
    let router = build_router(shared.clone());
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    let server = tokio::spawn(async move {
        axum::serve(listener, router)
            .with_graceful_shutdown(async move {
                let _ = shutdown_rx.await;
            })
            .await
            .ok();
    });
    (
        Handle {
            inner: OxllmHandle { addr, shutdown_tx },
            server,
            writer: writer_handle,
            shared,
        },
        db,
    )
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

/// Shut the test proxy down: signal Axum's graceful shutdown and wait (up
/// to 5s) for in-flight requests to drain, then stop the spend writer with
/// a bounded final flush so pending spend rows are persisted before
/// subsequent state assertions run.
async fn shutdown(handle: Handle) {
    let _ = handle.inner.shutdown_tx.send(());
    let _ = tokio::time::timeout(std::time::Duration::from_secs(5), handle.server).await;
    handle.writer.shutdown().await;
}

/// Poll an async condition every 50ms for up to 5s. Replaces fixed sleeps on
/// ~1s flush windows so slow CI runners cannot flake the assertions.
async fn wait_until<F, Fut>(what: &str, mut f: F)
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = bool>,
{
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while !f().await {
        assert!(
            std::time::Instant::now() < deadline,
            "timed out waiting for {what}"
        );
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
}

#[tokio::test]
async fn scope_lists_report_truncation_only_when_an_extra_row_exists() {
    let mock = MockServer::start();
    let (handle, db) = start_oxllm(proxy_config(&mock.base_url())).await;
    let client = reqwest::Client::new();
    let now = oxllm_server::now_iso();
    for i in 0..=1_000 {
        // Seed directly to keep this boundary regression independent of
        // thousands of HTTP create requests. Record types are inferred from
        // the repository methods (no extra integration-test dependency).
        let org = serde_json::from_value(json!({
            "org_id": format!("org-{i}"), "org_name": "Org",
            "created_at": now, "updated_at": now,
        }))
        .unwrap();
        db.create_organization(&org).await.unwrap();
        let project = serde_json::from_value(json!({
            "project_id": format!("project-{i}"), "project_name": "Project",
            "created_at": now, "updated_at": now,
        }))
        .unwrap();
        db.create_project(&project).await.unwrap();
        let customer = serde_json::from_value(json!({
            "end_user_id": format!("customer-{i}"), "spend": 0.0, "blocked": false,
            "created_at": now, "updated_at": now,
        }))
        .unwrap();
        db.create_customer(&customer).await.unwrap();
        let tag = serde_json::from_value(json!({
            "name": format!("tag-{i}"), "created_at": now, "updated_at": now,
        }))
        .unwrap();
        db.create_tag(&tag).await.unwrap();
        if i == 999 || i == 1_000 {
            for (endpoint, field) in [
                ("/organization/list", "organizations"),
                ("/project/list", "projects"),
                ("/customer/list", "customers"),
                ("/tag/list", "tags"),
            ] {
                let response = client
                    .get(handle.url(endpoint))
                    .bearer_auth(MASTER_KEY)
                    .send()
                    .await
                    .unwrap();
                assert_eq!(response.status(), 200, "{endpoint}");
                let body: Value = response.json().await.unwrap();
                assert_eq!(body[field].as_array().unwrap().len(), 1_000, "{endpoint}");
                assert_eq!(body["truncated"], i == 1_000, "{endpoint}");
            }
        }
    }
    shutdown(handle).await;
}

#[tokio::test]
async fn management_endpoints_require_master_key() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;

    let client = reqwest::Client::new();
    let resp = client
        .get(handle.url("/key/list"))
        .header("authorization", "Bearer wrong-key")
        .send()
        .await
        .expect("ok");
    assert_eq!(resp.status(), 401);

    let resp = client
        .get(handle.url("/key/list"))
        .send()
        .await
        .expect("ok");
    assert_eq!(resp.status(), 401, "missing auth header -> 401");

    shutdown(handle).await;
}

#[tokio::test]
async fn master_key_can_access_management() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;

    let client = reqwest::Client::new();
    let resp = client
        .get(handle.url("/key/list"))
        .header("authorization", format!("Bearer {MASTER_KEY}"))
        .send()
        .await
        .expect("ok");
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("json");
    assert_eq!(body["keys"].as_array().map(|a| a.len()), Some(0));

    shutdown(handle).await;
}

#[tokio::test]
async fn generate_key_then_auth_with_virtual_key() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;
    let _m = mock_chat(&mock);

    let client = reqwest::Client::new();

    let gen_resp = client
        .post(handle.url("/key/generate"))
        .header("authorization", format!("Bearer {MASTER_KEY}"))
        .json(&json!({"key_alias": "test-key"}))
        .send()
        .await
        .expect("ok");
    assert_eq!(gen_resp.status(), 200);
    let gen_body: Value = gen_resp.json().await.expect("json");
    let raw_key = gen_body["key"]
        .as_str()
        .expect("key in response")
        .to_string();
    let key_id = gen_body["key_id"]
        .as_str()
        .expect("key_id in response")
        .to_string();
    assert!(raw_key.starts_with("sk-ox-"));
    assert!(!key_id.is_empty());

    let listed = client
        .get(handle.url("/key/list"))
        .header("authorization", format!("Bearer {MASTER_KEY}"))
        .send()
        .await
        .expect("ok");
    let list_body: Value = listed.json().await.expect("json");
    let keys = list_body["keys"].as_array().expect("keys array");
    assert_eq!(keys.len(), 1);
    assert!(
        keys[0]["key_hash"]
            .as_str()
            .map(|s| s.is_empty())
            .unwrap_or(true),
        "hash must never be returned in /key/list"
    );
    assert_eq!(keys[0]["key_id"].as_str(), Some(key_id.as_str()));

    let chat = client
        .post(handle.url("/v1/chat/completions"))
        .header("authorization", format!("Bearer {raw_key}"))
        .json(&json!({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .expect("ok");
    assert_eq!(chat.status(), 200, "virtual key must authenticate");

    shutdown(handle).await;
}

#[tokio::test]
async fn deleted_key_is_rejected() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;

    let client = reqwest::Client::new();

    let gen_resp = client
        .post(handle.url("/key/generate"))
        .header("authorization", format!("Bearer {MASTER_KEY}"))
        .json(&json!({}))
        .send()
        .await
        .expect("ok");
    let gen_body: Value = gen_resp.json().await.expect("json");
    let raw_key = gen_body["key"].as_str().expect("key").to_string();
    let key_id = gen_body["key_id"].as_str().expect("key_id").to_string();

    let del = client
        .post(handle.url("/key/delete"))
        .header("authorization", format!("Bearer {MASTER_KEY}"))
        .json(&json!({"key_id": key_id}))
        .send()
        .await
        .expect("ok");
    assert_eq!(del.status(), 200);

    let chat = client
        .post(handle.url("/v1/chat/completions"))
        .header("authorization", format!("Bearer {raw_key}"))
        .json(&json!({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .expect("ok");
    assert_eq!(chat.status(), 401, "deleted key must be rejected");

    shutdown(handle).await;
}

#[tokio::test]
async fn master_key_still_works_for_chat() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;
    let _m = mock_chat(&mock);

    let client = reqwest::Client::new();
    let chat = client
        .post(handle.url("/v1/chat/completions"))
        .header("authorization", format!("Bearer {MASTER_KEY}"))
        .json(&json!({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .expect("ok");
    assert_eq!(chat.status(), 200);

    shutdown(handle).await;
}

#[tokio::test]
async fn unknown_token_is_rejected() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;

    let client = reqwest::Client::new();
    let chat = client
        .post(handle.url("/v1/chat/completions"))
        .header("authorization", "Bearer sk-ox-not-a-real-key")
        .json(&json!({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .expect("ok");
    assert_eq!(chat.status(), 401);

    shutdown(handle).await;
}

#[tokio::test]
async fn spend_logs_recorded_after_request() {
    let mock = MockServer::start();
    let (handle, db) = start_oxllm(proxy_config(&mock.base_url())).await;
    let _m = mock_chat(&mock);

    let client = reqwest::Client::new();

    let gen_resp = client
        .post(handle.url("/key/generate"))
        .header("authorization", format!("Bearer {MASTER_KEY}"))
        .json(&json!({"key_alias": "spend-key"}))
        .send()
        .await
        .expect("ok");
    let gen_body: Value = gen_resp.json().await.expect("json");
    let raw_key = gen_body["key"].as_str().expect("key").to_string();
    let key_id = gen_body["key_id"].as_str().expect("key_id").to_string();

    let chat = client
        .post(handle.url("/v1/chat/completions"))
        .header("authorization", format!("Bearer {raw_key}"))
        .json(&json!({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .expect("ok");
    assert_eq!(chat.status(), 200);

    for _ in 0..20 {
        let logs = db
            .query_spend(&SpendFilter {
                key_id: Some(key_id.clone()),
                ..Default::default()
            })
            .await
            .expect("query");
        if !logs.is_empty() {
            assert_eq!(logs[0].model, "gpt-4o");
            assert!(logs[0].cost.unwrap_or(-1.0) > 0.0);
            assert_eq!(logs[0].provider, "openai");
            shutdown(handle).await;
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    panic!("spend log never appeared after 2s");
}

#[tokio::test]
async fn spend_total_endpoint_reports_aggregate() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;
    let _m = mock_chat(&mock);

    let client = reqwest::Client::new();
    let chat = client
        .post(handle.url("/v1/chat/completions"))
        .header("authorization", format!("Bearer {MASTER_KEY}"))
        .json(&json!({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .expect("ok");
    assert_eq!(chat.status(), 200);

    let mut got = false;
    for _ in 0..20 {
        let resp = client
            .get(handle.url("/spend/total"))
            .header("authorization", format!("Bearer {MASTER_KEY}"))
            .send()
            .await
            .expect("ok");
        assert_eq!(resp.status(), 200);
        let body: Value = resp.json().await.expect("json");
        if body["total_requests"].as_u64().unwrap_or(0) >= 1 {
            assert!(
                body["total_cost"].as_f64().unwrap_or(0.0) > 0.0,
                "total_cost should be positive"
            );
            got = true;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    assert!(got, "spend never recorded");

    shutdown(handle).await;
}

#[tokio::test]
async fn spend_logs_endpoint_returns_data() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;
    let _m = mock_chat(&mock);

    let client = reqwest::Client::new();
    let chat = client
        .post(handle.url("/v1/chat/completions"))
        .header("authorization", format!("Bearer {MASTER_KEY}"))
        .json(&json!({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .expect("ok");
    assert_eq!(chat.status(), 200);

    let mut got = false;
    for _ in 0..20 {
        let resp = client
            .get(handle.url("/spend/logs"))
            .header("authorization", format!("Bearer {MASTER_KEY}"))
            .send()
            .await
            .expect("ok");
        assert_eq!(resp.status(), 200);
        let body: Value = resp.json().await.expect("json");
        if let Some(data) = body["data"].as_array() {
            if !data.is_empty() {
                assert!(body["total_cost"].as_f64().unwrap_or(0.0) > 0.0);
                assert!(body["total_requests"].as_u64().unwrap_or(0) >= 1);
                got = true;
                break;
            }
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    assert!(got, "spend logs endpoint never returned data");

    shutdown(handle).await;
}

#[tokio::test]
async fn team_and_user_crud_via_api() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;

    let client = reqwest::Client::new();
    let auth = format!("Bearer {MASTER_KEY}");

    let add_team = client
        .post(handle.url("/team/add"))
        .header("authorization", &auth)
        .json(&json!({"team_name": "Eng", "max_budget": 50.0}))
        .send()
        .await
        .expect("ok");
    assert_eq!(add_team.status(), 200);
    let tb: Value = add_team.json().await.expect("json");
    let team_id = tb["team_id"].as_str().expect("team_id").to_string();

    let info = client
        .get(handle.url(&format!("/team/info?team_id={team_id}")))
        .header("authorization", &auth)
        .send()
        .await
        .expect("ok");
    assert_eq!(info.status(), 200);
    let ib: Value = info.json().await.expect("json");
    assert_eq!(ib["team_name"].as_str(), Some("Eng"));

    let add_user = client
        .post(handle.url("/user/add"))
        .header("authorization", &auth)
        .json(&json!({"user_name": "Alice"}))
        .send()
        .await
        .expect("ok");
    assert_eq!(add_user.status(), 200);
    let ub: Value = add_user.json().await.expect("json");
    let user_id = ub["user_id"].as_str().expect("user_id").to_string();

    let list = client
        .get(handle.url("/user/list"))
        .header("authorization", &auth)
        .send()
        .await
        .expect("ok");
    let lb: Value = list.json().await.expect("json");
    assert_eq!(lb["users"].as_array().map(|a| a.len()), Some(1));
    assert_eq!(lb["users"][0]["user_id"].as_str(), Some(user_id.as_str()));

    shutdown(handle).await;
}

#[tokio::test]
async fn model_management_endpoints() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;

    let client = reqwest::Client::new();
    let auth = format!("Bearer {MASTER_KEY}");

    let list = client
        .get(handle.url("/model/list"))
        .header("authorization", &auth)
        .send()
        .await
        .expect("ok");
    assert_eq!(list.status(), 200);
    let lb: Value = list.json().await.expect("json");
    let data = lb["data"].as_array().expect("data array");
    assert_eq!(data.len(), 2);
    assert!(data.iter().any(|m| m["id"].as_str() == Some("gpt-4o")));
    assert!(data.iter().any(|m| m["id"].as_str() == Some("claude-3")));

    let info = client
        .get(handle.url("/model/info?model=gpt-4o"))
        .header("authorization", &auth)
        .send()
        .await
        .expect("ok");
    assert_eq!(info.status(), 200);
    let ib: Value = info.json().await.expect("json");
    assert_eq!(ib["model"].as_str(), Some("gpt-4o"));
    assert!(ib.get("input_cost_per_token").is_some());

    shutdown(handle).await;
}

// ---------------------------------------------------------------------------
// P0-4: per-key budget enforced even when no global limits are configured
// ---------------------------------------------------------------------------

#[tokio::test]
async fn per_key_budget_enforced_without_global_limits() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;
    let _m = mock_chat(&mock);

    let client = reqwest::Client::new();

    let gen_resp = client
        .post(handle.url("/key/generate"))
        .header("authorization", format!("Bearer {MASTER_KEY}"))
        .json(&json!({"key_alias": "budget-capped", "max_budget": 0.0001}))
        .send()
        .await
        .expect("ok");
    assert_eq!(gen_resp.status(), 200);
    let gb: Value = gen_resp.json().await.expect("json");
    let raw_key = gb["key"].as_str().expect("key").to_string();

    let first = client
        .post(handle.url("/v1/chat/completions"))
        .header("authorization", format!("Bearer {raw_key}"))
        .json(&json!({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5}))
        .send()
        .await
        .expect("ok");
    assert_eq!(
        first.status(),
        200,
        "first request under per-key budget should succeed"
    );

    let second = client
        .post(handle.url("/v1/chat/completions"))
        .header("authorization", format!("Bearer {raw_key}"))
        .json(&json!({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5}))
        .send()
        .await
        .expect("ok");
    assert_eq!(
        second.status(),
        429,
        "second request should be rejected after per-key budget exhausted"
    );

    shutdown(handle).await;
}

#[tokio::test]
async fn per_key_rpm_enforced_without_global_limits() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;
    let _m = mock_chat(&mock);

    let client = reqwest::Client::new();

    let gen_resp = client
        .post(handle.url("/key/generate"))
        .header("authorization", format!("Bearer {MASTER_KEY}"))
        .json(&json!({"key_alias": "rpm-capped", "rpm": 1}))
        .send()
        .await
        .expect("ok");
    assert_eq!(gen_resp.status(), 200);
    let gb: Value = gen_resp.json().await.expect("json");
    let raw_key = gb["key"].as_str().expect("key").to_string();

    let first = client
        .post(handle.url("/v1/chat/completions"))
        .header("authorization", format!("Bearer {raw_key}"))
        .json(&json!({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .expect("ok");
    assert_eq!(first.status(), 200, "first request should succeed");

    let second = client
        .post(handle.url("/v1/chat/completions"))
        .header("authorization", format!("Bearer {raw_key}"))
        .json(&json!({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .expect("ok");
    assert_eq!(
        second.status(),
        429,
        "second request should be rejected after per-key RPM exceeded"
    );

    shutdown(handle).await;
}

// ---------------------------------------------------------------------------
// Team limit inheritance (LiteLLM: tests/e2e/budgets/ team enforcement matrix)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn team_rpm_merged_with_key_enforced_without_global_limits() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;
    let _m = mock_chat(&mock);

    let client = reqwest::Client::new();
    let auth = format!("Bearer {MASTER_KEY}");

    let team_resp = client
        .post(handle.url("/team/add"))
        .header("authorization", &auth)
        .json(&json!({"team_name": "rpm-team", "rpm": 1}))
        .send()
        .await
        .expect("team");
    assert_eq!(team_resp.status(), 200);
    let tb: Value = team_resp.json().await.expect("json");
    let team_id = tb["team_id"].as_str().expect("team_id");

    let key_resp = client
        .post(handle.url("/key/generate"))
        .header("authorization", &auth)
        .json(&json!({
        "key_alias": "team-rpm-key",
        "team_id": team_id,
        "rpm": 100
        }))
        .send()
        .await
        .expect("key");
    assert_eq!(key_resp.status(), 200);
    let kb: Value = key_resp.json().await.expect("json");
    let raw_key = kb["key"].as_str().expect("key");

    let body = json!({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]});

    let first = client
        .post(handle.url("/v1/chat/completions"))
        .header("authorization", format!("Bearer {raw_key}"))
        .json(&body)
        .send()
        .await
        .expect("ok");
    assert_eq!(first.status(), 200);

    let second = client
        .post(handle.url("/v1/chat/completions"))
        .header("authorization", format!("Bearer {raw_key}"))
        .json(&body)
        .send()
        .await
        .expect("ok");
    assert_eq!(
        second.status(),
        429,
        "merged team rpm=1 should cap key rpm=100"
    );

    shutdown(handle).await;
}

/// Two virtual keys on the same team must share a team-level spend counter.
/// After one key exhausts the team budget, the second key is also rejected.
#[tokio::test]
async fn team_budget_aggregated_across_two_keys() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;
    let _m = mock_chat(&mock);

    let client = reqwest::Client::new();
    let auth = format!("Bearer {MASTER_KEY}");

    let team_resp = client
        .post(handle.url("/team/add"))
        .header("authorization", &auth)
        .json(&json!({"team_name": "budget-team", "max_budget": 0.0001}))
        .send()
        .await
        .expect("team");
    assert_eq!(team_resp.status(), 200);
    let tb: Value = team_resp.json().await.expect("json");
    let team_id = tb["team_id"].as_str().expect("team_id").to_string();

    let gen_resp_a = client
        .post(handle.url("/key/generate"))
        .header("authorization", &auth)
        .json(&json!({"key_alias": "team-key-a", "team_id": team_id, "max_budget": 100.0}))
        .send()
        .await
        .expect("key gen");
    assert_eq!(gen_resp_a.status(), 200);
    let ka: Value = gen_resp_a.json().await.expect("json");
    let key_a = ka["key"].as_str().expect("key").to_string();

    let gen_resp_b = client
        .post(handle.url("/key/generate"))
        .header("authorization", &auth)
        .json(&json!({"key_alias": "team-key-b", "team_id": team_id, "max_budget": 100.0}))
        .send()
        .await
        .expect("key gen");
    assert_eq!(gen_resp_b.status(), 200);
    let kb: Value = gen_resp_b.json().await.expect("json");
    let key_b = kb["key"].as_str().expect("key").to_string();

    let body = json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 5
    });

    let resp_a = client
        .post(handle.url("/v1/chat/completions"))
        .header("authorization", format!("Bearer {key_a}"))
        .json(&body)
        .send()
        .await
        .expect("ok");
    assert_eq!(
        resp_a.status(),
        200,
        "first request from key A should succeed"
    );

    let resp_b = client
        .post(handle.url("/v1/chat/completions"))
        .header("authorization", format!("Bearer {key_b}"))
        .json(&body)
        .send()
        .await
        .expect("ok");
    assert_eq!(
        resp_b.status(),
        429,
        "request from key B should be rejected — team budget exhausted by key A"
    );

    shutdown(handle).await;
}

// ---------------------------------------------------------------------------
// #26b: negative rate-limit / budget values must be rejected at write time
// ---------------------------------------------------------------------------

#[tokio::test]
async fn generate_key_rejects_negative_rpm() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/key/generate"))
        .header("authorization", format!("Bearer {MASTER_KEY}"))
        .json(&json!({"key_alias": "neg-rpm", "rpm": -1}))
        .send()
        .await
        .expect("ok");
    assert_eq!(
        resp.status(),
        400,
        "negative rpm must be rejected at /key/generate"
    );

    shutdown(handle).await;
}

#[tokio::test]
async fn generate_key_rejects_negative_tpm() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/key/generate"))
        .header("authorization", format!("Bearer {MASTER_KEY}"))
        .json(&json!({"tpm": -5}))
        .send()
        .await
        .expect("ok");
    assert_eq!(resp.status(), 400, "negative tpm must be rejected");

    shutdown(handle).await;
}

#[tokio::test]
async fn generate_key_rejects_negative_max_budget() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/key/generate"))
        .header("authorization", format!("Bearer {MASTER_KEY}"))
        .json(&json!({"max_budget": -0.5}))
        .send()
        .await
        .expect("ok");
    assert_eq!(resp.status(), 400, "negative max_budget must be rejected");

    shutdown(handle).await;
}

#[tokio::test]
async fn generate_key_accepts_zero_limits() {
    // rpm=0 / tpm=0 / max_budget=0 are valid (0 == "no quota"), only
    // negative values are forbidden.
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/key/generate"))
        .header("authorization", format!("Bearer {MASTER_KEY}"))
        .json(&json!({"rpm": 0, "tpm": 0, "max_budget": 0.0}))
        .send()
        .await
        .expect("ok");
    assert_eq!(resp.status(), 200, "zero limits must be accepted");

    shutdown(handle).await;
}

#[tokio::test]
async fn update_key_rejects_negative_rpm() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;

    let client = reqwest::Client::new();
    let auth = format!("Bearer {MASTER_KEY}");

    let gen_resp = client
        .post(handle.url("/key/generate"))
        .header("authorization", &auth)
        .json(&json!({}))
        .send()
        .await
        .expect("ok");
    let gb: Value = gen_resp.json().await.expect("json");
    let key_id = gb["key_id"].as_str().expect("key_id").to_string();

    let upd = client
        .post(handle.url("/key/update"))
        .header("authorization", &auth)
        .json(&json!({"key_id": key_id, "rpm": -10}))
        .send()
        .await
        .expect("ok");
    assert_eq!(upd.status(), 400, "/key/update must reject negative rpm");

    shutdown(handle).await;
}

// ---------------------------------------------------------------------------
// Bug #39: deleting a team that never existed must return 404 (previously
// returned {"status":"deleted"} unconditionally, masking operator typos).
// ---------------------------------------------------------------------------
#[tokio::test]
async fn delete_team_returns_404_for_nonexistent_team() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;

    let client = reqwest::Client::new();
    let auth = format!("Bearer {MASTER_KEY}");

    let resp = client
        .post(handle.url("/team/delete"))
        .header("authorization", &auth)
        .json(&json!({"team_id": "team-does-not-exist"}))
        .send()
        .await
        .expect("ok");
    assert_eq!(
        resp.status(),
        404,
        "deleting a nonexistent team must return 404, not 200"
    );
    let body: Value = resp.json().await.expect("json body");
    assert!(
        body["error"]["message"]
            .as_str()
            .map(|s| s.contains("not found"))
            .unwrap_or(false),
        "error body should mention 'not found', got: {body}"
    );

    shutdown(handle).await;
}

// ---------------------------------------------------------------------------
// Bug #39: deleting a team that does exist still works (regression guard).
// ---------------------------------------------------------------------------
#[tokio::test]
async fn delete_team_succeeds_for_existing_team() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;

    let client = reqwest::Client::new();
    let auth = format!("Bearer {MASTER_KEY}");

    let add = client
        .post(handle.url("/team/add"))
        .header("authorization", &auth)
        .json(&json!({"team_name": "ToDelete"}))
        .send()
        .await
        .expect("ok");
    assert_eq!(add.status(), 200);
    let body: Value = add.json().await.expect("json");
    let team_id = body["team_id"].as_str().expect("team_id").to_string();

    let del = client
        .post(handle.url("/team/delete"))
        .header("authorization", &auth)
        .json(&json!({"team_id": team_id}))
        .send()
        .await
        .expect("ok");
    assert_eq!(del.status(), 200);

    // Subsequent delete now returns 404 because the row is gone.
    let del_again = client
        .post(handle.url("/team/delete"))
        .header("authorization", &auth)
        .json(&json!({"team_id": team_id}))
        .send()
        .await
        .expect("ok");
    assert_eq!(
        del_again.status(),
        404,
        "after deletion, the row is gone so a second delete must 404"
    );

    shutdown(handle).await;
}

// ---------------------------------------------------------------------------
// Bug #39: deleting a key that never existed must return 404. A second
// delete against a soft-deleted key must remain idempotent (returns
// "deleted") because the row still exists in the table.
// ---------------------------------------------------------------------------
#[tokio::test]
async fn delete_key_404_for_never_existent_and_idempotent_for_soft_deleted() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;

    let client = reqwest::Client::new();
    let auth = format!("Bearer {MASTER_KEY}");

    // Never-existed key → 404.
    let resp = client
        .post(handle.url("/key/delete"))
        .header("authorization", &auth)
        .json(&json!({"key_id": "key-never-existed"}))
        .send()
        .await
        .expect("ok");
    assert_eq!(
        resp.status(),
        404,
        "deleting a key that never existed must return 404"
    );

    // Generate a real key, then soft-delete it twice. The first call is 200,
    // the second is also 200 (idempotent) because the row still exists.
    let gen_resp = client
        .post(handle.url("/key/generate"))
        .header("authorization", &auth)
        .json(&json!({"key_alias": "soft-delete-target"}))
        .send()
        .await
        .expect("ok");
    assert_eq!(gen_resp.status(), 200);
    let gb: Value = gen_resp.json().await.expect("json");
    let key_id = gb["key_id"].as_str().expect("key_id").to_string();

    let del1 = client
        .post(handle.url("/key/delete"))
        .header("authorization", &auth)
        .json(&json!({"key_id": key_id}))
        .send()
        .await
        .expect("ok");
    assert_eq!(del1.status(), 200, "first soft-delete should succeed");

    let del2 = client
        .post(handle.url("/key/delete"))
        .header("authorization", &auth)
        .json(&json!({"key_id": key_id}))
        .send()
        .await
        .expect("ok");
    assert_eq!(
        del2.status(),
        200,
        "second soft-delete must remain idempotent (200), not 404"
    );

    shutdown(handle).await;
}

// ---------------------------------------------------------------------------
// Model ACLs (allowed_models): a key generated with `models: ["gpt-4o"]`
// must be allowed to call gpt-4o but rejected (403) for claude-3. Master key
// is unaffected. Wildcard patterns are exercised separately.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn key_models_restricts_to_allowed_list() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;
    let _m = mock_chat(&mock);

    let client = reqwest::Client::new();
    let auth = format!("Bearer {MASTER_KEY}");

    // Generate a key whose allowed-models list contains only gpt-4o.
    let gen_resp = client
        .post(handle.url("/key/generate"))
        .header("authorization", &auth)
        .json(&json!({"key_alias": "acl", "models": ["gpt-4o"]}))
        .send()
        .await
        .expect("ok");
    assert_eq!(gen_resp.status(), 200, "generate should succeed");
    let gb: Value = gen_resp.json().await.expect("json");
    let raw_key = gb["key"].as_str().expect("key").to_string();

    // Allowed model: 200.
    let ok = client
        .post(handle.url("/v1/chat/completions"))
        .header("authorization", format!("Bearer {raw_key}"))
        .json(&json!({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .expect("ok");
    assert_eq!(ok.status(), 200, "gpt-4o must be allowed");

    // Disallowed model: 403.
    let blocked = client
        .post(handle.url("/v1/chat/completions"))
        .header("authorization", format!("Bearer {raw_key}"))
        .json(&json!({"model": "claude-3", "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .expect("ok");
    assert_eq!(
        blocked.status(),
        403,
        "claude-3 must be forbidden by the key's models list"
    );

    shutdown(handle).await;
}

#[tokio::test]
async fn key_models_wildcard_pattern_matches() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;
    let _m = mock_chat(&mock);

    let client = reqwest::Client::new();
    let auth = format!("Bearer {MASTER_KEY}");

    let gen_resp = client
        .post(handle.url("/key/generate"))
        .header("authorization", &auth)
        .json(&json!({"models": ["gpt-*"]}))
        .send()
        .await
        .expect("ok");
    let gb: Value = gen_resp.json().await.expect("json");
    let raw_key = gb["key"].as_str().expect("key").to_string();

    // gpt-4o matches gpt-* → allowed.
    let ok = client
        .post(handle.url("/v1/chat/completions"))
        .header("authorization", format!("Bearer {raw_key}"))
        .json(&json!({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .expect("ok");
    assert_eq!(ok.status(), 200, "gpt-4o should match gpt-*");

    // claude-3 doesn't match gpt-* → 403.
    let blocked = client
        .post(handle.url("/v1/chat/completions"))
        .header("authorization", format!("Bearer {raw_key}"))
        .json(&json!({"model": "claude-3", "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .expect("ok");
    assert_eq!(blocked.status(), 403);

    shutdown(handle).await;
}

#[tokio::test]
async fn master_key_bypasses_model_acls() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;
    let _m = mock_chat(&mock);

    let client = reqwest::Client::new();
    // Create a team with restrictive models, then a key attached to it.
    // Master-key traffic must still bypass the team restriction.
    let team = client
        .post(handle.url("/team/add"))
        .header("authorization", format!("Bearer {MASTER_KEY}"))
        .json(&json!({"team_name": "Restricted", "models": ["claude-*"]}))
        .send()
        .await
        .expect("ok");
    let tb: Value = team.json().await.expect("json");
    let team_id = tb["team_id"].as_str().expect("team_id").to_string();

    let gen_resp = client
        .post(handle.url("/key/generate"))
        .header("authorization", format!("Bearer {MASTER_KEY}"))
        .json(&json!({"team_id": team_id, "models": ["gpt-4o"]}))
        .send()
        .await
        .expect("ok");
    let _gb: Value = gen_resp.json().await.expect("json");

    // Master key calls gpt-4o even though no ACL was configured for it.
    let ok = client
        .post(handle.url("/v1/chat/completions"))
        .header("authorization", format!("Bearer {MASTER_KEY}"))
        .json(&json!({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .expect("ok");
    assert_eq!(ok.status(), 200, "master key must bypass ACLs");

    shutdown(handle).await;
}

#[tokio::test]
async fn team_models_intersect_with_key_models() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;
    let _m = mock_chat(&mock);

    let client = reqwest::Client::new();
    let auth = format!("Bearer {MASTER_KEY}");

    // Team only allows claude-*. Key allows gpt-4o. The intersection is
    // empty, so gpt-4o must be rejected by the team-level check.
    let team = client
        .post(handle.url("/team/add"))
        .header("authorization", &auth)
        .json(&json!({"team_name": "ClaudeOnly", "models": ["claude-*"]}))
        .send()
        .await
        .expect("ok");
    let tb: Value = team.json().await.expect("json");
    let team_id = tb["team_id"].as_str().expect("team_id").to_string();

    let gen_resp = client
        .post(handle.url("/key/generate"))
        .header("authorization", &auth)
        .json(&json!({"team_id": team_id, "models": ["gpt-4o"]}))
        .send()
        .await
        .expect("ok");
    let gb: Value = gen_resp.json().await.expect("json");
    let raw_key = gb["key"].as_str().expect("key").to_string();

    // gpt-4o passes the key check but fails the team check → 403.
    let blocked = client
        .post(handle.url("/v1/chat/completions"))
        .header("authorization", format!("Bearer {raw_key}"))
        .json(&json!({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .expect("ok");
    assert_eq!(
        blocked.status(),
        403,
        "team-level restriction must reject even when key allows the model"
    );

    shutdown(handle).await;
}

#[tokio::test]
async fn key_without_models_is_unrestricted() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;
    let _m = mock_chat(&mock);

    let client = reqwest::Client::new();
    let gen_resp = client
        .post(handle.url("/key/generate"))
        .header("authorization", format!("Bearer {MASTER_KEY}"))
        .json(&json!({}))
        .send()
        .await
        .expect("ok");
    let gb: Value = gen_resp.json().await.expect("json");
    let raw_key = gb["key"].as_str().expect("key").to_string();

    // No models restriction → must not be rejected by the ACL (i.e. status
    // is NOT 403). gpt-4o goes through the OpenAI-compatible path and is
    // fully mocked (200). claude-3 is configured but routed to a different
    // provider path; what we care about here is that it isn't forbidden.
    let ok = client
        .post(handle.url("/v1/chat/completions"))
        .header("authorization", format!("Bearer {raw_key}"))
        .json(&json!({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .expect("ok");
    assert_eq!(ok.status(), 200, "gpt-4o should succeed fully");

    let claude_resp = client
        .post(handle.url("/v1/chat/completions"))
        .header("authorization", format!("Bearer {raw_key}"))
        .json(&json!({"model": "claude-3", "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .expect("ok");
    assert_ne!(
        claude_resp.status(),
        403,
        "claude-3 must not be forbidden by ACL when no models restriction is set"
    );

    shutdown(handle).await;
}

#[tokio::test]
async fn key_info_returns_models_field() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;

    let client = reqwest::Client::new();
    let auth = format!("Bearer {MASTER_KEY}");

    let gen_resp = client
        .post(handle.url("/key/generate"))
        .header("authorization", &auth)
        .json(&json!({"models": ["gpt-4o", "claude-*"]}))
        .send()
        .await
        .expect("ok");
    let gb: Value = gen_resp.json().await.expect("json");
    let key_id = gb["key_id"].as_str().expect("key_id").to_string();

    let info = client
        .get(handle.url(&format!("/key/info?key_id={key_id}")))
        .header("authorization", &auth)
        .send()
        .await
        .expect("ok");
    assert_eq!(info.status(), 200);
    let ib: Value = info.json().await.expect("json");
    let models = ib["models"].as_str().expect("models is a string");
    assert!(
        models.contains("gpt-4o") && models.contains("claude-*"),
        "models field should preserve patterns, got: {models}"
    );

    shutdown(handle).await;
}

// ---------------------------------------------------------------------------
// DB-stored model config: /model/new, /model/update, /model/delete
// ---------------------------------------------------------------------------

fn add_model_body(name: &str, params: Value) -> Value {
    json!({
        "model_name": name,
        "oxllm_params": params,
    })
}

#[tokio::test]
async fn model_new_adds_to_db_and_list() {
    let mock = MockServer::start();
    let (handle, db) = start_oxllm(proxy_config(&mock.base_url())).await;

    let client = reqwest::Client::new();
    let auth = format!("Bearer {MASTER_KEY}");

    let resp = client
        .post(handle.url("/model/new"))
        .header("authorization", &auth)
        .json(&add_model_body(
            "my-dynamic",
            json!({"model": "openai/gpt-4o", "api_key": "sk-test"}),
        ))
        .send()
        .await
        .expect("ok");
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("json");
    let model_id = body["model_id"].as_str().expect("model_id").to_string();
    assert!(model_id.starts_with("model-"));

    // Persisted in DB.
    let records = db.list_models_db(None).await.expect("list");
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].model_id, model_id);

    // Visible in /model/list (merged with YAML).
    let list = client
        .get(handle.url("/model/list"))
        .header("authorization", &auth)
        .send()
        .await
        .expect("ok");
    let lb: Value = list.json().await.expect("json");
    let data = lb["data"].as_array().expect("data array");
    assert!(
        data.iter().any(|m| m["id"].as_str() == Some("my-dynamic")),
        "DB model should appear in /model/list: {lb}"
    );
    // YAML models are still there.
    assert!(data.iter().any(|m| m["id"].as_str() == Some("gpt-4o")));

    shutdown(handle).await;
}

#[tokio::test]
async fn model_update_renames_model_name() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;

    let client = reqwest::Client::new();
    let auth = format!("Bearer {MASTER_KEY}");

    let add = client
        .post(handle.url("/model/new"))
        .header("authorization", &auth)
        .json(&add_model_body(
            "before-rename",
            json!({"model": "openai/gpt-4o", "api_key": "sk-test"}),
        ))
        .send()
        .await
        .expect("ok");
    let ab: Value = add.json().await.expect("json");
    let model_id = ab["model_id"].as_str().expect("model_id").to_string();

    let upd = client
        .post(handle.url("/model/update"))
        .header("authorization", &auth)
        .json(&json!({
            "model_id": model_id,
            "model_name": "after-rename",
            "oxllm_params": {"model": "openai/gpt-4o-mini", "api_key": "sk-test"}
        }))
        .send()
        .await
        .expect("ok");
    assert_eq!(upd.status(), 200);
    let ub: Value = upd.json().await.expect("json");
    assert_eq!(ub["model_name"].as_str(), Some("after-rename"));
    assert_eq!(ub["oxllm_params"]["model"], "openai/gpt-4o-mini");

    let list = client
        .get(handle.url("/model/list"))
        .header("authorization", &auth)
        .send()
        .await
        .expect("ok");
    let lb: Value = list.json().await.expect("json");
    let data = lb["data"].as_array().expect("data array");
    assert!(
        data.iter()
            .any(|m| m["id"].as_str() == Some("after-rename")),
        "renamed model should appear: {lb}"
    );
    assert!(
        !data
            .iter()
            .any(|m| m["id"].as_str() == Some("before-rename")),
        "old name should be gone: {lb}"
    );

    shutdown(handle).await;
}

#[tokio::test]
async fn model_delete_removes_from_db_and_list() {
    let mock = MockServer::start();
    let (handle, db) = start_oxllm(proxy_config(&mock.base_url())).await;

    let client = reqwest::Client::new();
    let auth = format!("Bearer {MASTER_KEY}");

    let add = client
        .post(handle.url("/model/new"))
        .header("authorization", &auth)
        .json(&add_model_body(
            "to-delete",
            json!({"model": "openai/gpt-4o", "api_key": "sk-test"}),
        ))
        .send()
        .await
        .expect("ok");
    let ab: Value = add.json().await.expect("json");
    let model_id = ab["model_id"].as_str().expect("model_id").to_string();

    let del = client
        .post(handle.url("/model/delete"))
        .header("authorization", &auth)
        .json(&json!({"model_id": model_id}))
        .send()
        .await
        .expect("ok");
    assert_eq!(del.status(), 200);

    assert!(db.get_model(&model_id).await.expect("get").is_none());

    let list = client
        .get(handle.url("/model/list"))
        .header("authorization", &auth)
        .send()
        .await
        .expect("ok");
    let lb: Value = list.json().await.expect("json");
    let data = lb["data"].as_array().expect("data array");
    assert!(
        !data.iter().any(|m| m["id"].as_str() == Some("to-delete")),
        "deleted model should not appear: {lb}"
    );

    shutdown(handle).await;
}

#[tokio::test]
async fn model_delete_returns_404_for_unknown_id() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;

    let client = reqwest::Client::new();
    let auth = format!("Bearer {MASTER_KEY}");

    let resp = client
        .post(handle.url("/model/delete"))
        .header("authorization", &auth)
        .json(&json!({"model_id": "model-does-not-exist"}))
        .send()
        .await
        .expect("ok");
    assert_eq!(resp.status(), 404);

    shutdown(handle).await;
}

#[tokio::test]
async fn model_update_returns_404_for_unknown_id() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;

    let client = reqwest::Client::new();
    let auth = format!("Bearer {MASTER_KEY}");

    let resp = client
        .post(handle.url("/model/update"))
        .header("authorization", &auth)
        .json(&json!({
            "model_id": "model-does-not-exist",
            "model_name": "x",
            "oxllm_params": {"model": "openai/gpt-4o"}
        }))
        .send()
        .await
        .expect("ok");
    assert_eq!(resp.status(), 404);

    shutdown(handle).await;
}

#[tokio::test]
async fn model_new_rejects_empty_model_name() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;

    let client = reqwest::Client::new();
    let auth = format!("Bearer {MASTER_KEY}");

    let resp = client
        .post(handle.url("/model/new"))
        .header("authorization", &auth)
        .json(&add_model_body("", json!({"model": "openai/gpt-4o"})))
        .send()
        .await
        .expect("ok");
    assert_eq!(resp.status(), 400);

    shutdown(handle).await;
}

#[tokio::test]
async fn model_new_rejects_empty_model() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;

    let client = reqwest::Client::new();
    let auth = format!("Bearer {MASTER_KEY}");

    let resp = client
        .post(handle.url("/model/new"))
        .header("authorization", &auth)
        .json(&add_model_body("x", json!({"model": ""})))
        .send()
        .await
        .expect("ok");
    assert_eq!(resp.status(), 400);

    shutdown(handle).await;
}

/// C30 MEDIUM: a DB model whose `api_base` is rejected by the SSRF policy
/// (private CIDR while `allow_private_api_base = false`) used to return 200 —
/// `refresh_db_models` silently dropped the row with only a `tracing::warn!`.
/// The handler now re-checks the live deployments after refresh and surfaces a
/// 400, rolling back the stored row.
#[tokio::test]
async fn model_new_rejects_private_api_base_when_disallowed() {
    let _mock = MockServer::start();
    let cfg = ProxyConfig {
        model_list: vec![],
        oxllm_settings: OxllmSettings::default(),
        general_settings: GeneralSettings {
            master_key: Some(MASTER_KEY.to_string()),
            allow_private_api_base: Some(false),
            ..Default::default()
        },
        router_settings: Default::default(),
        extra: serde_json::Map::new(),
    };
    let (handle, db) = start_oxllm(cfg).await;

    let client = reqwest::Client::new();
    let auth = format!("Bearer {MASTER_KEY}");

    let resp = client
        .post(handle.url("/model/new"))
        .header("authorization", &auth)
        .json(&add_model_body(
            "private-model",
            json!({
                "model": "openai/gpt-4o",
                "api_base": "http://127.0.0.1:11434",
            }),
        ))
        .send()
        .await
        .expect("ok");
    assert_eq!(
        resp.status(),
        400,
        "private api_base with allow_private_api_base=false must be rejected: {}",
        resp.status()
    );

    // The rolled-back row must not linger in the DB.
    let records = db.list_models_db(None).await.expect("list");
    assert!(
        records.is_empty(),
        "rejected model must be rolled back, found {} rows",
        records.len()
    );

    shutdown(handle).await;
}

/// Two concurrent `POST /model/new` with the same `model_name` race the
/// `name_taken` TOCTOU check. Exactly one must win; the loser's row must be
/// rolled back so no durable orphan lingers in the `models` table (C65 LOW).
#[tokio::test]
async fn concurrent_model_new_same_name_leaves_single_row() {
    let mock = MockServer::start();
    let (handle, db) = start_oxllm(proxy_config(&mock.base_url())).await;

    let client = reqwest::Client::new();
    let auth = format!("Bearer {MASTER_KEY}");
    let url = handle.url("/model/new");
    let body = add_model_body(
        "race-model",
        json!({"model": "openai/gpt-4o", "api_key": "sk-test"}),
    );

    // Fire both concurrently so they observe the same pre-insert state.
    let (r1, r2) = tokio::join!(
        client
            .post(&url)
            .header("authorization", &auth)
            .json(&body)
            .send(),
        client
            .post(&url)
            .header("authorization", &auth)
            .json(&body)
            .send(),
    );
    let (r1, r2) = (r1.expect("ok"), r2.expect("ok"));

    // Exactly one succeeds; the loser is rejected with 400 (orphan rollback).
    let codes = vec![r1.status().as_u16(), r2.status().as_u16()];
    let ok = codes.iter().filter(|&&c| c == 200).count();
    let bad = codes.iter().filter(|&&c| c == 400).count();
    assert_eq!(
        ok, 1,
        "exactly one concurrent add must succeed, got statuses {codes:?}"
    );
    assert_eq!(
        bad, 1,
        "the loser must be rejected with 400, got statuses {codes:?}"
    );

    // No orphan row: the DB carries exactly one row for this model_name.
    let records = db
        .list_models_db(None)
        .await
        .expect("list")
        .into_iter()
        .filter(|r| r.model_name == "race-model")
        .collect::<Vec<_>>();
    assert_eq!(
        records.len(),
        1,
        "exactly one row must remain after the race, found {}",
        records.len()
    );

    shutdown(handle).await;
}

/// A model added via the management API must be routable for chat just like a
/// YAML deployment. This exercises the full path: POST /model/new → refresh →
/// router picks the new deployment → upstream is hit.
#[tokio::test]
async fn db_model_is_usable_for_chat_after_add() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;

    // Upstream mock for the dynamic model. The chat handler will rewrite the
    // path to the provider's expected path; the openai-compatible provider
    // hits /chat/completions.
    let _m = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .header("authorization", "Bearer sk-dynamic");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "id": "chatcmpl-dyn",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi from dynamic"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
            }));
    });

    let client = reqwest::Client::new();
    let auth = format!("Bearer {MASTER_KEY}");

    let add = client
        .post(handle.url("/model/new"))
        .header("authorization", &auth)
        .json(&add_model_body(
            "dynamic-model",
            json!({
                "model": "openai/gpt-4o",
                "api_key": "sk-dynamic",
                "api_base": mock.base_url(),
            }),
        ))
        .send()
        .await
        .expect("ok");
    assert_eq!(add.status(), 200);

    let chat = client
        .post(handle.url("/v1/chat/completions"))
        .header("authorization", &auth)
        .json(&json!({
            "model": "dynamic-model",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("ok");
    assert_eq!(
        chat.status(),
        200,
        "DB-added model must route to upstream. Body: {}",
        chat.text().await.unwrap_or_default()
    );

    shutdown(handle).await;
}

/// DB-stored models survive a config hot-reload: they are additive to whatever
/// the YAML currently carries.
#[tokio::test]
async fn db_models_survive_config_reload() {
    use oxllm_server::SharedState;
    use oxllm_server::state::refresh_db_models;

    let mock = MockServer::start();

    // Build the state by hand so we can call reload_from_config + refresh.
    let db = fresh_db().await;
    let writer = spend_writer::spawn(db.clone(), 1);
    let initial: SharedState = build_shared_state_with_db(
        proxy_config(&mock.base_url()),
        Some(writer.sender()),
        Some(db.clone()),
    )
    .expect("state");

    // Insert a DB model directly (bypasses the API to keep this test focused
    // on the reload path rather than the HTTP plumbing).
    db.create_model(&ModelConfigRecord {
        model_id: "model-reload-1".to_string(),
        model_name: "persisted-model".to_string(),
        oxllm_params: json!({"model": "openai/gpt-4o", "api_key": "sk-test"}),
        model_info: None,
        created_at: oxllm_server::now_iso(),
        updated_at: oxllm_server::now_iso(),
    })
    .await
    .expect("create");

    // First refresh picks up the DB row.
    refresh_db_models(&initial).await.expect("refresh");
    assert!(
        initial
            .load_full()
            .deployments
            .iter()
            .any(|d| d.model_name == "persisted-model"),
        "DB model should be present after initial refresh"
    );

    // Simulate a config hot-reload by rebuilding state from a new config. The
    // YAML has only `gpt-4o` / `claude-3`, so without re-merging the DB row
    // would vanish.
    let prev = initial.load_full();
    let reloaded = prev
        .reload_from_config(proxy_config(&mock.base_url()))
        .expect("reload");
    initial.store(std::sync::Arc::new(reloaded));
    refresh_db_models(&initial)
        .await
        .expect("refresh after reload");

    let after = initial.load_full();
    assert!(
        after
            .deployments
            .iter()
            .any(|d| d.model_name == "persisted-model"),
        "DB model must survive a config reload"
    );
    assert!(
        after.deployments.iter().any(|d| d.model_name == "gpt-4o"),
        "YAML models should still be present after reload"
    );

    // Ensure exactly one copy: no duplicates from the re-merge.
    let count = after
        .deployments
        .iter()
        .filter(|d| d.model_name == "persisted-model")
        .count();
    assert_eq!(count, 1, "DB model should not be duplicated after reload");
}

/// Helper: a `ProxyConfig` whose YAML `model_list` carries only `gpt-4o`, with
/// the given master key (so a reload changing the master key is observable in
/// `master_key_hash`).
fn single_yaml_config(mock_base_url: &str, master_key: &str) -> ProxyConfig {
    ProxyConfig {
        model_list: vec![ConfigModel {
            model_name: "gpt-4o".to_string(),
            oxllm_params: OxllmParams {
                model: "openai/gpt-4o".to_string(),
                api_key: Some("sk-upstream".to_string()),
                api_base: Some(mock_base_url.to_string()),
                timeout: Some(30.0),
                stream_timeout: Some(30),
                ..Default::default()
            },
            ..Default::default()
        }],
        oxllm_settings: OxllmSettings::default(),
        general_settings: GeneralSettings {
            master_key: Some(master_key.to_string()),
            allow_private_api_base: Some(true),
            ..Default::default()
        },
        router_settings: Default::default(),
        extra: serde_json::Map::new(),
    }
}

fn db_model(model_id: &str, name: &str) -> ModelConfigRecord {
    ModelConfigRecord {
        model_id: model_id.to_string(),
        model_name: name.to_string(),
        oxllm_params: json!({"model": "openai/gpt-4o", "api_key": "sk-test"}),
        model_info: None,
        created_at: oxllm_server::now_iso(),
        updated_at: oxllm_server::now_iso(),
    }
}

/// `refresh_db_models` must prune the router runtime stores for a DB model
/// that was deleted via the management API, mirroring the config hot-reload
/// path in `assemble`. Without pruning, the deleted model's cooldown entry
/// leaks forever under its old positional key (`"{model_name}:{index}"`).
///
/// This test cools a survivor whose index is unchanged (verifying its cooldown
/// is retained) and the doomed model itself (verifying its entry is pruned
/// rather than orphaned). It reads the actual deployment order from the live
/// state (DB rows come back newest-first, so insertion order is not the append
/// order) and deletes the highest-index model — which preserves every lower
/// index, so the survivor's cooldown key is stable across the refresh.
#[tokio::test]
async fn refresh_db_models_prunes_store_on_db_model_deletion() {
    use oxllm_server::SharedState;
    use oxllm_server::state::refresh_db_models;

    let mock = MockServer::start();
    let db = fresh_db().await;
    let writer = spend_writer::spawn(db.clone(), 1);

    // Three DB models appended after the single YAML model `gpt-4o`.
    db.create_model(&db_model("m-a", "db-a"))
        .await
        .expect("create");
    db.create_model(&db_model("m-b", "db-b"))
        .await
        .expect("create");
    db.create_model(&db_model("m-c", "db-c"))
        .await
        .expect("create");

    let shared: SharedState = build_shared_state_with_db(
        single_yaml_config(&mock.base_url(), "sk-master-key-1234"),
        Some(writer.sender()),
        Some(db.clone()),
    )
    .expect("state");
    refresh_db_models(&shared).await.expect("initial refresh");

    let state_after = shared.load_full();
    let deployments = state_after.deployments.clone();
    let router = state_after.router.clone();
    let cooldown = router.cooldown_store();
    // YAML `gpt-4o` is index 0; DB models follow at indices 1..n.
    assert!(
        deployments.len() >= 3,
        "expected YAML + 3 DB models, got {}",
        deployments.len()
    );

    // Survivor = first DB model (lowest DB index). Doomed = highest-index
    // deployment (deleting it preserves all lower indices).
    let survivor_idx = 1;
    let survivor = &deployments[survivor_idx];
    let survivor_key = router.cooldown_key(survivor);
    let doomed_idx = deployments.len() - 1;
    let doomed = &deployments[doomed_idx];
    let doomed_key = router.cooldown_key(doomed);
    assert_ne!(
        survivor.model_name, doomed.model_name,
        "survivor and doomed must be distinct models"
    );

    cooldown.mark(&survivor_key, std::time::Duration::from_secs(600));
    cooldown.mark(&doomed_key, std::time::Duration::from_secs(600));
    assert!(cooldown.is_cooling(&survivor_key));
    assert!(cooldown.is_cooling(&doomed_key));

    // Delete the doomed DB model by its model_id. The fixture maps
    // model_name "db-<x>" to model_id "m-<x>".
    let doomed_id = format!("m-{}", doomed.model_name.trim_start_matches("db-"));
    db.delete_model(&doomed_id).await.expect("delete");
    refresh_db_models(&shared)
        .await
        .expect("refresh after delete");

    // The survivor's cooldown is retained (keyed stably, independent of any
    // index shift from the DB refresh).
    assert!(
        cooldown.is_cooling(&survivor_key),
        "survivor cooldown must survive refresh_db_models"
    );
    // The deleted model's cooldown entry is pruned (no orphaned leak).
    assert!(
        !cooldown.is_cooling(&doomed_key),
        "deleted DB model cooldown must be pruned, not orphaned"
    );
}

/// `refresh_db_models` must not silently revert a concurrent config
/// hot-reload. It publishes via `ArcSwap::compare_and_swap` in an RCU retry
/// loop: if a reload lands between the refresh's `load_full` and its publish,
/// the refresh retries from the post-reload state instead of overwriting it
/// with a stale-config clone. This stress test hammers both paths on a
/// multi-thread runtime; a regression to a plain `store` surfaces as the
/// `master_key_hash` reverting to the pre-reload key.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn refresh_db_models_does_not_revert_concurrent_reload() {
    use oxllm_server::SharedState;
    use oxllm_server::auth::hash_key;
    use oxllm_server::state::refresh_db_models;

    let mock = MockServer::start();
    let db = fresh_db().await;
    db.create_model(&db_model("m-stress", "db-stress"))
        .await
        .expect("create");
    let writer = spend_writer::spawn(db.clone(), 1);

    let shared: SharedState = build_shared_state_with_db(
        single_yaml_config(&mock.base_url(), "sk-key-A-key-12345"),
        Some(writer.sender()),
        Some(db.clone()),
    )
    .expect("state");
    refresh_db_models(&shared).await.expect("initial refresh");

    let shared_reload = shared.clone();
    let reload_handle = tokio::spawn(async move {
        let config_b = single_yaml_config(&mock.base_url(), "sk-key-B-key-67890");
        for _ in 0..40 {
            let prev = shared_reload.load_full();
            let reloaded = prev.reload_from_config(config_b.clone()).expect("reload");
            shared_reload.store(Arc::new(reloaded));
            refresh_db_models(&shared_reload)
                .await
                .expect("refresh after reload");
        }
    });

    let shared_refresh = shared.clone();
    let refresh_handle = tokio::spawn(async move {
        for _ in 0..40 {
            refresh_db_models(&shared_refresh).await.ok();
        }
    });

    reload_handle.await.expect("reload task");
    refresh_handle.await.expect("refresh task");

    let final_state = shared.load_full();
    // The reload task's final action was reload(config B) + refresh, so the
    // live config must be config B. A racing refresh that used a plain store
    // would periodically revert this to config A.
    assert_eq!(
        final_state.master_key_hash,
        Some(hash_key(b"sk-key-B-key-67890")),
        "concurrent refresh_db_models must not revert the reloaded master key"
    );
    // The DB model survives the churn.
    assert!(
        final_state
            .deployments
            .iter()
            .any(|d| d.model_name == "db-stress"),
        "DB model must remain merged after concurrent reload + refresh"
    );
}

// ---------------------------------------------------------------------------
// Organization & Project CRUD + budget enforcement
// ---------------------------------------------------------------------------

#[tokio::test]
async fn organization_crud_via_api() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;
    let client = reqwest::Client::new();
    let auth = format!("Bearer {MASTER_KEY}");

    let add = client
        .post(handle.url("/organization/add"))
        .header("authorization", &auth)
        .json(&json!({"org_name": "Acme", "max_budget": 100.0, "rpm": 50}))
        .send()
        .await
        .expect("ok");
    assert_eq!(add.status(), 200);
    let ab: Value = add.json().await.expect("json");
    let org_id = ab["org_id"].as_str().expect("org_id").to_string();
    assert!(
        org_id.starts_with("org-"),
        "auto-generated org_id should use org- prefix, got {org_id}"
    );

    let info = client
        .get(handle.url(&format!("/organization/info?org_id={org_id}")))
        .header("authorization", &auth)
        .send()
        .await
        .expect("ok");
    assert_eq!(info.status(), 200);
    let ib: Value = info.json().await.expect("json");
    assert_eq!(ib["org_name"].as_str(), Some("Acme"));
    assert_eq!(ib["max_budget"].as_f64(), Some(100.0));

    let list = client
        .get(handle.url("/organization/list"))
        .header("authorization", &auth)
        .send()
        .await
        .expect("ok");
    let lb: Value = list.json().await.expect("json");
    assert_eq!(lb["organizations"].as_array().map(|a| a.len()), Some(1));

    let upd = client
        .post(handle.url("/organization/update"))
        .header("authorization", &auth)
        .json(&json!({"org_id": org_id, "org_name": "Acme Co", "max_budget": 200.0}))
        .send()
        .await
        .expect("ok");
    assert_eq!(upd.status(), 200);
    let ub: Value = upd.json().await.expect("json");
    assert_eq!(ub["org_name"].as_str(), Some("Acme Co"));
    assert_eq!(ub["max_budget"].as_f64(), Some(200.0));

    let del = client
        .post(handle.url("/organization/delete"))
        .header("authorization", &auth)
        .json(&json!({"org_id": org_id}))
        .send()
        .await
        .expect("ok");
    assert_eq!(del.status(), 200);

    let missing = client
        .get(handle.url(&format!("/organization/info?org_id={org_id}")))
        .header("authorization", &auth)
        .send()
        .await
        .expect("ok");
    assert_eq!(missing.status(), 404);

    shutdown(handle).await;
}

#[tokio::test]
async fn project_crud_via_api() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;
    let client = reqwest::Client::new();
    let auth = format!("Bearer {MASTER_KEY}");

    let add = client
        .post(handle.url("/project/add"))
        .header("authorization", &auth)
        .json(&json!({"project_name": "Atlas", "max_budget": 50.0}))
        .send()
        .await
        .expect("ok");
    assert_eq!(add.status(), 200);
    let ab: Value = add.json().await.expect("json");
    let project_id = ab["project_id"].as_str().expect("project_id").to_string();
    assert!(
        project_id.starts_with("project-"),
        "auto-generated project_id should use project- prefix, got {project_id}"
    );

    let info = client
        .get(handle.url(&format!("/project/info?project_id={project_id}")))
        .header("authorization", &auth)
        .send()
        .await
        .expect("ok");
    assert_eq!(info.status(), 200);
    let ib: Value = info.json().await.expect("json");
    assert_eq!(ib["project_name"].as_str(), Some("Atlas"));

    let list = client
        .get(handle.url("/project/list"))
        .header("authorization", &auth)
        .send()
        .await
        .expect("ok");
    let lb: Value = list.json().await.expect("json");
    assert_eq!(lb["projects"].as_array().map(|a| a.len()), Some(1));

    let del = client
        .post(handle.url("/project/delete"))
        .header("authorization", &auth)
        .json(&json!({"project_id": project_id}))
        .send()
        .await
        .expect("ok");
    assert_eq!(del.status(), 200);

    shutdown(handle).await;
}

#[tokio::test]
async fn key_generation_with_org_and_project_associations() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;
    let client = reqwest::Client::new();
    let auth = format!("Bearer {MASTER_KEY}");

    let org_id = "org-test-1";
    let project_id = "project-test-1";
    client
        .post(handle.url("/organization/add"))
        .header("authorization", &auth)
        .json(&json!({"org_id": org_id, "org_name": "Acme"}))
        .send()
        .await
        .expect("ok");
    client
        .post(handle.url("/project/add"))
        .header("authorization", &auth)
        .json(&json!({"project_id": project_id, "project_name": "Atlas", "org_id": org_id}))
        .send()
        .await
        .expect("ok");

    let gen_resp = client
        .post(handle.url("/key/generate"))
        .header("authorization", &auth)
        .json(&json!({
            "key_alias": "with-org-project",
            "org_id": org_id,
            "project_id": project_id,
        }))
        .send()
        .await
        .expect("ok");
    assert_eq!(gen_resp.status(), 200);
    let gb: Value = gen_resp.json().await.expect("json");
    let key_id = gb["key_id"].as_str().expect("key_id").to_string();

    let info = client
        .get(handle.url(&format!("/key/info?key_id={key_id}")))
        .header("authorization", &auth)
        .send()
        .await
        .expect("ok");
    let ib: Value = info.json().await.expect("json");
    assert_eq!(ib["org_id"].as_str(), Some(org_id));
    assert_eq!(ib["project_id"].as_str(), Some(project_id));

    shutdown(handle).await;
}

#[tokio::test]
async fn deleting_and_recreating_scopes_does_not_reattach_cached_keys() {
    let mock = MockServer::start();
    let (handle, db) = start_oxllm(proxy_config(&mock.base_url())).await;
    let upstream = mock_chat(&mock);
    let client = reqwest::Client::new();
    for (endpoint, body) in [
        (
            "/organization/add",
            json!({"org_id":"detach-org", "org_name":"Org", "max_budget":0.0}),
        ),
        (
            "/project/add",
            json!({"project_id":"detach-project", "project_name":"Project",
            "org_id":"detach-org", "max_budget":1.0}),
        ),
    ] {
        client
            .post(handle.url(endpoint))
            .bearer_auth(MASTER_KEY)
            .json(&body)
            .send()
            .await
            .unwrap()
            .error_for_status()
            .unwrap();
    }
    let generated: Value = client
        .post(handle.url("/key/generate"))
        .bearer_auth(MASTER_KEY)
        .json(&json!({"org_id":"detach-org", "project_id":"detach-project", "max_budget":1.0}))
        .send()
        .await
        .unwrap()
        .error_for_status()
        .unwrap()
        .json()
        .await
        .unwrap();
    let key = generated["key"].as_str().unwrap();
    let key_id = generated["key_id"].as_str().unwrap();
    let request =
        json!({"model":"gpt-4o", "messages":[{"role":"user", "content":"hi"}], "max_tokens":5});
    let warm = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth(key)
        .json(&request)
        .send()
        .await
        .unwrap();
    assert_eq!(
        warm.status(),
        429,
        "warm auth and scope caches with the zero-cap org"
    );
    let error: Value = warm.json().await.unwrap();
    assert_eq!(error["error"]["code"], "budget_exceeded");
    client
        .post(handle.url("/organization/delete"))
        .bearer_auth(MASTER_KEY)
        .json(&json!({"org_id":"detach-org"}))
        .send()
        .await
        .unwrap()
        .error_for_status()
        .unwrap();
    assert!(db.get_key(key_id).await.unwrap().unwrap().org_id.is_none());
    assert!(
        db.get_project("detach-project")
            .await
            .unwrap()
            .unwrap()
            .org_id
            .is_none()
    );
    client
        .post(handle.url("/organization/add"))
        .bearer_auth(MASTER_KEY)
        .json(&json!({"org_id":"detach-org", "org_name":"New org", "max_budget":0.0}))
        .send()
        .await
        .unwrap()
        .error_for_status()
        .unwrap();
    let allowed = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth(key)
        .json(&request)
        .send()
        .await
        .unwrap();
    assert_eq!(
        allowed.status(),
        200,
        "detached key must not regain a recreated org via stale auth cache"
    );
    let _: Value = allowed.json().await.unwrap();

    client
        .post(handle.url("/project/delete"))
        .bearer_auth(MASTER_KEY)
        .json(&json!({"project_id":"detach-project"}))
        .send()
        .await
        .unwrap()
        .error_for_status()
        .unwrap();
    client
        .post(handle.url("/project/add"))
        .bearer_auth(MASTER_KEY)
        .json(
            &json!({"project_id":"detach-project", "project_name":"New project", "max_budget":0.0}),
        )
        .send()
        .await
        .unwrap()
        .error_for_status()
        .unwrap();
    let allowed = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth(key)
        .json(&request)
        .send()
        .await
        .unwrap();
    assert_eq!(
        allowed.status(),
        200,
        "recreated project must not reattach a cached key"
    );
    let _: Value = allowed.json().await.unwrap();
    let record = db.get_key(key_id).await.unwrap().unwrap();
    assert!(record.org_id.is_none() && record.project_id.is_none());
    assert!(record.is_active);
    assert_eq!(
        record.max_budget,
        Some(1.0),
        "independent key cap survives scope deletion"
    );
    assert_eq!(upstream.hits(), 2);
    shutdown(handle).await;
}

#[tokio::test]
async fn multi_tag_requests_settle_cache_and_durable_spend_for_every_tag() {
    let mock = MockServer::start();
    let (handle, db) = start_oxllm(proxy_config(&mock.base_url())).await;
    let upstream = mock_chat(&mock);
    let client = reqwest::Client::new();
    for name in ["app", "prod"] {
        client
            .post(handle.url("/tag/new"))
            .bearer_auth(MASTER_KEY)
            .json(&json!({"name":name, "max_budget":1.0}))
            .send()
            .await
            .unwrap()
            .error_for_status()
            .unwrap();
    }
    client
        .post(handle.url("/customer/new"))
        .bearer_auth(MASTER_KEY)
        .json(&json!({"end_user_id":"multi-customer", "max_budget":1.0}))
        .send()
        .await
        .unwrap()
        .error_for_status()
        .unwrap();
    let generated: Value = client
        .post(handle.url("/key/generate"))
        .bearer_auth(MASTER_KEY)
        .json(&json!({}))
        .send()
        .await
        .unwrap()
        .error_for_status()
        .unwrap()
        .json()
        .await
        .unwrap();
    let key = generated["key"].as_str().unwrap();
    for _ in 0..2 {
        let response = client
            .post(handle.url("/v1/chat/completions"))
            .bearer_auth(key)
            .header("x-oxllm-tag", "prod")
            .json(
                &json!({"model":"gpt-4o", "user":"multi-customer", "max_tokens":5,
                "metadata":{"tags":["label-only","app","prod","app"]},
                "messages":[{"role":"user","content":"hi"}]}),
            )
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let body: Value = response.json().await.unwrap();
        assert_eq!(body["usage"]["total_tokens"], 15);
    }
    let expected = 0.00015; // Two calls, each 10 input + 5 output gpt-4o tokens.
    let cache = handle.shared.load_full().cache.clone();
    wait_until("multi-tag reservations settled", || async {
        let app = cache.counter_value("spend:tag:app").await;
        let prod = cache.counter_value("spend:tag:prod").await;
        app.is_some_and(|v| (v - expected).abs() < 1e-12)
            && prod.is_some_and(|v| (v - expected).abs() < 1e-12)
    })
    .await;
    wait_until("multi-tag events durably flushed", || async {
        db.count_spend(None, None).await.unwrap() == 2
    })
    .await;
    for name in ["app", "prod"] {
        let row = db.get_tag(name).await.unwrap().unwrap();
        assert!((row.spend.unwrap() - expected).abs() < 1e-12, "{name}");
    }
    let customer = db.get_customer("multi-customer").await.unwrap().unwrap();
    assert!((customer.spend - expected).abs() < 1e-12);
    assert!(db.get_tag("label-only").await.unwrap().is_none());
    let groups = db.query_spend_by_tag(None, None).await.unwrap();
    assert_eq!(groups.len(), 3);
    for row in groups {
        assert_eq!(
            row.total_requests, 2,
            "repeated input tags must not duplicate attribution"
        );
        assert!((row.total_spend - expected).abs() < 1e-12);
    }
    assert_eq!(upstream.hits(), 2);
    shutdown(handle).await;
}

#[tokio::test]
async fn org_budget_blocks_request_when_exceeded() {
    let mock = MockServer::start();
    let (handle, db) = start_oxllm(proxy_config(&mock.base_url())).await;
    let upstream = mock_chat(&mock);
    let client = reqwest::Client::new();
    let auth = format!("Bearer {MASTER_KEY}");

    // Org with a tiny budget — first request consumes it; second must reject.
    client
        .post(handle.url("/organization/add"))
        .header("authorization", &auth)
        .json(&json!({"org_id": "org-blocked", "org_name": "Blocked", "max_budget": 0.0001}))
        .send()
        .await
        .expect("ok");

    let gen_resp = client
        .post(handle.url("/key/generate"))
        .header("authorization", &auth)
        .json(&json!({"org_id": "org-blocked"}))
        .send()
        .await
        .expect("ok");
    let gb: Value = gen_resp.json().await.expect("json");
    let raw_key = gb["key"].as_str().expect("key").to_string();

    let body = json!({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5});
    let first = client
        .post(handle.url("/v1/chat/completions"))
        .header("authorization", format!("Bearer {raw_key}"))
        .json(&body)
        .send()
        .await
        .expect("ok");
    assert_eq!(
        first.status(),
        200,
        "first request under org budget should succeed"
    );

    let first_body: Value = first.json().await.expect("completed response");
    assert_eq!(first_body["usage"]["total_tokens"], 15);
    wait_until("first org request metered", || async {
        db.count_spend(None, None).await.expect("count") == 1
    })
    .await;

    let second = client
        .post(handle.url("/v1/chat/completions"))
        .header("authorization", format!("Bearer {raw_key}"))
        .json(&body)
        .send()
        .await
        .expect("ok");
    assert_eq!(
        second.status(),
        429,
        "second request should be rejected after org budget exhausted"
    );
    assert_eq!(
        upstream.hits(),
        1,
        "budget rejection must not reach upstream"
    );

    shutdown(handle).await;
}

#[tokio::test]
async fn project_budget_blocks_request_when_exceeded() {
    let mock = MockServer::start();
    let (handle, db) = start_oxllm(proxy_config(&mock.base_url())).await;
    let upstream = mock_chat(&mock);
    let client = reqwest::Client::new();
    let auth = format!("Bearer {MASTER_KEY}");

    client
        .post(handle.url("/project/add"))
        .header("authorization", &auth)
        .json(&json!({"project_id": "project-blocked", "project_name": "Blocked", "max_budget": 0.0001}))
        .send()
        .await
        .expect("ok");

    let gen_resp = client
        .post(handle.url("/key/generate"))
        .header("authorization", &auth)
        .json(&json!({"project_id": "project-blocked"}))
        .send()
        .await
        .expect("ok");
    let gb: Value = gen_resp.json().await.expect("json");
    let raw_key = gb["key"].as_str().expect("key").to_string();

    let body = json!({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5});
    let first = client
        .post(handle.url("/v1/chat/completions"))
        .header("authorization", format!("Bearer {raw_key}"))
        .json(&body)
        .send()
        .await
        .expect("ok");
    assert_eq!(
        first.status(),
        200,
        "first request under project budget should succeed"
    );

    let first_body: Value = first.json().await.expect("completed response");
    assert_eq!(first_body["usage"]["total_tokens"], 15);
    wait_until("first project request metered", || async {
        db.count_spend(None, None).await.expect("count") == 1
    })
    .await;

    let second = client
        .post(handle.url("/v1/chat/completions"))
        .header("authorization", format!("Bearer {raw_key}"))
        .json(&body)
        .send()
        .await
        .expect("ok");
    assert_eq!(
        second.status(),
        429,
        "second request should be rejected after project budget exhausted"
    );
    assert_eq!(
        upstream.hits(),
        1,
        "budget rejection must not reach upstream"
    );

    shutdown(handle).await;
}

// ---------------------------------------------------------------------------
// P2-3: GET /routing/state — live router-state inspection
// ---------------------------------------------------------------------------

#[tokio::test]
async fn routing_state_reports_strategy_checks_and_deployments() {
    let mock = MockServer::start();
    let mut config = proxy_config(&mock.base_url());
    // Health-check routing on so the built-in check shows up in
    // `pre_call_checks` (nothing else is registered by default).
    config.router_settings.enable_health_check_routing = Some(true);
    let (handle, _db) = start_oxllm(config).await;

    let client = reqwest::Client::new();
    let auth = format!("Bearer {MASTER_KEY}");

    // Management auth applies: wrong key → 401.
    let unauth = client
        .get(handle.url("/routing/state"))
        .header("authorization", "Bearer wrong-key")
        .send()
        .await
        .expect("ok");
    assert_eq!(unauth.status(), 401);

    let resp = client
        .get(handle.url("/routing/state"))
        .header("authorization", &auth)
        .send()
        .await
        .expect("ok");
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("json");

    assert_eq!(body["strategy"].as_str(), Some("simple_shuffle"));
    assert!(
        body["group_strategies"]
            .as_object()
            .expect("object")
            .is_empty(),
        "no routing-group overrides configured"
    );
    assert_eq!(body["pre_call_checks"], json!(["health_check"]));

    let deployments = body["deployments"].as_array().expect("deployments");
    assert_eq!(deployments.len(), 2);

    let gpt = deployments
        .iter()
        .find(|d| d["model_name"] == "gpt-4o")
        .expect("gpt-4o present");
    assert_eq!(gpt["provider_model"], "openai/gpt-4o");
    assert_eq!(gpt["cooling"], false);
    assert_eq!(gpt["healthy"], true);
    assert!(gpt["health_reason"].is_null());
    assert!(
        gpt.get("bandit_cells").is_none(),
        "no bandit cells recorded yet → key omitted"
    );
    // No secrets leak through the endpoint.
    assert!(
        gpt.get("api_key").is_none(),
        "api keys must never appear in /routing/state"
    );

    let claude = deployments
        .iter()
        .find(|d| d["model_name"] == "claude-3")
        .expect("claude-3 present");
    assert_eq!(claude["provider_model"], "anthropic/claude-3");

    shutdown(handle).await;
}

// ---------------------------------------------------------------------------
// Customer / budget / tag / team-model / cache-settings CRUD
// ---------------------------------------------------------------------------

#[tokio::test]
async fn customer_budget_tag_team_model_and_cache_settings() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;
    let client = reqwest::Client::new();
    let auth = format!("Bearer {MASTER_KEY}");

    // --- customer ---
    let created = client
        .post(handle.url("/customer/new"))
        .header("authorization", &auth)
        .json(&json!({
            "end_user_id": "eu-alice",
            "alias": "Alice",
            "max_budget": 25.0
        }))
        .send()
        .await
        .expect("ok");
    assert_eq!(created.status(), 200, "customer new");
    let body: Value = created.json().await.expect("json");
    assert_eq!(body["end_user_id"], "eu-alice");

    let listed = client
        .get(handle.url("/customer/list"))
        .header("authorization", &auth)
        .send()
        .await
        .expect("ok");
    assert_eq!(listed.status(), 200);
    let list_body: Value = listed.json().await.expect("json");
    assert_eq!(list_body["customers"].as_array().map(|a| a.len()), Some(1));

    let info = client
        .get(handle.url("/customer/info?end_user_id=eu-alice"))
        .header("authorization", &auth)
        .send()
        .await
        .expect("ok");
    assert_eq!(info.status(), 200);
    let info_body: Value = info.json().await.expect("json");
    assert_eq!(info_body["alias"], "Alice");
    assert_eq!(info_body["blocked"], false);
    assert_eq!(info_body["spend"], 0.0);

    let blocked = client
        .post(handle.url("/customer/block"))
        .header("authorization", &auth)
        .json(&json!({"end_user_id": "eu-alice"}))
        .send()
        .await
        .expect("ok");
    assert_eq!(blocked.status(), 200);
    let info2 = client
        .get(handle.url("/customer/info?end_user_id=eu-alice"))
        .header("authorization", &auth)
        .send()
        .await
        .expect("ok");
    let info2_body: Value = info2.json().await.expect("json");
    assert_eq!(info2_body["blocked"], true);

    // --- budget ---
    let bnew = client
        .post(handle.url("/budget/new"))
        .header("authorization", &auth)
        .json(&json!({
            "budget_id": "budget-e2e",
            "max_budget": 100.0,
            "rpm_limit": 60
        }))
        .send()
        .await
        .expect("ok");
    assert_eq!(bnew.status(), 200);
    let blist = client
        .get(handle.url("/budget/list"))
        .header("authorization", &auth)
        .send()
        .await
        .expect("ok");
    assert_eq!(blist.status(), 200);
    let blist_body: Value = blist.json().await.expect("json");
    assert_eq!(blist_body["budgets"].as_array().map(|a| a.len()), Some(1));
    assert_eq!(blist_body["budgets"][0]["budget_id"], "budget-e2e");

    // --- tag ---
    let tnew = client
        .post(handle.url("/tag/new"))
        .header("authorization", &auth)
        .json(&json!({
            "name": "prod",
            "description": "production traffic",
            "models": ["gpt-4o"]
        }))
        .send()
        .await
        .expect("ok");
    assert_eq!(tnew.status(), 200);
    let tlist = client
        .get(handle.url("/tag/list"))
        .header("authorization", &auth)
        .send()
        .await
        .expect("ok");
    assert_eq!(tlist.status(), 200);
    let tlist_body: Value = tlist.json().await.expect("json");
    assert_eq!(tlist_body["tags"].as_array().map(|a| a.len()), Some(1));
    assert_eq!(tlist_body["tags"][0]["name"], "prod");

    // --- team model add/delete ---
    let team = client
        .post(handle.url("/team/add"))
        .header("authorization", &auth)
        .json(&json!({
            "team_id": "team-models",
            "team_name": "Models",
            "models": ["gpt-4o"]
        }))
        .send()
        .await
        .expect("ok");
    assert_eq!(team.status(), 200);

    let madd = client
        .post(handle.url("/team/model/add"))
        .header("authorization", &auth)
        .json(&json!({
            "team_id": "team-models",
            "models": ["claude-*", "gpt-4o"]
        }))
        .send()
        .await
        .expect("ok");
    assert_eq!(madd.status(), 200);
    let madd_body: Value = madd.json().await.expect("json");
    let models = madd_body["models"].as_array().expect("models");
    assert_eq!(models.len(), 2, "unique append: {models:?}");
    assert!(models.iter().any(|m| m == "gpt-4o"));
    assert!(models.iter().any(|m| m == "claude-*"));

    let mdel = client
        .post(handle.url("/team/model/delete"))
        .header("authorization", &auth)
        .json(&json!({
            "team_id": "team-models",
            "models": ["gpt-4o"]
        }))
        .send()
        .await
        .expect("ok");
    assert_eq!(mdel.status(), 200);
    let mdel_body: Value = mdel.json().await.expect("json");
    assert_eq!(mdel_body["models"], json!(["claude-*"]));

    // --- cache settings GET ---
    let cget = client
        .get(handle.url("/cache/settings"))
        .header("authorization", &auth)
        .send()
        .await
        .expect("ok");
    assert_eq!(cget.status(), 200);
    let cget_body: Value = cget.json().await.expect("json");
    assert!(cget_body.get("cache").is_some());
    assert!(cget_body.get("enabled").is_some());

    let cpost = client
        .post(handle.url("/cache/settings"))
        .header("authorization", &auth)
        .json(&json!({
            "cache": { "type": "memory" },
            "cache_ttl_secs": 120
        }))
        .send()
        .await
        .expect("ok");
    assert_eq!(cpost.status(), 200, "cache settings post");
    let cpost_body: Value = cpost.json().await.expect("json");
    assert_eq!(cpost_body["status"], "ok");
    assert_eq!(cpost_body["cache_ttl_secs"], 120);

    shutdown(handle).await;
}

#[tokio::test]
async fn routing_state_reports_cooling_after_upstream_failures() {
    let mock = MockServer::start();
    // gpt-4o's upstream always 500s. `num_retries: 0` exhausts the retry
    // loop on the first attempt (no exponential-backoff sleep), which cools
    // the deployment — keeping the test snappy.
    let _m = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions");
        then.status(500)
            .json_body(json!({"error": {"message": "boom"}}));
    });
    let mut config = proxy_config(&mock.base_url());
    config.router_settings.routing.num_retries = Some(0);
    let (handle, _db) = start_oxllm(config).await;

    let client = reqwest::Client::new();
    let auth = format!("Bearer {MASTER_KEY}");

    let chat = client
        .post(handle.url("/v1/chat/completions"))
        .header("authorization", &auth)
        .json(&json!({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .expect("ok");
    assert_eq!(chat.status(), 500, "exhausted retries surface the 500");

    let resp = client
        .get(handle.url("/routing/state"))
        .header("authorization", &auth)
        .send()
        .await
        .expect("ok");
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("json");
    let deployments = body["deployments"].as_array().expect("deployments");

    let gpt = deployments
        .iter()
        .find(|d| d["model_name"] == "gpt-4o")
        .expect("gpt-4o present");
    assert_eq!(
        gpt["cooling"], true,
        "failed deployment must be cooling: {gpt}"
    );
    let claude = deployments
        .iter()
        .find(|d| d["model_name"] == "claude-3")
        .expect("claude-3 present");
    assert_eq!(
        claude["cooling"], false,
        "untouched deployment is not cooling"
    );

    shutdown(handle).await;
}

// ---------------------------------------------------------------------------
// C30: POST /model/update must redact provider secrets from its response.
// At-rest encryption only covers `oxllm_params.api_key`; every other
// provider secret (AWS Bedrock signing material, custom Authorization headers)
// is stored verbatim and would otherwise leak via this endpoint.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn model_update_redacts_secrets_in_response() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;

    let client = reqwest::Client::new();
    let auth = format!("Bearer {MASTER_KEY}");

    // Add a Bedrock-style deployment carrying several non-api_key secrets.
    let add = client
        .post(handle.url("/model/new"))
        .header("authorization", &auth)
        .json(&add_model_body(
            "bedrock-claude",
            json!({
                "model": "bedrock/anthropic.claude-3",
                "aws_secret_access_key": "AWSSECRET123",
                "aws_session_token": "AQoEXAMPLEH4aoAH0eNSPE",
                "authorization": "Bearer secret-bearer",
                "api_base": "https://bedrock-runtime.us-east-1.amazonaws.com",
            }),
        ))
        .send()
        .await
        .expect("ok");
    assert_eq!(add.status(), 200);
    let ab: Value = add.json().await.expect("json");
    let model_id = ab["model_id"].as_str().expect("model_id").to_string();

    // Update with a fresh api_key + the same secret fields; the response must
    // not echo any of the plaintext secret values back.
    let upd = client
        .post(handle.url("/model/update"))
        .header("authorization", &auth)
        .json(&json!({
            "model_id": model_id,
            "oxllm_params": {
                "model": "bedrock/anthropic.claude-3",
                "api_key": "sk-leaked-openai-key",
                "aws_secret_access_key": "AWSSECRET123",
                "aws_session_token": "AQoEXAMPLEH4aoAH0eNSPE",
                "authorization": "Bearer secret-bearer",
                "api_base": "https://bedrock-runtime.us-east-1.amazonaws.com",
            }
        }))
        .send()
        .await
        .expect("ok");
    assert_eq!(upd.status(), 200);
    let body_text = upd.text().await.expect("body text");
    assert!(
        !body_text.contains("AWSSECRET123"),
        "response must not leak aws_secret_access_key: {body_text}"
    );
    assert!(
        !body_text.contains("AQoEXAMPLEH4aoAH0eNSPE"),
        "response must not leak aws_session_token: {body_text}"
    );
    assert!(
        !body_text.contains("secret-bearer"),
        "response must not leak authorization header: {body_text}"
    );
    assert!(
        !body_text.contains("sk-leaked-openai-key"),
        "response must not leak api_key: {body_text}"
    );
    assert!(
        body_text.contains("***"),
        "response must contain the redaction placeholder"
    );
    // Non-secret fields survive redaction.
    assert!(
        body_text.contains("https://bedrock-runtime.us-east-1.amazonaws.com"),
        "non-secret api_base must remain: {body_text}"
    );

    shutdown(handle).await;
}

// ---------------------------------------------------------------------------
// C30 MEDIUM: a partial update_model must not silently wipe a previously
// stored api_key. update_model now deep-merges the new oxllm_params into the
// stored ones, so patching `weight` keeps the api_key. The definitive proof is
// a chat round-trip that still authenticates with the original key.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn model_update_partial_patch_preserves_stored_api_key() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;

    let _m = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .header("authorization", "Bearer sk-keep");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "id": "chatcmpl-keep",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            }));
    });

    let client = reqwest::Client::new();
    let auth = format!("Bearer {MASTER_KEY}");

    let add = client
        .post(handle.url("/model/new"))
        .header("authorization", &auth)
        .json(&add_model_body(
            "patched",
            json!({
                "model": "openai/gpt-4o",
                "api_key": "sk-keep",
                "api_base": mock.base_url(),
                "weight": 1,
            }),
        ))
        .send()
        .await
        .expect("ok");
    assert_eq!(add.status(), 200);
    let ab: Value = add.json().await.expect("json");
    let model_id = ab["model_id"].as_str().expect("model_id").to_string();

    // Partial patch: only tweak `weight`. The api_key is omitted, so a
    // full-replace would wipe it; a deep-merge preserves it.
    let upd = client
        .post(handle.url("/model/update"))
        .header("authorization", &auth)
        .json(&json!({
            "model_id": model_id,
            "oxllm_params": {"weight": 2}
        }))
        .send()
        .await
        .expect("ok");
    assert_eq!(upd.status(), 200);
    let ub: Value = upd.json().await.expect("json");
    assert_eq!(ub["oxllm_params"]["weight"], 2);
    // The api_key field is still present (redacted in the response).
    assert_eq!(ub["oxllm_params"]["api_key"], "***");

    // The model still authenticates upstream with the ORIGINAL key — the
    // definitive proof that the partial patch did not clear the credential.
    let chat = client
        .post(handle.url("/v1/chat/completions"))
        .header("authorization", &auth)
        .json(&json!({
            "model": "patched",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("ok");
    assert_eq!(
        chat.status(),
        200,
        "chat must still authenticate with the preserved api_key: {}",
        chat.text().await.unwrap_or_default()
    );

    shutdown(handle).await;
}

// ---------------------------------------------------------------------------
// C30: POST /cache/settings must reject contradictory field combinations.
// Previously `clear_cache: true` silently won over an inline `cache`, so an
// operator PATCHing both fields ended up with caching disabled and no error.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn cache_settings_rejects_clear_cache_and_cache_together() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;

    let client = reqwest::Client::new();
    let auth = format!("Bearer {MASTER_KEY}");

    let resp = client
        .post(handle.url("/cache/settings"))
        .header("authorization", &auth)
        .json(&json!({
            "clear_cache": true,
            "cache": {"type": "memory"},
        }))
        .send()
        .await
        .expect("ok");
    assert_eq!(resp.status(), 400);
    let body: Value = resp.json().await.expect("json");
    let msg = body["error"]["message"].as_str().unwrap_or("");
    assert!(
        msg.contains("clear_cache") && msg.contains("cache"),
        "error should explain the conflict, got: {msg}"
    );

    shutdown(handle).await;
}

#[tokio::test]
async fn cache_settings_rejects_clear_ttl_and_ttl_together() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;

    let client = reqwest::Client::new();
    let auth = format!("Bearer {MASTER_KEY}");

    let resp = client
        .post(handle.url("/cache/settings"))
        .header("authorization", &auth)
        .json(&json!({
            "clear_cache_ttl_secs": true,
            "cache_ttl_secs": 120,
        }))
        .send()
        .await
        .expect("ok");
    assert_eq!(resp.status(), 400);

    shutdown(handle).await;
}

// ---------------------------------------------------------------------------
// C30: GET /model/list must surface every provider backing a shared
// model_name (e.g. an OpenAI + Azure failover group), not just the first
// deployment that happened to win the dedup.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn model_list_surfaces_all_providers_for_shared_model_name() {
    let mock = MockServer::start();

    // A config with one model_name backed by two providers.
    let mut config = proxy_config(&mock.base_url());
    config.model_list.push(ConfigModel {
        model_name: "gpt-4o".to_string(),
        oxllm_params: OxllmParams {
            model: "azure/gpt-4o".to_string(),
            api_key: Some("sk-azure".to_string()),
            api_base: Some(mock.base_url()),
            ..Default::default()
        },
        ..Default::default()
    });
    let (handle, _db) = start_oxllm(config).await;

    let client = reqwest::Client::new();
    let auth = format!("Bearer {MASTER_KEY}");

    let list = client
        .get(handle.url("/model/list"))
        .header("authorization", &auth)
        .send()
        .await
        .expect("ok");
    assert_eq!(list.status(), 200);
    let lb: Value = list.json().await.expect("json");
    let data = lb["data"].as_array().expect("data array");

    let gpt = data
        .iter()
        .find(|m| m["id"].as_str() == Some("gpt-4o"))
        .expect("gpt-4o group present");
    let providers = gpt["providers"].as_array().expect("providers array");
    let provider_slugs: Vec<&str> = providers.iter().filter_map(|p| p.as_str()).collect();
    assert!(
        provider_slugs.contains(&"openai") && provider_slugs.contains(&"azure"),
        "both providers must be reported, got: {provider_slugs:?}"
    );

    shutdown(handle).await;
}

// ---------------------------------------------------------------------------
// C27 HIGH: /key/list truncated flag must actually fire when there are more
// than 1000 keys. Previously the DB repo clamped LIMIT to 1000, so the
// handler's `keys.len() > 1000` check was dead code and `truncated` was
// always false.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn key_list_truncated_flag_fires_over_1000_rows() {
    let mock = MockServer::start();
    let (handle, db) = start_oxllm(proxy_config(&mock.base_url())).await;

    let now = oxllm_server::now_iso();
    for i in 0..1001 {
        db.create_key(&oxllm_server::KeyRecord {
            key_id: format!("k-{i}"),
            key_hash: format!("h-{i}"),
            key_alias: None,
            team_id: None,
            user_id: None,
            org_id: None,
            project_id: None,
            max_budget: None,
            budget_duration: None,
            rpm: None,
            tpm: None,
            created_at: now.clone(),
            updated_at: now.clone(),
            expires_at: None,
            is_active: true,
            models: None,
            allowed_cache_controls: None,
        })
        .await
        .expect("create");
    }

    let client = reqwest::Client::new();
    let resp = client
        .get(handle.url("/key/list"))
        .header("authorization", format!("Bearer {MASTER_KEY}"))
        .send()
        .await
        .expect("ok");
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("json");
    assert_eq!(
        body["truncated"].as_bool(),
        Some(true),
        "truncated must be true when >1000 rows exist"
    );
    let keys = body["keys"].as_array().expect("keys array");
    assert_eq!(keys.len(), 1000, "must be capped at 1000");

    shutdown(handle).await;
}

// ---------------------------------------------------------------------------
// C27 HIGH: /team/unblock must NOT re-enable keys that were individually
// blocked via /key/block. Team block/unblock now uses a `blocked` column on
// the team row, not per-key `is_active`.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn team_unblock_preserves_individually_blocked_keys() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;
    let _m = mock_chat(&mock);

    let client = reqwest::Client::new();
    let auth = format!("Bearer {MASTER_KEY}");

    // Create a team.
    let resp = client
        .post(handle.url("/team/add"))
        .header("authorization", &auth)
        .json(&json!({"team_name": "Eng"}))
        .send()
        .await
        .expect("ok");
    let tb: Value = resp.json().await.expect("json");
    let team_id = tb["team_id"].as_str().expect("team_id").to_string();

    // Create two keys in the team.
    let mut keys = Vec::new();
    for _ in 0..2 {
        let resp = client
            .post(handle.url("/key/generate"))
            .header("authorization", &auth)
            .json(&json!({"team_id": team_id}))
            .send()
            .await
            .expect("ok");
        let body: Value = resp.json().await.expect("json");
        keys.push((
            body["key"].as_str().expect("raw key").to_string(),
            body["key_id"].as_str().expect("key_id").to_string(),
        ));
    }
    let (k1_raw, k1_id) = &keys[0];
    let (k2_raw, _k2_id) = &keys[1];

    // Individually block K1.
    let resp = client
        .post(handle.url("/key/block"))
        .header("authorization", &auth)
        .json(&json!({"key_id": k1_id}))
        .send()
        .await
        .expect("ok");
    assert_eq!(resp.status(), 200);

    // Block then unblock the team.
    let resp = client
        .post(handle.url("/team/block"))
        .header("authorization", &auth)
        .json(&json!({"team_id": team_id}))
        .send()
        .await
        .expect("ok");
    assert_eq!(resp.status(), 200);

    let resp = client
        .post(handle.url("/team/unblock"))
        .header("authorization", &auth)
        .json(&json!({"team_id": team_id}))
        .send()
        .await
        .expect("ok");
    assert_eq!(resp.status(), 200);

    // K1 (individually blocked) must stay rejected.
    let chat = client
        .post(handle.url("/v1/chat/completions"))
        .header("authorization", format!("Bearer {k1_raw}"))
        .json(&json!({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .expect("ok");
    assert_eq!(
        chat.status(),
        401,
        "individually blocked key must stay rejected after team/unblock"
    );

    // K2 (not individually blocked) must work again.
    let chat = client
        .post(handle.url("/v1/chat/completions"))
        .header("authorization", format!("Bearer {k2_raw}"))
        .json(&json!({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .expect("ok");
    assert_eq!(
        chat.status(),
        200,
        "unblocked key in unblocked team must authenticate"
    );

    shutdown(handle).await;
}

// ---------------------------------------------------------------------------
// C27 HIGH: /team/block must actually block all keys in the team.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn team_block_rejects_all_keys_in_team() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;
    let _m = mock_chat(&mock);

    let client = reqwest::Client::new();
    let auth = format!("Bearer {MASTER_KEY}");

    let resp = client
        .post(handle.url("/team/add"))
        .header("authorization", &auth)
        .json(&json!({"team_name": "Blocked"}))
        .send()
        .await
        .expect("ok");
    let tb: Value = resp.json().await.expect("json");
    let team_id = tb["team_id"].as_str().expect("team_id").to_string();

    let resp = client
        .post(handle.url("/key/generate"))
        .header("authorization", &auth)
        .json(&json!({"team_id": team_id}))
        .send()
        .await
        .expect("ok");
    let body: Value = resp.json().await.expect("json");
    let raw_key = body["key"].as_str().expect("raw key").to_string();

    // Before block: works.
    let chat = client
        .post(handle.url("/v1/chat/completions"))
        .header("authorization", format!("Bearer {raw_key}"))
        .json(&json!({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .expect("ok");
    assert_eq!(chat.status(), 200);

    // Block the team.
    let resp = client
        .post(handle.url("/team/block"))
        .header("authorization", &auth)
        .json(&json!({"team_id": team_id}))
        .send()
        .await
        .expect("ok");
    assert_eq!(resp.status(), 200);

    // After block: rejected.
    let chat = client
        .post(handle.url("/v1/chat/completions"))
        .header("authorization", format!("Bearer {raw_key}"))
        .json(&json!({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .expect("ok");
    assert_eq!(chat.status(), 401, "key in blocked team must be rejected");

    shutdown(handle).await;
}

// ---------------------------------------------------------------------------
// C27 MEDIUM: budget_duration validation at the API boundary.
// Malformed value → 400, valid value → 200.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn generate_key_rejects_malformed_budget_duration() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/key/generate"))
        .header("authorization", format!("Bearer {MASTER_KEY}"))
        .json(&json!({"budget_duration": "1x"}))
        .send()
        .await
        .expect("ok");
    assert_eq!(resp.status(), 400, "malformed budget_duration must be 400");

    shutdown(handle).await;
}

#[tokio::test]
async fn generate_key_accepts_valid_budget_duration() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/key/generate"))
        .header("authorization", format!("Bearer {MASTER_KEY}"))
        .json(&json!({"budget_duration": "30d"}))
        .send()
        .await
        .expect("ok");
    assert_eq!(resp.status(), 200, "valid budget_duration must succeed");

    shutdown(handle).await;
}

#[tokio::test]
async fn add_team_rejects_malformed_budget_duration() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/team/add"))
        .header("authorization", format!("Bearer {MASTER_KEY}"))
        .json(&json!({"team_name": "T", "budget_duration": "oops"}))
        .send()
        .await
        .expect("ok");
    assert_eq!(resp.status(), 400);

    shutdown(handle).await;
}

#[tokio::test]
async fn add_user_rejects_malformed_budget_duration() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/user/add"))
        .header("authorization", format!("Bearer {MASTER_KEY}"))
        .json(&json!({"user_name": "U", "budget_duration": "xx"}))
        .send()
        .await
        .expect("ok");
    assert_eq!(resp.status(), 400);

    shutdown(handle).await;
}

// ---------------------------------------------------------------------------
// C27 LOW: /key/update can now change expires.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn update_key_can_change_expires() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;

    let client = reqwest::Client::new();
    let auth = format!("Bearer {MASTER_KEY}");

    let resp = client
        .post(handle.url("/key/generate"))
        .header("authorization", &auth)
        .json(&json!({}))
        .send()
        .await
        .expect("ok");
    let body: Value = resp.json().await.expect("json");
    let key_id = body["key_id"].as_str().expect("key_id").to_string();

    // Update with a future expiry.
    let resp = client
        .post(handle.url("/key/update"))
        .header("authorization", &auth)
        .json(&json!({"key_id": key_id, "expires": "2999-01-01T00:00:00Z"}))
        .send()
        .await
        .expect("ok");
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("json");
    assert_eq!(body["expires_at"].as_str(), Some("2999-01-01T00:00:00Z"));

    // Malformed expiry → 400.
    let resp = client
        .post(handle.url("/key/update"))
        .header("authorization", &auth)
        .json(&json!({"key_id": key_id, "expires": "not-a-date"}))
        .send()
        .await
        .expect("ok");
    assert_eq!(resp.status(), 400);

    // Empty string clears the expiry.
    let resp = client
        .post(handle.url("/key/update"))
        .header("authorization", &auth)
        .json(&json!({"key_id": key_id, "expires": ""}))
        .send()
        .await
        .expect("ok");
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("json");
    assert!(
        body.get("expires_at").is_none() || body["expires_at"].is_null(),
        "expires_at should be cleared"
    );

    shutdown(handle).await;
}

// ---------------------------------------------------------------------------
// C27 LOW: /key/generate rejects non-finite max_budget (NaN / Infinity).
// serde_json rejects these at the parse boundary, but defense-in-depth.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn generate_key_rejects_nan_max_budget_at_api() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/key/generate"))
        .header("authorization", format!("Bearer {MASTER_KEY}"))
        .json(&json!({"max_budget": -1.0}))
        .send()
        .await
        .expect("ok");
    assert_eq!(resp.status(), 400);

    shutdown(handle).await;
}

// ---------------------------------------------------------------------------
// C28 MEDIUM: validate_budget_duration must be enforced at the API boundary
// for org / project / customer create and update (previously only enforced
// for keys / teams / users, so a typo like "1x" silently fell back to the
// 30d default at enforcement time).
// ---------------------------------------------------------------------------

#[tokio::test]
async fn org_add_rejects_malformed_budget_duration() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;
    let client = reqwest::Client::new();
    let auth = format!("Bearer {MASTER_KEY}");

    let resp = client
        .post(handle.url("/organization/add"))
        .header("authorization", &auth)
        .json(&json!({"org_name": "Acme", "budget_duration": "1x"}))
        .send()
        .await
        .expect("ok");
    assert_eq!(resp.status(), 400);

    shutdown(handle).await;
}

#[tokio::test]
async fn org_add_accepts_valid_budget_duration() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;
    let client = reqwest::Client::new();
    let auth = format!("Bearer {MASTER_KEY}");

    let resp = client
        .post(handle.url("/organization/add"))
        .header("authorization", &auth)
        .json(&json!({"org_name": "Acme", "budget_duration": "1mo"}))
        .send()
        .await
        .expect("ok");
    assert_eq!(resp.status(), 200);

    shutdown(handle).await;
}

#[tokio::test]
async fn org_update_rejects_malformed_budget_duration() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;
    let client = reqwest::Client::new();
    let auth = format!("Bearer {MASTER_KEY}");

    let add = client
        .post(handle.url("/organization/add"))
        .header("authorization", &auth)
        .json(&json!({"org_id": "org-bd", "org_name": "Acme"}))
        .send()
        .await
        .expect("ok");
    assert_eq!(add.status(), 200);

    let resp = client
        .post(handle.url("/organization/update"))
        .header("authorization", &auth)
        .json(&json!({"org_id": "org-bd", "budget_duration": "30days"}))
        .send()
        .await
        .expect("ok");
    assert_eq!(resp.status(), 400);

    shutdown(handle).await;
}

#[tokio::test]
async fn project_add_rejects_malformed_budget_duration() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;
    let client = reqwest::Client::new();
    let auth = format!("Bearer {MASTER_KEY}");

    let resp = client
        .post(handle.url("/project/add"))
        .header("authorization", &auth)
        .json(&json!({"project_name": "Atlas", "budget_duration": "1x"}))
        .send()
        .await
        .expect("ok");
    assert_eq!(resp.status(), 400);

    shutdown(handle).await;
}

#[tokio::test]
async fn project_update_rejects_malformed_budget_duration() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;
    let client = reqwest::Client::new();
    let auth = format!("Bearer {MASTER_KEY}");

    let add = client
        .post(handle.url("/project/add"))
        .header("authorization", &auth)
        .json(&json!({"project_id": "project-bd", "project_name": "Atlas"}))
        .send()
        .await
        .expect("ok");
    assert_eq!(add.status(), 200);

    let resp = client
        .post(handle.url("/project/update"))
        .header("authorization", &auth)
        .json(&json!({"project_id": "project-bd", "budget_duration": "oops"}))
        .send()
        .await
        .expect("ok");
    assert_eq!(resp.status(), 400);

    shutdown(handle).await;
}

#[tokio::test]
async fn customer_new_rejects_malformed_budget_duration() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;
    let client = reqwest::Client::new();
    let auth = format!("Bearer {MASTER_KEY}");

    let resp = client
        .post(handle.url("/customer/new"))
        .header("authorization", &auth)
        .json(&json!({"end_user_id": "eu-x", "budget_duration": "1x"}))
        .send()
        .await
        .expect("ok");
    assert_eq!(resp.status(), 400);

    shutdown(handle).await;
}

#[tokio::test]
async fn customer_update_rejects_malformed_budget_duration() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;
    let client = reqwest::Client::new();
    let auth = format!("Bearer {MASTER_KEY}");

    let add = client
        .post(handle.url("/customer/new"))
        .header("authorization", &auth)
        .json(&json!({"end_user_id": "eu-bd"}))
        .send()
        .await
        .expect("ok");
    assert_eq!(add.status(), 200);

    let resp = client
        .post(handle.url("/customer/update"))
        .header("authorization", &auth)
        .json(&json!({"end_user_id": "eu-bd", "budget_duration": "xx"}))
        .send()
        .await
        .expect("ok");
    assert_eq!(resp.status(), 400);

    shutdown(handle).await;
}

// ---------------------------------------------------------------------------
// C28 MEDIUM: validate_limits (negative / non-finite max_budget) is now
// enforced for customer and tag create/update. serde_json rejects NaN/Inf
// at the parse boundary, but a negative value reaches the handler.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn customer_new_rejects_negative_max_budget() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;
    let client = reqwest::Client::new();
    let auth = format!("Bearer {MASTER_KEY}");

    let resp = client
        .post(handle.url("/customer/new"))
        .header("authorization", &auth)
        .json(&json!({"end_user_id": "eu-neg", "max_budget": -0.01}))
        .send()
        .await
        .expect("ok");
    assert_eq!(resp.status(), 400);

    shutdown(handle).await;
}

#[tokio::test]
async fn customer_update_rejects_negative_max_budget() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;
    let client = reqwest::Client::new();
    let auth = format!("Bearer {MASTER_KEY}");

    let add = client
        .post(handle.url("/customer/new"))
        .header("authorization", &auth)
        .json(&json!({"end_user_id": "eu-neg2"}))
        .send()
        .await
        .expect("ok");
    assert_eq!(add.status(), 200);

    let resp = client
        .post(handle.url("/customer/update"))
        .header("authorization", &auth)
        .json(&json!({"end_user_id": "eu-neg2", "max_budget": -1.0}))
        .send()
        .await
        .expect("ok");
    assert_eq!(resp.status(), 400);

    shutdown(handle).await;
}

#[tokio::test]
async fn tag_new_rejects_negative_max_budget() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;
    let client = reqwest::Client::new();
    let auth = format!("Bearer {MASTER_KEY}");

    let resp = client
        .post(handle.url("/tag/new"))
        .header("authorization", &auth)
        .json(&json!({"name": "bad-tag", "max_budget": -5.0}))
        .send()
        .await
        .expect("ok");
    assert_eq!(resp.status(), 400);

    shutdown(handle).await;
}

#[tokio::test]
async fn tag_new_accepts_positive_max_budget() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;
    let client = reqwest::Client::new();
    let auth = format!("Bearer {MASTER_KEY}");

    let add = client
        .post(handle.url("/tag/new"))
        .header("authorization", &auth)
        .json(&json!({"name": "budget-tag", "max_budget": 10.0}))
        .send()
        .await
        .expect("ok");
    assert_eq!(add.status(), 200);

    let info = client
        .get(handle.url("/tag/info?name=budget-tag"))
        .header("authorization", &auth)
        .send()
        .await
        .expect("ok");
    assert_eq!(info.status(), 200);
    let body: Value = info.json().await.expect("json");
    assert_eq!(body["max_budget"].as_f64(), Some(10.0));

    let upd = client
        .post(handle.url("/tag/update"))
        .header("authorization", &auth)
        .json(&json!({"name": "budget-tag", "max_budget": 25.0}))
        .send()
        .await
        .expect("ok");
    assert_eq!(upd.status(), 200);
    let ub: Value = upd.json().await.expect("json");
    assert_eq!(ub["max_budget"].as_f64(), Some(25.0));

    shutdown(handle).await;
}

// ---------------------------------------------------------------------------
// C28 MEDIUM: `/customer/delete` now accepts LiteLLM's bulk `user_ids` shape
// (alongside the legacy single `end_user_id`), with all-or-404 semantics.
// ---------------------------------------------------------------------------

async fn c28_create_customer(handle: &Handle, client: &reqwest::Client, id: &str) {
    let auth = format!("Bearer {MASTER_KEY}");
    let resp = client
        .post(handle.url("/customer/new"))
        .header("authorization", &auth)
        .json(&json!({"end_user_id": id}))
        .send()
        .await
        .expect("ok");
    assert_eq!(resp.status(), 200, "create customer {id}");
}

#[tokio::test]
async fn customer_delete_single_id_preserves_legacy_shape() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;
    let client = reqwest::Client::new();
    c28_create_customer(&handle, &client, "eu-single").await;

    let resp = client
        .post(handle.url("/customer/delete"))
        .header("authorization", format!("Bearer {MASTER_KEY}"))
        .json(&json!({"end_user_id": "eu-single"}))
        .send()
        .await
        .expect("ok");
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("json");
    assert_eq!(body["status"], "deleted");
    assert_eq!(body["end_user_id"], "eu-single");
    assert!(
        body.get("deleted_customers").is_none(),
        "single-id response must keep the legacy shape"
    );

    let info = client
        .get(handle.url("/customer/info?end_user_id=eu-single"))
        .header("authorization", format!("Bearer {MASTER_KEY}"))
        .send()
        .await
        .expect("ok");
    assert_eq!(info.status(), 404);

    shutdown(handle).await;
}

#[tokio::test]
async fn customer_delete_bulk_user_ids_deletes_all() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;
    let client = reqwest::Client::new();
    c28_create_customer(&handle, &client, "eu-a").await;
    c28_create_customer(&handle, &client, "eu-b").await;
    c28_create_customer(&handle, &client, "eu-c").await;

    let resp = client
        .post(handle.url("/customer/delete"))
        .header("authorization", format!("Bearer {MASTER_KEY}"))
        .json(&json!({"user_ids": ["eu-a", "eu-b", "eu-c"]}))
        .send()
        .await
        .expect("ok");
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("json");
    assert_eq!(body["status"], "deleted");
    assert_eq!(body["deleted_customers"], 3);

    for id in ["eu-a", "eu-b", "eu-c"] {
        let info = client
            .get(handle.url(&format!("/customer/info?end_user_id={id}")))
            .header("authorization", format!("Bearer {MASTER_KEY}"))
            .send()
            .await
            .expect("ok");
        assert_eq!(info.status(), 404, "customer {id} should be gone");
    }

    shutdown(handle).await;
}

#[tokio::test]
async fn customer_delete_bulk_is_atomic_on_missing_id() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;
    let client = reqwest::Client::new();
    c28_create_customer(&handle, &client, "eu-real").await;

    // One id does not exist — request must 404 AND not delete anything.
    let resp = client
        .post(handle.url("/customer/delete"))
        .header("authorization", format!("Bearer {MASTER_KEY}"))
        .json(&json!({"user_ids": ["eu-real", "eu-ghost"]}))
        .send()
        .await
        .expect("ok");
    assert_eq!(resp.status(), 404);
    let body: Value = resp.json().await.expect("json");
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap_or("")
            .contains("eu-ghost"),
        "404 should name the missing id; got: {body}"
    );

    // The existing customer must still be there (all-or-nothing).
    let info = client
        .get(handle.url("/customer/info?end_user_id=eu-real"))
        .header("authorization", format!("Bearer {MASTER_KEY}"))
        .send()
        .await
        .expect("ok");
    assert_eq!(info.status(), 200, "existing customer must be untouched");

    shutdown(handle).await;
}

#[tokio::test]
async fn customer_delete_combines_end_user_id_and_user_ids() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;
    let client = reqwest::Client::new();
    c28_create_customer(&handle, &client, "eu-mix-a").await;
    c28_create_customer(&handle, &client, "eu-mix-b").await;

    // Both fields: dedupe (eu-mix-a appears in both) and delete the union.
    let resp = client
        .post(handle.url("/customer/delete"))
        .header("authorization", format!("Bearer {MASTER_KEY}"))
        .json(&json!({"end_user_id": "eu-mix-a", "user_ids": ["eu-mix-a", "eu-mix-b"]}))
        .send()
        .await
        .expect("ok");
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("json");
    assert_eq!(body["deleted_customers"], 2);

    shutdown(handle).await;
}

#[tokio::test]
async fn customer_delete_requires_end_user_id_or_user_ids() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;
    let client = reqwest::Client::new();

    let resp = client
        .post(handle.url("/customer/delete"))
        .header("authorization", format!("Bearer {MASTER_KEY}"))
        .json(&json!({}))
        .send()
        .await
        .expect("ok");
    assert_eq!(resp.status(), 400);

    shutdown(handle).await;
}

// ---------------------------------------------------------------------------
// C28 MEDIUM: stale `200 OK` on racy delete. A row deleted between the
// handler's `get_X` and its `update_X` must now surface as a 404, not a
// 200 with a stale body. We simulate the race by deleting the row directly
// via the DB handle between the get and the update.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn update_org_returns_404_when_row_deleted_midflight() {
    let mock = MockServer::start();
    let (handle, db) = start_oxllm(proxy_config(&mock.base_url())).await;
    let client = reqwest::Client::new();
    let auth = format!("Bearer {MASTER_KEY}");

    let add = client
        .post(handle.url("/organization/add"))
        .header("authorization", &auth)
        .json(&json!({"org_id": "org-race", "org_name": "Acme"}))
        .send()
        .await
        .expect("ok");
    assert_eq!(add.status(), 200);

    // Simulate the concurrent delete by removing the row directly.
    db.delete_organization("org-race").await.expect("delete");

    let resp = client
        .post(handle.url("/organization/update"))
        .header("authorization", &auth)
        .json(&json!({"org_id": "org-race", "org_name": "Renamed"}))
        .send()
        .await
        .expect("ok");
    assert_eq!(
        resp.status(),
        404,
        "update on a deleted row must surface as 404, not stale 200"
    );

    shutdown(handle).await;
}

#[tokio::test]
async fn update_project_returns_404_when_row_deleted_midflight() {
    let mock = MockServer::start();
    let (handle, db) = start_oxllm(proxy_config(&mock.base_url())).await;
    let client = reqwest::Client::new();
    let auth = format!("Bearer {MASTER_KEY}");

    let add = client
        .post(handle.url("/project/add"))
        .header("authorization", &auth)
        .json(&json!({"project_id": "project-race", "project_name": "Atlas"}))
        .send()
        .await
        .expect("ok");
    assert_eq!(add.status(), 200);

    db.delete_project("project-race").await.expect("delete");

    let resp = client
        .post(handle.url("/project/update"))
        .header("authorization", &auth)
        .json(&json!({"project_id": "project-race", "project_name": "Renamed"}))
        .send()
        .await
        .expect("ok");
    assert_eq!(resp.status(), 404);

    shutdown(handle).await;
}

#[tokio::test]
async fn update_customer_returns_404_when_row_deleted_midflight() {
    let mock = MockServer::start();
    let (handle, db) = start_oxllm(proxy_config(&mock.base_url())).await;
    let client = reqwest::Client::new();
    let auth = format!("Bearer {MASTER_KEY}");

    let add = client
        .post(handle.url("/customer/new"))
        .header("authorization", &auth)
        .json(&json!({"end_user_id": "eu-race"}))
        .send()
        .await
        .expect("ok");
    assert_eq!(add.status(), 200);

    db.delete_customer("eu-race").await.expect("delete");

    let resp = client
        .post(handle.url("/customer/update"))
        .header("authorization", &auth)
        .json(&json!({"end_user_id": "eu-race", "alias": "Renamed"}))
        .send()
        .await
        .expect("ok");
    assert_eq!(resp.status(), 404);

    shutdown(handle).await;
}

#[tokio::test]
async fn update_tag_returns_404_when_row_deleted_midflight() {
    let mock = MockServer::start();
    let (handle, db) = start_oxllm(proxy_config(&mock.base_url())).await;
    let client = reqwest::Client::new();
    let auth = format!("Bearer {MASTER_KEY}");

    let add = client
        .post(handle.url("/tag/new"))
        .header("authorization", &auth)
        .json(&json!({"name": "tag-race"}))
        .send()
        .await
        .expect("ok");
    assert_eq!(add.status(), 200);

    db.delete_tag("tag-race").await.expect("delete");

    let resp = client
        .post(handle.url("/tag/update"))
        .header("authorization", &auth)
        .json(&json!({"name": "tag-race", "description": "updated"}))
        .send()
        .await
        .expect("ok");
    assert_eq!(resp.status(), 404);

    shutdown(handle).await;
}

// ---------------------------------------------------------------------------
// C28 LOW: org `models` is now typed as `Vec<String>` (consistent with
// tag/customer `models`/`model_blocklist`). An object payload is rejected
// at the API boundary.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn org_add_rejects_non_array_models() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;
    let client = reqwest::Client::new();
    let auth = format!("Bearer {MASTER_KEY}");

    let resp = client
        .post(handle.url("/organization/add"))
        .header("authorization", &auth)
        .json(&json!({"org_name": "Acme", "models": {"gpt-4o": true}}))
        .send()
        .await
        .expect("ok");
    assert_eq!(
        resp.status(),
        422,
        "object payload should fail deserialization"
    );

    shutdown(handle).await;
}

#[tokio::test]
async fn org_add_persists_array_models() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;
    let client = reqwest::Client::new();
    let auth = format!("Bearer {MASTER_KEY}");

    let resp = client
        .post(handle.url("/organization/add"))
        .header("authorization", &auth)
        .json(
            &json!({"org_id": "org-models", "org_name": "Acme", "models": ["gpt-4o", "claude-*"]}),
        )
        .send()
        .await
        .expect("ok");
    assert_eq!(resp.status(), 200);

    let info = client
        .get(handle.url("/organization/info?org_id=org-models"))
        .header("authorization", &auth)
        .send()
        .await
        .expect("ok");
    let body: Value = info.json().await.expect("json");
    // `models` is stored as a JSON-encoded TEXT column; the API returns the
    // raw string (matching the existing schema contract for tag/customer
    // `models` / `model_blocklist`).
    assert_eq!(body["models"], json!("[\"gpt-4o\",\"claude-*\"]"));

    shutdown(handle).await;
}

// ---------------------------------------------------------------------------
// C28 LOW: `/customer/block` filters blanks and trims `user_ids` before
// the cap check, mirroring `serialize_string_list`.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn customer_block_filters_blank_user_ids() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;
    let client = reqwest::Client::new();
    let auth = format!("Bearer {MASTER_KEY}");

    c28_create_customer(&handle, &client, "eu-block-real").await;

    let resp = client
        .post(handle.url("/customer/block"))
        .header("authorization", &auth)
        .json(&json!({"user_ids": ["", "  ", "eu-block-real"]}))
        .send()
        .await
        .expect("ok");
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("json");
    assert_eq!(
        body["customers_affected"], 1,
        "blanks must not reach SQL; only the real id is affected"
    );

    shutdown(handle).await;
}

// ---------------------------------------------------------------------------
// C28 coverage gap: skills have zero e2e coverage. Add a CRUD cycle
// (new → info → update → list → delete).
// ---------------------------------------------------------------------------

#[tokio::test]
async fn skill_crud_via_api() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;
    let client = reqwest::Client::new();
    let auth = format!("Bearer {MASTER_KEY}");

    // new
    let resp = client
        .post(handle.url("/skill/new"))
        .header("authorization", &auth)
        .json(&json!({
            "name": "code-reviewer",
            "description": "Reviews code",
            "content": "You are a senior code reviewer.",
            "model_pattern": "claude-*"
        }))
        .send()
        .await
        .expect("ok");
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("json");
    let skill_id = body["id"].as_str().expect("id").to_string();
    assert!(skill_id.starts_with("skill-"));

    // info
    let info = client
        .get(handle.url(&format!("/skill/{skill_id}")))
        .header("authorization", &auth)
        .send()
        .await
        .expect("ok");
    assert_eq!(info.status(), 200);
    let ib: Value = info.json().await.expect("json");
    assert_eq!(ib["name"], "code-reviewer");
    assert_eq!(ib["model_pattern"], "claude-*");
    assert_eq!(ib["version"], "1.0", "default version is 1.0");

    // update (rename + clear model_pattern)
    let upd = client
        .post(handle.url("/skill/update"))
        .header("authorization", &auth)
        .json(&json!({"id": skill_id, "name": "code-reviewer-v2", "clear_model_pattern": true}))
        .send()
        .await
        .expect("ok");
    assert_eq!(upd.status(), 200);
    let ub: Value = upd.json().await.expect("json");
    assert_eq!(ub["name"], "code-reviewer-v2");
    assert!(
        ub.get("model_pattern").is_none() || ub["model_pattern"].is_null(),
        "model_pattern should be cleared"
    );

    // list
    let list = client
        .get(handle.url("/skill/list"))
        .header("authorization", &auth)
        .send()
        .await
        .expect("ok");
    assert_eq!(list.status(), 200);
    let lb: Value = list.json().await.expect("json");
    assert_eq!(lb["skills"].as_array().map(|a| a.len()), Some(1));

    // delete
    let del = client
        .post(handle.url("/skill/delete"))
        .header("authorization", &auth)
        .json(&json!({"id": skill_id}))
        .send()
        .await
        .expect("ok");
    assert_eq!(del.status(), 200);

    // 404 the second time (idempotent 404, matching /key/delete semantics).
    let del2 = client
        .post(handle.url("/skill/delete"))
        .header("authorization", &auth)
        .json(&json!({"id": skill_id}))
        .send()
        .await
        .expect("ok");
    assert_eq!(del2.status(), 404);

    shutdown(handle).await;
}

#[tokio::test]
async fn skill_update_returns_404_for_unknown_id() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;
    let client = reqwest::Client::new();
    let auth = format!("Bearer {MASTER_KEY}");

    let resp = client
        .post(handle.url("/skill/update"))
        .header("authorization", &auth)
        .json(&json!({"id": "skill-nonexistent", "name": "x"}))
        .send()
        .await
        .expect("ok");
    assert_eq!(resp.status(), 404);

    shutdown(handle).await;
}

// ---------------------------------------------------------------------------
// C27 MEDIUM: add_user / add_team with a duplicate explicit id must return
// 400 "already exists" (mirrors /organization/add). Previously the duplicate
// insert surfaced as a generic 500.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn add_user_returns_400_on_duplicate_explicit_id() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;
    let client = reqwest::Client::new();
    let auth = format!("Bearer {MASTER_KEY}");

    let first = client
        .post(handle.url("/user/add"))
        .header("authorization", &auth)
        .json(&json!({"user_name": "Alice", "user_id": "user-dup"}))
        .send()
        .await
        .expect("ok");
    assert_eq!(first.status(), 200);

    let second = client
        .post(handle.url("/user/add"))
        .header("authorization", &auth)
        .json(&json!({"user_name": "Alice Again", "user_id": "user-dup"}))
        .send()
        .await
        .expect("ok");
    assert_eq!(
        second.status(),
        400,
        "duplicate user_id must be 400, not 500"
    );
    let body: Value = second.json().await.expect("json");
    assert!(
        body["error"]["message"]
            .as_str()
            .map(|s| s.contains("already exists"))
            .unwrap_or(false),
        "error body should mention 'already exists', got: {body}"
    );

    shutdown(handle).await;
}

#[tokio::test]
async fn add_team_returns_400_on_duplicate_explicit_id() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;
    let client = reqwest::Client::new();
    let auth = format!("Bearer {MASTER_KEY}");

    let first = client
        .post(handle.url("/team/add"))
        .header("authorization", &auth)
        .json(&json!({"team_name": "Eng", "team_id": "team-dup"}))
        .send()
        .await
        .expect("ok");
    assert_eq!(first.status(), 200);

    let second = client
        .post(handle.url("/team/add"))
        .header("authorization", &auth)
        .json(&json!({"team_name": "Eng Again", "team_id": "team-dup"}))
        .send()
        .await
        .expect("ok");
    assert_eq!(
        second.status(),
        400,
        "duplicate team_id must be 400, not 500"
    );
    let body: Value = second.json().await.expect("json");
    assert!(
        body["error"]["message"]
            .as_str()
            .map(|s| s.contains("already exists"))
            .unwrap_or(false),
        "error body should mention 'already exists', got: {body}"
    );

    shutdown(handle).await;
}

// ---------------------------------------------------------------------------
// C27 MEDIUM: deleting a user must clear `user_id` on the user's keys so the
// user-level budget cap is removed explicitly (mirrors /team/delete). The key
// must keep authenticating; only the user scope is gone. After the cascade,
// `/key/info` should report `user_id: null` for the previously bound key.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn delete_user_clears_user_id_on_keys() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;
    let client = reqwest::Client::new();
    let auth = format!("Bearer {MASTER_KEY}");

    let user_resp = client
        .post(handle.url("/user/add"))
        .header("authorization", &auth)
        .json(&json!({"user_name": "Alice", "user_id": "user-to-delete", "max_budget": 1.0}))
        .send()
        .await
        .expect("ok");
    assert_eq!(user_resp.status(), 200);

    let gen_resp = client
        .post(handle.url("/key/generate"))
        .header("authorization", &auth)
        .json(&json!({"user_id": "user-to-delete", "key_alias": "alice-key"}))
        .send()
        .await
        .expect("ok");
    assert_eq!(gen_resp.status(), 200);
    let gb: Value = gen_resp.json().await.expect("json");
    let key_id = gb["key_id"].as_str().expect("key_id").to_string();

    // Sanity: the key is bound to the user before the delete.
    let info_before = client
        .get(handle.url(&format!("/key/info?key_id={key_id}")))
        .header("authorization", &auth)
        .send()
        .await
        .expect("ok");
    let ib: Value = info_before.json().await.expect("json");
    assert_eq!(ib["user_id"].as_str(), Some("user-to-delete"));

    let del = client
        .post(handle.url("/user/delete"))
        .header("authorization", &auth)
        .json(&json!({"user_id": "user-to-delete"}))
        .send()
        .await
        .expect("ok");
    assert_eq!(del.status(), 200);

    // After the cascade, the key row should no longer reference the deleted
    // user; the user scope has been cleanly detached.
    let info_after = client
        .get(handle.url(&format!("/key/info?key_id={key_id}")))
        .header("authorization", &auth)
        .send()
        .await
        .expect("ok");
    assert_eq!(info_after.status(), 200);
    let ia: Value = info_after.json().await.expect("json");
    assert!(
        ia["user_id"].as_str().is_none(),
        "user_id must be cleared on key after user delete, got: {ia}"
    );

    shutdown(handle).await;
}

// ---------------------------------------------------------------------------
// C27 LOW: /key/update and /team/update can now clear an rpm/tpm cap (back to
// unlimited) via `clear_rpm` / `clear_tpm`, mirroring `clear_max_budget`.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn update_key_clear_rpm_clears_the_cap() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;
    let client = reqwest::Client::new();
    let auth = format!("Bearer {MASTER_KEY}");

    let gen_resp = client
        .post(handle.url("/key/generate"))
        .header("authorization", &auth)
        .json(&json!({"rpm": 10, "tpm": 100}))
        .send()
        .await
        .expect("ok");
    let gb: Value = gen_resp.json().await.expect("json");
    let key_id = gb["key_id"].as_str().expect("key_id").to_string();

    let upd = client
        .post(handle.url("/key/update"))
        .header("authorization", &auth)
        .json(&json!({"key_id": key_id, "clear_rpm": true, "clear_tpm": true}))
        .send()
        .await
        .expect("ok");
    assert_eq!(upd.status(), 200);
    let ub: Value = upd.json().await.expect("json");
    assert!(ub["rpm"].as_i64().is_none(), "rpm must be cleared: {ub}");
    assert!(ub["tpm"].as_i64().is_none(), "tpm must be cleared: {ub}");

    shutdown(handle).await;
}

#[tokio::test]
async fn update_team_clear_rpm_clears_the_cap() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;
    let client = reqwest::Client::new();
    let auth = format!("Bearer {MASTER_KEY}");

    let add = client
        .post(handle.url("/team/add"))
        .header("authorization", &auth)
        .json(&json!({"team_name": "T", "rpm": 10, "tpm": 100}))
        .send()
        .await
        .expect("ok");
    let ab: Value = add.json().await.expect("json");
    let team_id = ab["team_id"].as_str().expect("team_id").to_string();

    let upd = client
        .post(handle.url("/team/update"))
        .header("authorization", &auth)
        .json(&json!({"team_id": team_id, "clear_rpm": true, "clear_tpm": true}))
        .send()
        .await
        .expect("ok");
    assert_eq!(upd.status(), 200);
    let ub: Value = upd.json().await.expect("json");
    assert!(ub["rpm"].as_i64().is_none(), "rpm must be cleared: {ub}");
    assert!(ub["tpm"].as_i64().is_none(), "tpm must be cleared: {ub}");

    shutdown(handle).await;
}

// ---------------------------------------------------------------------------
// C27 LOW: rpm / tpm values above u32::MAX used to be silently coerced to
// `None` (unlimited) at enforcement time. They must now be rejected at the
// API boundary with a 400 instead.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn add_team_rejects_rpm_above_u32_max() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;
    let client = reqwest::Client::new();
    let auth = format!("Bearer {MASTER_KEY}");

    let resp = client
        .post(handle.url("/team/add"))
        .header("authorization", &auth)
        .json(&json!({"team_name": "T", "rpm": u32::MAX as i64 + 1}))
        .send()
        .await
        .expect("ok");
    assert_eq!(resp.status(), 400, "rpm > u32::MAX must be rejected");

    shutdown(handle).await;
}

#[tokio::test]
async fn add_team_rejects_tpm_above_u32_max() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;
    let client = reqwest::Client::new();
    let auth = format!("Bearer {MASTER_KEY}");

    let resp = client
        .post(handle.url("/team/add"))
        .header("authorization", &auth)
        .json(&json!({"team_name": "T", "tpm": u32::MAX as i64 + 1}))
        .send()
        .await
        .expect("ok");
    assert_eq!(resp.status(), 400, "tpm > u32::MAX must be rejected");

    shutdown(handle).await;
}

// ---------------------------------------------------------------------------
// Customer runtime enforcement (blocked / model_blocklist / max_budget / spend)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn customer_block_rejects_chat_with_end_user() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;
    let _m = mock_chat(&mock);
    let client = reqwest::Client::new();
    let auth = format!("Bearer {MASTER_KEY}");

    let created = client
        .post(handle.url("/customer/new"))
        .header("authorization", &auth)
        .json(&json!({"end_user_id": "eu-block-me"}))
        .send()
        .await
        .expect("ok");
    assert_eq!(created.status(), 200);

    let blocked = client
        .post(handle.url("/customer/block"))
        .header("authorization", &auth)
        .json(&json!({"end_user_id": "eu-block-me"}))
        .send()
        .await
        .expect("ok");
    assert_eq!(blocked.status(), 200);

    let chat = client
        .post(handle.url("/v1/chat/completions"))
        .header("authorization", &auth)
        .json(&json!({
            "model": "gpt-4o",
            "user": "eu-block-me",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("ok");
    assert_eq!(
        chat.status(),
        403,
        "blocked end-user must be rejected: {}",
        chat.text().await.unwrap_or_default()
    );

    let unblocked = client
        .post(handle.url("/customer/unblock"))
        .header("authorization", &auth)
        .json(&json!({"end_user_id": "eu-block-me"}))
        .send()
        .await
        .expect("ok");
    assert_eq!(unblocked.status(), 200);

    let chat2 = client
        .post(handle.url("/v1/chat/completions"))
        .header("authorization", &auth)
        .json(&json!({
            "model": "gpt-4o",
            "user": "eu-block-me",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("ok");
    assert_eq!(
        chat2.status(),
        200,
        "unblocked end-user must succeed: {}",
        chat2.text().await.unwrap_or_default()
    );

    shutdown(handle).await;
}

#[tokio::test]
async fn customer_model_blocklist_rejects_matching_model() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;
    let _m = mock_chat(&mock);
    let client = reqwest::Client::new();
    let auth = format!("Bearer {MASTER_KEY}");

    let created = client
        .post(handle.url("/customer/new"))
        .header("authorization", &auth)
        .json(&json!({
            "end_user_id": "eu-bl",
            "model_blocklist": ["gpt-4*"]
        }))
        .send()
        .await
        .expect("ok");
    assert_eq!(created.status(), 200);

    let chat = client
        .post(handle.url("/v1/chat/completions"))
        .header("authorization", &auth)
        .json(&json!({
            "model": "gpt-4o",
            "user": "eu-bl",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("ok");
    assert_eq!(
        chat.status(),
        403,
        "blocklisted model must 403: {}",
        chat.text().await.unwrap_or_default()
    );
    let body: Value = chat.json().await.expect("json");
    let msg = body["error"]["message"].as_str().unwrap_or("");
    assert!(
        msg.contains("blocked") || msg.contains("eu-bl"),
        "expected blocklist message, got {msg}"
    );

    shutdown(handle).await;
}

#[tokio::test]
async fn customer_max_budget_enforced_on_virtual_key() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;
    let _m = mock_chat(&mock);
    let client = reqwest::Client::new();
    let auth = format!("Bearer {MASTER_KEY}");

    // Tiny customer budget; the BudgetHook reserve estimate for a chat
    // call is well above this, so the first request must reject.
    let created = client
        .post(handle.url("/customer/new"))
        .header("authorization", &auth)
        .json(&json!({
            "end_user_id": "eu-budget",
            "max_budget": 1e-12
        }))
        .send()
        .await
        .expect("ok");
    assert_eq!(created.status(), 200);

    // Virtual key required — master bypasses BudgetHook.
    let gen_resp = client
        .post(handle.url("/key/generate"))
        .header("authorization", &auth)
        .json(&json!({"key_alias": "vk-customer-budget"}))
        .send()
        .await
        .expect("ok");
    assert_eq!(gen_resp.status(), 200);
    let gb: Value = gen_resp.json().await.expect("json");
    let raw_key = gb["key"].as_str().expect("key").to_string();

    let chat = client
        .post(handle.url("/v1/chat/completions"))
        .header("authorization", format!("Bearer {raw_key}"))
        .json(&json!({
            "model": "gpt-4o",
            "user": "eu-budget",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 5
        }))
        .send()
        .await
        .expect("ok");
    assert_eq!(
        chat.status(),
        429,
        "customer over budget must 429, got {}: {}",
        chat.status(),
        chat.text().await.unwrap_or_default()
    );

    shutdown(handle).await;
}

#[tokio::test]
async fn customer_spend_accrues_after_chat() {
    let mock = MockServer::start();
    // Spend writer flushes every 1s in start_oxllm.
    let (handle, db) = start_oxllm(proxy_config(&mock.base_url())).await;
    let _m = mock_chat(&mock);
    let client = reqwest::Client::new();
    let auth = format!("Bearer {MASTER_KEY}");

    let created = client
        .post(handle.url("/customer/new"))
        .header("authorization", &auth)
        .json(&json!({"end_user_id": "eu-spend"}))
        .send()
        .await
        .expect("ok");
    assert_eq!(created.status(), 200);

    let chat = client
        .post(handle.url("/v1/chat/completions"))
        .header("authorization", &auth)
        .json(&json!({
            "model": "gpt-4o",
            "user": "eu-spend",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("ok");
    assert_eq!(chat.status(), 200, "chat must succeed");

    // Wait for the 1s spend-writer flush, polling until spend is visible.
    wait_until("customer spend accrual", || async {
        let info = client
            .get(handle.url("/customer/info?end_user_id=eu-spend"))
            .header("authorization", &auth)
            .send()
            .await
            .expect("ok");
        assert_eq!(info.status(), 200);
        let body: Value = info.json().await.expect("json");
        body["spend"].as_f64().unwrap_or(-1.0) > 0.0
    })
    .await;

    let info = client
        .get(handle.url("/customer/info?end_user_id=eu-spend"))
        .header("authorization", &auth)
        .send()
        .await
        .expect("ok");
    assert_eq!(info.status(), 200);
    let body: Value = info.json().await.expect("json");
    let spend = body["spend"].as_f64().unwrap_or(-1.0);
    assert!(
        spend > 0.0,
        "customer spend must accrue after billed chat, got {spend}: {body}"
    );

    // Also confirm the spend log carried end_user_id.
    let logs = db
        .query_spend(&SpendFilter {
            limit: Some(10),
            ..Default::default()
        })
        .await
        .expect("query spend");
    assert!(
        logs.iter()
            .any(|l| l.end_user_id.as_deref() == Some("eu-spend")),
        "spend_logs must attribute end_user_id, got: {logs:?}"
    );

    shutdown(handle).await;
}

// ---------------------------------------------------------------------------
// Skills runtime injection
// ---------------------------------------------------------------------------

#[tokio::test]
async fn skill_content_injected_into_upstream_system_prompt() {
    let mock = MockServer::start();
    let skill_marker = "UNIQUE_SKILL_CONTENT_FOR_E2E_TEST";
    // Mock that only matches when the skill content is present in the body.
    let m = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .header("authorization", "Bearer sk-upstream")
            .body_contains(skill_marker);
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "id": "chatcmpl-skill",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
            }));
    });

    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;
    let client = reqwest::Client::new();
    let auth = format!("Bearer {MASTER_KEY}");

    let created = client
        .post(handle.url("/skill/new"))
        .header("authorization", &auth)
        .json(&json!({
            "name": "e2e-skill",
            "content": skill_marker,
            "model_pattern": "gpt-4*"
        }))
        .send()
        .await
        .expect("ok");
    assert_eq!(
        created.status(),
        200,
        "skill create failed: {}",
        created.text().await.unwrap_or_default()
    );

    let chat = client
        .post(handle.url("/v1/chat/completions"))
        .header("authorization", &auth)
        .json(&json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("ok");
    assert_eq!(
        chat.status(),
        200,
        "chat with skill injection must reach mock: {}",
        chat.text().await.unwrap_or_default()
    );
    m.assert();

    shutdown(handle).await;
}

// ---------------------------------------------------------------------------
// Tag runtime spend metering (max_budget + durable spend)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn tag_max_budget_enforced_on_virtual_key() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;
    let _m = mock_chat(&mock);
    let client = reqwest::Client::new();
    let auth = format!("Bearer {MASTER_KEY}");

    let created = client
        .post(handle.url("/tag/new"))
        .header("authorization", &auth)
        .json(&json!({
            "name": "canary",
            "max_budget": 1e-12
        }))
        .send()
        .await
        .expect("ok");
    assert_eq!(created.status(), 200);

    // Virtual key required — master bypasses BudgetHook.
    let gen_resp = client
        .post(handle.url("/key/generate"))
        .header("authorization", &auth)
        .json(&json!({"key_alias": "vk-tag-budget"}))
        .send()
        .await
        .expect("ok");
    assert_eq!(gen_resp.status(), 200);
    let gb: Value = gen_resp.json().await.expect("json");
    let raw_key = gb["key"].as_str().expect("key").to_string();

    let chat = client
        .post(handle.url("/v1/chat/completions"))
        .header("authorization", format!("Bearer {raw_key}"))
        .header("x-oxllm-tag", "canary")
        .json(&json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 5
        }))
        .send()
        .await
        .expect("ok");
    assert_eq!(
        chat.status(),
        429,
        "tag over budget must 429, got {}: {}",
        chat.status(),
        chat.text().await.unwrap_or_default()
    );

    shutdown(handle).await;
}

#[tokio::test]
async fn tag_spend_accrues_after_chat() {
    let mock = MockServer::start();
    let (handle, db) = start_oxllm(proxy_config(&mock.base_url())).await;
    let _m = mock_chat(&mock);
    let client = reqwest::Client::new();
    let auth = format!("Bearer {MASTER_KEY}");

    let created = client
        .post(handle.url("/tag/new"))
        .header("authorization", &auth)
        .json(&json!({"name": "prod"}))
        .send()
        .await
        .expect("ok");
    assert_eq!(created.status(), 200);

    let chat = client
        .post(handle.url("/v1/chat/completions"))
        .header("authorization", &auth)
        .header("x-oxllm-tag", "prod")
        .json(&json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("ok");
    assert_eq!(chat.status(), 200, "chat must succeed");

    // Wait for the 1s spend-writer flush, polling until spend is visible.
    wait_until("tag spend accrual", || async {
        let info = client
            .get(handle.url("/tag/info?name=prod"))
            .header("authorization", &auth)
            .send()
            .await
            .expect("ok");
        assert_eq!(info.status(), 200);
        let body: Value = info.json().await.expect("json");
        body["spend"].as_f64().unwrap_or(-1.0) > 0.0
    })
    .await;

    let info = client
        .get(handle.url("/tag/info?name=prod"))
        .header("authorization", &auth)
        .send()
        .await
        .expect("ok");
    assert_eq!(info.status(), 200);
    let body: Value = info.json().await.expect("json");
    let spend = body["spend"].as_f64().unwrap_or(-1.0);
    assert!(
        spend > 0.0,
        "tag spend must accrue after billed chat, got {spend}: {body}"
    );

    // spend_logs.tag_name must attribute the request tag (v9).
    let logs = db
        .query_spend(&SpendFilter {
            limit: Some(10),
            ..Default::default()
        })
        .await
        .expect("query spend");
    assert!(
        logs.iter().any(|l| l.tag_name.as_deref() == Some("prod")),
        "spend_logs must attribute tag_name, got: {logs:?}"
    );

    shutdown(handle).await;
}

/// Real API path: unknown tag header still stamps spend_logs.tag_name
/// (no tag row required for log attribution).
#[tokio::test]
async fn spend_log_records_tag_name_from_header_without_tag_row() {
    let mock = MockServer::start();
    let (handle, db) = start_oxllm(proxy_config(&mock.base_url())).await;
    let _m = mock_chat(&mock);
    let client = reqwest::Client::new();
    let auth = format!("Bearer {MASTER_KEY}");

    let chat = client
        .post(handle.url("/v1/chat/completions"))
        .header("authorization", &auth)
        .header("x-oxllm-tag", "ephemeral-tag")
        .json(&json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("ok");
    assert_eq!(
        chat.status(),
        200,
        "unknown tag must not block chat: {}",
        chat.text().await.unwrap_or_default()
    );

    // Wait for the 1s spend-writer flush, polling until the tagged row lands.
    wait_until("ephemeral tag spend log", || async {
        let logs = db
            .query_spend(&SpendFilter {
                limit: Some(20),
                ..Default::default()
            })
            .await
            .expect("query");
        logs.iter()
            .any(|l| l.tag_name.as_deref() == Some("ephemeral-tag") && l.cost.is_some())
    })
    .await;

    let logs = db
        .query_spend(&SpendFilter {
            limit: Some(20),
            ..Default::default()
        })
        .await
        .expect("query");
    assert!(
        logs.iter()
            .any(|l| l.tag_name.as_deref() == Some("ephemeral-tag") && l.cost.is_some()),
        "spend log must carry tag_name from header even without a tags row: {logs:?}"
    );

    // No durable tag spend without a tags row.
    let missing = client
        .get(handle.url("/tag/info?name=ephemeral-tag"))
        .header("authorization", &auth)
        .send()
        .await
        .expect("ok");
    assert_eq!(missing.status(), 404);

    shutdown(handle).await;
}

#[tokio::test]
async fn tag_name_rejects_colon() {
    let mock = MockServer::start();
    let (handle, _db) = start_oxllm(proxy_config(&mock.base_url())).await;
    let client = reqwest::Client::new();
    let auth = format!("Bearer {MASTER_KEY}");

    let resp = client
        .post(handle.url("/tag/new"))
        .header("authorization", &auth)
        .json(&json!({"name": "bad:tag"}))
        .send()
        .await
        .expect("ok");
    assert_eq!(resp.status(), 400);
    let body: Value = resp.json().await.expect("json");
    let msg = body["error"]["message"].as_str().unwrap_or("");
    assert!(
        msg.contains("':'") || msg.contains("colon") || msg.contains("spend-key"),
        "expected colon-rejection message, got {msg}"
    );

    shutdown(handle).await;
}

/// The shared-production-key pattern: one virtual key serves several
/// applications, distinguished ONLY by request tags (body `metadata.tags`).
/// Each application's spend is attributed to its tag, per-app budgets block
/// only the over-budget app, and `/spend/tags` reports the breakdown.
#[tokio::test]
async fn shared_key_multi_app_tags_with_per_app_budgets() {
    let mock = MockServer::start();
    let (handle, db) = start_oxllm(proxy_config(&mock.base_url())).await;
    let _m = mock_chat(&mock);
    let client = reqwest::Client::new();
    let auth = format!("Bearer {MASTER_KEY}");

    // Two applications; only app-billing gets a budget (tightened later).
    for name in ["app-search", "app-billing"] {
        let created = client
            .post(handle.url("/tag/new"))
            .header("authorization", &auth)
            .json(&json!({"name": name}))
            .send()
            .await
            .expect("ok");
        assert_eq!(created.status(), 200, "tag {name} must be created");
    }

    // ONE shared production key with a roomy budget.
    let gen_resp = client
        .post(handle.url("/key/generate"))
        .header("authorization", &auth)
        .json(&json!({"key_alias": "prod-shared", "max_budget": 5.0}))
        .send()
        .await
        .expect("ok");
    assert_eq!(gen_resp.status(), 200);
    let gb: Value = gen_resp.json().await.expect("json");
    let raw_key = gb["key"].as_str().expect("key").to_string();
    let vk = format!("Bearer {raw_key}");

    let chat = |app: String| {
        let client = client.clone();
        let url = handle.url("/v1/chat/completions");
        let vk = vk.clone();
        async move {
            client
                .post(url)
                .header("authorization", &vk)
                .json(&json!({
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 5,
                    "metadata": {"tags": [app]},
                }))
                .send()
                .await
                .expect("ok")
        }
    };
    // Two searches, two billings — all through the same key.
    assert_eq!(chat("app-search".to_string()).await.status(), 200);
    assert_eq!(chat("app-search".to_string()).await.status(), 200);
    assert_eq!(chat("app-billing".to_string()).await.status(), 200);
    assert_eq!(chat("app-billing".to_string()).await.status(), 200);
    // A request carrying BOTH tags counts toward both ledgers.
    let both = client
        .post(handle.url("/v1/chat/completions"))
        .header("authorization", &vk)
        .json(&json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 5,
            "metadata": {"tags": ["app-search", "app-billing"]},
        }))
        .send()
        .await
        .expect("ok");
    assert_eq!(both.status(), 200);

    // The page and totals queries run in one HTTP request; if the newest
    // event flushes between them the page can look short and totals correct.
    // Wait for all five events to be durable before asserting filtered pages.
    wait_until("shared-key spend rows flushed", || async {
        db.count_spend(None, None).await.unwrap() == 5
    })
    .await;

    // /spend/logs?tag_name filters by tag (quoted-substring on the JSON
    // tags column, so app-search never matches a hypothetical app-searchx).
    let logs = |tag: String| {
        let client = client.clone();
        let url = handle.url(&format!("/spend/logs?tag_name={tag}"));
        let auth = auth.clone();
        async move {
            let r = client
                .get(url)
                .header("authorization", &auth)
                .send()
                .await
                .expect("ok");
            assert_eq!(r.status(), 200);
            r.json::<Value>().await.expect("json")
        }
    };
    let search_rows = logs("app-search".to_string()).await;
    let billing_rows = logs("app-billing".to_string()).await;
    assert_eq!(
        search_rows["data"].as_array().map(|a| a.len()),
        Some(3),
        "2 single-tag + 1 multi-tag rows: {search_rows}"
    );
    assert_eq!(
        billing_rows["data"].as_array().map(|a| a.len()),
        Some(3),
        "2 single-tag + 1 multi-tag rows: {billing_rows}"
    );
    assert_eq!(
        logs("zzz".to_string()).await["data"]
            .as_array()
            .map(|a| a.len()),
        Some(0)
    );

    // /spend/tags: the multi-tag row overlaps into BOTH buckets by design.
    let agg = client
        .get(handle.url("/spend/tags"))
        .header("authorization", &auth)
        .send()
        .await
        .expect("ok");
    assert_eq!(agg.status(), 200);
    let rows: Value = agg.json().await.expect("json");
    let by = |t: &str| {
        rows["data"]
            .as_array()
            .expect("data")
            .iter()
            .find(|r| r["tag_name"] == t)
            .unwrap_or_else(|| panic!("missing {t} in {rows}"))
            .clone()
    };
    assert_eq!(by("app-search")["total_requests"], 3);
    assert_eq!(by("app-billing")["total_requests"], 3);
    assert_eq!(
        by("app-search")["total_cost"],
        by("app-billing")["total_cost"],
        "identical traffic must bill identically per tag"
    );

    // Tighten ONLY app-billing's budget below what it already spent: its
    // next request must 429 while app-search keeps working on the same key.
    let upd = client
        .post(handle.url("/tag/update"))
        .header("authorization", &auth)
        .json(&json!({"name": "app-billing", "max_budget": 0.000001}))
        .send()
        .await
        .expect("ok");
    assert_eq!(upd.status(), 200);
    let blocked = chat("app-billing".to_string()).await;
    assert_eq!(
        blocked.status(),
        429,
        "over-budget app must 429, got {}: {}",
        blocked.status(),
        blocked.text().await.unwrap_or_default()
    );
    let allowed = chat("app-search".to_string()).await;
    assert_eq!(allowed.status(), 200, "the other app must be unaffected");

    shutdown(handle).await;
}
