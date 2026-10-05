//! Integration tests: generic HTTP callbacks, allowed_ips, pass_through_endpoints.

use std::collections::HashMap;
use std::sync::Arc;

use httpmock::MockServer;
use oxllm_server::app::{build_router, build_shared_state_with_db};
use oxllm_server::config::types::{
    ConfigModel, GeneralSettings, OxllmSettings, PassThroughEndpoint, ProxyConfig,
};
use oxllm_server::spend_writer;
use oxllm_server::{Database, OxllmParams, SharedState};
use serde_json::{Value, json};

const MASTER_KEY: &str = "sk-master-ops-test-key";
const VIRTUAL_KEY: &str = "sk-virtual-ops";

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

async fn create_virtual_key(db: &Arc<Database>) {
    use oxllm_server::KeyRecord;
    let now = oxllm_server::now_iso();
    db.create_team(&oxllm_server::TeamRecord {
        team_id: "team-ops-1".into(),
        team_name: "Ops test team".into(),
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
        key_id: "k-ops-1".to_string(),
        key_hash: oxllm_server::auth::hash_key_hex(VIRTUAL_KEY.as_bytes()),
        key_alias: Some("ops-test".to_string()),
        team_id: Some("team-ops-1".to_string()),
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

fn proxy_config_with_callback(mock_base_url: &str, callback_url: &str) -> ProxyConfig {
    let mut callback_settings = HashMap::new();
    let mut generic = HashMap::new();
    generic.insert("url".to_string(), json!(callback_url));
    callback_settings.insert("generic".to_string(), json!(generic));

    ProxyConfig {
        model_list: vec![ConfigModel {
            model_name: "gpt-4o".to_string(),
            oxllm_params: base_params(mock_base_url),
            ..Default::default()
        }],
        oxllm_settings: OxllmSettings {
            success_callback: Some(vec!["generic".to_string()]),
            callback_settings: Some(callback_settings),
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

fn proxy_config_with_allowed_ips(mock_base_url: &str, allowed: Vec<String>) -> ProxyConfig {
    ProxyConfig {
        model_list: vec![ConfigModel {
            model_name: "gpt-4o".to_string(),
            oxllm_params: base_params(mock_base_url),
            ..Default::default()
        }],
        oxllm_settings: OxllmSettings::default(),
        general_settings: GeneralSettings {
            master_key: Some(MASTER_KEY.to_string()),
            allowed_ips: Some(allowed),
            allow_private_api_base: Some(true),
            ..Default::default()
        },
        router_settings: Default::default(),
        extra: serde_json::Map::new(),
    }
}

fn proxy_config_with_passthrough(mock_base_url: &str, pass_through_target: &str) -> ProxyConfig {
    ProxyConfig {
        model_list: vec![ConfigModel {
            model_name: "gpt-4o".to_string(),
            oxllm_params: base_params(mock_base_url),
            ..Default::default()
        }],
        oxllm_settings: OxllmSettings::default(),
        general_settings: GeneralSettings {
            master_key: Some(MASTER_KEY.to_string()),
            pass_through_endpoints: Some(vec![PassThroughEndpoint {
                path: "/v1/custom-endpoint".to_string(),
                target: pass_through_target.to_string(),
                headers: Some(HashMap::new()),
            }]),
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

async fn shutdown(handle: OxllmHandle) {
    let _ = handle.shutdown_tx.send(());
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
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
async fn generic_callback_fires_on_successful_chat() {
    let upstream = MockServer::start();
    let callback = MockServer::start();

    let callback_hit = callback.mock(|when, then| {
        when.method(httpmock::Method::POST).path("/");
        then.status(202);
    });

    let handle = start_oxllm(proxy_config_with_callback(
        &upstream.base_url(),
        &callback.base_url(),
    ))
    .await;
    // Master-key traffic intentionally skips the callback (matching every
    // other exporter); create a virtual key to exercise the fire-and-forget
    // POST path with a non-master principal.
    create_virtual_key(&handle._db).await;
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
        elapsed < std::time::Duration::from_secs(3),
        "chat should not block on the callback (took {elapsed:?})"
    );

    for _ in 0..40 {
        if callback_hit.hits() >= 1 {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    assert!(
        callback_hit.hits() >= 1,
        "generic callback should have been POSTed at least once"
    );

    shutdown(handle).await;
}

#[tokio::test]
async fn allowed_ips_rejects_unlisted_peer() {
    let upstream = MockServer::start();
    let upstream_never = upstream.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions");
        then.status(500);
    });
    let handle = start_oxllm(proxy_config_with_allowed_ips(
        &upstream.base_url(),
        vec!["10.0.0.0/8".to_string()],
    ))
    .await;

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth(MASTER_KEY)
        .json(&json!({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .expect("req");
    assert_eq!(
        resp.status(),
        403,
        "loopback peer 127.0.0.1 is not in 10.0.0.0/8 → should be blocked"
    );
    assert_eq!(upstream_never.hits(), 0, "upstream must not be hit");

    shutdown(handle).await;
}

#[tokio::test]
async fn allowed_ips_allows_listed_peer() {
    let upstream = MockServer::start();
    let handle = start_oxllm(proxy_config_with_allowed_ips(
        &upstream.base_url(),
        vec!["127.0.0.0/8".to_string()],
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
        .expect("req");
    assert_eq!(
        resp.status(),
        200,
        "loopback peer 127.0.0.1 is in 127.0.0.0/8 → should be allowed"
    );

    shutdown(handle).await;
}

#[tokio::test]
async fn pass_through_endpoint_forwards_to_target() {
    let upstream = MockServer::start();
    let target = MockServer::start();

    let target_hit = target.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/v1/custom-endpoint/foo");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({"forwarded": true}));
    });
    let upstream_never = upstream.mock(|when, then| {
        when.method(httpmock::Method::POST);
        then.status(500);
    });

    let handle = start_oxllm(proxy_config_with_passthrough(
        &upstream.base_url(),
        &target.base_url(),
    ))
    .await;

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/custom-endpoint/foo"))
        .bearer_auth(MASTER_KEY)
        .json(&json!({"hello": "world"}))
        .send()
        .await
        .expect("passthrough");
    assert_eq!(resp.status(), 200, "passthrough should reach the target");
    let body: Value = resp.json().await.expect("json");
    assert_eq!(body["forwarded"].as_bool(), Some(true));

    assert_eq!(
        target_hit.hits(),
        1,
        "target should receive exactly one forwarded request"
    );
    assert_eq!(upstream_never.hits(), 0, "main upstream should not be hit");

    shutdown(handle).await;
}

#[tokio::test]
async fn pass_through_endpoint_requires_auth() {
    let upstream = MockServer::start();
    let target = MockServer::start();

    let target_hit = target.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/v1/custom-endpoint/foo");
        then.status(200);
    });

    let handle = start_oxllm(proxy_config_with_passthrough(
        &upstream.base_url(),
        &target.base_url(),
    ))
    .await;

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/custom-endpoint/foo"))
        .bearer_auth("sk-wrong")
        .json(&json!({}))
        .send()
        .await
        .expect("req");
    assert_eq!(resp.status(), 401, "passthrough must enforce auth");
    assert_eq!(target_hit.hits(), 0);

    shutdown(handle).await;
}

#[tokio::test]
async fn pass_through_endpoint_attaches_configured_headers() {
    let upstream = MockServer::start();
    let target = MockServer::start();

    let target_hit = target.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/v1/custom-endpoint/foo")
            .header("X-Custom-Token", "secret-value");
        then.status(200).json_body(json!({"ok": true}));
    });

    let config = ProxyConfig {
        model_list: vec![ConfigModel {
            model_name: "gpt-4o".to_string(),
            oxllm_params: base_params(&upstream.base_url()),
            ..Default::default()
        }],
        oxllm_settings: OxllmSettings::default(),
        general_settings: GeneralSettings {
            master_key: Some(MASTER_KEY.to_string()),
            pass_through_endpoints: Some(vec![PassThroughEndpoint {
                path: "/v1/custom-endpoint".to_string(),
                target: target.base_url(),
                headers: Some({
                    let mut m = HashMap::new();
                    m.insert("X-Custom-Token".to_string(), "secret-value".to_string());
                    m
                }),
            }]),
            allow_private_api_base: Some(true),
            ..Default::default()
        },
        router_settings: Default::default(),
        extra: serde_json::Map::new(),
    };

    let handle = start_oxllm(config).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/custom-endpoint/foo"))
        .bearer_auth(MASTER_KEY)
        .json(&json!({}))
        .send()
        .await
        .expect("req");
    assert_eq!(resp.status(), 200);
    assert_eq!(target_hit.hits(), 1, "target should receive the request");

    shutdown(handle).await;
}

// ---------------------------------------------------------------------------
// Bug #28: configured pass-through targets must not let an attacker escape
// the target's path prefix via `..` segments (the configured secret headers
// would otherwise be attached to the escaped request). Most HTTP clients
// (reqwest included) normalize `..` client-side via the `url` crate, so we
// send a raw HTTP request over a TcpStream to exercise the server-side fix.
// ---------------------------------------------------------------------------
#[tokio::test]
async fn pass_through_endpoint_rejects_path_traversal_escape() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let upstream = MockServer::start();
    let target = MockServer::start();

    // Target has a path prefix `/secret-prefix` — attacks that escape it
    // would leak the configured `X-Secret` header to other endpoints.
    let target_url = format!("{}/secret-prefix", target.base_url());

    // Anything that lands under /secret-prefix is legitimate.
    let _legit = target.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path_matches(regex::Regex::new(r"^/secret-prefix").unwrap());
        then.status(200).json_body(json!({"ok": true}));
    });
    // An escape attempt would land here (or anywhere else under /).
    let escape = target.mock(|when, then| {
        when.method(httpmock::Method::POST).path("/admin");
        then.status(200).json_body(json!({"leaked": true}));
    });

    let mut config = proxy_config_with_passthrough(&upstream.base_url(), &target_url);
    config.general_settings.pass_through_endpoints = config
        .general_settings
        .pass_through_endpoints
        .map(|mut eps| {
            for ep in &mut eps {
                ep.headers = Some({
                    let mut m = HashMap::new();
                    m.insert("X-Secret".to_string(), "topsecret".to_string());
                    m
                });
            }
            eps
        });

    let handle = start_oxllm(config).await;

    // Build a raw HTTP request whose URL contains `..` segments that, after
    // normalization, escape `/secret-prefix`. reqwest's url crate would
    // normalize this client-side, so we have to bypass it.
    //
    // /v1/custom-endpoint/a/b/../../../../../admin
    //   → /secret-prefix/v1/custom-endpoint/a/b/../../../../../admin
    //   → /admin   (climbs out of /secret-prefix/v1/custom-endpoint/a/b,
    //               then /secret-prefix itself)
    //
    // Counting pops: a/b → a → custom-endpoint → v1 → secret-prefix → /
    // requires five `..` segments.
    let raw_request = format!(
        "POST /v1/custom-endpoint/a/b/../../../../../admin HTTP/1.1\r\n\
         host: {host}\r\n\
         authorization: Bearer {master}\r\n\
         connection: close\r\n\
         content-length: 2\r\n\
         \r\n\
         {}",
        "{}",
        host = handle.addr,
        master = MASTER_KEY,
    );

    let mut stream = tokio::net::TcpStream::connect(handle.addr)
        .await
        .expect("connect");
    stream
        .write_all(raw_request.as_bytes())
        .await
        .expect("write");

    let mut buf = Vec::new();
    let _ = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        stream.read_to_end(&mut buf),
    )
    .await
    .expect("read within timeout");

    let response = String::from_utf8_lossy(&buf);
    let status_line = response.lines().next().unwrap_or("");
    assert!(
        status_line.contains("403"),
        "path traversal escape must be rejected with 403, got: {status_line}\nfull response:\n{response}"
    );
    assert_eq!(
        escape.hits(),
        0,
        "target must not receive the escaped request (would leak X-Secret)"
    );

    shutdown(handle).await;
}
