//! Integration tests: JWT auth, realtime WebSocket, body limits, smuggling guard, error sanitization.

use httpmock::MockServer;
use oxllm_server::app::{build_router, build_shared_state};
use oxllm_server::config::types::{ConfigModel, GeneralSettings, OxllmSettings, ProxyConfig};
use oxllm_server::{OxllmParams, SharedState};
use serde_json::{Value, json};

const MASTER_KEY: &str = "sk-master-security";

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
            master_key: Some(MASTER_KEY.to_string()),
            allow_private_api_base: Some(true),
            ..Default::default()
        },
        router_settings: Default::default(),
        extra: serde_json::Map::new(),
    }
}

fn proxy_config_with_jwt(mock_base_url: &str, jwt_secret: &str) -> ProxyConfig {
    let mut cfg = proxy_config(mock_base_url);
    cfg.general_settings.jwt_secret = Some(jwt_secret.to_string());
    cfg.general_settings.jwt_audience = Some("oxllm".to_string());
    cfg
}

async fn start_oxllm(config: ProxyConfig) -> (OxllmHandle, SharedState) {
    let shared = build_shared_state(config).expect("build state");
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let addr = listener.local_addr().expect("addr");
    let router = build_router(shared.clone());
    // Provide ConnectInfo like `run()` does (into_make_service_with_connect_info)
    // so IP-keyed middleware (auth brake, IP allowlist) can attribute the
    // peer address instead of silently skipping enforcement.
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
    (OxllmHandle { addr, shutdown_tx }, shared)
}

fn mock_chat_ok(mock: &MockServer) -> httpmock::Mock<'_> {
    mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .header("authorization", "Bearer sk-upstream");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
            "id": "chatcmpl-8",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-4o",
            "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
            }));
    })
}

fn make_jwt(claims: &Value, secret: &str) -> String {
    use jsonwebtoken::{Algorithm, EncodingKey, Header};
    let header = Header::new(Algorithm::HS256);
    jsonwebtoken::encode(
        &header,
        claims,
        &EncodingKey::from_secret(secret.as_bytes()),
    )
    .expect("encode jwt")
}

fn future_exp() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
        + 3600
}

#[tokio::test]
async fn jwt_valid_token_authenticates() {
    let mock = MockServer::start();
    let _m = mock_chat_ok(&mock);

    let config = proxy_config_with_jwt(&mock.base_url(), "jwt-e2e-secret-0123456789");
    let (handle, _shared) = start_oxllm(config).await;

    let claims = json!({
    "sub": "user-1",
    "team_id": "team-1",
    "exp": future_exp(),
    "aud": "oxllm",
    });
    let token = make_jwt(&claims, "jwt-e2e-secret-0123456789");

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth(&token)
        .json(&json!({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .expect("send");

    assert_eq!(resp.status(), 200);
    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn jwt_only_mode_requires_and_validates_jwt() {
    let mock = MockServer::start();
    let upstream = mock_chat_ok(&mock);

    let mut config = proxy_config_with_jwt(&mock.base_url(), "jwt-e2e-secret-0123456789");
    config.general_settings.master_key = None;
    let (handle, _shared) = start_oxllm(config).await;
    let client = reqwest::Client::new();
    let body = json!({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]});

    let missing = client
        .post(handle.url("/v1/chat/completions"))
        .json(&body)
        .send()
        .await
        .expect("send missing jwt request");
    assert_eq!(missing.status(), 401);

    let invalid = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth("not-a-jwt")
        .json(&body)
        .send()
        .await
        .expect("send invalid jwt request");
    assert_eq!(invalid.status(), 401);

    let token = make_jwt(
        &json!({"sub": "user-1", "exp": future_exp(), "aud": "oxllm"}),
        "jwt-e2e-secret-0123456789",
    );
    let valid = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth(token)
        .json(&body)
        .send()
        .await
        .expect("send valid jwt request");
    assert_eq!(valid.status(), 200);
    upstream.assert_hits(1);
    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn jwt_expired_token_rejected() {
    let mock = MockServer::start();

    let config = proxy_config_with_jwt(&mock.base_url(), "jwt-e2e-secret-0123456789");
    let (handle, _shared) = start_oxllm(config).await;

    let claims = json!({
    "sub": "user-1",
    "exp": 1,
    "aud": "oxllm",
    });
    let token = make_jwt(&claims, "jwt-e2e-secret-0123456789");

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth(&token)
        .json(&json!({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .expect("send");

    assert_eq!(resp.status(), 401);
    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn jwt_wrong_secret_rejected() {
    let mock = MockServer::start();

    let config = proxy_config_with_jwt(&mock.base_url(), "jwt-e2e-secret-0123456789");
    let (handle, _shared) = start_oxllm(config).await;

    let claims = json!({"sub": "user-1", "exp": future_exp(), "aud": "oxllm"});
    let token = make_jwt(&claims, "different-secret");

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth(&token)
        .json(&json!({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .expect("send");

    assert_eq!(resp.status(), 401);
    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn master_key_still_works_with_jwt_enabled() {
    let mock = MockServer::start();
    let _m = mock_chat_ok(&mock);

    let config = proxy_config_with_jwt(&mock.base_url(), "jwt-e2e-secret-0123456789");
    let (handle, _shared) = start_oxllm(config).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth(MASTER_KEY)
        .json(&json!({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .expect("send");

    assert_eq!(resp.status(), 200);
    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn upstream_error_message_truncated() {
    let mock = MockServer::start();
    let long_message = "x".repeat(2000);
    let _m = mock.mock(|when, then| {
        when.method(httpmock::Method::POST).path("/rerank");
        then.status(500)
            .header("server", "upstream-server/1.0")
            .header("x-request-id", "leak-id")
            .body(long_message.clone());
    });

    let config = proxy_config(&mock.base_url());
    let (handle, _shared) = start_oxllm(config).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/rerank"))
        .bearer_auth(MASTER_KEY)
        .json(&json!({"model": "gpt-4o", "query": "q"}))
        .send()
        .await
        .expect("send");

    assert_eq!(resp.status(), 500);
    assert!(
        resp.headers().get("server").is_none(),
        "server header stripped"
    );
    // The upstream's leaked `x-request-id` must be stripped. oxllm stamps its
    // own request id on every response (H8), so the header may be present —
    // but it must never carry the upstream's leaked value.
    let xrid = resp
        .headers()
        .get("x-request-id")
        .and_then(|v| v.to_str().ok());
    assert_ne!(
        xrid,
        Some("leak-id"),
        "upstream x-request-id must be stripped"
    );
    let body = resp.text().await.expect("body");
    assert!(
        body.chars().count() <= 501,
        "error body should be truncated to ~500 chars, got {}",
        body.chars().count()
    );
    assert!(body.ends_with('…'));
    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn body_size_limit_enforced() {
    let mock = MockServer::start();

    let mut config = proxy_config(&mock.base_url());
    config.general_settings.max_request_body_size_mb = Some(1);
    let (handle, _shared) = start_oxllm(config).await;

    let big_payload = "a".repeat(2 * 1024 * 1024);

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth(MASTER_KEY)
        .header("content-type", "application/json")
        .body(format!(
            r#"{{"model":"gpt-4o","messages":[{{"role":"user","content":"{big_payload}"}}]}}"#
        ))
        .send()
        .await
        .expect("send");

    assert!(
        resp.status() == 413 || resp.status() == 400,
        "oversized body should be rejected (413/400), got {}",
        resp.status()
    );
    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn request_smuggling_rejected() {
    let mock = MockServer::start();
    let config = proxy_config(&mock.base_url());
    let (handle, _shared) = start_oxllm(config).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth(MASTER_KEY)
        .header("content-length", "10")
        .header("transfer-encoding", "chunked")
        .body("hello world")
        .send()
        .await
        .expect("send");

    assert_eq!(
        resp.status(),
        400,
        "ambiguous Content-Length + Transfer-Encoding must be rejected"
    );
    let _ = handle.shutdown_tx.send(());
}

/// Brute-force protection must be exercised through the live proxy: fire
/// real bad-auth requests at the router and assert the block response, so a
/// regression in the middleware wiring (brake not consulted, failures not
/// recorded, IP attribution broken) fails here instead of passing silently.
#[tokio::test]
async fn auth_brute_force_blocks_after_threshold() {
    let mock = MockServer::start();
    let upstream = mock_chat_ok(&mock);

    // Route the brake at a synthetic client IP. The test client connects from
    // loopback, which `auth_brake_ip` treats as a *shared* peer and skips
    // unless the peer is a trusted proxy — so trust loopback (exactly like a
    // real deployment behind nginx) and pin the client via X-Forwarded-For.
    let mut config = proxy_config(&mock.base_url());
    config.general_settings.trusted_proxy_cidrs = Some(vec!["127.0.0.0/8".to_string()]);
    let (handle, shared) = start_oxllm(config).await;

    assert!(
        shared.load_full().auth_brake.is_some(),
        "master_key config must wire the auth brake into the live router"
    );

    let client = reqwest::Client::new();
    let body = json!({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]});
    let attacker = "203.0.113.7";

    // Drive the failure count to the default policy threshold (10 failures
    // in a 60s window) through real HTTP requests, so the test proves the
    // middleware records failures — not just that AuthBrake works standalone.
    for i in 0..10 {
        let resp = client
            .post(handle.url("/v1/chat/completions"))
            .header("x-forwarded-for", attacker)
            .bearer_auth(format!("sk-wrong-{i}"))
            .json(&body)
            .send()
            .await
            .expect("send bad-auth request");
        assert_eq!(
            resp.status(),
            401,
            "attempt {i} (below threshold) must be a plain auth rejection"
        );
    }

    // Past the threshold the brake must 429 the client — even with the
    // CORRECT master key, proving the block gate runs before authentication.
    let blocked = client
        .post(handle.url("/v1/chat/completions"))
        .header("x-forwarded-for", attacker)
        .bearer_auth(MASTER_KEY)
        .json(&body)
        .send()
        .await
        .expect("send post-threshold request");
    assert_eq!(
        blocked.status(),
        429,
        "brute-forcing client must be blocked after the failure threshold"
    );
    assert!(
        blocked.headers().contains_key("retry-after"),
        "block response should carry a retry-after header"
    );
    let text = blocked.text().await.expect("body");
    assert!(
        text.contains("too many auth failures"),
        "expected the auth-brake block message, got: {text}"
    );

    // The block is per client IP: a different caller is unaffected and the
    // request still reaches the upstream.
    let other = client
        .post(handle.url("/v1/chat/completions"))
        .header("x-forwarded-for", "203.0.113.8")
        .bearer_auth(MASTER_KEY)
        .json(&body)
        .send()
        .await
        .expect("send request from another client IP");
    assert_eq!(
        other.status(),
        200,
        "the brake must block only the offending client IP"
    );
    upstream.assert_hits(1);

    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn realtime_missing_model_returns_400() {
    let mock = MockServer::start();
    let config = proxy_config(&mock.base_url());
    let (handle, _shared) = start_oxllm(config).await;

    let client = reqwest::Client::new();
    let resp = client
        .get(handle.url("/v1/realtime"))
        .bearer_auth(MASTER_KEY)
        .header("upgrade", "websocket")
        .header("connection", "upgrade")
        .send()
        .await
        .expect("send");

    assert_eq!(
        resp.status(),
        400,
        "realtime without model query param should return 400"
    );
    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn realtime_unknown_model_rejected() {
    use tokio_tungstenite::tungstenite::client::IntoClientRequest;

    let mock = MockServer::start();
    let config = proxy_config(&mock.base_url());
    let (handle, _shared) = start_oxllm(config).await;

    let ws_url = format!("ws://{}/v1/realtime?model=does-not-exist", handle.addr);
    let mut req = ws_url.into_client_request().expect("build request");
    req.headers_mut().insert(
        "authorization",
        format!("Bearer {MASTER_KEY}").parse().unwrap(),
    );

    let result = tokio_tungstenite::connect_async(req).await;
    assert!(result.is_err(), "unknown model should reject the upgrade");
    if let Err(e) = result {
        let msg = format!("{e}");
        assert!(
            msg.contains("400") || msg.contains("Bad Request"),
            "expected invalid-model 400, got: {msg}"
        );
    }
    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn realtime_proxy_roundtrips_messages() {
    use futures_util::{SinkExt, StreamExt};
    use tokio_tungstenite::tungstenite::client::IntoClientRequest;
    use tokio_tungstenite::tungstenite::protocol::Message as TungsteniteMessage;

    let upstream_listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind upstream");
    let upstream_addr = upstream_listener.local_addr().expect("addr");

    tokio::spawn(async move {
        use tokio_tungstenite::accept_async;
        loop {
            let Ok((stream, _)) = upstream_listener.accept().await else {
                continue;
            };
            tokio::spawn(async move {
                let Ok(mut ws) = accept_async(stream).await else {
                    return;
                };
                while let Some(Ok(msg)) = ws.next().await {
                    if matches!(msg, TungsteniteMessage::Close(_)) {
                        break;
                    }
                    if let TungsteniteMessage::Text(t) = &msg {
                        let echoed = format!("echo:{t}");
                        let _ = ws.send(TungsteniteMessage::Text(echoed.into())).await;
                    }
                }
            });
        }
    });

    let ws_base = format!("http://{}", upstream_addr);
    let mut config = proxy_config(&ws_base);
    config.model_list[0].model_name = "gpt-4o-realtime".to_string();

    let (handle, _shared) = start_oxllm(config).await;

    let ws_url = format!("ws://{}/v1/realtime?model=gpt-4o-realtime", handle.addr);
    let mut req = ws_url.into_client_request().expect("build request");
    req.headers_mut().insert(
        "authorization",
        format!("Bearer {MASTER_KEY}").parse().unwrap(),
    );

    let (mut client_ws, _resp) = tokio_tungstenite::connect_async(req)
        .await
        .expect("connect client ws");

    client_ws
        .send(TungsteniteMessage::Text("ping".into()))
        .await
        .expect("send");

    let reply = tokio::time::timeout(std::time::Duration::from_secs(5), client_ws.next())
        .await
        .expect("timeout")
        .expect("none")
        .expect("err");

    match reply {
        TungsteniteMessage::Text(t) => assert_eq!(t.as_str(), "echo:ping"),
        other => panic!("expected text echo, got {other:?}"),
    }

    let _ = client_ws.close(None).await;
    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn realtime_auth_via_query_param_succeeds() {
    use futures_util::{SinkExt, StreamExt};
    use tokio_tungstenite::tungstenite::client::IntoClientRequest;
    use tokio_tungstenite::tungstenite::protocol::Message as TungsteniteMessage;

    let upstream_listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind upstream");
    let upstream_addr = upstream_listener.local_addr().expect("addr");

    tokio::spawn(async move {
        use tokio_tungstenite::accept_async;
        loop {
            let Ok((stream, _)) = upstream_listener.accept().await else {
                continue;
            };
            tokio::spawn(async move {
                let Ok(mut ws) = accept_async(stream).await else {
                    return;
                };
                while let Some(Ok(msg)) = ws.next().await {
                    if matches!(msg, TungsteniteMessage::Close(_)) {
                        break;
                    }
                    if let TungsteniteMessage::Text(t) = &msg {
                        let echoed = format!("echo:{t}");
                        let _ = ws.send(TungsteniteMessage::Text(echoed.into())).await;
                    }
                }
            });
        }
    });

    let ws_base = format!("http://{}", upstream_addr);
    let mut config = proxy_config(&ws_base);
    config.model_list[0].model_name = "gpt-4o-realtime".to_string();

    let (handle, _shared) = start_oxllm(config).await;

    let ws_url = format!(
        "ws://{}/v1/realtime?model=gpt-4o-realtime&authorization={MASTER_KEY}",
        handle.addr
    );
    let req = ws_url.into_client_request().expect("build request");

    let (mut client_ws, _resp) = tokio_tungstenite::connect_async(req)
        .await
        .expect("query-param auth should succeed");

    client_ws
        .send(TungsteniteMessage::Text("ping".into()))
        .await
        .expect("send");

    let reply = tokio::time::timeout(std::time::Duration::from_secs(5), client_ws.next())
        .await
        .expect("timeout")
        .expect("none")
        .expect("err");

    match reply {
        TungsteniteMessage::Text(t) => assert_eq!(t.as_str(), "echo:ping"),
        other => panic!("expected text echo, got {other:?}"),
    }

    let _ = client_ws.close(None).await;
    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn realtime_auth_via_query_param_rejected_when_wrong() {
    use tokio_tungstenite::tungstenite::client::IntoClientRequest;

    let mock = MockServer::start();
    let mut config = proxy_config(&mock.base_url());
    config.model_list[0].model_name = "gpt-4o-realtime".to_string();

    let (handle, _shared) = start_oxllm(config).await;

    let ws_url = format!(
        "ws://{}/v1/realtime?model=gpt-4o-realtime&authorization=sk-wrong",
        handle.addr
    );
    let req = ws_url.into_client_request().expect("build request");

    let result = tokio_tungstenite::connect_async(req).await;
    assert!(
        result.is_err(),
        "wrong query-param key should reject the upgrade"
    );
    if let Err(e) = result {
        let msg = format!("{e}");
        assert!(
            msg.contains("401") || msg.contains("Unauthorized"),
            "expected 401, got: {msg}"
        );
    }
    let _ = handle.shutdown_tx.send(());
}

// Bug F2: realtime runs pre_call (budget reservation) BEFORE check_limits.
// When the deployment RPM is exhausted, the handler must call post_failure
// before returning the 429 — otherwise the reservation stays pinned for the
// entire budget window.
#[tokio::test]
async fn realtime_releases_budget_when_deployment_rpm_exhausted() {
    let mock = MockServer::start();

    // Open mode (no master key) with a global max_budget so the BudgetHook
    // reserves an estimate for the open principal; deployment rpm=1 so
    // check_limits can be exhausted.
    let mut config = proxy_config(&mock.base_url());
    config.general_settings.master_key = None;
    config.oxllm_settings.max_budget = Some(1.0);
    config.model_list[0].model_name = "gpt-4o-realtime".to_string();
    config.model_list[0].oxllm_params.model = "openai/gpt-4o-realtime".to_string();
    config.model_list[0].oxllm_params.rpm = Some(1);

    let (handle, shared) = start_oxllm(config).await;

    let state = shared.load_full();
    let _ = state
        .cache
        .incr("dep_rpm:gpt-4o-realtime:openai/gpt-4o-realtime", 1.0, None)
        .await;

    let resp = reqwest::Client::new()
        .get(handle.url("/v1/realtime?model=gpt-4o-realtime"))
        .header("upgrade", "websocket")
        .header("connection", "upgrade")
        .header("sec-websocket-version", "13")
        .header("sec-websocket-key", "dGhlIHNhbXBsZSBub25jZQ==")
        .send()
        .await
        .expect("send");

    assert_eq!(
        resp.status(),
        429,
        "deployment rpm exhausted must reject before the upgrade"
    );

    let spend_key = format!(
        "spend:{}",
        oxllm_server::auth::AuthUser::open().key_hash_hex()
    );
    let spend = state.cache.counter_value(&spend_key).await;
    let leaked = spend.map(|v| v.abs() >= 1e-9).unwrap_or(false);
    assert!(
        !leaked,
        "budget reservation leaked on realtime deployment 429: spend counter = {spend:?} (should be ~0)"
    );

    let _ = handle.shutdown_tx.send(());
}

// C21 HIGH #2: realtime token spend was never metered — the
// `RealtimeProvider::extract_usage` hook existed but the host never called it,
// and sessions always settled via post_failure (releasing the reservation in
// full, net spend zero). Now the proxy extracts usage from upstream events and
// settles the actual cost via post_success at session close.
#[tokio::test]
async fn realtime_session_meters_reported_token_usage() {
    use futures_util::{SinkExt, StreamExt};
    use tokio_tungstenite::tungstenite::client::IntoClientRequest;
    use tokio_tungstenite::tungstenite::protocol::Message as TungsteniteMessage;

    let upstream_listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind upstream");
    let upstream_addr = upstream_listener.local_addr().expect("addr");

    tokio::spawn(async move {
        use tokio_tungstenite::accept_async;
        loop {
            let Ok((stream, _)) = upstream_listener.accept().await else {
                continue;
            };
            tokio::spawn(async move {
                let Ok(mut ws) = accept_async(stream).await else {
                    return;
                };
                while let Some(Ok(msg)) = ws.next().await {
                    if matches!(msg, TungsteniteMessage::Close(_)) {
                        break;
                    }
                    if let TungsteniteMessage::Text(_) = &msg {
                        // Reply with a response.done carrying usage so the
                        // identity provider's extract_usage surfaces it.
                        let done = r#"{"type":"response.done","response":{"usage":{"input_tokens":10,"output_tokens":20,"total_tokens":30}}}"#;
                        let _ = ws.send(TungsteniteMessage::Text(done.into())).await;
                    }
                }
            });
        }
    });

    // Open mode + a global max_budget so the BudgetHook reserves for the open
    // principal; "gpt-4o" is a priced model so the settled actual cost is
    // non-zero.
    let ws_base = format!("http://{}", upstream_addr);
    let mut config = proxy_config(&ws_base);
    config.general_settings.master_key = None;
    config.oxllm_settings.max_budget = Some(1.0);
    config.model_list[0].model_name = "gpt-4o".to_string();
    config.model_list[0].oxllm_params.model = "openai/gpt-4o".to_string();

    let (handle, shared) = start_oxllm(config).await;
    let state = shared.load_full();

    let ws_url = format!("ws://{}/v1/realtime?model=gpt-4o", handle.addr);
    let req = ws_url.into_client_request().expect("build request");

    let (mut client_ws, _resp) = tokio_tungstenite::connect_async(req)
        .await
        .expect("connect client ws");

    client_ws
        .send(TungsteniteMessage::Text("ping".into()))
        .await
        .expect("send");

    // Receive the response.done the upstream sent back. meter_upstream_text
    // runs before the frame is forwarded to the client, so by the time we
    // receive it the session accumulator already holds the usage.
    let _ = tokio::time::timeout(std::time::Duration::from_secs(5), client_ws.next())
        .await
        .expect("timeout waiting for upstream response.done")
        .expect("stream closed")
        .expect("ws error");

    let _ = client_ws.close(None).await;

    let spend_key = format!(
        "spend:{}",
        oxllm_server::auth::AuthUser::open().key_hash_hex()
    );

    // The session settles asynchronously in the on_upgrade teardown once the
    // pump returns. Poll the spend counter until it reflects the debit.
    let mut settled = false;
    for _ in 0..40 {
        if let Some(v) = state.cache.counter_value(&spend_key).await {
            if v.abs() >= 1e-12 {
                settled = true;
                break;
            }
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    assert!(
        settled,
        "realtime session with reported usage must settle non-zero spend"
    );

    let _ = handle.shutdown_tx.send(());
}
// C21 MEDIUM #4: setup-failure paths in handle_realtime_upgrade returned errors
// AFTER pre_call reserved budget/RPM, leaking the reservation for the whole
// budget window. Each must now call post_failure before propagating. This
// exercises the unknown-provider-slug path (resolve_realtime_provider fails
// post-reserve) without depending on ambient AWS env vars.
#[tokio::test]
async fn realtime_releases_budget_when_provider_unresolvable() {
    let mock = MockServer::start();
    let mut config = proxy_config(&mock.base_url());
    config.general_settings.master_key = None;
    config.oxllm_settings.max_budget = Some(1.0);
    config.model_list[0].model_name = "rt-unknown".to_string();
    config.model_list[0].oxllm_params.model = "acme/rt-model".to_string();

    let (handle, shared) = start_oxllm(config).await;
    let state = shared.load_full();

    let resp = reqwest::Client::new()
        .get(handle.url("/v1/realtime?model=rt-unknown"))
        .header("upgrade", "websocket")
        .header("connection", "upgrade")
        .header("sec-websocket-version", "13")
        .header("sec-websocket-key", "dGhlIHNhbXBsZSBub25jZQ==")
        .send()
        .await
        .expect("send");

    // Unknown slug → 400 (bad_request), and the reservation must be released.
    assert_eq!(
        resp.status(),
        400,
        "unknown provider slug must surface as 400, got {}",
        resp.status()
    );

    let spend_key = format!(
        "spend:{}",
        oxllm_server::auth::AuthUser::open().key_hash_hex()
    );
    let mut released = false;
    for _ in 0..40 {
        let spend = state.cache.counter_value(&spend_key).await;
        let leaked = spend.map(|v| v.abs() >= 1e-9).unwrap_or(false);
        if !leaked {
            released = true;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    assert!(
        released,
        "budget reservation leaked on unresolvable-realtime-provider setup failure"
    );

    let _ = handle.shutdown_tx.send(());
}

/// HIGH #1: a realtime session whose upstream closes without ever emitting a
/// usage event (no `response.done`) must settle as a **successful unmetered**
/// session — the budget reservation is released (spend ~0) and no error-rate
/// metric is inflated. Previously `settle_realtime_session` called
/// `post_failure` with a synthetic network error for every unmetered session,
/// recording them as failures. The spend release alone cannot tell the two
/// apart (both release), so this test also scrapes `/metrics` for the unique
/// model label: a `post_success` settlement records
/// `oxllm_requests_total{status="success"}` and no `status="error"` series,
/// while the buggy `post_failure` path records the exact opposite.
#[tokio::test]
async fn realtime_unmetered_session_settles_as_success_not_failure() {
    use futures_util::StreamExt;
    use tokio_tungstenite::tungstenite::client::IntoClientRequest;

    // Install the Prometheus recorder before the request so the settlement
    // hook's recording is captured (records before install are no-ops).
    oxllm_server::metrics::install();

    let upstream_listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind upstream");
    let upstream_addr = upstream_listener.local_addr().expect("addr");

    tokio::spawn(async move {
        use tokio_tungstenite::accept_async;
        loop {
            let Ok((stream, _)) = upstream_listener.accept().await else {
                continue;
            };
            tokio::spawn(async move {
                let Ok(mut ws) = accept_async(stream).await else {
                    return;
                };
                // Accept the connection then immediately close WITHOUT ever
                // sending a response.done usage event. This simulates a
                // clean session that ends before the upstream reports usage.
                let _ = ws.close(None).await;
            });
        }
    });

    // A model label unique to this test: the metric series is global to the
    // process, so a parallel test must not be able to pollute it.
    let model = "gpt-4o-realtime-unmetered-probe";
    let ws_base = format!("http://{}", upstream_addr);
    let mut config = proxy_config(&ws_base);
    config.general_settings.master_key = None;
    config.oxllm_settings.max_budget = Some(1.0);
    config.model_list[0].model_name = model.to_string();
    config.model_list[0].oxllm_params.model = format!("openai/{model}");

    let (handle, shared) = start_oxllm(config).await;
    let state = shared.load_full();

    let ws_url = format!("ws://{}/v1/realtime?model={model}", handle.addr);
    let req = ws_url.into_client_request().expect("build request");
    let (mut client_ws, _resp) = tokio_tungstenite::connect_async(req)
        .await
        .expect("connect client ws");

    // Wait for the upstream to close (session ends without usage).
    let _ = tokio::time::timeout(std::time::Duration::from_secs(5), client_ws.next()).await;
    let _ = client_ws.close(None).await;

    let spend_key = format!(
        "spend:{}",
        oxllm_server::auth::AuthUser::open().key_hash_hex()
    );

    // Poll the spend counter: the pre-call reservation must be released
    // (spend returns to ~0) — proving the unmetered session settled at all.
    let mut released = false;
    for _ in 0..40 {
        let spend = state.cache.counter_value(&spend_key).await;
        let leaked = spend.map(|v| v.abs() >= 1e-9).unwrap_or(false);
        if !leaked {
            released = true;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    assert!(
        released,
        "unmetered realtime session must release the budget reservation (spend ~0)"
    );

    // Distinguish post_success from post_failure via the settlement hook's own
    // metric: the success series must appear and the error series must not.
    let success_series =
        format!("oxllm_requests_total{{model=\"{model}\",provider=\"openai\",status=\"success\"}}");
    let error_fragment = format!("model=\"{model}\"");
    let client = reqwest::Client::new();
    let mut rendered = String::new();
    let mut recorded_success = false;
    for _ in 0..40 {
        rendered = client
            .get(handle.url("/metrics"))
            .send()
            .await
            .expect("metrics request")
            .text()
            .await
            .expect("metrics body");
        if rendered
            .lines()
            .any(|line| line.starts_with(&success_series))
        {
            recorded_success = true;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    assert!(
        recorded_success,
        "unmetered session must settle through post_success, but no success series \
         was recorded; /metrics had:\n{rendered}"
    );
    let error_series = rendered.lines().any(|line| {
        line.starts_with("oxllm_requests_total{")
            && line.contains(&error_fragment)
            && line.contains("status=\"error\"")
    });
    assert!(
        !error_series,
        "post_success must not also record an error series:\n{rendered}"
    );

    let _ = handle.shutdown_tx.send(());
}

async fn realtime_disconnect_peer(
    final_usage: Option<Value>,
) -> (std::net::SocketAddr, tokio::task::JoinHandle<bool>) {
    use futures_util::{SinkExt, StreamExt};
    use tokio_tungstenite::tungstenite::Message;
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let task = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let mut upstream = tokio_tungstenite::accept_async(stream).await.unwrap();
        let request = upstream.next().await.unwrap().unwrap();
        assert_eq!(
            serde_json::from_str::<Value>(request.to_text().unwrap()).unwrap()["type"],
            "response.create"
        );
        upstream
            .send(Message::Text(
                json!({
                    "type":"response.created","response":{"id":"r1"}
                })
                .to_string()
                .into(),
            ))
            .await
            .unwrap();
        upstream
            .send(Message::Text(
                json!({
                    "type":"response.text.delta","response_id":"r1","delta":"hello"
                })
                .to_string()
                .into(),
            ))
            .await
            .unwrap();
        while let Some(Ok(message)) = upstream.next().await {
            match message {
                Message::Text(text) => {
                    let event: Value = serde_json::from_str(&text).unwrap();
                    if event["type"] == "response.cancel" {
                        if let Some(usage) = final_usage {
                            upstream
                                .send(Message::Text(
                                    json!({
                                        "type":"response.done","response":{
                                            "id":"r1","status":"cancelled","usage":usage
                                        }
                                    })
                                    .to_string()
                                    .into(),
                                ))
                                .await
                                .unwrap();
                            upstream.close(None).await.unwrap();
                        } else {
                            // Deliberately withhold both usage and close. The
                            // gateway's drain deadline must end the session.
                            tokio::time::timeout(std::time::Duration::from_secs(5), async {
                                while let Some(Ok(message)) = upstream.next().await {
                                    if matches!(message, Message::Close(_)) {
                                        return;
                                    }
                                }
                            })
                            .await
                            .expect("gateway must bound the usage drain");
                        }
                        return true;
                    }
                }
                Message::Close(_) => return false,
                _ => {}
            }
        }
        false
    });
    (address, task)
}

type TestRealtimeSocket =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

async fn receive_generated_realtime_delta(client: &mut TestRealtimeSocket) {
    use futures_util::StreamExt;
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        while let Some(Ok(message)) = client.next().await {
            if let Ok(text) = message.to_text() {
                if serde_json::from_str::<Value>(text).unwrap()["type"] == "response.text.delta" {
                    return;
                }
            }
        }
        panic!("session ended before generated output");
    })
    .await
    .expect("generated output must arrive");
}

fn realtime_budget_config(address: std::net::SocketAddr) -> ProxyConfig {
    let mut config = proxy_config(&format!("http://{address}"));
    config.general_settings.master_key = None;
    config.oxllm_settings.max_budget = Some(1.0);
    config.model_list[0].model_name = "rt-alias".into();
    config.model_list[0].oxllm_params.input_cost_per_token = Some(0.00001);
    config.model_list[0].oxllm_params.output_cost_per_token = Some(0.00002);
    config
}

#[tokio::test]
async fn realtime_disconnect_cancels_and_drains_actual_usage_with_alias_pricing() {
    use futures_util::SinkExt;
    use tokio_tungstenite::tungstenite::Message;
    let (address, peer) = realtime_disconnect_peer(Some(json!({
        "input_tokens":2,"output_tokens":3,"total_tokens":5
    })))
    .await;
    let (handle, shared) = start_oxllm(realtime_budget_config(address)).await;
    let state = shared.load_full();
    let (mut client, _) = tokio_tungstenite::connect_async(format!(
        "ws://{}/v1/realtime?model=rt-alias",
        handle.addr
    ))
    .await
    .unwrap();
    client
        .send(Message::Text(r#"{"type":"response.create"}"#.into()))
        .await
        .unwrap();
    receive_generated_realtime_delta(&mut client).await;
    client.close(None).await.unwrap();
    assert!(
        tokio::time::timeout(std::time::Duration::from_secs(5), peer)
            .await
            .unwrap()
            .unwrap(),
        "disconnect must send cancellation before closing upstream"
    );
    let spend_key = format!(
        "spend:{}",
        oxllm_server::auth::AuthUser::open().key_hash_hex()
    );
    // Check the exact cost, not merely nonzero (which could still be the
    // pre-call reservation). Both public-alias lookup and override loss fail.
    let expected = 2.0 * 0.00001 + 3.0 * 0.00002;
    tokio::time::timeout(std::time::Duration::from_secs(5), async {
        loop {
            if state
                .cache
                .counter_value(&spend_key)
                .await
                .is_some_and(|v| (v - expected).abs() < 1e-12)
            {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("drained usage must settle the exact reported token cost");
    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn realtime_partial_output_without_usage_retains_estimate_and_records_unknown_cost() {
    use futures_util::SinkExt;
    use tokio_tungstenite::tungstenite::Message;
    let (address, peer) = realtime_disconnect_peer(None).await;
    let (handle, shared) = start_oxllm(realtime_budget_config(address)).await;
    let (tx, mut rx) = tokio::sync::mpsc::channel(4);
    let mut updated = (*shared.load_full()).clone();
    updated.spend_tx = Some(tx);
    shared.store(std::sync::Arc::new(updated));
    let state = shared.load_full();
    let (mut client, _) = tokio_tungstenite::connect_async(format!(
        "ws://{}/v1/realtime?model=rt-alias",
        handle.addr
    ))
    .await
    .unwrap();
    client
        .send(Message::Text(r#"{"type":"response.create"}"#.into()))
        .await
        .unwrap();
    receive_generated_realtime_delta(&mut client).await;
    let spend_key = format!(
        "spend:{}",
        oxllm_server::auth::AuthUser::open().key_hash_hex()
    );
    let reserved = state.cache.counter_value(&spend_key).await.unwrap();
    assert!(reserved > 0.0);
    client.close(None).await.unwrap();
    assert!(
        tokio::time::timeout(std::time::Duration::from_secs(5), peer)
            .await
            .unwrap()
            .unwrap()
    );
    let entry = tokio::time::timeout(std::time::Duration::from_secs(5), rx.recv())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(entry.status.as_deref(), Some("partial"));
    assert!(entry.cost.is_none());
    assert!(entry.total_tokens.is_none());
    let settled = state.cache.counter_value(&spend_key).await.unwrap();
    assert!(
        (settled - reserved).abs() < 1e-12,
        "unreported delivered output must not be refunded to zero"
    );
    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn realtime_connect_enforces_current_private_address_policy() {
    use futures_util::StreamExt;
    use tokio_tungstenite::tungstenite::Message;
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let (handle, shared) =
        start_oxllm(realtime_budget_config(listener.local_addr().unwrap())).await;
    // Simulate a destination becoming forbidden after initial validation.
    // The live WS connect must independently enforce the current policy.
    let mut updated = (*shared.load_full()).clone();
    let mut config = (*updated.config).clone();
    config.general_settings.allow_private_api_base = Some(false);
    updated.config = std::sync::Arc::new(config);
    shared.store(std::sync::Arc::new(updated));
    let state = shared.load_full();
    let (mut client, _) = tokio_tungstenite::connect_async(format!(
        "ws://{}/v1/realtime?model=rt-alias",
        handle.addr
    ))
    .await
    .unwrap();
    let message = tokio::time::timeout(std::time::Duration::from_secs(3), client.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(matches!(message, Message::Close(Some(close)) if u16::from(close.code) == 1011));
    assert!(
        tokio::time::timeout(std::time::Duration::from_millis(100), listener.accept())
            .await
            .is_err(),
        "no connection or bearer header may reach a forbidden destination"
    );
    let spend_key = format!(
        "spend:{}",
        oxllm_server::auth::AuthUser::open().key_hash_hex()
    );
    tokio::time::timeout(std::time::Duration::from_secs(3), async {
        loop {
            if state
                .cache
                .counter_value(&spend_key)
                .await
                .unwrap_or(0.0)
                .abs()
                < 1e-12
            {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("connect rejection must release its reservation");
    let _ = handle.shutdown_tx.send(());
}
