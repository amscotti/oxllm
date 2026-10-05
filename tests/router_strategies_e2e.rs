//! Integration tests: router strategy failover semantics — bounded pick
//! loops under `cooldown_time: 0`, deployment-fault 4xx sibling failover
//! (LiteLLM AuthenticationError retry parity), round-robin alternation
//! through the proxy, and the non-secret shape of health metric labels.

use httpmock::{Mock, MockServer};
use oxllm_server::OxllmParams;
use oxllm_server::app::{build_router, build_shared_state};
use oxllm_server::config::types::{
    ConfigModel, GeneralSettings, OxllmSettings, ProxyConfig, RouterSettings, RoutingKnobs,
};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::time::{Duration, Instant};

struct OxllmHandle {
    addr: std::net::SocketAddr,
    shutdown_tx: tokio::sync::oneshot::Sender<()>,
}

impl OxllmHandle {
    fn url(&self, path: &str) -> String {
        format!("http://{}{path}", self.addr)
    }
}

fn params(base: &str, key: &str) -> OxllmParams {
    OxllmParams {
        model: "openai/gpt-4o".to_string(),
        api_key: Some(key.to_string()),
        api_base: Some(base.to_string()),
        timeout: Some(30.0),
        stream_timeout: Some(30),
        ..Default::default()
    }
}

fn model_entry(name: &str, base: &str, key: &str) -> ConfigModel {
    ConfigModel {
        model_name: name.to_string(),
        oxllm_params: params(base, key),
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

fn setup_status_mock<'a>(mock: &'a MockServer, status: u16, key: &str) -> Mock<'a> {
    mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .header("authorization", format!("Bearer {key}"));
        then.status(status)
            .header("content-type", "application/json")
            .json_body(json!({
            "error": {
            "message": "upstream failure",
            "type": if status >= 500 { "server_error" } else { "invalid_request_error" },
            "code": "internal_error"
            }
            }));
    })
}

fn build_config(
    model_list: Vec<ConfigModel>,
    routing: RoutingKnobs,
    router_settings_extra: RouterSettings,
) -> ProxyConfig {
    let routing_groups = router_settings_extra.routing_groups;
    ProxyConfig {
        model_list,
        oxllm_settings: OxllmSettings {
            routing,
            ..Default::default()
        },
        general_settings: GeneralSettings {
            allow_private_api_base: Some(true),
            ..Default::default()
        },
        router_settings: RouterSettings {
            routing_groups,
            ..Default::default()
        },
        extra: serde_json::Map::new(),
    }
}

async fn start_oxllm(
    config: ProxyConfig,
) -> anyhow::Result<(OxllmHandle, oxllm_server::SharedState)> {
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

fn request_body(model: &str) -> Value {
    json!({
    "model": model,
    "messages": [{"role": "user", "content": "hi"}],
    })
}

async fn send(handle: &OxllmHandle, model: &str) -> reqwest::Response {
    reqwest::Client::new()
        .post(handle.url("/v1/chat/completions"))
        .json(&request_body(model))
        .send()
        .await
        .expect("request should succeed")
}

// ---------------------------------------------------------------------------
// cooldown_time: 0 — the chat retry loop must terminate (bounded picks)
// ---------------------------------------------------------------------------

/// A `cooldown_time` of 0 never excludes a failed deployment from rotation
/// (the mark expires instantly), so the outer pick loop — not the cooldown
/// store — is the only termination guarantee. The loop must bound itself at
/// `deployments × (num_retries + 1)` picks: with one deployment and
/// `num_retries: 1` the upstream sees exactly 4 attempts (2 picks × 2
/// attempts) and the request resolves with an error instead of hanging.
#[tokio::test]
async fn zero_cooldown_chat_loop_terminates_with_bounded_attempts() {
    let mock = MockServer::start();
    let upstream = setup_status_mock(&mock, 500, "sk-test-key");

    let config = build_config(
        vec![model_entry("gpt-4o", &mock.base_url(), "sk-test-key")],
        RoutingKnobs {
            num_retries: Some(1),
            cooldown_time: Some(0),
            ..Default::default()
        },
        RouterSettings::default(),
    );
    let (handle, _shared) = start_oxllm(config).await.expect("server starts");

    let started = Instant::now();
    let resp = tokio::time::timeout(Duration::from_secs(30), send(&handle, "gpt-4o"))
        .await
        .expect("request must terminate (bounded pick loop)");
    let elapsed = started.elapsed();

    assert!(
        !resp.status().is_success(),
        "a persistently-500 upstream must surface an error"
    );
    assert!(
        elapsed < Duration::from_secs(20),
        "bounded loop must not spin; took {elapsed:?}"
    );
    assert_eq!(
        upstream.hits(),
        4,
        "1 deployment × (num_retries 1 + 1) picks × 2 attempts per pick = 4 \
         upstream hits — the loop must not retry the zero-cooldown \
         deployment forever"
    );

    let _ = handle.shutdown_tx.send(());
}

// ---------------------------------------------------------------------------
// 4xx sibling failover (LiteLLM AuthenticationError retry parity)
// ---------------------------------------------------------------------------

/// With a per-status allowed-fails policy (`401: 0`) and a healthy sibling,
/// a 401-ing deployment is cooled and the request fails over instead of
/// failing the caller — LiteLLM retries AuthenticationError "if other
/// deployments available". Every request must return 200, and the broken
/// deployment must be hit at most once before its cooldown keeps traffic
/// away.
#[tokio::test]
async fn auth_error_fails_over_to_healthy_sibling() {
    let mock_bad = MockServer::start();
    let mock_good = MockServer::start();
    let bad = setup_status_mock(&mock_bad, 401, "sk-bad-key");
    let good = setup_ok_mock(&mock_good, "from-good", "sk-good-key");

    let config = build_config(
        vec![
            model_entry("gpt-4o", &mock_bad.base_url(), "sk-bad-key"),
            model_entry("gpt-4o", &mock_good.base_url(), "sk-good-key"),
        ],
        RoutingKnobs {
            allowed_fails_policy: Some(HashMap::from([("401".to_string(), 0)])),
            ..Default::default()
        },
        RouterSettings::default(),
    );
    let (handle, _shared) = start_oxllm(config).await.expect("server starts");

    for _ in 0..10 {
        let resp = send(&handle, "gpt-4o").await;
        assert_eq!(
            resp.status(),
            200,
            "a 401 from one deployment must fail over to the healthy sibling"
        );
        let body: Value = resp.json().await.expect("json");
        assert_eq!(body["choices"][0]["message"]["content"], "from-good");
    }
    assert_eq!(
        good.hits(),
        10,
        "every request must be served by the healthy sibling"
    );
    assert!(
        bad.hits() <= 1,
        "the 401-ing deployment must be cooled after its first failure \
         (policy limit 0), got {} hits",
        bad.hits()
    );

    let _ = handle.shutdown_tx.send(());
}

/// Without a policy covering the status, a client-caused 4xx must NOT fan
/// out across deployments: the error returns immediately and the sibling is
/// untouched.
#[tokio::test]
async fn uncovered_client_error_does_not_fan_out() {
    let mock_bad = MockServer::start();
    let mock_good = MockServer::start();
    let bad = setup_status_mock(&mock_bad, 400, "sk-bad-key");
    let good = setup_ok_mock(&mock_good, "from-good", "sk-good-key");

    let config = build_config(
        vec![
            model_entry("gpt-4o", &mock_bad.base_url(), "sk-bad-key"),
            model_entry("gpt-4o", &mock_good.base_url(), "sk-good-key"),
        ],
        RoutingKnobs::default(),
        RouterSettings::default(),
    );
    let (handle, _shared) = start_oxllm(config).await.expect("server starts");

    let mut saw_400 = false;
    let mut ok_count = 0;
    for _ in 0..10 {
        let resp = send(&handle, "gpt-4o").await;
        if resp.status() == 400 {
            saw_400 = true;
        } else {
            assert_eq!(resp.status(), 200);
            ok_count += 1;
        }
    }
    assert!(saw_400, "a 400 no policy covers must surface to the caller");
    // One upstream hit per request: a request that drew the broken
    // deployment returns its 400 immediately — it must not be retried onto
    // the sibling (which would add a second hit for that request).
    assert_eq!(
        bad.hits() + good.hits(),
        10,
        "client-caused 4xx must not fan out across deployments (bad={}, good={})",
        bad.hits(),
        good.hits()
    );
    assert_eq!(bad.hits(), 10 - ok_count);

    let _ = handle.shutdown_tx.send(());
}

// ---------------------------------------------------------------------------
// round_robin through the proxy
// ---------------------------------------------------------------------------

/// Strict alternation across two upstreams by request order, plus the
/// strategy header plumbing.
#[tokio::test]
async fn round_robin_strictly_alternates_upstreams() {
    let mock_a = MockServer::start();
    let mock_b = MockServer::start();
    let a = setup_ok_mock(&mock_a, "from-a", "key-a");
    let b = setup_ok_mock(&mock_b, "from-b", "key-b");

    let config = build_config(
        vec![
            model_entry("gpt-4o", &mock_a.base_url(), "key-a"),
            model_entry("gpt-4o", &mock_b.base_url(), "key-b"),
        ],
        RoutingKnobs {
            routing_strategy: Some("round_robin".to_string()),
            ..Default::default()
        },
        RouterSettings::default(),
    );
    let (handle, _shared) = start_oxllm(config).await.expect("server starts");

    let mut sequence: Vec<&str> = Vec::new();
    for _ in 0..6 {
        let resp = send(&handle, "gpt-4o").await;
        assert_eq!(resp.status(), 200);
        assert_eq!(
            resp.headers()
                .get("x-oxllm-router-strategy")
                .and_then(|v| v.to_str().ok()),
            Some("round_robin")
        );
        let body: Value = resp.json().await.expect("json");
        let content = body["choices"][0]["message"]["content"]
            .as_str()
            .expect("str");
        sequence.push(match content {
            "from-a" => "a",
            "from-b" => "b",
            other => panic!("unexpected content {other}"),
        });
    }
    assert_eq!(a.hits(), 3);
    assert_eq!(b.hits(), 3);
    for pair in sequence.windows(2) {
        assert_ne!(
            pair[0], pair[1],
            "round robin must strictly alternate, saw {sequence:?}"
        );
    }

    let _ = handle.shutdown_tx.send(());
}

/// The per-model-group counter survives a config hot-reload that keeps the
/// same deployments: the rotation position must not reset to
/// `deployments[0]` mid-stream.
#[tokio::test]
async fn round_robin_position_survives_config_reload() {
    let mock_a = MockServer::start();
    let mock_b = MockServer::start();
    let _a = setup_ok_mock(&mock_a, "from-a", "key-a");
    let _b = setup_ok_mock(&mock_b, "from-b", "key-b");

    let make_config = || {
        build_config(
            vec![
                model_entry("gpt-4o", &mock_a.base_url(), "key-a"),
                model_entry("gpt-4o", &mock_b.base_url(), "key-b"),
            ],
            RoutingKnobs {
                routing_strategy: Some("round_robin".to_string()),
                ..Default::default()
            },
            RouterSettings::default(),
        )
    };
    let (handle, shared) = start_oxllm(make_config()).await.expect("server starts");

    let content_of = |resp: reqwest::Response| async move {
        let body: Value = resp.json().await.expect("json");
        body["choices"][0]["message"]["content"]
            .as_str()
            .expect("str")
            .to_string()
    };
    let first = content_of(send(&handle, "gpt-4o").await).await;
    let second = content_of(send(&handle, "gpt-4o").await).await;
    assert_ne!(first, second, "rotation must alternate before reload");

    // Hot-reload the same config: the router is rebuilt from the preserved
    // stores, so the rotation position continues rather than resetting.
    let old = shared.load_full();
    let new_router = oxllm_core::router::Router::rebuild_from_stores(
        old.router.config().clone(),
        old.router.cooldown_arc(),
        old.router.latency_store_arc(),
        old.router.usage_store_arc(),
        old.router.health_store_arc(),
        old.router.bandit_store_arc(),
        old.router.round_robin_counters_arc(),
    );
    let mut new_state = (*old).clone();
    new_state.router = std::sync::Arc::new(new_router);
    shared.store(std::sync::Arc::new(new_state));

    let third = content_of(send(&handle, "gpt-4o").await).await;
    assert_eq!(
        third, first,
        "after a same-config reload the rotation must continue from its \
         saved position (next in {first} → {second} → ?), not restart at \
         deployments[0]"
    );

    let _ = handle.shutdown_tx.send(());
}

// ---------------------------------------------------------------------------
// Health metric label hygiene
// ---------------------------------------------------------------------------

/// `oxllm_health_check_unhealthy_total`'s `deployment` label must be the
/// non-secret `deployment_key` (`{model_name}:{provider_model}`) — never the
/// stable cooldown key, whose 16-hex digest covers credential material and
/// would give a low-privilege `/metrics` scraper an offline verification
/// oracle for guessed API keys.
#[tokio::test]
#[serial_test::serial]
async fn health_unhealthy_metric_label_is_not_the_stable_cooldown_key() {
    oxllm_server::metrics::install();
    let config = ProxyConfig {
        model_list: vec![model_entry(
            "gpt-4o",
            "http://127.0.0.1:1",
            "sk-unreachable-key",
        )],
        oxllm_settings: OxllmSettings::default(),
        general_settings: GeneralSettings {
            allow_private_api_base: Some(true),
            ..Default::default()
        },
        router_settings: RouterSettings {
            enable_health_check_routing: Some(true),
            health_check_failure_threshold: Some(1),
            ..Default::default()
        },
        extra: serde_json::Map::new(),
    };
    let (handle, _shared) = start_oxllm(config).await.expect("server starts");

    let resp = tokio::time::timeout(Duration::from_secs(30), send(&handle, "gpt-4o"))
        .await
        .expect("request terminates");
    assert!(!resp.status().is_success());

    let metrics = reqwest::get(handle.url("/metrics"))
        .await
        .expect("metrics fetch")
        .text()
        .await
        .expect("metrics body");
    let line = metrics
        .lines()
        .find(|l| l.starts_with("oxllm_health_check_unhealthy_total"))
        .expect("unhealthy metric emitted after connection failures");
    assert!(
        line.contains(r#"deployment="gpt-4o:openai/gpt-4o""#),
        "label must be the non-secret deployment_key, got: {line}"
    );
    // No label anywhere on the endpoint may carry the stable-key digest
    // shape (`{model}:{16-hex}`), which covers credential material.
    for l in metrics.lines().filter(|l| l.contains("deployment=\"")) {
        let label = l
            .split("deployment=\"")
            .nth(1)
            .and_then(|rest| rest.split('"').next())
            .unwrap_or("");
        let looks_like_stable_key = label.rsplit(':').next().is_some_and(|digest| {
            digest.len() == 16 && digest.chars().all(|c| c.is_ascii_hexdigit())
        });
        assert!(
            !looks_like_stable_key,
            "credential-derived stable-key shape must not surface on /metrics: {l}"
        );
    }

    let _ = handle.shutdown_tx.send(());
}
