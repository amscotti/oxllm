//! Integration tests: least_busy, weighted shuffle, content-policy fallbacks, retry_policy, model_group_alias.

use httpmock::{Mock, MockServer};
use oxllm_server::OxllmParams;
use oxllm_server::SharedState;
use oxllm_server::app::{build_router, build_shared_state};
use oxllm_server::config::types::{
    ConfigModel, GeneralSettings, OxllmSettings, ProxyConfig, RouterSettings, RoutingKnobs,
};
use serde_json::{Value, json};
use std::collections::HashMap;

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

fn model_entry_with_model(name: &str, litellm_model: &str, base: &str, key: &str) -> ConfigModel {
    let mut p = params(base, key);
    p.model = litellm_model.to_string();
    ConfigModel {
        model_name: name.to_string(),
        oxllm_params: p,
        ..Default::default()
    }
}

fn model_entry_with_weight(name: &str, base: &str, key: &str, weight: Option<u32>) -> ConfigModel {
    let mut p = params(base, key);
    p.weight = weight;
    ConfigModel {
        model_name: name.to_string(),
        oxllm_params: p,
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

fn setup_fail_mock<'a>(mock: &'a MockServer, status: u16, key: &str) -> Mock<'a> {
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

fn setup_content_policy_mock<'a>(mock: &'a MockServer, key: &str) -> Mock<'a> {
    mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .header("authorization", format!("Bearer {key}"));
        then.status(400)
            .header("content-type", "application/json")
            .json_body(json!({
            "error": {
            "code": "content_policy_violation",
            "message": "Output blocked by content filtering policy",
            "type": "invalid_request_error"
            }
            }));
    })
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

fn build_config_with_strategy(
    model_list: Vec<ConfigModel>,
    settings: OxllmSettings,
    router_settings: RouterSettings,
) -> ProxyConfig {
    ProxyConfig {
        model_list,
        oxllm_settings: settings,
        general_settings: GeneralSettings {
            allow_private_api_base: Some(true),
            ..Default::default()
        },
        router_settings,
        extra: serde_json::Map::new(),
    }
}

fn make_request_body(model: &str) -> Value {
    json!({
    "model": model,
    "messages": [{"role": "user", "content": "hi"}],
    })
}

async fn send_request(handle: &OxllmHandle, model: &str) -> reqwest::Response {
    reqwest::Client::new()
        .post(handle.url("/v1/chat/completions"))
        .json(&make_request_body(model))
        .send()
        .await
        .expect("request should succeed")
}

// ---------------------------------------------------------------------------
// least_busy
// ---------------------------------------------------------------------------

/// When one deployment has in-flight traffic recorded, `least_busy` should
/// prefer the idle one. We pre-mark the busy deployment by hitting it once
/// and relying on the fact that the in-flight count is incremented around
/// the upstream call. Since requests complete quickly, we instead drive the
/// strategy directly by checking the router's recorded in-flight counts.
#[tokio::test]
async fn least_busy_strategy_prefers_idle_deployment() {
    let mock_a = MockServer::start();
    let mock_b = MockServer::start();

    let _a = setup_ok_mock(&mock_a, "from-a", "key-a");
    let _b = setup_ok_mock(&mock_b, "from-b", "key-b");

    let config = build_config_with_strategy(
        vec![
            model_entry_with_model("gpt-4o", "openai/gpt-4o-a", &mock_a.base_url(), "key-a"),
            model_entry_with_model("gpt-4o", "openai/gpt-4o-b", &mock_b.base_url(), "key-b"),
        ],
        OxllmSettings {
            routing: RoutingKnobs {
                routing_strategy: Some("least_busy".to_string()),
                ..Default::default()
            },
            ..Default::default()
        },
        RouterSettings::default(),
    );
    let (handle, shared) = start_oxllm(config).await.expect("server starts");

    // Pre-mark deployment A as much busier than B via the shared router,
    // using the stable per-deployment key (the same key space the strategies
    // and host feed now use — sibling deployments sharing a provider model
    // get distinct counters).
    let a_dep: oxllm_core::Deployment =
        model_entry_with_model("gpt-4o", "openai/gpt-4o-a", &mock_a.base_url(), "key-a").into();
    shared
        .load_full()
        .router
        .record_in_flight(&oxllm_core::router::stable_key(&a_dep), 10);

    let mut saw_b = false;
    for _ in 0..40 {
        let resp = send_request(&handle, "gpt-4o").await;
        assert_eq!(resp.status(), 200);
        let body: Value = resp.json().await.expect("json");
        if body["choices"][0]["message"]["content"] == "from-b" {
            saw_b = true;
        }
    }
    assert!(
        saw_b,
        "least_busy should route to the idle deployment at least once"
    );

    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn least_busy_strategy_no_data_falls_back_to_shuffle() {
    let mock_a = MockServer::start();
    let mock_b = MockServer::start();

    let _a = setup_ok_mock(&mock_a, "from-a", "key-a");
    let _b = setup_ok_mock(&mock_b, "from-b", "key-b");

    let config = build_config_with_strategy(
        vec![
            model_entry("gpt-4o", &mock_a.base_url(), "key-a"),
            model_entry("gpt-4o", &mock_b.base_url(), "key-b"),
        ],
        OxllmSettings {
            routing: RoutingKnobs {
                routing_strategy: Some("least_busy".to_string()),
                ..Default::default()
            },
            ..Default::default()
        },
        RouterSettings::default(),
    );
    let (handle, _shared) = start_oxllm(config).await.expect("server starts");

    // With no in-flight data recorded, the strategy falls back to shuffle.
    let mut contents = std::collections::HashSet::new();
    for _ in 0..40 {
        let resp = send_request(&handle, "gpt-4o").await;
        assert_eq!(resp.status(), 200);
        let body: Value = resp.json().await.expect("json");
        if let Some(c) = body["choices"][0]["message"]["content"].as_str() {
            contents.insert(c.to_string());
        }
    }
    assert!(
        contents.len() > 1,
        "shuffle fallback should spread requests"
    );

    let _ = handle.shutdown_tx.send(());
}

// ---------------------------------------------------------------------------
// Weighted shuffle
// ---------------------------------------------------------------------------

#[tokio::test]
async fn weighted_shuffle_distributes_proportionally() {
    let mock_heavy = MockServer::start();
    let mock_light = MockServer::start();

    let heavy_hits = setup_ok_mock(&mock_heavy, "heavy", "key-heavy");
    let _light = setup_ok_mock(&mock_light, "light", "key-light");

    let config = build_config_with_strategy(
        vec![
            model_entry_with_weight("gpt-4o", &mock_heavy.base_url(), "key-heavy", Some(9)),
            model_entry_with_weight("gpt-4o", &mock_light.base_url(), "key-light", Some(1)),
        ],
        OxllmSettings::default(),
        RouterSettings::default(),
    );
    let (handle, _shared) = start_oxllm(config).await.expect("server starts");

    let total = 200;
    for _ in 0..total {
        let resp = send_request(&handle, "gpt-4o").await;
        assert_eq!(resp.status(), 200);
    }

    let hits = heavy_hits.hits();
    let ratio = hits as f64 / total as f64;
    assert!(
        (ratio - 0.9).abs() < 0.12,
        "expected ~0.9 ratio for weight=9 vs weight=1, got {ratio:.3} ({hits}/{total})"
    );

    let _ = handle.shutdown_tx.send(());
}

// ---------------------------------------------------------------------------
// content_policy_fallbacks
// ---------------------------------------------------------------------------

#[tokio::test]
async fn content_policy_fallback_triggers_on_400_content_filter() {
    let mock_primary = MockServer::start();
    let mock_fallback = MockServer::start();

    let _primary_cp = setup_content_policy_mock(&mock_primary, "pk");
    let _fb_ok = setup_ok_mock(&mock_fallback, "from-policy-fallback", "fbk");

    let config = build_config_with_strategy(
        vec![
            model_entry("gpt-4o", &mock_primary.base_url(), "pk"),
            model_entry("guarded-claude", &mock_fallback.base_url(), "fbk"),
        ],
        OxllmSettings {
            routing: RoutingKnobs {
                content_policy_fallbacks: Some(json!([{"gpt-4o": ["guarded-claude"]}])),
                ..Default::default()
            },
            ..Default::default()
        },
        RouterSettings::default(),
    );
    let (handle, _shared) = start_oxllm(config).await.expect("server starts");

    let resp = send_request(&handle, "gpt-4o").await;
    assert_eq!(resp.status(), 200);
    assert_eq!(
        resp.headers()
            .get("x-oxllm-fallback-used")
            .and_then(|v| v.to_str().ok()),
        Some("guarded-claude")
    );
    let body: Value = resp.json().await.expect("json");
    assert_eq!(
        body["choices"][0]["message"]["content"],
        "from-policy-fallback"
    );

    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn content_policy_fallback_skipped_for_non_policy_400() {
    let mock_primary = MockServer::start();
    let mock_fallback = MockServer::start();

    let _primary_400 = mock_primary.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .header("authorization", "Bearer pk");
        then.status(400)
            .header("content-type", "application/json")
            .json_body(json!({
            "error": {"message": "invalid model", "type": "invalid_request_error"}
            }));
    });
    let fb = setup_ok_mock(&mock_fallback, "should-not-happen", "fbk");

    let config = build_config_with_strategy(
        vec![
            model_entry("gpt-4o", &mock_primary.base_url(), "pk"),
            model_entry("guarded-claude", &mock_fallback.base_url(), "fbk"),
        ],
        OxllmSettings {
            routing: RoutingKnobs {
                content_policy_fallbacks: Some(json!([{"gpt-4o": ["guarded-claude"]}])),
                ..Default::default()
            },
            ..Default::default()
        },
        RouterSettings::default(),
    );
    let (handle, _shared) = start_oxllm(config).await.expect("server starts");

    let resp = send_request(&handle, "gpt-4o").await;
    assert_eq!(resp.status(), 400);
    assert!(
        resp.headers().get("x-oxllm-fallback-used").is_none(),
        "non-policy 400 should not trigger content-policy fallback"
    );
    fb.assert_hits(0);

    let _ = handle.shutdown_tx.send(());
}

// ---------------------------------------------------------------------------
// retry_policy
// ---------------------------------------------------------------------------

#[tokio::test]
async fn retry_policy_retries_429_more_than_500() {
    let mock_429 = MockServer::start();
    let mock_500 = MockServer::start();

    let fail_429 = setup_fail_mock(&mock_429, 429, "key-429");
    let fail_500 = setup_fail_mock(&mock_500, 500, "key-500");

    let mut policy = HashMap::new();
    policy.insert("429".to_string(), 5);
    policy.insert("500".to_string(), 1);

    let config = build_config_with_strategy(
        vec![
            model_entry("gpt-4o", &mock_429.base_url(), "key-429"),
            model_entry("gpt-500", &mock_500.base_url(), "key-500"),
        ],
        OxllmSettings {
            routing: RoutingKnobs {
                num_retries: Some(0),
                retry_policy: Some(policy),
                ..Default::default()
            },
            ..Default::default()
        },
        RouterSettings::default(),
    );
    let (handle, _shared) = start_oxllm(config).await.expect("server starts");

    let resp_429 = send_request(&handle, "gpt-4o").await;
    assert_eq!(resp_429.status(), 429);
    // 1 initial + 5 retries = 6 hits.
    let hits_429 = fail_429.hits();
    assert!(
        hits_429 >= 5,
        "retry_policy[429]=5 should produce at least 5 retries, got {hits_429}"
    );

    let resp_500 = send_request(&handle, "gpt-500").await;
    assert_eq!(resp_500.status(), 500);
    let hits_500 = fail_500.hits();
    assert!(
        hits_500 <= 2,
        "retry_policy[500]=1 should produce at most 2 total hits (1 + 1 retry), got {hits_500}"
    );

    let _ = handle.shutdown_tx.send(());
}

// ---------------------------------------------------------------------------
// model_group_alias
// ---------------------------------------------------------------------------

#[tokio::test]
async fn model_group_alias_routes_to_aliased_models() {
    let mock_a = MockServer::start();
    let mock_b = MockServer::start();

    let _a = setup_ok_mock(&mock_a, "from-gpt-4o", "key-a");
    let _b = setup_ok_mock(&mock_b, "from-gpt-4-turbo", "key-b");

    let mut aliases = HashMap::new();
    aliases.insert(
        "gpt-4".to_string(),
        vec!["gpt-4o".to_string(), "gpt-4-turbo".to_string()],
    );

    let config = build_config_with_strategy(
        vec![
            model_entry("gpt-4o", &mock_a.base_url(), "key-a"),
            model_entry("gpt-4-turbo", &mock_b.base_url(), "key-b"),
        ],
        OxllmSettings::default(),
        RouterSettings {
            model_group_alias: Some(aliases),
            ..Default::default()
        },
    );
    let (handle, _shared) = start_oxllm(config).await.expect("server starts");

    let mut contents = std::collections::HashSet::new();
    for _ in 0..40 {
        let resp = send_request(&handle, "gpt-4").await;
        assert_eq!(resp.status(), 200);
        let body: Value = resp.json().await.expect("json");
        if let Some(c) = body["choices"][0]["message"]["content"].as_str() {
            contents.insert(c.to_string());
        }
    }
    assert!(
        contents.len() > 1,
        "alias should route to multiple aliased models, got: {contents:?}"
    );

    let _ = handle.shutdown_tx.send(());
}
