//! Integration tests: cost tracking, rate limits, response cache, and budgets.

mod common;

use httpmock::{Mock, MockServer};
use oxllm_server::Cache;
use oxllm_server::OxllmParams;
use oxllm_server::app::{build_router, build_shared_state};
use oxllm_server::config::types::{ConfigModel, GeneralSettings, OxllmSettings, ProxyConfig};
use serde_json::{Value, json};
use std::sync::Arc;
use std::time::Duration;

struct OxllmHandle {
    addr: std::net::SocketAddr,
    shutdown_tx: tokio::sync::oneshot::Sender<()>,
}

impl OxllmHandle {
    fn url(&self, path: &str) -> String {
        format!("http://{}{path}", self.addr)
    }
}

fn proxy_config(mock_base_url: &str, settings: OxllmSettings) -> ProxyConfig {
    ProxyConfig {
        model_list: vec![ConfigModel {
            model_name: "gpt-4o".to_string(),
            oxllm_params: OxllmParams {
                model: "openai/gpt-4o".to_string(),
                api_key: Some("sk-test-key".to_string()),
                api_base: Some(mock_base_url.to_string()),
                timeout: Some(30.0),
                stream_timeout: Some(30),
                ..Default::default()
            },
            ..Default::default()
        }],
        oxllm_settings: settings,
        general_settings: GeneralSettings {
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

fn mock_chat_response(mock: &MockServer) -> Mock<'_> {
    mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .header("authorization", "Bearer sk-test-key");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
            "id": "chatcmpl-cost-test",
            "object": "chat.completion",
            "created": 1234567890,
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

fn chat_request(model: &str) -> Value {
    json!({
    "model": model,
    "messages": [{"role": "user", "content": "hi"}]
    })
}

#[tokio::test]
async fn response_has_cost_header() {
    let mock = MockServer::start();
    let config = proxy_config(&mock.base_url(), OxllmSettings::default());
    let handle = start_oxllm(config).await.expect("server starts");
    let mock_mock = mock_chat_response(&mock);

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .json(&chat_request("gpt-4o"))
        .send()
        .await
        .expect("ok");

    assert_eq!(resp.status(), 200);
    let cost_header = resp
        .headers()
        .get("x-oxllm-response-cost")
        .expect("cost header should be present");
    let cost_str = cost_header.to_str().expect("cost header parses");
    let cost: f64 = cost_str.parse().expect("cost parses as float");
    let expected = 10.0 * 0.0000025 + 5.0 * 0.00001;
    assert!(
        (cost - expected).abs() < 1e-9,
        "got {cost}, expected {expected}"
    );

    mock_mock.assert_hits(1);
    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn response_has_request_id_header() {
    let mock = MockServer::start();
    let config = proxy_config(&mock.base_url(), OxllmSettings::default());
    let handle = start_oxllm(config).await.expect("server starts");
    let _m = mock_chat_response(&mock);

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .json(&chat_request("gpt-4o"))
        .send()
        .await
        .expect("ok");

    assert_eq!(resp.status(), 200);
    assert!(
        resp.headers().contains_key("x-oxllm-request-id"),
        "request id header should be set"
    );
    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn rpm_limit_rejects_after_threshold() {
    let mock = MockServer::start();
    let settings = OxllmSettings {
        rpm: Some(2),
        ..Default::default()
    };
    let config = proxy_config(&mock.base_url(), settings);
    let handle = start_oxllm(config).await.expect("server starts");
    let mock_mock = mock_chat_response(&mock);

    let client = reqwest::Client::new();

    for _ in 0..2 {
        let resp = client
            .post(handle.url("/v1/chat/completions"))
            .json(&chat_request("gpt-4o"))
            .send()
            .await
            .expect("ok");
        assert_eq!(resp.status(), 200, "first 2 requests should succeed");
    }

    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .json(&chat_request("gpt-4o"))
        .send()
        .await
        .expect("ok");
    assert_eq!(
        resp.status(),
        429,
        "third request should be rate-limited (rpm=2)"
    );
    let body: Value = resp.json().await.expect("body");
    assert_eq!(body["error"]["type"], "rate_limit_error");

    mock_mock.assert_hits(2);
    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn budget_exceeded_rejects_after_threshold() {
    let mock = MockServer::start();
    let settings = OxllmSettings {
        max_budget: Some(0.0001),
        budget_duration: Some("30d".to_string()),
        ..Default::default()
    };
    let config = proxy_config(&mock.base_url(), settings);
    let handle = start_oxllm(config).await.expect("server starts");
    let mock_mock = mock_chat_response(&mock);

    let client = reqwest::Client::new();
    let mut request = chat_request("gpt-4o");
    request["max_tokens"] = json!(5);

    let mut successes = 0;
    let mut rejected: Option<reqwest::Response> = None;
    for _ in 0..10 {
        let resp = client
            .post(handle.url("/v1/chat/completions"))
            .json(&request)
            .send()
            .await
            .expect("ok");
        if resp.status() == 200 {
            successes += 1;
        } else if resp.status() == 429 {
            rejected = Some(resp);
            break;
        }
    }
    assert!(successes > 0, "at least one request should succeed");
    let rejected = rejected.expect("should eventually be budget-rejected");

    // The budget rejection must be distinguishable from a plain RPM 429
    // (`proxy_rate_limited`) and must NOT synthesize the 60s Retry-After:
    // budget windows reset on daily/monthly horizons, so a 60s header would
    // mislead clients into hot-retrying a cap that won't lift for a long time.
    assert!(
        rejected.headers().get("retry-after").is_none(),
        "budget_exceeded must not synthesize a Retry-After header"
    );
    let body: Value = rejected.json().await.expect("body");
    assert_eq!(
        body["error"]["code"], "budget_exceeded",
        "budget rejection must carry the budget_exceeded code, got {body}"
    );

    mock_mock.assert_hits(successes);
    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn cache_hit_returns_cached_response() {
    let mock = MockServer::start();
    let settings = OxllmSettings {
        cache_ttl_secs: Some(300),
        ..Default::default()
    };
    let config = proxy_config(&mock.base_url(), settings);
    let handle = start_oxllm(config).await.expect("server starts");
    let mock_mock = mock_chat_response(&mock);

    let client = reqwest::Client::new();
    let cached_body = json!({
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "say cached"}],
    "cache": {"insert": true}
    });

    let resp1 = client
        .post(handle.url("/v1/chat/completions"))
        .json(&cached_body)
        .send()
        .await
        .expect("ok");
    assert_eq!(resp1.status(), 200);
    assert!(
        !resp1.headers().contains_key("x-oxllm-cache-hit"),
        "first response should not be a cache hit"
    );

    let resp2 = client
        .post(handle.url("/v1/chat/completions"))
        .json(&cached_body)
        .send()
        .await
        .expect("ok");
    assert_eq!(resp2.status(), 200, "second response should succeed");
    assert!(
        resp2.headers().contains_key("x-oxllm-cache-hit"),
        "second response should be a cache hit"
    );

    mock_mock.assert_hits(1);
    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn no_cache_when_not_opted_in() {
    let mock = MockServer::start();
    let settings = OxllmSettings {
        cache_ttl_secs: Some(300),
        ..Default::default()
    };
    let config = proxy_config(&mock.base_url(), settings);
    let handle = start_oxllm(config).await.expect("server starts");
    let mock_mock = mock_chat_response(&mock);

    let client = reqwest::Client::new();
    let body = json!({
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "no cache"}]
    });

    for _ in 0..2 {
        let resp = client
            .post(handle.url("/v1/chat/completions"))
            .json(&body)
            .send()
            .await
            .expect("ok");
        assert_eq!(resp.status(), 200);
        assert!(
            !resp.headers().contains_key("x-oxllm-cache-hit"),
            "no cache hit without opt-in"
        );
    }

    mock_mock.assert_hits(2);
    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn unknown_model_has_no_cost_header() {
    let mock = MockServer::start();
    let mut config = proxy_config(&mock.base_url(), OxllmSettings::default());
    config.model_list[0].model_name = "unpriced-model".into();
    config.model_list[0].oxllm_params.model = "openai/unpriced-model".into();
    let handle = start_oxllm(config).await.expect("server starts");

    let _unknown_mock = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
            "id": "x",
            "object": "chat.completion",
            "created": 1,
            "model": "made-up",
            "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "hi"},
            "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
            }));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .json(&chat_request("unpriced-model"))
        .send()
        .await
        .expect("ok");

    assert_eq!(resp.status(), 200);
    assert!(
        !resp.headers().contains_key("x-oxllm-response-cost"),
        "no cost header for unknown model"
    );
    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn unpriced_reported_alias_keeps_the_known_serving_cost() {
    let mock = MockServer::start();
    let handle = start_oxllm(proxy_config(&mock.base_url(), OxllmSettings::default()))
        .await
        .unwrap();
    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST).path("/chat/completions")
            .json_body_partial(json!({"model":"gpt-4o"}).to_string());
        then.status(200).json_body(json!({"id":"alias","model":"unpriced-reported-alias",
            "choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],
            "usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}));
    });
    let response = reqwest::Client::new()
        .post(handle.url("/v1/chat/completions"))
        .json(&chat_request("gpt-4o"))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let cost: f64 = response.headers()["x-oxllm-response-cost"]
        .to_str()
        .unwrap()
        .parse()
        .unwrap();
    assert!((cost - 0.000075).abs() < 1e-12);
    assert_eq!(response.json::<Value>().await.unwrap()["model"], "gpt-4o");
    upstream.assert_hits(1);
    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn scheduler_tick_resets_expired_counters() {
    use oxllm_server::InMemoryCache;
    use oxllm_server::hooks::BudgetHook;
    use oxllm_server::scheduler::BudgetResetScheduler;

    let cache = Arc::new(InMemoryCache::default());
    let _ = cache.incr("spend:abc", 5.0, None).await;
    assert_eq!(cache.counter_value("spend:abc").await, Some(5.0));

    let hook = Arc::new(BudgetHook::with_duration(
        cache.clone(),
        Some(10.0),
        Duration::from_millis(1),
    ));
    // The scheduler re-reads the GLOBAL `budget_duration` from the live shared
    // config on every tick (NOT the per-hook duration). The shared config must
    // therefore set a short window, otherwise `tick_once` falls back to the
    // canonical 30-day default and the counter is untouched (a counter created
    // moments ago is nowhere near 30 days old).
    let settings = OxllmSettings {
        budget_duration: Some("1ms".to_string()),
        ..Default::default()
    };
    let scheduler = BudgetResetScheduler::new(
        cache.clone(),
        hook,
        oxllm_server::app::build_shared_state(proxy_config("http://127.0.0.1:0", settings))
            .expect("state"),
    );

    tokio::time::sleep(Duration::from_millis(20)).await;
    scheduler.tick_once().await;
    // Post-M9, `reset_spend_counter_if_due` REMOVES the TTL-less counter entry
    // under one write lock (matching the Redis `delete` path). The counter is
    // semantically zero for all downstream purposes (every caller reads
    // `counter_value(...).unwrap_or(0.0)`), but the explicit `None` is the
    // current contract — mirroring `scheduler.rs::tests`.
    assert_eq!(
        cache.counter_value("spend:abc").await,
        None,
        "spend counter must be removed after the global window elapsed"
    );
}

#[tokio::test]
async fn scheduler_does_not_reset_within_window() {
    use oxllm_server::InMemoryCache;
    use oxllm_server::hooks::BudgetHook;
    use oxllm_server::scheduler::BudgetResetScheduler;

    let cache = Arc::new(InMemoryCache::default());
    let _ = cache.incr("spend:abc", 5.0, None).await;
    assert_eq!(cache.counter_value("spend:abc").await, Some(5.0));

    let hook = Arc::new(BudgetHook::with_duration(
        cache.clone(),
        Some(10.0),
        Duration::from_secs(3600),
    ));
    let scheduler = BudgetResetScheduler::new(
        cache.clone(),
        hook,
        oxllm_server::app::build_shared_state(crate::common::proxy_config("http://127.0.0.1:0"))
            .expect("state"),
    );

    scheduler.tick_once().await;
    assert_eq!(
        cache.counter_value("spend:abc").await,
        Some(5.0),
        "spend should NOT be reset while window is active"
    );
}

/// Build a proxy config with custom per-deployment pricing overrides.
fn proxy_config_with_pricing(
    mock_base_url: &str,
    input_cost_per_token: f64,
    output_cost_per_token: f64,
) -> ProxyConfig {
    ProxyConfig {
        model_list: vec![ConfigModel {
            model_name: "gpt-4o".to_string(),
            oxllm_params: OxllmParams {
                model: "openai/gpt-4o".to_string(),
                api_key: Some("sk-test-key".to_string()),
                api_base: Some(mock_base_url.to_string()),
                timeout: Some(30.0),
                stream_timeout: Some(30),
                input_cost_per_token: Some(input_cost_per_token),
                output_cost_per_token: Some(output_cost_per_token),
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

/// Custom pricing override on a deployment wins over the embedded price
/// table. The response cost header must reflect the override rates.
#[tokio::test]
async fn custom_pricing_override_used_for_cost_header() {
    let mock = MockServer::start();
    // Use rates that are clearly different from gpt-4o's table rates
    // (input=2.5e-6, output=1e-5).
    let config = proxy_config_with_pricing(&mock.base_url(), 0.001, 0.002);
    let handle = start_oxllm(config).await.expect("server starts");
    let mock_mock = mock_chat_response(&mock);

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .json(&chat_request("gpt-4o"))
        .send()
        .await
        .expect("ok");

    assert_eq!(resp.status(), 200);
    let cost_str = resp
        .headers()
        .get("x-oxllm-response-cost")
        .expect("cost header present")
        .to_str()
        .expect("parses");
    let cost: f64 = cost_str.parse().expect("float");
    // 10 prompt * 0.001 + 5 completion * 0.002 = 0.02
    let expected = 10.0 * 0.001 + 5.0 * 0.002;
    assert!(
        (cost - expected).abs() < 1e-9,
        "override should drive cost: got {cost}, expected {expected}"
    );
    let table_cost = 10.0 * 0.0000025 + 5.0 * 0.00001;
    assert!(
        (cost - table_cost).abs() > 1e-9,
        "override cost must differ from price table ({table_cost})"
    );

    mock_mock.assert_hits(1);
    let _ = handle.shutdown_tx.send(());
}

/// Key with RPM=10, after 3 successful requests, the response should report
/// `x-ratelimit-remaining-requests` decrementing. The 4th request sees
/// remaining=6 (10 - 4 in-flight including itself).
#[tokio::test]
async fn ratelimit_remaining_requests_header_decrements() {
    let mock = MockServer::start();
    let settings = OxllmSettings {
        rpm: Some(10),
        ..Default::default()
    };
    let config = proxy_config(&mock.base_url(), settings);
    let handle = start_oxllm(config).await.expect("server starts");
    let mock_mock = mock_chat_response(&mock);

    let client = reqwest::Client::new();

    // Issue 3 prior requests to fill the counter.
    for _ in 0..3 {
        let resp = client
            .post(handle.url("/v1/chat/completions"))
            .json(&chat_request("gpt-4o"))
            .send()
            .await
            .expect("ok");
        assert_eq!(resp.status(), 200);
    }

    // 4th request: counter was 3 before, becomes 4 after, so remaining = 6.
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .json(&chat_request("gpt-4o"))
        .send()
        .await
        .expect("ok");
    assert_eq!(resp.status(), 200);
    let remaining = resp
        .headers()
        .get("x-ratelimit-remaining-requests")
        .expect("x-ratelimit-remaining-requests header should be set")
        .to_str()
        .expect("parses");
    assert_eq!(remaining, "6", "after 4 of 10 RPM used, 6 should remain");
    let reset = resp
        .headers()
        .get("x-ratelimit-reset-requests")
        .expect("x-ratelimit-reset-requests header should be set")
        .to_str()
        .expect("parses");
    assert_eq!(reset, "60s");

    mock_mock.assert_hits(4);
    let _ = handle.shutdown_tx.send(());
}

/// Master key bypasses rate limits → response must NOT carry any
/// `x-ratelimit-*` headers.
#[tokio::test]
async fn ratelimit_headers_absent_for_master_key() {
    let mock = MockServer::start();
    let settings = OxllmSettings {
        rpm: Some(10),
        tpm: Some(1000),
        ..Default::default()
    };
    let mut config = proxy_config(&mock.base_url(), settings);
    config.general_settings.master_key = Some("sk-master-key-test-1".to_string());
    let handle = start_oxllm(config).await.expect("server starts");
    let mock_mock = mock_chat_response(&mock);

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .header("authorization", "Bearer sk-master-key-test-1")
        .json(&chat_request("gpt-4o"))
        .send()
        .await
        .expect("ok");

    assert_eq!(resp.status(), 200);
    assert!(
        resp.headers()
            .get("x-ratelimit-remaining-requests")
            .is_none(),
        "master key must not get rate-limit headers"
    );
    assert!(
        resp.headers().get("x-ratelimit-remaining-tokens").is_none(),
        "master key must not get rate-limit headers"
    );
    assert!(
        resp.headers().get("x-ratelimit-reset-requests").is_none(),
        "master key must not get rate-limit headers"
    );
    assert!(
        resp.headers().get("x-ratelimit-reset-tokens").is_none(),
        "master key must not get rate-limit headers"
    );

    mock_mock.assert_hits(1);
    let _ = handle.shutdown_tx.send(());
}
