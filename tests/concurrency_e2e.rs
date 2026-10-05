//! Integration tests: concurrency / race conditions.
//!
//! Fires multiple concurrent requests and verifies correctness under
//! parallel load: RPM enforcement, budget enforcement, in-flight counter
//! accuracy, and cache-write safety.

mod common;

use common::{proxy_config, start_oxllm};
use httpmock::{Mock, MockServer};
use oxllm_server::OxllmParams;
use oxllm_server::config::types::{
    ConfigModel, GeneralSettings, OxllmSettings, ProxyConfig, RoutingKnobs,
};
use serde_json::{Value, json};
use std::time::Duration;

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn mock_chat_response<'a>(mock: &'a MockServer, key: &str) -> Mock<'a> {
    mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .header("authorization", format!("Bearer {key}"));
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
            "id": "chatcmpl-conc",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-4o",
            "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
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

/// Build a config with a global RPM limit and optional max_tokens in the
/// request body.
fn config_with_rpm(mock_base_url: &str, rpm: u32) -> ProxyConfig {
    let mut cfg = proxy_config(mock_base_url);
    cfg.oxllm_settings.rpm = Some(rpm);
    cfg
}

/// Build a config with a global max_budget.
fn config_with_budget(mock_base_url: &str, max_budget: f64) -> ProxyConfig {
    let mut cfg = proxy_config(mock_base_url);
    cfg.oxllm_settings.max_budget = Some(max_budget);
    cfg.oxllm_settings.budget_duration = Some("30d".to_string());
    cfg
}

// ---------------------------------------------------------------------------
// 1. Concurrent RPM enforcement
// ---------------------------------------------------------------------------

/// A key with `rpm: 5`. Fire 20 concurrent requests. Assert exactly 5
/// succeed (200) and 15 are rejected (429).
#[tokio::test]
async fn concurrent_rpm_allows_exactly_limit_then_rejects() {
    let mock = MockServer::start();
    let config = config_with_rpm(&mock.base_url(), 5);
    let (handle, _shared) = start_oxllm(config).await.expect("server starts");
    let _m = mock_chat_response(&mock, "sk-test-key");

    let client = reqwest::Client::new();
    let url = handle.url("/v1/chat/completions");
    let body = chat_request("gpt-4o");

    let futs = (0..20u32).map(|_| {
        let client = client.clone();
        let url = url.clone();
        let body = body.clone();
        async move {
            client
                .post(&url)
                .json(&body)
                .send()
                .await
                .expect("send")
                .status()
        }
    });
    let statuses: Vec<reqwest::StatusCode> = futures_util::future::join_all(futs).await;

    let ok = statuses.iter().filter(|s| s.is_success()).count();
    let too_many = statuses
        .iter()
        .filter(|s| **s == reqwest::StatusCode::TOO_MANY_REQUESTS)
        .count();

    assert_eq!(
        ok, 5,
        "exactly 5 requests should succeed under rpm=5, got {ok}"
    );
    assert_eq!(
        too_many, 15,
        "exactly 15 requests should be rejected (429), got {too_many}"
    );

    handle.shutdown();
}

// ---------------------------------------------------------------------------
// 2. Concurrent budget enforcement
// ---------------------------------------------------------------------------

/// A key with `max_budget: 0.10`. Fire 20 concurrent requests (each costing
/// ~$0.01). Assert total spend never exceeds the budget significantly —
/// the atomic reservation via `incr_if_below` bounds overshoot.
#[tokio::test]
async fn concurrent_budget_bounded_overshoot() {
    let mock = MockServer::start();
    let config = config_with_budget(&mock.base_url(), 0.10);
    let (handle, _shared) = start_oxllm(config).await.expect("server starts");
    let _m = mock_chat_response(&mock, "sk-test-key");

    let client = reqwest::Client::new();
    let url = handle.url("/v1/chat/completions");

    // Each request specifies max_tokens=1000 → the budget hook reserves
    // ~$0.01 per request (1000 output tokens × $10/M). With a $0.10 budget
    // roughly 10 should be admitted.
    let mut body = chat_request("gpt-4o");
    body["max_tokens"] = json!(1000);

    let futs = (0..20u32).map(|_| {
        let client = client.clone();
        let url = url.clone();
        let body = body.clone();
        async move {
            client
                .post(&url)
                .json(&body)
                .send()
                .await
                .expect("send")
                .status()
        }
    });
    let statuses: Vec<reqwest::StatusCode> = futures_util::future::join_all(futs).await;

    let ok = statuses.iter().filter(|s| s.is_success()).count();
    let rejected = statuses
        .iter()
        .filter(|s| **s == reqwest::StatusCode::TOO_MANY_REQUESTS)
        .count();

    assert!(ok > 0, "at least some requests should succeed");
    assert!(
        rejected > 0,
        "budget should reject some concurrent requests, got {rejected} rejections"
    );
    // The atomic reservation bounds overshoot: at ~$0.01/req and a $0.10
    // budget, no more than ~12 should succeed (allowing small estimation
    // slack). 20 succeeding would indicate a broken reservation.
    assert!(
        ok <= 15,
        "concurrent budget should not admit {ok}/20 requests against a $0.10 budget at ~$0.01/req"
    );

    handle.shutdown();
}

// ---------------------------------------------------------------------------
// 3. Concurrent in-flight counter accuracy (least_busy)
// ---------------------------------------------------------------------------

fn least_busy_config(mock_a: &str, mock_b: &str) -> ProxyConfig {
    ProxyConfig {
        model_list: vec![
            ConfigModel {
                model_name: "gpt-4o".to_string(),
                oxllm_params: OxllmParams {
                    model: "openai/gpt-4o-a".to_string(),
                    api_key: Some("sk-a".to_string()),
                    api_base: Some(mock_a.to_string()),
                    timeout: Some(30.0),
                    stream_timeout: Some(30),
                    ..Default::default()
                },
                ..Default::default()
            },
            ConfigModel {
                model_name: "gpt-4o".to_string(),
                oxllm_params: OxllmParams {
                    model: "openai/gpt-4o-b".to_string(),
                    api_key: Some("sk-b".to_string()),
                    api_base: Some(mock_b.to_string()),
                    timeout: Some(30.0),
                    stream_timeout: Some(30),
                    ..Default::default()
                },
                ..Default::default()
            },
        ],
        oxllm_settings: OxllmSettings {
            routing: RoutingKnobs {
                routing_strategy: Some("least_busy".to_string()),
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

/// Configure `least_busy` with 2 deployments. Fire 10 concurrent requests
/// (with an upstream delay so they overlap). Assert both deployments
/// receive traffic — the in-flight counter steers requests away from the
/// busy deployment.
#[tokio::test]
async fn concurrent_least_busy_distributes_across_deployments() {
    let mock_a = MockServer::start();
    let mock_b = MockServer::start();

    // Add a delay so requests are genuinely in-flight concurrently — the
    // in-flight counter is non-zero while the upstream call is pending,
    // causing `least_busy` to prefer the idle deployment.
    let mock_a_guard = mock_a.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .header("authorization", "Bearer sk-a");
        then.status(200)
            .delay(Duration::from_millis(100))
            .json_body(json!({
            "id": "a", "object": "chat.completion", "model": "gpt-4o",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "from-a"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
            }));
    });
    let mock_b_guard = mock_b.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .header("authorization", "Bearer sk-b");
        then.status(200)
            .delay(Duration::from_millis(100))
            .json_body(json!({
            "id": "b", "object": "chat.completion", "model": "gpt-4o",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "from-b"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
            }));
    });

    let config = least_busy_config(&mock_a.base_url(), &mock_b.base_url());
    let (handle, _shared) = start_oxllm(config).await.expect("server starts");

    let client = reqwest::Client::new();
    let url = handle.url("/v1/chat/completions");
    let body = chat_request("gpt-4o");

    let futs = (0..10u32).map(|_| {
        let client = client.clone();
        let url = url.clone();
        let body = body.clone();
        async move {
            let resp = client.post(&url).json(&body).send().await.expect("send");
            resp.status()
        }
    });
    let statuses: Vec<reqwest::StatusCode> = futures_util::future::join_all(futs).await;
    for s in &statuses {
        assert!(s.is_success(), "request failed with {s}");
    }

    let hits_a = mock_a_guard.hits();
    let hits_b = mock_b_guard.hits();
    assert_eq!(
        hits_a + hits_b,
        10,
        "all 10 concurrent requests should reach an upstream"
    );
    assert!(
        hits_a > 0 && hits_b > 0,
        "least_busy should distribute traffic across both deployments (a={hits_a}, b={hits_b}), not send all to one"
    );

    handle.shutdown();
}

// ---------------------------------------------------------------------------
// 4. Concurrent cache writes don't corrupt
// ---------------------------------------------------------------------------

fn config_with_cache(mock_base_url: &str, ttl_secs: u64) -> ProxyConfig {
    let mut cfg = proxy_config(mock_base_url);
    cfg.oxllm_settings.cache_ttl_secs = Some(ttl_secs);
    cfg
}

/// Fire 10 concurrent requests with the same body and `cache: {insert: true}`.
/// Assert no panic, no corruption, and the cache works on a subsequent read
/// (upstream hit exactly once).
#[tokio::test]
async fn concurrent_cache_writes_do_not_corrupt() {
    let mock = MockServer::start();
    let config = config_with_cache(&mock.base_url(), 300);
    let (handle, _shared) = start_oxllm(config).await.expect("server starts");
    let chat_hit = mock_chat_response(&mock, "sk-test-key");

    let client = reqwest::Client::new();
    let url = handle.url("/v1/chat/completions");
    let body = json!({
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "concurrent cache probe"}],
    "cache": {"insert": true}
    });

    // Fire 10 concurrent requests with the identical body.
    let futs = (0..10u32).map(|_| {
        let client = client.clone();
        let url = url.clone();
        let body = body.clone();
        async move {
            let resp = client.post(&url).json(&body).send().await.expect("send");
            resp.status()
        }
    });
    let statuses: Vec<reqwest::StatusCode> = futures_util::future::join_all(futs).await;

    for s in &statuses {
        assert!(s.is_success(), "concurrent cache request failed with {s}");
    }

    // The first concurrent request populates the cache; the rest may also
    // reach upstream if they all miss simultaneously (thundering herd).
    // Either way, no panic / corruption should occur. Verify the cache is
    // now stable by sending one more request — it MUST be served from
    // cache (upstream not hit again).
    let upstream_after_burst = chat_hit.hits();
    let resp = client.post(&url).json(&body).send().await.expect("read");
    assert!(resp.status().is_success());
    assert_eq!(
        chat_hit.hits(),
        upstream_after_burst,
        "subsequent identical request should be served from cache, not upstream"
    );

    handle.shutdown();
}
