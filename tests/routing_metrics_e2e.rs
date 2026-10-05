//! Integration tests: passthrough surfaces, Prometheus metrics, routing strategies, DualCache.

mod common;

use common::{http_client, metric_value};
use httpmock::MockServer;
use oxllm_server::Cache as CacheTrait;
use oxllm_server::OxllmParams;
use oxllm_server::app::{build_router, build_shared_state};
use oxllm_server::config::types::{
    ConfigModel, GeneralSettings, OxllmSettings, ProxyConfig, RoutingKnobs,
};
use serde_json::{Value, json};

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

fn multi_model_config(base_expensive: &str, base_cheap: &str) -> ProxyConfig {
    ProxyConfig {
        model_list: vec![
            ConfigModel {
                model_name: "gpt-4o".to_string(),
                oxllm_params: OxllmParams {
                    model: "openai/gpt-4o".to_string(),
                    api_key: Some("sk-test-key".to_string()),
                    api_base: Some(base_expensive.to_string()),
                    timeout: Some(30.0),
                    ..Default::default()
                },
                ..Default::default()
            },
            ConfigModel {
                model_name: "gpt-4o".to_string(),
                oxllm_params: OxllmParams {
                    model: "openai/gpt-4o-mini".to_string(),
                    api_key: Some("sk-test-key".to_string()),
                    api_base: Some(base_cheap.to_string()),
                    timeout: Some(30.0),
                    ..Default::default()
                },
                ..Default::default()
            },
        ],
        oxllm_settings: OxllmSettings {
            routing: RoutingKnobs {
                routing_strategy: Some("lowest_cost".to_string()),
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

fn anthropic_config(mock_base_url: &str) -> ProxyConfig {
    ProxyConfig {
        model_list: vec![ConfigModel {
            model_name: "claude".to_string(),
            oxllm_params: OxllmParams {
                model: "anthropic/claude-3-5-sonnet".to_string(),
                api_key: Some("sk-test-key".to_string()),
                api_base: Some(mock_base_url.to_string()),
                timeout: Some(30.0),
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

async fn start_oxllm(config: ProxyConfig) -> anyhow::Result<OxllmHandle> {
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
    Ok(OxllmHandle { addr, shutdown_tx })
}

async fn start_oxllm_with_shared(
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

// ---------------------------------------------------------------------------
// Metrics endpoint
// ---------------------------------------------------------------------------

#[tokio::test]
#[serial_test::serial]
async fn metrics_endpoint_returns_prometheus_format() {
    oxllm_server::metrics::install();
    oxllm_server::metrics::record_request("prometheus-format-probe", "openai", "success");
    let mock = MockServer::start();
    let config = proxy_config(&mock.base_url(), OxllmSettings::default());
    let handle = start_oxllm(config).await.expect("server starts");

    let resp = http_client()
        .get(handle.url("/metrics"))
        .send()
        .await
        .expect("request succeeds");
    assert!(resp.status().is_success());
    assert!(
        resp.headers()[reqwest::header::CONTENT_TYPE]
            .to_str()
            .expect("content type")
            .starts_with("text/plain"),
        "metrics must use Prometheus text exposition"
    );
    let body = resp.text().await.expect("body");
    assert!(
        body.contains("# TYPE oxllm_requests_total counter")
            && body.contains("# HELP oxllm_requests_total "),
        "registered counters must expose both TYPE and HELP: {body}"
    );
    assert_eq!(
        metric_value(
            &body,
            "oxllm_requests_total",
            &[("model", "prometheus-format-probe")]
        ),
        Some(1.0),
        "an empty or metadata-only response must not pass"
    );

    let _ = handle.shutdown_tx.send(());
}

// ---------------------------------------------------------------------------
// Images passthrough
// ---------------------------------------------------------------------------

#[tokio::test]
async fn images_generations_passthrough() {
    let mock = MockServer::start();
    mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/images/generations")
            .header("authorization", "Bearer sk-test-key");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
            "created": 1234567890,
            "data": [{"url": "https://example.com/image.png"}]
            }));
    });

    let config = proxy_config(&mock.base_url(), OxllmSettings::default());
    let handle = start_oxllm(config).await.expect("server starts");

    let resp = http_client()
        .post(handle.url("/v1/images/generations"))
        .json(&json!({
        "model": "gpt-4o",
        "prompt": "a cat"
        }))
        .send()
        .await
        .expect("request succeeds");

    assert!(resp.status().is_success());
    let body: Value = resp.json().await.expect("json body");
    assert!(body["data"].is_array());

    let _ = handle.shutdown_tx.send(());
}

// ---------------------------------------------------------------------------
// Audio speech passthrough
// ---------------------------------------------------------------------------

#[tokio::test]
async fn audio_speech_passthrough() {
    let mock = MockServer::start();
    mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/audio/speech")
            .header("authorization", "Bearer sk-test-key");
        then.status(200)
            .header("content-type", "audio/mpeg")
            .body(b"fake-audio-bytes");
    });

    let config = proxy_config(&mock.base_url(), OxllmSettings::default());
    let handle = start_oxllm(config).await.expect("server starts");

    let resp = http_client()
        .post(handle.url("/v1/audio/speech"))
        .json(&json!({
        "model": "gpt-4o",
        "input": "hello",
        "voice": "alloy"
        }))
        .send()
        .await
        .expect("request succeeds");

    assert!(resp.status().is_success());
    let bytes = resp.bytes().await.expect("body bytes");
    assert!(!bytes.is_empty());

    let _ = handle.shutdown_tx.send(());
}

// ---------------------------------------------------------------------------
// Moderations passthrough
// ---------------------------------------------------------------------------

#[tokio::test]
async fn moderations_passthrough() {
    let mock = MockServer::start();
    mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/moderations")
            .header("authorization", "Bearer sk-test-key");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
            "id": "modr-test",
            "model": "text-moderation-latest",
            "results": [{"flagged": false}]
            }));
    });

    let config = proxy_config(&mock.base_url(), OxllmSettings::default());
    let handle = start_oxllm(config).await.expect("server starts");

    let resp = http_client()
        .post(handle.url("/v1/moderations"))
        .json(&json!({
        "model": "gpt-4o",
        "input": "hello world"
        }))
        .send()
        .await
        .expect("request succeeds");

    assert!(resp.status().is_success());
    let body: Value = resp.json().await.expect("json body");
    assert_eq!(body["results"][0]["flagged"], false);

    let _ = handle.shutdown_tx.send(());
}

// ---------------------------------------------------------------------------
// Routing strategy: lowest_cost picks the cheaper model
// ---------------------------------------------------------------------------

#[tokio::test]
async fn lowest_cost_strategy_picks_cheaper_model() {
    let mock_expensive = MockServer::start();
    let mock_cheap = MockServer::start();

    mock_expensive.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
            "id": "chatcmpl-expensive",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4o",
            "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "hi"},
            "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}
            }));
    });

    mock_cheap.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
            "id": "chatcmpl-cheap",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4o-mini",
            "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "hi"},
            "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}
            }));
    });

    let config = multi_model_config(&mock_expensive.base_url(), &mock_cheap.base_url());
    let handle = start_oxllm(config).await.expect("server starts");

    let resp = http_client()
        .post(handle.url("/v1/chat/completions"))
        .json(&json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("request succeeds");

    assert!(resp.status().is_success());
    let body: Value = resp.json().await.expect("json body");
    assert_eq!(
        body["model"], "gpt-4o-mini",
        "lowest_cost should pick the cheaper model (gpt-4o-mini)"
    );

    let _ = handle.shutdown_tx.send(());
}

// ---------------------------------------------------------------------------
// Provider passthrough: /anthropic/v1/messages forwards to Anthropic
// ---------------------------------------------------------------------------

#[tokio::test]
async fn provider_passthrough_forwards_to_anthropic() {
    let mock = MockServer::start();
    let _m = mock.mock(|when, then| {
        when.method(httpmock::Method::POST);
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
            "id": "msg-test",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "hello"}],
            "model": "claude-3-5-sonnet",
            "stop_reason": "end_turn"
            }));
    });

    let config = anthropic_config(&mock.base_url());
    let handle = start_oxllm(config).await.expect("server starts");

    let resp = http_client()
        .post(handle.url("/anthropic/v1/messages"))
        .header("content-type", "application/json")
        .body(r#"{"model":"claude","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}"#)
        .send()
        .await
        .expect("request succeeds");

    assert!(resp.status().is_success());
    let body: Value = resp.json().await.expect("json body");
    assert_eq!(body["type"], "message");
    assert_eq!(body["content"][0]["text"], "hello");

    let _ = handle.shutdown_tx.send(());
}

// ---------------------------------------------------------------------------
// Metrics recorded after a chat request
// ---------------------------------------------------------------------------

#[tokio::test]
#[serial_test::serial]
async fn metrics_recorded_after_chat_request() {
    oxllm_server::metrics::install();
    let model = "metrics-chat-probe";
    let mock = MockServer::start();
    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
            "id": "chatcmpl-metrics-test",
            "object": "chat.completion",
            "created": 1234567890,
            "model": model,
            "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "hi"},
            "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
            }));
    });

    let mut config = proxy_config(&mock.base_url(), OxllmSettings::default());
    config.model_list[0].model_name = model.to_string();
    config.oxllm_settings.cache =
        Some(serde_json::from_value(json!({"type": "memory"})).expect("memory cache settings"));
    let handle = start_oxllm(config).await.expect("server starts");
    let client = http_client();
    let baseline = oxllm_server::metrics::render();
    let request_labels = [
        ("model", model),
        ("provider", "openai"),
        ("status", "success"),
    ];
    let requests_before =
        metric_value(&baseline, "oxllm_requests_total", &request_labels).unwrap_or(0.0);
    for completed in 1..=2 {
        let resp = client
            .post(handle.url("/v1/chat/completions"))
            .json(&json!({
                "model": model,
                "messages": [{"role": "user", "content": "hi"}],
                "cache": {"use_cache": true, "insert": true}
            }))
            .send()
            .await
            .expect("chat request succeeds");
        assert_eq!(
            resp.status(),
            200,
            "chat must succeed before checking its metrics"
        );
        if completed == 2 {
            assert_eq!(resp.headers()["x-oxllm-cache-hit"], "true");
        }
        let body: Value = resp.json().await.expect("chat body");
        assert_eq!(body["id"], "chatcmpl-metrics-test");

        let metrics_body = client
            .get(handle.url("/metrics"))
            .send()
            .await
            .expect("metrics request succeeds")
            .error_for_status()
            .expect("metrics status")
            .text()
            .await
            .expect("metrics body");
        // The cached body carries no provider prefix, so its provider is
        // intentionally unknown. Sum only this unique model's request series.
        let requests: f64 = metrics_body
            .lines()
            .filter(|line| line.starts_with("oxllm_requests_total{"))
            .filter(|line| line.contains(&format!("model=\"{model}\"")))
            .map(|line| {
                line.split_whitespace()
                    .last()
                    .unwrap()
                    .parse::<f64>()
                    .unwrap()
            })
            .sum();
        assert_eq!(requests - requests_before, f64::from(completed));
        assert_eq!(
            metric_value(
                &metrics_body,
                "oxllm_tokens_total",
                &[("model", model), ("type", "prompt")]
            ),
            Some(10.0)
        );
        assert_eq!(
            metric_value(
                &metrics_body,
                "oxllm_tokens_total",
                &[("model", model), ("type", "completion")]
            ),
            Some(5.0)
        );
        assert_eq!(
            metric_value(
                &metrics_body,
                "oxllm_router_decisions_total",
                &[("model", model)]
            ),
            Some(1.0)
        );
        if completed == 2 {
            assert_eq!(
                metric_value(
                    &metrics_body,
                    "oxllm_prompt_cache_hits_total",
                    &[("model", model)]
                ),
                Some(1.0)
            );
            for name in [
                "oxllm_router_decisions_total",
                "oxllm_prompt_cache_hits_total",
            ] {
                assert!(metrics_body.contains(&format!("# HELP {name} ")));
                assert!(metrics_body.contains(&format!("# TYPE {name} counter")));
            }
        }
    }
    upstream.assert_hits(1);

    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn lowest_latency_strategy_prefers_fast_provider() {
    let mock_fast = MockServer::start();
    let mock_slow = MockServer::start();

    mock_fast.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
            "id": "chatcmpl-fast",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4o-fast",
            "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "hi"},
            "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}
            }));
    });

    mock_slow.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions");
        then.status(200)
            .header("content-type", "application/json")
            .delay(std::time::Duration::from_millis(500))
            .json_body(json!({
            "id": "chatcmpl-slow",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4o-slow",
            "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "hi"},
            "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}
            }));
    });

    let config = ProxyConfig {
        model_list: vec![
            ConfigModel {
                model_name: "gpt-4o".to_string(),
                oxllm_params: OxllmParams {
                    model: "openai/gpt-4o-fast".to_string(),
                    api_key: Some("sk-test-key".to_string()),
                    api_base: Some(mock_fast.base_url()),
                    timeout: Some(30.0),
                    ..Default::default()
                },
                ..Default::default()
            },
            ConfigModel {
                model_name: "gpt-4o".to_string(),
                oxllm_params: OxllmParams {
                    model: "openai/gpt-4o-slow".to_string(),
                    api_key: Some("sk-test-key".to_string()),
                    api_base: Some(mock_slow.base_url()),
                    timeout: Some(30.0),
                    ..Default::default()
                },
                ..Default::default()
            },
        ],
        oxllm_settings: OxllmSettings {
            routing: RoutingKnobs {
                routing_strategy: Some("lowest_latency".to_string()),
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
    };

    let handle = start_oxllm(config).await.expect("server starts");
    let client = http_client();

    let body = json!({
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "hi"}]
    });

    for _ in 0..2 {
        let _ = client
            .post(handle.url("/v1/chat/completions"))
            .json(&body)
            .send()
            .await
            .expect("warmup request succeeds");
    }

    let mut fast_count = 0;
    for _ in 0..5 {
        let resp = client
            .post(handle.url("/v1/chat/completions"))
            .json(&body)
            .send()
            .await
            .expect("request succeeds");
        let result: Value = resp.json().await.expect("json body");
        if result["model"] == "gpt-4o-fast" {
            fast_count += 1;
        }
    }

    let _ = handle.shutdown_tx.send(());

    assert_eq!(
        fast_count, 5,
        "after warmup, all requests should prefer the fast provider"
    );
}

#[tokio::test]
async fn latency_recorded_in_router_store() {
    let mock = MockServer::start();
    mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions");
        then.status(200)
            .header("content-type", "application/json")
            .delay(std::time::Duration::from_millis(100))
            .json_body(json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4o",
            "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "hi"},
            "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}
            }));
    });

    let config = proxy_config(&mock.base_url(), OxllmSettings::default());
    let (handle, shared) = start_oxllm_with_shared(config.clone())
        .await
        .expect("server starts");
    // The latency store is keyed by the stable per-deployment key.
    let dep: oxllm_core::Deployment = config.model_list[0].clone().into();
    let store_key = oxllm_core::router::stable_key(&dep);

    {
        let state = shared.load_full();
        assert!(
            state.router.latency_store().p50(&store_key).is_none(),
            "latency store should be empty before any request"
        );
    }

    let _ = http_client()
        .post(handle.url("/v1/chat/completions"))
        .json(&json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}]
        }))
        .send()
        .await
        .expect("request succeeds");

    let state = shared.load_full();
    let store = state.router.latency_store();
    assert!(
        store.p50(&store_key).is_some(),
        "latency store should have samples after a request"
    );

    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn dual_cache_constructed_when_redis_config_present() {
    let config_yaml = "\
general_settings:
  allow_private_api_base: true
model_list:
  - model_name: gpt-4o
    oxllm_params:
      model: openai/gpt-4o
      api_key: sk-test
oxllm_settings:
  cache:
    type: redis
    host: 127.0.0.1
    port: 1
";
    let config = oxllm_server::parse_config_str(config_yaml).expect("config parses");

    assert!(config.oxllm_settings.cache.is_some());
    let cache_cfg = config.oxllm_settings.cache.as_ref().unwrap();
    assert_eq!(cache_cfg.cache_type.as_deref(), Some("redis"));
    assert_eq!(cache_cfg.host.as_deref(), Some("127.0.0.1"));
    assert_eq!(cache_cfg.port, Some(1));

    let memory = std::sync::Arc::new(oxllm_server::InMemoryCache::default());
    let cache = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        oxllm_server::app::build_cache_for_test(&config, memory.clone()),
    )
    .await
    .expect("build_cache should not hang");

    cache.set("test_key", "test_val".to_string(), None).await;
    let val = CacheTrait::get(&*memory, "test_key").await;
    assert_eq!(
        val.as_deref(),
        Some("test_val"),
        "in-memory layer should have the value"
    );

    let _ = cache.incr("test_counter", 1.0, None).await.unwrap();
    let counter = memory.counter_value("test_counter").await;
    assert_eq!(
        counter,
        Some(1.0),
        "in-memory layer should have the incremented counter"
    );
}

#[tokio::test]
async fn cache_config_defaults_to_memory() {
    let config_yaml = "\
model_list:
  - model_name: gpt-4o
    oxllm_params:
      model: openai/gpt-4o
      api_key: sk-test
oxllm_settings:
  cache:
    type: memory
";
    let config = oxllm_server::parse_config_str(config_yaml).expect("config parses");

    let cache_cfg = config.oxllm_settings.cache.as_ref().expect("cache config");
    assert_eq!(cache_cfg.cache_type.as_deref(), Some("memory"));

    let memory = std::sync::Arc::new(oxllm_server::InMemoryCache::default());
    let cache = oxllm_server::app::build_cache_for_test(&config, memory.clone()).await;

    cache.set("k", "v".to_string(), None).await;
    assert_eq!(cache.get("k").await.as_deref(), Some("v"));
}

// ---------------------------------------------------------------------------
// P0-1: passthrough requires auth when master_key is set
// ---------------------------------------------------------------------------

#[tokio::test]
async fn passthrough_rejects_unauthenticated_when_master_key_set() {
    let mock = MockServer::start();
    mock.mock(|when, then| {
        when.method(httpmock::Method::POST).path("/v1/messages");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({"id": "x", "type": "message", "role": "assistant"}));
    });

    let config = ProxyConfig {
        model_list: vec![ConfigModel {
            model_name: "claude".to_string(),
            oxllm_params: OxllmParams {
                model: "anthropic/claude-3-5-sonnet".to_string(),
                api_key: Some("sk-test-key".to_string()),
                api_base: Some(mock.base_url()),
                timeout: Some(30.0),
                ..Default::default()
            },
            ..Default::default()
        }],
        oxllm_settings: OxllmSettings::default(),
        general_settings: GeneralSettings {
            master_key: Some("sk-master-key-test-1".to_string()),
            allow_private_api_base: Some(true),
            ..Default::default()
        },
        router_settings: Default::default(),
        extra: serde_json::Map::new(),
    };
    let handle = start_oxllm(config).await.expect("server starts");

    let resp = http_client()
        .post(handle.url("/anthropic/v1/messages"))
        .header("content-type", "application/json")
        .body(r#"{"model":"claude","max_tokens":1,"messages":[]}"#)
        .send()
        .await
        .expect("request succeeds");

    assert_eq!(
        resp.status(),
        401,
        "passthrough without auth should be rejected when master_key is set"
    );

    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn passthrough_succeeds_with_master_key() {
    let mock = MockServer::start();
    let _mock_mock = mock.mock(|when, then| {
        when.method(httpmock::Method::POST);
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({"id": "x", "type": "message", "role": "assistant"}));
    });

    let config = ProxyConfig {
        model_list: vec![ConfigModel {
            model_name: "claude".to_string(),
            oxllm_params: OxllmParams {
                model: "anthropic/claude-3-5-sonnet".to_string(),
                api_key: Some("sk-test-key".to_string()),
                api_base: Some(mock.base_url()),
                timeout: Some(30.0),
                ..Default::default()
            },
            ..Default::default()
        }],
        oxllm_settings: OxllmSettings::default(),
        general_settings: GeneralSettings {
            master_key: Some("sk-master-key-test-1".to_string()),
            allow_private_api_base: Some(true),
            ..Default::default()
        },
        router_settings: Default::default(),
        extra: serde_json::Map::new(),
    };
    let handle = start_oxllm(config).await.expect("server starts");

    let resp = http_client()
        .post(handle.url("/anthropic/v1/messages"))
        .header("authorization", "Bearer sk-master-key-test-1")
        .header("content-type", "application/json")
        .body(r#"{"model":"claude","max_tokens":1,"messages":[]}"#)
        .send()
        .await
        .expect("request succeeds");

    assert!(
        resp.status().is_success(),
        "passthrough with master key should succeed"
    );

    let _ = handle.shutdown_tx.send(());
}

// ---------------------------------------------------------------------------
// P0-5: master key bypasses budget/rate limits
// ---------------------------------------------------------------------------

#[tokio::test]
async fn master_key_bypasses_global_max_budget() {
    let mock = MockServer::start();
    mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .header("authorization", "Bearer sk-test-key");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
            "id": "x",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-4o",
            "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "hi"},
            "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 1000, "total_tokens": 2000}
            }));
    });

    let config = ProxyConfig {
        model_list: vec![ConfigModel {
            model_name: "gpt-4o".to_string(),
            oxllm_params: OxllmParams {
                model: "openai/gpt-4o".to_string(),
                api_key: Some("sk-test-key".to_string()),
                api_base: Some(mock.base_url()),
                timeout: Some(30.0),
                ..Default::default()
            },
            ..Default::default()
        }],
        oxllm_settings: OxllmSettings {
            max_budget: Some(0.01),
            ..Default::default()
        },
        general_settings: GeneralSettings {
            master_key: Some("sk-master-key-test-1".to_string()),
            allow_private_api_base: Some(true),
            ..Default::default()
        },
        router_settings: Default::default(),
        extra: serde_json::Map::new(),
    };
    let handle = start_oxllm(config).await.expect("server starts");

    let client = http_client();
    for _ in 0..5 {
        let resp = client
            .post(handle.url("/v1/chat/completions"))
            .header("authorization", "Bearer sk-master-key-test-1")
            .json(&json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}]
            }))
            .send()
            .await
            .expect("request succeeds");
        assert!(
            resp.status().is_success(),
            "master key should bypass budget across repeated requests"
        );
    }

    let _ = handle.shutdown_tx.send(());
}
