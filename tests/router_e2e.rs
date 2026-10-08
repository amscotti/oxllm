//! Integration tests: routing, retries, fallbacks, and wildcard model matching.

mod common;

use common::{OxllmHandle, http_client, start_oxllm};
use httpmock::{Mock, MockServer};
use oxllm_server::OxllmParams;
use oxllm_server::config::types::{
    ConfigModel, GeneralSettings, OxllmSettings, ProxyConfig, RoutingGroupConfig, RoutingKnobs,
};
use oxllm_server::pre_call_checks::prompt_caching::PromptCachingCheck;
use serde_json::{Value, json};

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

fn model_entry_with_litellm_model(
    name: &str,
    litellm_model: &str,
    base: &str,
    key: &str,
) -> ConfigModel {
    let mut p = params(base, key);
    p.model = litellm_model.to_string();
    ConfigModel {
        model_name: name.to_string(),
        oxllm_params: p,
        ..Default::default()
    }
}

fn build_config(model_list: Vec<ConfigModel>, settings: OxllmSettings) -> ProxyConfig {
    ProxyConfig {
        model_list,
        oxllm_settings: settings,
        general_settings: GeneralSettings {
            master_key: None,
            allow_private_api_base: Some(true),
            ..Default::default()
        },
        router_settings: Default::default(),
        extra: serde_json::Map::new(),
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

fn error_body(status: u16, message: &str) -> Value {
    json!({
    "error": {
    "message": message,
    "type": if status >= 500 { "server_error" } else { "invalid_request_error" },
    "code": if status == 400 { "context_length_exceeded" } else { "internal_error" }
    }
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
            .json_body(error_body(status, "upstream failure"));
    })
}

fn make_request_body(model: &str) -> Value {
    json!({
    "model": model,
    "messages": [{"role": "user", "content": "hi"}],
    })
}

async fn send_request(handle: &OxllmHandle, model: &str) -> reqwest::Response {
    http_client()
        .post(handle.url("/v1/chat/completions"))
        .json(&make_request_body(model))
        .send()
        .await
        .expect("request should succeed")
}

/// Wait until `key` leaves the router's cooldown store, polling every 50ms
/// for up to 3s. Replaces fixed sleeps on the 1s test cooldown with a
/// deterministic condition (no timing margin to flake).
/// The stable cooldown key of the deployment at `index` in the live router.
async fn cooldown_key_of(shared: &oxllm_server::SharedState, index: usize) -> String {
    let state = shared.load_full();
    state.router.cooldown_key(&state.deployments[index])
}

async fn wait_until_not_cooling(shared: &oxllm_server::SharedState, key: &str) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(3);
    while shared.load_full().router.cooldown_store().is_cooling(key) {
        assert!(
            std::time::Instant::now() < deadline,
            "deployment {key} still cooling after 3s"
        );
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
}

#[tokio::test]
async fn test_load_distribution_across_deployments() {
    let mock1 = MockServer::start();
    let mock2 = MockServer::start();

    let _m1 = setup_ok_mock(&mock1, "from-dep1", "key1");
    let _m2 = setup_ok_mock(&mock2, "from-dep2", "key2");

    let config = build_config(
        vec![
            model_entry("gpt-4o", &mock1.base_url(), "key1"),
            model_entry("gpt-4o", &mock2.base_url(), "key2"),
        ],
        OxllmSettings::default(),
    );
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let mut contents = std::collections::HashSet::new();
    for _ in 0..50 {
        let resp = send_request(&handle, "gpt-4o").await;
        assert_eq!(resp.status(), 200);
        // Router observability headers: the default strategy serves every
        // request via a direct pick (no fallback, no reroute, no pin).
        assert_eq!(
            resp.headers()
                .get("x-oxllm-router-strategy")
                .and_then(|v| v.to_str().ok()),
            Some("simple_shuffle"),
            "every response must carry the router strategy header"
        );
        assert_eq!(
            resp.headers()
                .get("x-oxllm-router-decision")
                .and_then(|v| v.to_str().ok()),
            Some("picked"),
            "every response must carry the router decision header"
        );
        let body: Value = resp.json().await.expect("json body");
        if let Some(c) = body["choices"][0]["message"]["content"].as_str() {
            contents.insert(c.to_string());
        }
    }

    assert!(
        contents.len() > 1,
        "requests should be distributed across multiple deployments, got: {contents:?}"
    );

    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn test_complexity_router_classifies_once_and_records_tier_metric() {
    let mock = MockServer::start();
    let _ok = setup_ok_mock(&mock, "from-complexity", "key-c");

    let mut config = build_config(
        vec![model_entry("gpt-4o", &mock.base_url(), "key-c")],
        OxllmSettings {
            routing: RoutingKnobs {
                routing_strategy: Some("complexity_router".to_string()),
                ..Default::default()
            },
            ..Default::default()
        },
    );
    config.router_settings.complexity_router_config =
        Some(oxllm_core::router::ComplexityRouterConfig {
            tiers: std::collections::HashMap::from([(
                "simple".to_string(),
                vec!["gpt-4o".to_string()],
            )]),
            ..Default::default()
        });
    let (handle, _shared) = start_oxllm(config).await.expect("server should start");

    // The in-process harness does not run `app::run` (which installs the
    // Prometheus recorder), so install it here BEFORE the request fires
    // the metric.
    let _ = oxllm_server::metrics::render();

    let resp = send_request(&handle, "gpt-4o").await;
    assert_eq!(resp.status(), 200);
    // The complexity-routed request still carries the router headers.
    assert_eq!(
        resp.headers()
            .get("x-oxllm-router-strategy")
            .and_then(|v| v.to_str().ok()),
        Some("complexity_router")
    );
    let body: Value = resp.json().await.expect("json");
    assert_eq!(body["choices"][0]["message"]["content"], "from-complexity");

    // "hi" scores 0 → Simple. The classification ran once per request in
    // `ChatOrchestrator::run` (stamped onto the context; the strategy pick
    // reuses it), so the tier counter reads exactly 1.
    let rendered = oxllm_server::metrics::render();
    assert!(
        rendered.contains("oxllm_complexity_tier_total{tier=\"simple\"} 1"),
        "the once-per-request classification must increment the tier metric exactly once, got:\n{rendered}"
    );

    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn test_retry_succeeds_on_second_deployment() {
    oxllm_server::metrics::install();
    let model = "retry-proof-model";
    let mock_fail = MockServer::start();
    let mock_ok = MockServer::start();

    let fail_mock = setup_fail_mock(&mock_fail, 500, "key-fail");
    let ok_mock = setup_ok_mock(&mock_ok, "from-ok", "key-ok");

    let config = build_config(
        vec![
            model_entry_ordered(model, &mock_fail.base_url(), "key-fail", 0),
            model_entry_ordered(model, &mock_ok.base_url(), "key-ok", 1),
        ],
        OxllmSettings {
            routing: RoutingKnobs {
                num_retries: Some(1),
                allowed_fails: Some(10),
                ..Default::default()
            },
            ..Default::default()
        },
    );
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    // Priority tiers force the failed deployment to be tried first. A test
    // that sends random requests can pass without ever exercising a retry.
    let resp = send_request(&handle, model).await;
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("successful retry body");
    assert_eq!(body["choices"][0]["message"]["content"], "from-ok");
    fail_mock.assert_hits(2); // initial attempt + one same-deployment retry
    ok_mock.assert_hits(1); // then rotate to the healthy sibling
    assert_eq!(
        common::metric_value(
            &oxllm_server::metrics::render(),
            "oxllm_upstream_retries_total",
            &[("model", model), ("provider", "openai")],
        ),
        Some(1.0),
        "only the additional same-deployment attempt counts as a retry"
    );

    let resp = send_request(&handle, model).await;
    assert_eq!(resp.status(), 200);
    fail_mock.assert_hits(2); // failed deployment is still cooling
    ok_mock.assert_hits(2);

    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn test_all_deployments_fail_triggers_fallback() {
    let mock_primary = MockServer::start();
    let mock_fallback = MockServer::start();

    let _primary_fail = setup_fail_mock(&mock_primary, 500, "primary-key");
    let _fb_ok = setup_ok_mock(&mock_fallback, "from-fallback", "fb-key");

    let config = build_config(
        vec![
            model_entry("gpt-4o", &mock_primary.base_url(), "primary-key"),
            model_entry("claude-backup", &mock_fallback.base_url(), "fb-key"),
        ],
        OxllmSettings {
            routing: RoutingKnobs {
                num_retries: Some(2),
                fallbacks: Some(json!([{"gpt-4o": ["claude-backup"]}])),
                ..Default::default()
            },
            ..Default::default()
        },
    );
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let resp = send_request(&handle, "gpt-4o").await;
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("json");
    assert_eq!(body["choices"][0]["message"]["content"], "from-fallback");

    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn test_fallback_header_is_set() {
    let mock_primary = MockServer::start();
    let mock_fallback = MockServer::start();

    let _primary_fail = setup_fail_mock(&mock_primary, 500, "pk");
    let _fb_ok = setup_ok_mock(&mock_fallback, "fb-content", "fbk");

    let config = build_config(
        vec![
            model_entry("gpt-4o", &mock_primary.base_url(), "pk"),
            model_entry("gpt-4o-mini", &mock_fallback.base_url(), "fbk"),
        ],
        OxllmSettings {
            routing: RoutingKnobs {
                num_retries: Some(2),
                fallbacks: Some(json!([{"gpt-4o": ["gpt-4o-mini"]}])),
                ..Default::default()
            },
            ..Default::default()
        },
    );
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let resp = send_request(&handle, "gpt-4o").await;
    assert_eq!(resp.status(), 200);
    assert_eq!(
        resp.headers()
            .get("x-oxllm-fallback-used")
            .and_then(|v| v.to_str().ok()),
        Some("gpt-4o-mini")
    );

    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn test_4xx_does_not_trigger_fallback() {
    let mock_primary = MockServer::start();
    let mock_fallback = MockServer::start();

    let _primary_4xx = mock_primary.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .header("authorization", "Bearer pk");
        then.status(400)
            .header("content-type", "application/json")
            .json_body(json!({
            "error": {
            "message": "bad request: invalid model",
            "type": "invalid_request_error",
            "code": "invalid_model"
            }
            }));
    });
    let fb_ok = setup_ok_mock(&mock_fallback, "should-not-happen", "fbk");

    let config = build_config(
        vec![
            model_entry("gpt-4o", &mock_primary.base_url(), "pk"),
            model_entry("gpt-4o-mini", &mock_fallback.base_url(), "fbk"),
        ],
        OxllmSettings {
            routing: RoutingKnobs {
                num_retries: Some(3),
                fallbacks: Some(json!([{"gpt-4o": ["gpt-4o-mini"]}])),
                ..Default::default()
            },
            ..Default::default()
        },
    );
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let resp = send_request(&handle, "gpt-4o").await;
    assert_eq!(resp.status(), 400);
    assert!(
        resp.headers().get("x-oxllm-fallback-used").is_none(),
        "fallback header should not be set on 4xx"
    );
    fb_ok.assert_hits(0);

    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn test_wildcard_routing_unconfigured_model() {
    let mock = MockServer::start();

    let _m = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .header("authorization", "Bearer wild-key");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(success_body("from-wildcard"));
    });

    let config = build_config(
        vec![model_entry_with_litellm_model(
            "*",
            "openai/*",
            &mock.base_url(),
            "wild-key",
        )],
        OxllmSettings::default(),
    );
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let resp = send_request(&handle, "openai/gpt-4o-mini").await;
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("json");
    assert_eq!(body["choices"][0]["message"]["content"], "from-wildcard");

    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn test_context_window_fallback() {
    let mock_primary = MockServer::start();
    let mock_fallback = MockServer::start();

    let _primary_ctx = mock_primary.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .header("authorization", "Bearer pk");
        then.status(400)
            .header("content-type", "application/json")
            .json_body(json!({
            "error": {
            "message": "context length exceeded",
            "type": "invalid_request_error",
            "code": "context_length_exceeded"
            }
            }));
    });
    let _fb_ok = setup_ok_mock(&mock_fallback, "larger-model-ok", "fbk");

    let config = build_config(
        vec![
            model_entry("small-model", &mock_primary.base_url(), "pk"),
            model_entry("big-model", &mock_fallback.base_url(), "fbk"),
        ],
        OxllmSettings {
            routing: RoutingKnobs {
                num_retries: Some(2),
                context_window_fallbacks: Some(json!([{"small-model": ["big-model"]}])),
                ..Default::default()
            },
            ..Default::default()
        },
    );
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let resp = send_request(&handle, "small-model").await;
    assert_eq!(resp.status(), 200);
    assert_eq!(
        resp.headers()
            .get("x-oxllm-fallback-used")
            .and_then(|v| v.to_str().ok()),
        Some("big-model")
    );

    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn test_wildcard_star_matches_any_provider_model() {
    let mock = MockServer::start();

    let _m = setup_ok_mock(&mock, "any-model", "any-key");

    let config = build_config(
        vec![model_entry_with_litellm_model(
            "*",
            "openai/*",
            &mock.base_url(),
            "any-key",
        )],
        OxllmSettings::default(),
    );
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let resp = send_request(&handle, "openai/gpt-99-turbo").await;
    assert_eq!(resp.status(), 200);

    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn test_single_deployment_retries_same_deployment() {
    let mock = MockServer::start();

    let fail_mock = setup_fail_mock(&mock, 500, "key");

    let config = build_config(
        vec![model_entry("gpt-4o", &mock.base_url(), "key")],
        OxllmSettings {
            routing: RoutingKnobs {
                num_retries: Some(2),
                ..Default::default()
            },
            ..Default::default()
        },
    );
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let resp = send_request(&handle, "gpt-4o").await;
    assert_eq!(resp.status(), 500);

    fail_mock.assert_hits(3);

    let _ = handle.shutdown_tx.send(());
}

/// Safety invariant: once a streaming response starts (2xx received), a
/// mid-stream error must NOT trigger fallback to another model. The client
/// already received SSE bytes; switching models would corrupt the response.
#[tokio::test]
async fn test_streaming_mid_stream_error_does_not_fallback() {
    let mock_primary = MockServer::start();
    let mock_fallback = MockServer::start();

    let _primary_stream = mock_primary.mock(|when, then| {
 when.method(httpmock::Method::POST)
 .path("/chat/completions")
 .header("authorization", "Bearer pk")
 .json_body_partial(json!({"stream": true}).to_string());
 then.status(200)
 .header("content-type", "text/event-stream")
 .body(
 "data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi\"},\"finish_reason\":null}]}\n\n");
 });
    let fb_ok = setup_ok_mock(&mock_fallback, "should-not-stream", "fbk");

    let config = build_config(
        vec![
            model_entry("gpt-4o", &mock_primary.base_url(), "pk"),
            model_entry("gpt-4o-mini", &mock_fallback.base_url(), "fbk"),
        ],
        OxllmSettings {
            routing: RoutingKnobs {
                num_retries: Some(2),
                fallbacks: Some(json!([{"gpt-4o": ["gpt-4o-mini"]}])),
                ..Default::default()
            },
            ..Default::default()
        },
    );
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let resp = http_client()
        .post(handle.url("/v1/chat/completions"))
        .json(&json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": true
        }))
        .send()
        .await
        .expect("request");

    assert_eq!(resp.status(), 200);
    let ct = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(ct.contains("text/event-stream"), "got content-type: {ct}");

    assert!(
        resp.headers().get("x-oxllm-fallback-used").is_none(),
        "fallback header must not be set — mid-stream errors do not fall back"
    );
    fb_ok.assert_hits(0);

    let _ = handle.shutdown_tx.send(());
}

/// Anthropic-shaped context-window error should trigger context-window
/// fallback (previously only OpenAI-shaped codes were detected).
#[tokio::test]
async fn test_anthropic_context_window_error_triggers_fallback() {
    let mock_primary = MockServer::start();
    let mock_fallback = MockServer::start();

    let _primary_ctx = mock_primary.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .header("authorization", "Bearer pk");
        then.status(400)
            .header("content-type", "application/json")
            .json_body(json!({
            "error": {
            "type": "invalid_request_error",
            "message": "prompt is too long: 300000 tokens > 200000 maximum"
            }
            }));
    });
    let _fb_ok = setup_ok_mock(&mock_fallback, "larger-model-ok", "fbk");

    let config = build_config(
        vec![
            model_entry("small-model", &mock_primary.base_url(), "pk"),
            model_entry("big-model", &mock_fallback.base_url(), "fbk"),
        ],
        OxllmSettings {
            routing: RoutingKnobs {
                num_retries: Some(2),
                context_window_fallbacks: Some(json!([{"small-model": ["big-model"]}])),
                ..Default::default()
            },
            ..Default::default()
        },
    );
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let resp = send_request(&handle, "small-model").await;
    assert_eq!(resp.status(), 200);
    assert_eq!(
        resp.headers()
            .get("x-oxllm-fallback-used")
            .and_then(|v| v.to_str().ok()),
        Some("big-model")
    );

    let _ = handle.shutdown_tx.send(());
}

/// Gemini-shaped context-window error should trigger context-window
/// fallback.
#[tokio::test]
async fn test_gemini_context_window_error_triggers_fallback() {
    let mock_primary = MockServer::start();
    let mock_fallback = MockServer::start();

    let _primary_ctx = mock_primary.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .header("authorization", "Bearer pk");
        then.status(400)
            .header("content-type", "application/json")
            .json_body(json!({
            "error": {
            "message": "Request exceeds context limit: 1000000 tokens",
            "status": "INVALID_ARGUMENT"
            }
            }));
    });
    let _fb_ok = setup_ok_mock(&mock_fallback, "larger-model-ok", "fbk");

    let config = build_config(
        vec![
            model_entry("small-model", &mock_primary.base_url(), "pk"),
            model_entry("big-model", &mock_fallback.base_url(), "fbk"),
        ],
        OxllmSettings {
            routing: RoutingKnobs {
                num_retries: Some(2),
                context_window_fallbacks: Some(json!([{"small-model": ["big-model"]}])),
                ..Default::default()
            },
            ..Default::default()
        },
    );
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let resp = send_request(&handle, "small-model").await;
    assert_eq!(resp.status(), 200);
    assert_eq!(
        resp.headers()
            .get("x-oxllm-fallback-used")
            .and_then(|v| v.to_str().ok()),
        Some("big-model")
    );

    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn test_wildcard_cohere_routes_to_cohere_v2_chat() {
    let mock = MockServer::start();

    let _m = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat")
            .header("authorization", "Bearer cohere-key");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
            "id": "cohere-1",
            "finish_reason": "COMPLETE",
            "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": "from-cohere"}]
            },
            "usage": {
            "billed_units": {"input_tokens": 5, "output_tokens": 3},
            "tokens": {"input_tokens": 6, "output_tokens": 4}
            }
            }));
    });

    let mut p = params(&mock.base_url(), "cohere-key");
    p.model = "cohere/command-r-plus".to_string();
    let entry = ConfigModel {
        model_name: "*".to_string(),
        oxllm_params: p,
        ..Default::default()
    };

    let config = build_config(vec![entry], OxllmSettings::default());
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let resp = send_request(&handle, "cohere/command-r-plus").await;
    let status = resp.status();
    let body: Value = resp.json().await.expect("json");
    assert_eq!(status, 200, "body: {body}");
    assert_eq!(body["choices"][0]["message"]["content"], "from-cohere");
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
    assert_eq!(body["usage"]["prompt_tokens"], 5);
    assert_eq!(body["usage"]["completion_tokens"], 3);
    assert_eq!(body["usage"]["total_tokens"], 8);
    assert_eq!(body["object"], "chat.completion");

    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn test_wildcard_azure_routes_with_deployment_url_and_api_key_header() {
    let mock = MockServer::start();

    let _m = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path_matches(
                regex::Regex::new(r"^/openai/deployments/[^/]+/chat/completions$").unwrap(),
            )
            .query_param("api-version", "2024-02-15-preview")
            .header("api-key", "azure-key");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(success_body("from-azure"));
    });

    let mut p = params(&mock.base_url(), "azure-key");
    p.model = "azure/my-gpt4-deployment".to_string();
    p.api_version = Some("2024-02-15-preview".to_string());
    let entry = ConfigModel {
        model_name: "*".to_string(),
        oxllm_params: p,
        ..Default::default()
    };

    let config = build_config(vec![entry], OxllmSettings::default());
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let resp = send_request(&handle, "azure/my-gpt4-deployment").await;
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("json");
    assert_eq!(body["choices"][0]["message"]["content"], "from-azure");

    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn test_azure_legacy_deployments_chat_route() {
    let mock = MockServer::start();

    let _m = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path_matches(
                regex::Regex::new(r"^/openai/deployments/[^/]+/chat/completions$").unwrap(),
            )
            .query_param("api-version", "2024-02-15-preview")
            .header("api-key", "azure-key");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(success_body("via-legacy-route"));
    });

    let mut p = params(&mock.base_url(), "azure-key");
    p.model = "azure/my-gpt4-deployment".to_string();
    p.api_version = Some("2024-02-15-preview".to_string());
    let entry = ConfigModel {
        model_name: "my-gpt4-deployment".to_string(),
        oxllm_params: p,
        ..Default::default()
    };

    let config = build_config(vec![entry], OxllmSettings::default());
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let body = json!({
    "messages": [{"role": "user", "content": "hi"}],
    });
    let resp = http_client()
        .post(handle.url("/openai/deployments/my-gpt4-deployment/chat/completions"))
        .json(&body)
        .send()
        .await
        .expect("request should succeed");
    assert_eq!(resp.status(), 200);
    let v: Value = resp.json().await.expect("json");
    assert_eq!(v["choices"][0]["message"]["content"], "via-legacy-route");

    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn test_azure_legacy_engines_chat_route() {
    let mock = MockServer::start();

    let _m = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path_matches(
                regex::Regex::new(r"^/openai/deployments/[^/]+/chat/completions$").unwrap(),
            )
            .header("api-key", "azure-key");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(success_body("via-engines-route"));
    });

    let mut p = params(&mock.base_url(), "azure-key");
    p.model = "azure/my-engine".to_string();
    p.api_version = Some("2024-02-15-preview".to_string());
    let entry = ConfigModel {
        model_name: "my-engine".to_string(),
        oxllm_params: p,
        ..Default::default()
    };

    let config = build_config(vec![entry], OxllmSettings::default());
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let body = json!({
    "messages": [{"role": "user", "content": "hi"}],
    });
    let resp = http_client()
        .post(handle.url("/engines/my-engine/chat/completions"))
        .json(&body)
        .send()
        .await
        .expect("request should succeed");
    assert_eq!(resp.status(), 200);
    let v: Value = resp.json().await.expect("json");
    assert_eq!(v["choices"][0]["message"]["content"], "via-engines-route");

    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn test_azure_legacy_deployments_embeddings_route() {
    let mock = MockServer::start();

    let _m = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path_matches(regex::Regex::new(r"^/openai/deployments/[^/]+/embeddings$").unwrap())
            .header("api-key", "azure-key");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
            "model": "my-embed",
            "usage": {"prompt_tokens": 2, "total_tokens": 2}
            }));
    });

    let mut p = params(&mock.base_url(), "azure-key");
    p.model = "azure/my-embed".to_string();
    p.api_version = Some("2024-02-15-preview".to_string());
    let entry = ConfigModel {
        model_name: "my-embed".to_string(),
        oxllm_params: p,
        ..Default::default()
    };

    let config = build_config(vec![entry], OxllmSettings::default());
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let body = json!({"input": "hello"});
    let resp = http_client()
        .post(handle.url("/openai/deployments/my-embed/embeddings"))
        .json(&body)
        .send()
        .await
        .expect("request should succeed");
    assert_eq!(resp.status(), 200);
    let v: Value = resp.json().await.expect("json");
    assert_eq!(v["data"][0]["embedding"], json!([0.1, 0.2]));

    let _ = handle.shutdown_tx.send(());
}

/// End-to-end coverage for the Azure provider's streaming path. The unit
/// tests exercise `transform_stream_chunk` with hand-crafted chunks, but no
/// httpmock-based test previously verified the full pipeline (SSE framing
/// through the shared parser, usage accounting on the final chunk, and
/// `[DONE]` handling) for Azure. A regression in any of these would not be
/// caught otherwise. Mirrors `openai_streaming_with_usage_chunk` in
/// `openai_e2e.rs`, but routes through the Azure deployment path and asserts
/// the `api-key` header (not `Authorization: Bearer`) reaches the upstream.
#[tokio::test]
async fn test_azure_streaming_chat_completion_with_usage() {
    let mock = MockServer::start();

    let sse_body = concat!(
        "data: {\"id\":\"chatcmpl-az-stream\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"my-gpt4-deployment\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"chatcmpl-az-stream\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"my-gpt4-deployment\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"chatcmpl-az-stream\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"my-gpt4-deployment\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5,\"total_tokens\":15}}\n\n",
        "data: [DONE]\n\n"
    );

    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path_matches(
                regex::Regex::new(r"^/openai/deployments/[^/]+/chat/completions$").unwrap(),
            )
            .query_param("api-version", "2024-02-15-preview")
            .header("api-key", "azure-key")
            .json_body_partial(json!({"stream": true}).to_string());
        then.status(200)
            .header("content-type", "text/event-stream")
            .body(sse_body);
    });

    let mut p = params(&mock.base_url(), "azure-key");
    p.model = "azure/my-gpt4-deployment".to_string();
    p.api_version = Some("2024-02-15-preview".to_string());
    let entry = ConfigModel {
        model_name: "*".to_string(),
        oxllm_params: p,
        ..Default::default()
    };

    let config = build_config(vec![entry], OxllmSettings::default());
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let resp = http_client()
        .post(handle.url("/v1/chat/completions"))
        .json(&json!({
            "model": "azure/my-gpt4-deployment",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": true,
        }))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);
    let text = resp.text().await.expect("body");

    assert!(
        text.contains("\"completion_tokens\":5"),
        "streamed body should contain the usage chunk: {text}"
    );
    assert!(
        text.contains("\"total_tokens\":15"),
        "streamed body should contain total tokens: {text}"
    );
    assert!(
        text.contains("[DONE]"),
        "stream must terminate with [DONE]: {text}"
    );

    upstream.assert_hits(1);
    let _ = handle.shutdown_tx.send(());
}

/// When the primary deployment times out, the proxy must mark it cooling
/// and route the next attempt to a configured fallback model. The
/// `x-oxllm-fallback-used` header must reflect the model that ultimately
/// served the request.
///
/// `num_retries: 0` ensures the timeout surfaces on the first attempt (no
/// exponential-backoff sleep), which keeps the test snappy and forces the
/// retry loop to fall through to the regular fallback chain.
#[tokio::test]
async fn upstream_timeout_triggers_retry_on_different_deployment() {
    let mock_primary = MockServer::start();
    let mock_fallback = MockServer::start();

    // Primary: 200 but delayed past the per-deployment timeout → 504 in the
    // chat orchestrator, classified Retryable.
    mock_primary.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .header("authorization", "Bearer pk");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(success_body("late-from-primary"))
            .delay(std::time::Duration::from_secs(5));
    });
    // Fallback: returns promptly.
    let fb_mock = setup_ok_mock(&mock_fallback, "from-fallback", "fbk");

    let mut primary = model_entry("gpt-4o", &mock_primary.base_url(), "pk");
    primary.oxllm_params.timeout = Some(0.2); // 200 ms — primary always times out
    let fallback = model_entry("gpt-4o-mini", &mock_fallback.base_url(), "fbk");

    let config = build_config(
        vec![primary, fallback],
        OxllmSettings {
            routing: RoutingKnobs {
                num_retries: Some(0),
                fallbacks: Some(json!([{"gpt-4o": ["gpt-4o-mini"]}])),
                ..Default::default()
            },
            ..Default::default()
        },
    );
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(15))
        .build()
        .expect("client");
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .json(&make_request_body("gpt-4o"))
        .send()
        .await
        .expect("request should reach the proxy");

    assert_eq!(
        resp.status(),
        200,
        "primary timed out → fallback must succeed; got {} body {:?}",
        resp.status(),
        resp.text().await.unwrap_or_default()
    );
    assert_eq!(
        resp.headers()
            .get("x-oxllm-fallback-used")
            .and_then(|v| v.to_str().ok()),
        Some("gpt-4o-mini"),
        "x-oxllm-fallback-used must point at the model that served the request"
    );
    // The fallback deployment was actually hit.
    fb_mock.assert_hits(1);

    let _ = handle.shutdown_tx.send(());
}

// ---------------------------------------------------------------------------
// Prompt-caching deployment check + allowed-fails policy (Batch 1)
// ---------------------------------------------------------------------------

/// The API-key hash used in open mode (no master key configured): lowercase
/// hex of 32 zero bytes, matching `AuthUser::open().key_hash_hex`.
const OPEN_KEY_HASH: &str = "0000000000000000000000000000000000000000000000000000000000000000";

/// A request body whose system prompt is ~2000 estimated tokens — above the
/// default 1024-token prompt-caching threshold.
fn big_prompt_body(model: &str) -> Value {
    let text = "lorem ipsum dolor sit amet ".repeat(400);
    json!({
    "model": model,
    "messages": [
    {"role": "system", "content": text},
    {"role": "user", "content": "hi"}
    ],
    })
}

async fn post_body(handle: &OxllmHandle, body: &Value) -> reqwest::Response {
    http_client()
        .post(handle.url("/v1/chat/completions"))
        .json(body)
        .send()
        .await
        .expect("request should succeed")
}

/// Resolve which deployment a pin value (the deployment's stable cooldown
/// key) belongs to, by matching it against the live router's keys.
async fn pinned_index(shared: &oxllm_server::SharedState, pin: &str) -> usize {
    let state = shared.load_full();
    state
        .deployments
        .iter()
        .position(|d| state.router.cooldown_key(d) == pin)
        .unwrap_or_else(|| panic!("pin value must be a deployment cooldown key, got {pin}"))
}

/// Pin swap: a request with a cacheable (large) prompt is pinned to the
/// deployment that served it; subsequent identical prompts must all route
/// to that deployment, never to its sibling.
#[tokio::test]
async fn test_prompt_cache_pins_large_prompt_to_serving_deployment() {
    let mock1 = MockServer::start();
    let mock2 = MockServer::start();

    let m1 = setup_ok_mock(&mock1, "from-dep1", "key1");
    let m2 = setup_ok_mock(&mock2, "from-dep2", "key2");

    let config = build_config(
        vec![
            model_entry("gpt-4o", &mock1.base_url(), "key1"),
            model_entry("gpt-4o", &mock2.base_url(), "key2"),
        ],
        OxllmSettings {
            prompt_caching_enabled: Some(true),
            ..Default::default()
        },
    );
    let (handle, shared) = start_oxllm(config).await.expect("server should start");

    let state = shared.load_full();
    let pcache = PromptCachingCheck::new(state.cache.clone(), None);
    let body = big_prompt_body("gpt-4o");

    // First request writes the pin for the deployment that served it.
    let resp = post_body(&handle, &body).await;
    assert_eq!(resp.status(), 200);

    let (hash, _) = pcache.prompt_hash(&body, "gpt-4o").expect("hash");
    let pinned = pcache
        .preferred_deployment(hash, OPEN_KEY_HASH)
        .await
        .expect("pin must be written after a successful large-prompt call");
    let pinned_idx = pinned_index(&shared, &pinned).await;
    assert!(pinned_idx < 2, "pin index out of range: {pinned}");

    // Subsequent requests with the same prompt must all hit the pinned
    // deployment — the pin overrides the shuffle.
    for _ in 0..10 {
        let resp = post_body(&handle, &body).await;
        assert_eq!(resp.status(), 200);
    }
    let hits = [m1.hits(), m2.hits()];
    assert_eq!(
        hits[pinned_idx], 11,
        "pinned deployment must serve every request (hits: {hits:?})"
    );
    assert_eq!(
        hits[1 - pinned_idx],
        0,
        "unpinned deployment must not be hit (hits: {hits:?})"
    );

    let _ = handle.shutdown_tx.send(());
}

/// Pipeline path: with `optional_pre_call_checks: ["prompt_caching"]` the
/// pin steers the pick inside the router's pre-call check pipeline instead
/// of via the legacy post-pick prefer. Pinning behavior is unchanged: a
/// large prompt is pinned to the deployment that served it and subsequent
/// identical prompts all route to it, never to its sibling.
#[tokio::test]
async fn test_prompt_cache_pipeline_path_via_optional_pre_call_checks() {
    let mock1 = MockServer::start();
    let mock2 = MockServer::start();

    let m1 = setup_ok_mock(&mock1, "from-dep1", "key1");
    let m2 = setup_ok_mock(&mock2, "from-dep2", "key2");

    let mut config = build_config(
        vec![
            model_entry("gpt-4o", &mock1.base_url(), "key1"),
            model_entry("gpt-4o", &mock2.base_url(), "key2"),
        ],
        OxllmSettings {
            prompt_caching_enabled: Some(true),
            ..Default::default()
        },
    );
    config.router_settings.optional_pre_call_checks = Some(vec!["prompt_caching".to_string()]);
    let (handle, shared) = start_oxllm(config).await.expect("server should start");

    let state = shared.load_full();
    let pcache = PromptCachingCheck::new(state.cache.clone(), None);
    let body = big_prompt_body("gpt-4o");

    // First request writes the pin for the deployment that served it.
    let resp = post_body(&handle, &body).await;
    assert_eq!(resp.status(), 200);

    let (hash, _) = pcache.prompt_hash(&body, "gpt-4o").expect("hash");
    let pinned = pcache
        .preferred_deployment(hash, OPEN_KEY_HASH)
        .await
        .expect("pin must be written after a successful large-prompt call");
    let pinned_idx = pinned_index(&shared, &pinned).await;
    assert!(pinned_idx < 2, "pin index out of range: {pinned}");

    // Subsequent requests with the same prompt must all hit the pinned
    // deployment — the pipeline filters candidates to the pin pre-pick.
    for _ in 0..10 {
        let resp = post_body(&handle, &body).await;
        assert_eq!(resp.status(), 200);
    }
    let hits = [m1.hits(), m2.hits()];
    assert_eq!(
        hits[pinned_idx], 11,
        "pinned deployment must serve every request (hits: {hits:?})"
    );
    assert_eq!(
        hits[1 - pinned_idx],
        0,
        "unpinned deployment must not be hit (hits: {hits:?})"
    );

    let _ = handle.shutdown_tx.send(());
}

/// Small prompts stay below the caching threshold: no pin is written and
/// routing is unaffected.
#[tokio::test]
async fn test_prompt_cache_ignores_small_prompts() {
    let mock = MockServer::start();

    let _m = setup_ok_mock(&mock, "from-dep", "key");

    let config = build_config(
        vec![model_entry("gpt-4o", &mock.base_url(), "key")],
        OxllmSettings {
            prompt_caching_enabled: Some(true),
            ..Default::default()
        },
    );
    let (handle, shared) = start_oxllm(config).await.expect("server should start");

    let resp = send_request(&handle, "gpt-4o").await;
    assert_eq!(resp.status(), 200);

    let state = shared.load_full();
    let pcache = PromptCachingCheck::new(state.cache.clone(), None);
    let (hash, _) = pcache
        .prompt_hash(&make_request_body("gpt-4o"), "gpt-4o")
        .expect("hash");
    assert!(
        pcache
            .preferred_deployment(hash, OPEN_KEY_HASH)
            .await
            .is_none(),
        "small prompts must not write a pin"
    );

    let _ = handle.shutdown_tx.send(());
}

/// Allowed fails: a deployment cools as soon as its failure count EXCEEDS
/// `allowed_fails` (exclusive `>` semantics), instead of exhausting
/// `num_retries` first. With `allowed_fails: 2` the deployment tolerates 2
/// failures and cools on the 3rd.
#[tokio::test]
async fn test_allowed_fails_cools_deployment_after_threshold() {
    let mock = MockServer::start();

    let fail = setup_fail_mock(&mock, 500, "key");

    let config = build_config(
        vec![model_entry("gpt-4o", &mock.base_url(), "key")],
        OxllmSettings {
            routing: RoutingKnobs {
                num_retries: Some(5),
                allowed_fails: Some(2),
                ..Default::default()
            },
            ..Default::default()
        },
    );
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    // Three failures exceed the allowed_fails limit of 2 (count=3 > limit=2
    // under exclusive `>` semantics) and cool the deployment — the remaining
    // retries are skipped.
    let resp = send_request(&handle, "gpt-4o").await;
    assert_eq!(resp.status(), 500);
    fail.assert_hits(3);

    // While cooling, the next request never reaches the upstream.
    // Returns 429 (model exists but all deployments unavailable), not 404.
    let resp = send_request(&handle, "gpt-4o").await;
    assert_eq!(
        resp.status(),
        429,
        "no deployment available while the only one cools"
    );
    fail.assert_hits(3);

    let _ = handle.shutdown_tx.send(());
}

/// 4xx policy: with `allowed_fails_policy: {"401": 0}` a single 401 cools
/// the deployment immediately (401s are never retried). Exclusive `>`
/// semantics: count=1 > limit=0 → LimitReached on the first failure.
#[tokio::test]
async fn test_allowed_fails_policy_cools_immediately_on_401() {
    let mock = MockServer::start();

    let fail = setup_fail_mock(&mock, 401, "key");

    let policy = std::collections::HashMap::from([("401".to_string(), 0u32)]);
    let config = build_config(
        vec![model_entry("gpt-4o", &mock.base_url(), "key")],
        OxllmSettings {
            routing: RoutingKnobs {
                num_retries: Some(3),
                allowed_fails_policy: Some(policy),
                ..Default::default()
            },
            ..Default::default()
        },
    );
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    // The first 401 cools the deployment (count=1 > limit=0); the client
    // error is surfaced.
    let resp = send_request(&handle, "gpt-4o").await;
    assert_eq!(resp.status(), 401);
    fail.assert_hits(1);

    // Cooling: the next request is rejected without touching the upstream.
    // Returns 429 (model exists but all deployments unavailable), not 404.
    let resp = send_request(&handle, "gpt-4o").await;
    assert_eq!(resp.status(), 429);
    fail.assert_hits(1);

    let _ = handle.shutdown_tx.send(());
}

/// Pin rewrite: when the pinned deployment fails over to a sibling, the pin
/// is rewritten to the deployment that actually served the request, and the
/// next request follows the rewritten pin.
#[tokio::test]
async fn test_prompt_cache_pin_rewritten_after_failover() {
    let mock1 = MockServer::start();
    let mock2 = MockServer::start();

    let ok1 = setup_ok_mock(&mock1, "from-dep1", "key1");
    let ok2 = setup_ok_mock(&mock2, "from-dep2", "key2");

    let config = build_config(
        vec![
            model_entry("gpt-4o", &mock1.base_url(), "key1"),
            model_entry("gpt-4o", &mock2.base_url(), "key2"),
        ],
        OxllmSettings {
            prompt_caching_enabled: Some(true),
            routing: RoutingKnobs {
                // No retries: the failing deployment cools after one
                // attempt so the failover is immediate.
                num_retries: Some(0),
                ..Default::default()
            },
            ..Default::default()
        },
    );
    let (handle, shared) = start_oxllm(config).await.expect("server should start");

    let state = shared.load_full();
    let pcache = PromptCachingCheck::new(state.cache.clone(), None);
    let body = big_prompt_body("gpt-4o");

    // Request 1 succeeds and writes the pin.
    let resp = post_body(&handle, &body).await;
    assert_eq!(resp.status(), 200);
    let (hash, _) = pcache.prompt_hash(&body, "gpt-4o").expect("hash");
    let pinned = pcache
        .preferred_deployment(hash, OPEN_KEY_HASH)
        .await
        .expect("pin written");
    let pinned_idx = pinned_index(&shared, &pinned).await;
    assert!(pinned_idx < 2, "pin index out of range: {pinned}");

    // The pinned deployment starts failing; its sibling stays healthy.
    if pinned_idx == 0 {
        ok1.delete_async().await;
        let _f = setup_fail_mock(&mock1, 500, "key1");
    } else {
        ok2.delete_async().await;
        let _f = setup_fail_mock(&mock2, 500, "key2");
    }

    // Request 2: the pin steers to the failing deployment, the failover
    // moves to the healthy sibling, and the pin is rewritten to it.
    let resp = post_body(&handle, &body).await;
    assert_eq!(
        resp.status(),
        200,
        "failover to the healthy deployment must succeed"
    );
    let rewritten = pcache
        .preferred_deployment(hash, OPEN_KEY_HASH)
        .await
        .expect("pin still present");
    assert_eq!(
        rewritten,
        state
            .router
            .cooldown_key(&state.deployments[1 - pinned_idx]),
        "pin must be rewritten to the deployment that served after failover"
    );

    // Request 3 follows the rewritten pin and succeeds.
    let resp = post_body(&handle, &body).await;
    assert_eq!(resp.status(), 200);

    let _ = handle.shutdown_tx.send(());
}

// ---------------------------------------------------------------------------
// Budget limiter (per-deployment + per-provider) — Batch 2
// ---------------------------------------------------------------------------

/// Deployment budget: once the actual spend of a deployment reaches its
/// `max_budget`, the deployment is skipped (429 surfaced) and never called
/// again.
#[tokio::test]
async fn test_deployment_budget_enforced_after_spend_accumulates() {
    let mock = MockServer::start();
    let ok = setup_ok_mock(&mock, "from-dep", "key");

    let mut entry = model_entry("gpt-4o", &mock.base_url(), "key");
    // Budget that fits exactly one estimated request (the check atomically
    // reserves an estimate, then settles to the actual cost): the mock's
    // 5+5-token usage costs ~6.25e-5 USD, so after the first call settles,
    // the counter is over the cap and the second request is rejected.
    entry.oxllm_params.max_budget = Some(4e-5);

    let config = build_config(vec![entry], OxllmSettings::default());
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    // First request: the estimate (tiny with `max_tokens: 1`) fits under
    // the cap → served.
    let resp = post_body(
        &handle,
        &json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1
        }),
    )
    .await;
    assert_eq!(resp.status(), 200);
    ok.assert_hits(1);

    // Second request: the settled spend now exceeds the cap → 429 without
    // touching the upstream. `max_tokens: 1` keeps this request's estimate
    // tiny (a fresh counter would admit it), so the rejection is
    // attributable to the settled spend alone — not to the default
    // 16384-token output estimate.
    let resp = post_body(
        &handle,
        &json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1
        }),
    )
    .await;
    assert_eq!(
        resp.status(),
        429,
        "over-budget deployment must be rejected with 429"
    );
    ok.assert_hits(1);

    let _ = handle.shutdown_tx.send(());
}

/// Provider budget: a provider-level cap gates every deployment of that
/// provider, even when the deployments themselves carry no `max_budget`.
#[tokio::test]
async fn test_provider_budget_enforced_after_spend_accumulates() {
    let mock = MockServer::start();
    let ok = setup_ok_mock(&mock, "from-dep", "key");

    let provider_budgets = std::collections::HashMap::from([(
        "openai".to_string(),
        oxllm_core::router::ProviderBudgetConfig {
            // Fits exactly one estimated request (see the deployment-budget
            // test above); the settled mock usage pushes the counter over.
            budget_limit: 4e-5,
            time_period: "1d".to_string(),
        },
    )]);
    let mut config = build_config(
        vec![model_entry("gpt-4o", &mock.base_url(), "key")],
        OxllmSettings::default(),
    );
    config.router_settings.provider_budget_config = Some(provider_budgets);
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let resp = post_body(
        &handle,
        &json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1
        }),
    )
    .await;
    assert_eq!(resp.status(), 200);
    ok.assert_hits(1);

    // Second request: `max_tokens: 1` keeps the estimate tiny (a fresh
    // counter would admit it), so the 429 is attributable to the settled
    // provider spend alone.
    let resp = post_body(
        &handle,
        &json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1
        }),
    )
    .await;
    assert_eq!(
        resp.status(),
        429,
        "over-budget provider must be rejected with 429"
    );
    ok.assert_hits(1);

    let _ = handle.shutdown_tx.send(());
}

/// Budget failover: with two deployments of the same model, an over-budget
/// deployment is skipped and the request is served by its sibling.
#[tokio::test]
async fn test_budget_exceeded_falls_back_to_next_deployment() {
    let mock_capped = MockServer::start();
    let mock_free = MockServer::start();

    let capped = setup_ok_mock(&mock_capped, "from-capped", "key-capped");
    let free = setup_ok_mock(&mock_free, "from-free", "key-free");

    let mut capped_entry = model_entry("gpt-4o", &mock_capped.base_url(), "key-capped");
    // Tiny-but-positive cap: the capped deployment serves at most once
    // before its spend counter exceeds the budget.
    capped_entry.oxllm_params.max_budget = Some(1e-9);
    let free_entry = model_entry("gpt-4o", &mock_free.base_url(), "key-free");

    let config = build_config(vec![capped_entry, free_entry], OxllmSettings::default());
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let mut capped_responses = 0;
    for i in 0..5 {
        let resp = send_request(&handle, "gpt-4o").await;
        assert_eq!(resp.status(), 200, "request {i} must succeed via failover");
        let body: Value = resp.json().await.expect("json");
        let content = body["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("");
        if content == "from-capped" {
            capped_responses += 1;
        } else {
            assert_eq!(content, "from-free", "request {i} served by: {content}");
        }
    }
    assert!(
        capped_responses <= 1,
        "over-budget deployment must serve at most once, got {capped_responses} responses"
    );
    assert!(
        capped.hits() <= 1,
        "over-budget deployment must be called at most once, got {} hits",
        capped.hits()
    );
    assert_eq!(
        free.hits() + capped.hits(),
        5,
        "every request lands upstream exactly once (free: {}, capped: {})",
        free.hits(),
        capped.hits()
    );

    let _ = handle.shutdown_tx.send(());
}

/// No budgets configured: routing is unaffected (existing behavior). This is
/// covered by every other test in this file; here we pin it explicitly with
/// a deployment that would trip a zero cap if one were implied.
#[tokio::test]
async fn test_no_budget_means_no_limit() {
    let mock = MockServer::start();
    let ok = setup_ok_mock(&mock, "from-dep", "key");

    let config = build_config(
        vec![model_entry("gpt-4o", &mock.base_url(), "key")],
        OxllmSettings::default(),
    );
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    for _ in 0..3 {
        let resp = send_request(&handle, "gpt-4o").await;
        assert_eq!(resp.status(), 200);
    }
    ok.assert_hits(3);

    let _ = handle.shutdown_tx.send(());
}

// ---------------------------------------------------------------------------
// Health-check routing — Batch 2
// ---------------------------------------------------------------------------

/// With health-check routing enabled, a deployment that refuses connections
/// is marked unhealthy after `health_check_failure_threshold` consecutive
/// connection failures.
#[tokio::test]
async fn test_health_check_marks_dead_deployment_unhealthy() {
    // Dead deployment: nothing listens on 127.0.0.1:1, so every attempt is
    // a fast connection-refused (a Network error in the orchestrator).
    let mut dead_entry = model_entry("gpt-4o", "http://127.0.0.1:1", "key-dead");
    dead_entry.oxllm_params.timeout = Some(2.0);

    let mut config = build_config(
        vec![dead_entry],
        OxllmSettings {
            routing: RoutingKnobs {
                // No retries: keeps the failure fast; the deployment still
                // cools after the first attempt.
                num_retries: Some(0),
                ..Default::default()
            },
            ..Default::default()
        },
    );
    config.router_settings.enable_health_check_routing = Some(true);
    config.router_settings.health_check_failure_threshold = Some(1);
    let (handle, shared) = start_oxllm(config).await.expect("server should start");

    let resp = send_request(&handle, "gpt-4o").await;
    // A reachability failure is an upstream-side problem: 502 Bad Gateway,
    // not a proxy-internal 500.
    assert_eq!(resp.status(), 502, "network failure surfaces to the client");

    let state = shared.load_full();
    assert!(
        !state.router.health_store().is_healthy(
            &state.router.cooldown_key(&state.deployments[0]),
            std::time::Duration::from_secs(60)
        ),
        "the dead deployment must be marked unhealthy after the failure threshold"
    );

    let _ = handle.shutdown_tx.send(());
}

/// With health-check routing enabled, a dead deployment in a pair is
/// filtered out once marked unhealthy; every request is served by the
/// healthy sibling (failover on the first failing attempt).
#[tokio::test]
async fn test_health_check_routing_skips_dead_deployment() {
    let mock_ok = MockServer::start();
    let ok = setup_ok_mock(&mock_ok, "from-healthy", "key-ok");

    let mut dead_entry = model_entry("gpt-4o", "http://127.0.0.1:1", "key-dead");
    dead_entry.oxllm_params.timeout = Some(2.0);
    let ok_entry = model_entry("gpt-4o", &mock_ok.base_url(), "key-ok");

    let mut config = build_config(
        vec![dead_entry, ok_entry],
        OxllmSettings {
            routing: RoutingKnobs {
                num_retries: Some(0),
                ..Default::default()
            },
            ..Default::default()
        },
    );
    config.router_settings.enable_health_check_routing = Some(true);
    config.router_settings.health_check_failure_threshold = Some(1);
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    for i in 0..5 {
        let resp = send_request(&handle, "gpt-4o").await;
        assert_eq!(
            resp.status(),
            200,
            "request {i} must succeed via the healthy deployment"
        );
        let body: Value = resp.json().await.expect("json");
        assert_eq!(body["choices"][0]["message"]["content"], "from-healthy");
    }
    assert_eq!(
        ok.hits(),
        5,
        "every request lands on the healthy deployment"
    );

    let _ = handle.shutdown_tx.send(());
}

/// Health-check routing disabled (default): a dead deployment stays in
/// rotation and its failures surface to the client — existing behavior is
/// preserved.
#[tokio::test]
async fn test_health_check_disabled_keeps_dead_deployment_in_rotation() {
    let mut dead_entry = model_entry("gpt-4o", "http://127.0.0.1:1", "key-dead");
    dead_entry.oxllm_params.timeout = Some(2.0);

    let config = build_config(
        vec![dead_entry],
        OxllmSettings {
            routing: RoutingKnobs {
                num_retries: Some(0),
                ..Default::default()
            },
            ..Default::default()
        },
    );
    let (handle, shared) = start_oxllm(config).await.expect("server should start");

    let resp = send_request(&handle, "gpt-4o").await;
    // A reachability failure is an upstream-side problem: 502 Bad Gateway,
    // not a proxy-internal 500.
    assert_eq!(resp.status(), 502, "network failure surfaces to the client");

    let state = shared.load_full();
    assert!(
        state.router.health_store().is_healthy(
            &state.router.cooldown_key(&state.deployments[0]),
            std::time::Duration::from_secs(60)
        ),
        "with health-check routing disabled no health state may be recorded"
    );

    let _ = handle.shutdown_tx.send(());
}

/// H1 regression: a *locally-synthesized* timeout (the upstream never
/// produced a response) must degrade the deployment's health like any other
/// connection failure. Previously a timeout was classified as
/// `Retryable{504}` and fell into `record_connection_success`, deleting the
/// health entry — a hung deployment could never become unhealthy.
#[tokio::test]
async fn test_health_check_marks_hung_deployment_unhealthy() {
    let mock_hung = MockServer::start();
    // Hung deployment: 200 but delayed well past the per-deployment
    // timeout, so every attempt ends in a client-side timeout (no response
    // ever arrives).
    mock_hung.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .header("authorization", "Bearer key-hung");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(success_body("too-late"))
            .delay(std::time::Duration::from_secs(5));
    });

    let mut hung_entry = model_entry("gpt-4o", &mock_hung.base_url(), "key-hung");
    hung_entry.oxllm_params.timeout = Some(0.2); // 200 ms — always times out

    let mut config = build_config(
        vec![hung_entry],
        OxllmSettings {
            routing: RoutingKnobs {
                num_retries: Some(0),
                ..Default::default()
            },
            ..Default::default()
        },
    );
    config.router_settings.enable_health_check_routing = Some(true);
    config.router_settings.health_check_failure_threshold = Some(1);
    let (handle, shared) = start_oxllm(config).await.expect("server should start");

    let resp = send_request(&handle, "gpt-4o").await;
    assert_eq!(resp.status(), 504, "client-side timeout surfaces as 504");

    let state = shared.load_full();
    assert!(
        !state.router.health_store().is_healthy(
            &state.router.cooldown_key(&state.deployments[0]),
            std::time::Duration::from_secs(60)
        ),
        "a hung deployment (local timeout, no response) must be marked unhealthy"
    );

    let _ = handle.shutdown_tx.send(());
}

/// H1 counterpart: a *genuine* upstream 504 (an HTTP response arrived)
/// proves reachability, so it must NOT degrade the deployment's health —
/// the failure count resets via `record_connection_success`, matching the
/// rule that any HTTP response proves connectivity.
#[tokio::test]
async fn test_health_check_genuine_upstream_504_stays_healthy() {
    let mock_504 = MockServer::start();
    let _m504 = setup_fail_mock(&mock_504, 504, "key-504");

    let entry = model_entry("gpt-4o", &mock_504.base_url(), "key-504");

    let mut config = build_config(
        vec![entry],
        OxllmSettings {
            routing: RoutingKnobs {
                num_retries: Some(0),
                ..Default::default()
            },
            ..Default::default()
        },
    );
    config.router_settings.enable_health_check_routing = Some(true);
    config.router_settings.health_check_failure_threshold = Some(1);
    let (handle, shared) = start_oxllm(config).await.expect("server should start");

    let resp = send_request(&handle, "gpt-4o").await;
    assert_eq!(
        resp.status(),
        504,
        "the upstream 504 surfaces to the client"
    );

    let state = shared.load_full();
    assert!(
        state.router.health_store().is_healthy(
            &state.router.cooldown_key(&state.deployments[0]),
            std::time::Duration::from_secs(60)
        ),
        "a genuine upstream 504 proves reachability and must not degrade health"
    );

    let _ = handle.shutdown_tx.send(());
}

/// Flapping: a deployment whose health checks fail is marked unhealthy and
/// filtered out; once it recovers (and its cooldown + stale health entry
/// expire) it re-enters rotation and serves traffic again.
#[tokio::test]
async fn test_health_recovers_and_receives_traffic() {
    let mock_flap = MockServer::start();
    let mock_ok = MockServer::start();

    // Flapping deployment, initially hung: 200 but delayed well past the
    // per-deployment timeout, so every attempt ends in a client-side
    // timeout, which degrades connectivity health (H1).
    let hung = mock_flap.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .header("authorization", "Bearer key-flap");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(success_body("too-late"))
            .delay(std::time::Duration::from_secs(5));
    });
    let _ok = setup_ok_mock(&mock_ok, "from-healthy", "key-ok");

    let mut flap_entry = model_entry("gpt-4o", &mock_flap.base_url(), "key-flap");
    flap_entry.oxllm_params.timeout = Some(0.2); // 200 ms — hung always times out
    let ok_entry = model_entry("gpt-4o", &mock_ok.base_url(), "key-ok");

    let mut config = build_config(
        vec![flap_entry, ok_entry],
        OxllmSettings {
            routing: RoutingKnobs {
                num_retries: Some(0),
                // One-second cooldown so the recovered deployment re-enters
                // rotation quickly once its health entry goes stale.
                cooldown_time: Some(1),
                ..Default::default()
            },
            ..Default::default()
        },
    );
    config.router_settings.enable_health_check_routing = Some(true);
    config.router_settings.health_check_failure_threshold = Some(1);
    config.router_settings.health_check_staleness_threshold_secs = Some(1);
    let (handle, shared) = start_oxllm(config).await.expect("server should start");

    // Phase 1: the hung deployment is marked unhealthy after its first
    // attempt; every request fails over to the healthy sibling.
    let state = shared.load_full();
    let mut marked_unhealthy = false;
    for _ in 0..20 {
        let resp = send_request(&handle, "gpt-4o").await;
        assert_eq!(resp.status(), 200, "the healthy sibling must serve");
        let body: Value = resp.json().await.expect("json");
        assert_eq!(body["choices"][0]["message"]["content"], "from-healthy");
        if !state.router.health_store().is_healthy(
            &state.router.cooldown_key(&state.deployments[0]),
            std::time::Duration::from_secs(60),
        ) {
            marked_unhealthy = true;
            break;
        }
    }
    assert!(
        marked_unhealthy,
        "the hung deployment must be marked unhealthy within 20 requests"
    );

    // Phase 2: the deployment recovers — the hung mock is swapped for a
    // healthy one on the same upstream — and the cooldown + health
    // staleness window elapses, making it a candidate again.
    hung.delete_async().await;
    let flap_ok = setup_ok_mock(&mock_flap, "from-flap", "key-flap");
    // Wait for the 1s cooldown to elapse instead of a fixed sleep with a
    // ~500ms margin that can flake on slow CI.
    let flap_key = cooldown_key_of(&shared, 0).await;
    wait_until_not_cooling(&shared, &flap_key).await;

    // Phase 3: the recovered deployment re-enters rotation, serves traffic,
    // and the success clears its health entry.
    let mut flap_served = 0;
    for _ in 0..20 {
        let resp = send_request(&handle, "gpt-4o").await;
        assert_eq!(resp.status(), 200);
        let body: Value = resp.json().await.expect("json");
        if body["choices"][0]["message"]["content"] == "from-flap" {
            flap_served += 1;
        }
    }
    assert!(
        flap_served >= 1,
        "recovered deployment must receive traffic again (flap hits: {})",
        flap_ok.hits()
    );
    assert!(
        state.router.health_store().is_healthy(
            &state.router.cooldown_key(&state.deployments[0]),
            std::time::Duration::from_secs(60)
        ),
        "a successful call after recovery must clear the health entry"
    );

    let _ = handle.shutdown_tx.send(());
}

// ---------------------------------------------------------------------------
// Priority-order tiers (oxllm_params.order)
// ---------------------------------------------------------------------------

fn model_entry_ordered(name: &str, base: &str, key: &str, order: u32) -> ConfigModel {
    let mut p = params(base, key);
    p.order = Some(order);
    ConfigModel {
        model_name: name.to_string(),
        oxllm_params: p,
        ..Default::default()
    }
}

#[tokio::test]
async fn test_order_tiers_prefer_lower_order() {
    let mock_low = MockServer::start();
    let mock_high = MockServer::start();

    let high_priority = setup_ok_mock(&mock_high, "from-order-1", "key-high");
    let low_priority = setup_ok_mock(&mock_low, "from-order-2", "key-low");

    let config = build_config(
        vec![
            model_entry_ordered("gpt-4o", &mock_low.base_url(), "key-low", 2),
            model_entry_ordered("gpt-4o", &mock_high.base_url(), "key-high", 1),
        ],
        OxllmSettings::default(),
    );
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    for _ in 0..20 {
        let resp = send_request(&handle, "gpt-4o").await;
        assert_eq!(resp.status(), 200);
        let body: Value = resp.json().await.expect("json");
        assert_eq!(
            body["choices"][0]["message"]["content"], "from-order-1",
            "lower order = higher priority: every request must hit the order-1 deployment"
        );
    }
    assert!(
        high_priority.hits() >= 20,
        "order-1 deployment must serve all traffic"
    );
    assert_eq!(
        low_priority.hits(),
        0,
        "order-2 deployment must be idle while the order-1 tier is healthy"
    );

    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn test_order_tiers_escalate_when_top_tier_fails() {
    let mock_low = MockServer::start();
    let mock_high = MockServer::start();

    let _high_fail = setup_fail_mock(&mock_high, 500, "key-high");
    let _low_ok = setup_ok_mock(&mock_low, "from-order-2", "key-low");

    let config = build_config(
        vec![
            model_entry_ordered("gpt-4o", &mock_high.base_url(), "key-high", 1),
            model_entry_ordered("gpt-4o", &mock_low.base_url(), "key-low", 2),
        ],
        OxllmSettings {
            routing: RoutingKnobs {
                num_retries: Some(1),
                ..Default::default()
            },
            ..Default::default()
        },
    );
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    // The first request burns the order-1 tier's retry budget, cools it,
    // and escalates to the order-2 tier within the same request.
    for _ in 0..10 {
        let resp = send_request(&handle, "gpt-4o").await;
        assert_eq!(
            resp.status(),
            200,
            "once the order-1 tier cools, the order-2 tier must serve"
        );
        let body: Value = resp.json().await.expect("json");
        assert_eq!(body["choices"][0]["message"]["content"], "from-order-2");
    }

    let _ = handle.shutdown_tx.send(());
}

// ---------------------------------------------------------------------------
// default_fallbacks (unknown model group)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_default_fallbacks_unknown_model_serves_default() {
    let mock = MockServer::start();
    let _ok = setup_ok_mock(&mock, "from-default", "key-a");

    let config = build_config(
        vec![model_entry("model-a", &mock.base_url(), "key-a")],
        OxllmSettings {
            routing: RoutingKnobs {
                default_fallbacks: Some(json!(["model-a"])),
                ..Default::default()
            },
            ..Default::default()
        },
    );
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let resp = send_request(&handle, "unknown-model").await;
    assert_eq!(
        resp.status(),
        200,
        "an unknown model group must be rerouted via default_fallbacks"
    );
    assert_eq!(
        resp.headers()
            .get("x-oxllm-fallback-used")
            .and_then(|v| v.to_str().ok()),
        Some("model-a")
    );
    let body: Value = resp.json().await.expect("json");
    assert_eq!(body["choices"][0]["message"]["content"], "from-default");

    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn test_default_fallbacks_walks_chain_to_first_reachable() {
    let mock = MockServer::start();
    let _ok = setup_ok_mock(&mock, "from-model-b", "key-b");

    // "ghost" has no deployments; the chain must skip it and land on
    // "model-b".
    let config = build_config(
        vec![model_entry("model-b", &mock.base_url(), "key-b")],
        OxllmSettings {
            routing: RoutingKnobs {
                default_fallbacks: Some(json!(["ghost", "model-b"])),
                ..Default::default()
            },
            ..Default::default()
        },
    );
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let resp = send_request(&handle, "unknown-model").await;
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("json");
    assert_eq!(body["choices"][0]["message"]["content"], "from-model-b");

    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn test_default_fallbacks_not_used_for_known_model() {
    let mock_primary = MockServer::start();
    let mock_default = MockServer::start();

    let _primary_fail = setup_fail_mock(&mock_primary, 500, "pk");
    let default_ok = setup_ok_mock(&mock_default, "from-default", "dk");

    // "gpt-4o" exists but always fails; default_fallbacks must NOT engage
    // (that is what `fallbacks` is for), so the request errors out and the
    // default model stays idle.
    let config = build_config(
        vec![
            model_entry("gpt-4o", &mock_primary.base_url(), "pk"),
            model_entry("model-a", &mock_default.base_url(), "dk"),
        ],
        OxllmSettings {
            routing: RoutingKnobs {
                num_retries: Some(1),
                default_fallbacks: Some(json!(["model-a"])),
                ..Default::default()
            },
            ..Default::default()
        },
    );
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let resp = send_request(&handle, "gpt-4o").await;
    assert_eq!(
        resp.status(),
        500,
        "a KNOWN model that fails must surface its own error, not default_fallbacks"
    );
    assert_eq!(
        default_ok.hits(),
        0,
        "default fallback deployment must stay idle for known models"
    );

    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn test_unknown_model_without_default_fallbacks_still_404() {
    let mock = MockServer::start();
    let _ok = setup_ok_mock(&mock, "from-model-a", "key-a");

    let config = build_config(
        vec![model_entry("model-a", &mock.base_url(), "key-a")],
        OxllmSettings::default(),
    );
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let resp = send_request(&handle, "unknown-model").await;
    assert_eq!(
        resp.status(),
        404,
        "no default_fallbacks configured → the model-not-found error is preserved"
    );
    let body: Value = resp.json().await.expect("json");
    assert_eq!(body["error"]["code"], "model_not_found");

    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn test_default_fallbacks_exhausted_chain_preserves_404() {
    let mock = MockServer::start();
    let _ok = setup_ok_mock(&mock, "from-model-a", "key-a");

    // Chain only names a model with no deployments → exhausted → original
    // model-not-found error.
    let config = build_config(
        vec![model_entry("model-a", &mock.base_url(), "key-a")],
        OxllmSettings {
            routing: RoutingKnobs {
                default_fallbacks: Some(json!(["ghost"])),
                ..Default::default()
            },
            ..Default::default()
        },
    );
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let resp = send_request(&handle, "unknown-model").await;
    assert_eq!(resp.status(), 404);
    let body: Value = resp.json().await.expect("json");
    assert_eq!(body["error"]["code"], "model_not_found");

    let _ = handle.shutdown_tx.send(());
}

// ---------------------------------------------------------------------------
// Adaptive router (Thompson-sampling bandit)
// ---------------------------------------------------------------------------

/// Router settings for the adaptive e2e tests: bandit engaged from the
/// first request (no cold start), no retries (the failing deployment
/// rotates out immediately instead of burning attempts), 1s cooldown.
fn adaptive_config(epsilon: f64, cold_start: u32, model_list: Vec<ConfigModel>) -> ProxyConfig {
    let mut config = build_config(
        model_list,
        OxllmSettings {
            routing: RoutingKnobs {
                routing_strategy: Some("adaptive_router".to_string()),
                num_retries: Some(0),
                cooldown_time: Some(1),
                ..Default::default()
            },
            ..Default::default()
        },
    );
    config.router_settings.adaptive_router_config =
        Some(oxllm_core::router::AdaptiveRouterConfig {
            quality_weight: 1.0,
            cost_weight: 0.0,
            epsilon,
            cold_start_min_observations: cold_start,
        });
    config
}

#[tokio::test]
async fn test_adaptive_router_shifts_traffic_to_healthy_deployment() {
    let mock_bad = MockServer::start();
    let mock_good = MockServer::start();
    let bad = setup_fail_mock(&mock_bad, 500, "key-bad");
    let good = setup_ok_mock(&mock_good, "from-good", "key-good");

    let config = adaptive_config(
        0.05,
        0,
        vec![
            model_entry("gpt-4o", &mock_bad.base_url(), "key-bad"),
            model_entry("gpt-4o", &mock_good.base_url(), "key-good"),
        ],
    );
    let (handle, shared) = start_oxllm(config).await.expect("server should start");

    // Warm-up: every request succeeds (the bad deployment fails over to the
    // good one), and each success strengthens the good deployment's
    // posterior via the learning hook.
    for i in 0..30 {
        let resp = send_request(&handle, "gpt-4o").await;
        assert_eq!(resp.status(), 200, "warm-up request {i} must succeed");
    }

    let good_before = good.hits();
    let bad_before = bad.hits();

    // Measurement window: with the good cell at ~Beta(31, 1) vs the bad
    // cell's Beta(1, 1) prior, the bandit should nearly always pick the
    // good deployment first (expected bad first-picks ≈ 5-6%: Thompson
    // loss rate ~1/32 plus ε/2 exploration).
    const WINDOW: usize = 40;
    for i in 0..WINDOW {
        let resp = send_request(&handle, "gpt-4o").await;
        assert_eq!(resp.status(), 200, "request {i} must succeed");
    }
    let good_delta = good.hits() - good_before;
    let bad_delta = bad.hits() - bad_before;
    assert_eq!(
        good_delta, WINDOW,
        "every request must land on the healthy deployment"
    );
    assert!(
        bad_delta <= WINDOW / 4,
        "traffic must shift away from the failing deployment ({bad_delta}/{WINDOW} hits)"
    );

    // The bandit learned the good deployment's successes: the request text
    // ("hi") classifies as Other, and the good deployment is index 1.
    let state = shared.load_full();
    let cell = (
        oxllm_core::router::RequestType::Other,
        state.router.cooldown_key(&state.deployments[1]),
    );
    assert!(
        state.router.bandit_store().observations(&cell) >= 70.0,
        "the learning hook must accumulate successes for the serving deployment"
    );

    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn test_adaptive_router_records_failures_in_bandit_store() {
    let mock_bad = MockServer::start();
    let _bad = setup_fail_mock(&mock_bad, 500, "key-bad");

    // A single always-500 deployment: every request fails after the pick,
    // so the cell's beta grows. Each failed request debits EXACTLY ONCE —
    // the per-pick debit fires at the cooling point when the deployment's
    // retries are exhausted, and the stamped cell is cleared there so the
    // once-per-request `post_failure` hook skips (unstamped no-op path).
    // The 1s cooldown must expire between requests or the deployment is
    // not pickable (no pick → no stamped cell → nothing recorded).
    let config = adaptive_config(
        0.0,
        0,
        vec![model_entry("gpt-4o", &mock_bad.base_url(), "key-bad")],
    );
    let (handle, shared) = start_oxllm(config).await.expect("server should start");
    let key = cooldown_key_of(&shared, 0).await;

    for i in 0..2 {
        let resp = send_request(&handle, "gpt-4o").await;
        assert_eq!(resp.status(), 500, "request {i} must fail");
        if i == 0 {
            wait_until_not_cooling(&shared, &key).await;
        }
    }

    let state = shared.load_full();
    let cell = (oxllm_core::router::RequestType::Other, key.clone());
    let snapshot = state.router.bandit_store().sample_snapshot(&[cell]);
    assert_eq!(snapshot[0].alpha, 1.0, "no successes recorded");
    assert_eq!(
        snapshot[0].beta, 3.0,
        "two failed requests debit once each (prior 1 + 2 × per-pick cooling debit)"
    );

    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn test_adaptive_router_debits_intermediate_failures() {
    let mock_bad = MockServer::start();
    let mock_good = MockServer::start();
    let _bad = setup_fail_mock(&mock_bad, 500, "key-bad");
    let _good = setup_ok_mock(&mock_good, "from-good", "key-good");

    // Priority tiers make the failing deployment the deterministic first
    // pick (order 0 serves before order 1 while both are available): every
    // request tries the bad deployment first and fails over to the good
    // one. The per-pick failure debit must grow the bad cell's beta even
    // though the once-per-request hook only ever credits the serving
    // (good) deployment — before per-pick debits, a failing-but-always-
    // rescued deployment learned nothing and its cell never left cold
    // start.
    let config = adaptive_config(
        0.0,
        2,
        vec![
            model_entry_ordered("gpt-4o", &mock_bad.base_url(), "key-bad", 0),
            model_entry_ordered("gpt-4o", &mock_good.base_url(), "key-good", 1),
        ],
    );
    let (handle, shared) = start_oxllm(config).await.expect("server should start");
    let bad_key = cooldown_key_of(&shared, 0).await;
    let good_key = cooldown_key_of(&shared, 1).await;

    const REQUESTS: usize = 3;
    for i in 0..REQUESTS {
        let resp = send_request(&handle, "gpt-4o").await;
        assert_eq!(
            resp.status(),
            200,
            "request {i} must be rescued by the healthy deployment"
        );
        let body: Value = resp.json().await.expect("json");
        assert_eq!(body["choices"][0]["message"]["content"], "from-good");
        // Wait out the 1s cooldown so the bad deployment is pickable again.
        wait_until_not_cooling(&shared, &bad_key).await;
    }

    let state = shared.load_full();
    let bad_cell = (oxllm_core::router::RequestType::Other, bad_key);
    let good_cell = (oxllm_core::router::RequestType::Other, good_key);
    let snapshot = state
        .router
        .bandit_store()
        .sample_snapshot(&[bad_cell.clone(), good_cell.clone()]);
    assert_eq!(
        snapshot[0].beta,
        1.0 + REQUESTS as f64,
        "every rescued failure must debit the failing deployment's cell"
    );
    assert_eq!(
        snapshot[0].alpha, 1.0,
        "the failing deployment never serves a request"
    );
    assert_eq!(
        snapshot[1].alpha,
        1.0 + REQUESTS as f64,
        "the serving deployment keeps its per-request success credits"
    );
    assert_eq!(snapshot[1].beta, 1.0);
    // Cold start (min_observations = 2) requires EVERY arm warm: with
    // per-pick debits the failing arm accrues observations too, so the
    // group exits cold start for this request type.
    for (name, cell) in [("bad", &bad_cell), ("good", &good_cell)] {
        assert!(
            state.router.bandit_store().observations(cell) >= 2.0,
            "the {name} arm must warm out of cold start"
        );
    }

    let _ = handle.shutdown_tx.send(());
}

/// Router settings for the mixed-strategy adaptive tests: a non-adaptive
/// global strategy, one adaptive routing group, and a fallback chain from
/// the primary to the fallback model.
fn mixed_strategy_config(adaptive_group: &str, model_list: Vec<ConfigModel>) -> ProxyConfig {
    let mut config = build_config(
        model_list,
        OxllmSettings {
            routing: RoutingKnobs {
                num_retries: Some(0),
                fallbacks: Some(json!([{"gpt-4o": ["gpt-4o-fb"]}])),
                ..Default::default()
            },
            ..Default::default()
        },
    );
    config.router_settings.routing_groups = Some(std::collections::HashMap::from([(
        adaptive_group.to_string(),
        RoutingGroupConfig {
            routing: RoutingKnobs {
                routing_strategy: Some("adaptive_router".to_string()),
                ..Default::default()
            },
        },
    )]));
    config.router_settings.adaptive_router_config =
        Some(oxllm_core::router::AdaptiveRouterConfig {
            cold_start_min_observations: 0,
            ..Default::default()
        });
    config
}

#[tokio::test]
async fn test_adaptive_primary_non_adaptive_fallback_stamps_nothing() {
    let mock_primary = MockServer::start();
    let mock_fallback = MockServer::start();
    let _primary_fail = setup_fail_mock(&mock_primary, 500, "pk");
    let _fallback_ok = setup_ok_mock(&mock_fallback, "from-fallback", "fk");

    // The primary's group is adaptive; the fallback's group is NOT. When
    // the primary fails and the fallback serves, the fallback's success
    // must not be credited to a bandit cell (the bandit never sampled one
    // there — the non-adaptive pick clears the stamp) — only the primary's
    // own cell learns, via the per-pick failure debit at the cooling
    // point.
    let config = mixed_strategy_config(
        "gpt-4o",
        vec![
            model_entry("gpt-4o", &mock_primary.base_url(), "pk"),
            model_entry("gpt-4o-fb", &mock_fallback.base_url(), "fk"),
        ],
    );
    let (handle, shared) = start_oxllm(config).await.expect("server should start");

    let resp = send_request(&handle, "gpt-4o").await;
    assert_eq!(resp.status(), 200, "the fallback must rescue the request");
    let body: Value = resp.json().await.expect("json");
    assert_eq!(body["choices"][0]["message"]["content"], "from-fallback");

    // The cell keys are the deployments' stable cooldown keys: the fallback
    // deployment is the second model_list entry → index 1.
    let state = shared.load_full();
    let primary_cell = (
        oxllm_core::router::RequestType::Other,
        state.router.cooldown_key(&state.deployments[0]),
    );
    let fallback_cell = (
        oxllm_core::router::RequestType::Other,
        state.router.cooldown_key(&state.deployments[1]),
    );
    let snapshot = state
        .router
        .bandit_store()
        .sample_snapshot(&[primary_cell, fallback_cell]);
    assert_eq!(
        snapshot[0].beta, 2.0,
        "the adaptive primary's exhausted retries debit its own cell"
    );
    assert_eq!(
        snapshot[0].alpha, 1.0,
        "the failed primary must not be credited for the rescued request"
    );
    assert_eq!(
        snapshot[1],
        oxllm_core::router::BetaCell::default(),
        "a non-adaptive leg's outcome must never touch the bandit"
    );

    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn test_non_adaptive_primary_adaptive_fallback_leg_learns() {
    let mock_primary = MockServer::start();
    let mock_fallback = MockServer::start();
    let _primary_fail = setup_fail_mock(&mock_primary, 500, "pk");
    let _fallback_ok = setup_ok_mock(&mock_fallback, "from-fallback", "fk");

    // The reverse direction: the primary is non-adaptive and the fallback
    // group resolves to the adaptive router. `ChatOrchestrator::run`
    // classifies the request once (adaptive is configured somewhere) and
    // the adaptive fallback leg reuses that classification — its success
    // must warm the fallback deployment's cell while the primary's
    // non-adaptive failure leaves no bandit trace.
    let config = mixed_strategy_config(
        "gpt-4o-fb",
        vec![
            model_entry("gpt-4o", &mock_primary.base_url(), "pk"),
            model_entry("gpt-4o-fb", &mock_fallback.base_url(), "fk"),
        ],
    );
    let (handle, shared) = start_oxllm(config).await.expect("server should start");

    let resp = send_request(&handle, "gpt-4o").await;
    assert_eq!(resp.status(), 200, "the fallback must rescue the request");
    let body: Value = resp.json().await.expect("json");
    assert_eq!(body["choices"][0]["message"]["content"], "from-fallback");

    // The cell keys are the deployments' stable cooldown keys: the fallback
    // deployment is the second model_list entry → index 1.
    let state = shared.load_full();
    let primary_cell = (
        oxllm_core::router::RequestType::Other,
        state.router.cooldown_key(&state.deployments[0]),
    );
    let fallback_cell = (
        oxllm_core::router::RequestType::Other,
        state.router.cooldown_key(&state.deployments[1]),
    );
    let snapshot = state
        .router
        .bandit_store()
        .sample_snapshot(&[primary_cell, fallback_cell]);
    assert_eq!(
        snapshot[0],
        oxllm_core::router::BetaCell::default(),
        "a non-adaptive leg's failure must not debit the bandit"
    );
    assert_eq!(
        snapshot[1].alpha, 2.0,
        "the adaptive fallback leg's success must credit its cell"
    );
    assert_eq!(snapshot[1].beta, 1.0);

    let _ = handle.shutdown_tx.send(());
}
