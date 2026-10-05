//! Integration tests: core chat completions lifecycle (httpmock upstream).

mod common;

use common::start_oxllm;
use httpmock::{Mock, MockServer};
use oxllm_server::OxllmParams;
use oxllm_server::config::types::{
    ConfigModel, GeneralSettings, OxllmSettings, ProxyConfig, RoutingKnobs,
};
use serde_json::{Value, json};

fn make_proxy_config(mock_base_url: &str, master_key: Option<&str>) -> ProxyConfig {
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
        oxllm_settings: OxllmSettings::default(),
        general_settings: GeneralSettings {
            master_key: master_key.map(String::from),
            allow_private_api_base: Some(true),
            ..Default::default()
        },
        router_settings: Default::default(),
        extra: serde_json::Map::new(),
    }
}

fn setup_mock_non_stream(mock: &MockServer) -> Mock<'_> {
    mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .header("authorization", "Bearer sk-test-key");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
            "id": "chatcmpl-test-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4o",
            "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from the mock!"},
            "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
            }));
    })
}

fn setup_mock_stream(mock: &MockServer) -> Mock<'_> {
    let sse_body = concat!(
        "data: {\"id\":\"chatcmpl-stream-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"chatcmpl-stream-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"chatcmpl-stream-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"!\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"chatcmpl-stream-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
        "data: [DONE]\n\n"
    );
    mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .header("authorization", "Bearer sk-test-key")
            .json_body_partial(json!({"stream": true}).to_string());
        then.status(200)
            .header("content-type", "text/event-stream")
            .body(sse_body);
    })
}

fn make_request_body(model: &str, stream: bool) -> Value {
    json!({
    "model": model,
    "messages": [{"role": "user", "content": "Say hello"}],
    "stream": stream
    })
}

#[tokio::test]
async fn checked_tag_rejections_stop_chat_and_embeddings_fallbacks() {
    let mock = MockServer::start();
    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST);
        then.status(200).header("content-type", "application/json").json_body(json!({
            "choices":[{"message":{"role":"assistant","content":"unexpected"},"finish_reason":"stop"}],
            "data":[{"embedding":[0.1],"index":0}],"usage":{"prompt_tokens":1,"total_tokens":1}
        }));
    });
    for (path, primary_cooling) in [
        ("/v1/chat/completions", false),
        ("/v1/embeddings", false),
        ("/openai/v1/chat/completions", false),
        ("/v1/chat/completions", true),
        ("/v1/embeddings", true),
    ] {
        let mut config = make_proxy_config(&mock.base_url(), None);
        config.model_list[0].model_name = "primary".to_string();
        config.model_list[0].model_info = Some(oxllm_core::deployment::ModelInfo {
            tags: vec![
                if primary_cooling {
                    "allowed"
                } else {
                    "blocked"
                }
                .to_string(),
            ],
            ..Default::default()
        });
        let mut fallback = config.model_list[0].clone();
        fallback.model_name = "fallback".to_string();
        fallback.model_info.as_mut().unwrap().tags = vec!["blocked".to_string()];
        let mut last = fallback.clone();
        last.model_name = "last".to_string();
        last.model_info.as_mut().unwrap().tags = vec!["allowed".to_string()];
        config.model_list.extend([fallback, last]);
        config.router_settings = serde_json::from_value(json!({
            "routing_strategy":"tag_based", "num_retries":0,
            "fallbacks":[{"primary":["fallback","last"]}],
            "routing_groups":{"fallback":{"routing_strategy":"round_robin"}}
        }))
        .unwrap();
        let (handle, shared) = start_oxllm(config).await.unwrap();
        if primary_cooling {
            let state = shared.load_full();
            state
                .router
                .mark_cooldown(&state.router.cooldown_key(&state.deployments[0]));
        }
        let response = common::http_client()
            .post(handle.url(path))
            .json(&json!({
                "model":"primary", "messages":[{"role":"user","content":"hi"}], "input":"hi",
                "metadata":{"tags":["&allowed"]}
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(
            response.status(),
            400,
            "{path}, primary cooling={primary_cooling}"
        );
        let error: Value = response.json().await.unwrap();
        assert_eq!(error["error"]["code"], "routing_error");
        handle.shutdown();
    }
    upstream.assert_hits(0);
}

#[tokio::test]
async fn test_non_streaming_chat() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), None);
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let mock_mock = setup_mock_non_stream(&mock);

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .json(&make_request_body("gpt-4o", false))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("body should be JSON");
    assert_eq!(body["id"], "chatcmpl-test-123");
    assert_eq!(body["object"], "chat.completion");
    assert_eq!(
        body["choices"][0]["message"]["content"],
        "Hello from the mock!"
    );

    mock_mock.assert_hits(1);
    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn test_streaming_chat() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), None);
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let mock_mock = setup_mock_stream(&mock);

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .json(&make_request_body("gpt-4o", true))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);
    let text = resp.text().await.expect("body should be text");

    assert!(
        text.contains("chat.completion.chunk"),
        "should contain chunk data: {text}"
    );
    assert!(
        text.contains("\"content\":\"Hi\""),
        "should contain Hi delta: {text}"
    );
    assert!(
        text.contains("\"content\":\"!\""),
        "should contain ! delta: {text}"
    );
    assert!(
        text.contains("\"finish_reason\":\"stop\""),
        "should contain stop: {text}"
    );
    assert!(text.contains("[DONE]"), "should contain [DONE]: {text}");

    mock_mock.assert_hits(1);
    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn test_auth_no_key_returns_401() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), Some("sk-master-secret"));
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .json(&make_request_body("gpt-4o", false))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 401);
    let body: Value = resp.json().await.expect("body should be JSON");
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert_eq!(body["error"]["code"], "invalid_api_key");

    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn test_auth_wrong_key_returns_401() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), Some("sk-master-secret"));
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .header("authorization", "Bearer sk-wrong-key")
        .json(&make_request_body("gpt-4o", false))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 401);
    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn test_auth_correct_key_returns_200() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), Some("sk-master-secret"));
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let _mock_mock = setup_mock_non_stream(&mock);

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .header("authorization", "Bearer sk-master-secret")
        .json(&make_request_body("gpt-4o", false))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);
    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn test_open_mode_no_auth_required() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), None);
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let _mock_mock = setup_mock_non_stream(&mock);

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .json(&make_request_body("gpt-4o", false))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);
    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn test_list_models() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), None);
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let client = reqwest::Client::new();
    let resp = client
        .get(handle.url("/v1/models"))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("body should be JSON");
    assert_eq!(body["object"], "list");
    assert!(body["data"].is_array(), "data should be an array");
    let data = body["data"].as_array().expect("data should be array");
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["id"], "gpt-4o");
    assert_eq!(data[0]["object"], "model");

    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn test_health_endpoint() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), None);
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let client = reqwest::Client::new();
    let resp = client
        .get(handle.url("/health"))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("body should be JSON");
    assert_eq!(body["status"], "healthy");

    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn test_readiness_endpoint() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), None);
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let client = reqwest::Client::new();
    let resp = client
        .get(handle.url("/health/readiness"))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("body should be JSON");
    assert_eq!(body["status"], "healthy");

    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn test_model_not_found() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), None);
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .json(&make_request_body("nonexistent-model", false))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 404);
    let body: Value = resp.json().await.expect("body should be JSON");
    assert_eq!(body["error"]["code"], "model_not_found");
    assert!(
        body["error"]["message"]
            .as_str()
            .expect("message should be string")
            .contains("nonexistent-model")
    );

    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn test_chat_completions_alias_route() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), None);
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let mock_mock = setup_mock_non_stream(&mock);

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/chat/completions"))
        .json(&make_request_body("gpt-4o", false))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);
    mock_mock.assert_hits(1);
    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn test_missing_model_field() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), None);
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .json(&json!({"messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 400);
    let body: Value = resp.json().await.expect("body should be JSON");
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert!(
        body["error"]["message"]
            .as_str()
            .expect("message")
            .contains("model")
    );

    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn test_health_does_not_require_auth() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), Some("sk-master-secret"));
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let client = reqwest::Client::new();
    let resp = client
        .get(handle.url("/health"))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);

    let _ = handle.shutdown_tx.send(());
}

fn make_cached_request_body(model: &str, stream: bool) -> Value {
    json!({
    "model": model,
    "messages": [{"role": "user", "content": "Say hello"}],
    "stream": stream,
    "cache": {"insert": true}
    })
}

#[tokio::test]
async fn test_cache_hit_consumes_rpm() {
    let mock = MockServer::start();
    let config = ProxyConfig {
        model_list: vec![ConfigModel {
            model_name: "gpt-4o".to_string(),
            oxllm_params: OxllmParams {
                model: "openai/gpt-4o".to_string(),
                api_key: Some("sk-test-key".to_string()),
                api_base: Some(mock.base_url()),
                timeout: Some(30.0),
                stream_timeout: Some(30),
                ..Default::default()
            },
            ..Default::default()
        }],
        oxllm_settings: OxllmSettings {
            rpm: Some(2),
            ..Default::default()
        },
        general_settings: GeneralSettings {
            allow_private_api_base: Some(true),
            ..Default::default()
        },
        router_settings: Default::default(),
        extra: serde_json::Map::new(),
    };
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let upstream_mock = setup_mock_non_stream(&mock);

    let client = reqwest::Client::new();

    // Request 1: cache miss — consumes 1 RPM, populates the cache.
    let resp1 = client
        .post(handle.url("/v1/chat/completions"))
        .json(&make_cached_request_body("gpt-4o", false))
        .send()
        .await
        .expect("first request");
    assert_eq!(resp1.status(), 200);

    // Request 2: cache hit — still consumes an RPM slot (request received and
    // processed). Total RPM now 2 (== limit).
    let resp2 = client
        .post(handle.url("/v1/chat/completions"))
        .json(&make_cached_request_body("gpt-4o", false))
        .send()
        .await
        .expect("cache-hit request");
    assert_eq!(
        resp2.status(),
        200,
        "cache hit within RPM limit should succeed"
    );
    assert_eq!(
        resp2
            .headers()
            .get("x-oxllm-cache-hit")
            .and_then(|v| v.to_str().ok()),
        Some("true"),
        "request 2 should be a cache hit"
    );

    // Request 3: cache hit again — RPM limit (2) now exceeded, so even though
    // the response is cached, the rate-limit pre-check rejects it with 429.
    let resp3 = client
        .post(handle.url("/v1/chat/completions"))
        .json(&make_cached_request_body("gpt-4o", false))
        .send()
        .await
        .expect("third request");
    assert_eq!(
        resp3.status(),
        429,
        "cache hits must consume RPM; after the limit is reached further requests are rejected"
    );

    upstream_mock.assert_hits(1);

    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn test_cache_streaming_does_not_hit_non_streaming_entry() {
    // After fix for cache-key stream-field inclusion: streaming and
    // non-streaming requests have separate cache keys. A streaming request
    // must NOT serve a cache entry stored by a non-streaming request.
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), None);
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let upstream_mock = setup_mock_non_stream(&mock);
    let client = reqwest::Client::new();

    // Seed cache with non-streaming request
    let resp1 = client
        .post(handle.url("/v1/chat/completions"))
        .json(&make_cached_request_body("gpt-4o", false))
        .send()
        .await
        .expect("seed request");
    assert_eq!(resp1.status(), 200);

    // Streaming request should NOT hit the non-streaming cache entry
    let resp2 = client
        .post(handle.url("/v1/chat/completions"))
        .json(&make_cached_request_body("gpt-4o", true))
        .send()
        .await
        .expect("streaming request");
    assert_eq!(resp2.status(), 200);
    assert_ne!(
        resp2
            .headers()
            .get("x-oxllm-cache-hit")
            .and_then(|v| v.to_str().ok()),
        Some("true"),
        "streaming request must not hit a non-streaming cache entry"
    );

    upstream_mock.assert_hits(2);

    let _ = handle.shutdown_tx.send(());
}

fn make_multi_model_config(
    mock_base_url: &str,
    primary: &str,
    fb1: &str,
    fb2: &str,
) -> ProxyConfig {
    fn dep(name: &str, base: &str) -> ConfigModel {
        ConfigModel {
            model_name: name.to_string(),
            oxllm_params: OxllmParams {
                model: format!("openai/{name}"),
                api_key: Some("sk-test-key".to_string()),
                api_base: Some(base.to_string()),
                timeout: Some(30.0),
                stream_timeout: Some(30),
                ..Default::default()
            },
            ..Default::default()
        }
    }
    ProxyConfig {
        model_list: vec![
            dep(primary, mock_base_url),
            dep(fb1, mock_base_url),
            dep(fb2, mock_base_url),
        ],
        oxllm_settings: OxllmSettings {
            routing: RoutingKnobs {
                fallbacks: Some(json!([{ primary: [fb1, fb2] }])),
                num_retries: Some(0),
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

#[tokio::test]
async fn test_fallback_continues_on_4xx() {
    let mock = MockServer::start();
    let config = make_multi_model_config(&mock.base_url(), "primary", "fb1", "fb2");
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let primary_mock = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .json_body_partial(json!({"model": "primary"}).to_string());
        then.status(500)
            .json_body(json!({"error": {"message": "boom"}}));
    });
    let fb1_mock = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .json_body_partial(json!({"model": "fb1"}).to_string());
        then.status(400)
            .json_body(json!({"error": {"message": "bad"}}));
    });
    let fb2_mock = mock.mock(|when, then| {
 when.method(httpmock::Method::POST)
 .path("/chat/completions")
 .json_body_partial(json!({"model": "fb2"}).to_string());
 then.status(200).header("content-type", "application/json").json_body(json!({
 "id": "ok-from-fb2",
 "object": "chat.completion",
 "created": 1,
 "model": "fb2",
 "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
 "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
 }));
 });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .json(&json!({
        "model": "primary",
        "messages": [{"role": "user", "content": "hi"}],
        }))
        .send()
        .await
        .expect("request");

    assert_eq!(resp.status(), 200, "should succeed via fb2");
    assert_eq!(
        resp.headers()
            .get("x-oxllm-fallback-used")
            .and_then(|v| v.to_str().ok()),
        Some("fb2"),
        "fallback-used header should reflect the model that succeeded"
    );
    let body: Value = resp.json().await.expect("json");
    assert_eq!(body["id"], "ok-from-fb2");

    primary_mock.assert_hits(1);
    fb1_mock.assert_hits(1);
    fb2_mock.assert_hits(1);

    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn test_fallback_exhausted_returns_last_error() {
    let mock = MockServer::start();
    let config = make_multi_model_config(&mock.base_url(), "primary", "fb1", "fb2");
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let _primary_mock = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .json_body_partial(json!({"model": "primary"}).to_string());
        then.status(500)
            .json_body(json!({"error": {"message": "boom"}}));
    });
    let _fb1_mock = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .json_body_partial(json!({"model": "fb1"}).to_string());
        then.status(400)
            .json_body(json!({"error": {"message": "bad-1"}}));
    });
    let _fb2_mock = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .json_body_partial(json!({"model": "fb2"}).to_string());
        then.status(400)
            .json_body(json!({"error": {"message": "bad-2"}}));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .json(&json!({
        "model": "primary",
        "messages": [{"role": "user", "content": "hi"}],
        }))
        .send()
        .await
        .expect("request");

    assert_eq!(
        resp.status(),
        400,
        "all fallbacks 4xx → last error surfaces"
    );
    assert_eq!(
        resp.headers()
            .get("x-oxllm-fallback-used")
            .and_then(|v| v.to_str().ok()),
        Some("fb2"),
        "header should reflect the last fallback tried"
    );

    let _ = handle.shutdown_tx.send(());
}

// ---------------------------------------------------------------------------
// Network / timeout handling: the proxy must classify upstream transport
// failures predictably. A timeout surfaces as 504 (gateway timeout); an
// unreachable upstream surfaces as a 5xx server error after retries are
// exhausted. See docs/troubleshooting.md.
// ---------------------------------------------------------------------------

/// Pointing the deployment at a TCP port nothing is listening on must
/// surface as a server error to the client (the proxy must not hang, return
/// 200, or leak the transport error verbatim). The exact status is 500
/// (`ApiError::internal`) because chat's retry loop maps a non-timeout
/// transport failure to `UpstreamError::Network`, which is itself classified
/// `Retryable` — exhausting retries on a single deployment returns the
/// stashed retryable outcome, never a 2xx.
#[tokio::test]
async fn upstream_connection_refused_returns_5xx() {
    // Bind a TCP listener just to grab a free port, then drop it so nothing
    // is accepting connections. The proxy will get ECONNREFUSED on the
    // upstream call.
    let free_port = {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        listener.local_addr().expect("addr").port()
    };
    let unreachable_base = format!("http://127.0.0.1:{free_port}");

    // `num_retries: 0` so the loop bails out on the first failure instead of
    // spending several seconds in exponential backoff.
    let mut config = make_proxy_config(&unreachable_base, None);
    config.oxllm_settings.routing.num_retries = Some(0);

    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(15))
        .build()
        .expect("client");
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .json(&make_request_body("gpt-4o", false))
        .send()
        .await
        .expect("request should reach the proxy");

    assert!(
        resp.status().is_server_error(),
        "expected 5xx for refused upstream, got {}",
        resp.status()
    );

    let _ = handle.shutdown_tx.send(());
}

/// When the upstream takes longer than the deployment's `timeout`, the
/// proxy must classify it as a 504 Gateway Timeout (not 500 / 502).
#[tokio::test]
async fn upstream_timeout_returns_504() {
    let mock = MockServer::start();
    mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .header("authorization", "Bearer sk-test-key");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "id": "slow",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "late"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            }))
            // httpmock applies the delay before responding.
            .delay(std::time::Duration::from_secs(5));
    });

    let mut config = make_proxy_config(&mock.base_url(), None);
    // 0.2 s timeout against a 5 s delayed upstream → guaranteed timeout.
    // `num_retries: 0` keeps the test snappy (no backoff sleeps).
    config.model_list[0].oxllm_params.timeout = Some(0.2);
    config.oxllm_settings.routing.num_retries = Some(0);

    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(15))
        .build()
        .expect("client");
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .json(&make_request_body("gpt-4o", false))
        .send()
        .await
        .expect("request should reach the proxy");

    assert_eq!(
        resp.status(),
        504,
        "upstream timeout must map to 504 Gateway Timeout, got {} body {:?}",
        resp.status(),
        resp.text().await.unwrap_or_default()
    );

    let _ = handle.shutdown_tx.send(());
}

/// MEDIUM (C11): a stream whose upstream never delivers a usage chunk
/// (provider ignores the injected `stream_options.include_usage`) must
/// still be billed. Without estimation the router-budget settle records
/// only the nominal unknown cost (0.0001); with it, the deployment spend
/// counter advances by an estimated cost derived from the pre-call prompt
/// estimate plus the tokenized streamed completion.
#[tokio::test]
async fn test_streaming_zero_usage_still_billed() {
    let mock = MockServer::start();
    let mut config = make_proxy_config(&mock.base_url(), None);
    config.model_list[0].oxllm_params.max_budget = Some(10.0);

    let long_completion = "tokenizable completion text ".repeat(60);
    let chunk_head = "{\"id\":\"c1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,";
    let chunk_tail = "}]}\n\n";
    let role_chunk = format!(
        "data: {chunk_head}\"delta\":{{\"role\":\"assistant\",\"content\":\"\"}},\"finish_reason\":null{chunk_tail}"
    );
    let content_chunk = format!(
        "data: {chunk_head}\"delta\":{{\"content\":\"{long_completion}\"}},\"finish_reason\":null{chunk_tail}"
    );
    let stop_chunk =
        format!("data: {chunk_head}\"delta\":{{}},\"finish_reason\":\"stop\"{chunk_tail}");
    let sse_body = format!("{role_chunk}{content_chunk}{stop_chunk}data: [DONE]\n\n");
    mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .header("authorization", "Bearer sk-test-key")
            .json_body_partial(json!({"stream": true}).to_string());
        then.status(200)
            .header("content-type", "text/event-stream")
            .body(sse_body);
    });

    let (handle, state) = start_oxllm(config).await.expect("server should start");

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(15))
        .build()
        .expect("client");
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .json(&make_request_body("gpt-4o", true))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);
    let text = resp.text().await.expect("body should be text");
    assert!(
        text.contains("tokenizable completion text"),
        "content: {text}"
    );
    assert!(text.contains("[DONE]"), "terminator: {text}");

    let state = state.load_full();
    let spend = state
        .cache
        .counter_value("spend:dep:gpt-4o:openai/gpt-4o")
        .await
        .unwrap_or(0.0);
    assert!(
        spend > 0.001,
        "estimated usage must advance the deployment spend counter past the \
         nominal unknown cost (0.0001); got {spend}"
    );

    let _ = handle.shutdown_tx.send(());
}
