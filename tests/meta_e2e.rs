//! Integration tests for the Meta provider (httpmock upstream).
//!
//! Meta uses OpenAiCompatProvider with:
//! - Default base: https://api.meta.ai/v1
//! - Strips `meta/` prefix from model names
//! - Bearer token auth
//! - Identity response transform (reasoning usage details pass through)
//! - SSE streaming via shared parse_openai_stream_chunk
//!
//! Models are like `meta/muse-spark-1.1`.

mod common;

use common::start_oxllm;
use httpmock::MockServer;
use oxllm_server::OxllmParams;
use oxllm_server::config::types::{ConfigModel, GeneralSettings, OxllmSettings, ProxyConfig};
use serde_json::{Value, json};

fn make_proxy_config(mock_base_url: &str, master_key: Option<&str>) -> ProxyConfig {
    ProxyConfig {
        model_list: vec![ConfigModel {
            model_name: "muse-spark-1.1".to_string(),
            oxllm_params: OxllmParams {
                model: "meta/muse-spark-1.1".to_string(),
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

fn meta_chat_body() -> Value {
    json!({
        "model": "muse-spark-1.1",
        "messages": [{"role": "user", "content": "Say hello"}],
    })
}

#[tokio::test]
async fn meta_non_streaming_chat() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), None);
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .header("authorization", "Bearer sk-test-key")
            .json_body_partial(json!({"model": "muse-spark-1.1"}).to_string());
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "id": "chatcmpl-meta-1",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "muse-spark-1.1",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello from Meta!"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
            }));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .json(&meta_chat_body())
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("body should be JSON");
    assert_eq!(body["id"], "chatcmpl-meta-1");
    assert_eq!(body["object"], "chat.completion");
    assert_eq!(body["choices"][0]["message"]["content"], "Hello from Meta!");
    assert_eq!(body["model"], "muse-spark-1.1");

    upstream.assert_hits(1);
    handle.shutdown();
}

#[tokio::test]
async fn meta_streaming_chat() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), None);
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let sse_body = concat!(
        "data: {\"id\":\"chatcmpl-meta-stream-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"muse-spark-1.1\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"chatcmpl-meta-stream-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"muse-spark-1.1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"chatcmpl-meta-stream-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"muse-spark-1.1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"!\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"chatcmpl-meta-stream-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"muse-spark-1.1\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
        "data: [DONE]\n\n"
    );

    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .header("authorization", "Bearer sk-test-key");
        then.status(200)
            .header("content-type", "text/event-stream")
            .body(sse_body);
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .json(&json!({
            "model": "muse-spark-1.1",
            "messages": [{"role": "user", "content": "Count to 3"}],
            "stream": true,
        }))
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
        text.contains("\"finish_reason\":\"stop\""),
        "should contain stop: {text}"
    );
    assert!(text.contains("[DONE]"), "should contain [DONE]: {text}");

    upstream.assert_hits(1);
    handle.shutdown();
}

#[tokio::test]
async fn meta_custom_api_base() {
    let mock = MockServer::start();
    let custom_base = mock.base_url();

    let config = ProxyConfig {
        model_list: vec![ConfigModel {
            model_name: "muse-spark-1.1".to_string(),
            oxllm_params: OxllmParams {
                model: "meta/muse-spark-1.1".to_string(),
                api_key: Some("sk-test-key".to_string()),
                api_base: Some(format!("{custom_base}/custom/v1")),
                timeout: Some(30.0),
                stream_timeout: Some(30),
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
    };
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/custom/v1/chat/completions")
            .header("authorization", "Bearer sk-test-key");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "id": "chatcmpl-custom",
                "object": "chat.completion",
                "created": 1,
                "model": "muse-spark-1.1",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "custom base"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            }));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .json(&meta_chat_body())
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("json");
    assert_eq!(body["choices"][0]["message"]["content"], "custom base");

    upstream.assert_hits(1);
    handle.shutdown();
}

#[tokio::test]
async fn meta_request_with_tool_calls() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), None);
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .json_body_partial(
                json!({
                    "tools": [{
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get weather",
                            "parameters": {"type": "object", "properties": {}}
                        }
                    }],
                    "tool_choice": "auto",
                })
                .to_string(),
            );
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "id": "chatcmpl-tool",
                "object": "chat.completion",
                "created": 1,
                "model": "muse-spark-1.1",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [{
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": "{\"location\": \"SF\"}"
                            }
                        }]
                    },
                    "finish_reason": "tool_calls"
                }],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}
            }));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .json(&json!({
            "model": "muse-spark-1.1",
            "messages": [{"role": "user", "content": "What's the weather in SF?"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {}}
                }
            }],
            "tool_choice": "auto",
        }))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("json");
    assert_eq!(body["choices"][0]["finish_reason"], "tool_calls");
    assert!(body["choices"][0]["message"]["tool_calls"].is_array());

    upstream.assert_hits(1);
    handle.shutdown();
}

#[tokio::test]
async fn meta_json_object_mode() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), None);
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .json_body_partial(json!({"response_format": {"type": "json_object"}}).to_string());
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "id": "chatcmpl-json",
                "object": "chat.completion",
                "created": 1,
                "model": "muse-spark-1.1",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "{\"key\": \"value\"}"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
            }));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .json(&json!({
            "model": "muse-spark-1.1",
            "messages": [{"role": "user", "content": "Return JSON"}],
            "response_format": {"type": "json_object"},
        }))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);

    upstream.assert_hits(1);
    handle.shutdown();
}

#[tokio::test]
async fn meta_json_schema_mode() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), None);
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .json_body_partial(
                json!({
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "x",
                            "schema": {"type": "object", "properties": {"answer": {"type": "string"}}}
                        }
                    }
                })
                .to_string(),
            );
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "id": "chatcmpl-schema",
                "object": "chat.completion",
                "model": "muse-spark-1.1",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "{\"answer\": \"42\"}"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
            }));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .json(&json!({
            "model": "muse-spark-1.1",
            "messages": [{"role": "user", "content": "Return schema JSON"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "x",
                    "schema": {"type": "object", "properties": {"answer": {"type": "string"}}}
                }
            },
        }))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);

    upstream.assert_hits(1);
    handle.shutdown();
}

#[tokio::test]
async fn meta_streaming_with_usage_chunk() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), None);
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let sse_body = concat!(
        "data: {\"id\":\"chatcmpl-usage-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"muse-spark-1.1\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"chatcmpl-usage-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"muse-spark-1.1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"chatcmpl-usage-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"muse-spark-1.1\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5,\"total_tokens\":15}}\n\n",
        "data: [DONE]\n\n"
    );

    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .json_body_partial(json!({"stream": true}).to_string());
        then.status(200)
            .header("content-type", "text/event-stream")
            .body(sse_body);
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .json(&json!({
            "model": "muse-spark-1.1",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": true,
        }))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);
    let text = resp.text().await.expect("body");

    assert!(
        text.contains("\"prompt_tokens\":10"),
        "should contain usage: {text}"
    );
    assert!(
        text.contains("\"total_tokens\":15"),
        "should contain total tokens: {text}"
    );

    upstream.assert_hits(1);
    handle.shutdown();
}

#[tokio::test]
async fn meta_drop_params_true_strips_unknown() {
    let mock = MockServer::start();
    let mut config = make_proxy_config(&mock.base_url(), None);
    config.oxllm_settings.drop_params = Some(true);
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    // "Trap" mock (registered first, so it wins on a first-match policy): any
    // request STILL carrying `unknown_param` is rejected with 400. A
    // `json_body_partial` subset match on the success mock alone would not
    // catch a stray `unknown_param` key (subset match ignores extra keys), so
    // the trap is what actually proves the stripping.
    let _trap = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .json_body_partial(json!({"unknown_param": "should_be_dropped"}).to_string());
        then.status(400)
            .json_body(json!({"error": "unknown_param reached upstream"}));
    });

    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .header("authorization", "Bearer sk-test-key")
            .json_body_partial(json!({"model": "muse-spark-1.1"}).to_string());
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "id": "chatcmpl-drop",
                "object": "chat.completion",
                "created": 1,
                "model": "muse-spark-1.1",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            }));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .json(&json!({
            "model": "muse-spark-1.1",
            "messages": [{"role": "user", "content": "Hi"}],
            "unknown_param": "should_be_dropped",
        }))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);

    upstream.assert_hits(1);
    handle.shutdown();
}

#[tokio::test]
async fn meta_auth_required() {
    let mock = MockServer::start();

    // Part 1: proxy with no master key — upstream mock asserts Bearer sk-test-key.
    let config = make_proxy_config(&mock.base_url(), None);
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .header("authorization", "Bearer sk-test-key");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "id": "chatcmpl-auth",
                "object": "chat.completion",
                "created": 1,
                "model": "muse-spark-1.1",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "auth ok"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            }));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .json(&meta_chat_body())
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);
    upstream.assert_hits(1);
    handle.shutdown();

    // Part 2: proxy with master_key — request without auth header → 401.
    let config = make_proxy_config(&mock.base_url(), Some("sk-master-secret"));
    let (handle2, _state2) = start_oxllm(config).await.expect("server should start");

    let client2 = reqwest::Client::new();
    let resp2 = client2
        .post(handle2.url("/v1/chat/completions"))
        .json(&meta_chat_body())
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp2.status(), 401);
    let body2: Value = resp2.json().await.expect("json");
    assert_eq!(body2["error"]["code"], "invalid_api_key");

    handle2.shutdown();
}

#[tokio::test]
async fn meta_upstream_4xx_propagates() {
    let mock = MockServer::start();
    let mut config = make_proxy_config(&mock.base_url(), None);
    config.oxllm_settings.routing.num_retries = Some(0);
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .header("authorization", "Bearer sk-test-key");
        then.status(429)
            .header("content-type", "application/json")
            .json_body(json!({
                "error": {
                    "message": "Rate limit exceeded",
                    "type": "rate_limit_error",
                    "code": "rate_limit_exceeded"
                }
            }));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .json(&meta_chat_body())
        .send()
        .await
        .expect("request should succeed");

    assert!(
        resp.status().is_client_error(),
        "expected 4xx, got {}",
        resp.status()
    );

    upstream.assert_hits(1);
    handle.shutdown();
}

#[tokio::test]
async fn meta_upstream_5xx_propagates() {
    let mock = MockServer::start();
    let mut config = make_proxy_config(&mock.base_url(), None);
    config.oxllm_settings.routing.num_retries = Some(0);
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .header("authorization", "Bearer sk-test-key");
        then.status(502)
            .header("content-type", "application/json")
            .json_body(json!({
                "error": {
                    "message": "Bad gateway",
                    "type": "api_error",
                    "code": "bad_gateway"
                }
            }));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .json(&meta_chat_body())
        .send()
        .await
        .expect("request should succeed");

    assert!(
        resp.status().is_server_error(),
        "expected 5xx, got {}",
        resp.status()
    );

    upstream.assert_hits(1);
    handle.shutdown();
}

#[tokio::test]
async fn meta_model_not_found() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), None);
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .json(&json!({
            "model": "nonexistent-model",
            "messages": [{"role": "user", "content": "hi"}],
        }))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 400);
    let body: Value = resp.json().await.expect("json");
    assert_eq!(body["error"]["code"], "model_not_found");
    assert!(
        body["error"]["message"]
            .as_str()
            .expect("message should be string")
            .contains("nonexistent-model")
    );

    handle.shutdown();
}

#[tokio::test]
async fn meta_list_models() {
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
    let body: Value = resp.json().await.expect("json");
    assert_eq!(body["object"], "list");
    assert!(body["data"].is_array(), "data should be an array");
    let data = body["data"].as_array().expect("data should be array");
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["id"], "muse-spark-1.1");
    assert_eq!(data[0]["object"], "model");

    handle.shutdown();
}

#[tokio::test]
async fn meta_request_with_seed() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), None);
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .json_body_partial(json!({"seed": 42}).to_string());
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "id": "chatcmpl-seed",
                "object": "chat.completion",
                "created": 1,
                "model": "muse-spark-1.1",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "seeded response"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            }));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .json(&json!({
            "model": "muse-spark-1.1",
            "messages": [{"role": "user", "content": "Hi"}],
            "seed": 42,
        }))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);

    upstream.assert_hits(1);
    handle.shutdown();
}

#[tokio::test]
async fn meta_request_with_max_completion_tokens() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), None);
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .json_body_partial(json!({"max_completion_tokens": 100}).to_string());
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "id": "chatcmpl-mct",
                "object": "chat.completion",
                "created": 1,
                "model": "muse-spark-1.1",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            }));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .json(&json!({
            "model": "muse-spark-1.1",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_completion_tokens": 100,
        }))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);

    upstream.assert_hits(1);
    handle.shutdown();
}

#[tokio::test]
async fn meta_request_with_stop_sequences() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), None);
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .json_body_partial(json!({"stop": ["\n", "done"]}).to_string());
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "id": "chatcmpl-stop",
                "object": "chat.completion",
                "created": 1,
                "model": "muse-spark-1.1",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "stopped"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            }));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .json(&json!({
            "model": "muse-spark-1.1",
            "messages": [{"role": "user", "content": "Hi"}],
            "stop": ["\n", "done"],
        }))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);

    upstream.assert_hits(1);
    handle.shutdown();
}

#[tokio::test]
async fn meta_request_with_reasoning_effort() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), None);
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .json_body_partial(json!({"reasoning_effort": "high"}).to_string());
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "id": "chatcmpl-reason",
                "object": "chat.completion",
                "created": 1,
                "model": "muse-spark-1.1",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "reasoned"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            }));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .json(&json!({
            "model": "muse-spark-1.1",
            "messages": [{"role": "user", "content": "Think hard"}],
            "reasoning_effort": "high",
        }))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);

    upstream.assert_hits(1);
    handle.shutdown();
}

#[tokio::test]
async fn meta_reasoning_usage_details_surface() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), None);
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    // Meta's reasoning models report reasoning token counts under
    // `usage.completion_tokens_details.reasoning_tokens`; the identity
    // transform must surface them to the client unchanged.
    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .header("authorization", "Bearer sk-test-key")
            .json_body_partial(json!({"model": "muse-spark-1.1"}).to_string());
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "id": "chatcmpl-reasoning-usage",
                "object": "chat.completion",
                "created": 1,
                "model": "muse-spark-1.1",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "reasoned answer"},
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 40,
                    "total_tokens": 50,
                    "completion_tokens_details": {"reasoning_tokens": 27}
                }
            }));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .json(&meta_chat_body())
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("body should be JSON");
    let usage = &body["usage"];
    assert_eq!(usage["completion_tokens"], 40);
    assert_eq!(
        usage["completion_tokens_details"]["reasoning_tokens"], 27,
        "reasoning_tokens must surface in client response"
    );

    upstream.assert_hits(1);
    handle.shutdown();
}

#[tokio::test]
async fn meta_streaming_reasoning_usage_details() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), None);
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    // The final SSE chunk carries usage with reasoning token details; the
    // stream pump must pass them through to the client.
    let sse_body = concat!(
        "data: {\"id\":\"chatcmpl-meta-reasoning-stream\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"muse-spark-1.1\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"chatcmpl-meta-reasoning-stream\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"muse-spark-1.1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Thought\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"chatcmpl-meta-reasoning-stream\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"muse-spark-1.1\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":40,\"total_tokens\":50,\"completion_tokens_details\":{\"reasoning_tokens\":27}}}\n\n",
        "data: [DONE]\n\n"
    );

    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .header("authorization", "Bearer sk-test-key");
        then.status(200)
            .header("content-type", "text/event-stream")
            .body(sse_body);
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .json(&json!({
            "model": "muse-spark-1.1",
            "messages": [{"role": "user", "content": "Solve it"}],
            "stream": true,
        }))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);
    let text = resp.text().await.expect("body");

    assert!(
        text.contains("\"reasoning_tokens\":27"),
        "should pass reasoning_tokens through in usage chunk: {text}"
    );
    assert!(
        text.contains("\"completion_tokens_details\""),
        "should include completion_tokens_details: {text}"
    );

    upstream.assert_hits(1);
    handle.shutdown();
}

#[tokio::test]
async fn meta_stream_in_body_error_propagates() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), None);
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    // A rate-limit frame before content uses the normal retry/error contract.
    let sse_body = concat!(
        "data: {\"error\": {\"message\": \"boom\", \"code\": 429}}\n\n",
        "data: [DONE]\n\n"
    );

    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .header("authorization", "Bearer sk-test-key");
        then.status(200)
            .header("content-type", "text/event-stream")
            .body(sse_body);
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .json(&json!({
            "model": "muse-spark-1.1",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": true,
        }))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 429);
    let text = resp.text().await.expect("body");

    assert!(
        text.contains("\"error\""),
        "should surface an error envelope: {text}"
    );
    assert!(
        text.contains("boom"),
        "should carry upstream message: {text}"
    );
    assert!(
        text.contains("upstream_rate_limited"),
        "should map numeric 429 to upstream_rate_limited code: {text}"
    );

    upstream.assert_hits(3);
    handle.shutdown();
}
