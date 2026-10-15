//! Integration tests for the DeepSeek provider (httpmock upstream).
//!
//! DeepSeek uses OpenAiCompatProvider with:
//! - Default base: https://api.deepseek.com/v1
//! - Strips `deepseek/` prefix from model names
//! - Bearer token auth
//! - Identity response transform (reasoning_content passes through)
//! - SSE streaming via shared parse_openai_stream_chunk

mod common;

use common::start_oxllm;
use httpmock::MockServer;
use oxllm_server::OxllmParams;
use oxllm_server::config::types::{ConfigModel, GeneralSettings, OxllmSettings, ProxyConfig};
use serde_json::{Value, json};

fn make_proxy_config(mock_base_url: &str, master_key: Option<&str>) -> ProxyConfig {
    ProxyConfig {
        model_list: vec![ConfigModel {
            model_name: "deepseek-chat".to_string(),
            oxllm_params: OxllmParams {
                model: "deepseek/deepseek-chat".to_string(),
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

fn deepseek_chat_body() -> Value {
    json!({
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": "Say hello"}],
    })
}

#[tokio::test]
async fn deepseek_non_streaming_chat() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), None);
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .header("authorization", "Bearer sk-test-key")
            .json_body_partial(json!({"model": "deepseek-chat"}).to_string());
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "id": "chatcmpl-deepseek-1",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "deepseek-chat",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello from DeepSeek!"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
            }));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .json(&deepseek_chat_body())
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("body should be JSON");
    assert_eq!(body["id"], "chatcmpl-deepseek-1");
    assert_eq!(body["object"], "chat.completion");
    assert_eq!(
        body["choices"][0]["message"]["content"],
        "Hello from DeepSeek!"
    );
    assert_eq!(body["model"], "deepseek-chat");

    upstream.assert_hits(1);
    handle.shutdown();
}

#[tokio::test]
async fn deepseek_streaming_chat() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), None);
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let sse_body = concat!(
        "data: {\"id\":\"chatcmpl-deepseek-stream-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"deepseek-chat\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"chatcmpl-deepseek-stream-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"deepseek-chat\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"chatcmpl-deepseek-stream-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"deepseek-chat\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"!\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"chatcmpl-deepseek-stream-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"deepseek-chat\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
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
            "model": "deepseek-chat",
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
async fn deepseek_custom_api_base() {
    let mock = MockServer::start();
    let custom_base = mock.base_url();

    let config = ProxyConfig {
        model_list: vec![ConfigModel {
            model_name: "deepseek-chat".to_string(),
            oxllm_params: OxllmParams {
                model: "deepseek/deepseek-chat".to_string(),
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
                "model": "deepseek-chat",
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
        .json(&deepseek_chat_body())
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
async fn deepseek_request_with_tool_calls() {
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
                "model": "deepseek-chat",
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
            "model": "deepseek-chat",
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
async fn deepseek_json_object_mode() {
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
                "model": "deepseek-chat",
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
            "model": "deepseek-chat",
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
async fn deepseek_json_schema_mode() {
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
                "model": "deepseek-chat",
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
            "model": "deepseek-chat",
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
async fn deepseek_streaming_with_usage_chunk() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), None);
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let sse_body = concat!(
        "data: {\"id\":\"chatcmpl-usage-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"deepseek-chat\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"chatcmpl-usage-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"deepseek-chat\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"chatcmpl-usage-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"deepseek-chat\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5,\"total_tokens\":15}}\n\n",
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
            "model": "deepseek-chat",
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
async fn deepseek_drop_params_true_strips_unknown() {
    let mock = MockServer::start();
    let mut config = make_proxy_config(&mock.base_url(), None);
    config.oxllm_settings.drop_params = Some(true);
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    // The mock verifies the upstream request does NOT contain `unknown_param`.
    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .header("authorization", "Bearer sk-test-key")
            .json_body_partial(json!({"model": "deepseek-chat"}).to_string());
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "id": "chatcmpl-drop",
                "object": "chat.completion",
                "created": 1,
                "model": "deepseek-chat",
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
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "Hi"}],
            "unknown_param": "should_be_dropped",
        }))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);

    // The mock only matched requests WITHOUT unknown_param, so a hit confirms it was stripped.
    upstream.assert_hits(1);
    handle.shutdown();
}

#[tokio::test]
async fn deepseek_auth_required() {
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
                "model": "deepseek-chat",
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
        .json(&deepseek_chat_body())
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
        .json(&deepseek_chat_body())
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp2.status(), 401);
    let body2: Value = resp2.json().await.expect("json");
    assert_eq!(body2["error"]["code"], "invalid_api_key");

    handle2.shutdown();
}

#[tokio::test]
async fn deepseek_upstream_4xx_propagates() {
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
        .json(&deepseek_chat_body())
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
async fn deepseek_upstream_5xx_propagates() {
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
        .json(&deepseek_chat_body())
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
async fn deepseek_model_not_found() {
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
async fn deepseek_list_models() {
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
    assert_eq!(data[0]["id"], "deepseek-chat");
    assert_eq!(data[0]["object"], "model");

    handle.shutdown();
}

#[tokio::test]
async fn deepseek_request_with_seed() {
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
                "model": "deepseek-chat",
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
            "model": "deepseek-chat",
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
async fn deepseek_request_with_max_completion_tokens() {
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
                "model": "deepseek-chat",
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
            "model": "deepseek-chat",
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
async fn deepseek_request_with_stop_sequences() {
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
                "model": "deepseek-chat",
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
            "model": "deepseek-chat",
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
async fn deepseek_request_with_reasoning_effort() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), None);
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .json_body_partial(json!({"thinking": {"type": "enabled"}}).to_string());
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "id": "chatcmpl-reason",
                "object": "chat.completion",
                "created": 1,
                "model": "deepseek-chat",
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
            "model": "deepseek-chat",
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
async fn deepseek_reasoner_non_streaming_surfaces_reasoning_content() {
    let mock = MockServer::start();
    let config = ProxyConfig {
        model_list: vec![ConfigModel {
            model_name: "deepseek-reasoner".to_string(),
            oxllm_params: OxllmParams {
                model: "deepseek/deepseek-reasoner".to_string(),
                api_key: Some("sk-test-key".to_string()),
                api_base: Some(mock.base_url().to_string()),
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

    // DeepSeek's OpenAI-compatible API returns `reasoning_content` on the
    // assistant message; the identity transform must pass it through.
    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .header("authorization", "Bearer sk-test-key")
            .json_body_partial(json!({"model": "deepseek-reasoner"}).to_string());
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "id": "chatcmpl-reasoner-1",
                "object": "chat.completion",
                "created": 1,
                "model": "deepseek-reasoner",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Final answer",
                        "reasoning_content": "Let me think step by step."
                    },
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
            }));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .json(&json!({
            "model": "deepseek-reasoner",
            "messages": [{"role": "user", "content": "Solve it"}],
        }))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("json");
    let message = &body["choices"][0]["message"];
    assert_eq!(message["content"], "Final answer");
    assert_eq!(message["reasoning_content"], "Let me think step by step.");

    upstream.assert_hits(1);
    handle.shutdown();
}

#[tokio::test]
async fn deepseek_reasoner_streaming_emits_reasoning_content_deltas() {
    let mock = MockServer::start();
    let config = ProxyConfig {
        model_list: vec![ConfigModel {
            model_name: "deepseek-reasoner".to_string(),
            oxllm_params: OxllmParams {
                model: "deepseek/deepseek-reasoner".to_string(),
                api_key: Some("sk-test-key".to_string()),
                api_base: Some(mock.base_url().to_string()),
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

    // DeepSeek reasoner streams `reasoning_content` deltas before the content
    // deltas; the shared SSE parser must forward them unchanged.
    let sse_body = concat!(
        "data: {\"id\":\"chatcmpl-reasoner-stream-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"deepseek-reasoner\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"reasoning_content\":\"\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"chatcmpl-reasoner-stream-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"deepseek-reasoner\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"Let me think\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"chatcmpl-reasoner-stream-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"deepseek-reasoner\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\" step by step.\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"chatcmpl-reasoner-stream-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"deepseek-reasoner\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Final\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"chatcmpl-reasoner-stream-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"deepseek-reasoner\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
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
            "model": "deepseek-reasoner",
            "messages": [{"role": "user", "content": "Solve it"}],
            "stream": true,
        }))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);
    let text = resp.text().await.expect("body");

    assert!(
        text.contains("\"reasoning_content\":\"Let me think\""),
        "should contain reasoning delta: {text}"
    );
    assert!(
        text.contains("\"reasoning_content\":\" step by step.\""),
        "should contain second reasoning delta: {text}"
    );
    assert!(
        text.contains("\"content\":\"Final\""),
        "should contain content delta: {text}"
    );

    upstream.assert_hits(1);
    handle.shutdown();
}

#[tokio::test]
async fn deepseek_stream_in_body_error_propagates() {
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
            "model": "deepseek-chat",
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
