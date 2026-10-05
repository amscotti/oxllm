//! Integration tests for the OpenAI provider (httpmock upstream).
//!
//! OpenAI is a full provider module with reasoning-model handling:
//! - Default base: https://api.openai.com/v1
//! - Strips `openai/` prefix from model names
//! - Bearer token auth, plus optional `openai-organization` /
//!   `openai-project` headers from `oxllm_params.extra`
//! - Identity response transform (plus null-token sanitization)
//! - SSE streaming via shared parse_openai_stream_chunk
//! - Reasoning models (o1/o3/o4, gpt-5, `-reasoning`): strips `temperature`,
//!   maps `max_tokens` → `max_completion_tokens`, converts `system` →
//!   `developer`, strips o-series-rejected sampling params
//! - OpenAI publishes NO `supported_params` list → even with
//!   `drop_params: true`, unknown params are kept

mod common;

use common::start_oxllm;
use httpmock::MockServer;
use oxllm_server::OxllmParams;
use oxllm_server::config::types::{ConfigModel, GeneralSettings, OxllmSettings, ProxyConfig};
use serde_json::{Value, json};

/// Build a one-deployment OpenAI config pointing at `mock_base_url`.
///
/// `model_key` is the full litellm model id (`openai/gpt-4o`,
/// `openai/o1-mini`, …). The client-facing `model_name` mirrors the key so
/// requests can address the deployment as-is. Wildcard keys (`openai/*`)
/// map to `model_name: "*"` so the router matches any `openai/...` model.
fn make_proxy_config(mock_base_url: &str, model_key: &str) -> ProxyConfig {
    let model_name = if model_key.contains('*') {
        "*".to_string()
    } else {
        model_key.to_string()
    };
    ProxyConfig {
        model_list: vec![ConfigModel {
            model_name,
            oxllm_params: OxllmParams {
                model: model_key.to_string(),
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
            allow_private_api_base: Some(true),
            ..Default::default()
        },
        router_settings: Default::default(),
        extra: serde_json::Map::new(),
    }
}

/// Standard chat completion body addressed to the deployment `model`.
fn openai_chat_body(model: &str) -> Value {
    json!({
        "model": model,
        "messages": [{"role": "user", "content": "Say hello"}],
    })
}

/// Standard chat completion response echoing `model` with `content`.
fn chat_completion(model: &str, content: &str) -> Value {
    json!({
        "id": "chatcmpl-openai-1",
        "object": "chat.completion",
        "created": 1234567890,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    })
}

// ---------------------------------------------------------------------------
// Standard matrix
// ---------------------------------------------------------------------------

#[tokio::test]
async fn openai_non_streaming_chat() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), "openai/gpt-4o");
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .header("authorization", "Bearer sk-test-key")
            .json_body_partial(json!({"model": "gpt-4o"}).to_string());
        then.status(200)
            .header("content-type", "application/json")
            .json_body(chat_completion("gpt-4o", "Hello from OpenAI!"));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .json(&openai_chat_body("openai/gpt-4o"))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("body should be JSON");
    assert_eq!(body["id"], "chatcmpl-openai-1");
    assert_eq!(body["object"], "chat.completion");
    assert_eq!(
        body["choices"][0]["message"]["content"],
        "Hello from OpenAI!"
    );
    assert_eq!(body["model"], "gpt-4o");

    upstream.assert_hits(1);
    handle.shutdown();
}

#[tokio::test]
async fn openai_streaming_chat() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), "openai/gpt-4o");
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let sse_body = concat!(
        "data: {\"id\":\"chatcmpl-openai-stream-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"chatcmpl-openai-stream-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"chatcmpl-openai-stream-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"!\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"chatcmpl-openai-stream-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
        "data: [DONE]\n\n"
    );

    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .header("authorization", "Bearer sk-test-key")
            .json_body_partial(json!({"model": "gpt-4o", "stream": true}).to_string());
        then.status(200)
            .header("content-type", "text/event-stream")
            .body(sse_body);
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .json(&json!({
            "model": "openai/gpt-4o",
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
async fn openai_custom_api_base() {
    let mock = MockServer::start();
    let custom_base = mock.base_url();

    let config = ProxyConfig {
        model_list: vec![ConfigModel {
            model_name: "openai/gpt-4o".to_string(),
            oxllm_params: OxllmParams {
                model: "openai/gpt-4o".to_string(),
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
            .json_body(chat_completion("gpt-4o", "custom base"));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .json(&openai_chat_body("openai/gpt-4o"))
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
async fn openai_request_with_tool_calls() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), "openai/gpt-4o");
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
                "model": "gpt-4o",
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
            "model": "openai/gpt-4o",
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
async fn openai_json_object_mode() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), "openai/gpt-4o");
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .json_body_partial(json!({"response_format": {"type": "json_object"}}).to_string());
        then.status(200)
            .header("content-type", "application/json")
            .json_body(chat_completion("gpt-4o", "{\"key\": \"value\"}"));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .json(&json!({
            "model": "openai/gpt-4o",
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
async fn openai_json_schema_mode() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), "openai/gpt-4o");
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
            .json_body(chat_completion("gpt-4o", "{\"answer\": \"42\"}"));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .json(&json!({
            "model": "openai/gpt-4o",
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
async fn openai_streaming_with_usage_chunk() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), "openai/gpt-4o");
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let sse_body = concat!(
        "data: {\"id\":\"chatcmpl-usage-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"chatcmpl-usage-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"chatcmpl-usage-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5,\"total_tokens\":15}}\n\n",
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
            "model": "openai/gpt-4o",
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
async fn openai_drop_params_true_keeps_unknown_params() {
    let mock = MockServer::start();
    let mut config = make_proxy_config(&mock.base_url(), "openai/gpt-4o");
    config.oxllm_settings.drop_params = Some(true);
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    // INVERTED vs xAI/Groq: OpenAI publishes no `supported_params` list, so
    // `drop_params: true` must NOT strip unknown params. The mock only
    // matches a body that still carries `unknown_param`.
    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .header("authorization", "Bearer sk-test-key")
            .json_body_partial(json!({"unknown_param": "should_be_kept"}).to_string());
        then.status(200)
            .header("content-type", "application/json")
            .json_body(chat_completion("gpt-4o", "ok"));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .json(&json!({
            "model": "openai/gpt-4o",
            "messages": [{"role": "user", "content": "Hi"}],
            "unknown_param": "should_be_kept",
        }))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);

    upstream.assert_hits(1);
    handle.shutdown();
}

#[tokio::test]
async fn openai_auth_required() {
    let mock = MockServer::start();

    // Part 1: proxy with no master key — upstream mock asserts Bearer sk-test-key.
    let config = make_proxy_config(&mock.base_url(), "openai/gpt-4o");
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .header("authorization", "Bearer sk-test-key");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(chat_completion("gpt-4o", "auth ok"));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .json(&openai_chat_body("openai/gpt-4o"))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);
    upstream.assert_hits(1);
    handle.shutdown();

    // Part 2: proxy with master_key — request without auth header → 401.
    let mut config = make_proxy_config(&mock.base_url(), "openai/gpt-4o");
    config.general_settings.master_key = Some("sk-master-secret".to_string());
    let (handle2, _state2) = start_oxllm(config).await.expect("server should start");

    let client2 = reqwest::Client::new();
    let resp2 = client2
        .post(handle2.url("/v1/chat/completions"))
        .json(&openai_chat_body("openai/gpt-4o"))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp2.status(), 401);
    let body2: Value = resp2.json().await.expect("json");
    assert_eq!(body2["error"]["code"], "invalid_api_key");

    handle2.shutdown();
}

#[tokio::test]
async fn openai_upstream_4xx_propagates() {
    let mock = MockServer::start();
    let mut config = make_proxy_config(&mock.base_url(), "openai/gpt-4o");
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
        .json(&openai_chat_body("openai/gpt-4o"))
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
async fn openai_upstream_5xx_propagates() {
    let mock = MockServer::start();
    let mut config = make_proxy_config(&mock.base_url(), "openai/gpt-4o");
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
        .json(&openai_chat_body("openai/gpt-4o"))
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
async fn openai_model_not_found() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), "openai/gpt-4o");
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

    assert_eq!(resp.status(), 404);
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
async fn openai_list_models() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), "openai/gpt-4o");
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
    assert_eq!(data[0]["id"], "openai/gpt-4o");
    assert_eq!(data[0]["object"], "model");

    handle.shutdown();
}

#[tokio::test]
async fn openai_request_with_seed() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), "openai/gpt-4o");
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .json_body_partial(json!({"seed": 42}).to_string());
        then.status(200)
            .header("content-type", "application/json")
            .json_body(chat_completion("gpt-4o", "seeded response"));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .json(&json!({
            "model": "openai/gpt-4o",
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
async fn openai_request_with_max_completion_tokens() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), "openai/gpt-4o");
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .json_body_partial(json!({"max_completion_tokens": 100}).to_string());
        then.status(200)
            .header("content-type", "application/json")
            .json_body(chat_completion("gpt-4o", "ok"));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .json(&json!({
            "model": "openai/gpt-4o",
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
async fn openai_request_with_stop_sequences() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), "openai/gpt-4o");
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .json_body_partial(json!({"stop": ["\n", "done"]}).to_string());
        then.status(200)
            .header("content-type", "application/json")
            .json_body(chat_completion("gpt-4o", "stopped"));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .json(&json!({
            "model": "openai/gpt-4o",
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
async fn openai_request_with_reasoning_effort() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), "openai/gpt-4o");
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .json_body_partial(json!({"reasoning_effort": "high"}).to_string());
        then.status(200)
            .header("content-type", "application/json")
            .json_body(chat_completion("gpt-4o", "reasoned"));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .json(&json!({
            "model": "openai/gpt-4o",
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

// ---------------------------------------------------------------------------
// OpenAI-specific
// ---------------------------------------------------------------------------

#[tokio::test]
async fn openai_embedding_request() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), "openai/text-embedding-3-small");
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/embeddings")
            .header("authorization", "Bearer sk-test-key")
            .json_body_partial(json!({"model": "text-embedding-3-small"}).to_string());
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}],
                "model": "text-embedding-3-small",
                "usage": {"prompt_tokens": 2, "total_tokens": 2}
            }));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/embeddings"))
        .json(&json!({
            "model": "openai/text-embedding-3-small",
            "input": "hello",
        }))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("json");
    assert_eq!(body["object"], "list");
    assert_eq!(body["data"][0]["object"], "embedding");
    assert_eq!(body["data"][0]["embedding"], json!([0.1, 0.2, 0.3]));

    upstream.assert_hits(1);
    handle.shutdown();
}

#[tokio::test]
async fn openai_request_with_sampling_params() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), "openai/gpt-4o");
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .json_body_partial(
                json!({
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "presence_penalty": 0.1,
                    "frequency_penalty": 0.2,
                })
                .to_string(),
            );
        then.status(200)
            .header("content-type", "application/json")
            .json_body(chat_completion("gpt-4o", "sampled"));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .json(&json!({
            "model": "openai/gpt-4o",
            "messages": [{"role": "user", "content": "Hi"}],
            "temperature": 0.7,
            "top_p": 0.9,
            "presence_penalty": 0.1,
            "frequency_penalty": 0.2,
        }))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);

    upstream.assert_hits(1);
    handle.shutdown();
}

#[tokio::test]
async fn openai_reasoning_model_strips_temperature() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), "openai/o1-mini");
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    // Exact body equality: if `temperature` were forwarded, the mock would
    // not match (the upstream body must be exactly the object below).
    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .json_body(json!({
                "model": "o1-mini",
                "messages": [{"role": "user", "content": "Reason for me"}],
            }));
        then.status(200)
            .header("content-type", "application/json")
            .json_body(chat_completion("o1-mini", "reasoned"));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .json(&json!({
            "model": "openai/o1-mini",
            "messages": [{"role": "user", "content": "Reason for me"}],
            "temperature": 0.7,
        }))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);

    upstream.assert_hits(1);
    handle.shutdown();
}

#[tokio::test]
async fn openai_reasoning_model_maps_max_tokens_to_max_completion_tokens() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), "openai/o1-mini");
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    // Exact body equality: asserts `max_completion_tokens` is present AND
    // `max_tokens` is gone (o-series models reject it with HTTP 400).
    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .json_body(json!({
                "model": "o1-mini",
                "messages": [{"role": "user", "content": "Reason for me"}],
                "max_completion_tokens": 50,
            }));
        then.status(200)
            .header("content-type", "application/json")
            .json_body(chat_completion("o1-mini", "reasoned"));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .json(&json!({
            "model": "openai/o1-mini",
            "messages": [{"role": "user", "content": "Reason for me"}],
            "max_tokens": 50,
        }))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);

    upstream.assert_hits(1);
    handle.shutdown();
}

#[tokio::test]
async fn openai_reasoning_model_converts_system_role_to_developer() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), "openai/o1-mini");
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    // Exact body equality: the first message must be `role: developer` —
    // any leftover `system` role fails the match.
    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .json_body(json!({
                "model": "o1-mini",
                "messages": [
                    {"role": "developer", "content": "You are a genius"},
                    {"role": "user", "content": "Think hard"},
                ],
            }));
        then.status(200)
            .header("content-type", "application/json")
            .json_body(chat_completion("o1-mini", "reasoned"));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .json(&json!({
            "model": "openai/o1-mini",
            "messages": [
                {"role": "system", "content": "You are a genius"},
                {"role": "user", "content": "Think hard"},
            ],
        }))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);

    upstream.assert_hits(1);
    handle.shutdown();
}

#[tokio::test]
async fn openai_organization_and_project_headers_forwarded() {
    let mock = MockServer::start();
    let mut config = make_proxy_config(&mock.base_url(), "openai/gpt-4o");
    config.model_list[0].oxllm_params.extra = json!({
        "organization": "org-1",
        "project": "proj-1",
    })
    .as_object()
    .expect("extra map")
    .clone();
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .header("authorization", "Bearer sk-test-key")
            .header("openai-organization", "org-1")
            .header("openai-project", "proj-1");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(chat_completion("gpt-4o", "routed"));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .json(&openai_chat_body("openai/gpt-4o"))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);

    upstream.assert_hits(1);
    handle.shutdown();
}

#[tokio::test]
async fn openai_wildcard_model_deployment() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), "openai/*");
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .header("authorization", "Bearer sk-test-key")
            .json_body_partial(json!({"model": "gpt-4o-mini"}).to_string());
        then.status(200)
            .header("content-type", "application/json")
            .json_body(chat_completion("gpt-4o-mini", "wildcard"));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .json(&json!({
            "model": "openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": "Hi"}],
        }))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("json");
    assert_eq!(body["choices"][0]["message"]["content"], "wildcard");

    upstream.assert_hits(1);
    handle.shutdown();
}
