//! Integration tests for the Anthropic provider (httpmock upstream).
//!
//! Anthropic uses the AnthropicProvider with:
//! - Default base: https://api.anthropic.com/v1 (provider appends `/messages`)
//! - Strips `anthropic/` prefix from model names
//! - `x-api-key` + `anthropic-version` header auth (not Bearer)
//! - Request transform: system → top-level system, max_tokens injected,
//!   tools → input_schema, cache_control preserved, reasoning_effort → thinking
//! - Anthropic content-block responses → OpenAI chat.completion shape
//! - SSE streaming via `event:`/`data:` frames (not bare data lines)

mod common;

use common::start_oxllm;
use httpmock::MockServer;
use oxllm_server::OxllmParams;
use oxllm_server::config::types::{ConfigModel, GeneralSettings, OxllmSettings, ProxyConfig};
use serde_json::{Value, json};

fn make_proxy_config(mock_base_url: &str, model_key: &str) -> ProxyConfig {
    ProxyConfig {
        model_list: vec![ConfigModel {
            model_name: "claude".to_string(),
            oxllm_params: OxllmParams {
                model: model_key.to_string(),
                api_key: Some("sk-anthropic-test".to_string()),
                // Anthropic's default base is `https://api.anthropic.com/v1`
                // and the provider appends `/messages`, so requests land on
                // `{mock_base_url}/v1/messages`.
                api_base: Some(format!("{mock_base_url}/v1")),
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
    }
}

fn anthropic_chat_body(model: &str) -> Value {
    json!({
        "model": model,
        "messages": [{"role": "user", "content": "Say hello"}],
    })
}

/// Anthropic-shaped non-streaming response body.
fn anthropic_response(content: &str) -> Value {
    json!({
        "id": "msg_01",
        "type": "message",
        "role": "assistant",
        "model": "claude-3-5-sonnet",
        "content": [{"type": "text", "text": content}],
        "stop_reason": "end_turn",
        "stop_sequence": null,
        "usage": {"input_tokens": 10, "output_tokens": 5}
    })
}

/// Build an SSE body from `event:`/`data:` frames.
fn anthropic_sse(events: Vec<(&str, Value)>) -> String {
    let mut body = String::new();
    for (event, data) in events {
        body.push_str(&format!("event: {event}\ndata: {data}\n\n"));
    }
    body
}

fn sse_message_start() -> (&'static str, Value) {
    (
        "message_start",
        json!({
            "type": "message_start",
            "message": {
                "id": "msg_stream_01",
                "type": "message",
                "role": "assistant",
                "model": "claude-3-5-sonnet",
                "content": [],
                "stop_reason": null,
                "stop_sequence": null,
                "usage": {"input_tokens": 10, "output_tokens": 0}
            }
        }),
    )
}

fn sse_text_block_start(index: u64) -> (&'static str, Value) {
    (
        "content_block_start",
        json!({
            "type": "content_block_start",
            "index": index,
            "content_block": {"type": "text", "text": ""}
        }),
    )
}

fn sse_text_delta(index: u64, text: &str) -> (&'static str, Value) {
    (
        "content_block_delta",
        json!({
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "text_delta", "text": text}
        }),
    )
}

fn sse_tool_block_start(index: u64, id: &str, name: &str) -> (&'static str, Value) {
    (
        "content_block_start",
        json!({
            "type": "content_block_start",
            "index": index,
            "content_block": {"type": "tool_use", "id": id, "name": name, "input": {}}
        }),
    )
}

fn sse_input_json_delta(index: u64, partial: &str) -> (&'static str, Value) {
    (
        "content_block_delta",
        json!({
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "input_json_delta", "partial_json": partial}
        }),
    )
}

fn sse_block_stop(index: u64) -> (&'static str, Value) {
    (
        "content_block_stop",
        json!({"type": "content_block_stop", "index": index}),
    )
}

fn sse_message_delta(stop_reason: &str, output_tokens: u64) -> (&'static str, Value) {
    (
        "message_delta",
        json!({
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": null},
            "usage": {"output_tokens": output_tokens}
        }),
    )
}

fn sse_message_stop() -> (&'static str, Value) {
    ("message_stop", json!({"type": "message_stop"}))
}

fn sse_error_event(error_type: &str, message: &str) -> (&'static str, Value) {
    (
        "error",
        json!({"type": "error", "error": {"type": error_type, "message": message}}),
    )
}

// ---------------------------------------------------------------------------
// Core chat
// ---------------------------------------------------------------------------

#[tokio::test]
async fn anthropic_non_streaming_chat() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), "anthropic/claude-3-5-sonnet");
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/v1/messages")
            .header("x-api-key", "sk-anthropic-test")
            .header("anthropic-version", "2023-06-01")
            .json_body_partial(json!({"model": "claude-3-5-sonnet"}).to_string());
        then.status(200)
            .header("content-type", "application/json")
            .json_body(anthropic_response("Hello from Anthropic!"));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .header("authorization", "Bearer sk-master-key-test-1")
        .json(&anthropic_chat_body("claude"))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("body should be JSON");
    assert_eq!(body["id"], "msg_01");
    assert_eq!(body["object"], "chat.completion");
    assert_eq!(
        body["choices"][0]["message"]["content"],
        "Hello from Anthropic!"
    );
    assert_eq!(body["model"], "claude");
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
    assert_eq!(body["usage"]["prompt_tokens"], 10);
    assert_eq!(body["usage"]["completion_tokens"], 5);
    assert_eq!(body["usage"]["total_tokens"], 15);

    upstream.assert_hits(1);
    handle.shutdown();
}

#[tokio::test]
async fn anthropic_streaming_chat() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), "anthropic/claude-3-5-sonnet");
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let mut sse_body = anthropic_sse(vec![
        sse_message_start(),
        sse_text_block_start(0),
        sse_text_delta(0, "Hi"),
        sse_text_delta(0, "!"),
        sse_message_delta("end_turn", 3),
        sse_message_stop(),
    ]);
    sse_body.push_str("data: [DONE]\n\n");

    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/v1/messages")
            .header("x-api-key", "sk-anthropic-test")
            .header("anthropic-version", "2023-06-01")
            .json_body_partial(json!({"stream": true}).to_string());
        then.status(200)
            .header("content-type", "text/event-stream")
            .body(sse_body);
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .header("authorization", "Bearer sk-master-key-test-1")
        .json(&json!({
            "model": "claude",
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
        text.contains("\"content\":\"!\""),
        "should contain ! delta: {text}"
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
async fn anthropic_custom_api_base() {
    let mock = MockServer::start();
    let custom_base = mock.base_url();

    let config = ProxyConfig {
        model_list: vec![ConfigModel {
            model_name: "claude".to_string(),
            oxllm_params: OxllmParams {
                model: "anthropic/claude-3-5-sonnet".to_string(),
                api_key: Some("sk-anthropic-test".to_string()),
                api_base: Some(format!("{custom_base}/custom/v1")),
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
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/custom/v1/messages")
            .header("x-api-key", "sk-anthropic-test")
            .header("anthropic-version", "2023-06-01");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(anthropic_response("custom base"));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .header("authorization", "Bearer sk-master-key-test-1")
        .json(&anthropic_chat_body("claude"))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("json");
    assert_eq!(body["choices"][0]["message"]["content"], "custom base");

    upstream.assert_hits(1);
    handle.shutdown();
}

// ---------------------------------------------------------------------------
// Request transforms
// ---------------------------------------------------------------------------

#[tokio::test]
async fn anthropic_system_message_extracted_to_top_level() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), "anthropic/claude-3-5-sonnet");
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/v1/messages")
            .json_body_partial(
                json!({
                    "system": "You are a helpful assistant",
                    "messages": [{"role": "user", "content": "Say hello"}]
                })
                .to_string(),
            );
        then.status(200)
            .header("content-type", "application/json")
            .json_body(anthropic_response("ok"));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .header("authorization", "Bearer sk-master-key-test-1")
        .json(&json!({
            "model": "claude",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Say hello"}
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
async fn anthropic_multiple_system_messages_become_block_array() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), "anthropic/claude-3-5-sonnet");
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/v1/messages")
            .json_body_partial(
                json!({
                    "system": [
                        {"type": "text", "text": "System one"},
                        {"type": "text", "text": "System two"}
                    ],
                    "messages": [{"role": "user", "content": "Hi"}]
                })
                .to_string(),
            );
        then.status(200)
            .header("content-type", "application/json")
            .json_body(anthropic_response("ok"));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .header("authorization", "Bearer sk-master-key-test-1")
        .json(&json!({
            "model": "claude",
            "messages": [
                {"role": "system", "content": "System one"},
                {"role": "system", "content": "System two"},
                {"role": "user", "content": "Hi"}
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
async fn anthropic_request_with_tool_calls() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), "anthropic/claude-3-5-sonnet");
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/v1/messages")
            .json_body_partial(
                json!({
                    "tools": [{
                        "name": "get_weather",
                        "description": "Get weather",
                        "input_schema": {"type": "object", "properties": {}}
                    }],
                    "tool_choice": {"type": "auto"}
                })
                .to_string(),
            );
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "id": "msg_tool_01",
                "type": "message",
                "role": "assistant",
                "model": "claude-3-5-sonnet",
                "content": [{
                    "type": "tool_use",
                    "id": "toolu_01",
                    "name": "get_weather",
                    "input": {"location": "SF"}
                }],
                "stop_reason": "tool_use",
                "stop_sequence": null,
                "usage": {"input_tokens": 20, "output_tokens": 10}
            }));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .header("authorization", "Bearer sk-master-key-test-1")
        .json(&json!({
            "model": "claude",
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
    let tool_call = &body["choices"][0]["message"]["tool_calls"][0];
    assert_eq!(tool_call["id"], "toolu_01");
    assert_eq!(tool_call["type"], "function");
    assert_eq!(tool_call["function"]["name"], "get_weather");
    assert_eq!(tool_call["function"]["arguments"], "{\"location\":\"SF\"}");

    upstream.assert_hits(1);
    handle.shutdown();
}

#[tokio::test]
async fn anthropic_streaming_tool_call() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), "anthropic/claude-3-5-sonnet");
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let mut sse_body = anthropic_sse(vec![
        sse_message_start(),
        sse_tool_block_start(0, "toolu_01", "get_weather"),
        sse_input_json_delta(0, "{\"location\": "),
        sse_input_json_delta(0, "\"SF\"}"),
        sse_block_stop(0),
        sse_message_delta("tool_use", 10),
        sse_message_stop(),
    ]);
    sse_body.push_str("data: [DONE]\n\n");

    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/v1/messages")
            .json_body_partial(json!({"stream": true}).to_string());
        then.status(200)
            .header("content-type", "text/event-stream")
            .body(sse_body);
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .header("authorization", "Bearer sk-master-key-test-1")
        .json(&json!({
            "model": "claude",
            "messages": [{"role": "user", "content": "What's the weather in SF?"}],
            "stream": true,
        }))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);
    let text = resp.text().await.expect("body should be text");

    assert!(
        text.contains("\"name\":\"get_weather\""),
        "should contain tool name: {text}"
    );
    assert!(
        text.contains("\"tool_calls\""),
        "should contain tool_calls deltas: {text}"
    );
    assert!(
        text.contains("\"finish_reason\":\"tool_calls\""),
        "should contain tool_calls finish: {text}"
    );
    assert!(text.contains("[DONE]"), "should contain [DONE]: {text}");

    upstream.assert_hits(1);
    handle.shutdown();
}

#[tokio::test]
async fn anthropic_schema_less_json_mode_returns_an_unsupported_error() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), "anthropic/claude-3-opus");
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let upstream = mock.mock(|when, then| {
        // Any attempt to call this upstream would succeed and fail the local
        // 400 assertion below, regardless of the attempted wire encoding.
        when.method(httpmock::Method::POST);
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "id": "msg_json_01",
                "type": "message",
                "role": "assistant",
                "model": "claude-3-opus",
                "content": [{"type":"text","text":"{}"}],
                "stop_reason": "end_turn",
                "stop_sequence": null,
                "usage": {"input_tokens": 10, "output_tokens": 5}
            }));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .header("authorization", "Bearer sk-master-key-test-1")
        .json(&json!({
            "model": "claude",
            "messages": [{"role": "user", "content": "Return JSON"}],
            "response_format": {"type": "json_object"},
        }))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 400);
    let body: Value = resp.json().await.expect("json");
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("no native representation")
    );
    upstream.assert_hits(0);
    handle.shutdown();
}

#[tokio::test]
async fn anthropic_json_schema_mode() {
    let mock = MockServer::start();
    // Schemas are forwarded natively, leaving model support to the provider.
    let config = make_proxy_config(&mock.base_url(), "anthropic/claude-sonnet-4-5");
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/v1/messages")
            .json_body_partial(
                json!({
                    "output_config": {
                        "format": {
                            "type": "json_schema",
                            "schema": {
                                "type": "object",
                                "properties": {"answer": {"type": "string"}},
                                "additionalProperties": false
                            }
                        }
                    }
                })
                .to_string(),
            );
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "id": "msg_schema_01",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-5",
                "content": [{"type": "text", "text": "{\"answer\": \"42\"}"}],
                "stop_reason": "end_turn",
                "stop_sequence": null,
                "usage": {"input_tokens": 10, "output_tokens": 5}
            }));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .header("authorization", "Bearer sk-master-key-test-1")
        .json(&json!({
            "model": "claude",
            "messages": [{"role": "user", "content": "Return schema JSON"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "x",
                    "schema": {"type": "object", "properties": {"answer": {"type": "string"}}, "additionalProperties":false}
                }
            },
        }))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("json");
    assert_eq!(
        body["choices"][0]["message"]["content"],
        "{\"answer\": \"42\"}"
    );

    upstream.assert_hits(1);
    handle.shutdown();
}

#[tokio::test]
async fn anthropic_request_with_max_completion_tokens() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), "anthropic/claude-3-5-sonnet");
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/v1/messages")
            .json_body_partial(json!({"max_tokens": 100}).to_string());
        then.status(200)
            .header("content-type", "application/json")
            .json_body(anthropic_response("ok"));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .header("authorization", "Bearer sk-master-key-test-1")
        .json(&json!({
            "model": "claude",
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
async fn anthropic_request_with_stop_sequences() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), "anthropic/claude-3-5-sonnet");
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/v1/messages")
            .json_body_partial(json!({"stop_sequences": ["END"]}).to_string());
        then.status(200)
            .header("content-type", "application/json")
            .json_body(anthropic_response("ok"));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .header("authorization", "Bearer sk-master-key-test-1")
        .json(&json!({
            "model": "claude",
            "messages": [{"role": "user", "content": "Hi"}],
            "stop": "END",
        }))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);

    upstream.assert_hits(1);
    handle.shutdown();
}

#[tokio::test]
async fn anthropic_request_with_sampling_params() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), "anthropic/claude-3-5-sonnet");
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/v1/messages")
            .json_body_partial(json!({"temperature": 0.7, "top_p": 0.9, "top_k": 50}).to_string());
        then.status(200)
            .header("content-type", "application/json")
            .json_body(anthropic_response("ok"));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .header("authorization", "Bearer sk-master-key-test-1")
        .json(&json!({
            "model": "claude",
            "messages": [{"role": "user", "content": "Hi"}],
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
        }))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);

    upstream.assert_hits(1);
    handle.shutdown();
}

#[tokio::test]
async fn anthropic_request_with_reasoning_effort() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), "anthropic/claude-3-5-sonnet");
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/v1/messages")
            .json_body_partial(
                json!({
                    "thinking": {"type": "enabled", "budget_tokens": 4096},
                    "max_tokens": 20000
                })
                .to_string(),
            );
        then.status(200)
            .header("content-type", "application/json")
            .json_body(anthropic_response("reasoned"));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .header("authorization", "Bearer sk-master-key-test-1")
        .json(&json!({
            "model": "claude",
            "messages": [{"role": "user", "content": "Think hard"}],
            "reasoning_effort": "high",
            "max_completion_tokens": 20000,
        }))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);

    upstream.assert_hits(1);
    handle.shutdown();
}

#[tokio::test]
async fn anthropic_user_maps_to_metadata_user_id() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), "anthropic/claude-3-5-sonnet");
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/v1/messages")
            .json_body_partial(json!({"metadata": {"user_id": "user-123"}}).to_string());
        then.status(200)
            .header("content-type", "application/json")
            .json_body(anthropic_response("ok"));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .header("authorization", "Bearer sk-master-key-test-1")
        .json(&json!({
            "model": "claude",
            "messages": [{"role": "user", "content": "Hi"}],
            "user": "user-123",
        }))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);

    upstream.assert_hits(1);
    handle.shutdown();
}

#[tokio::test]
async fn anthropic_cache_control_preserved() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), "anthropic/claude-3-5-sonnet");
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/v1/messages")
            .json_body_partial(
                json!({
                    "system": [{
                        "type": "text",
                        "text": "You are a caching system",
                        "cache_control": {"type": "ephemeral"}
                    }]
                })
                .to_string(),
            );
        then.status(200)
            .header("content-type", "application/json")
            .json_body(anthropic_response("ok"));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .header("authorization", "Bearer sk-master-key-test-1")
        .json(&json!({
            "model": "claude",
            "messages": [
                {
                    "role": "system",
                    "content": [{
                        "type": "text",
                        "text": "You are a caching system",
                        "cache_control": {"type": "ephemeral"}
                    }]
                },
                {"role": "user", "content": "Hi"}
            ],
        }))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);

    upstream.assert_hits(1);
    handle.shutdown();
}

// ---------------------------------------------------------------------------
// Errors / auth
// ---------------------------------------------------------------------------

#[tokio::test]
async fn anthropic_drop_params_true_strips_unknown() {
    let mock = MockServer::start();
    let mut config = make_proxy_config(&mock.base_url(), "anthropic/claude-3-5-sonnet");
    config.oxllm_settings.drop_params = Some(true);
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/v1/messages")
            .header("x-api-key", "sk-anthropic-test")
            .json_body_partial(json!({"model": "claude-3-5-sonnet"}).to_string());
        then.status(200)
            .header("content-type", "application/json")
            .json_body(anthropic_response("ok"));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .header("authorization", "Bearer sk-master-key-test-1")
        .json(&json!({
            "model": "claude",
            "messages": [{"role": "user", "content": "Hi"}],
            "frequency_penalty": 1.0,
        }))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);

    upstream.assert_hits(1);
    handle.shutdown();
}

#[tokio::test]
async fn anthropic_wildcard_model_deployment() {
    let mock = MockServer::start();

    let config = ProxyConfig {
        model_list: vec![ConfigModel {
            model_name: "*".to_string(),
            oxllm_params: OxllmParams {
                model: "anthropic/*".to_string(),
                api_key: Some("sk-anthropic-test".to_string()),
                api_base: Some(format!("{}/v1", mock.base_url())),
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
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/v1/messages")
            .json_body_partial(json!({"model": "claude-3-5-sonnet"}).to_string());
        then.status(200)
            .header("content-type", "application/json")
            .json_body(anthropic_response("wildcard served"));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .header("authorization", "Bearer sk-master-key-test-1")
        .json(&anthropic_chat_body("anthropic/claude-3-5-sonnet"))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("json");
    assert_eq!(body["choices"][0]["message"]["content"], "wildcard served");

    upstream.assert_hits(1);
    handle.shutdown();
}

#[tokio::test]
async fn anthropic_upstream_4xx_propagates() {
    let mock = MockServer::start();
    let mut config = make_proxy_config(&mock.base_url(), "anthropic/claude-3-5-sonnet");
    config.oxllm_settings.routing.num_retries = Some(0);
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/v1/messages")
            .header("x-api-key", "sk-anthropic-test");
        then.status(400)
            .header("content-type", "application/json")
            .json_body(json!({
                "error": {
                    "message": "Bad Request",
                    "type": "invalid_request_error",
                    "code": "invalid_request_error"
                }
            }));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .header("authorization", "Bearer sk-master-key-test-1")
        .json(&anthropic_chat_body("claude"))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 400);
    let body: Value = resp.json().await.expect("json");
    assert_eq!(body["error"]["message"], "Bad Request");

    upstream.assert_hits(1);
    handle.shutdown();
}

#[tokio::test]
async fn anthropic_upstream_5xx_propagates() {
    let mock = MockServer::start();
    let mut config = make_proxy_config(&mock.base_url(), "anthropic/claude-3-5-sonnet");
    config.oxllm_settings.routing.num_retries = Some(0);
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/v1/messages")
            .header("x-api-key", "sk-anthropic-test");
        then.status(500)
            .header("content-type", "application/json")
            .json_body(json!({
                "error": {
                    "message": "Internal error",
                    "type": "api_error",
                    "code": "internal_error"
                }
            }));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .header("authorization", "Bearer sk-master-key-test-1")
        .json(&anthropic_chat_body("claude"))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 500);
    let body: Value = resp.json().await.expect("json");
    assert_eq!(body["error"]["message"], "Internal error");

    upstream.assert_hits(1);
    handle.shutdown();
}

#[tokio::test]
async fn anthropic_provider_model_not_found_stays_404() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), "anthropic/claude-3-5-sonnet");
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/v1/messages")
            .header("x-api-key", "sk-anthropic-test");
        then.status(404)
            .header("content-type", "application/json")
            .json_body(json!({
                "error": {
                    "message": "Model not found",
                    "type": "not_found_error",
                    "code": "not_found_error"
                }
            }));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .header("authorization", "Bearer sk-master-key-test-1")
        .json(&anthropic_chat_body("claude"))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 404);
    let body: Value = resp.json().await.expect("json");
    assert_eq!(body["error"]["message"], "Model not found");

    upstream.assert_hits(1);
    handle.shutdown();
}

#[tokio::test]
async fn anthropic_auth_required() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), "anthropic/claude-3-5-sonnet");
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/v1/messages")
            .header("x-api-key", "sk-anthropic-test");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(anthropic_response("should not be reached"));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .json(&anthropic_chat_body("claude"))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 401);
    let body: Value = resp.json().await.expect("json");
    assert_eq!(body["error"]["code"], "invalid_api_key");

    upstream.assert_hits(0);
    handle.shutdown();
}

// ---------------------------------------------------------------------------
// Response / stream
// ---------------------------------------------------------------------------

#[tokio::test]
async fn anthropic_thinking_response_surfaces_reasoning_content() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), "anthropic/claude-3-5-sonnet");
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/v1/messages")
            .header("x-api-key", "sk-anthropic-test");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "id": "msg_thinking_01",
                "type": "message",
                "role": "assistant",
                "model": "claude-3-5-sonnet",
                "content": [
                    {"type": "thinking", "thinking": "Let me think step by step.", "signature": "sig_01"},
                    {"type": "text", "text": "Final answer"}
                ],
                "stop_reason": "end_turn",
                "stop_sequence": null,
                "usage": {"input_tokens": 12, "output_tokens": 30}
            }));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .header("authorization", "Bearer sk-master-key-test-1")
        .json(&anthropic_chat_body("claude"))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("json");
    let message = &body["choices"][0]["message"];
    assert_eq!(message["content"], "Final answer");
    assert_eq!(message["reasoning_content"], "Let me think step by step.");
    assert_eq!(
        message["thinking_blocks"][0]["type"], "thinking",
        "raw thinking block should be preserved"
    );

    upstream.assert_hits(1);
    handle.shutdown();
}

#[tokio::test]
async fn anthropic_streaming_with_usage_chunk() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), "anthropic/claude-3-5-sonnet");
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let mut sse_body = anthropic_sse(vec![
        sse_message_start(),
        sse_text_block_start(0),
        sse_text_delta(0, "Hello"),
        sse_message_delta("end_turn", 5),
        sse_message_stop(),
    ]);
    sse_body.push_str("data: [DONE]\n\n");

    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/v1/messages")
            .json_body_partial(json!({"stream": true}).to_string());
        then.status(200)
            .header("content-type", "text/event-stream")
            .body(sse_body);
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .header("authorization", "Bearer sk-master-key-test-1")
        .json(&json!({
            "model": "claude",
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
        "should contain prompt tokens from message_start: {text}"
    );
    assert!(
        text.contains("\"completion_tokens\":5"),
        "should contain output tokens from message_delta: {text}"
    );
    assert!(
        text.contains("\"total_tokens\":15"),
        "should contain total tokens: {text}"
    );

    upstream.assert_hits(1);
    handle.shutdown();
}

#[tokio::test]
async fn anthropic_stream_error_event_propagates() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), "anthropic/claude-3-5-sonnet");
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let sse_body = anthropic_sse(vec![
        sse_message_start(),
        sse_text_block_start(0),
        sse_text_delta(0, "partial response"),
        sse_error_event("overloaded_error", "Our servers are overloaded"),
    ]);

    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/v1/messages")
            .json_body_partial(json!({"stream": true}).to_string());
        then.status(200)
            .header("content-type", "text/event-stream")
            .body(sse_body);
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .header("authorization", "Bearer sk-master-key-test-1")
        .json(&json!({
            "model": "claude",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": true,
        }))
        .send()
        .await
        .expect("request should succeed");

    // The SSE stream was already established (200), so the mid-stream
    // overloaded_error surfaces as a terminal error event, not an HTTP 529.
    assert_eq!(resp.status(), 200);
    let text = resp.text().await.expect("body");

    assert!(
        text.contains("Our servers are overloaded"),
        "should surface the upstream error message: {text}"
    );
    assert!(
        text.contains("[DONE]"),
        "should terminate the stream: {text}"
    );

    upstream.assert_hits(1);
    handle.shutdown();
}

#[tokio::test]
async fn anthropic_list_models() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), "anthropic/claude-3-5-sonnet");
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let client = reqwest::Client::new();
    let resp = client
        .get(handle.url("/v1/models"))
        .header("authorization", "Bearer sk-master-key-test-1")
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("json");
    assert_eq!(body["object"], "list");
    assert!(body["data"].is_array(), "data should be an array");
    let data = body["data"].as_array().expect("data should be array");
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["id"], "claude");
    assert_eq!(data[0]["object"], "model");

    handle.shutdown();
}

// ---------------------------------------------------------------------------
// Extra headers
// ---------------------------------------------------------------------------

#[tokio::test]
async fn anthropic_extra_headers_beta_merged() {
    let mock = MockServer::start();
    let mut config = make_proxy_config(&mock.base_url(), "anthropic/claude-3-5-sonnet");
    config.model_list[0].oxllm_params.extra_headers = Some(std::collections::HashMap::from([(
        "anthropic-beta".to_string(),
        "prompt-caching-2024-07-31".to_string(),
    )]));
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/v1/messages")
            .header("x-api-key", "sk-anthropic-test")
            .header("anthropic-version", "2023-06-01")
            .header("anthropic-beta", "prompt-caching-2024-07-31");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(anthropic_response("beta ok"));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .header("authorization", "Bearer sk-master-key-test-1")
        .json(&anthropic_chat_body("claude"))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("json");
    assert_eq!(body["choices"][0]["message"]["content"], "beta ok");

    upstream.assert_hits(1);
    handle.shutdown();
}

// ---------------------------------------------------------------------------
// Dynamic anthropic-beta header (collect_beta_headers wired into the host)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn anthropic_computed_beta_header_sent_for_web_search() {
    // HIGH issue: the host must call `collect_beta_headers` and merge the
    // result into the upstream request, otherwise gated features 400. A
    // request with `web_search_options` must reach the upstream carrying the
    // `web-fetch-2025-09-10` beta tag.
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), "anthropic/claude-3-5-sonnet");
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/v1/messages")
            .header("x-api-key", "sk-anthropic-test")
            .header("anthropic-version", "2023-06-01")
            .header("anthropic-beta", "web-fetch-2025-09-10");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(anthropic_response("searched"));
    });

    let client = reqwest::Client::new();
    let body = json!({
        "model": "claude",
        "messages": [{"role": "user", "content": "search the web"}],
        "web_search_options": {"search_context_size": "low"},
    });
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .header("authorization", "Bearer sk-master-key-test-1")
        .json(&body)
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);
    upstream.assert_hits(1);
    handle.shutdown();
}

#[tokio::test]
async fn anthropic_computed_beta_merged_with_deployment_header() {
    // The computed beta must be MERGED with a deployment-supplied
    // `anthropic-beta` (not replace it). Here the deployment sets
    // prompt-caching and the request triggers web-fetch; both must appear.
    let mock = MockServer::start();
    let mut config = make_proxy_config(&mock.base_url(), "anthropic/claude-3-5-sonnet");
    config.model_list[0].oxllm_params.extra_headers = Some(std::collections::HashMap::from([(
        "anthropic-beta".to_string(),
        "prompt-caching-2024-07-31".to_string(),
    )]));
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/v1/messages")
            .header(
                "anthropic-beta",
                "web-fetch-2025-09-10, prompt-caching-2024-07-31",
            );
        then.status(200)
            .header("content-type", "application/json")
            .json_body(anthropic_response("merged"));
    });

    let client = reqwest::Client::new();
    let body = json!({
        "model": "claude",
        "messages": [{"role": "user", "content": "search"}],
        "web_search_options": {"search_context_size": "low"},
    });
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .header("authorization", "Bearer sk-master-key-test-1")
        .json(&body)
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);
    upstream.assert_hits(1);
    handle.shutdown();
}
