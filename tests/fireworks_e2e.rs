//! Integration tests for the Fireworks AI provider (httpmock upstream).
//!
//! Fireworks uses OpenAiCompatProvider with:
//! - Default base: https://api.fireworks.ai/inference/v1
//! - Strips `fireworks_ai/` prefix from model names
//! - Supports embeddings
//! - Bearer token auth
//! - 27+ supported OpenAI params
//! - SSE streaming via shared parse_openai_stream_chunk

mod common;

use common::start_oxllm;
use httpmock::MockServer;
use oxllm_server::OxllmParams;
use oxllm_server::config::types::{ConfigModel, GeneralSettings, OxllmSettings, ProxyConfig};
use serde_json::{Value, json};

fn proxy_config(mock_base_url: &str, master_key: Option<&str>) -> ProxyConfig {
    ProxyConfig {
        model_list: vec![ConfigModel {
            model_name: "fireworks-llama".to_string(),
            oxllm_params: OxllmParams {
                model: "fireworks_ai/accounts/fireworks/models/llama-v3p1-70b-instruct".to_string(),
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

fn chat_body() -> Value {
    json!({
        "model": "fireworks-llama",
        "messages": [{"role": "user", "content": "Say hello"}],
    })
}

#[tokio::test]
async fn fireworks_non_streaming_chat() {
    let mock = MockServer::start();
    let config = proxy_config(&mock.base_url(), None);
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .header("authorization", "Bearer sk-test-key")
            .json_body_partial(
                json!({"model": "accounts/fireworks/models/llama-v3p1-70b-instruct"}).to_string(),
            );
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "id": "chatcmpl-fw-1",
                "object": "chat.completion",
                "created": 1234567890,
                "model": "accounts/fireworks/models/llama-v3p1-70b-instruct",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello from Fireworks!"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
            }));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .json(&chat_body())
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("body should be JSON");
    assert_eq!(body["id"], "chatcmpl-fw-1");
    assert_eq!(body["object"], "chat.completion");
    assert_eq!(
        body["choices"][0]["message"]["content"],
        "Hello from Fireworks!"
    );
    assert_eq!(body["model"], "fireworks-llama");

    upstream.assert_hits(1);
    handle.shutdown();
}

#[tokio::test]
async fn fireworks_streaming_chat() {
    let mock = MockServer::start();
    let config = proxy_config(&mock.base_url(), None);
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let sse_body = concat!(
        "data: {\"id\":\"chatcmpl-fw-stream-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"accounts/fireworks/models/llama-v3p1-70b-instruct\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"chatcmpl-fw-stream-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"accounts/fireworks/models/llama-v3p1-70b-instruct\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"chatcmpl-fw-stream-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"accounts/fireworks/models/llama-v3p1-70b-instruct\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"!\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"chatcmpl-fw-stream-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"accounts/fireworks/models/llama-v3p1-70b-instruct\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
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
            "model": "fireworks-llama",
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
async fn fireworks_custom_api_base() {
    let mock = MockServer::start();
    let custom_base = mock.base_url();

    let config = ProxyConfig {
        model_list: vec![ConfigModel {
            model_name: "fireworks-llama".to_string(),
            oxllm_params: OxllmParams {
                model: "fireworks_ai/accounts/fireworks/models/llama-v3p1-70b-instruct".to_string(),
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
                "id": "chatcmpl-custom-fw",
                "object": "chat.completion",
                "created": 1,
                "model": "accounts/fireworks/models/llama-v3p1-70b-instruct",
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
        .json(&chat_body())
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
async fn fireworks_request_with_tool_calls() {
    let mock = MockServer::start();
    let config = proxy_config(&mock.base_url(), None);
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
                "id": "chatcmpl-tool-fw",
                "object": "chat.completion",
                "created": 1,
                "model": "accounts/fireworks/models/llama-v3p1-70b-instruct",
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
            "model": "fireworks-llama",
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
async fn fireworks_json_object_mode() {
    let mock = MockServer::start();
    let config = proxy_config(&mock.base_url(), None);
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .json_body_partial(json!({"response_format": {"type": "json_object"}}).to_string());
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "id": "chatcmpl-json-fw",
                "object": "chat.completion",
                "created": 1,
                "model": "accounts/fireworks/models/llama-v3p1-70b-instruct",
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
            "model": "fireworks-llama",
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
async fn fireworks_json_schema_mode() {
    let mock = MockServer::start();
    let config = proxy_config(&mock.base_url(), None);
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
                "id": "chatcmpl-schema-fw",
                "object": "chat.completion",
                "model": "accounts/fireworks/models/llama-v3p1-70b-instruct",
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
            "model": "fireworks-llama",
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
async fn fireworks_streaming_with_usage_chunk() {
    let mock = MockServer::start();
    let config = proxy_config(&mock.base_url(), None);
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let sse_body = concat!(
        "data: {\"id\":\"chatcmpl-usage-fw-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"accounts/fireworks/models/llama-v3p1-70b-instruct\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"chatcmpl-usage-fw-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"accounts/fireworks/models/llama-v3p1-70b-instruct\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"chatcmpl-usage-fw-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"accounts/fireworks/models/llama-v3p1-70b-instruct\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5,\"total_tokens\":15}}\n\n",
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
            "model": "fireworks-llama",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": true,
            "stream_options": {"include_usage": true},
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
async fn fireworks_request_with_seed() {
    let mock = MockServer::start();
    let config = proxy_config(&mock.base_url(), None);
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .json_body_partial(json!({"seed": 42}).to_string());
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "id": "chatcmpl-seed-fw",
                "object": "chat.completion",
                "created": 1,
                "model": "accounts/fireworks/models/llama-v3p1-70b-instruct",
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
            "model": "fireworks-llama",
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
async fn fireworks_request_with_max_completion_tokens() {
    let mock = MockServer::start();
    let config = proxy_config(&mock.base_url(), None);
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .json_body_partial(json!({"max_completion_tokens": 100}).to_string());
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "id": "chatcmpl-mct-fw",
                "object": "chat.completion",
                "created": 1,
                "model": "accounts/fireworks/models/llama-v3p1-70b-instruct",
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
            "model": "fireworks-llama",
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
async fn fireworks_request_with_stop_sequences() {
    let mock = MockServer::start();
    let config = proxy_config(&mock.base_url(), None);
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .json_body_partial(json!({"stop": ["\n", "done"]}).to_string());
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "id": "chatcmpl-stop-fw",
                "object": "chat.completion",
                "created": 1,
                "model": "accounts/fireworks/models/llama-v3p1-70b-instruct",
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
            "model": "fireworks-llama",
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
async fn fireworks_request_with_reasoning_effort() {
    let mock = MockServer::start();
    let config = proxy_config(&mock.base_url(), None);
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .json_body_partial(json!({"reasoning_effort": "high"}).to_string());
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "id": "chatcmpl-reason-fw",
                "object": "chat.completion",
                "created": 1,
                "model": "accounts/fireworks/models/llama-v3p1-70b-instruct",
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
            "model": "fireworks-llama",
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
async fn fireworks_request_with_sampling_params() {
    let mock = MockServer::start();
    let config = proxy_config(&mock.base_url(), None);
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
            .json_body(json!({
                "id": "chatcmpl-sampling-fw",
                "object": "chat.completion",
                "created": 1,
                "model": "accounts/fireworks/models/llama-v3p1-70b-instruct",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "sampled"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            }));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .json(&json!({
            "model": "fireworks-llama",
            "messages": [{"role": "user", "content": "Tune it"}],
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
async fn fireworks_auth_required() {
    let mock = MockServer::start();

    let config = proxy_config(&mock.base_url(), None);
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .header("authorization", "Bearer sk-test-key");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "id": "chatcmpl-auth-fw",
                "object": "chat.completion",
                "created": 1,
                "model": "accounts/fireworks/models/llama-v3p1-70b-instruct",
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
        .json(&chat_body())
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);
    upstream.assert_hits(1);
    handle.shutdown();

    let config2 = proxy_config(&mock.base_url(), Some("sk-master-secret"));
    let (handle2, _state2) = start_oxllm(config2).await.expect("server should start");

    let client2 = reqwest::Client::new();
    let resp2 = client2
        .post(handle2.url("/v1/chat/completions"))
        .json(&chat_body())
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp2.status(), 401);
    let body2: Value = resp2.json().await.expect("json");
    assert_eq!(body2["error"]["code"], "invalid_api_key");

    handle2.shutdown();
}

#[tokio::test]
async fn fireworks_upstream_4xx_propagates() {
    let mock = MockServer::start();
    let mut config = proxy_config(&mock.base_url(), None);
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
        .json(&chat_body())
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
async fn fireworks_upstream_5xx_propagates() {
    let mock = MockServer::start();
    let mut config = proxy_config(&mock.base_url(), None);
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
        .json(&chat_body())
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
async fn fireworks_model_not_found() {
    let mock = MockServer::start();
    let config = proxy_config(&mock.base_url(), None);
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
async fn fireworks_list_models() {
    let mock = MockServer::start();
    let config = proxy_config(&mock.base_url(), None);
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
    assert_eq!(data[0]["id"], "fireworks-llama");
    assert_eq!(data[0]["object"], "model");

    handle.shutdown();
}

#[tokio::test]
async fn fireworks_embedding_request() {
    let mock = MockServer::start();
    let config = proxy_config(&mock.base_url(), None);
    let (handle, _state) = start_oxllm(config).await.expect("server should start");

    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/embeddings")
            .header("authorization", "Bearer sk-test-key")
            .json_body_partial(
                json!({"model": "accounts/fireworks/models/llama-v3p1-70b-instruct", "input": "hello"})
                    .to_string(),
            );
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "object": "list",
                "data": [{
                    "object": "embedding",
                    "index": 0,
                    "embedding": [0.1, 0.2, 0.3]
                }],
                "model": "accounts/fireworks/models/llama-v3p1-70b-instruct",
                "usage": {"prompt_tokens": 1, "total_tokens": 1}
            }));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/embeddings"))
        .json(&json!({
            "model": "fireworks-llama",
            "input": "hello",
        }))
        .send()
        .await
        .expect("request should succeed");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("json");
    assert_eq!(body["object"], "list");
    assert!(body["data"].is_array(), "data should be an array");
    assert_eq!(body["data"][0]["embedding"][0], 0.1);

    upstream.assert_hits(1);
    handle.shutdown();
}
