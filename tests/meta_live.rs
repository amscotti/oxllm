//! Live integration test against the Meta API.
//!
//! Requires `META_API_KEY` to be set in the environment.
//! Run with: `META_API_KEY=... cargo test --test meta_live --all-features -- --nocapture`
//!
//! NOTE: muse-spark-1.1 is a REASONING model: it spends tokens on reasoning
//! before answering. With small max_tokens it returns `content: null` with
//! `finish_reason: "length"` and reasoning_tokens set in usage. Tests that
//! need content must leave room for both thinking and visible output. Streaming
//! and JSON-mode checks use 2048 tokens and fail if content/usage is missing;
//! the separate small-budget test intentionally only checks the HTTP contract.

use oxllm_server::OxllmParams;
use oxllm_server::app::{build_router, build_shared_state};
use oxllm_server::config::types::{ConfigModel, GeneralSettings, OxllmSettings, ProxyConfig};
use serde_json::{Value, json};

const META_API_KEY_ENV: &str = "META_API_KEY";
const DEFAULT_MODEL: &str = "meta/muse-spark-1.1";

struct OxllmHandle {
    addr: std::net::SocketAddr,
    shutdown_tx: tokio::sync::oneshot::Sender<()>,
}

impl OxllmHandle {
    fn url(&self, path: &str) -> String {
        format!("http://{}{path}", self.addr)
    }
    fn shutdown(self) {
        let _ = self.shutdown_tx.send(());
    }
}

fn get_api_key() -> Option<String> {
    std::env::var(META_API_KEY_ENV)
        .ok()
        .filter(|k| !k.is_empty())
}

async fn start_oxllm(config: ProxyConfig) -> Option<OxllmHandle> {
    let shared = build_shared_state(config).expect("configured live proxy state");
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind live proxy");
    let addr = listener.local_addr().expect("live proxy address");
    let router = build_router(shared.clone());
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    tokio::spawn(async move {
        axum::serve(
            listener,
            router.into_make_service_with_connect_info::<std::net::SocketAddr>(),
        )
        .with_graceful_shutdown(async {
            let _ = shutdown_rx.await;
        })
        .await
        .expect("serve live proxy");
    });
    Some(OxllmHandle { addr, shutdown_tx })
}

fn proxy_config_for(model_name: &str, model_key: &str, key: &str) -> ProxyConfig {
    ProxyConfig {
        model_list: vec![ConfigModel {
            model_name: model_name.to_string(),
            oxllm_params: OxllmParams {
                model: model_key.to_string(),
                api_key: Some(key.to_string()),
                api_base: None,
                timeout: Some(60.0),
                stream_timeout: Some(60),
                ..Default::default()
            },
            ..Default::default()
        }],
        oxllm_settings: OxllmSettings::default(),
        general_settings: GeneralSettings {
            master_key: Some("sk-test-key-master-1".to_string()),
            allow_private_api_base: Some(true),
            ..Default::default()
        },
        router_settings: Default::default(),
        extra: serde_json::Map::new(),
    }
}

#[tokio::test]
async fn meta_non_streaming_chat() {
    let Some(key) = get_api_key() else {
        eprintln!("skipping: {META_API_KEY_ENV} not set");
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for("muse", DEFAULT_MODEL, &key)).await else {
        eprintln!("skipping: failed to start oxllm");
        return;
    };

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth("sk-test-key-master-1")
        .json(&json!({
            "model": "muse",
            "messages": [{"role": "user", "content": "Say hello in one word"}],
            "max_tokens": 512,
        }))
        .send()
        .await
        .expect("request failed");

    assert_eq!(
        resp.status(),
        200,
        "Meta request failed: {}",
        resp.text().await.unwrap_or_default()
    );
    let body: Value = resp.json().await.expect("json");
    let content = body["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("");
    assert!(!content.is_empty(), "response had empty content");
    assert!(
        body["usage"]["total_tokens"].as_u64().is_some(),
        "expected usage to be present"
    );
    eprintln!("Meta response: {content}");

    handle.shutdown();
}

#[tokio::test]
async fn meta_streaming_chat() {
    let Some(key) = get_api_key() else {
        eprintln!("skipping: {META_API_KEY_ENV} not set");
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for("muse", DEFAULT_MODEL, &key)).await else {
        eprintln!("skipping: failed to start oxllm");
        return;
    };

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth("sk-test-key-master-1")
        .json(&json!({
            "model": "muse",
            "messages": [{"role": "user", "content": "Count to 3"}],
            "stream": true,
            "max_tokens": 2048,
        }))
        .send()
        .await
        .expect("request failed");

    assert_eq!(resp.status(), 200, "Meta streaming request failed");
    let body = resp.text().await.expect("body");
    assert!(body.contains("data: "), "expected SSE data, got: {body}");
    assert!(body.contains("[DONE]"), "expected [DONE] terminator");

    let full_text: String = body
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .take_while(|data| *data != "[DONE]")
        .filter_map(|data| serde_json::from_str::<Value>(data).ok())
        .filter_map(|chunk| {
            chunk["choices"][0]["delta"]["content"]
                .as_str()
                .map(|s| s.to_string())
        })
        .collect();
    assert!(
        !full_text.is_empty(),
        "expected visible streamed content: {body}"
    );
    eprintln!("Meta streaming OK — streamed {} chars", full_text.len());

    handle.shutdown();
}

#[tokio::test]
async fn meta_tool_call() {
    let Some(key) = get_api_key() else {
        eprintln!("skipping: {META_API_KEY_ENV} not set");
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for("muse", DEFAULT_MODEL, &key)).await else {
        eprintln!("skipping: failed to start oxllm");
        return;
    };

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth("sk-test-key-master-1")
        .json(&json!({
            "model": "muse",
            "messages": [
                {"role": "system", "content": "You have a function get_weather(city: string). Call it for Paris."},
                {"role": "user", "content": "What is the weather in Paris?"}
            ],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "The city name"
                            }
                        },
                        "required": ["city"]
                    }
                }
            }],
            "tool_choice": "auto",
            "max_tokens": 1024,
        }))
        .send()
        .await
        .expect("request failed");

    assert_eq!(
        resp.status(),
        200,
        "Meta tool call failed: {}",
        resp.text().await.unwrap_or_default()
    );
    let body: Value = resp.json().await.expect("json");
    let tool_calls = body["choices"][0]["message"]["tool_calls"]
        .as_array()
        .expect("expected tool_calls array");
    assert!(!tool_calls.is_empty(), "at least one tool call");
    assert!(
        tool_calls[0]["id"].as_str().is_some_and(|s| !s.is_empty()),
        "expected non-empty id"
    );
    assert_eq!(tool_calls[0]["type"], "function");
    assert_eq!(tool_calls[0]["function"]["name"], "get_weather");

    handle.shutdown();
}

#[tokio::test]
async fn meta_json_object_mode() {
    let Some(key) = get_api_key() else {
        eprintln!("skipping: {META_API_KEY_ENV} not set");
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for("muse", DEFAULT_MODEL, &key)).await else {
        eprintln!("skipping: failed to start oxllm");
        return;
    };

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth("sk-test-key-master-1")
        .json(&json!({
            "model": "muse",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that always responds with valid JSON."},
                {"role": "user", "content": "Create a JSON object for a person named Alice who is 25 years old, with fields 'name' and 'age'."}
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": 2048,
        }))
        .send()
        .await
        .expect("request failed");

    assert_eq!(
        resp.status(),
        200,
        "Meta JSON mode failed: {}",
        resp.text().await.unwrap_or_default()
    );
    let body: Value = resp.json().await.expect("json");
    let content = body["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("");
    assert!(!content.is_empty(), "expected visible JSON content: {body}");
    let parsed: Value = serde_json::from_str(content).expect("response must be valid JSON");
    assert!(
        parsed.get("name").is_some(),
        "JSON must contain 'name' field"
    );
    assert!(parsed.get("age").is_some(), "JSON must contain 'age' field");

    handle.shutdown();
}

#[tokio::test]
async fn meta_multiple_models() {
    let Some(api_key) = get_api_key() else {
        eprintln!("skipping: {META_API_KEY_ENV} not set");
        return;
    };

    let models: Vec<(&str, &str)> = vec![("muse-a", DEFAULT_MODEL), ("muse-b", DEFAULT_MODEL)];

    let config = ProxyConfig {
        model_list: models
            .iter()
            .map(|(name, model_key)| ConfigModel {
                model_name: name.to_string(),
                oxllm_params: OxllmParams {
                    model: model_key.to_string(),
                    api_key: Some(api_key.clone()),
                    api_base: None,
                    timeout: Some(60.0),
                    stream_timeout: Some(60),
                    ..Default::default()
                },
                ..Default::default()
            })
            .collect(),
        oxllm_settings: OxllmSettings::default(),
        general_settings: GeneralSettings {
            master_key: Some("sk-test-key-master-1".to_string()),
            allow_private_api_base: Some(true),
            ..Default::default()
        },
        router_settings: Default::default(),
        extra: serde_json::Map::new(),
    };

    let Some(handle) = start_oxllm(config).await else {
        eprintln!("skipping: failed to start oxllm");
        return;
    };

    let client = reqwest::Client::new();
    for (model_name, _) in &models {
        let resp = client
            .post(handle.url("/v1/chat/completions"))
            .bearer_auth("sk-test-key-master-1")
            .json(&json!({
                "model": model_name,
                "messages": [{"role": "user", "content": "Say hello in one word"}],
                "max_tokens": 512,
            }))
            .send()
            .await
            .expect("request failed");

        assert_eq!(
            resp.status(),
            200,
            "model {model_name} failed: {}",
            resp.text().await.unwrap_or_default()
        );
        eprintln!("{model_name}: OK (status 200)");
    }

    handle.shutdown();
}

#[tokio::test]
async fn meta_streaming_with_include_usage() {
    let Some(key) = get_api_key() else {
        eprintln!("skipping: {META_API_KEY_ENV} not set");
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for("muse", DEFAULT_MODEL, &key)).await else {
        eprintln!("skipping: failed to start oxllm");
        return;
    };

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth("sk-test-key-master-1")
        .json(&json!({
            "model": "muse",
            "messages": [{"role": "user", "content": "Count to 5"}],
            "stream": true,
            "stream_options": {"include_usage": true},
            "max_tokens": 2048,
        }))
        .send()
        .await
        .expect("request failed");

    assert_eq!(resp.status(), 200, "Meta streaming with usage failed");
    let body = resp.text().await.expect("body");
    assert!(body.contains("data: "), "expected SSE data, got: {body}");
    assert!(body.contains("[DONE]"), "expected [DONE] terminator");

    let usage_chunks: Vec<Value> = body
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .take_while(|data| *data != "[DONE]")
        .filter_map(|data| serde_json::from_str::<Value>(data).ok())
        .filter(|chunk| {
            chunk["usage"].is_object() && chunk["choices"].as_array().is_some_and(Vec::is_empty)
        })
        .collect();

    let full_text: String = body
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .take_while(|data| *data != "[DONE]")
        .filter_map(|data| serde_json::from_str::<Value>(data).ok())
        .filter_map(|chunk| {
            chunk["choices"][0]["delta"]["content"]
                .as_str()
                .map(|s| s.to_string())
        })
        .collect();
    assert!(
        !full_text.is_empty(),
        "expected visible streamed content: {body}"
    );
    assert_eq!(
        usage_chunks.len(),
        1,
        "include_usage must expose one final usage chunk: {body}"
    );
    let usage = &usage_chunks[0]["usage"];
    assert!(
        usage["prompt_tokens"].as_u64().is_some(),
        "expected prompt_tokens in usage"
    );
    assert!(
        usage["completion_tokens"].as_u64().is_some(),
        "expected completion_tokens in usage"
    );

    eprintln!("Streamed text ({}) chars: {full_text}", full_text.len());

    handle.shutdown();
}

#[tokio::test]
async fn meta_with_seed() {
    let Some(key) = get_api_key() else {
        eprintln!("skipping: {META_API_KEY_ENV} not set");
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for("muse", DEFAULT_MODEL, &key)).await else {
        eprintln!("skipping: failed to start oxllm");
        return;
    };

    let client = reqwest::Client::new();
    for i in 0..2 {
        let resp = client
            .post(handle.url("/v1/chat/completions"))
            .bearer_auth("sk-test-key-master-1")
            .json(&json!({
                "model": "muse",
                "messages": [{"role": "user", "content": "Say hello in one word"}],
                "seed": 42,
                "max_tokens": 512,
            }))
            .send()
            .await
            .expect("request failed");

        assert_eq!(
            resp.status(),
            200,
            "request {i} failed: {}",
            resp.text().await.unwrap_or_default()
        );
        eprintln!("Request {i}: OK (status 200)");
    }

    handle.shutdown();
}

#[tokio::test]
async fn meta_with_stop_sequence() {
    let Some(key) = get_api_key() else {
        eprintln!("skipping: {META_API_KEY_ENV} not set");
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for("muse", DEFAULT_MODEL, &key)).await else {
        eprintln!("skipping: failed to start oxllm");
        return;
    };

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth("sk-test-key-master-1")
        .json(&json!({
            "model": "muse",
            "messages": [{"role": "user", "content": "Say 'Hello world' exactly"}],
            "stop": ["world"],
            "max_tokens": 512,
        }))
        .send()
        .await
        .expect("request failed");

    if resp.status() != 200 {
        eprintln!(
            "skipping: upstream does not support stop sequences (status {})",
            resp.status()
        );
        handle.shutdown();
        return;
    }
    let body: Value = resp.json().await.expect("json");
    let content = body["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("");
    eprintln!("Response with stop sequence: {content}");
    assert!(!content.is_empty(), "response had empty content");
    assert!(
        !content.contains("world"),
        "response should not contain the stop sequence 'world', got: {content}"
    );

    handle.shutdown();
}

#[tokio::test]
async fn meta_max_completion_tokens_small_returns_200() {
    let Some(key) = get_api_key() else {
        eprintln!("skipping: {META_API_KEY_ENV} not set");
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for("muse", DEFAULT_MODEL, &key)).await else {
        eprintln!("skipping: failed to start oxllm");
        return;
    };

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth("sk-test-key-master-1")
        .json(&json!({
            "model": "muse",
            "messages": [{"role": "user", "content": "Write a very long essay about AI"}],
            "max_completion_tokens": 10,
        }))
        .send()
        .await
        .expect("request failed");

    assert_eq!(
        resp.status(),
        200,
        "Meta max_completion_tokens failed: {}",
        resp.text().await.unwrap_or_default()
    );
    let body: Value = resp.json().await.expect("json");
    let finish_reason = body["choices"][0]["finish_reason"].as_str().unwrap_or("");
    eprintln!(
        "Small max_completion_tokens response: finish_reason={finish_reason} (content may be null due to reasoning)"
    );

    handle.shutdown();
}

#[tokio::test]
async fn meta_max_tokens_small_returns_200() {
    let Some(key) = get_api_key() else {
        eprintln!("skipping: {META_API_KEY_ENV} not set");
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for("muse", DEFAULT_MODEL, &key)).await else {
        eprintln!("skipping: failed to start oxllm");
        return;
    };

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth("sk-test-key-master-1")
        .json(&json!({
            "model": "muse",
            "messages": [{"role": "user", "content": "Write a very long essay about AI"}],
            "max_tokens": 10,
        }))
        .send()
        .await
        .expect("request failed");

    assert_eq!(
        resp.status(),
        200,
        "Meta max_tokens failed: {}",
        resp.text().await.unwrap_or_default()
    );
    let body: Value = resp.json().await.expect("json");
    let finish_reason = body["choices"][0]["finish_reason"].as_str().unwrap_or("");
    eprintln!(
        "Small max_tokens response: finish_reason={finish_reason} (content may be null due to reasoning)"
    );

    handle.shutdown();
}

#[tokio::test]
async fn meta_long_context() {
    let Some(key) = get_api_key() else {
        eprintln!("skipping: {META_API_KEY_ENV} not set");
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for("muse", DEFAULT_MODEL, &key)).await else {
        eprintln!("skipping: failed to start oxllm");
        return;
    };

    let paragraph = "The history of artificial intelligence dates back to ancient times, \
        when philosophers contemplated the nature of thought and intelligence. \
        In the 1950s, the field of AI was formally established, and researchers began \
        developing programs that could solve problems and play games. Over the decades, \
        AI has evolved from simple rule-based systems to complex neural networks that \
        can learn from data. Today, AI powers everything from virtual assistants to \
        self-driving cars, and continues to advance at a rapid pace. ";
    let long_prompt = format!(
        "{}Based on the above text, what is one major milestone in AI history?",
        paragraph.repeat(30)
    );

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth("sk-test-key-master-1")
        .json(&json!({
            "model": "muse",
            "messages": [{"role": "user", "content": long_prompt}],
            "max_tokens": 1024,
        }))
        .send()
        .await
        .expect("request failed");

    assert_eq!(
        resp.status(),
        200,
        "Meta long context failed: {}",
        resp.text().await.unwrap_or_default()
    );
    let body: Value = resp.json().await.expect("json");
    let content = body["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("");
    assert!(!content.is_empty(), "response had empty content");
    assert!(
        content.len() > 20,
        "response seems too short for context-based answer: {content}"
    );
    eprintln!(
        "Long context response ({len} chars): {content}",
        len = content.len()
    );

    handle.shutdown();
}

#[tokio::test]
async fn meta_invalid_auth_returns_401() {
    let Some(key) = get_api_key() else {
        eprintln!("skipping: {META_API_KEY_ENV} not set");
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for("muse", DEFAULT_MODEL, &key)).await else {
        eprintln!("skipping: failed to start oxllm");
        return;
    };

    let client = reqwest::Client::new();
    let resp = client
        .get(handle.url("/v1/models"))
        .bearer_auth("wrong-key")
        .send()
        .await
        .expect("request failed");

    assert_eq!(
        resp.status(),
        401,
        "expected 401 with wrong auth, got {}",
        resp.status()
    );

    handle.shutdown();
}

#[tokio::test]
async fn meta_nonexistent_model() {
    let Some(key) = get_api_key() else {
        eprintln!("skipping: {META_API_KEY_ENV} not set");
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for("muse", DEFAULT_MODEL, &key)).await else {
        eprintln!("skipping: failed to start oxllm");
        return;
    };

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth("sk-test-key-master-1")
        .json(&json!({
            "model": "does-not-exist",
            "messages": [{"role": "user", "content": "Hello"}],
        }))
        .send()
        .await
        .expect("request failed");

    let status = resp.status();
    assert!(
        status == 400 || status == 404,
        "expected 400 or 404 for nonexistent model, got {status}: {}",
        resp.text().await.unwrap_or_default()
    );

    handle.shutdown();
}

#[tokio::test]
async fn meta_health_endpoint() {
    let Some(key) = get_api_key() else {
        eprintln!("skipping: {META_API_KEY_ENV} not set");
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for("muse", DEFAULT_MODEL, &key)).await else {
        eprintln!("skipping: failed to start oxllm");
        return;
    };

    let client = reqwest::Client::new();
    let resp = client
        .get(handle.url("/health"))
        .send()
        .await
        .expect("request failed");

    assert_eq!(resp.status(), 200, "expected 200 from /health");

    handle.shutdown();
}

#[tokio::test]
async fn meta_models_endpoint() {
    let Some(key) = get_api_key() else {
        eprintln!("skipping: {META_API_KEY_ENV} not set");
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for("muse", DEFAULT_MODEL, &key)).await else {
        eprintln!("skipping: failed to start oxllm");
        return;
    };

    let client = reqwest::Client::new();
    let resp = client
        .get(handle.url("/v1/models"))
        .bearer_auth("sk-test-key-master-1")
        .send()
        .await
        .expect("request failed");

    assert_eq!(resp.status(), 200, "expected 200 from /v1/models");
    let body: Value = resp.json().await.expect("json");
    let models: Vec<&str> = body["data"]
        .as_array()
        .expect("data must be an array")
        .iter()
        .filter_map(|m| m["id"].as_str())
        .collect();
    assert!(
        models.contains(&"muse"),
        "expected 'muse' in model list, got: {models:?}"
    );

    handle.shutdown();
}

#[tokio::test]
async fn meta_reasoning_tokens_in_usage() {
    let Some(key) = get_api_key() else {
        eprintln!("skipping: {META_API_KEY_ENV} not set");
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for("muse", DEFAULT_MODEL, &key)).await else {
        eprintln!("skipping: failed to start oxllm");
        return;
    };

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth("sk-test-key-master-1")
        .json(&json!({
            "model": "muse",
            "messages": [{"role": "user", "content": "Think step by step: what is 17 times 23?"}],
            "max_tokens": 512,
        }))
        .send()
        .await
        .expect("request failed");

    assert_eq!(
        resp.status(),
        200,
        "Meta reasoning usage failed: {}",
        resp.text().await.unwrap_or_default()
    );
    let body: Value = resp.json().await.expect("json");
    let usage = &body["usage"];
    let details = usage["completion_tokens_details"].as_object();
    eprintln!("usage: {usage}");
    if let Some(details) = details {
        if let Some(reasoning_tokens) = details.get("reasoning_tokens").and_then(|v| v.as_u64()) {
            eprintln!("reasoning_tokens = {reasoning_tokens}");
            assert!(
                reasoning_tokens > 0,
                "expected reasoning_tokens > 0 for a reasoning model"
            );
        } else {
            eprintln!("reasoning_tokens field not present, skipping assertion");
        }
    } else {
        eprintln!("completion_tokens_details not present, skipping assertion");
    }

    handle.shutdown();
}

#[tokio::test]
async fn meta_seed_and_temperature() {
    let Some(key) = get_api_key() else {
        eprintln!("skipping: {META_API_KEY_ENV} not set");
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for("muse", DEFAULT_MODEL, &key)).await else {
        eprintln!("skipping: failed to start oxllm");
        return;
    };

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth("sk-test-key-master-1")
        .json(&json!({
            "model": "muse",
            "messages": [{"role": "user", "content": "Say hello in one word"}],
            "seed": 42,
            "temperature": 0.5,
            "max_tokens": 512,
        }))
        .send()
        .await
        .expect("request failed");

    assert_eq!(
        resp.status(),
        200,
        "Meta seed+temperature failed: {}",
        resp.text().await.unwrap_or_default()
    );

    handle.shutdown();
}
