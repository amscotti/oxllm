//! Live integration test against the Perplexity API.
//!
//! Requires `PERPLEXITY_API_KEY` to be set in the environment.
//! Run with: `PERPLEXITY_API_KEY=pplx-... cargo test --test perplexity_live --all-features -- --nocapture`
//!
//! Notes on Perplexity behavior observed against the live API:
//! - Provider calls are serialized by `serial_test` because Perplexity
//!   rate-limits concurrent requests. Default cargo test parallelism is safe.
//! - No `/models` endpoint exists (404); model names verified by direct calls.
//! - `n` > 1 is rejected ("Multiple samples are not supported") — tests never send `n`.
//! - `max_tokens` must be at least 16; `max_completion_tokens` is also accepted.
//! - `response_format.type` only accepts `json_schema` or `text` — `json_object` is rejected.
//! - Tool calling is only supported on `sonar-pro` and `sonar-reasoning-pro`.
//! - `sonar-reasoning` is deprecated; `sonar-reasoning-pro` is the current reasoner.

use oxllm_server::OxllmParams;
use oxllm_server::app::{build_router, build_shared_state};
use oxllm_server::config::types::{ConfigModel, GeneralSettings, OxllmSettings, ProxyConfig};
use serde_json::{Value, json};

const PERPLEXITY_API_KEY_ENV: &str = "PERPLEXITY_API_KEY";
const DEFAULT_MODEL: &str = "perplexity/sonar";
const SECOND_MODEL: &str = "perplexity/sonar-pro";
const REASONER_MODEL: &str = "perplexity/sonar-reasoning-pro";

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
    std::env::var(PERPLEXITY_API_KEY_ENV)
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
#[serial_test::serial(perplexity_live)]
async fn perplexity_non_streaming_chat() {
    let Some(key) = get_api_key() else {
        eprintln!("skipping: {PERPLEXITY_API_KEY_ENV} not set");
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for("sonar", DEFAULT_MODEL, &key)).await else {
        eprintln!("skipping: failed to start oxllm");
        return;
    };

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth("sk-test-key-master-1")
        .json(&json!({
            "model": "sonar",
            "messages": [{"role": "user", "content": "Say hello in one word"}],
            "max_tokens": 16,
        }))
        .send()
        .await
        .expect("request failed");

    assert_eq!(
        resp.status(),
        200,
        "Perplexity request failed: {}",
        resp.text().await.unwrap_or_default()
    );
    let body: Value = resp.json().await.expect("json");
    let content = body["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("");
    assert!(!content.is_empty(), "response had empty content");
    assert!(
        body["usage"].is_object(),
        "expected usage in response, got: {}",
        body
    );
    eprintln!("Perplexity response: {content}");
    eprintln!("usage: {}", body["usage"]);

    handle.shutdown();
}

#[tokio::test]
#[serial_test::serial(perplexity_live)]
async fn perplexity_streaming_chat() {
    let Some(key) = get_api_key() else {
        eprintln!("skipping: {PERPLEXITY_API_KEY_ENV} not set");
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for("sonar", DEFAULT_MODEL, &key)).await else {
        eprintln!("skipping: failed to start oxllm");
        return;
    };

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth("sk-test-key-master-1")
        .json(&json!({
            "model": "sonar",
            "messages": [{"role": "user", "content": "Count to 3"}],
            "stream": true,
            "max_tokens": 50,
        }))
        .send()
        .await
        .expect("request failed");

    assert_eq!(resp.status(), 200, "Perplexity streaming request failed");
    let body = resp.text().await.expect("body");
    assert!(body.contains("data: "), "expected SSE data, got: {body}");
    assert!(body.contains("[DONE]"), "expected [DONE] terminator");
    eprintln!("Perplexity streaming OK — received {} bytes", body.len());

    handle.shutdown();
}

#[tokio::test]
#[serial_test::serial(perplexity_live)]
async fn perplexity_tool_call() {
    let Some(key) = get_api_key() else {
        eprintln!("skipping: {PERPLEXITY_API_KEY_ENV} not set");
        return;
    };

    // Tool calling is only supported on sonar-pro / sonar-reasoning-pro.
    let Some(handle) = start_oxllm(proxy_config_for("sonar-pro", SECOND_MODEL, &key)).await else {
        eprintln!("skipping: failed to start oxllm");
        return;
    };

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth("sk-test-key-master-1")
        .json(&json!({
            "model": "sonar-pro",
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
            // sonar is search-grounded and nondeterministic about CHOOSING a
            // tool (it may answer from search results instead); force the
            // required-tool form so the tool_calls plumbing is what's tested.
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
            "max_tokens": 300,
        }))
        .send()
        .await
        .expect("request failed");

    assert_eq!(
        resp.status(),
        200,
        "Perplexity tool call failed: {}",
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
#[serial_test::serial(perplexity_live)]
async fn perplexity_json_object_mode() {
    let Some(key) = get_api_key() else {
        eprintln!("skipping: {PERPLEXITY_API_KEY_ENV} not set");
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for("sonar", DEFAULT_MODEL, &key)).await else {
        eprintln!("skipping: failed to start oxllm");
        return;
    };

    // Perplexity rejects response_format.type "json_object" (400) — it only
    // accepts "json_schema" or "text". Use json_schema with a strict schema
    // and verify the returned content is valid JSON.
    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth("sk-test-key-master-1")
        .json(&json!({
            "model": "sonar",
            "messages": [
                {"role": "user", "content": "Create a JSON object for a person named Alice who is 25 years old, with fields 'name' and 'age'."}
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "person",
                    "strict": true,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "age": {"type": "number"}
                        },
                        "required": ["name", "age"],
                        "additionalProperties": false
                    }
                }
            },
            "max_tokens": 200,
        }))
        .send()
        .await
        .expect("request failed");

    assert_eq!(
        resp.status(),
        200,
        "Perplexity JSON mode failed: {}",
        resp.text().await.unwrap_or_default()
    );
    let body: Value = resp.json().await.expect("json");
    let content = body["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("");
    assert!(!content.is_empty(), "response had empty content");
    let parsed: Value = serde_json::from_str(content).expect("response must be valid JSON");
    assert!(
        parsed.get("name").is_some(),
        "JSON must contain 'name' field"
    );
    assert!(parsed.get("age").is_some(), "JSON must contain 'age' field");

    handle.shutdown();
}

#[tokio::test]
#[serial_test::serial(perplexity_live)]
async fn perplexity_multiple_models() {
    let Some(api_key) = get_api_key() else {
        eprintln!("skipping: {PERPLEXITY_API_KEY_ENV} not set");
        return;
    };

    let models: Vec<(&str, &str)> = vec![("sonar", DEFAULT_MODEL), ("sonar-pro", SECOND_MODEL)];

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
                "max_tokens": 16,
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
        let body: Value = resp.json().await.expect("json");
        let content = body["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("");
        assert!(
            !content.is_empty(),
            "model {model_name} returned empty content"
        );
        eprintln!("{model_name}: {content}");
    }

    handle.shutdown();
}

#[tokio::test]
#[serial_test::serial(perplexity_live)]
async fn perplexity_streaming_with_include_usage() {
    let Some(key) = get_api_key() else {
        eprintln!("skipping: {PERPLEXITY_API_KEY_ENV} not set");
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for("sonar", DEFAULT_MODEL, &key)).await else {
        eprintln!("skipping: failed to start oxllm");
        return;
    };

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth("sk-test-key-master-1")
        .json(&json!({
            "model": "sonar",
            "messages": [{"role": "user", "content": "Count to 5"}],
            "stream": true,
            "stream_options": {"include_usage": true},
            "max_tokens": 50,
        }))
        .send()
        .await
        .expect("request failed");

    assert_eq!(resp.status(), 200, "Perplexity streaming with usage failed");
    let body = resp.text().await.expect("body");
    assert!(body.contains("data: "), "expected SSE data, got: {body}");
    assert!(body.contains("[DONE]"), "expected [DONE] terminator");
    assert!(body.contains("usage"), "expected usage in stream response");
    assert!(body.contains("prompt_tokens"), "expected prompt_tokens");
    assert!(
        body.contains("completion_tokens"),
        "expected completion_tokens"
    );

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
    assert!(!full_text.is_empty(), "no text collected from stream");
    eprintln!("Streamed text ({}) chars: {full_text}", full_text.len());

    handle.shutdown();
}

#[tokio::test]
#[serial_test::serial(perplexity_live)]
async fn perplexity_with_seed() {
    let Some(key) = get_api_key() else {
        eprintln!("skipping: {PERPLEXITY_API_KEY_ENV} not set");
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for("sonar", DEFAULT_MODEL, &key)).await else {
        eprintln!("skipping: failed to start oxllm");
        return;
    };

    let client = reqwest::Client::new();
    for i in 0..2 {
        let resp = client
            .post(handle.url("/v1/chat/completions"))
            .bearer_auth("sk-test-key-master-1")
            .json(&json!({
                "model": "sonar",
                "messages": [{"role": "user", "content": "Say hello in one word"}],
                "seed": 42,
                "max_tokens": 16,
            }))
            .send()
            .await
            .expect("request failed");

        assert_eq!(resp.status(), 200, "request {i} failed");
        let body: Value = resp.json().await.expect("json");
        let content = body["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("");
        assert!(!content.is_empty(), "request {i} returned empty content");
        eprintln!("Request {i}: {content}");
    }

    handle.shutdown();
}

#[tokio::test]
#[serial_test::serial(perplexity_live)]
async fn perplexity_max_tokens() {
    let Some(key) = get_api_key() else {
        eprintln!("skipping: {PERPLEXITY_API_KEY_ENV} not set");
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for("sonar", DEFAULT_MODEL, &key)).await else {
        eprintln!("skipping: failed to start oxllm");
        return;
    };

    // Perplexity enforces max_tokens >= 16 (10 is rejected with 400).
    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth("sk-test-key-master-1")
        .json(&json!({
            "model": "sonar",
            "messages": [{"role": "user", "content": "Write a very long essay about AI"}],
            "max_tokens": 16,
        }))
        .send()
        .await
        .expect("request failed");

    assert_eq!(
        resp.status(),
        200,
        "Perplexity max tokens failed: {}",
        resp.text().await.unwrap_or_default()
    );
    let body: Value = resp.json().await.expect("json");
    let content = body["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("");
    eprintln!(
        "Short response ({len} chars): {content}",
        len = content.len()
    );
    assert!(!content.is_empty(), "response had empty content");
    assert!(
        content.len() < 200,
        "response should be short with max_tokens=16, got {} chars: {content}",
        content.len()
    );

    handle.shutdown();
}

#[tokio::test]
#[serial_test::serial(perplexity_live)]
async fn perplexity_long_context() {
    let Some(key) = get_api_key() else {
        eprintln!("skipping: {PERPLEXITY_API_KEY_ENV} not set");
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for("sonar", DEFAULT_MODEL, &key)).await else {
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
        paragraph.repeat(40)
    );

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth("sk-test-key-master-1")
        .json(&json!({
            "model": "sonar",
            "messages": [{"role": "user", "content": long_prompt}],
            "max_tokens": 100,
        }))
        .send()
        .await
        .expect("request failed");

    assert_eq!(
        resp.status(),
        200,
        "Perplexity long context failed: {}",
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
#[serial_test::serial(perplexity_live)]
async fn perplexity_invalid_auth_returns_401() {
    let Some(key) = get_api_key() else {
        eprintln!("skipping: {PERPLEXITY_API_KEY_ENV} not set");
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for("sonar", DEFAULT_MODEL, &key)).await else {
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
#[serial_test::serial(perplexity_live)]
async fn perplexity_nonexistent_model() {
    let Some(key) = get_api_key() else {
        eprintln!("skipping: {PERPLEXITY_API_KEY_ENV} not set");
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for("sonar", DEFAULT_MODEL, &key)).await else {
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
        "expected 400 or 404 for nonexistent model, got {status}"
    );

    handle.shutdown();
}

#[tokio::test]
#[serial_test::serial(perplexity_live)]
async fn perplexity_health_endpoint() {
    let Some(key) = get_api_key() else {
        eprintln!("skipping: {PERPLEXITY_API_KEY_ENV} not set");
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for("sonar", DEFAULT_MODEL, &key)).await else {
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
#[serial_test::serial(perplexity_live)]
async fn perplexity_models_endpoint() {
    let Some(key) = get_api_key() else {
        eprintln!("skipping: {PERPLEXITY_API_KEY_ENV} not set");
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for("sonar", DEFAULT_MODEL, &key)).await else {
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
        models.contains(&"sonar"),
        "expected 'sonar' in model list, got: {models:?}"
    );

    handle.shutdown();
}

#[tokio::test]
#[serial_test::serial(perplexity_live)]
async fn perplexity_with_seed_and_temperature() {
    let Some(key) = get_api_key() else {
        eprintln!("skipping: {PERPLEXITY_API_KEY_ENV} not set");
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for("sonar", DEFAULT_MODEL, &key)).await else {
        eprintln!("skipping: failed to start oxllm");
        return;
    };

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth("sk-test-key-master-1")
        .json(&json!({
            "model": "sonar",
            "messages": [{"role": "user", "content": "Say hello in one word"}],
            "seed": 42,
            "temperature": 0.5,
            "max_tokens": 16,
        }))
        .send()
        .await
        .expect("request failed");

    assert_eq!(
        resp.status(),
        200,
        "Perplexity seed+temperature failed: {}",
        resp.text().await.unwrap_or_default()
    );
    let body: Value = resp.json().await.expect("json");
    let content = body["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("");
    assert!(!content.is_empty(), "response had empty content");
    eprintln!("Seed+temperature response: {content}");

    handle.shutdown();
}

#[tokio::test]
#[serial_test::serial(perplexity_live)]
async fn perplexity_reasoner_model() {
    let Some(key) = get_api_key() else {
        eprintln!("skipping: {PERPLEXITY_API_KEY_ENV} not set");
        return;
    };

    // sonar-reasoning-pro is the current reasoner model (sonar-reasoning is
    // deprecated and rejected with 400). If it becomes unavailable, the
    // request will fail loudly rather than being silently skipped.
    let Some(handle) = start_oxllm(proxy_config_for("reasoner", REASONER_MODEL, &key)).await else {
        eprintln!("skipping: failed to start oxllm");
        return;
    };

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth("sk-test-key-master-1")
        .json(&json!({
            "model": "reasoner",
            "messages": [{"role": "user", "content": "What is 2+2? Answer with just the number."}],
            // Reasoning is part of the output budget; leave room for visible
            // text instead of testing a randomly truncated thinking preamble.
            "reasoning_effort": "low",
            "max_tokens": 2048,
        }))
        .send()
        .await
        .expect("request failed");

    assert_eq!(
        resp.status(),
        200,
        "Perplexity reasoner failed: {}",
        resp.text().await.unwrap_or_default()
    );
    let body: Value = resp.json().await.expect("json");
    let content = body["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("");
    assert!(!content.is_empty(), "response had empty content: {body}");
    let reasoning = body["choices"][0]["message"]["reasoning_content"]
        .as_str()
        .unwrap_or("");
    if reasoning.is_empty() {
        eprintln!("reasoner response had no reasoning_content field");
    } else {
        eprintln!("reasoner reasoning_content: {reasoning}");
    }
    eprintln!("Reasoner model response: {content}");

    handle.shutdown();
}

#[tokio::test]
#[serial_test::serial(perplexity_live)]
async fn perplexity_reasoner_streaming() {
    let Some(key) = get_api_key() else {
        eprintln!("skipping: {PERPLEXITY_API_KEY_ENV} not set");
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for("reasoner", REASONER_MODEL, &key)).await else {
        eprintln!("skipping: failed to start oxllm");
        return;
    };

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth("sk-test-key-master-1")
        .json(&json!({
            "model": "reasoner",
            "messages": [{"role": "user", "content": "What is 2+2? Answer with just the number."}],
            "stream": true,
            "reasoning_effort": "low",
            "max_tokens": 2048,
        }))
        .send()
        .await
        .expect("request failed");

    assert_eq!(resp.status(), 200, "Perplexity reasoner streaming failed");
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
    assert!(!full_text.is_empty(), "no text collected from stream");
    eprintln!(
        "Reasoner streamed text ({}) chars: {full_text}",
        full_text.len()
    );

    handle.shutdown();
}
