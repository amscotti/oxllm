//! Live integration test against the Fireworks AI API.
//!
//! Requires `FIREWORKS_API_KEY` to be set in the environment.
//! Run with: `FIREWORKS_API_KEY=fw_... cargo test --test fireworks_live -- --nocapture`

use oxllm_server::OxllmParams;
use oxllm_server::app::{build_router, build_shared_state};
use oxllm_server::config::types::{ConfigModel, GeneralSettings, OxllmSettings, ProxyConfig};
use serde_json::{Value, json};

const FIREWORKS_API_KEY_ENV: &str = "FIREWORKS_API_KEY";
// Fireworks retires undated model aliases, and most of the current catalog
// (deepseek-v4-flash-0731, glm-5p3-flash, qwen3p7-plus, ...) is reasoning-first:
// with small max_tokens the budget is consumed by reasoning_content and
// `content` stays empty. kimi-k3 emits plain content immediately, which the
// short-prompt assertions here need.
const DEFAULT_MODEL: &str = "accounts/fireworks/models/kimi-k3";
const FAST_MODEL: &str = "accounts/fireworks/models/glm-5p3-flash";
const EMBEDDING_MODEL: &str = "accounts/fireworks/models/qwen3-embedding-8b";

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
    std::env::var(FIREWORKS_API_KEY_ENV)
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

fn proxy_config(key: &str) -> ProxyConfig {
    ProxyConfig {
        model_list: vec![ConfigModel {
            model_name: "llama".to_string(),
            oxllm_params: OxllmParams {
                model: format!("fireworks_ai/{DEFAULT_MODEL}"),
                api_key: Some(key.to_string()),
                api_base: None,
                timeout: Some(30.0),
                stream_timeout: Some(30),
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

fn proxy_config_for(model_name: &str, model_key: &str) -> ProxyConfig {
    let key = get_api_key().expect("FIREWORKS_API_KEY must be set");
    ProxyConfig {
        model_list: vec![ConfigModel {
            model_name: model_name.to_string(),
            oxllm_params: OxllmParams {
                model: model_key.to_string(),
                api_key: Some(key),
                api_base: None,
                timeout: Some(30.0),
                stream_timeout: Some(30),
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
async fn fireworks_non_streaming_chat() {
    let Some(_key) = get_api_key() else {
        eprintln!("skipping: {FIREWORKS_API_KEY_ENV} not set");
        return;
    };

    let Some(handle) = start_oxllm(proxy_config(&_key)).await else {
        eprintln!("skipping: failed to start oxllm");
        return;
    };

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth("sk-test-key-master-1")
        .json(&json!({
            "model": "llama",
            "messages": [{"role": "user", "content": "Say hello in one word"}],
            "max_tokens": 10,
        }))
        .send()
        .await
        .expect("request failed");

    assert_eq!(
        resp.status(),
        200,
        "Fireworks request failed: {}",
        resp.text().await.unwrap_or_default()
    );
    let body: Value = resp.json().await.expect("json");
    let content = body["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("");
    assert!(!content.is_empty(), "response had empty content");
    eprintln!("Fireworks response: {content}");

    handle.shutdown();
}

#[tokio::test]
async fn fireworks_tool_call() {
    let Some(_key) = get_api_key() else {
        eprintln!("skipping: {FIREWORKS_API_KEY_ENV} not set");
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for(
        "llama",
        &format!("fireworks_ai/{DEFAULT_MODEL}"),
    ))
    .await
    else {
        eprintln!("skipping: failed to start oxllm");
        return;
    };

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth("sk-test-key-master-1")
        .json(&json!({
            "model": "llama",
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
            // Verify tool plumbing independently of the model's decision to
            // answer in text instead of choosing a tool.
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
        }))
        .send()
        .await
        .expect("request failed");

    assert_eq!(
        resp.status(),
        200,
        "Fireworks tool call failed: {}",
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
async fn fireworks_json_object_mode() {
    let Some(_key) = get_api_key() else {
        eprintln!("skipping: {FIREWORKS_API_KEY_ENV} not set");
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for(
        "llama",
        &format!("fireworks_ai/{DEFAULT_MODEL}"),
    ))
    .await
    else {
        eprintln!("skipping: failed to start oxllm");
        return;
    };

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth("sk-test-key-master-1")
        .json(&json!({
            "model": "llama",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that always responds with valid JSON."},
                {"role": "user", "content": "Create a JSON object for a person named Alice who is 25 years old, with fields 'name' and 'age'."}
            ],
            "response_format": {"type": "json_object"}
        }))
        .send()
        .await
        .expect("request failed");

    assert_eq!(
        resp.status(),
        200,
        "Fireworks JSON mode failed: {}",
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
async fn fireworks_with_seed() {
    let Some(_key) = get_api_key() else {
        eprintln!("skipping: {FIREWORKS_API_KEY_ENV} not set");
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for(
        "llama",
        &format!("fireworks_ai/{DEFAULT_MODEL}"),
    ))
    .await
    else {
        eprintln!("skipping: failed to start oxllm");
        return;
    };

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth("sk-test-key-master-1")
        .json(&json!({
            "model": "llama",
            "messages": [{"role": "user", "content": "Say hello in one word"}],
            "seed": 42,
            "max_tokens": 10,
        }))
        .send()
        .await
        .expect("request failed");

    assert_eq!(
        resp.status(),
        200,
        "Fireworks request failed: {}",
        resp.text().await.unwrap_or_default()
    );
    let body: Value = resp.json().await.expect("json");
    let content = body["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("");
    assert!(!content.is_empty(), "response had empty content");
    eprintln!("Fireworks response (seed=42): {content}");

    handle.shutdown();
}

#[tokio::test]
async fn fireworks_with_stop_sequence() {
    let Some(_key) = get_api_key() else {
        eprintln!("skipping: {FIREWORKS_API_KEY_ENV} not set");
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for(
        "llama",
        &format!("fireworks_ai/{DEFAULT_MODEL}"),
    ))
    .await
    else {
        eprintln!("skipping: failed to start oxllm");
        return;
    };

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth("sk-test-key-master-1")
        .json(&json!({
            "model": "llama",
            "messages": [{"role": "user", "content": "Say 'Hello world' exactly"}],
            "stop": ["Hello"],
            "max_tokens": 20,
        }))
        .send()
        .await
        .expect("request failed");

    assert_eq!(
        resp.status(),
        200,
        "Fireworks stop sequence failed: {}",
        resp.text().await.unwrap_or_default()
    );
    let body: Value = resp.json().await.expect("json");
    let content = body["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("");
    eprintln!("Response with stop sequence: {content}");
    assert!(!content.is_empty(), "response had empty content");
    assert!(
        !content.contains("Hello"),
        "response should not contain the stop sequence 'Hello', got: {content}"
    );

    handle.shutdown();
}

#[tokio::test]
async fn fireworks_max_completion_tokens() {
    let Some(_key) = get_api_key() else {
        eprintln!("skipping: {FIREWORKS_API_KEY_ENV} not set");
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for(
        "llama",
        &format!("fireworks_ai/{DEFAULT_MODEL}"),
    ))
    .await
    else {
        eprintln!("skipping: failed to start oxllm");
        return;
    };

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth("sk-test-key-master-1")
        .json(&json!({
            "model": "llama",
            "messages": [{"role": "user", "content": "Write a very long essay about AI"}],
            "max_completion_tokens": 5,
        }))
        .send()
        .await
        .expect("request failed");

    assert_eq!(
        resp.status(),
        200,
        "Fireworks max tokens failed: {}",
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
        content.len() < 50,
        "response should be short with max_completion_tokens=5, got {} chars: {content}",
        content.len()
    );

    handle.shutdown();
}

#[tokio::test]
async fn fireworks_long_context() {
    let Some(_key) = get_api_key() else {
        eprintln!("skipping: {FIREWORKS_API_KEY_ENV} not set");
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for(
        "llama",
        &format!("fireworks_ai/{DEFAULT_MODEL}"),
    ))
    .await
    else {
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
            "model": "llama",
            "messages": [{"role": "user", "content": long_prompt}],
            "max_tokens": 100,
        }))
        .send()
        .await
        .expect("request failed");

    assert_eq!(
        resp.status(),
        200,
        "Fireworks long context failed: {}",
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
async fn fireworks_multiple_models() {
    let Some(api_key) = get_api_key() else {
        eprintln!("skipping: {FIREWORKS_API_KEY_ENV} not set");
        return;
    };

    let kimi_model = format!("fireworks_ai/{DEFAULT_MODEL}");
    let glm_model = format!("fireworks_ai/{FAST_MODEL}");
    let models: Vec<(&str, &str)> = vec![("kimi", &kimi_model), ("glm-flash", &glm_model)];

    let config = ProxyConfig {
        model_list: models
            .iter()
            .map(|(name, model_key)| ConfigModel {
                model_name: name.to_string(),
                oxllm_params: OxllmParams {
                    model: model_key.to_string(),
                    api_key: Some(api_key.clone()),
                    api_base: None,
                    timeout: Some(30.0),
                    stream_timeout: Some(30),
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
                // Room for the FAST_MODEL's reasoning preamble: a 10-token
                // budget is consumed by reasoning_content before any content.
                "max_tokens": 400,
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
async fn fireworks_streaming_chat() {
    let Some(_key) = get_api_key() else {
        eprintln!("skipping: {FIREWORKS_API_KEY_ENV} not set");
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for(
        "llama",
        "fireworks_ai/accounts/fireworks/models/nemotron-3-ultra-nvfp4",
    ))
    .await
    else {
        eprintln!("skipping: failed to start oxllm");
        return;
    };

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth("sk-test-key-master-1")
        .json(&json!({
            "model": "llama",
            "messages": [{"role": "user", "content": "Count to 3"}],
            "stream": true,
            "max_tokens": 50,
        }))
        .send()
        .await
        .expect("request failed");

    assert_eq!(resp.status(), 200, "Fireworks streaming request failed");
    let body = resp.text().await.expect("body");
    assert!(body.contains("data: "), "expected SSE data, got: {body}");
    assert!(body.contains("[DONE]"), "expected [DONE] terminator");
    eprintln!("Fireworks streaming OK — received {} bytes", body.len());

    handle.shutdown();
}

#[tokio::test]
async fn fireworks_streaming_with_include_usage() {
    let Some(_key) = get_api_key() else {
        eprintln!("skipping: {FIREWORKS_API_KEY_ENV} not set");
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for(
        "llama",
        "fireworks_ai/accounts/fireworks/models/nemotron-3-ultra-nvfp4",
    ))
    .await
    else {
        eprintln!("skipping: failed to start oxllm");
        return;
    };

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth("sk-test-key-master-1")
        .json(&json!({
            "model": "llama",
            "messages": [{"role": "user", "content": "Count to 5"}],
            "stream": true,
            "stream_options": {"include_usage": true},
            "max_tokens": 50,
        }))
        .send()
        .await
        .expect("request failed");

    assert_eq!(resp.status(), 200, "Fireworks streaming with usage failed");
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
async fn fireworks_with_seed_and_temperature() {
    let Some(_key) = get_api_key() else {
        eprintln!("skipping: {FIREWORKS_API_KEY_ENV} not set");
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for(
        "llama",
        &format!("fireworks_ai/{DEFAULT_MODEL}"),
    ))
    .await
    else {
        eprintln!("skipping: failed to start oxllm");
        return;
    };

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth("sk-test-key-master-1")
        .json(&json!({
            "model": "llama",
            "messages": [{"role": "user", "content": "Say hello in one word"}],
            "seed": 42,
            "temperature": 0.5,
            "max_tokens": 10,
        }))
        .send()
        .await
        .expect("request failed");

    assert_eq!(
        resp.status(),
        200,
        "Fireworks seed+temperature failed: {}",
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
async fn fireworks_embedding_request() {
    let Some(api_key) = get_api_key() else {
        eprintln!("skipping: {FIREWORKS_API_KEY_ENV} not set");
        return;
    };

    let config = ProxyConfig {
        model_list: vec![ConfigModel {
            model_name: "embed".to_string(),
            oxllm_params: OxllmParams {
                model: format!("fireworks_ai/{EMBEDDING_MODEL}"),
                api_key: Some(api_key),
                api_base: None,
                timeout: Some(30.0),
                stream_timeout: Some(30),
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
    };
    let Some(handle) = start_oxllm(config).await else {
        eprintln!("skipping: failed to start oxllm");
        return;
    };

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/embeddings"))
        .bearer_auth("sk-test-key-master-1")
        .json(&json!({
            "model": "embed",
            "input": "hello world",
        }))
        .send()
        .await
        .expect("request failed");

    assert_eq!(
        resp.status(),
        200,
        "Fireworks embedding failed: {}",
        resp.text().await.unwrap_or_default()
    );
    let body: Value = resp.json().await.expect("json");
    let embedding = body["data"][0]["embedding"]
        .as_array()
        .expect("expected embedding array");
    assert!(!embedding.is_empty(), "embedding should not be empty");
    eprintln!("Embedding dimension: {}", embedding.len());

    handle.shutdown();
}

#[tokio::test]
async fn fireworks_invalid_auth_returns_401() {
    let Some(_key) = get_api_key() else {
        eprintln!("skipping: {FIREWORKS_API_KEY_ENV} not set");
        return;
    };

    let Some(handle) = start_oxllm(proxy_config(&_key)).await else {
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
async fn fireworks_nonexistent_model() {
    let Some(_key) = get_api_key() else {
        eprintln!("skipping: {FIREWORKS_API_KEY_ENV} not set");
        return;
    };

    let Some(handle) = start_oxllm(proxy_config(&_key)).await else {
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
async fn fireworks_health_endpoint() {
    let Some(_key) = get_api_key() else {
        eprintln!("skipping: {FIREWORKS_API_KEY_ENV} not set");
        return;
    };

    let Some(handle) = start_oxllm(proxy_config(&_key)).await else {
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
async fn fireworks_models_endpoint() {
    let Some(_key) = get_api_key() else {
        eprintln!("skipping: {FIREWORKS_API_KEY_ENV} not set");
        return;
    };

    let Some(handle) = start_oxllm(proxy_config(&_key)).await else {
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
        models.contains(&"llama"),
        "expected 'llama' in model list, got: {models:?}"
    );

    handle.shutdown();
}
