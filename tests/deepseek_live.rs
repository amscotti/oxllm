//! Live integration test against the DeepSeek API.
//!
//! Requires `DEEPSEEK_API_KEY` to be set in the environment.
//! Run with: `DEEPSEEK_API_KEY=sk-... cargo test --test deepseek_live`

use oxllm_server::OxllmParams;
use oxllm_server::app::{build_router, build_shared_state};
use oxllm_server::config::types::{ConfigModel, GeneralSettings, OxllmSettings, ProxyConfig};
use serde_json::{Value, json};

const DEEPSEEK_API_KEY_ENV: &str = "DEEPSEEK_API_KEY";
const DEFAULT_MODEL: &str = "deepseek/deepseek-v4-flash";
const REASONER_MODEL: &str = "deepseek/deepseek-v4-pro";

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
    std::env::var(DEEPSEEK_API_KEY_ENV)
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

fn proxy_config_for(model_name: &str, model_key: &str) -> ProxyConfig {
    let key = get_api_key().expect("DEEPSEEK_API_KEY must be set");
    ProxyConfig {
        model_list: vec![ConfigModel {
            model_name: model_name.to_string(),
            oxllm_params: OxllmParams {
                model: model_key.to_string(),
                api_key: Some(key),
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
async fn deepseek_non_streaming_chat() {
    let Some(_key) = get_api_key() else {
        eprintln!("skipping: {DEEPSEEK_API_KEY_ENV} not set");
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for("deepseek-v4-flash", DEFAULT_MODEL)).await
    else {
        eprintln!("skipping: failed to start oxllm");
        return;
    };

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth("sk-test-key-master-1")
        .json(&json!({
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "Say hello in one word"}],
        }))
        .send()
        .await
        .expect("request failed");

    assert_eq!(
        resp.status(),
        200,
        "DeepSeek request failed: {}",
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
    eprintln!("DeepSeek response: {content}");
    eprintln!("usage: {}", body["usage"]);

    handle.shutdown();
}

#[tokio::test]
async fn deepseek_streaming_chat() {
    let Some(_key) = get_api_key() else {
        eprintln!("skipping: {DEEPSEEK_API_KEY_ENV} not set");
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for("deepseek-v4-flash", DEFAULT_MODEL)).await
    else {
        eprintln!("skipping: failed to start oxllm");
        return;
    };

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth("sk-test-key-master-1")
        .json(&json!({
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "Count to 3"}],
            "stream": true,
        }))
        .send()
        .await
        .expect("request failed");

    assert_eq!(resp.status(), 200, "DeepSeek streaming request failed");
    let body = resp.text().await.expect("body");
    assert!(body.contains("data: "), "expected SSE data, got: {body}");
    assert!(body.contains("[DONE]"), "expected [DONE] terminator");
    eprintln!("DeepSeek streaming OK — received {} bytes", body.len());

    handle.shutdown();
}

#[tokio::test]
async fn deepseek_tool_call() {
    let Some(_key) = get_api_key() else {
        eprintln!("skipping: {DEEPSEEK_API_KEY_ENV} not set");
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for("deepseek-v4-flash", DEFAULT_MODEL)).await
    else {
        eprintln!("skipping: failed to start oxllm");
        return;
    };

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth("sk-test-key-master-1")
        .json(&json!({
            "model": "deepseek-v4-flash",
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
            "tool_choice": "auto"
        }))
        .send()
        .await
        .expect("request failed");

    assert_eq!(
        resp.status(),
        200,
        "DeepSeek tool call failed: {}",
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
async fn deepseek_json_object_mode() {
    let Some(_key) = get_api_key() else {
        eprintln!("skipping: {DEEPSEEK_API_KEY_ENV} not set");
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for("deepseek-v4-flash", DEFAULT_MODEL)).await
    else {
        eprintln!("skipping: failed to start oxllm");
        return;
    };

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth("sk-test-key-master-1")
        .json(&json!({
            "model": "deepseek-v4-flash",
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
        "DeepSeek JSON mode failed: {}",
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
async fn deepseek_multiple_models() {
    let Some(api_key) = get_api_key() else {
        eprintln!("skipping: {DEEPSEEK_API_KEY_ENV} not set");
        return;
    };

    let models: Vec<(&str, &str)> = vec![
        ("deepseek-v4-flash", DEFAULT_MODEL),
        ("deepseek-v4-pro", REASONER_MODEL),
    ];

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
async fn deepseek_streaming_with_include_usage() {
    let Some(_key) = get_api_key() else {
        eprintln!("skipping: {DEEPSEEK_API_KEY_ENV} not set");
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for("deepseek-v4-flash", DEFAULT_MODEL)).await
    else {
        eprintln!("skipping: failed to start oxllm");
        return;
    };

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth("sk-test-key-master-1")
        .json(&json!({
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "Count to 5"}],
            "stream": true,
            "stream_options": {"include_usage": true},
        }))
        .send()
        .await
        .expect("request failed");

    assert_eq!(resp.status(), 200, "DeepSeek streaming with usage failed");
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
async fn deepseek_with_seed() {
    let Some(_key) = get_api_key() else {
        eprintln!("skipping: {DEEPSEEK_API_KEY_ENV} not set");
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for("deepseek-v4-flash", DEFAULT_MODEL)).await
    else {
        eprintln!("skipping: failed to start oxllm");
        return;
    };

    let client = reqwest::Client::new();
    for i in 0..2 {
        let resp = client
            .post(handle.url("/v1/chat/completions"))
            .bearer_auth("sk-test-key-master-1")
            .json(&json!({
                "model": "deepseek-v4-flash",
                "messages": [{"role": "user", "content": "Say hello in one word"}],
                "seed": 42,
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
async fn deepseek_with_stop_sequence() {
    let Some(_key) = get_api_key() else {
        eprintln!("skipping: {DEEPSEEK_API_KEY_ENV} not set");
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for("deepseek-v4-flash", DEFAULT_MODEL)).await
    else {
        eprintln!("skipping: failed to start oxllm");
        return;
    };

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth("sk-test-key-master-1")
        .json(&json!({
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "Say 'Hello world' exactly"}],
            "stop": ["world"],
        }))
        .send()
        .await
        .expect("request failed");

    assert_eq!(
        resp.status(),
        200,
        "DeepSeek stop sequence failed: {}",
        resp.text().await.unwrap_or_default()
    );
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
    let finish_reason = body["choices"][0]["finish_reason"].as_str().unwrap_or("");
    assert_eq!(finish_reason, "stop", "expected finish_reason 'stop'");

    handle.shutdown();
}

#[tokio::test]
async fn deepseek_max_completion_tokens() {
    let Some(_key) = get_api_key() else {
        eprintln!("skipping: {DEEPSEEK_API_KEY_ENV} not set");
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for("deepseek-v4-flash", DEFAULT_MODEL)).await
    else {
        eprintln!("skipping: failed to start oxllm");
        return;
    };

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth("sk-test-key-master-1")
        .json(&json!({
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "Write a very long essay about AI"}],
            "max_completion_tokens": 5,
            "max_tokens": 5,
            // v4-flash reasons by default and would consume the whole 5-token
            // budget on reasoning (leaving empty content); the test validates
            // the max_completion_tokens passthrough, so disable reasoning.
            "reasoning_effort": "none",
        }))
        .send()
        .await
        .expect("request failed");

    assert_eq!(
        resp.status(),
        200,
        "DeepSeek max tokens failed: {}",
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
async fn deepseek_long_context() {
    let Some(_key) = get_api_key() else {
        eprintln!("skipping: {DEEPSEEK_API_KEY_ENV} not set");
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for("deepseek-v4-flash", DEFAULT_MODEL)).await
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
        paragraph.repeat(20)
    );
    eprintln!(
        "Long prompt length: ~{} words (~{} tokens)",
        long_prompt.split_whitespace().count(),
        long_prompt.split_whitespace().count() * 4 / 3
    );

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth("sk-test-key-master-1")
        .json(&json!({
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": long_prompt}],
            "max_completion_tokens": 200,
        }))
        .send()
        .await
        .expect("request failed");

    assert_eq!(
        resp.status(),
        200,
        "DeepSeek long context failed: {}",
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
async fn deepseek_invalid_auth_returns_401() {
    let Some(_key) = get_api_key() else {
        eprintln!("skipping: {DEEPSEEK_API_KEY_ENV} not set");
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for("deepseek-v4-flash", DEFAULT_MODEL)).await
    else {
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
async fn deepseek_nonexistent_model() {
    let Some(_key) = get_api_key() else {
        eprintln!("skipping: {DEEPSEEK_API_KEY_ENV} not set");
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for("deepseek-v4-flash", DEFAULT_MODEL)).await
    else {
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
async fn deepseek_health_endpoint() {
    let Some(_key) = get_api_key() else {
        eprintln!("skipping: {DEEPSEEK_API_KEY_ENV} not set");
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for("deepseek-v4-flash", DEFAULT_MODEL)).await
    else {
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
async fn deepseek_models_endpoint() {
    let Some(_key) = get_api_key() else {
        eprintln!("skipping: {DEEPSEEK_API_KEY_ENV} not set");
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for("deepseek-v4-flash", DEFAULT_MODEL)).await
    else {
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
        models.contains(&"deepseek-v4-flash"),
        "expected 'deepseek-v4-flash' in model list, got: {models:?}"
    );

    handle.shutdown();
}

#[tokio::test]
async fn deepseek_with_seed_and_temperature() {
    let Some(_key) = get_api_key() else {
        eprintln!("skipping: {DEEPSEEK_API_KEY_ENV} not set");
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for("deepseek-v4-flash", DEFAULT_MODEL)).await
    else {
        eprintln!("skipping: failed to start oxllm");
        return;
    };

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth("sk-test-key-master-1")
        .json(&json!({
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "Say hello in one word"}],
            "seed": 42,
            "temperature": 0.5,
        }))
        .send()
        .await
        .expect("request failed");

    assert_eq!(
        resp.status(),
        200,
        "DeepSeek seed+temperature failed: {}",
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
async fn deepseek_reasoner_model() {
    let Some(_key) = get_api_key() else {
        eprintln!("skipping: {DEEPSEEK_API_KEY_ENV} not set");
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for("deepseek-v4-pro", REASONER_MODEL)).await
    else {
        eprintln!("skipping: failed to start oxllm");
        return;
    };

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth("sk-test-key-master-1")
        .json(&json!({
            "model": "deepseek-v4-pro",
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "max_completion_tokens": 200,
        }))
        .send()
        .await
        .expect("request failed");

    assert_eq!(
        resp.status(),
        200,
        "DeepSeek reasoner failed: {}",
        resp.text().await.unwrap_or_default()
    );
    let body: Value = resp.json().await.expect("json");
    let content = body["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("");
    assert!(!content.is_empty(), "response had empty content");
    let reasoning = body["choices"][0]["message"]["reasoning_content"]
        .as_str()
        .unwrap_or("");
    if reasoning.is_empty() {
        eprintln!("reasoning_content NOT present in response");
    } else {
        eprintln!(
            "reasoning_content present ({} chars): {reasoning}",
            reasoning.len()
        );
    }
    eprintln!("DeepSeek reasoner response: {content}");

    handle.shutdown();
}

#[tokio::test]
async fn deepseek_reasoner_streaming() {
    let Some(_key) = get_api_key() else {
        eprintln!("skipping: {DEEPSEEK_API_KEY_ENV} not set");
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for("deepseek-v4-pro", REASONER_MODEL)).await
    else {
        eprintln!("skipping: failed to start oxllm");
        return;
    };

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth("sk-test-key-master-1")
        .json(&json!({
            "model": "deepseek-v4-pro",
            "messages": [{"role": "user", "content": "What is the capital of France? Answer in one word."}],
            "stream": true,
        }))
        .send()
        .await
        .expect("request failed");

    assert_eq!(
        resp.status(),
        200,
        "DeepSeek reasoner streaming failed: {}",
        resp.text().await.unwrap_or_default()
    );
    let body = resp.text().await.expect("body");
    assert!(body.contains("data: "), "expected SSE data, got: {body}");
    assert!(body.contains("[DONE]"), "expected [DONE] terminator");

    let mut reasoning_text = String::new();
    let mut content_text = String::new();
    for line in body.lines().filter_map(|l| l.strip_prefix("data: ")) {
        if line == "[DONE]" {
            continue;
        }
        let Ok(chunk) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        let delta = &chunk["choices"][0]["delta"];
        if let Some(part) = delta["reasoning_content"].as_str() {
            reasoning_text.push_str(part);
        }
        if let Some(part) = delta["content"].as_str() {
            content_text.push_str(part);
        }
    }
    assert!(
        !reasoning_text.is_empty() || !content_text.is_empty(),
        "no text collected from reasoner stream"
    );
    eprintln!(
        "Reasoner stream: reasoning={} chars, content={} chars",
        reasoning_text.len(),
        content_text.len()
    );
    eprintln!("Reasoning text: {reasoning_text}");
    eprintln!("Content text: {content_text}");

    handle.shutdown();
}
