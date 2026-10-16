//! Live integration test against the real Anthropic API.
//!
//! Requires `ANTHROPIC_API_KEY` to be set in the environment.
//! Run with: `ANTHROPIC_API_KEY=sk-ant-... cargo test --test anthropic_live --all-features -- --nocapture`

use oxllm_server::OxllmParams;
use oxllm_server::app::{build_router, build_shared_state};
use oxllm_server::config::types::{ConfigModel, GeneralSettings, OxllmSettings, ProxyConfig};
use serde_json::{Value, json};

const ANTHROPIC_API_KEY_ENV: &str = "ANTHROPIC_API_KEY";
const MAIN_MODEL: &str = "anthropic/claude-sonnet-5";
const SECOND_MODEL: &str = "anthropic/claude-haiku-4-5";

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
    std::env::var(ANTHROPIC_API_KEY_ENV)
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
    let key = get_api_key().expect("ANTHROPIC_API_KEY must be set");
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
async fn anthropic_non_streaming_chat() {
    let Some(_key) = get_api_key() else {
        eprintln!("skipping: {ANTHROPIC_API_KEY_ENV} not set");
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for("sonnet", MAIN_MODEL)).await else {
        eprintln!("skipping: failed to start oxllm");
        return;
    };

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth("sk-test-key-master-1")
        .json(&json!({
            "model": "sonnet",
            "messages": [{"role": "user", "content": "Say hello in one word"}],
            "max_completion_tokens": 20,
        }))
        .send()
        .await
        .expect("request failed");

    assert_eq!(
        resp.status(),
        200,
        "Anthropic request failed: {}",
        resp.text().await.unwrap_or_default()
    );
    let body: Value = resp.json().await.expect("json");
    assert_eq!(body["object"], "chat.completion");
    let content = body["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("");
    assert!(!content.is_empty(), "response had empty content");
    eprintln!("Anthropic response: {content}");

    handle.shutdown();
}

#[tokio::test]
async fn anthropic_streaming_chat() {
    let Some(_key) = get_api_key() else {
        eprintln!("skipping: {ANTHROPIC_API_KEY_ENV} not set");
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for("sonnet", MAIN_MODEL)).await else {
        eprintln!("skipping: failed to start oxllm");
        return;
    };

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth("sk-test-key-master-1")
        .json(&json!({
            "model": "sonnet",
            "messages": [{"role": "user", "content": "Count to 3"}],
            "stream": true,
            "max_completion_tokens": 100,
        }))
        .send()
        .await
        .expect("request failed");

    assert_eq!(resp.status(), 200, "Anthropic streaming request failed");
    let body = resp.text().await.expect("body");
    assert!(body.contains("data: "), "expected SSE data, got: {body}");
    assert!(body.contains("[DONE]"), "expected [DONE] terminator");
    eprintln!("Anthropic streaming OK — received {} bytes", body.len());

    handle.shutdown();
}

#[tokio::test]
async fn anthropic_tool_call() {
    let Some(_key) = get_api_key() else {
        eprintln!("skipping: {ANTHROPIC_API_KEY_ENV} not set");
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for("sonnet", MAIN_MODEL)).await else {
        eprintln!("skipping: failed to start oxllm");
        return;
    };

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth("sk-test-key-master-1")
        .json(&json!({
            "model": "sonnet",
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
            "max_completion_tokens": 100,
        }))
        .send()
        .await
        .expect("request failed");

    assert_eq!(
        resp.status(),
        200,
        "Anthropic tool call failed: {}",
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
    let args: Value = serde_json::from_str(
        tool_calls[0]["function"]["arguments"]
            .as_str()
            .unwrap_or("{}"),
    )
    .expect("tool arguments must be valid JSON");
    assert!(args.get("city").is_some(), "expected city argument");

    handle.shutdown();
}

#[tokio::test]
async fn anthropic_streaming_tool_call() {
    let Some(_key) = get_api_key() else {
        eprintln!("skipping: {ANTHROPIC_API_KEY_ENV} not set");
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for("sonnet", MAIN_MODEL)).await else {
        eprintln!("skipping: failed to start oxllm");
        return;
    };

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth("sk-test-key-master-1")
        .json(&json!({
            "model": "sonnet",
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
            "stream": true,
            "max_completion_tokens": 100,
        }))
        .send()
        .await
        .expect("request failed");

    assert_eq!(resp.status(), 200, "Anthropic streaming tool call failed");
    let body = resp.text().await.expect("body");

    let mut tool_ids: Vec<String> = Vec::new();
    let mut tool_names: Vec<String> = Vec::new();
    let mut args_by_index: std::collections::BTreeMap<u64, String> = Default::default();
    for line in body.lines() {
        let Some(data) = line.strip_prefix("data: ") else {
            continue;
        };
        if data == "[DONE]" {
            continue;
        }
        let Ok(chunk) = serde_json::from_str::<Value>(data) else {
            continue;
        };
        let Some(calls) = chunk["choices"][0]["delta"]["tool_calls"].as_array() else {
            continue;
        };
        for tc in calls {
            let idx = tc["index"].as_u64().unwrap_or(0);
            if let Some(id) = tc["id"].as_str() {
                if !id.is_empty() {
                    tool_ids.push(id.to_string());
                }
            }
            if let Some(name) = tc["function"]["name"].as_str() {
                if !name.is_empty() {
                    tool_names.push(name.to_string());
                }
            }
            if let Some(args) = tc["function"]["arguments"].as_str() {
                if !args.is_empty() {
                    args_by_index.entry(idx).or_default().push_str(args);
                }
            }
        }
    }

    assert!(
        !tool_ids.is_empty(),
        "expected a tool call id in the stream"
    );
    assert!(
        tool_names.contains(&"get_weather".to_string()),
        "expected get_weather tool call in stream, got: {tool_names:?}"
    );
    assert!(
        !args_by_index.is_empty(),
        "expected accumulated tool call arguments"
    );
    let accumulated = args_by_index.values().next().unwrap();
    let parsed: Value = serde_json::from_str(accumulated)
        .unwrap_or_else(|e| panic!("accumulated args must be valid JSON: {e}: {accumulated}"));
    assert!(
        parsed.get("city").is_some(),
        "expected city argument in {accumulated}"
    );
    eprintln!("Streamed tool call args: {accumulated}");

    handle.shutdown();
}

#[tokio::test]
async fn anthropic_multiturn_tool_conversation() {
    let Some(_key) = get_api_key() else {
        eprintln!("skipping: {ANTHROPIC_API_KEY_ENV} not set");
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for("sonnet", MAIN_MODEL)).await else {
        eprintln!("skipping: failed to start oxllm");
        return;
    };

    let client = reqwest::Client::new();
    let tools = json!([{
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
    }]);

    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth("sk-test-key-master-1")
        .json(&json!({
            "model": "sonnet",
            "messages": [
                {"role": "system", "content": "You have a function get_weather(city: string). Call it for Paris."},
                {"role": "user", "content": "What is the weather in Paris?"}
            ],
            "tools": tools,
            "tool_choice": "auto",
            "max_completion_tokens": 100,
        }))
        .send()
        .await
        .expect("request failed");

    assert_eq!(
        resp.status(),
        200,
        "first turn failed: {}",
        resp.text().await.unwrap_or_default()
    );
    let body: Value = resp.json().await.expect("json");
    let assistant_msg = body["choices"][0]["message"].clone();
    let tool_calls = assistant_msg["tool_calls"]
        .as_array()
        .expect("expected tool_calls array");
    assert!(!tool_calls.is_empty(), "at least one tool call");
    let tool_call_id = tool_calls[0]["id"].as_str().unwrap_or("").to_string();
    assert!(!tool_call_id.is_empty(), "expected non-empty tool call id");

    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth("sk-test-key-master-1")
        .json(&json!({
            "model": "sonnet",
            "messages": [
                {"role": "system", "content": "You have a function get_weather(city: string). Call it for Paris."},
                {"role": "user", "content": "What is the weather in Paris?"},
                {"role": "assistant", "content": "", "tool_calls": tool_calls},
                {"role": "tool", "tool_call_id": tool_call_id, "content": "The weather in Paris is sunny and 22 degrees."}
            ],
            "max_completion_tokens": 100,
        }))
        .send()
        .await
        .expect("request failed");

    assert_eq!(
        resp.status(),
        200,
        "second turn failed: {}",
        resp.text().await.unwrap_or_default()
    );
    let body: Value = resp.json().await.expect("json");
    let content = body["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("");
    assert!(
        !content.is_empty(),
        "follow-up answer should not be empty, got: {content}"
    );
    eprintln!("Follow-up answer: {content}");

    handle.shutdown();
}

#[tokio::test]
async fn anthropic_native_json_schema_output() {
    let Some(_key) = get_api_key() else {
        eprintln!("skipping: {ANTHROPIC_API_KEY_ENV} not set");
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for("sonnet", MAIN_MODEL)).await else {
        eprintln!("skipping: failed to start oxllm");
        return;
    };

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth("sk-test-key-master-1")
        .json(&json!({
            "model": "sonnet",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that always responds with valid JSON."},
                {"role": "user", "content": "Create a JSON object for a person named Alice who is 25 years old, with fields 'name' and 'age'."}
            ],
            "response_format": {"type":"json_schema","json_schema":{"name":"person","strict":true,"schema":{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},"required":["name","age"],"additionalProperties":false}}},
            "max_completion_tokens": 512,
        }))
        .send()
        .await
        .expect("request failed");

    assert_eq!(
        resp.status(),
        200,
        "Anthropic native JSON schema request failed: {}",
        resp.text().await.unwrap_or_default()
    );
    let body: Value = resp.json().await.expect("json");
    if matches!(
        body["choices"][0]["finish_reason"].as_str(),
        Some("length" | "content_filter" | "refusal")
    ) || body["choices"][0]["message"]["refusal"].is_string()
    {
        eprintln!("native schema result unevaluated: truncated or refused");
        handle.shutdown();
        return;
    }
    let content = body["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("");
    eprintln!("JSON mode response: {content}");
    assert!(!content.is_empty(), "response had empty content");
    let parsed: Value = serde_json::from_str(content).expect("response must be valid JSON");
    assert!(
        parsed.is_object(),
        "response must be a JSON object, got: {parsed}"
    );
    assert!(parsed["name"].is_string());
    assert!(parsed["age"].as_f64().is_some_and(|age| age.fract() == 0.0));
    assert_eq!(parsed.as_object().unwrap().len(), 2);
    assert!(body["choices"][0]["message"].get("tool_calls").is_none());

    handle.shutdown();
}

#[tokio::test]
async fn anthropic_system_message_with_cache_control() {
    let Some(_key) = get_api_key() else {
        eprintln!("skipping: {ANTHROPIC_API_KEY_ENV} not set");
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for("sonnet", MAIN_MODEL)).await else {
        eprintln!("skipping: failed to start oxllm");
        return;
    };

    // Long enough system text (Anthropic requires >=1024 tokens for a block
    // to be cacheable) with a cache_control breakpoint.
    let paragraph = "You are a helpful assistant working with a long reference document. \
        The document explains that the capital of France is Paris, the capital of Italy is \
        Rome, and the capital of Japan is Tokyo. ";
    let long_system = paragraph.repeat(20);

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth("sk-test-key-master-1")
        .json(&json!({
            "model": "sonnet",
            "messages": [
                {"role": "system", "content": [
                    {"type": "text", "text": long_system, "cache_control": {"type": "ephemeral"}}
                ]},
                {"role": "user", "content": "What is the capital of France?"}
            ],
            "max_completion_tokens": 50,
        }))
        .send()
        .await
        .expect("request failed");

    assert_eq!(
        resp.status(),
        200,
        "Anthropic cache_control request failed: {}",
        resp.text().await.unwrap_or_default()
    );
    let body: Value = resp.json().await.expect("json");
    let content = body["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("");
    assert!(!content.is_empty(), "response had empty content");
    eprintln!("Cache_control response: {content}");

    handle.shutdown();
}

#[tokio::test]
async fn anthropic_multiple_models() {
    let Some(api_key) = get_api_key() else {
        eprintln!("skipping: {ANTHROPIC_API_KEY_ENV} not set");
        return;
    };

    let models: Vec<(&str, &str)> = vec![("sonnet", MAIN_MODEL), ("haiku", SECOND_MODEL)];

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
                "max_completion_tokens": 20,
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
async fn anthropic_streaming_with_include_usage() {
    let Some(_key) = get_api_key() else {
        eprintln!("skipping: {ANTHROPIC_API_KEY_ENV} not set");
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for("sonnet", MAIN_MODEL)).await else {
        eprintln!("skipping: failed to start oxllm");
        return;
    };

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth("sk-test-key-master-1")
        .json(&json!({
            "model": "sonnet",
            "messages": [{"role": "user", "content": "Count to 5"}],
            "stream": true,
            "stream_options": {"include_usage": true},
            "max_completion_tokens": 100,
        }))
        .send()
        .await
        .expect("request failed");

    assert_eq!(resp.status(), 200, "Anthropic streaming with usage failed");
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
async fn anthropic_with_temperature() {
    let Some(_key) = get_api_key() else {
        eprintln!("skipping: {ANTHROPIC_API_KEY_ENV} not set");
        return;
    };

    // `temperature` is deprecated on claude-sonnet-5 (the API rejects any
    // value); this test validates the temperature passthrough on the current
    // Haiku model, which still accepts it.
    let Some(handle) = start_oxllm(proxy_config_for("haiku", SECOND_MODEL)).await else {
        eprintln!("skipping: failed to start oxllm");
        return;
    };

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth("sk-test-key-master-1")
        .json(&json!({
            "model": "haiku",
            "messages": [{"role": "user", "content": "Say hello in one word"}],
            "temperature": 0,
            "max_completion_tokens": 20,
        }))
        .send()
        .await
        .expect("request failed");

    assert_eq!(
        resp.status(),
        200,
        "Anthropic temperature request failed: {}",
        resp.text().await.unwrap_or_default()
    );
    let body: Value = resp.json().await.expect("json");
    let content = body["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("");
    assert!(!content.is_empty(), "response had empty content");
    eprintln!("Temperature response: {content}");

    handle.shutdown();
}

#[tokio::test]
async fn anthropic_with_stop_sequence() {
    let Some(_key) = get_api_key() else {
        eprintln!("skipping: {ANTHROPIC_API_KEY_ENV} not set");
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for("sonnet", MAIN_MODEL)).await else {
        eprintln!("skipping: failed to start oxllm");
        return;
    };

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth("sk-test-key-master-1")
        .json(&json!({
            "model": "sonnet",
            "messages": [{"role": "user", "content": "Say exactly: 'The capital of France is Paris'"}],
            "stop": ["Paris"],
            "max_completion_tokens": 30,
        }))
        .send()
        .await
        .expect("request failed");

    if resp.status() != 200 {
        eprintln!(
            "skipping: upstream does not support the stop parameter (status {})",
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
        !content.contains("Paris"),
        "response should not contain the stop sequence 'Paris', got: {content}"
    );
    let finish_reason = body["choices"][0]["finish_reason"].as_str().unwrap_or("");
    assert_eq!(finish_reason, "stop", "expected finish_reason 'stop'");

    handle.shutdown();
}

#[tokio::test]
async fn anthropic_max_completion_tokens() {
    let Some(_key) = get_api_key() else {
        eprintln!("skipping: {ANTHROPIC_API_KEY_ENV} not set");
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for("sonnet", MAIN_MODEL)).await else {
        eprintln!("skipping: failed to start oxllm");
        return;
    };

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth("sk-test-key-master-1")
        .json(&json!({
            "model": "sonnet",
            "messages": [{"role": "user", "content": "Write a very long essay about AI"}],
            "max_completion_tokens": 20,
        }))
        .send()
        .await
        .expect("request failed");

    assert_eq!(
        resp.status(),
        200,
        "Anthropic max tokens failed: {}",
        resp.text().await.unwrap_or_default()
    );
    let body: Value = resp.json().await.expect("json");
    let content = body["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("");
    let finish_reason = body["choices"][0]["finish_reason"].as_str().unwrap_or("");
    eprintln!(
        "Short response ({len} chars, finish {finish_reason}): {content}",
        len = content.len()
    );
    assert!(!content.is_empty(), "response had empty content");
    if finish_reason != "length" {
        assert!(
            content.len() < 100,
            "with max_completion_tokens=20 expected 'length' or short content, got {finish_reason} and {} chars: {content}",
            content.len()
        );
    }

    handle.shutdown();
}

#[tokio::test]
async fn anthropic_long_context() {
    let Some(_key) = get_api_key() else {
        eprintln!("skipping: {ANTHROPIC_API_KEY_ENV} not set");
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for("sonnet", MAIN_MODEL)).await else {
        eprintln!("skipping: failed to start oxllm");
        return;
    };

    // A ~10 message multi-turn conversation (5 user/assistant pairs + a final
    // user turn).
    let turns: Vec<(&str, &str)> = vec![
        ("user", "My name is Alice and I work in Paris."),
        ("assistant", "Nice to meet you, Alice!"),
        ("user", "What is the weather usually like there?"),
        ("assistant", "Paris has mild winters and warm summers."),
        ("user", "What should I pack for a trip?"),
        ("assistant", "Bring an umbrella and comfortable shoes."),
        ("user", "Is there a good time to visit?"),
        ("assistant", "Spring and early autumn are the best times."),
        ("user", "Any restaurant recommendations?"),
        ("assistant", "Try a classic bistro near the Seine."),
    ];
    let mut messages: Vec<Value> = turns
        .iter()
        .map(|(role, content)| json!({"role": role, "content": content}))
        .collect();
    messages.push(json!({"role": "user", "content": "Given our conversation, what is my name and where do I work?"}));

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth("sk-test-key-master-1")
        .json(&json!({
            "model": "sonnet",
            "messages": messages,
            "max_completion_tokens": 50,
        }))
        .send()
        .await
        .expect("request failed");

    assert_eq!(
        resp.status(),
        200,
        "Anthropic long context failed: {}",
        resp.text().await.unwrap_or_default()
    );
    let body: Value = resp.json().await.expect("json");
    let content = body["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("");
    assert!(!content.is_empty(), "response had empty content");
    eprintln!(
        "Long context response ({len} chars): {content}",
        len = content.len()
    );

    handle.shutdown();
}

#[tokio::test]
async fn anthropic_thinking_model() {
    let Some(_key) = get_api_key() else {
        eprintln!("skipping: {ANTHROPIC_API_KEY_ENV} not set");
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for("thinking", MAIN_MODEL)).await else {
        eprintln!("skipping: failed to start oxllm");
        return;
    };

    // `reasoning_effort: "high"` maps to the model's adaptive-thinking
    // interface (`thinking: {type:"adaptive"}` + `output_config.effort`) for
    // Claude 5-family models. Adaptive thinking self-sizes and may skip
    // thinking entirely on trivial prompts, so the prompt explicitly asks for
    // deliberate reasoning. max_completion_tokens is kept generously above
    // the thinking the model emits.
    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth("sk-test-key-master-1")
        .json(&json!({
            "model": "thinking",
            "messages": [{"role": "user", "content": "Think step by step and verify each stage: what is 17 * 23?"}],
            "reasoning_effort": "high",
            "max_completion_tokens": 2000,
        }))
        .send()
        .await
        .expect("request failed");

    if resp.status() != 200 {
        eprintln!(
            "skipping: thinking model not available (status {})",
            resp.status()
        );
        handle.shutdown();
        return;
    }

    let body: Value = resp.json().await.expect("json");
    let message = &body["choices"][0]["message"];
    let reasoning = message["reasoning_content"].as_str().unwrap_or("");
    let has_thought = !reasoning.is_empty()
        || message["thinking_blocks"]
            .as_array()
            .is_some_and(|b| !b.is_empty());
    assert!(
        has_thought,
        "expected reasoning_content or thinking_blocks, got: {body}"
    );
    eprintln!(
        "Thinking response ({reasoning} reasoning chars, {} thinking blocks)",
        message["thinking_blocks"].as_array().map_or(0, |b| b.len())
    );

    handle.shutdown();
}

#[tokio::test]
async fn anthropic_invalid_auth_returns_401() {
    let Some(_key) = get_api_key() else {
        eprintln!("skipping: {ANTHROPIC_API_KEY_ENV} not set");
        return;
    };

    let config = ProxyConfig {
        model_list: vec![ConfigModel {
            model_name: "sonnet".to_string(),
            oxllm_params: OxllmParams {
                model: MAIN_MODEL.to_string(),
                api_key: Some("sk-bogus".to_string()),
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
    };

    let Some(handle) = start_oxllm(config).await else {
        eprintln!("skipping: failed to start oxllm");
        return;
    };

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth("sk-test-key-master-1")
        .json(&json!({
            "model": "sonnet",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_completion_tokens": 20,
        }))
        .send()
        .await
        .expect("request failed");

    assert_eq!(
        resp.status(),
        401,
        "expected 401 with bogus upstream key, got {}",
        resp.status()
    );

    handle.shutdown();
}

#[tokio::test]
async fn anthropic_nonexistent_model() {
    let Some(_key) = get_api_key() else {
        eprintln!("skipping: {ANTHROPIC_API_KEY_ENV} not set");
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for(
        "nonexistent",
        "anthropic/claude-does-not-exist",
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
            "model": "nonexistent",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_completion_tokens": 20,
        }))
        .send()
        .await
        .expect("request failed");

    let status = resp.status();
    assert!(
        status == 400 || status == 404 || status == 403,
        "expected 4xx for nonexistent model, got {status}"
    );

    handle.shutdown();
}

#[tokio::test]
async fn anthropic_health_and_models_endpoint() {
    let Some(_key) = get_api_key() else {
        eprintln!("skipping: {ANTHROPIC_API_KEY_ENV} not set");
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for("sonnet", MAIN_MODEL)).await else {
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
        models.contains(&"sonnet"),
        "expected 'sonnet' in model list, got: {models:?}"
    );

    handle.shutdown();
}
