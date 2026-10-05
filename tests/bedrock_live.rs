//! Live integration tests against the real AWS Bedrock API through the
//! oxllm server.
//!
//! The Bedrock credential resolver is ENV-ONLY: the test process must have
//! `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_REGION` exported
//! (extract the key values from `~/.aws/credentials` — never hard-code them
//! here). When the env vars are unset the suite prints "skipping" and returns.
//!
//! Run with:
//! ```sh
//! export AWS_ACCESS_KEY_ID=<id> AWS_SECRET_ACCESS_KEY=<secret> AWS_REGION=us-east-1
//! cargo test --test bedrock_live --all-features -- --nocapture
//! ```

use oxllm_server::OxllmParams;
use oxllm_server::app::{build_router, build_shared_state};
use oxllm_server::config::types::{ConfigModel, GeneralSettings, OxllmSettings, ProxyConfig};
use serde_json::{Value, json};

const AWS_ACCESS_KEY_ENV: &str = "AWS_ACCESS_KEY_ID";
const AWS_SECRET_KEY_ENV: &str = "AWS_SECRET_ACCESS_KEY";
const AWS_REGION_ENV: &str = "AWS_REGION";

const MASTER_KEY: &str = "sk-test-key-master-1";
const CLAUDE_MODEL: &str = "claude";
const CLAUDE_MODEL_KEY: &str = "bedrock/us.anthropic.claude-sonnet-4-6";

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

/// Read the AWS credential env vars. The test module is skipped entirely
/// (early-return with a "skipping" message) when any of them is unset.
fn get_aws_env() -> Option<(String, String, String)> {
    let id = std::env::var(AWS_ACCESS_KEY_ENV)
        .ok()
        .filter(|s| !s.is_empty())?;
    let secret = std::env::var(AWS_SECRET_KEY_ENV)
        .ok()
        .filter(|s| !s.is_empty())?;
    let region = std::env::var(AWS_REGION_ENV)
        .ok()
        .filter(|s| !s.is_empty())?;
    Some((id, secret, region))
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

fn base_config(model_list: Vec<ConfigModel>) -> ProxyConfig {
    ProxyConfig {
        model_list,
        oxllm_settings: OxllmSettings::default(),
        general_settings: GeneralSettings {
            master_key: Some(MASTER_KEY.to_string()),
            allow_private_api_base: Some(true),
            ..Default::default()
        },
        router_settings: Default::default(),
        extra: serde_json::Map::new(),
    }
}

/// Bedrock creds come from the environment (`AWS_ACCESS_KEY_ID` /
/// `AWS_SECRET_ACCESS_KEY` / `AWS_REGION`) — no `api_key` in the config.
fn proxy_config_for(model_name: &str, model_key: &str) -> ProxyConfig {
    base_config(vec![ConfigModel {
        model_name: model_name.to_string(),
        oxllm_params: OxllmParams {
            model: model_key.to_string(),
            api_key: None,
            api_base: None,
            timeout: Some(60.0),
            stream_timeout: Some(60),
            ..Default::default()
        },
        ..Default::default()
    }])
}

fn skip_aws_env() {
    eprintln!(
        "skipping: {AWS_ACCESS_KEY_ENV}/{AWS_SECRET_KEY_ENV}/{AWS_REGION_ENV} not set (export them, e.g. from ~/.aws/credentials profile 'anthropic-cr-bedrock')"
    );
}

/// POST a chat completion and return (status, parsed body).
async fn post_chat(
    client: &reqwest::Client,
    url: &str,
    body: &Value,
) -> (reqwest::StatusCode, Value) {
    let resp = client
        .post(url)
        .bearer_auth(MASTER_KEY)
        .json(body)
        .send()
        .await
        .expect("request failed");
    let status = resp.status();
    let text = resp.text().await.unwrap_or_default();
    match serde_json::from_str::<Value>(&text) {
        Ok(v) => (status, v),
        Err(_) => (status, json!({ "raw": text })),
    }
}

#[tokio::test]
async fn bedrock_non_streaming_chat() {
    let Some(_env) = get_aws_env() else {
        skip_aws_env();
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for(CLAUDE_MODEL, CLAUDE_MODEL_KEY)).await else {
        eprintln!("skipping: failed to start oxllm");
        return;
    };

    let client = reqwest::Client::new();
    let (status, body) = post_chat(
        &client,
        &handle.url("/v1/chat/completions"),
        &json!({
            "model": CLAUDE_MODEL,
            "messages": [{"role": "user", "content": "Say hello in one word"}],
            "max_tokens": 20,
        }),
    )
    .await;

    assert_eq!(status, 200, "Bedrock request failed: {body}");
    let content = body["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("");
    assert!(!content.is_empty(), "response had empty content");
    assert!(
        body["usage"]["prompt_tokens"].as_u64().is_some()
            && body["usage"]["completion_tokens"].as_u64().is_some(),
        "expected usage with prompt/completion tokens, got: {body}"
    );
    eprintln!("Bedrock response: {content}");
    eprintln!("usage: {}", body["usage"]);

    handle.shutdown();
}

#[tokio::test]
async fn bedrock_streaming_chat() {
    let Some(_env) = get_aws_env() else {
        skip_aws_env();
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for(CLAUDE_MODEL, CLAUDE_MODEL_KEY)).await else {
        eprintln!("skipping: failed to start oxllm");
        return;
    };

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth(MASTER_KEY)
        .json(&json!({
            "model": CLAUDE_MODEL,
            "messages": [{"role": "user", "content": "Count to 5"}],
            "stream": true,
            "max_tokens": 100,
        }))
        .send()
        .await
        .expect("request failed");

    assert_eq!(resp.status(), 200, "Bedrock streaming request failed");
    let body = resp.text().await.expect("body");
    assert!(body.contains("data: "), "expected SSE data, got: {body}");
    assert!(body.contains("[DONE]"), "expected [DONE] terminator");

    // Anthropic-on-Bedrock streams `message_start` / `content_block_delta`
    // events (delivered as eventstream frames, re-emitted as SSE lines).
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
    eprintln!("Streamed text ({} chars): {full_text}", full_text.len());

    handle.shutdown();
}

#[tokio::test]
async fn bedrock_json_object_mode() {
    let Some(_env) = get_aws_env() else {
        skip_aws_env();
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for(CLAUDE_MODEL, CLAUDE_MODEL_KEY)).await else {
        eprintln!("skipping: failed to start oxllm");
        return;
    };

    let client = reqwest::Client::new();
    let (status, body) = post_chat(
        &client,
        &handle.url("/v1/chat/completions"),
        &json!({
            "model": CLAUDE_MODEL,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that always responds with valid JSON."},
                {"role": "user", "content": "Create a JSON object for a person named Alice who is 25 years old, with fields 'name' and 'age'."}
            ],
            "response_format": {"type": "json_object"},
        }),
    )
    .await;

    assert_eq!(status, 200, "Bedrock JSON mode failed: {body}");
    let content = body["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("");
    assert!(!content.is_empty(), "response had empty content");
    // Sonnet 4.6 takes the native `output_config` path: `json_object`
    // without a schema becomes a permissive schema, so the model may
    // return any valid JSON object (it has been observed returning `{}`
    // or inventing its own fields). Assert the JSON-mode contract — a
    // parseable JSON object — rather than specific fields.
    let parsed: Value = serde_json::from_str(content)
        .unwrap_or_else(|_| panic!("response must be valid JSON, got: {content}"));
    assert!(
        parsed.is_object(),
        "response must be a JSON object, got: {content}"
    );
    eprintln!("JSON mode response: {content}");

    handle.shutdown();
}

#[tokio::test]
async fn bedrock_multiple_models() {
    let Some(_env) = get_aws_env() else {
        skip_aws_env();
        return;
    };

    let models: Vec<(&str, &str)> = vec![
        (CLAUDE_MODEL, CLAUDE_MODEL_KEY),
        // Legacy `us.anthropic.claude-3-haiku-20240307-v1:0` returns
        // "Access denied ... marked by provider as Legacy" on this account;
        // the modern haiku-4.5 cross-region profile is accessible.
        (
            "haiku",
            "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        ),
    ];

    let config = base_config(
        models
            .iter()
            .map(|(name, model_key)| ConfigModel {
                model_name: name.to_string(),
                oxllm_params: OxllmParams {
                    model: model_key.to_string(),
                    api_key: None,
                    api_base: None,
                    timeout: Some(60.0),
                    stream_timeout: Some(60),
                    ..Default::default()
                },
                ..Default::default()
            })
            .collect(),
    );

    let Some(handle) = start_oxllm(config).await else {
        eprintln!("skipping: failed to start oxllm");
        return;
    };

    let client = reqwest::Client::new();
    for (model_name, _) in &models {
        let (status, body) = post_chat(
            &client,
            &handle.url("/v1/chat/completions"),
            &json!({
                "model": model_name,
                "messages": [{"role": "user", "content": "Say hello in one word"}],
                "max_tokens": 20,
            }),
        )
        .await;

        assert_eq!(status, 200, "model {model_name} failed: {body}");
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
async fn bedrock_with_stop_sequence() {
    let Some(_env) = get_aws_env() else {
        skip_aws_env();
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for(CLAUDE_MODEL, CLAUDE_MODEL_KEY)).await else {
        eprintln!("skipping: failed to start oxllm");
        return;
    };

    let client = reqwest::Client::new();
    let (status, body) = post_chat(
        &client,
        &handle.url("/v1/chat/completions"),
        &json!({
            "model": CLAUDE_MODEL,
            "messages": [{"role": "user", "content": "Say 'Hello world' exactly"}],
            "stop": ["world"],
            "max_tokens": 20,
        }),
    )
    .await;

    assert_eq!(status, 200, "Bedrock stop sequence failed: {body}");
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
async fn bedrock_max_tokens() {
    let Some(_env) = get_aws_env() else {
        skip_aws_env();
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for(CLAUDE_MODEL, CLAUDE_MODEL_KEY)).await else {
        eprintln!("skipping: failed to start oxllm");
        return;
    };

    let client = reqwest::Client::new();
    let (status, body) = post_chat(
        &client,
        &handle.url("/v1/chat/completions"),
        &json!({
            "model": CLAUDE_MODEL,
            "messages": [{"role": "user", "content": "Write a very long essay about AI"}],
            "max_tokens": 20,
        }),
    )
    .await;

    assert_eq!(status, 200, "Bedrock max tokens failed: {body}");
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
        "response should be short with max_tokens=20, got {} chars: {content}",
        content.len()
    );
    let finish_reason = body["choices"][0]["finish_reason"].as_str().unwrap_or("");
    assert!(
        matches!(finish_reason, "length" | "stop"),
        "expected finish_reason 'length' or 'stop', got '{finish_reason}'"
    );
    eprintln!("finish_reason: {finish_reason}");

    handle.shutdown();
}

#[tokio::test]
async fn bedrock_long_context() {
    let Some(_env) = get_aws_env() else {
        skip_aws_env();
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for(CLAUDE_MODEL, CLAUDE_MODEL_KEY)).await else {
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
    let (status, body) = post_chat(
        &client,
        &handle.url("/v1/chat/completions"),
        &json!({
            "model": CLAUDE_MODEL,
            "messages": [{"role": "user", "content": long_prompt}],
            "max_tokens": 100,
        }),
    )
    .await;

    assert_eq!(status, 200, "Bedrock long context failed: {body}");
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
async fn bedrock_invalid_auth_returns_401() {
    let Some(_env) = get_aws_env() else {
        skip_aws_env();
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for(CLAUDE_MODEL, CLAUDE_MODEL_KEY)).await else {
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
async fn bedrock_nonexistent_model() {
    let Some(_env) = get_aws_env() else {
        skip_aws_env();
        return;
    };

    // A configured model name whose Bedrock id does not exist — the request
    // reaches Bedrock, which answers with a ValidationException (400).
    let config = base_config(vec![
        ConfigModel {
            model_name: CLAUDE_MODEL.to_string(),
            oxllm_params: OxllmParams {
                model: CLAUDE_MODEL_KEY.to_string(),
                api_key: None,
                api_base: None,
                timeout: Some(60.0),
                stream_timeout: Some(60),
                ..Default::default()
            },
            ..Default::default()
        },
        ConfigModel {
            model_name: "badmodel".to_string(),
            oxllm_params: OxllmParams {
                model: "bedrock/us.anthropic.claude-does-not-exist-v9:0".to_string(),
                api_key: None,
                api_base: None,
                timeout: Some(60.0),
                stream_timeout: Some(60),
                ..Default::default()
            },
            ..Default::default()
        },
    ]);

    let Some(handle) = start_oxllm(config).await else {
        eprintln!("skipping: failed to start oxllm");
        return;
    };

    let client = reqwest::Client::new();
    let (status, body) = post_chat(
        &client,
        &handle.url("/v1/chat/completions"),
        &json!({
            "model": "badmodel",
            "messages": [{"role": "user", "content": "Hello"}],
        }),
    )
    .await;

    assert!(
        status == 400 || status == 404,
        "expected 400 or 404 for nonexistent model, got {status}: {body}"
    );
    eprintln!("Nonexistent model rejected with {status}: {body}");

    handle.shutdown();
}

#[tokio::test]
async fn bedrock_health_endpoint() {
    let Some(_env) = get_aws_env() else {
        skip_aws_env();
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for(CLAUDE_MODEL, CLAUDE_MODEL_KEY)).await else {
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
async fn bedrock_models_endpoint() {
    let Some(_env) = get_aws_env() else {
        skip_aws_env();
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for(CLAUDE_MODEL, CLAUDE_MODEL_KEY)).await else {
        eprintln!("skipping: failed to start oxllm");
        return;
    };

    let client = reqwest::Client::new();
    let resp = client
        .get(handle.url("/v1/models"))
        .bearer_auth(MASTER_KEY)
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
        models.contains(&CLAUDE_MODEL),
        "expected '{CLAUDE_MODEL}' in model list, got: {models:?}"
    );

    handle.shutdown();
}

#[tokio::test]
async fn bedrock_system_prompt_and_temperature() {
    let Some(_env) = get_aws_env() else {
        skip_aws_env();
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for(CLAUDE_MODEL, CLAUDE_MODEL_KEY)).await else {
        eprintln!("skipping: failed to start oxllm");
        return;
    };

    let client = reqwest::Client::new();
    let (status, body) = post_chat(
        &client,
        &handle.url("/v1/chat/completions"),
        &json!({
            "model": CLAUDE_MODEL,
            "messages": [
                {"role": "system", "content": "You are a terse assistant. Reply with a single word."},
                {"role": "user", "content": "Say hello"}
            ],
            "temperature": 0.7,
            "max_tokens": 10,
        }),
    )
    .await;

    assert_eq!(
        status, 200,
        "Bedrock system prompt + temperature failed: {body}"
    );
    let content = body["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("");
    assert!(!content.is_empty(), "response had empty content");
    eprintln!("System prompt + temperature response: {content}");

    handle.shutdown();
}

#[tokio::test]
async fn bedrock_tool_call() {
    let Some(_env) = get_aws_env() else {
        skip_aws_env();
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for(CLAUDE_MODEL, CLAUDE_MODEL_KEY)).await else {
        eprintln!("skipping: failed to start oxllm");
        return;
    };

    let client = reqwest::Client::new();
    let (status, body) = post_chat(
        &client,
        &handle.url("/v1/chat/completions"),
        &json!({
            "model": CLAUDE_MODEL,
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
        }),
    )
    .await;

    assert_eq!(status, 200, "Bedrock tool call failed: {body}");
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
    eprintln!("Tool call: {}", tool_calls[0]);

    handle.shutdown();
}

#[tokio::test]
async fn bedrock_seed() {
    let Some(_env) = get_aws_env() else {
        skip_aws_env();
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for(CLAUDE_MODEL, CLAUDE_MODEL_KEY)).await else {
        eprintln!("skipping: failed to start oxllm");
        return;
    };

    // Anthropic does not support `seed` — the transform drops it. Just
    // assert the request still succeeds.
    let client = reqwest::Client::new();
    let (status, body) = post_chat(
        &client,
        &handle.url("/v1/chat/completions"),
        &json!({
            "model": CLAUDE_MODEL,
            "messages": [{"role": "user", "content": "Say hello in one word"}],
            "seed": 42,
            "max_tokens": 10,
        }),
    )
    .await;

    assert_eq!(status, 200, "Bedrock request with seed failed: {body}");
    let content = body["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("");
    assert!(!content.is_empty(), "response had empty content");
    eprintln!("Seed response: {content}");

    handle.shutdown();
}

#[tokio::test]
async fn bedrock_streaming_include_usage() {
    let Some(_env) = get_aws_env() else {
        skip_aws_env();
        return;
    };

    let Some(handle) = start_oxllm(proxy_config_for(CLAUDE_MODEL, CLAUDE_MODEL_KEY)).await else {
        eprintln!("skipping: failed to start oxllm");
        return;
    };

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth(MASTER_KEY)
        .json(&json!({
            "model": CLAUDE_MODEL,
            "messages": [{"role": "user", "content": "Count to 5"}],
            "stream": true,
            "stream_options": {"include_usage": true},
            "max_tokens": 50,
        }))
        .send()
        .await
        .expect("request failed");

    assert_eq!(resp.status(), 200, "Bedrock streaming with usage failed");
    let body = resp.text().await.expect("body");
    assert!(body.contains("data: "), "expected SSE data, got: {body}");
    assert!(body.contains("[DONE]"), "expected [DONE] terminator");

    // The host synthesises a usage chunk before [DONE] when
    // `stream_options.include_usage` is set.
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
