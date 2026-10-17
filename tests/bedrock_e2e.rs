//! Integration tests for the AWS Bedrock provider.
//!
//! Verifies end-to-end flow: an OpenAI-shaped client request is routed to a
//! Bedrock deployment, transformed to the Anthropic Messages body shape,
//! signed with AWS SigV4, forwarded to the mock upstream, and the native
//! Anthropic response is transformed back into the OpenAI chat-completion
//! shape.

use httpmock::MockServer;
use oxllm_server::OxllmParams;
use oxllm_server::app::{build_router, build_shared_state};
use oxllm_server::config::types::{ConfigModel, GeneralSettings, OxllmSettings, ProxyConfig};
use serde_json::{Value, json};

struct OxllmHandle {
    addr: std::net::SocketAddr,
    shutdown_tx: tokio::sync::oneshot::Sender<()>,
}

impl OxllmHandle {
    fn url(&self, path: &str) -> String {
        format!("http://{}{path}", self.addr)
    }
}

async fn start_oxllm(config: ProxyConfig) -> anyhow::Result<OxllmHandle> {
    let shared = build_shared_state(config)?;

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;
    let router = build_router(shared);

    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    tokio::spawn(async move {
        axum::serve(listener, router)
            .with_graceful_shutdown(async move {
                let _ = shutdown_rx.await;
            })
            .await
            .ok();
    });

    Ok(OxllmHandle { addr, shutdown_tx })
}

fn bedrock_config(mock_base_url: &str, model_name: &str, configured_model: &str) -> ProxyConfig {
    let mut extra = serde_json::Map::new();
    extra.insert(
        "aws_access_key_id".to_string(),
        serde_json::Value::String("AKIATESTKEY".to_string()),
    );
    extra.insert(
        "aws_secret_access_key".to_string(),
        serde_json::Value::String("supersecretkey".to_string()),
    );
    extra.insert(
        "aws_region".to_string(),
        serde_json::Value::String("us-east-1".to_string()),
    );

    ProxyConfig {
        model_list: vec![ConfigModel {
            model_name: model_name.to_string(),
            oxllm_params: OxllmParams {
                model: configured_model.to_string(),
                api_base: Some(mock_base_url.to_string()),
                timeout: Some(30.0),
                stream_timeout: Some(30),
                extra,
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

#[tokio::test]
async fn bedrock_anthropic_non_streaming_signs_and_routes_correctly() {
    let mock = MockServer::start();
    let config = bedrock_config(
        &mock.base_url(),
        "bedrock-claude",
        "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
    );
    let handle = start_oxllm(config).await.expect("server start");

    let _m = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path_matches(
                regex::Regex::new(r"^/model/anthropic\.claude-3-5-sonnet-20241022-v2:0/invoke$")
                    .unwrap(),
            )
            .header_exists("authorization")
            .header_exists("x-amz-date")
            .header_exists("x-amz-content-sha256")
            .json_body_partial(
                json!({
                "anthropic_version": "bedrock-2023-05-31",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 100,
                })
                .to_string(),
            );
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
            "id": "msg_bedrock_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-5-sonnet-20241022-v2:0",
            "content": [{"type": "text", "text": "Hello from Bedrock!"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 4},
            }));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .json(&json!({
        "model": "bedrock-claude",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 100,
        }))
        .send()
        .await
        .expect("request");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("json");
    assert_eq!(body["object"], "chat.completion");
    assert_eq!(body["id"], "msg_bedrock_1");
    assert_eq!(body["model"], "bedrock-claude");
    assert_eq!(
        body["choices"][0]["message"]["content"],
        "Hello from Bedrock!"
    );
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
    assert_eq!(body["usage"]["prompt_tokens"], 5);
    assert_eq!(body["usage"]["completion_tokens"], 4);
    assert_eq!(body["usage"]["total_tokens"], 9);

    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn bedrock_authorization_header_is_sigv4_shaped() {
    let mock = MockServer::start();
    let config = bedrock_config(
        &mock.base_url(),
        "bedrock-claude",
        "bedrock/anthropic.claude-3-haiku",
    );
    let handle = start_oxllm(config).await.expect("server start");

    let _m = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path_contains("/model/anthropic.claude-3-haiku/invoke")
            .header_exists("authorization")
            .header_exists("x-amz-date")
            .header_exists("x-amz-content-sha256")
            .header_exists("host");
        then.status(200).json_body(json!({
            "id": "msg", "model": "claude-3-haiku",
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .json(&json!({
        "model": "bedrock-claude",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 5,
        }))
        .send()
        .await
        .expect("request");

    assert_eq!(resp.status(), 200);
    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn bedrock_strips_bedrock_prefix_in_url() {
    let mock = MockServer::start();
    let config = bedrock_config(
        &mock.base_url(),
        "bedrock-claude",
        "bedrock/anthropic.claude-3-5-sonnet",
    );
    let handle = start_oxllm(config).await.expect("server start");

    let _m = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            // Must NOT contain a "bedrock/" segment in the URL path.
            .path("/model/anthropic.claude-3-5-sonnet/invoke");
        then.status(200).json_body(json!({
            "id": "msg", "model": "claude",
            "content": [{"type": "text", "text": "x"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .json(&json!({
        "model": "bedrock-claude",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 1,
        }))
        .send()
        .await
        .expect("request");

    assert_eq!(resp.status(), 200);
    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn bedrock_geo_prefixes_keep_signed_runtime_paths_and_family_features() {
    for (model, nova) in [
        ("jp.amazon.nova-2-lite-v1:0", true),
        ("au.anthropic.claude-sonnet-4-5-20250929-v1:0", false),
    ] {
        for explicit_converse in [false, true] {
            let mock = MockServer::start();
            let converse = nova || explicit_converse;
            let api = if converse { "converse" } else { "invoke" };
            let mut config = bedrock_config(
                &mock.base_url(),
                "regional-model",
                &format!("bedrock/{model}"),
            );
            if explicit_converse {
                config.model_list[0]
                    .oxllm_params
                    .extra
                    .insert("use_converse_api".into(), json!(true));
            }
            config.router_settings.routing.num_retries = Some(0);
            let handle = start_oxllm(config).await.expect("server start");
            let mut request = json!({
                "model": "regional-model", "max_tokens": 4096, "reasoning_effort": "low",
                "messages": [
                    {"role": "system", "content": [{"type": "text", "text": "instructions", "cache_control": {"type": "ephemeral", "ttl": "1h"}}]},
                    {"role": "user", "content": "look up the weather"}
                ],
                "tools": [{"type": "function", "function": {"name": "get.weather", "parameters": {"type": "object"}}, "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
                "tool_choice": "auto"
            });
            let thinking = json!({"type": "enabled", "budget_tokens": 1024});
            let serial = json!({"type": "auto", "disable_parallel_tool_use": true});
            let expected = if nova {
                request["web_search_options"] = json!({});
                json!({
                    "inferenceConfig": {"maxTokens": 4096},
                    "additionalModelRequestFields": {"reasoningConfig": {"type": "enabled", "maxReasoningEffort": "low"}},
                    "toolConfig": {"tools": [
                        {"toolSpec": {"name": "get_weather"}},
                        {"systemTool": {"name": "nova_grounding"}}
                    ], "toolChoice": {"auto": {}}}
                })
            } else {
                request["top_k"] = json!(20);
                request["parallel_tool_calls"] = json!(false);
                if converse {
                    json!({
                        "inferenceConfig": {"maxTokens": 4096},
                        "additionalModelRequestFields": {"thinking": thinking, "top_k": 20, "tool_choice": serial},
                        "system": [{"text": "instructions"}, {"cachePoint": {"type": "default", "ttl": "1h"}}],
                        "toolConfig": {"tools": [{"toolSpec": {"name": "get_weather"}}]}
                    })
                } else {
                    json!({
                        "anthropic_version": "bedrock-2023-05-31", "max_tokens": 4096,
                        "thinking": thinking, "top_k": 20, "tool_choice": serial,
                        "system": request["messages"][0]["content"],
                        "tools": [{"name": "get_weather", "cache_control": {"type": "ephemeral", "ttl": "1h"}}]
                    })
                }
            };
            let upstream = mock.mock(|when, then| {
                when.method(httpmock::Method::POST)
                    .path(format!("/model/{model}/{api}"))
                    .header_exists("authorization")
                    .header_exists("x-amz-date")
                    .header_exists("x-amz-content-sha256")
                    .json_body_partial(expected.to_string());
                let response = if converse {
                    json!({
                        "output": {"message": {"role": "assistant", "content": [{"text": "regional answer"}]}},
                        "stopReason": "end_turn", "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15}
                    })
                } else {
                    json!({
                        "id": "msg-regional", "type": "message", "role": "assistant", "model": model,
                        "content": [{"type": "text", "text": "regional answer"}],
                        "stop_reason": "end_turn", "usage": {"input_tokens": 10, "output_tokens": 5}
                    })
                };
                then.status(200).json_body(response);
            });
            let client = reqwest::Client::new();
            let response = client
                .post(handle.url("/v1/chat/completions"))
                .json(&request)
                .send()
                .await
                .expect("request");
            let status = response.status();
            let body: Value = response.json().await.expect("json");
            assert_eq!(
                status, 200,
                "{model}, explicit_converse={explicit_converse}: {body}"
            );
            assert_eq!(body["model"], "regional-model");
            assert_eq!(body["choices"][0]["message"]["content"], "regional answer");

            if nova {
                request["top_k"] = json!(20);
                let response = client
                    .post(handle.url("/v1/chat/completions"))
                    .json(&request)
                    .send()
                    .await
                    .expect("unsupported top_k request");
                let status = response.status();
                let body: Value = response.json().await.expect("error json");
                assert_eq!(status, 400, "Nova 2 top_k must still fail locally: {body}");
                assert!(body["error"]["message"].as_str().unwrap().contains("top_k"));
            }
            upstream.assert_hits(1);
            let _ = handle.shutdown_tx.send(());
        }
    }
}

#[tokio::test]
async fn bedrock_noncanonical_geo_heads_keep_custom_invoke_fields_unless_converse_is_explicit() {
    for model in [
        "jp.jp.amazon.nova-2-lite-v1:0",
        "au.au.amazon.nova-2-lite-v1:0",
        "jp.au.amazon.nova-2-lite-v1:0",
        "au.jp.amazon.nova-2-lite-v1:0",
        "us.eu.amazon.nova-2-lite-v1:0",
        "jp.us-east-1.amazon.nova-2-lite-v1:0",
        "us-east-1.au.amazon.nova-2-lite-v1:0",
        "jp..amazon.nova-2-lite-v1:0",
        "au.-amazon.nova-2-lite-v1:0",
        "au.jp.anthropic.claude-sonnet-4-5-20250929-v1:0",
    ] {
        for explicit_converse in [false, true] {
            let mock = MockServer::start();
            let mut config = bedrock_config(
                &mock.base_url(),
                "custom-model",
                &format!("bedrock/{model}"),
            );
            config.model_list[0]
                .oxllm_params
                .extra
                .insert("use_converse_api".into(), json!(explicit_converse));
            config.router_settings.routing.num_retries = Some(0);
            let handle = start_oxllm(config).await.expect("server start");
            let request = json!({
                "model": "custom-model", "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 32, "custom_field": {"preserve": true}
            });
            let api = if explicit_converse {
                "converse"
            } else {
                "invoke"
            };
            let expected = if explicit_converse {
                json!({"messages": [{"role": "user", "content": [{"text": "hello"}]}], "inferenceConfig": {"maxTokens": 32}})
            } else {
                let mut body = request.clone();
                body["model"] = json!(model);
                body
            };
            let upstream = mock.mock(|when, then| {
                when.method(httpmock::Method::POST)
                    .path(format!("/model/{model}/{api}"))
                    .header_exists("authorization")
                    .header_exists("x-amz-date")
                    .header_exists("x-amz-content-sha256")
                    .json_body_partial(expected.to_string());
                let response = if explicit_converse {
                    json!({
                        "output": {"message": {"role": "assistant", "content": [{"text": "custom answer"}]}},
                        "stopReason": "end_turn", "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}
                    })
                } else {
                    json!({
                        "id": "chatcmpl-custom", "object": "chat.completion", "model": model,
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": "custom answer"}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
                    })
                };
                then.status(200).json_body(response);
            });
            let response = reqwest::Client::new()
                .post(handle.url("/v1/chat/completions"))
                .json(&request)
                .send()
                .await
                .expect("request");
            let status = response.status();
            let body: Value = response.json().await.expect("json");
            assert_eq!(
                status, 200,
                "{model}, explicit_converse={explicit_converse}: {body}"
            );
            assert_eq!(body["model"], "custom-model");
            assert_eq!(body["choices"][0]["message"]["content"], "custom answer");
            upstream.assert_hits(1);
            let _ = handle.shutdown_tx.send(());
        }
    }
}
