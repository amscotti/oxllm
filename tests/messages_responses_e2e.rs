//! Integration tests: `/v1/messages` (Anthropic) and `/v1/responses`.

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

fn make_proxy_config(mock_base_url: &str, model_name: &str, model: &str) -> ProxyConfig {
    ProxyConfig {
        model_list: vec![ConfigModel {
            model_name: model_name.to_string(),
            oxllm_params: OxllmParams {
                model: model.to_string(),
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

#[tokio::test]
async fn native_messages_adds_null_stop_sequence_without_rewriting_the_response() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), "claude", "anthropic/claude-3-5-sonnet");
    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST).path("/messages");
        then.status(200).json_body(json!({
            "id": "msg-native-null-stop",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-5-sonnet",
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 2, "output_tokens": 1}
        }));
    });
    let handle = start_oxllm(config).await.unwrap();
    let response = reqwest::Client::new()
        .post(handle.url("/anthropic/v1/messages"))
        .json(&json!({
            "model": "claude", "max_tokens": 10,
            "messages": [{"role": "user", "content": "hello"}]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let body: Value = response.json().await.unwrap();
    assert!(body["stop_sequence"].is_null());
    assert_eq!(body["id"], "msg-native-null-stop");
    assert_eq!(body["content"][0]["text"], "ok");
    upstream.assert_hits(1);
    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn malformed_tool_result_text_is_rejected_before_any_upstream_call() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), "claude", "openai/gpt-4o");
    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST);
        then.status(200).json_body(json!({}));
    });
    let handle = start_oxllm(config).await.unwrap();
    for invalid in [Value::Null, json!(42)] {
        for path in [
            "/v1/messages",
            "/v1/messages/count_tokens",
            "/utils/token_counter",
        ] {
            let response = reqwest::Client::new()
                .post(handle.url(path))
                .json(&json!({
                    "model": "claude",
                    "max_tokens": 10,
                    "messages": [{"role": "user", "content": [{
                        "type": "tool_result", "tool_use_id": "call-1",
                        "content": [{"type": "text", "text": invalid}],
                    }]}],
                }))
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), 400, "{path}: {invalid}");
            let body = response.text().await.unwrap();
            assert!(body.contains("tool_result text must be a string"), "{body}");
        }
    }
    upstream.assert_hits(0);
    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn translated_messages_forward_client_betas_and_keep_pdf_documents() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), "claude", "anthropic/claude-3-5-sonnet");
    let document = json!({"type":"document","source":{"type":"base64","media_type":"application/pdf","data":"JVBERi0="}});
    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/messages")
            .header("anthropic-beta", "pdfs-2024-09-25")
            .header("x-api-key", "sk-test-key")
            .json_body_partial(
                json!({"messages":[{"role":"user","content":[document.clone()]}]}).to_string(),
            );
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
                "id":"msg-pdf","type":"message","role":"assistant","model":"claude-3-5-sonnet",
                "content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn",
                "usage":{"input_tokens":2,"output_tokens":1}
            }));
    });
    let handle = start_oxllm(config).await.unwrap();
    let response = reqwest::Client::new().post(handle.url("/v1/messages"))
        .header("anthropic-beta", "pdfs-2024-09-25")
        .json(&json!({"model":"claude","max_tokens":10,"messages":[{"role":"user","content":[document]}]}))
        .send().await.unwrap();
    assert_eq!(response.status(), 200);
    upstream.assert_hits(1);
    let _ = handle.shutdown_tx.send(());
}

// ===========================================================================
// /v1/messages — Anthropic Messages input API
// ===========================================================================

#[tokio::test]
async fn test_messages_non_streaming_openai_upstream() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), "claude", "openai/gpt-4o");
    let handle = start_oxllm(config).await.expect("server start");

    let _m = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .json_body_partial(
                json!({
                "messages": [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "Say hi"},
                ],
                "max_tokens": 100,
                })
                .to_string(),
            );
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "model": "gpt-4o",
            "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Hello!"},
            "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}
            }));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/messages"))
        .json(&json!({
        "model": "claude",
        "system": "Be brief.",
        "messages": [{"role": "user", "content": "Say hi"}],
        "max_tokens": 100,
        }))
        .send()
        .await
        .expect("request");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("json body");
    assert_eq!(body["type"], "message");
    assert_eq!(body["role"], "assistant");
    assert_eq!(body["content"][0]["type"], "text");
    assert_eq!(body["content"][0]["text"], "Hello!");
    assert_eq!(body["stop_reason"], "end_turn");
    assert_eq!(body["usage"]["input_tokens"], 10);
    assert_eq!(body["usage"]["output_tokens"], 3);

    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn test_messages_streaming_openai_upstream() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), "claude", "openai/gpt-4o");
    let handle = start_oxllm(config).await.expect("server start");

    let sse_body = concat!(
        "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"!\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
        "data: {\"id\":\"chatcmpl-1\",\"model\":\"gpt-4o\",\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":2,\"total_tokens\":7}}\n\n",
        "data: [DONE]\n\n"
    );

    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .json_body_partial(json!({"stream_options": {"include_usage": true}}).to_string());
        then.status(200)
            .header("content-type", "text/event-stream")
            .body(sse_body);
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/messages"))
        .json(&json!({
        "model": "claude",
        "messages": [{"role": "user", "content": "Say hi"}],
        "max_tokens": 100,
        "stream": true,
        }))
        .send()
        .await
        .expect("request");

    assert_eq!(resp.status(), 200);
    let text = resp.text().await.expect("text");

    assert!(
        text.contains("message_start"),
        "missing message_start: {text}"
    );
    assert!(
        text.contains("\"text_delta\""),
        "missing text_delta: {text}"
    );
    assert!(text.contains("Hello"), "missing Hello: {text}");
    assert!(
        text.contains("message_delta"),
        "missing message_delta: {text}"
    );
    assert!(
        text.contains("\"end_turn\""),
        "missing end_turn stop_reason: {text}"
    );
    assert!(
        text.contains("message_stop"),
        "missing message_stop: {text}"
    );
    let deltas: Vec<Value> = text
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter_map(|data| serde_json::from_str::<Value>(data).ok())
        .filter(|event| event["type"] == "message_delta")
        .collect();
    assert_eq!(deltas.len(), 1, "exactly one terminal delta: {text}");
    assert_eq!(deltas[0]["usage"]["input_tokens"], 5);
    assert_eq!(deltas[0]["usage"]["output_tokens"], 2);
    upstream.assert_hits(1);

    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn test_messages_missing_model_returns_400() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), "claude", "openai/gpt-4o");
    let handle = start_oxllm(config).await.expect("server start");

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/messages"))
        .json(&json!({"messages": [], "max_tokens": 10}))
        .send()
        .await
        .expect("request");

    assert_eq!(resp.status(), 400);
    let _ = handle.shutdown_tx.send(());
}

// ===========================================================================
// /v1/responses — OpenAI Responses API
// ===========================================================================

#[tokio::test]
async fn responses_schema_format_reaches_openai_and_anthropic_for_json_and_streams() {
    let schema = json!({"type":"object", "properties":{"answer":{"type":"string"}},
        "required":["answer"], "additionalProperties":false});
    for model in ["openai/gpt-4o", "anthropic/claude-haiku-4-5"] {
        for stream in [false, true] {
            let mock = MockServer::start_async().await;
            let anthropic = model.starts_with("anthropic/");
            let expected = if anthropic {
                json!({"output_config":{"format":{"type":"json_schema", "schema":schema}}})
            } else {
                json!({"response_format":{"type":"json_schema", "json_schema":{
                    "name":"answer", "strict":true, "description":"A structured answer", "schema":schema
                }}})
            };
            let upstream = mock.mock_async(|when, then| {
                when.method(httpmock::Method::POST)
                    .path(if anthropic { "/messages" } else { "/chat/completions" })
                    .json_body_partial(expected.to_string());
                if stream {
                    let chunks = if anthropic {
                        vec![
                            json!({"type":"message_start", "message":{"id":"msg-schema", "model":"claude-haiku-4-5", "usage":{"input_tokens":5, "output_tokens":0}}}),
                            json!({"type":"content_block_delta", "index":0, "delta":{"type":"text_delta", "text":"{\"answer\":\"ok\"}"}}),
                            json!({"type":"message_delta", "delta":{"stop_reason":"end_turn"}, "usage":{"output_tokens":3}}),
                            json!({"type":"message_stop"}),
                        ]
                    } else {
                        vec![
                            json!({"id":"chatcmpl-schema", "model":"gpt-4o", "choices":[{"index":0, "delta":{"content":"{\"answer\":\"ok\"}"}, "finish_reason":null}]}),
                            json!({"choices":[{"index":0, "delta":{}, "finish_reason":"stop"}]}),
                            json!({"choices":[], "usage":{"prompt_tokens":5, "completion_tokens":3, "total_tokens":8}}),
                        ]
                    };
                    let mut body = chunks.iter().map(|chunk| format!("data: {chunk}\n\n")).collect::<String>();
                    if !anthropic { body.push_str("data: [DONE]\n\n"); }
                    then.status(200).header("content-type", "text/event-stream").body(body);
                } else {
                    then.status(200).json_body(if anthropic {
                        json!({"id":"msg-schema", "type":"message", "role":"assistant", "model":"claude-haiku-4-5",
                            "content":[{"type":"text", "text":"{\"answer\":\"ok\"}"}], "stop_reason":"end_turn",
                            "usage":{"input_tokens":5, "output_tokens":3}})
                    } else {
                        json!({"id":"chatcmpl-schema", "model":"gpt-4o", "choices":[{"index":0,
                            "message":{"role":"assistant", "content":"{\"answer\":\"ok\"}"}, "finish_reason":"stop"}],
                            "usage":{"prompt_tokens":5, "completion_tokens":3, "total_tokens":8}})
                    });
                }
            }).await;
            let handle = start_oxllm(make_proxy_config(&mock.base_url(), "structured", model))
                .await
                .unwrap();
            let response = reqwest::Client::new()
                .post(handle.url("/v1/responses"))
                .json(
                    &json!({"model":"structured", "input":"answer", "stream":stream,
                    "text":{"format":{"type":"json_schema", "name":"answer", "strict":true,
                        "description":"A structured answer", "schema":schema}}}),
                )
                .send()
                .await
                .unwrap();
            assert_eq!(
                response.status(),
                200,
                "{model}, stream={stream}: {}",
                response.text().await.unwrap()
            );
            let body: Value = if stream {
                let text = response.text().await.unwrap();
                text.lines()
                    .filter_map(|line| line.strip_prefix("data: "))
                    .filter_map(|data| serde_json::from_str::<Value>(data).ok())
                    .find(|event| event["type"] == "response.completed")
                    .expect("completed event")["response"]
                    .clone()
            } else {
                response.json().await.unwrap()
            };
            assert_eq!(body["status"], "completed");
            assert_eq!(
                serde_json::from_str::<Value>(
                    body["output"][0]["content"][0]["text"].as_str().unwrap()
                )
                .unwrap(),
                json!({"answer":"ok"})
            );
            assert_eq!(body["usage"]["input_tokens"], 5);
            assert_eq!(body["usage"]["output_tokens"], 3);
            upstream.assert_hits_async(1).await;
            let _ = handle.shutdown_tx.send(());
        }
    }
}

#[tokio::test]
async fn responses_invalid_formats_are_local_client_errors_without_cooldown() {
    let mock = MockServer::start_async().await;
    let upstream = mock
        .mock_async(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/chat/completions");
            then.status(200).json_body(
                json!({"id":"chatcmpl-healthy", "model":"gpt-4o", "choices":[{
            "index":0, "message":{"role":"assistant", "content":"ok"}, "finish_reason":"stop"
        }], "usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}),
            );
        })
        .await;
    let handle = start_oxllm(make_proxy_config(
        &mock.base_url(),
        "structured",
        "openai/gpt-4o",
    ))
    .await
    .unwrap();
    let client = reqwest::Client::new();
    for format in [
        json!("invalid"),
        json!({"type":"unsupported"}),
        json!({"type":"json_schema", "name":"answer"}),
        json!({"type":"json_schema", "name":"answer", "schema":{}, "strict":"true"}),
        json!({"type":"json_schema", "name":"two words", "schema":{"type":"object"}}),
        json!({"type":"json_schema", "name":"a".repeat(65), "schema":{"type":"object"}}),
    ] {
        let response = client
            .post(handle.url("/v1/responses"))
            .json(&json!({"model":"structured", "input":"answer", "text":{"format":format}}))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 400);
        let body: Value = response.json().await.unwrap();
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("text.format")
        );
    }
    upstream.assert_hits_async(0).await;
    let response = client
        .post(handle.url("/v1/responses"))
        .json(&json!({"model":"structured", "input":"answer"}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    upstream.assert_hits_async(1).await;
    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn test_responses_non_streaming() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), "gpt-4o", "openai/gpt-4o");
    let handle = start_oxllm(config).await.expect("server start");

    let _m = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .json_body_partial(
                json!({
                "model": "gpt-4o",
                "messages": [
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "Hello!"},
                ],
                })
                .to_string(),
            );
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
            "id": "chatcmpl-abc123",
            "object": "chat.completion",
            "model": "gpt-4o",
            "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Hi there!"},
            "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
            }));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/responses"))
        .json(&json!({
        "model": "gpt-4o",
        "input": "Hello!",
        "instructions": "Be helpful.",
        }))
        .send()
        .await
        .expect("request");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("json body");
    assert_eq!(body["object"], "response");
    assert!(body["id"].as_str().expect("id").starts_with("resp_"));
    assert_eq!(body["model"], "gpt-4o");
    assert_eq!(body["output"][0]["type"], "message");
    assert_eq!(body["output"][0]["role"], "assistant");
    assert_eq!(body["output"][0]["content"][0]["type"], "output_text");
    assert_eq!(body["output"][0]["content"][0]["text"], "Hi there!");
    assert_eq!(body["usage"]["input_tokens"], 10);
    assert_eq!(body["usage"]["output_tokens"], 5);
    assert_eq!(body["usage"]["total_tokens"], 15);
    assert_eq!(body["status"], "completed");

    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn test_responses_array_input() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), "gpt-4o", "openai/gpt-4o");
    let handle = start_oxllm(config).await.expect("server start");

    let _m = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
            "id": "chatcmpl-x",
            "object": "chat.completion",
            "model": "gpt-4o",
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
        .post(handle.url("/v1/responses"))
        .json(&json!({
        "model": "gpt-4o",
        "input": [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
        {"role": "user", "content": "Bye"},
        ],
        }))
        .send()
        .await
        .expect("request");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("json");
    assert_eq!(body["object"], "response");

    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn test_responses_streaming() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), "gpt-4o", "openai/gpt-4o");
    let handle = start_oxllm(config).await.expect("server start");

    let sse_body = concat!(
        "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
        "data: [DONE]\n\n"
    );

    let _m = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions");
        then.status(200)
            .header("content-type", "text/event-stream")
            .body(sse_body);
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/responses"))
        .json(&json!({
        "model": "gpt-4o",
        "input": "Hi",
        "stream": true,
        }))
        .send()
        .await
        .expect("request");

    assert_eq!(resp.status(), 200);
    let text = resp.text().await.expect("text");

    assert!(
        text.contains("response.created"),
        "missing response.created: {text}"
    );
    assert!(
        text.contains("response.output_text.delta"),
        "missing response.output_text.delta: {text}"
    );
    assert!(text.contains("Hi"), "missing Hi delta: {text}");
    assert!(
        text.contains("response.completed"),
        "missing response.completed: {text}"
    );

    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn test_responses_missing_model_returns_400() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), "gpt-4o", "openai/gpt-4o");
    let handle = start_oxllm(config).await.expect("server start");

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/responses"))
        .json(&json!({"input": "Hi"}))
        .send()
        .await
        .expect("request");

    assert_eq!(resp.status(), 400);
    let _ = handle.shutdown_tx.send(());
}

// ===========================================================================
// Auth on new endpoints
// ===========================================================================

#[tokio::test]
async fn test_messages_endpoint_requires_auth_when_master_key_set() {
    let mock = MockServer::start();
    let mut config = make_proxy_config(&mock.base_url(), "claude", "openai/gpt-4o");
    config.general_settings.master_key = Some("sk-master-secret".to_string());
    let handle = start_oxllm(config).await.expect("server start");

    let client = reqwest::Client::new();

    let resp_no_auth = client
        .post(handle.url("/v1/messages"))
        .json(&json!({"model": "claude", "messages": [], "max_tokens": 10}))
        .send()
        .await
        .expect("request");
    assert_eq!(resp_no_auth.status(), 401);

    let resp_bad_auth = client
        .post(handle.url("/v1/messages"))
        .header("authorization", "Bearer sk-wrong")
        .json(&json!({"model": "claude", "messages": [], "max_tokens": 10}))
        .send()
        .await
        .expect("request");
    assert_eq!(resp_bad_auth.status(), 401);

    let _m = mock.mock(|when, then| {
 when.method(httpmock::Method::POST).path("/chat/completions");
 then.status(200)
 .header("content-type", "application/json")
 .json_body(json!({
 "id": "x", "object": "chat.completion", "model": "gpt-4o",
 "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
 "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
 }));
 });

    let resp_good_auth = client
 .post(handle.url("/v1/messages"))
 .header("authorization", "Bearer sk-master-secret")
 .json(&json!({"model": "claude", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10}))
 .send()
 .await
 .expect("request");
    assert_eq!(resp_good_auth.status(), 200);

    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn test_responses_endpoint_requires_auth_when_master_key_set() {
    let mock = MockServer::start();
    let mut config = make_proxy_config(&mock.base_url(), "gpt-4o", "openai/gpt-4o");
    config.general_settings.master_key = Some("sk-master-secret".to_string());
    let handle = start_oxllm(config).await.expect("server start");

    let client = reqwest::Client::new();

    let resp_no_auth = client
        .post(handle.url("/v1/responses"))
        .json(&json!({"model": "gpt-4o", "input": "Hi"}))
        .send()
        .await
        .expect("request");
    assert_eq!(resp_no_auth.status(), 401);

    let resp_bad_auth = client
        .post(handle.url("/v1/responses"))
        .header("authorization", "Bearer wrong")
        .json(&json!({"model": "gpt-4o", "input": "Hi"}))
        .send()
        .await
        .expect("request");
    assert_eq!(resp_bad_auth.status(), 401);

    let _ = handle.shutdown_tx.send(());
}

// ===========================================================================
// Cross-provider: Anthropic input routed to Anthropic provider
// ===========================================================================

#[tokio::test]
async fn test_messages_anthropic_input_to_anthropic_provider() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), "claude", "anthropic/claude-3-5-sonnet");
    let handle = start_oxllm(config).await.expect("server start");

    let _m = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/messages")
            .header("x-api-key", "sk-test-key");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
            "id": "msg_abc",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-5-sonnet",
            "content": [{"type": "text", "text": "Bonjour!"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 3},
            }));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/messages"))
        .json(&json!({
        "model": "claude",
        "system": "Be brief.",
        "messages": [{"role": "user", "content": "Say hi in French"}],
        "max_tokens": 100,
        }))
        .send()
        .await
        .expect("request");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("json");
    assert_eq!(body["type"], "message");
    assert_eq!(body["content"][0]["type"], "text");
    assert_eq!(body["content"][0]["text"], "Bonjour!");
    assert_eq!(body["stop_reason"], "end_turn");
    assert_eq!(body["usage"]["input_tokens"], 10);
    assert_eq!(body["usage"]["output_tokens"], 3);

    let _ = handle.shutdown_tx.send(());
}

// ===========================================================================
// C18: orchestrator transparency headers survive the Anthropic conversion
// ===========================================================================

/// The non-streaming `/v1/messages` conversion must carry the
/// `x-oxllm-response-cost`, `x-oxllm-request-id`, `x-request-id`,
/// `x-oxllm-router-strategy`, `x-oxllm-router-decision`, and at least one
/// `x-ratelimit-*` header onto the Anthropic-shaped response. Previously
/// both conversion paths built a brand-new `Response` and dropped every
/// header the orchestrator stamped (while `/v1/chat/completions` preserved
/// them all).
#[tokio::test]
async fn messages_non_streaming_preserves_transparency_headers() {
    let mock = MockServer::start();
    let mut config = make_proxy_config(&mock.base_url(), "claude", "openai/gpt-4o");
    // Set a global RPM limit so the orchestrator stamps `x-ratelimit-*`
    // headers; without a limit configured no ratelimit header is emitted and
    // the preservation assertion would be vacuous.
    config.oxllm_settings.rpm = Some(100);
    let handle = start_oxllm(config).await.expect("server start");

    let _m = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "model": "gpt-4o",
            "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Hello!"},
            "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}
            }));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/messages"))
        .json(&json!({
        "model": "claude",
        "messages": [{"role": "user", "content": "Say hi"}],
        "max_tokens": 100,
        }))
        .send()
        .await
        .expect("request");

    assert_eq!(resp.status(), 200);
    let headers = resp.headers();
    assert!(
        headers.contains_key("x-oxllm-response-cost"),
        "x-oxllm-response-cost must be preserved: {headers:?}"
    );
    assert!(
        headers.contains_key("x-oxllm-request-id"),
        "x-oxllm-request-id must be preserved: {headers:?}"
    );
    assert!(
        headers.contains_key("x-request-id"),
        "x-request-id must be preserved: {headers:?}"
    );
    assert_eq!(
        headers
            .get("x-oxllm-router-strategy")
            .and_then(|v| v.to_str().ok()),
        Some("simple_shuffle"),
        "x-oxllm-router-strategy must be preserved: {headers:?}"
    );
    assert!(
        headers.contains_key("x-oxllm-router-decision"),
        "x-oxllm-router-decision must be preserved: {headers:?}"
    );
    assert!(
        headers.contains_key("x-ratelimit-remaining-requests"),
        "an x-ratelimit-* header must be preserved: {headers:?}"
    );

    let _ = handle.shutdown_tx.send(());
}

/// The streaming `/v1/messages` conversion must carry the same header
/// families as the non-streaming path so an Anthropic SDK client observes
/// routing / throttling transparency regardless of `stream`. Note:
/// `x-oxllm-response-cost` is computed in the async post-success hook after
/// the stream completes, so it is not stamped on the streaming response
/// headers (same as `/v1/chat/completions` streaming); the unit test
/// `convert_json_preserves_transparency_headers` covers the cost-header
/// preservation directly.
#[tokio::test]
async fn messages_streaming_preserves_transparency_headers() {
    let mock = MockServer::start();
    let mut config = make_proxy_config(&mock.base_url(), "claude", "openai/gpt-4o");
    config.oxllm_settings.rpm = Some(100);
    let handle = start_oxllm(config).await.expect("server start");

    let sse_body = concat!(
        "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":2,\"total_tokens\":7}}\n\n",
        "data: [DONE]\n\n"
    );

    let _m = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions");
        then.status(200)
            .header("content-type", "text/event-stream")
            .body(sse_body);
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/messages"))
        .json(&json!({
        "model": "claude",
        "messages": [{"role": "user", "content": "Say hi"}],
        "max_tokens": 100,
        "stream": true,
        }))
        .send()
        .await
        .expect("request");

    assert_eq!(resp.status(), 200);
    let headers = resp.headers();
    assert!(
        headers.contains_key("x-oxllm-request-id"),
        "x-oxllm-request-id must be preserved on stream: {headers:?}"
    );
    assert!(
        headers.contains_key("x-request-id"),
        "x-request-id must be preserved on stream: {headers:?}"
    );
    assert_eq!(
        headers
            .get("x-oxllm-router-strategy")
            .and_then(|v| v.to_str().ok()),
        Some("simple_shuffle"),
        "x-oxllm-router-strategy must be preserved on stream: {headers:?}"
    );
    assert!(
        headers.contains_key("x-ratelimit-remaining-requests"),
        "an x-ratelimit-* header must be preserved on stream: {headers:?}"
    );

    let _ = handle.shutdown_tx.send(());
}

/// `/v1/messages` surfaces a 429 from the upstream OpenAI pipeline as an
/// Anthropic-shaped error with `error.type = "rate_limit_error"` so an
/// Anthropic SDK client applies its rate-limit retry behaviour. Previously
/// every upstream status was mapped to the generic `"api_error"`.
#[tokio::test]
async fn messages_upstream_rate_limit_surfaces_as_rate_limit_error_type() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), "claude", "openai/gpt-4o");
    let handle = start_oxllm(config).await.expect("server start");

    let _m = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions");
        then.status(429).json_body(json!({
            "error": {"message": "Rate limit reached", "type": "rate_limit_exceeded"}
        }));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/messages"))
        .json(&json!({
        "model": "claude",
        "messages": [{"role": "user", "content": "Say hi"}],
        "max_tokens": 100,
        }))
        .send()
        .await
        .expect("request");

    assert_eq!(resp.status(), 429);
    let body: Value = resp.json().await.expect("json");
    assert_eq!(body["type"], "error", "error envelope: {body}");
    assert_eq!(
        body["error"]["type"], "rate_limit_error",
        "Anthropic error.type must reflect the upstream status: {body}"
    );
    assert_eq!(body["error"]["message"], "Rate limit reached");

    let _ = handle.shutdown_tx.send(());
}

/// A 401 from the upstream must surface as `authentication_error` so the
/// Anthropic SDK surfaces it to the user rather than retrying.
#[tokio::test]
async fn messages_upstream_unauthorized_surfaces_as_authentication_error_type() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), "claude", "openai/gpt-4o");
    let handle = start_oxllm(config).await.expect("server start");

    let _m = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions");
        then.status(401).json_body(json!({
            "error": {"message": "Invalid API key", "type": "invalid_request_error"}
        }));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/messages"))
        .json(&json!({
        "model": "claude",
        "messages": [{"role": "user", "content": "Say hi"}],
        "max_tokens": 100,
        }))
        .send()
        .await
        .expect("request");

    assert_eq!(resp.status(), 401);
    let body: Value = resp.json().await.expect("json");
    assert_eq!(
        body["error"]["type"], "authentication_error",
        "401 must map to authentication_error: {body}"
    );

    let _ = handle.shutdown_tx.send(());
}
