//! Integration tests: `/v1/embeddings` and legacy `/v1/completions`.

use httpmock::MockServer;
use oxllm_server::OxllmParams;
use oxllm_server::SharedState;
use oxllm_server::app::{build_router, build_shared_state};
use oxllm_server::config::types::{
    ConfigModel, GeneralSettings, OxllmSettings, ProxyConfig, RoutingKnobs,
};
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
    let (handle, _shared) = start_oxllm_with_state(config).await?;
    Ok(handle)
}

async fn start_oxllm_with_state(config: ProxyConfig) -> anyhow::Result<(OxllmHandle, SharedState)> {
    let shared = build_shared_state(config)?;

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;
    let router = build_router(shared.clone());

    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    tokio::spawn(async move {
        axum::serve(listener, router)
            .with_graceful_shutdown(async move {
                let _ = shutdown_rx.await;
            })
            .await
            .ok();
    });

    Ok((OxllmHandle { addr, shutdown_tx }, shared))
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
async fn test_embeddings_endpoint_openai() {
    let mock = MockServer::start();
    let config = make_proxy_config(
        &mock.base_url(),
        "text-embedding-3-small",
        "openai/text-embedding-3-small",
    );
    let handle = start_oxllm(config).await.expect("server start");

    let _m = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/embeddings")
            .header("authorization", "Bearer sk-test-key")
            .json_body_partial(json!({"input": "hello"}).to_string());
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}],
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 2, "total_tokens": 2}
            }));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/embeddings"))
        .json(&json!({
        "model": "text-embedding-3-small",
        "input": "hello",
        }))
        .send()
        .await
        .expect("request");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("json body");
    assert_eq!(body["object"], "list");
    assert_eq!(body["data"][0]["embedding"][0], 0.1);

    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn test_embeddings_missing_model_returns_400() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), "emb", "openai/emb");
    let handle = start_oxllm(config).await.expect("server start");

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/embeddings"))
        .json(&json!({"input": "hi"}))
        .send()
        .await
        .expect("request");

    assert_eq!(resp.status(), 400);
    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn test_embeddings_unknown_model_returns_404() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), "known", "openai/known");
    let handle = start_oxllm(config).await.expect("server start");

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/embeddings"))
        .json(&json!({"model": "does-not-exist", "input": "x"}))
        .send()
        .await
        .expect("request");

    assert_eq!(resp.status(), 404);
    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn test_completions_endpoint_forwarded_as_chat() {
    let mock = MockServer::start();
    let config = make_proxy_config(
        &mock.base_url(),
        "gpt-3.5-turbo-instruct",
        "openai/gpt-3.5-turbo-instruct",
    );
    let handle = start_oxllm(config).await.expect("server start");

    let _m = mock.mock(|when, then| {
 when.method(httpmock::Method::POST)
 .path("/chat/completions");
 then.status(200).header("content-type", "application/json").json_body(json!({
 "id": "chatcmpl-1",
 "object": "chat.completion",
 "model": "gpt-3.5-turbo-instruct",
 "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
 "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
 }));
 });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/completions"))
        .json(&json!({
        "model": "gpt-3.5-turbo-instruct",
        "prompt": "Hello",
        "max_tokens": 5,
        }))
        .send()
        .await
        .expect("request");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("json");
    assert_eq!(body["object"], "text_completion");
    assert_eq!(body["choices"][0]["text"], "ok");
    assert!(body["choices"][0].get("message").is_none());

    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn test_completions_streaming_returns_text_completion_chunks() {
    let mock = MockServer::start();
    let config = make_proxy_config(
        &mock.base_url(),
        "gpt-3.5-turbo-instruct",
        "openai/gpt-3.5-turbo-instruct",
    );
    let handle = start_oxllm(config).await.expect("server start");

    let sse_body = concat!(
        "data: {\"id\":\"cmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-3.5-turbo-instruct\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"cmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-3.5-turbo-instruct\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"cmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-3.5-turbo-instruct\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"!\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"cmpl-1\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"gpt-3.5-turbo-instruct\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":2,\"total_tokens\":4}}\n\n",
        "data: [DONE]\n\n"
    );
    let upstream_mock = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .json_body_partial(json!({"stream": true}).to_string());
        then.status(200)
            .header("content-type", "text/event-stream")
            .body(sse_body);
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/completions"))
        .json(&json!({
        "model": "gpt-3.5-turbo-instruct",
        "prompt": "Hello",
        "stream": true,
        }))
        .send()
        .await
        .expect("request");

    assert_eq!(resp.status(), 200);
    assert_eq!(
        resp.headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok()),
        Some("text/event-stream"),
        "streaming completions must return an SSE content type"
    );
    assert!(
        resp.headers()
            .get("x-oxllm-request-id")
            .and_then(|v| v.to_str().ok())
            .is_some(),
        "x-oxllm-request-id header must be preserved on the streaming response"
    );

    let text = resp.text().await.expect("body text");

    assert!(
        text.contains("\"object\":\"text_completion\""),
        "chunks must be rewritten to text_completion: {text}"
    );
    assert!(
        !text.contains("chat.completion.chunk"),
        "no chat.completion.chunk should leak through: {text}"
    );
    assert!(
        text.contains("\"text\":\"Hi\""),
        "first delta content must become text: {text}"
    );
    assert!(
        text.contains("\"text\":\"!\""),
        "second delta content must become text: {text}"
    );
    assert!(
        text.contains("\"finish_reason\":\"stop\""),
        "finish_reason must be preserved: {text}"
    );
    assert!(
        text.contains("\"logprobs\":null"),
        "logprobs must be defaulted to null: {text}"
    );
    assert!(
        text.contains("\"prompt_tokens\":2"),
        "usage chunk must pass through: {text}"
    );
    assert!(text.contains("[DONE]"), "missing [DONE] sentinel: {text}");
    assert!(
        !text.contains("\"delta\""),
        "no delta field should remain on choices: {text}"
    );

    upstream_mock.assert_hits(1);
    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn test_completions_streaming_does_not_fallback_after_first_byte() {
    // Stream safety invariant: once the orchestrator returns a streaming
    // response (200 OK + first byte), any later failure must be surfaced as
    // a truncated stream — never retried against a fallback. We model this by
    // configuring a fallback chain, having the primary emit a valid 200 SSE
    // stream, and asserting the fallback deployment is never hit.
    fn dep(name: &str, base: &str) -> ConfigModel {
        ConfigModel {
            model_name: name.to_string(),
            oxllm_params: OxllmParams {
                model: format!("openai/{name}"),
                api_key: Some("sk-test-key".to_string()),
                api_base: Some(base.to_string()),
                timeout: Some(30.0),
                stream_timeout: Some(30),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    let mock = MockServer::start();
    let config = ProxyConfig {
        model_list: vec![
            dep("primary", &mock.base_url()),
            dep("fb1", &mock.base_url()),
        ],
        oxllm_settings: OxllmSettings {
            routing: RoutingKnobs {
                fallbacks: Some(json!([{"primary": ["fb1"]}])),
                num_retries: Some(0),
                ..Default::default()
            },
            ..Default::default()
        },
        general_settings: GeneralSettings {
            allow_private_api_base: Some(true),
            ..Default::default()
        },
        router_settings: Default::default(),
        extra: serde_json::Map::new(),
    };
    let handle = start_oxllm(config).await.expect("server start");

    let primary_sse = concat!(
        "data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\n",
        "data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"!\"},\"finish_reason\":\"stop\"}]}\n\n",
        "data: [DONE]\n\n"
    );
    let primary_mock = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .json_body_partial(json!({"model": "primary", "stream": true}).to_string());
        then.status(200)
            .header("content-type", "text/event-stream")
            .body(primary_sse);
    });
    let fb_mock = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .json_body_partial(json!({"model": "fb1"}).to_string());
        then.status(200).json_body(json!({"id": "fb"}));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/completions"))
        .json(&json!({
        "model": "primary",
        "prompt": "hi",
        "stream": true,
        }))
        .send()
        .await
        .expect("request");

    assert_eq!(resp.status(), 200);
    let text = resp.text().await.expect("body");
    assert!(
        text.contains("\"object\":\"text_completion\""),
        "streamed chunks must be text_completion: {text}"
    );
    assert!(
        text.contains("\"text\":\"hi\""),
        "primary's content must reach the client: {text}"
    );

    primary_mock.assert_hits(1);
    assert_eq!(
        fb_mock.hits(),
        0,
        "fallback must not be hit once the primary stream has committed (no fallback after first byte)"
    );

    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn test_anthropic_non_streaming_chat() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), "claude", "anthropic/claude-3-5-sonnet");
    let handle = start_oxllm(config).await.expect("server start");

    let _m = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/messages")
            .header("x-api-key", "sk-test-key")
            .header("anthropic-version", "2023-06-01");
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
        .post(handle.url("/v1/chat/completions"))
        .json(&json!({
        "model": "claude",
        "messages": [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Say hi in French"},
        ],
        "max_tokens": 100,
        }))
        .send()
        .await
        .expect("request");

    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("json");
    assert_eq!(body["object"], "chat.completion");
    assert_eq!(body["id"], "msg_abc");
    assert_eq!(body["model"], "claude-3-5-sonnet");
    assert_eq!(body["choices"][0]["message"]["role"], "assistant");
    assert_eq!(body["choices"][0]["message"]["content"], "Bonjour!");
    assert_eq!(body["choices"][0]["finish_reason"], "stop");
    assert_eq!(body["usage"]["prompt_tokens"], 10);
    assert_eq!(body["usage"]["completion_tokens"], 3);
    assert_eq!(body["usage"]["total_tokens"], 13);

    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn test_anthropic_request_is_transformed_to_messages_shape() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), "claude", "anthropic/claude-3-5-sonnet");
    let handle = start_oxllm(config).await.expect("server start");

    let m = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/messages")
            .json_body_partial(
                json!({
                "model": "claude-3-5-sonnet",
                "system": "Be brief.",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 50,
                })
                .to_string(),
            );
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
            "id": "msg", "model": "claude", "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn", "usage": {"input_tokens": 1, "output_tokens": 1},
            }));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .json(&json!({
        "model": "claude",
        "messages": [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hi"},
        ],
        "max_tokens": 50,
        }))
        .send()
        .await
        .expect("request");

    assert_eq!(resp.status(), 200);
    m.assert_hits(1);
    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn test_anthropic_streaming_chat() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), "claude", "anthropic/claude-3-5-sonnet");
    let handle = start_oxllm(config).await.expect("server start");

    let sse_body = concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude-3-5-sonnet\",\"role\":\"assistant\",\"content\":[],\"stop_reason\":null,\"usage\":{\"input_tokens\":3,\"output_tokens\":0}}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi\"}}\n\n",
        "event: message_delta\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":1}}\n\n",
        "event: message_stop\n",
        "data: {\"type\":\"message_stop\"}\n\n"
    );

    let _m = mock.mock(|when, then| {
        when.method(httpmock::Method::POST).path("/messages");
        then.status(200)
            .header("content-type", "text/event-stream")
            .body(sse_body);
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .json(&json!({
        "model": "claude",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 5,
        "stream": true,
        }))
        .send()
        .await
        .expect("request");

    assert_eq!(resp.status(), 200);
    let text = resp.text().await.expect("text");
    assert!(
        text.contains("chat.completion.chunk"),
        "missing chunk obj: {text}"
    );
    assert!(
        text.contains("\"content\":\"Hi\""),
        "missing Hi delta: {text}"
    );
    assert!(
        text.contains("\"finish_reason\":\"stop\""),
        "missing finish: {text}"
    );

    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn test_drop_params_request_succeeds_with_unsupported_keys_for_anthropic() {
    let mock = MockServer::start();
    let mut config = make_proxy_config(&mock.base_url(), "claude", "anthropic/claude-3-5-sonnet");
    config.oxllm_settings.drop_params = Some(true);
    let handle = start_oxllm(config).await.expect("server start");

    let m = mock.mock(|when, then| {
        when.method(httpmock::Method::POST).path("/messages");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
            "id": "msg", "model": "claude", "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn", "usage": {"input_tokens": 1, "output_tokens": 1},
            }));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .json(&json!({
        "model": "claude",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 5,
        "frequency_penalty": 0.5,
        "logprobs": true,
        }))
        .send()
        .await
        .expect("request");

    assert_eq!(resp.status(), 200);
    m.assert_hits(1);
    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn test_unknown_provider_slug_returns_400() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), "weird", "madeup/weird-model");
    let handle = start_oxllm(config).await.expect("server start");

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .json(&json!({
        "model": "weird",
        "messages": [{"role": "user", "content": "Hi"}],
        }))
        .send()
        .await
        .expect("request");

    assert_eq!(resp.status(), 400);
    let body: Value = resp.json().await.expect("json");
    assert!(
        body["error"]["message"]
            .as_str()
            .expect("message")
            .contains("unsupported provider")
    );

    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn test_embeddings_alias_route() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), "emb", "openai/emb");
    let handle = start_oxllm(config).await.expect("server start");

    let _m = mock.mock(|when, then| {
        when.method(httpmock::Method::POST).path("/embeddings");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.5]}],
            "model": "emb",
            "usage": {"prompt_tokens": 1, "total_tokens": 1},
            }));
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/embeddings"))
        .json(&json!({"model": "emb", "input": "x"}))
        .send()
        .await
        .expect("request");

    assert_eq!(resp.status(), 200);
    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn test_completions_alias_route() {
    let mock = MockServer::start();
    let config = make_proxy_config(&mock.base_url(), "gpt", "openai/gpt");
    let handle = start_oxllm(config).await.expect("server start");

    let _m = mock.mock(|when, then| {
 when.method(httpmock::Method::POST).path("/chat/completions");
 then.status(200).header("content-type", "application/json").json_body(json!({
 "id": "x", "object": "chat.completion", "model": "gpt",
 "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
 "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
 }));
 });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/completions"))
        .json(&json!({"model": "gpt", "prompt": "hi"}))
        .send()
        .await
        .expect("request");

    assert_eq!(resp.status(), 200);
    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn test_embeddings_triggers_rate_limit_after_n_requests() {
    let mock = MockServer::start();
    let mut config = make_proxy_config(
        &mock.base_url(),
        "text-embedding-3-small",
        "openai/text-embedding-3-small",
    );
    config.oxllm_settings.rpm = Some(2);
    let handle = start_oxllm(config).await.expect("server start");

    let _m = mock.mock(|when, then| {
        when.method(httpmock::Method::POST).path("/embeddings");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.1]}],
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 2, "total_tokens": 2}
            }));
    });

    let client = reqwest::Client::new();
    let body = json!({"model": "text-embedding-3-small", "input": "hello"});

    let resp1 = client
        .post(handle.url("/v1/embeddings"))
        .json(&body)
        .send()
        .await
        .expect("first request");
    assert_eq!(resp1.status(), 200);

    let resp2 = client
        .post(handle.url("/v1/embeddings"))
        .json(&body)
        .send()
        .await
        .expect("second request");
    assert_eq!(resp2.status(), 200);

    let resp3 = client
        .post(handle.url("/v1/embeddings"))
        .json(&body)
        .send()
        .await
        .expect("third request should be rate limited");
    assert_eq!(
        resp3.status(),
        429,
        "embeddings must respect the RPM limit after exceeding it"
    );

    let _ = handle.shutdown_tx.send(());
}

// ---------------------------------------------------------------------------
// Bug F7: embeddings must retry across deployments, cool failed ones, and
// map upstream timeouts to 504 (matching the chat path).
// ---------------------------------------------------------------------------

/// Build a config with two deployments serving the same model name, each
/// pointing at its own upstream base URL.
fn two_deployment_config(model_name: &str, base_a: &str, base_b: &str) -> ProxyConfig {
    let entry = |base: &str, key: &str| ConfigModel {
        model_name: model_name.to_string(),
        oxllm_params: OxllmParams {
            model: format!("openai/{model_name}"),
            api_key: Some(key.to_string()),
            api_base: Some(base.to_string()),
            timeout: Some(30.0),
            ..Default::default()
        },
        ..Default::default()
    };
    ProxyConfig {
        model_list: vec![entry(base_a, "sk-key-a"), entry(base_b, "sk-key-b")],
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
async fn test_embeddings_fails_over_to_second_deployment() {
    let mock_fail = MockServer::start();
    let fail_upstream = mock_fail.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/embeddings")
            .header("authorization", "Bearer sk-key-a");
        then.status(500)
            .header("content-type", "application/json")
            .json_body(json!({"error": {"message": "boom"}}));
    });
    let mock_ok = MockServer::start();
    let ok_upstream = mock_ok.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/embeddings")
            .header("authorization", "Bearer sk-key-b");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.9]}],
            "model": "emb-failover",
            "usage": {"prompt_tokens": 1, "total_tokens": 1}
            }));
    });

    let config = two_deployment_config("emb-failover", &mock_fail.base_url(), &mock_ok.base_url());
    let (handle, _shared) = start_oxllm_with_state(config).await.expect("server start");

    let resp = reqwest::Client::new()
        .post(handle.url("/v1/embeddings"))
        .json(&json!({"model": "emb-failover", "input": "hello"}))
        .send()
        .await
        .expect("request");
    assert_eq!(
        resp.status(),
        200,
        "a failed deployment must fail over to the healthy twin"
    );
    let body: Value = resp.json().await.expect("json");
    assert_eq!(body["data"][0]["embedding"][0], 0.9);

    // Exactly one request lands on the healthy upstream. When shuffle drew
    // the failing twin first, it is retried up to `num_retries` (2) — 3
    // upstream hits — before its cooldown moves traffic to the twin; when
    // the healthy twin was drawn first it is never hit. (The previous
    // `<= 1` assertion contradicted the retry policy and made this test a
    // coin flip on which deployment shuffle picked.)
    assert_eq!(ok_upstream.hits(), 1);
    assert!(
        matches!(fail_upstream.hits(), 0 | 3),
        "failing upstream must be either never picked (0) or retried to \
         exhaustion then cooled (num_retries 2 + 1 = 3 hits), got {}",
        fail_upstream.hits()
    );

    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn test_embeddings_marks_failed_deployment_cooling() {
    let mock = MockServer::start();
    mock.mock(|when, then| {
        when.method(httpmock::Method::POST).path("/embeddings");
        then.status(500)
            .header("content-type", "application/json")
            .json_body(json!({"error": {"message": "boom"}}));
    });

    let config = make_proxy_config(&mock.base_url(), "emb-cool", "openai/emb-cool");
    let (handle, shared) = start_oxllm_with_state(config).await.expect("server start");

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/embeddings"))
        .json(&json!({"model": "emb-cool", "input": "hello"}))
        .send()
        .await
        .expect("request");
    assert_eq!(resp.status(), 500, "the upstream 500 must surface");

    let state = shared.load_full();
    let cooldown_key = state.router.cooldown_key(&state.deployments[0]);
    assert!(
        state.router.cooldown_store().is_cooling(&cooldown_key),
        "failed embeddings deployment must be marked cooling"
    );

    // While cooling, the router skips the only deployment, so the next
    // request finds no candidate. The model exists but all its deployments
    // are unavailable → 429 (not 404, which means "model doesn't exist").
    let resp2 = client
        .post(handle.url("/v1/embeddings"))
        .json(&json!({"model": "emb-cool", "input": "hello"}))
        .send()
        .await
        .expect("second request");
    assert_eq!(
        resp2.status(),
        429,
        "a cooled deployment must be skipped by the router; 429 = model exists but unavailable"
    );

    let _ = handle.shutdown_tx.send(());
}

#[tokio::test]
async fn test_embeddings_timeout_returns_504() {
    let mock = MockServer::start();
    mock.mock(|when, then| {
        when.method(httpmock::Method::POST).path("/embeddings");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.1]}],
            "model": "emb-slow",
            "usage": {"prompt_tokens": 1, "total_tokens": 1}
            }))
            .delay(std::time::Duration::from_secs(5));
    });

    let mut config = make_proxy_config(&mock.base_url(), "emb-slow", "openai/emb-slow");
    config.model_list[0].oxllm_params.timeout = Some(0.2);
    let handle = start_oxllm(config).await.expect("server start");

    let resp = reqwest::Client::new()
        .post(handle.url("/v1/embeddings"))
        .json(&json!({"model": "emb-slow", "input": "hello"}))
        .send()
        .await
        .expect("request");
    assert_eq!(
        resp.status(),
        504,
        "upstream timeout must map to 504 (gateway timeout), matching chat"
    );

    let _ = handle.shutdown_tx.send(());
}

// ---------------------------------------------------------------------------
// C20: the embeddings upstream error-body read must stream into a bounded
// buffer (MAX_ERROR_BODY_BYTES) rather than buffering the whole payload via
// `response.bytes()` first — otherwise a malicious upstream streaming a
// multi-GB error body pins it all in heap before the size check fires.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_embeddings_rejects_oversized_upstream_error_body() {
    let mock = MockServer::start();
    let mut config = make_proxy_config(&mock.base_url(), "emb-big-err", "openai/emb-big-err");
    // Fail fast: do not retry the bounded-read failure across attempts.
    config.oxllm_settings.routing.num_retries = Some(0);
    let handle = start_oxllm(config).await.expect("server start");

    let oversized_error_body = "x".repeat(oxllm_server::security::MAX_ERROR_BODY_BYTES * 4);
    mock.mock(|when, then| {
        when.method(httpmock::Method::POST).path("/embeddings");
        then.status(500)
            .header("content-type", "application/json")
            .body(oversized_error_body);
    });

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/embeddings"))
        .json(&json!({"model": "emb-big-err", "input": "hello"}))
        .send()
        .await
        .expect("request");

    // The streaming bounded read aborts at MAX_ERROR_BODY_BYTES and surfaces a
    // bounded 5xx instead of hanging or OOMing on the oversized error payload.
    assert!(
        resp.status().is_server_error(),
        "oversized upstream error body must yield a bounded server error, got {}",
        resp.status()
    );
    let body: Value = resp.json().await.expect("json body");
    let msg = body["error"]["message"]
        .as_str()
        .expect("error message")
        .to_string();
    assert!(
        msg.contains("exceeds") && msg.contains("limit"),
        "error should report the body-size limit, got: {msg}"
    );

    let _ = handle.shutdown_tx.send(());
}

// ---------------------------------------------------------------------------
// C20: a cache hit must stamp `x-oxllm-response-cost` for consistency with
// the cache-miss path (previously the header was dropped on cache hits).
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_embeddings_cache_hit_stamps_cost_header() {
    let mock = MockServer::start();
    let config = make_proxy_config(
        &mock.base_url(),
        "text-embedding-3-small",
        "openai/text-embedding-3-small",
    );
    let handle = start_oxllm(config).await.expect("server start");

    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST).path("/embeddings");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 2, "total_tokens": 2}
            }));
    });

    let client = reqwest::Client::new();
    // `cache: {"insert": true}` opts the request into the response cache so
    // the second identical call short-circuits in pre_call.
    let body = json!({
    "model": "text-embedding-3-small",
    "input": "hello",
    "cache": {"insert": true}
    });

    // First request: cache miss, hits upstream, stores the response.
    let resp1 = client
        .post(handle.url("/v1/embeddings"))
        .json(&body)
        .send()
        .await
        .expect("first request");
    assert_eq!(resp1.status(), 200);
    assert!(
        resp1.headers().get("x-oxllm-response-cost").is_some(),
        "cache miss must stamp the cost header"
    );
    assert!(
        resp1.headers().get("x-oxllm-cache-hit").is_none(),
        "cache miss must NOT carry the cache-hit header"
    );
    let _ = resp1.text().await;

    // Second identical request: cache hit (the upstream mock must NOT be hit
    // again). The cost header must still be present, matching the miss path.
    let resp2 = client
        .post(handle.url("/v1/embeddings"))
        .json(&body)
        .send()
        .await
        .expect("second request");
    assert_eq!(resp2.status(), 200);
    assert!(
        resp2.headers().get("x-oxllm-response-cost").is_some(),
        "cache hit must stamp the cost header for consistency with cache misses"
    );
    assert_eq!(
        resp2
            .headers()
            .get("x-oxllm-cache-hit")
            .and_then(|v| v.to_str().ok()),
        Some("true"),
        "cache hit must stamp x-oxllm-cache-hit: true (parity with chat)"
    );

    upstream.assert_hits(1);
    let _ = handle.shutdown_tx.send(());
}

// ---------------------------------------------------------------------------
// C20: an embeddings miss must stamp the `x-ratelimit-*` header set (parity
// with the chat handler, which clients/observability rely on for backoff).
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_embeddings_miss_stamps_ratelimit_headers() {
    let mock = MockServer::start();
    let config = ProxyConfig {
        model_list: vec![ConfigModel {
            model_name: "text-embedding-3-small".to_string(),
            oxllm_params: OxllmParams {
                model: "openai/text-embedding-3-small".to_string(),
                api_key: Some("sk-test-key".to_string()),
                api_base: Some(mock.base_url()),
                timeout: Some(30.0),
                stream_timeout: Some(30),
                ..Default::default()
            },
            ..Default::default()
        }],
        oxllm_settings: OxllmSettings {
            rpm: Some(10),
            ..Default::default()
        },
        general_settings: GeneralSettings {
            allow_private_api_base: Some(true),
            ..Default::default()
        },
        router_settings: Default::default(),
        extra: serde_json::Map::new(),
    };
    let handle = start_oxllm(config).await.expect("server start");

    mock.mock(|when, then| {
        when.method(httpmock::Method::POST).path("/embeddings");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 2, "total_tokens": 2}
            }));
    });

    let client = reqwest::Client::new();
    let body = json!({
    "model": "text-embedding-3-small",
    "input": "hello"
    });

    let resp = client
        .post(handle.url("/v1/embeddings"))
        .json(&body)
        .send()
        .await
        .expect("request");
    assert_eq!(resp.status(), 200);
    assert_eq!(
        resp.headers()
            .get("x-ratelimit-limit-requests")
            .and_then(|v| v.to_str().ok()),
        Some("10"),
        "miss must stamp the RPM limit header (parity with chat)"
    );
    let remaining = resp
        .headers()
        .get("x-ratelimit-remaining-requests")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u32>().ok())
        .expect("x-ratelimit-remaining-requests present");
    assert!(
        remaining < 10,
        "remaining must reflect the post-reservation count, got {remaining}"
    );

    let _ = handle.shutdown_tx.send(());
}
