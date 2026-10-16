//! Integration tests: semantic caching end-to-end through the proxy.
//!
//! Verifies that an exact-match MISS followed by a similar request yields a
//! semantic-cache HIT, with the upstream chat completions endpoint hit only
//! once.

use httpmock::MockServer;
use oxllm_server::OxllmParams;
use oxllm_server::app::{build_router, build_shared_state};
use oxllm_server::config::types::{
    CacheConfig, ConfigModel, GeneralSettings, OxllmSettings, ProxyConfig, SemanticCacheConfig,
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

fn config_with_semantic_cache(
    chat_base_url: &str,
    embed_base_url: &str,
    threshold: f64,
) -> ProxyConfig {
    ProxyConfig {
        model_list: vec![
            ConfigModel {
                model_name: "gpt-4o".to_string(),
                oxllm_params: OxllmParams {
                    model: "openai/gpt-4o".to_string(),
                    api_key: Some("sk-chat".to_string()),
                    api_base: Some(chat_base_url.to_string()),
                    timeout: Some(30.0),
                    stream_timeout: Some(30),
                    ..Default::default()
                },
                ..Default::default()
            },
            ConfigModel {
                model_name: "text-embedding-3-small".to_string(),
                oxllm_params: OxllmParams {
                    model: "openai/text-embedding-3-small".to_string(),
                    api_key: Some("sk-embed".to_string()),
                    api_base: Some(embed_base_url.to_string()),
                    timeout: Some(30.0),
                    stream_timeout: Some(30),
                    ..Default::default()
                },
                ..Default::default()
            },
        ],
        oxllm_settings: OxllmSettings {
            cache: Some(CacheConfig {
                cache_type: Some("memory".to_string()),
                host: None,
                port: None,
                semantic_cache: Some(SemanticCacheConfig {
                    enabled: true,
                    embedding_model: "text-embedding-3-small".to_string(),
                    similarity_threshold: Some(threshold),
                    max_entries: Some(100),
                    ttl_secs: Some(3600),
                    api_base: None,
                    api_key: None,
                    ..Default::default()
                }),
                ..Default::default()
            }),
            ..Default::default()
        },
        general_settings: GeneralSettings {
            allow_private_api_base: Some(true),
            ..Default::default()
        },
        router_settings: Default::default(),
        extra: serde_json::Map::new(),
    }
}

#[tokio::test]
async fn semantic_hits_keep_tool_and_output_schema_scopes_exact() {
    for tools in [false, true] {
        let chat = MockServer::start_async().await;
        let embeddings = MockServer::start_async().await;
        let chat_hit=chat.mock_async(|when,then| {
            when.method(httpmock::Method::POST).path("/chat/completions");
            then.status(200).json_body(json!({"id":"scoped","model":"gpt-4o","choices":[{"index":0,"message":{"role":"assistant","content":"{\"answer\":\"ok\"}"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":4,"total_tokens":9}}));
        }).await;
        let embedding_hit=embeddings.mock_async(|when,then| {
            when.method(httpmock::Method::POST).path("/embeddings");
            then.status(200).json_body(json!({"object":"list","model":"text-embedding-3-small","data":[{"object":"embedding","index":0,"embedding":[0.12,0.45,0.78]}],"usage":{"prompt_tokens":5,"total_tokens":5}}));
        }).await;
        let handle = start_oxllm(config_with_semantic_cache(
            &chat.base_url(),
            &embeddings.base_url(),
            0.95,
        ))
        .await
        .unwrap();
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .unwrap();
        for (index, (text, other, cached)) in [
            (
                "Please extract the answer for the first record.",
                "one",
                false,
            ),
            (
                "Please extract the answer for the second record.",
                "two",
                false,
            ),
            (
                "Please extract the answer for the first record again.",
                "one",
                true,
            ),
            (
                "Please extract the answer for the second record again.",
                "two",
                true,
            ),
        ]
        .into_iter()
        .enumerate()
        {
            let schema = json!({"type":"object","properties":{"answer":{"type":"string","enum":["ok",other]}},"required":["answer"],"additionalProperties":false});
            let mut payload = json!({"model":"gpt-4o","messages":[{"role":"user","content":text}],"cache":{"insert":true}});
            if tools {
                payload["tools"] = json!([{"type":"function","function":{"name":"extract","strict":true,"parameters":schema}}]);
                payload["tool_choice"] = json!("auto");
            } else {
                payload["response_format"] = json!({"type":"json_schema","json_schema":{"name":"answer","strict":true,"schema":schema}});
            }
            let response = client
                .post(handle.url("/v1/chat/completions"))
                .json(&payload)
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), 200, "tools={tools} request={index}");
            assert_eq!(
                response.headers().contains_key("x-oxllm-cache-hit"),
                cached,
                "tools={tools} request={index}"
            );
            let body: Value = response.json().await.unwrap();
            assert_eq!(
                body["choices"][0]["message"]["content"],
                "{\"answer\":\"ok\"}"
            );
        }
        chat_hit.assert_hits_async(2).await;
        assert!(
            embedding_hit.hits_async().await >= 4,
            "semantic matching must actually fetch embeddings"
        );
        let _ = handle.shutdown_tx.send(());
    }
}

/// Two chat requests that look different but should produce identical
/// embeddings (the mock returns the same vector for any input) → the second
/// must short-circuit via the semantic cache.
#[tokio::test]
async fn semantic_cache_serves_second_similar_request() {
    let chat_mock = MockServer::start();
    let embed_mock = MockServer::start();

    let chat_hit = chat_mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .header("authorization", "Bearer sk-chat");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
            "id": "chatcmpl-semantic-1",
            "object": "chat.completion",
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Paris"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11}
            }));
    });

    // Every embedding request returns the SAME vector — any two requests will
    // have cosine similarity 1.0 and exceed the 0.95 threshold.
    let _embed_hit = embed_mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/embeddings")
            .header("authorization", "Bearer sk-embed");
        then.status(200).json_body(json!({
        "object": "list",
        "data": [{"object": "embedding", "index": 0, "embedding": [0.12, 0.45, 0.78]}],
        "model": "text-embedding-3-small",
        "usage": {"prompt_tokens": 5, "total_tokens": 5}
        }));
    });

    let config = config_with_semantic_cache(&chat_mock.base_url(), &embed_mock.base_url(), 0.95);
    let handle = start_oxllm(config).await.expect("server start");

    let client = reqwest::Client::new();

    // First request: cache miss → upstream called → response cached.
    let resp1 = client
        .post(handle.url("/v1/chat/completions"))
        .json(&json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "cache": {"insert": true}
        }))
        .send()
        .await
        .expect("first request");
    assert_eq!(resp1.status(), 200);
    let body1: Value = resp1.json().await.expect("body");
    assert_eq!(body1["choices"][0]["message"]["content"], "Paris");

    // Second, semantically similar request: exact cache MISS but semantic
    // cache HIT → upstream NOT called again.
    let resp2 = client
        .post(handle.url("/v1/chat/completions"))
        .json(&json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "What's the capital of France?"}],
        "cache": {"insert": true}
        }))
        .send()
        .await
        .expect("second request");
    assert_eq!(resp2.status(), 200);
    let body2: Value = resp2.json().await.expect("body");
    assert_eq!(
        body2["choices"][0]["message"]["content"], "Paris",
        "second request should return the cached response"
    );

    assert_eq!(
        chat_hit.hits(),
        1,
        "upstream chat endpoint should be hit exactly once"
    );

    let _ = handle.shutdown_tx.send(());
}

/// When the embedding fetch fails, the request must fall through to the
/// upstream without blocking or erroring.
#[tokio::test]
async fn semantic_cache_embedding_failure_falls_through() {
    let chat_mock = MockServer::start();
    let embed_mock = MockServer::start();

    let chat_hit = chat_mock.mock(|when, then| {
        when.method(httpmock::Method::POST).path("/chat/completions");
        then.status(200).json_body(json!({
        "id": "ok",
        "object": "chat.completion",
        "model": "gpt-4o",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        }));
    });
    let _embed_fail = embed_mock.mock(|when, then| {
        when.method(httpmock::Method::POST).path("/embeddings");
        then.status(500);
    });

    let config = config_with_semantic_cache(&chat_mock.base_url(), &embed_mock.base_url(), 0.95);
    let handle = start_oxllm(config).await.expect("server start");

    let resp = reqwest::Client::new()
        .post(handle.url("/v1/chat/completions"))
        .json(&json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
        "cache": {"insert": true}
        }))
        .send()
        .await
        .expect("request");
    assert_eq!(resp.status(), 200);
    assert_eq!(chat_hit.hits(), 1, "upstream must still be called");

    let _ = handle.shutdown_tx.send(());
}

/// Two different requests with orthogonal embeddings (below threshold) must
/// both hit the upstream — no false-positive semantic match.
#[tokio::test]
async fn semantic_cache_does_not_match_below_threshold() {
    let chat_mock = MockServer::start();
    let embed_mock = MockServer::start();

    let chat_hit = chat_mock.mock(|when, then| {
        when.method(httpmock::Method::POST).path("/chat/completions");
        then.status(200).json_body(json!({
        "id": "unique",
        "object": "chat.completion",
        "model": "gpt-4o",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        }));
    });

    // Return orthogonal embeddings for different inputs. httpmock can't vary
    // by body easily here, so instead set a threshold of 1.0 — only identical
    // vectors (cosine = 1.0) match. Two requests with different stashed
    // embeddings (the mock returns the same one, but the *stored* and *query*
    // vectors are identical) would still match... so we instead use distinct
    // responses by querying different request bodies and asserting the
    // upstream is hit twice because each stores a different entry that doesn't
    // match the next query. The simplest reliable setup: vary the embedding
    // by `input` using httpmock's body matcher.
    let embed_a = embed_mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/embeddings")
            .json_body_partial(json!({"input": "alpha question"}).to_string());
        then.status(200).json_body(json!({
        "data": [{"embedding": [1.0, 0.0]}]
        }));
    });
    let embed_b = embed_mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/embeddings")
            .json_body_partial(json!({"input": "beta question"}).to_string());
        then.status(200).json_body(json!({
        "data": [{"embedding": [0.0, 1.0]}]
        }));
    });

    let config = config_with_semantic_cache(&chat_mock.base_url(), &embed_mock.base_url(), 0.95);
    let handle = start_oxllm(config).await.expect("server start");
    let client = reqwest::Client::new();

    for content in ["alpha question", "beta question"] {
        let resp = client
            .post(handle.url("/v1/chat/completions"))
            .json(&json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": content}],
            "cache": {"insert": true}
            }))
            .send()
            .await
            .expect("request");
        assert_eq!(resp.status(), 200);
    }

    assert_eq!(
        chat_hit.hits(),
        2,
        "two orthogonal requests must not share a semantic cache entry"
    );

    // The two-upstream-hits assertion above only proves the semantic layer did
    // not serve a cached entry if the embedding mocks actually fired. If these
    // partial matchers stopped matching (e.g. the forwarded body shape
    // changed), every embedding fetch would 404 and the semantic layer would
    // fall through — the upstream would be hit twice for the wrong reason and
    // the test would pass vacuously. Guard against that.
    assert!(
        embed_a.hits() >= 1,
        "the below-threshold path must provably exercise the alpha embedding mock \
         (otherwise the semantic layer never ran)"
    );
    assert!(
        embed_b.hits() >= 1,
        "the below-threshold path must provably exercise the beta embedding mock \
         (otherwise the semantic layer never ran)"
    );

    let _ = handle.shutdown_tx.send(());
}
