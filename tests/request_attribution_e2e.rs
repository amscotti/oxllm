//! HTTP correlation and durable attribution across translated and native calls.

mod common;

use std::sync::Arc;
use std::time::Duration;

use httpmock::{Method::POST, MockServer};
use oxllm_server::app::{build_router, build_shared_state_with_db};
use oxllm_server::config::types::ProxyConfig;
use oxllm_server::{Database, SharedState, SpendLog, spend_writer};
use serde_json::{Value, json};

const MASTER_KEY: &str = "sk-synthetic-attribution-master";

struct Proxy {
    handle: common::OxllmHandle,
    shared: SharedState,
    db: Arc<Database>,
    server: tokio::task::JoinHandle<std::io::Result<()>>,
    writer: spend_writer::SpendWriterHandle,
}

impl Proxy {
    fn url(&self, path: &str) -> String {
        self.handle.url(path)
    }

    async fn shutdown(self) {
        self.handle.shutdown();
        tokio::time::timeout(Duration::from_secs(5), self.server)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        self.writer.shutdown().await;
    }
}

fn config(upstream: &MockServer) -> ProxyConfig {
    serde_json::from_value(json!({
        "model_list":[{"model_name":"chat", "oxllm_params":{
            "model":"openai/gpt-4o", "api_key":"synthetic-upstream-key", "api_base":upstream.base_url()
        }}],
        "general_settings":{"allow_private_api_base":true, "master_key":MASTER_KEY},
        "router_settings":{"num_retries":0}
    })).unwrap()
}

fn zero_rates(params: &mut oxllm_core::OxllmParams) {
    params.input_cost_per_token = Some(0.0);
    params.output_cost_per_token = Some(0.0);
    params.cache_read_input_token_cost = Some(0.0);
    params.cache_creation_input_token_cost = Some(0.0);
    params.cache_creation_input_token_cost_above_1hr = Some(0.0);
    params.output_cost_per_reasoning_token = Some(0.0);
}

async fn start(upstream: &MockServer) -> Proxy {
    start_with_config(config(upstream)).await
}

async fn start_with_config(config: ProxyConfig) -> Proxy {
    let db = Database::connect("sqlite::memory:").await.unwrap();
    db.migrate().await.unwrap();
    let (writer, writer_handle) = spend_writer::spawn_with_shutdown(db.clone(), 60, 100);
    let shared =
        build_shared_state_with_db(config, Some(writer.sender()), Some(db.clone())).unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let router = build_router(shared.clone());
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
    let server = tokio::spawn(async move {
        axum::serve(listener, router)
            .with_graceful_shutdown(async move {
                let _ = shutdown_rx.await;
            })
            .await
    });
    Proxy {
        handle: common::OxllmHandle { addr, shutdown_tx },
        shared,
        db,
        server,
        writer: writer_handle,
    }
}

fn reply() -> Value {
    json!({"id":"chatcmpl-attribution", "model":"gpt-4o", "choices":[{
        "index":0, "message":{"role":"assistant", "content":"ok"}, "finish_reason":"stop"
    }], "usage":{"prompt_tokens":5, "completion_tokens":1, "total_tokens":6}})
}

fn body(path: &str, input: &str) -> Value {
    match path {
        "/v1/responses" => json!({"model":"chat", "input":input, "max_output_tokens":5}),
        "/v1/completions" => json!({"model":"chat", "prompt":input, "max_tokens":5}),
        "/v1/embeddings" | "/openai/v1/embeddings" => json!({"model":"chat", "input":input}),
        "/v1/rerank" => json!({"model":"chat", "query":input, "documents":["document"]}),
        "/v1/audio/speech" => json!({"model":"chat", "input":input, "voice":"synthetic"}),
        _ => json!({"model":"chat", "messages":[{"role":"user", "content":input}], "max_tokens":5}),
    }
}

async fn key(proxy: &Proxy, policy: Value) -> String {
    let response = common::http_client()
        .post(proxy.url("/key/generate"))
        .bearer_auth(MASTER_KEY)
        .json(&policy)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200, "{}", response.text().await.unwrap());
    let value: Value = response.json().await.unwrap();
    value["key"].as_str().unwrap().to_string()
}

fn request_id(response: &reqwest::Response) -> String {
    correlated_request_id(response.headers(), response.url().path())
}

fn correlated_request_id(headers: &axum::http::HeaderMap, path: &str) -> String {
    let id = headers
        .get("x-oxllm-request-id")
        .unwrap_or_else(|| panic!("missing x-oxllm-request-id on {path}"))
        .to_str()
        .unwrap_or_else(|_| panic!("non-text x-oxllm-request-id on {path}"));
    assert!(!id.is_empty(), "blank correlation id on {path}");
    for alias in ["x-request-id", "x-litellm-call-id"] {
        assert_eq!(
            headers.get(alias).and_then(|value| value.to_str().ok()),
            Some(id),
            "{alias} mismatch on {path}"
        );
    }
    assert_ne!(
        id, "client-supplied-id",
        "client must not own the id on {path}"
    );
    id.to_string()
}

async fn row(db: &Database, request_id: &str) -> SpendLog {
    tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            if let Some(row) = db.query_spend_by_request_id(request_id).await.unwrap() {
                return row;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("attributed ledger entry")
}

#[tokio::test]
async fn session_and_call_ids_survive_translations_native_calls_and_header_precedence() {
    let upstream = MockServer::start_async().await;
    let wire = upstream
        .mock_async(|when, then| {
            when.method(POST);
            then.status(200).json_body(reply());
        })
        .await;
    let proxy = start(&upstream).await;
    let secret = key(&proxy, json!({"models":["chat"], "max_budget":1.0})).await;
    let client = common::http_client();
    let unauthorized = client
        .post(proxy.url("/v1/chat/completions"))
        .bearer_auth("synthetic-invalid-key")
        .header("x-litellm-session-id", "unauthorized-session")
        .json(&body("/v1/chat/completions", "hello"))
        .send()
        .await
        .unwrap();
    assert_eq!(unauthorized.status(), 401);
    request_id(&unauthorized);
    let mut ids = std::collections::HashSet::new();
    for (path, native, legacy, expected) in [
        (
            "/v1/chat/completions",
            None,
            Some(" legacy-session "),
            Some("legacy-session"),
        ),
        ("/v1/messages", None, None, Some("body-session")),
        (
            "/v1/responses",
            Some("  "),
            Some("shared-session"),
            Some("shared-session"),
        ),
        (
            "/v1/completions",
            Some(" native-session "),
            Some("ignored-session"),
            Some("native-session"),
        ),
        ("/v1/embeddings", None, None, None),
        (
            "/v1/embeddings",
            None,
            Some("embedding-session"),
            Some("embedding-session"),
        ),
        (
            "/openai/v1/embeddings",
            None,
            Some("raw-embedding-session"),
            Some("raw-embedding-session"),
        ),
        (
            "/v1/rerank",
            None,
            Some("rerank-session"),
            Some("rerank-session"),
        ),
        (
            "/v1/audio/speech",
            None,
            Some("speech-session"),
            Some("speech-session"),
        ),
        (
            "/openai/v1/chat/completions",
            Some("native-session"),
            Some("ignored-session"),
            Some("native-session"),
        ),
    ] {
        let mut payload = body(path, "hello");
        if path == "/v1/messages" {
            payload["metadata"] = json!({"user_id":{"session_id":"body-session"}});
        }
        let mut request = client
            .post(proxy.url(path))
            .bearer_auth(&secret)
            .header("x-litellm-call-id", "client-supplied-id")
            .json(&payload);
        if let Some(value) = native {
            request = request.header("x-oxllm-session-id", value);
        }
        if let Some(value) = legacy {
            request = request.header("x-litellm-session-id", value);
        }
        let response = request.send().await.unwrap();
        assert_eq!(
            response.status(),
            200,
            "{path}: {}",
            response.text().await.unwrap()
        );
        let id = request_id(&response);
        assert!(ids.insert(id.clone()));
        let recorded = row(&proxy.db, &id).await;
        assert_eq!(recorded.session_id.as_deref(), expected, "{path}");
        assert_eq!(recorded.status.as_deref(), Some("success"));
        let info = client
            .get(proxy.url(&format!("/request/logs/{id}")))
            .bearer_auth(MASTER_KEY)
            .send()
            .await
            .unwrap()
            .json::<Value>()
            .await
            .unwrap();
        assert_eq!(info.get("session_id").and_then(Value::as_str), expected);
    }
    for path in ["/spend/logs", "/spend/logs/v2"] {
        let result: Value = client
            .get(proxy.url(path))
            .bearer_auth(MASTER_KEY)
            .query(&[("session_id", "native-session"), ("limit", "1")])
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(
            result["total_requests"], 2,
            "session totals describe the full scope"
        );
        assert_eq!(result["data"].as_array().unwrap().len(), 1);
        assert_eq!(result["data"][0]["session_id"], "native-session");
        assert!(result["total_cost"].as_f64().unwrap() > 0.0);
    }
    wire.assert_hits_async(10).await;
    proxy.shutdown().await;
}

#[tokio::test]
async fn bounded_sessions_keep_routing_and_ledger_identity_consistent() {
    let upstream = MockServer::start_async().await;
    let wire = upstream
        .mock_async(|when, then| {
            when.method(POST);
            then.status(200).json_body(reply());
        })
        .await;
    let mut config = config(&upstream);
    config.router_settings.routing.routing_strategy = Some("session_affinity".into());
    let mut other = config.model_list[0].clone();
    other.oxllm_params.api_key = Some("synthetic-other-key".into());
    config.model_list.push(other);
    let proxy = start_with_config(config).await;
    let secret = key(&proxy, json!({"models":["chat"]})).await;
    let long = (0..512u32)
        .map(|value| format!("{:08x}", value.wrapping_mul(0x9e37_79b9)))
        .collect::<String>();
    let expected = &long[..256];
    let mut serving = None;
    for (path, header) in [
        (
            "/v1/chat/completions",
            Some(("x-oxllm-session-id", long.as_str())),
        ),
        (
            "/v1/chat/completions",
            Some(("x-litellm-session-id", expected)),
        ),
        ("/v1/messages", None),
    ] {
        let mut payload = body(path, "hello");
        if header.is_none() {
            payload["metadata"] = json!({"user_id":{"session_id":long}});
        }
        let mut request = common::http_client()
            .post(proxy.url(path))
            .bearer_auth(&secret)
            .json(&payload);
        if let Some((name, value)) = header {
            request = request.header(name, value);
        }
        let response = request.send().await.unwrap();
        assert_eq!(response.status(), 200);
        let selected = response.headers()["x-oxllm-model-id"].clone();
        if let Some(previous) = &serving {
            assert_eq!(&selected, previous);
        }
        serving = Some(selected);
        let recorded = row(&proxy.db, &request_id(&response)).await;
        assert_eq!(recorded.session_id.as_deref(), Some(expected));
    }
    wire.assert_hits_async(3).await;
    proxy.shutdown().await;
}

#[tokio::test]
async fn oversized_end_user_ids_are_local_errors_without_customer_aliasing() {
    let upstream = MockServer::start_async().await;
    let wire = upstream
        .mock_async(|when, then| {
            when.method(POST);
            then.status(200).json_body(reply());
        })
        .await;
    let proxy = start(&upstream).await;
    let client = common::http_client();
    let prefix = "a".repeat(256);
    let oversized = format!("{prefix}x");
    let created = client
        .post(proxy.url("/customer/new"))
        .bearer_auth(MASTER_KEY)
        .json(&json!({"end_user_id":prefix, "max_budget":0.0}))
        .send()
        .await
        .unwrap();
    assert_eq!(created.status(), 200);
    let create_oversized = client
        .post(proxy.url("/customer/new"))
        .bearer_auth(MASTER_KEY)
        .json(&json!({"end_user_id":oversized}))
        .send()
        .await
        .unwrap();
    assert_eq!(
        create_oversized.status(),
        400,
        "oversized customer creation must fail before writing"
    );
    assert!(proxy.db.get_customer(&oversized).await.unwrap().is_none());
    let secret = key(&proxy, json!({"models":["chat"]})).await;
    for path in [
        "/v1/chat/completions",
        "/v1/messages",
        "/v1/responses",
        "/v1/completions",
        "/v1/embeddings",
        "/v1/rerank",
        "/v1/audio/speech",
        "/openai/v1/embeddings",
        "/v1/messages/count_tokens",
        "/anthropic/v1/messages/count_tokens",
    ] {
        let mut payload = body(path, "hello");
        payload["user"] = json!(oversized);
        let response = client
            .post(proxy.url(path))
            .bearer_auth(&secret)
            .json(&payload)
            .send()
            .await
            .unwrap();
        assert_eq!(
            response.status(),
            400,
            "{path}: {}",
            response.text().await.unwrap()
        );
        request_id(&response);
        let value: Value = response.json().await.unwrap();
        assert!(
            value["error"]["message"]
                .as_str()
                .unwrap()
                .contains("256 bytes"),
            "{path}: {value}"
        );
    }
    let mut payload = body("/v1/chat/completions", "hello");
    payload["user"] = json!(prefix);
    let rejected = client
        .post(proxy.url("/v1/chat/completions"))
        .bearer_auth(&secret)
        .json(&payload)
        .send()
        .await
        .unwrap();
    assert_eq!(
        rejected.status(),
        429,
        "the exact 256-byte customer still receives its policy"
    );
    wire.assert_hits_async(0).await;
    let customer = proxy.db.get_customer(&prefix).await.unwrap().unwrap();
    assert_eq!(customer.spend, 0.0);
    proxy.shutdown().await;
}

#[tokio::test]
async fn denied_and_invalid_requests_keep_session_correlation_without_duplicate_rows() {
    let upstream = MockServer::start_async().await;
    let failed = upstream
        .mock_async(|when, then| {
            when.method(POST)
                .path("/chat/completions")
                .body_contains("rejected-upstream");
            then.status(400)
                .json_body(json!({"error":{"message":"invalid request"}}));
        })
        .await;
    let proxy = start(&upstream).await;
    let denied = key(&proxy, json!({"models":["other"]})).await;
    let capped = key(&proxy, json!({"models":["chat"], "max_budget":0.0})).await;
    let ordinary = key(&proxy, json!({"models":["chat"]})).await;
    let mut malformed = body("/v1/responses", "hello");
    malformed["text"] = json!({"format":{"type":"json_schema", "name":"answer"}});
    let client = common::http_client();
    for (path, secret, payload, status) in [
        (
            "/v1/chat/completions",
            &denied,
            body("/v1/chat/completions", "hello"),
            403,
        ),
        (
            "/v1/chat/completions",
            &capped,
            body("/v1/chat/completions", "hello"),
            429,
        ),
        (
            "/v1/chat/completions",
            &ordinary,
            body("/v1/chat/completions", "rejected-upstream"),
            400,
        ),
        ("/v1/responses", &ordinary, malformed, 400),
    ] {
        let response = client
            .post(proxy.url(path))
            .bearer_auth(secret)
            .header("x-litellm-session-id", "failure-session")
            .json(&payload)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), status);
        let id = request_id(&response);
        let recorded = row(&proxy.db, &id).await;
        assert_eq!(recorded.session_id.as_deref(), Some("failure-session"));
        assert_eq!(recorded.status.as_deref(), Some("failure"));
        assert_eq!(recorded.cost, Some(0.0));
    }
    failed.assert_hits_async(1).await;
    let state = proxy.shared.load_full();
    assert!(
        !state
            .router
            .cooldown_store()
            .is_cooling(&state.router.cooldown_key(&state.deployments[0]))
    );
    let db = proxy.db.clone();
    proxy.shutdown().await;
    let filter = oxllm_server::SpendFilter {
        session_id: Some("failure-session".into()),
        ..Default::default()
    };
    assert_eq!(db.count_spend_filtered(&filter).await.unwrap(), 4);
}

#[tokio::test]
async fn cache_hits_get_fresh_call_ids_and_preserve_cumulative_spend_aliases() {
    let upstream = MockServer::start_async().await;
    let wire = upstream
        .mock_async(|when, then| {
            when.method(POST).path("/chat/completions");
            then.status(200).json_body(reply());
        })
        .await;
    let proxy = start(&upstream).await;
    let secret = key(&proxy, json!({"models":["chat"], "max_budget":1.0})).await;
    let mut payload = body("/v1/chat/completions", "cached");
    payload["cache"] = json!({"insert":true});
    let mut ids = Vec::new();
    let mut spends = Vec::new();
    for session in ["cache-first", "cache-second"] {
        let response = common::http_client()
            .post(proxy.url("/v1/chat/completions"))
            .bearer_auth(&secret)
            .header("x-litellm-session-id", session)
            .json(&payload)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        ids.push(request_id(&response));
        let spend = response.headers()["x-oxllm-key-spend"].clone();
        assert_eq!(response.headers()["x-litellm-key-spend"], spend);
        spends.push(spend);
        if session == "cache-second" {
            assert_eq!(response.headers()["x-oxllm-cache-hit"], "true");
        }
        let _: Value = response.json().await.unwrap();
    }
    assert_ne!(ids[0], ids[1]);
    assert_eq!(spends[0], spends[1]);
    wire.assert_hits_async(1).await;
    let db = proxy.db.clone();
    proxy.shutdown().await;
    let rows = db.query_spend(&Default::default()).await.unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].request_id.as_deref(), Some(ids[0].as_str()));
    assert_eq!(rows[0].session_id.as_deref(), Some("cache-first"));
}

#[tokio::test]
async fn completed_and_partial_streams_keep_session_identity_across_bridges() {
    let upstream = MockServer::start_async().await;
    let first = json!({"id":"chatcmpl-stream", "model":"gpt-4o", "choices":[{
        "index":0, "delta":{"content":"output"}, "finish_reason":null
    }], "usage":{"prompt_tokens":5, "completion_tokens":1, "total_tokens":6}});
    let complete = upstream
        .mock_async(|when, then| {
            when.method(POST)
                .path("/chat/completions")
                .body_contains("complete");
            then.status(200)
                .header("content-type", "text/event-stream")
                .body(format!(
                    "data: {first}\n\ndata: {}\n\ndata: [DONE]\n\n",
                    json!({"choices":[{"index":0, "delta":{}, "finish_reason":"stop"}],
                "usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}})
                ));
        })
        .await;
    let partial = upstream
        .mock_async(|when, then| {
            when.method(POST)
                .path("/chat/completions")
                .body_contains("partial");
            then.status(200)
                .header("content-type", "text/event-stream")
                .body(format!(
                    "data: {first}\n\ndata: {}\n\n",
                    json!({"error":{"message":"upstream error", "type":"api_error", "code":500}})
                ));
        })
        .await;
    let proxy = start(&upstream).await;
    let secret = key(&proxy, json!({"models":["chat"], "max_budget":1.0})).await;
    for (path, interrupted) in [
        ("/v1/chat/completions", false),
        ("/v1/messages", false),
        ("/v1/responses", false),
        ("/v1/chat/completions", true),
    ] {
        let mut payload = body(path, if interrupted { "partial" } else { "complete" });
        payload["stream"] = json!(true);
        payload["stream_options"] = json!({"include_usage":true});
        let response = common::http_client()
            .post(proxy.url(path))
            .bearer_auth(&secret)
            .header("x-litellm-session-id", "stream-session")
            .json(&payload)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let id = request_id(&response);
        assert!(response.headers().get("x-oxllm-key-spend").is_none());
        assert!(response.headers().get("x-litellm-key-spend").is_none());
        let text = response.text().await.unwrap();
        assert!(text.contains("output"));
        if interrupted {
            assert!(text.contains("error"));
        }
        let recorded = row(&proxy.db, &id).await;
        assert_eq!(recorded.session_id.as_deref(), Some("stream-session"));
        assert_eq!(
            recorded.status.as_deref(),
            Some(if interrupted { "partial" } else { "success" })
        );
        assert!(recorded.cost.unwrap() > 0.0);
    }
    complete.assert_hits_async(3).await;
    partial.assert_hits_async(1).await;
    let db = proxy.db.clone();
    proxy.shutdown().await;
    let filter = oxllm_server::SpendFilter {
        session_id: Some("stream-session".into()),
        ..Default::default()
    };
    assert_eq!(db.count_spend_filtered(&filter).await.unwrap(), 4);
}

#[tokio::test]
async fn missing_inference_models_are_invalid_parameters_while_resources_remain_not_found() {
    let upstream = MockServer::start_async().await;
    let wire = upstream
        .mock_async(|when, then| {
            when.method(POST);
            then.status(200).json_body(reply());
        })
        .await;
    let proxy = start(&upstream).await;
    let client = common::http_client();
    for path in [
        "/v1/chat/completions",
        "/v1/messages",
        "/v1/responses",
        "/v1/completions",
        "/v1/embeddings",
        "/openai/v1/embeddings",
        "/v1/messages/count_tokens",
        "/v1/rerank",
        "/openai/v1/chat/completions",
        "/anthropic/v1/messages/count_tokens",
    ] {
        let mut payload = if path == "/v1/rerank" {
            json!({"query":"hello", "documents":["document"]})
        } else {
            body(path, "hello")
        };
        payload["model"] = json!("missing-model");
        let response = client
            .post(proxy.url(path))
            .bearer_auth(MASTER_KEY)
            .json(&payload)
            .send()
            .await
            .unwrap();
        assert_eq!(
            response.status(),
            400,
            "{path}: {}",
            response.text().await.unwrap()
        );
        request_id(&response);
        let value: Value = response.json().await.unwrap();
        if path == "/v1/messages" || path == "/v1/messages/count_tokens" {
            assert_eq!(value["type"], "error");
            assert_eq!(value["error"]["type"], "invalid_request_error");
            assert!(
                value["error"].get("code").is_none(),
                "Anthropic-native routes keep the Anthropic envelope"
            );
        } else {
            assert_eq!(value["error"]["code"], "model_not_found");
        }
    }
    let response = client
        .get(proxy.url("/model/info"))
        .bearer_auth(MASTER_KEY)
        .query(&[("model", "missing-model")])
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 404);
    wire.assert_hits_async(0).await;
    proxy.shutdown().await;
}

#[tokio::test]
async fn oversized_unknown_model_names_keep_their_failure_row_without_poisoning_the_next_call() {
    let upstream = MockServer::start_async().await;
    let wire = upstream
        .mock_async(|when, then| {
            when.method(POST);
            then.status(200).json_body(reply());
        })
        .await;
    let proxy = start(&upstream).await;
    let mut payload = body("/v1/chat/completions", "hello");
    let long = (0..512u32)
        .map(|value| format!("{:08x}", value.wrapping_mul(0x9e37_79b9)))
        .collect::<String>();
    payload["model"] = json!(long);
    let failed = common::http_client()
        .post(proxy.url("/v1/chat/completions"))
        .bearer_auth(MASTER_KEY)
        .json(&payload)
        .send()
        .await
        .unwrap();
    assert_eq!(failed.status(), 400);
    let recorded = row(&proxy.db, &request_id(&failed)).await;
    assert_eq!(recorded.status.as_deref(), Some("failure"));
    assert_eq!(recorded.cost, Some(0.0));
    assert_eq!(recorded.requested_model.as_deref(), Some(long.as_str()));
    assert!(recorded.model.len() <= 512 && recorded.model.contains("...#sha256:"));
    let healthy = common::http_client()
        .post(proxy.url("/v1/chat/completions"))
        .bearer_auth(MASTER_KEY)
        .json(&body("/v1/chat/completions", "hello"))
        .send()
        .await
        .unwrap();
    assert_eq!(healthy.status(), 200);
    assert_eq!(
        row(&proxy.db, &request_id(&healthy))
            .await
            .status
            .as_deref(),
        Some("success")
    );
    wire.assert_hits_async(1).await;
    let db = proxy.db.clone();
    proxy.shutdown().await;
    assert_eq!(db.count_spend(None, None).await.unwrap(), 2);
}

#[tokio::test]
async fn missing_model_default_fallbacks_and_genuine_upstream_not_found_keep_their_semantics() {
    for path in ["/v1/chat/completions", "/v1/embeddings"] {
        let upstream = MockServer::start_async().await;
        let rescued = upstream
            .mock_async(|when, then| {
                when.method(POST).body_contains("rescued");
                then.status(200).json_body(reply());
            })
            .await;
        let missing_upstream = upstream.mock_async(|when, then| {
            when.method(POST).body_contains("upstream-missing");
            then.status(404).json_body(json!({"error":{"message":"provider model not found", "code":"model_not_found"}}));
        }).await;
        let mut config = config(&upstream);
        config.router_settings.routing.default_fallbacks = Some(json!(["chat"]));
        let proxy = start_with_config(config).await;
        let mut payload = body(path, "rescued");
        payload["model"] = json!("missing-model");
        let response = common::http_client()
            .post(proxy.url(path))
            .bearer_auth(MASTER_KEY)
            .json(&payload)
            .send()
            .await
            .unwrap();
        assert_eq!(
            response.status(),
            200,
            "{path}: {}",
            response.text().await.unwrap()
        );
        assert_eq!(response.headers()["x-oxllm-attempted-fallbacks"], "1");
        let response = common::http_client()
            .post(proxy.url(path))
            .bearer_auth(MASTER_KEY)
            .json(&body(path, "upstream-missing"))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 404);
        rescued.assert_hits_async(1).await;
        missing_upstream.assert_hits_async(1).await;
        proxy.shutdown().await;
    }
}

#[tokio::test]
async fn configured_unavailable_groups_are_retryable_across_raw_and_realtime_routes() {
    use tokio_tungstenite::tungstenite::client::IntoClientRequest;
    let upstream = MockServer::start_async().await;
    let wire = upstream
        .mock_async(|when, then| {
            when.method(POST);
            then.status(200).json_body(reply());
        })
        .await;
    let mut config = config(&upstream);
    let mut counting = config.model_list[0].clone();
    counting.model_name = "counting".into();
    counting.oxllm_params.model = "anthropic/claude-haiku-4-5".into();
    config.model_list.push(counting);
    let proxy = start_with_config(config).await;
    let state = proxy.shared.load_full();
    for deployment in &state.deployments {
        state.router.mark_cooldown_for(
            &state.router.cooldown_key(deployment),
            &deployment.model_name,
        );
    }
    for path in [
        "/v1/chat/completions",
        "/v1/messages",
        "/v1/responses",
        "/v1/completions",
        "/v1/embeddings",
        "/v1/messages/count_tokens",
        "/anthropic/v1/messages/count_tokens",
        "/v1/rerank",
        "/v1/audio/speech",
        "/openai/v1/chat/completions",
        "/openai/v1/embeddings",
    ] {
        let mut payload = body(path, "hello");
        let model = if path.ends_with("count_tokens") {
            "counting"
        } else {
            "chat"
        };
        payload["model"] = json!(model);
        let response = common::http_client()
            .post(proxy.url(path))
            .bearer_auth(MASTER_KEY)
            .json(&payload)
            .send()
            .await
            .unwrap();
        assert_eq!(
            response.status(),
            429,
            "{path}: {}",
            response.text().await.unwrap()
        );
        request_id(&response);
        assert_eq!(response.headers()["retry-after"], "60", "{path}");
        let value: Value = response.json().await.unwrap();
        assert_eq!(
            value["error"]["message"],
            format!("all deployments for '{model}' are rate-limited or unavailable"),
            "{path}"
        );
        if path == "/v1/messages" || path == "/v1/messages/count_tokens" {
            assert_eq!(value["type"], "error");
            assert_eq!(value["error"]["type"], "rate_limit_error");
        } else {
            assert_eq!(value["error"]["code"], "proxy_rate_limited");
        }
    }
    let mut request = proxy
        .url("/v1/realtime?model=chat")
        .replace("http://", "ws://")
        .into_client_request()
        .unwrap();
    request.headers_mut().insert(
        "authorization",
        format!("Bearer {MASTER_KEY}").parse().unwrap(),
    );
    let error = tokio_tungstenite::connect_async(request)
        .await
        .expect_err("cooling group cannot upgrade");
    let tokio_tungstenite::tungstenite::Error::Http(response) = error else {
        panic!("HTTP rejection expected: {error}");
    };
    assert_eq!(response.status().as_u16(), 429);
    correlated_request_id(response.headers(), "/v1/realtime");
    assert_eq!(response.headers()["retry-after"], "60");
    let value: Value = serde_json::from_slice(
        response
            .body()
            .as_deref()
            .expect("realtime HTTP rejection body"),
    )
    .unwrap();
    assert_eq!(value["error"]["code"], "proxy_rate_limited");
    assert_eq!(
        value["error"]["message"],
        "all deployments for 'chat' are rate-limited or unavailable"
    );
    wire.assert_hits_async(0).await;
    proxy.shutdown().await;
}

#[tokio::test]
async fn native_relays_cannot_forge_gateway_spend_snapshots() {
    for (status, streaming) in [(200, false), (400, false), (200, true)] {
        let upstream = MockServer::start_async().await;
        let wire = upstream
            .mock_async(|when, then| {
                when.method(POST).path("/v1/chat/completions");
                let then = then
                    .status(status)
                    .header("x-oxllm-key-spend", "999")
                    .header("x-litellm-key-spend", "888")
                    .header("x-oxllm-response-cost", "0.001");
                if streaming {
                    then.header("content-type", "text/event-stream")
                        .body("data: [DONE]\n\n");
                } else if status == 400 {
                    then.json_body(json!({"error":{"message":"invalid request"}}));
                } else {
                    then.json_body(reply());
                }
            })
            .await;
        let proxy = start(&upstream).await;
        let response = common::http_client().post(proxy.url("/openai/v1/chat/completions")).bearer_auth(MASTER_KEY)
            .json(&json!({"model":"chat", "messages":[{"role":"user", "content":"hello"}], "stream":streaming})).send().await.unwrap();
        assert_eq!(response.status(), status);
        assert_eq!(
            response.headers()["x-oxllm-response-cost"]
                .to_str()
                .unwrap()
                .parse::<f64>()
                .unwrap(),
            0.001,
            "billing receipts must survive snapshot stripping"
        );
        for header in ["x-oxllm-key-spend", "x-litellm-key-spend"] {
            assert!(
                response.headers().get(header).is_none(),
                "upstream owns no caller spend snapshot"
            );
        }
        response.bytes().await.unwrap();
        wire.assert_hits_async(1).await;
        proxy.shutdown().await;
    }
}

#[tokio::test]
async fn explicitly_free_groups_fit_positive_budgets_without_relaxing_mixed_groups() {
    for mixed in [false, true] {
        let upstream = MockServer::start_async().await;
        let wire = upstream
            .mock_async(|when, then| {
                when.method(POST);
                then.status(200).json_body(reply());
            })
            .await;
        let mut config = config(&upstream);
        zero_rates(&mut config.model_list[0].oxllm_params);
        if mixed {
            let mut paid = config.model_list[0].clone();
            paid.oxllm_params.api_key = Some("synthetic-paid-key".into());
            paid.oxllm_params.output_cost_per_token = Some(0.001);
            config.model_list.push(paid);
        }
        let proxy = start_with_config(config).await;
        let secret = key(&proxy, json!({"models":["chat"], "max_budget":0.00001})).await;
        let response = common::http_client()
            .post(proxy.url("/v1/chat/completions"))
            .bearer_auth(secret)
            .json(&body("/v1/chat/completions", "hello"))
            .send()
            .await
            .unwrap();
        assert_eq!(
            response.status(),
            if mixed { 429 } else { 200 },
            "mixed={mixed}: {}",
            response.text().await.unwrap()
        );
        if !mixed {
            assert_eq!(
                response.headers()["x-oxllm-response-cost"]
                    .to_str()
                    .unwrap()
                    .parse::<f64>()
                    .unwrap(),
                0.0
            );
            assert_eq!(response.headers()["x-oxllm-key-spend"], "0");
        }
        let recorded = row(&proxy.db, &request_id(&response)).await;
        assert_eq!(recorded.cost, Some(0.0));
        wire.assert_hits_async(if mixed { 0 } else { 1 }).await;
        proxy.shutdown().await;
    }
}

#[tokio::test]
async fn a_paid_fallback_must_top_up_a_free_groups_zero_reservation() {
    let upstream = MockServer::start_async().await;
    let primary = upstream
        .mock_async(|when, then| {
            when.method(POST)
                .header("authorization", "Bearer synthetic-upstream-key");
            then.status(503)
                .json_body(json!({"error":{"message":"unavailable"}}));
        })
        .await;
    let paid = upstream
        .mock_async(|when, then| {
            when.method(POST)
                .header("authorization", "Bearer synthetic-paid-key");
            then.status(200).json_body(reply());
        })
        .await;
    let mut config = config(&upstream);
    zero_rates(&mut config.model_list[0].oxllm_params);
    let mut fallback = config.model_list[0].clone();
    fallback.model_name = "paid".into();
    fallback.oxllm_params.api_key = Some("synthetic-paid-key".into());
    fallback.oxllm_params.input_cost_per_token = Some(0.001);
    fallback.oxllm_params.output_cost_per_token = Some(0.002);
    config.model_list.push(fallback);
    config.router_settings.routing.fallbacks = Some(json!([{"chat":["paid"]}]));
    let proxy = start_with_config(config).await;
    let secret = key(
        &proxy,
        json!({"models":["chat", "paid"], "max_budget":0.00001}),
    )
    .await;
    let response = common::http_client()
        .post(proxy.url("/v1/chat/completions"))
        .bearer_auth(secret)
        .json(&body("/v1/chat/completions", "hello"))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 429);
    let value: Value = response.json().await.unwrap();
    assert_eq!(value["error"]["code"], "budget_exceeded");
    primary.assert_hits_async(1).await;
    paid.assert_hits_async(0).await;
    proxy.shutdown().await;
}

#[tokio::test]
async fn mixed_members_and_paid_fallbacks_top_up_and_settle_actual_spend() {
    for cross_group in [false, true] {
        let upstream = MockServer::start_async().await;
        let first = upstream
            .mock_async(|when, then| {
                when.method(POST)
                    .header("authorization", "Bearer synthetic-upstream-key");
                then.status(503)
                    .json_body(json!({"error":{"message":"unavailable"}}));
            })
            .await;
        let paid = upstream
            .mock_async(|when, then| {
                when.method(POST)
                    .header("authorization", "Bearer synthetic-paid-key");
                then.status(200).json_body(reply());
            })
            .await;
        let mut config = config(&upstream);
        zero_rates(&mut config.model_list[0].oxllm_params);
        let mut next = config.model_list[0].clone();
        next.model_name = if cross_group { "paid" } else { "chat" }.into();
        next.oxllm_params.api_key = Some("synthetic-paid-key".into());
        next.oxllm_params.input_cost_per_token = Some(0.001);
        next.oxllm_params.output_cost_per_token = Some(0.002);
        config.model_list.push(next);
        config.router_settings.routing.routing_strategy = Some("round_robin".into());
        if cross_group {
            config.router_settings.routing.fallbacks = Some(json!([{"chat":["paid"]}]));
        }
        let proxy = start_with_config(config).await;
        let secret = key(&proxy, json!({"models":["chat", "paid"], "max_budget":1.0})).await;
        let counter = oxllm_server::hooks::spend_keys::spend_key(
            &oxllm_server::auth::hash_key_hex(secret.as_bytes()),
        );
        let state = proxy.shared.load_full();
        state.cache.incr(&counter, 0.125, None).await.unwrap();
        let response = common::http_client()
            .post(proxy.url("/v1/chat/completions"))
            .bearer_auth(secret)
            .json(&body("/v1/chat/completions", "hello"))
            .send()
            .await
            .unwrap();
        assert_eq!(
            response.status(),
            200,
            "cross_group={cross_group}: {}",
            response.text().await.unwrap()
        );
        let cost: f64 = response.headers()["x-oxllm-response-cost"]
            .to_str()
            .unwrap()
            .parse()
            .unwrap();
        let spend: f64 = response.headers()["x-oxllm-key-spend"]
            .to_str()
            .unwrap()
            .parse()
            .unwrap();
        assert!((cost - 0.007).abs() < 1e-12);
        assert!(
            (spend - 0.132).abs() < 1e-12,
            "no estimates or failed-attempt charges may remain"
        );
        assert!((state.cache.counter_value(&counter).await.unwrap() - spend).abs() < 1e-12);
        let recorded = row(&proxy.db, &request_id(&response)).await;
        assert_eq!(recorded.status.as_deref(), Some("success"));
        assert!((recorded.cost.unwrap() - cost).abs() < 1e-12);
        first.assert_hits_async(1).await;
        paid.assert_hits_async(1).await;
        let db = proxy.db.clone();
        proxy.shutdown().await;
        assert_eq!(db.count_spend(None, None).await.unwrap(), 1);
    }
}

#[tokio::test]
async fn positive_literal_tag_budgets_settle_and_survive_management_lifecycle() {
    let upstream = MockServer::start_async().await;
    let wire = upstream
        .mock_async(|when, then| {
            when.method(POST);
            then.status(200).json_body(reply());
        })
        .await;
    let proxy = start(&upstream).await;
    let client = common::http_client();
    let created = client
        .post(proxy.url("/tag/new"))
        .bearer_auth(MASTER_KEY)
        .json(&json!({"name":"product:demo", "max_budget":1.0}))
        .send()
        .await
        .unwrap();
    assert_eq!(created.status(), 200);
    let secret = key(&proxy, json!({"models":["chat"]})).await;
    let call = || {
        client
            .post(proxy.url("/v1/chat/completions"))
            .bearer_auth(&secret)
            .header("x-litellm-tags", "product:demo")
            .json(&body("/v1/chat/completions", "hello"))
    };
    let first = call().send().await.unwrap();
    assert_eq!(first.status(), 200);
    let cost: f64 = first.headers()["x-oxllm-response-cost"]
        .to_str()
        .unwrap()
        .parse()
        .unwrap();
    row(&proxy.db, &request_id(&first)).await;
    let tag = proxy.db.get_tag("product:demo").await.unwrap().unwrap();
    assert!((tag.spend.unwrap() - cost).abs() < 1e-12);
    let counter = proxy
        .shared
        .load_full()
        .cache
        .counter_value(&oxllm_server::hooks::spend_keys::tag_spend_key(
            "product:demo",
        ))
        .await
        .unwrap();
    assert!((counter - cost).abs() < 1e-12);
    let listed: Value = client
        .get(proxy.url("/tag/list"))
        .bearer_auth(MASTER_KEY)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(
        listed["tags"]
            .as_array()
            .unwrap()
            .iter()
            .any(|tag| tag["name"] == "product:demo")
    );
    let updated = client
        .post(proxy.url("/tag/update"))
        .bearer_auth(MASTER_KEY)
        .json(&json!({"name":"product:demo", "max_budget":cost, "description":"updated"}))
        .send()
        .await
        .unwrap();
    assert_eq!(updated.status(), 200);
    let rejected = call().send().await.unwrap();
    assert_eq!(rejected.status(), 429);
    row(&proxy.db, &request_id(&rejected)).await;
    wire.assert_hits_async(1).await;
    let deleted = client
        .post(proxy.url("/tag/delete"))
        .bearer_auth(MASTER_KEY)
        .json(&json!({"name":"product:demo"}))
        .send()
        .await
        .unwrap();
    assert_eq!(deleted.status(), 200);
    let restored = call().send().await.unwrap();
    assert_eq!(restored.status(), 200);
    row(&proxy.db, &request_id(&restored)).await;
    let logs: Value = client
        .get(proxy.url("/spend/logs"))
        .bearer_auth(MASTER_KEY)
        .query(&[("tag_name", "product:demo")])
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(logs["total_requests"], 3);
    assert!((logs["total_cost"].as_f64().unwrap() - 2.0 * cost).abs() < 1e-12);
    wire.assert_hits_async(2).await;
    proxy.shutdown().await;
}

#[tokio::test]
async fn literal_tag_pairs_remain_whole_in_required_and_excluded_routing_selectors() {
    let upstream = MockServer::start_async().await;
    let wire = upstream
        .mock_async(|when, then| {
            when.method(POST)
                .header("authorization", "Bearer synthetic-upstream-key");
            then.status(200).json_body(reply());
        })
        .await;
    let mut config = config(&upstream);
    config.router_settings.routing.routing_strategy = Some("tag_based".into());
    config.model_list[0].model_info =
        Some(serde_json::from_value(json!({"tags":["env:prod"]})).unwrap());
    let mut other = config.model_list[0].clone();
    other.oxllm_params.api_key = Some("synthetic-other-key".into());
    other.model_info = Some(serde_json::from_value(json!({"tags":["env:test"]})).unwrap());
    config.model_list.push(other);
    let proxy = start_with_config(config).await;
    for tag in ["&env:prod", "!env:test", "&env:missing"] {
        let response = common::http_client()
            .post(proxy.url("/v1/chat/completions"))
            .bearer_auth(MASTER_KEY)
            .header("x-litellm-tag", tag)
            .json(&body("/v1/chat/completions", "hello"))
            .send()
            .await
            .unwrap();
        assert_eq!(
            response.status(),
            if tag == "&env:missing" { 400 } else { 200 }
        );
    }
    wire.assert_hits_async(2).await;
    proxy.shutdown().await;
}
