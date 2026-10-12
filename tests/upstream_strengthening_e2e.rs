//! Wire and ledger regressions for gateway accounting and recovery corrections.

mod common;

use std::sync::Arc;
use std::time::Duration;

use httpmock::{Method::POST, MockServer};
use oxllm_server::app::{build_router, build_shared_state_with_db};
use oxllm_server::config::types::ProxyConfig;
use oxllm_server::{Database, SharedState, spend_writer};
use serde_json::{Value, json};

const MASTER_KEY: &str = "sk-strengthening-synthetic-master";

fn response(model: &str, text: &str, receipt: Option<f64>) -> Value {
    let mut body = json!({"id":"strengthened", "model":model, "object":"chat.completion",
        "choices":[{"index":0,"message":{"role":"assistant","content":text},"finish_reason":"stop"}],
        "usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}});
    if let Some(receipt) = receipt {
        body["usage"]["cost"] = json!(receipt);
    }
    body
}

fn config(upstream: &MockServer) -> ProxyConfig {
    serde_json::from_value(json!({
        "model_list":[{"model_name":"primary", "oxllm_params":{
            "model":"openai/gpt-4o", "api_key":"synthetic-key", "api_base":upstream.base_url()
        }}], "general_settings":{"master_key":MASTER_KEY,"allow_private_api_base":true},
        "router_settings":{"num_retries":0,"cooldown_time":0}
    }))
    .unwrap()
}

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
    async fn shutdown(self) -> Arc<Database> {
        self.handle.shutdown();
        tokio::time::timeout(Duration::from_secs(10), self.server)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        self.writer.shutdown().await;
        self.db
    }
}

async fn start(config: ProxyConfig) -> Proxy {
    let db = Database::connect("sqlite::memory:").await.unwrap();
    db.migrate().await.unwrap();
    let (writer, writer_handle) = spend_writer::spawn_with_shutdown(db.clone(), 60, 100);
    let shared =
        build_shared_state_with_db(config, Some(writer.sender()), Some(db.clone())).unwrap();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
    let router = build_router(shared.clone());
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

#[tokio::test]
async fn aggregator_receipts_agree_across_json_stream_budget_and_persisted_spend() {
    for streaming in [false, true] {
        for (receipt, explicit_prices, expected) in [
            (0.017, None, 0.017),
            (0.0, None, 0.0),
            (0.017, Some((0.002, 0.005)), 0.015),
            (0.017, Some((0.0, 0.0)), 0.0),
        ] {
            let upstream = MockServer::start_async().await;
            let wire = upstream.mock_async(|when, then| {
                when.method(POST).path("/chat/completions");
                if streaming {
                    then.status(200).header("content-type","text/event-stream").body(format!(
                        "data: {}\n\ndata: {}\n\ndata: [DONE]\n\n",
                        json!({"model":"openai/gpt-4o", "choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":null}]}),
                        json!({"model":"openai/gpt-4o","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6,"cost":receipt}})
                    ));
                } else { then.status(200).json_body(response("openai/gpt-4o","ok",Some(receipt))); }
            }).await;
            let mut config = config(&upstream);
            let params = &mut config.model_list[0].oxllm_params;
            params.model = "openrouter/openai/gpt-4o".into();
            if let Some((input, output)) = explicit_prices {
                params.input_cost_per_token = Some(input);
                params.output_cost_per_token = Some(output);
            }
            let proxy = start(config).await;
            let client = common::http_client();
            let generated = client
                .post(proxy.url("/key/generate"))
                .bearer_auth(MASTER_KEY)
                .json(&json!({"max_budget":10.0}))
                .send()
                .await
                .unwrap();
            assert_eq!(generated.status(), 200);
            let key: Value = generated.json().await.unwrap();
            let reply = client.post(proxy.url("/v1/chat/completions"))
                .bearer_auth(key["key"].as_str().unwrap())
                .json(&json!({"model":"primary","max_tokens":5,"stream":streaming,"stream_options":{"include_usage":true},
                    "messages":[{"role":"user","content":"hello"}]})).send().await.unwrap();
            assert_eq!(reply.status(), 200, "{}", reply.text().await.unwrap());
            if !streaming {
                let cost: f64 = reply.headers()["x-oxllm-response-cost"]
                    .to_str()
                    .unwrap()
                    .parse()
                    .unwrap();
                let counter: f64 = reply.headers()["x-oxllm-key-spend"]
                    .to_str()
                    .unwrap()
                    .parse()
                    .unwrap();
                assert!((cost - expected).abs() < 1e-12);
                assert!((counter - expected).abs() < 1e-12);
            } else {
                assert!(reply.headers().get("x-oxllm-key-spend").is_none());
            }
            let body = reply.text().await.unwrap();
            if streaming {
                assert!(body.contains("[DONE]"));
            }
            let hash = oxllm_server::auth::hash_key_hex(key["key"].as_str().unwrap().as_bytes());
            let cache = proxy.shared.load_full().cache.clone();
            let rows = proxy
                .shutdown()
                .await
                .query_spend(&Default::default())
                .await
                .unwrap();
            assert_eq!(rows.len(), 1);
            assert_eq!(rows[0].status.as_deref(), Some("success"));
            assert!((rows[0].cost.unwrap() - expected).abs() < 1e-12);
            assert_eq!(rows[0].key_id.as_deref(), key["key_id"].as_str());
            assert_eq!(rows[0].completion_tokens, Some(1));
            let counter = cache
                .get(&oxllm_server::hooks::spend_keys::spend_key(&hash))
                .await
                .expect("the tracked key has a settled budget counter");
            let counter: f64 = counter.parse().unwrap();
            assert!(
                (counter - expected).abs() < 1e-12,
                "JSON and SSE settlement must debit the same measured cost"
            );
            wire.assert_hits_async(1).await;
        }
    }
}

#[tokio::test]
async fn explicitly_free_native_deployments_remain_free_for_receipt_only_responses() {
    for provider in ["openrouter", "perplexity"] {
        let upstream = MockServer::start_async().await;
        let mut body = response("private-free-model", "ok", None);
        body["usage"] = json!({"cost":0.017});
        let wire = upstream
            .mock_async(|when, then| {
                when.method(POST).path("/v1/chat/completions");
                then.status(200).json_body(body.clone());
            })
            .await;
        let mut config = config(&upstream);
        let params = &mut config.model_list[0].oxllm_params;
        params.model = format!("{provider}/private-free-model");
        params.input_cost_per_token = Some(0.0);
        params.output_cost_per_token = Some(0.0);
        let proxy = start(config).await;
        let client = common::http_client();
        let key: Value = client
            .post(proxy.url("/key/generate"))
            .bearer_auth(MASTER_KEY)
            .json(&json!({"max_budget":10.0}))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        let reply = client.post(proxy.url(&format!("/{provider}/v1/chat/completions")))
            .bearer_auth(key["key"].as_str().unwrap())
            .json(&json!({"model":"primary","max_tokens":5,"messages":[{"role":"user","content":"hello"}]}))
            .send().await.unwrap();
        assert_eq!(reply.status(), 200);
        let charged: f64 = reply.headers()["x-oxllm-response-cost"]
            .to_str()
            .unwrap()
            .parse()
            .unwrap();
        assert_eq!(
            charged, 0.0,
            "an explicit free override must not bill a receipt without token counters"
        );
        assert_eq!(
            reply.json::<Value>().await.unwrap(),
            body,
            "native receipt data remains provider-owned"
        );
        let hash = oxllm_server::auth::hash_key_hex(key["key"].as_str().unwrap().as_bytes());
        let cache = proxy.shared.load_full().cache.clone();
        let rows = proxy
            .shutdown()
            .await
            .query_spend(&Default::default())
            .await
            .unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].status.as_deref(), Some("success"));
        assert_eq!(rows[0].cost, Some(0.0));
        assert_eq!(rows[0].total_tokens, None);
        assert_eq!(
            cache
                .get(&oxllm_server::hooks::spend_keys::spend_key(&hash))
                .await
                .unwrap()
                .parse::<f64>()
                .unwrap(),
            0.0
        );
        wire.assert_hits_async(1).await;
    }
}

#[tokio::test]
async fn receiptless_aggregator_json_and_sse_price_the_serving_model() {
    for streaming in [false, true] {
        let upstream = MockServer::start_async().await;
        let wire = upstream.mock_async(|when, then| {
            when.method(POST).path("/chat/completions");
            if streaming {
                then.status(200).header("content-type", "text/event-stream").body(format!(
                    "data: {}\n\ndata: {}\n\ndata: [DONE]\n\n",
                    json!({"model":"openai/gpt-4o-mini","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":null}]}),
                    json!({"model":"openai/gpt-4o-mini","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}),
                ));
            } else {
                then.status(200).json_body(response("openai/gpt-4o-mini", "ok", None));
            }
        }).await;
        let mut config = config(&upstream);
        config.model_list[0].oxllm_params.model = "openrouter/openai/gpt-4o".into();
        let proxy = start(config).await;
        let client = common::http_client();
        let key: Value = client
            .post(proxy.url("/key/generate"))
            .bearer_auth(MASTER_KEY)
            .json(&json!({"max_budget":10.0}))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        let reply = client.post(proxy.url("/v1/chat/completions"))
            .bearer_auth(key["key"].as_str().unwrap())
            .json(&json!({"model":"primary","max_tokens":5,"stream":streaming,
                "stream_options":{"include_usage":true},"messages":[{"role":"user","content":"hello"}]}))
            .send().await.unwrap();
        assert_eq!(reply.status(), 200);
        // The serving mini card is $0.15/$0.60 per million tokens, not the
        // $2.50/$10.00 card for the originally configured upstream selector.
        let expected = 5.0 * 0.15e-6 + 0.60e-6;
        if !streaming {
            let cost: f64 = reply.headers()["x-oxllm-response-cost"]
                .to_str()
                .unwrap()
                .parse()
                .unwrap();
            assert!((cost - expected).abs() < 1e-12);
        }
        let body = reply.text().await.unwrap();
        if streaming {
            assert!(body.contains("[DONE]"));
        }
        let hash = oxllm_server::auth::hash_key_hex(key["key"].as_str().unwrap().as_bytes());
        let cache = proxy.shared.load_full().cache.clone();
        let rows = proxy
            .shutdown()
            .await
            .query_spend(&Default::default())
            .await
            .unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].model, "openai/gpt-4o-mini");
        assert!(
            (rows[0].cost.unwrap() - expected).abs() < 1e-12,
            "stream={streaming}: receiptless responses must use the serving card"
        );
        let counter: f64 = cache
            .get(&oxllm_server::hooks::spend_keys::spend_key(&hash))
            .await
            .unwrap()
            .parse()
            .unwrap();
        assert!((counter - expected).abs() < 1e-12);
        wire.assert_hits_async(1).await;
    }
}

#[tokio::test]
async fn exact_policy_codes_select_policy_fallbacks_before_commit_only() {
    for stage in ["json", "prelude", "committed"] {
        let upstream = MockServer::start_async().await;
        let primary = upstream.mock_async(|when, then| {
            when.method(POST).path("/chat/completions").json_body_partial(json!({"model":"gpt-4o"}).to_string());
            let error = json!({"error":{"code":"cyber_policy","message":"request declined"}});
            if stage == "json" { then.status(400).json_body(error); }
            else {
                let prefix = if stage == "committed" {
                    format!("data: {}\n\n",json!({"model":"gpt-4o","choices":[{"index":0,"delta":{"content":"partial"},"finish_reason":null}]}))
                } else { String::new() };
                then.status(200).header("content-type","text/event-stream").body(format!("{prefix}data: {error}\n\n"));
            }
        }).await;
        let fallback = upstream.mock_async(|when, then| {
            when.method(POST).path("/chat/completions").json_body_partial(json!({"model":"gpt-4o-mini"}).to_string());
            if stage == "json" { then.status(200).json_body(response("gpt-4o-mini","recovered",None)); }
            else { then.status(200).header("content-type","text/event-stream").body(format!(
                "data: {}\n\ndata: [DONE]\n\n",json!({"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"recovered"},"finish_reason":null}]})) ); }
        }).await;
        let mut config = config(&upstream);
        let mut backup = config.model_list[0].clone();
        backup.model_name = "backup".into();
        backup.oxllm_params.model = "openai/gpt-4o-mini".into();
        config.model_list.push(backup);
        config.router_settings.routing.content_policy_fallbacks =
            Some(serde_json::from_value(json!([{"primary":["backup"]}])).unwrap());
        let proxy = start(config).await;
        let reply = common::http_client().post(proxy.url("/v1/chat/completions")).bearer_auth(MASTER_KEY)
            .json(&json!({"model":"primary","stream":stage != "json","max_tokens":5,"messages":[{"role":"user","content":"hello"}]}))
            .send().await.unwrap();
        assert_eq!(reply.status(), 200, "{}", reply.text().await.unwrap());
        let body = reply.text().await.unwrap();
        if stage == "committed" {
            assert!(body.contains("partial"));
            assert!(!body.contains("recovered"));
        } else {
            assert!(body.contains("recovered"));
        }
        primary.assert_hits_async(1).await;
        fallback
            .assert_hits_async(usize::from(stage != "committed"))
            .await;
        proxy.shutdown().await;
    }
}

#[tokio::test]
async fn native_response_policy_errors_preserve_wire_bytes_without_cooling_healthy_routes() {
    for nested in [false, true] {
        for (code, explicit_policy, cooling) in [
            ("cyber_policy", false, false),
            ("invalid_request_error", false, false),
            ("not_cyber_policy", false, true),
            ("cyber_policy", true, true),
        ] {
            let upstream = MockServer::start_async().await;
            let error = json!({"code":code,"message":"request declined"});
            let event = if nested {
                json!({"type":"response.failed","response":{"id":"failed-response","status":"failed","error":error}})
            } else {
                json!({"type":"error","error":error})
            };
            let wire = format!(
                "event: {}\ndata: {event}\n\n",
                event["type"].as_str().unwrap()
            );
            let mocked = upstream
                .mock_async(|when, then| {
                    when.method(POST).path("/v1/responses");
                    then.status(200)
                        .header("content-type", "text/event-stream")
                        .body(&wire);
                })
                .await;
            let mut config = config(&upstream);
            config.router_settings.routing.allowed_fails = Some(0);
            config.router_settings.routing.cooldown_time = Some(60);
            if explicit_policy {
                config.router_settings.routing.allowed_fails_policy =
                    Some(std::collections::HashMap::from([("400".into(), 0)]));
            }
            let proxy = start(config).await;
            let client = common::http_client();
            let generated: Value = client
                .post(proxy.url("/key/generate"))
                .bearer_auth(MASTER_KEY)
                .json(&json!({"max_budget":10.0}))
                .send()
                .await
                .unwrap()
                .json()
                .await
                .unwrap();
            let reply = client
                .post(proxy.url("/openai/v1/responses"))
                .bearer_auth(generated["key"].as_str().unwrap())
                .json(
                    &json!({"model":"primary","input":"hello","stream":true,"max_output_tokens":5}),
                )
                .send()
                .await
                .unwrap();
            assert_eq!(
                reply.status(),
                200,
                "the native stream has committed its HTTP status"
            );
            assert_eq!(
                reply.text().await.unwrap(),
                wire,
                "native error events retain their wire shape"
            );
            let state = proxy.shared.load_full();
            let key = state.router.cooldown_key(&state.deployments[0]);
            assert_eq!(
                state.router.cooldown_store().is_cooling(&key),
                cooling,
                "{code}, nested={nested}"
            );
            assert_eq!(
                state.router.cooldown_store().failure_count(&key, 400),
                u32::from(explicit_policy)
            );
            assert_eq!(
                state.router.cooldown_store().failure_count(&key, 500),
                u32::from(code == "not_cyber_policy")
            );
            let hash =
                oxllm_server::auth::hash_key_hex(generated["key"].as_str().unwrap().as_bytes());
            let cache = state.cache.clone();
            let rows = proxy
                .shutdown()
                .await
                .query_spend(&Default::default())
                .await
                .unwrap();
            assert_eq!(rows.len(), 1);
            assert_eq!(rows[0].status.as_deref(), Some("failure"));
            assert_eq!(rows[0].cost, Some(0.0));
            assert_eq!(rows[0].total_tokens, None);
            let counter = cache
                .get(&oxllm_server::hooks::spend_keys::spend_key(&hash))
                .await
                .unwrap();
            assert_eq!(
                counter.parse::<f64>().unwrap(),
                0.0,
                "failed native streams release the key reservation"
            );
            mocked.assert_hits_async(1).await;
        }
    }
}

#[tokio::test]
async fn context_window_client_errors_still_count_toward_configured_global_cooldown() {
    let upstream = MockServer::start_async().await;
    let wire = upstream
        .mock_async(|when, then| {
            when.method(POST).path("/chat/completions");
            then.status(400).json_body(json!({
                "error":{"code":"context_length_exceeded","message":"prompt is too long"}
            }));
        })
        .await;
    let mut config = config(&upstream);
    config.router_settings.routing.allowed_fails = Some(0);
    config.router_settings.routing.cooldown_time = Some(60);
    let proxy = start(config).await;
    let reply = common::http_client()
        .post(proxy.url("/v1/chat/completions"))
        .bearer_auth(MASTER_KEY)
        .json(&json!({"model":"primary","max_tokens":5,"messages":[{"role":"user","content":"hello"}]}))
        .send()
        .await
        .unwrap();
    assert_eq!(reply.status(), 400, "{}", reply.text().await.unwrap());
    let state = proxy.shared.load_full();
    let key = state.router.cooldown_key(&state.deployments[0]);
    assert!(
        state.router.cooldown_store().is_cooling(&key),
        "a classified context-window 400 must keep counting toward a configured \
         global allowed_fails; the terminal-relay opt-in must not suppress it"
    );
    assert_eq!(state.router.cooldown_store().failure_count(&key, 400), 1);
    wire.assert_hits_async(1).await;
    proxy.shutdown().await;
}
