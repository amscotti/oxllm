//! Regressions for request attribution, admission, billing and management contracts.

mod common;

use std::sync::Arc;
use std::time::Duration;

use httpmock::{Method::POST, MockServer};
use oxllm_server::app::{build_router, build_shared_state_with_db};
use oxllm_server::config::types::ProxyConfig;
use oxllm_server::{Database, SharedState, spend_writer};
use serde_json::{Value, json};

const MASTER_KEY: &str = "sk-r4-synthetic-master-key";

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

async fn start(config: ProxyConfig) -> Proxy {
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
    json!({"id":"r4", "model":"gpt-4o", "object":"chat.completion", "choices":[{
        "index":0, "message":{"role":"assistant", "content":"ok"}, "finish_reason":"stop"
    }], "usage":{"prompt_tokens":5, "completion_tokens":1, "total_tokens":6}})
}

fn config(upstream: &MockServer) -> ProxyConfig {
    serde_json::from_value(json!({
        "model_list":[{"model_name":"chat", "oxllm_params":{
            "model":"openai/gpt-4o", "api_key":"synthetic-key", "api_base":upstream.base_url()
        }}],
        "general_settings":{"allow_private_api_base":true, "master_key":MASTER_KEY},
        "router_settings":{"num_retries":0}
    }))
    .unwrap()
}

async fn wait_for_rows(db: &Database, count: usize) -> Vec<oxllm_server::SpendLog> {
    let mut observed = 0;
    let result = tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            let rows = db
                .query_spend(&Default::default())
                .await
                .expect("query spend rows during asynchronous read-back");
            observed = rows.len();
            if observed >= count {
                return rows;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    })
    .await;
    result.unwrap_or_else(|_| {
        panic!("expected at least {count} spend rows within 30 seconds; last observed {observed}")
    })
}

async fn generate_key(proxy: &Proxy, policy: Value) -> Value {
    let response = common::http_client()
        .post(proxy.url("/key/generate"))
        .bearer_auth(MASTER_KEY)
        .json(&policy)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200, "{}", response.text().await.unwrap());
    response.json().await.unwrap()
}

#[tokio::test]
async fn client_errors_do_not_poison_single_or_multi_leg_groups() {
    for (path, upstream_path) in [
        ("/v1/chat/completions", "/chat/completions"),
        ("/v1/embeddings", "/embeddings"),
    ] {
        for status in [400, 422] {
            for legs in [1, 2] {
                let upstream = MockServer::start_async().await;
                let mut config = config(&upstream);
                for leg in 1..legs {
                    let mut deployment = config.model_list[0].clone();
                    deployment.oxllm_params.api_key = Some(format!("synthetic-leg-{leg}"));
                    config.model_list.push(deployment);
                }
                config.router_settings.routing.allowed_fails = Some(0);
                config.router_settings.routing.cooldown_time = Some(60);
                config.router_settings.routing.retry_policy =
                    Some(std::collections::HashMap::from([
                        ("BadRequestErrorRetries".into(), 0),
                        ("AuthenticationErrorRetries".into(), 0),
                        ("ContentPolicyViolationErrorRetries".into(), 0),
                    ]));
                // Parse through the real config path so aliases receive their
                // translations rather than only testing programmatic settings.
                let config =
                    oxllm_server::config::parse_config_str(&serde_yml::to_string(&config).unwrap())
                        .unwrap();
                let failed = upstream
                    .mock_async(|when, then| {
                        when.method(POST)
                            .path(upstream_path)
                            .body_contains("bogus_param_xyz");
                        then.status(status).json_body(
                            json!({"error":{"message":"Unknown parameter: bogus_param_xyz"}}),
                        );
                    })
                    .await;
                let healthy = upstream
                    .mock_async(|when, then| {
                        when.method(POST).path(upstream_path).matches(|request| {
                            request
                                .body
                                .as_ref()
                                .and_then(|body| serde_json::from_slice::<Value>(body).ok())
                                .is_some_and(|body| body.get("bogus_param_xyz").is_none())
                        });
                        then.status(200).json_body(if path == "/v1/embeddings" {
                            json!({"object":"list", "model":"gpt-4o", "data":[],
                            "usage":{"prompt_tokens":5,"total_tokens":5}})
                        } else {
                            reply()
                        });
                    })
                    .await;
                let proxy = start(config).await;
                let client = common::http_client();
                let body = if path == "/v1/embeddings" {
                    json!({"model":"chat", "input":"hello"})
                } else {
                    json!({"model":"chat", "messages":[{"role":"user", "content":"hello"}], "max_tokens":5})
                };
                let mut invalid = body.clone();
                invalid["bogus_param_xyz"] = json!(1);
                let response = client
                    .post(proxy.url(path))
                    .bearer_auth(MASTER_KEY)
                    .json(&invalid)
                    .send()
                    .await
                    .unwrap();
                assert_eq!(response.status(), status);
                let state = proxy.shared.load_full();
                for deployment in &state.deployments {
                    let key = state.router.cooldown_key(deployment);
                    assert!(
                        !state.router.cooldown_store().is_cooling(&key),
                        "{path}: {status}, legs={legs}"
                    );
                    assert_eq!(state.router.cooldown_store().failure_count(&key, status), 0);
                    assert_eq!(state.router.usage_store().in_flight(&key), 0);
                }
                let response = client
                    .post(proxy.url(path))
                    .bearer_auth(MASTER_KEY)
                    .json(&body)
                    .send()
                    .await
                    .unwrap();
                assert_eq!(
                    response.status(),
                    200,
                    "{status}, legs={legs}: {}",
                    response.text().await.unwrap()
                );
                failed.assert_hits_async(1).await;
                healthy.assert_hits_async(1).await;
                proxy.shutdown().await;
            }
        }
    }
}

#[tokio::test]
async fn top_level_tags_are_attributed_and_stripped_across_translated_surfaces() {
    let upstream = MockServer::start_async().await;
    let wire = upstream
        .mock_async(|when, then| {
            when.method(POST)
                .path("/chat/completions")
                .matches(|request| {
                    request
                        .body
                        .as_ref()
                        .and_then(|body| serde_json::from_slice::<Value>(body).ok())
                        .is_some_and(|body| {
                            body.get("tags").is_none() && body.pointer("/metadata/tags").is_none()
                        })
                });
            then.status(200).json_body(reply());
        })
        .await;
    let proxy = start(config(&upstream)).await;
    for path in [
        "/v1/chat/completions",
        "/v1/messages",
        "/v1/responses",
        "/v1/completions",
    ] {
        let response = common::http_client()
            .post(proxy.url(path))
            .bearer_auth(MASTER_KEY)
            .header("x-oxllm-tag", "header,top")
            .json(
                &json!({"model":"chat", "max_tokens":5, "input":"hello", "prompt":"hello",
                    "messages":[{"role":"user", "content":"hello"}],
                    "tags":[" top ","shared"], "metadata":{"tags":["metadata","shared"]}
                }),
            )
            .send()
            .await
            .unwrap();
        assert_eq!(
            response.status(),
            200,
            "{path}: {}",
            response.text().await.unwrap()
        );
    }
    wire.assert_hits_async(4).await;
    let db = proxy.db.clone();
    proxy.shutdown().await;
    let rows = db.query_spend(&Default::default()).await.unwrap();
    assert_eq!(rows.len(), 4);
    for row in rows {
        assert_eq!(row.tag_name.as_deref(), Some("top"));
        assert_eq!(
            serde_json::from_str::<Vec<String>>(row.tags.as_deref().unwrap()).unwrap(),
            ["top", "shared", "metadata", "header"]
        );
    }
}

#[tokio::test]
async fn header_tag_aliases_reach_the_ledger_and_tag_budgets_across_surfaces() {
    let upstream = MockServer::start_async().await;
    let wire = upstream
        .mock_async(|when, then| {
            when.method(POST)
                .header("authorization", "Bearer synthetic-key")
                .matches(|request| {
                    request.headers.as_ref().is_some_and(|headers| {
                        headers.iter().all(|(name, _)| {
                            !name.starts_with("x-litellm-tag") && !name.starts_with("x-oxllm-tag")
                        })
                    })
                });
            then.status(200).json_body(reply());
        })
        .await;
    let mut config = config(&upstream);
    config.router_settings.routing.routing_strategy = Some("tag_based".into());
    config.model_list[0].model_info = Some(
        serde_json::from_value(json!({
            "tags":["product:demo", "firm:example", "user:synthetic"]
        }))
        .unwrap(),
    );
    let mut other = config.model_list[0].clone();
    other.oxllm_params.api_key = Some("synthetic-other-leg".into());
    other.model_info = Some(serde_json::from_value(json!({"tags":["product:other"]})).unwrap());
    config.model_list.push(other);
    let proxy = start(config).await;
    let client = common::http_client();
    for (index, path) in [
        "/v1/chat/completions",
        "/v1/messages",
        "/v1/responses",
        "/v1/completions",
        "/v1/embeddings",
    ]
    .into_iter()
    .enumerate()
    {
        let response = client
            .post(proxy.url(path))
            .bearer_auth(MASTER_KEY)
            .header("x-litellm-tags", "product:demo, firm:example")
            .header("x-litellm-tag", "firm:example, user:synthetic")
            .json(
                &json!({"model":"chat", "max_tokens":5, "input":"hello", "prompt":"hello",
                "messages":[{"role":"user", "content":"hello"}]}),
            )
            .send()
            .await
            .unwrap();
        assert_eq!(
            response.status(),
            200,
            "{path}: {}",
            response.text().await.unwrap()
        );
        let rows = wait_for_rows(&proxy.db, index + 1).await;
        let row = rows.first().unwrap();
        assert_eq!(row.tag_name.as_deref(), Some("product:demo"));
        assert_eq!(
            serde_json::from_str::<Vec<String>>(row.tags.as_deref().unwrap()).unwrap(),
            ["product:demo", "firm:example", "user:synthetic"]
        );
    }
    let response = client
        .post(proxy.url("/tag/new"))
        .bearer_auth(MASTER_KEY)
        .json(&json!({"name":"product:blocked", "max_budget":0.0}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200, "{}", response.text().await.unwrap());
    let key = generate_key(&proxy, json!({"models":["chat"]})).await;
    let response = client.post(proxy.url("/v1/chat/completions"))
        .bearer_auth(key["key"].as_str().unwrap()).header("x-litellm-tags", "product:blocked")
        .json(&json!({"model":"chat", "messages":[{"role":"user", "content":"hello"}], "max_tokens":5}))
        .send().await.unwrap();
    assert_eq!(
        response.status(),
        429,
        "tag alias must participate in admission"
    );
    let rows = wait_for_rows(&proxy.db, 6).await;
    assert_eq!(rows[0].tag_name.as_deref(), Some("product:blocked"));
    assert_eq!(rows[0].status.as_deref(), Some("failure"));
    wire.assert_hits_async(5).await;
    proxy.shutdown().await;
}

#[tokio::test]
async fn vertex_reasoning_usage_reaches_client_cost_and_spend_for_json_and_sse() {
    for provider in ["vertex_ai", "gemini"] {
        for stream in [false, true] {
            let upstream = MockServer::start_async().await;
            let usage = json!({"prompt_tokens":5, "completion_tokens":1, "total_tokens":11,
            "completion_tokens_details":{"reasoning_tokens":5}});
            let wire = upstream.mock_async(|when, then| {
            when.method(POST).path(if provider == "vertex_ai" {
                "/v1beta1/projects/synthetic-project/locations/us-central1/endpoints/openapi/chat/completions"
            } else { "/chat/completions" });
            if stream {
                then.status(200).header("content-type", "text/event-stream").body(format!(
                    "data: {}\n\ndata: {}\n\ndata: [DONE]\n\n",
                    json!({"model":"gemini-3-flash", "choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":null}]}),
                    json!({"model":"gemini-3-flash", "choices":[], "usage":usage})
                ));
            } else {
                let mut body = reply();
                body["model"] = json!("gemini-3-flash");
                body["usage"] = usage;
                then.status(200).json_body(body);
            }
        }).await;
            let mut config = config(&upstream);
            let params = &mut config.model_list[0].oxllm_params;
            params.model = format!("{provider}/gemini-3-flash");
            params.input_cost_per_token = Some(5e-7);
            params.output_cost_per_token = Some(3e-6);
            if provider == "vertex_ai" {
                params
                    .extra
                    .insert("vertex_project".into(), json!("synthetic-project"));
            }
            let proxy = start(config).await;
            let response = common::http_client()
            .post(proxy.url("/v1/chat/completions"))
            .bearer_auth(MASTER_KEY)
            .json(
                &json!({"model":"chat", "stream":stream, "stream_options":{"include_usage":true},
                "messages":[{"role":"user", "content":"hello"}], "max_tokens":10}),
            )
            .send()
            .await
            .unwrap();
            assert_eq!(response.status(), 200, "{}", response.text().await.unwrap());
            if !stream {
                let cost: f64 = response.headers()["x-oxllm-response-cost"]
                    .to_str()
                    .unwrap()
                    .parse()
                    .unwrap();
                assert!((cost - 2.05e-5).abs() < 1e-12);
                let body: Value = response.json().await.unwrap();
                assert_eq!(body["usage"]["completion_tokens"], 6);
                assert_eq!(
                    body["usage"]["completion_tokens_details"]["reasoning_tokens"],
                    5
                );
            } else {
                let body = response.text().await.unwrap();
                let chunks = body
                    .lines()
                    .filter_map(|line| line.strip_prefix("data: "))
                    .filter_map(|data| serde_json::from_str::<Value>(data).ok())
                    .collect::<Vec<_>>();
                let usage = chunks.iter().find_map(|chunk| chunk.get("usage")).unwrap();
                assert_eq!(usage["completion_tokens"], 6);
                assert_eq!(usage["completion_tokens_details"]["reasoning_tokens"], 5);
            }
            wire.assert_hits_async(1).await;
            let db = proxy.db.clone();
            proxy.shutdown().await;
            let rows = db.query_spend(&Default::default()).await.unwrap();
            assert_eq!(rows.len(), 1);
            assert_eq!(rows[0].completion_tokens, Some(6));
            assert_eq!(rows[0].total_tokens, Some(11));
            assert!((rows[0].cost.unwrap() - 2.05e-5).abs() < 1e-12);
        }
    }
}

#[tokio::test]
async fn key_limit_aliases_round_trip_and_enforce_the_created_policy() {
    let upstream = MockServer::start_async().await;
    let wire = upstream
        .mock_async(|when, then| {
            when.method(POST).path("/chat/completions");
            then.status(200).json_body(reply());
        })
        .await;
    let proxy = start(config(&upstream)).await;
    let client = common::http_client();
    let response = client
        .post(proxy.url("/key/generate"))
        .bearer_auth(MASTER_KEY)
        .json(&json!({"key_alias":"limited", "models":["chat"], "rpm_limit":1, "tpm_limit":500}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200, "{}", response.text().await.unwrap());
    let created: Value = response.json().await.unwrap();
    let id = created["key_id"].as_str().unwrap();
    let secret = created["key"].as_str().unwrap();
    let info: Value = client
        .get(proxy.url("/key/info"))
        .bearer_auth(MASTER_KEY)
        .query(&[("key_id", id)])
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(info["rpm"], 1);
    assert_eq!(info["tpm"], 500);
    assert_eq!(info["models"], json!(["chat"]));
    assert_eq!(info["key_alias"], "limited");
    assert!(info.get("key_hash").is_none());
    let body =
        json!({"model":"chat", "max_tokens":5, "messages":[{"role":"user", "content":"hello"}]});
    assert_eq!(
        client
            .post(proxy.url("/v1/chat/completions"))
            .bearer_auth(secret)
            .json(&body)
            .send()
            .await
            .unwrap()
            .status(),
        200
    );
    assert_eq!(
        client
            .post(proxy.url("/v1/chat/completions"))
            .bearer_auth(secret)
            .json(&body)
            .send()
            .await
            .unwrap()
            .status(),
        429,
        "rpm_limit must not silently become unlimited"
    );
    let response = client
        .post(proxy.url("/key/update"))
        .bearer_auth(MASTER_KEY)
        .json(&json!({"key_id":id, "rpm_limit":2, "tpm_limit":600, "models":["chat","other"]}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let updated: Value = response.json().await.unwrap();
    assert_eq!(updated["rpm"], 2);
    assert_eq!(updated["tpm"], 600);
    assert_eq!(updated["models"], json!(["chat", "other"]));
    let listed: Value = client
        .get(proxy.url("/key/list"))
        .bearer_auth(MASTER_KEY)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(listed["keys"][0]["models"], json!(["chat", "other"]));
    assert!(listed["keys"][0].get("key_hash").is_none());
    wire.assert_hits_async(1).await;
    proxy.shutdown().await;
}

#[tokio::test]
async fn unknown_or_conflicting_key_fields_fail_before_mutation() {
    let upstream = MockServer::start_async().await;
    let proxy = start(config(&upstream)).await;
    let client = common::http_client();
    for body in [
        json!({"metadata":{"application":"one"}}),
        json!({"tags":["one"]}),
        json!({"rpm_limt":1}),
        json!({"max_parallel_requests":1}),
        json!({"permissions":{}}),
        json!({"rpm":1, "rpm_limit":2}),
        json!({"tpm":1, "tpm_limit":2}),
    ] {
        let response = client
            .post(proxy.url("/key/generate"))
            .bearer_auth(MASTER_KEY)
            .json(&body)
            .send()
            .await
            .unwrap();
        assert_eq!(
            response.status(),
            400,
            "{body}: {}",
            response.text().await.unwrap()
        );
        assert!(proxy.db.list_keys(None).await.unwrap().is_empty());
    }
    let key: Value = client
        .post(proxy.url("/key/generate"))
        .bearer_auth(MASTER_KEY)
        .json(&json!({"rpm":5}))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    for mut update in [
        json!({"rpm_limit":1, "metadata":{}}),
        json!({"rpm":1, "rpm_limit":2}),
        json!({"tpm":1, "tpm_limit":2}),
        json!({"permissions":{}}),
    ] {
        update["key_id"] = key["key_id"].clone();
        let response = client
            .post(proxy.url("/key/update"))
            .bearer_auth(MASTER_KEY)
            .json(&update)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 400, "{update}");
        let persisted = proxy
            .db
            .get_key(key["key_id"].as_str().unwrap())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(persisted.rpm, Some(5));
        assert_eq!(persisted.tpm, None);
    }
    proxy.shutdown().await;
}

#[tokio::test]
async fn denials_and_upstream_failures_are_reported_once_without_billing_or_body_storage() {
    let upstream = MockServer::start_async().await;
    let failed = upstream
        .mock_async(|when, then| {
            when.method(POST).body_contains("fail-private-prompt");
            then.status(503)
                .json_body(json!({"error":{"message":"sk-private-upstream-detail"}}));
        })
        .await;
    let healthy = upstream
        .mock_async(|when, then| {
            when.method(POST).matches(|request| {
                request.body.as_ref().is_some_and(|body| {
                    !String::from_utf8_lossy(body).contains("fail-private-prompt")
                })
            });
            then.status(200).json_body(reply());
        })
        .await;
    let mut config = config(&upstream);
    config.oxllm_settings.max_budget = Some(10.0);
    config.router_settings.routing.cooldown_time = Some(0);
    let proxy = start(config).await;
    let client = common::http_client();
    let mut keys = Vec::new();
    for policy in [
        json!({"models":["other"]}),
        json!({"max_budget":1e-12}),
        json!({}),
    ] {
        let created: Value = client
            .post(proxy.url("/key/generate"))
            .bearer_auth(MASTER_KEY)
            .json(&policy)
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        keys.push(created);
    }
    let body = json!({"model":"chat", "max_tokens":5,
        "messages":[{"role":"user", "content":"hello-private-prompt"}], "tags":["audit"]});
    let mut request_ids = Vec::new();
    for (key, status, body) in [
        (&keys[0], 403, body.clone()),
        (&keys[1], 429, body.clone()),
        (
            &keys[2],
            503,
            json!({"model":"chat", "max_tokens":5,
            "messages":[{"role":"user", "content":"fail-private-prompt"}], "tags":["audit"]}),
        ),
    ] {
        let response = client
            .post(proxy.url("/v1/chat/completions"))
            .bearer_auth(key["key"].as_str().unwrap())
            .header("x-oxllm-tag", "header-audit")
            .json(&body)
            .send()
            .await
            .unwrap();
        assert_eq!(
            response.status(),
            status,
            "{}",
            response.text().await.unwrap()
        );
        request_ids.push((
            response.headers()["x-request-id"]
                .to_str()
                .unwrap()
                .to_string(),
            key["key_id"].as_str().unwrap().to_owned(),
        ));
    }
    let failures = wait_for_rows(&proxy.db, 3).await;
    assert_eq!(
        failures.len(),
        3,
        "rollback and terminal hooks must not duplicate request rows"
    );
    for (request_id, key_id) in &request_ids {
        let matching = failures
            .iter()
            .filter(|row| row.request_id.as_ref() == Some(request_id))
            .collect::<Vec<_>>();
        assert_eq!(matching.len(), 1, "one row for each request identity");
        let row = matching[0];
        assert_eq!(row.key_id.as_ref(), Some(key_id));
        assert_eq!(row.status.as_deref(), Some("failure"));
        assert_eq!(row.cost, Some(0.0));
        assert_eq!(row.requested_model.as_deref(), Some("chat"));
        assert!(row.prompt_tokens.is_none() && row.completion_tokens.is_none());
        let tags: Vec<String> = serde_json::from_str(row.tags.as_deref().unwrap()).unwrap();
        assert_eq!(tags, ["audit", "header-audit"]);
    }
    let listed: Value = client
        .get(proxy.url("/spend/logs"))
        .bearer_auth(MASTER_KEY)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(
        !listed.to_string().contains("private-prompt")
            && !listed.to_string().contains("private-upstream-detail")
    );
    assert_eq!(listed["total_cost"], 0.0);
    let global: Value = client
        .get(proxy.url("/global/spend"))
        .bearer_auth(MASTER_KEY)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(global["max_budget"], 10.0);
    let response = client
        .post(proxy.url("/v1/chat/completions"))
        .bearer_auth(keys[2]["key"].as_str().unwrap())
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let budget_spend: f64 = response.headers()["x-oxllm-key-spend"]
        .to_str()
        .unwrap()
        .parse()
        .unwrap();
    let rows = wait_for_rows(&proxy.db, 4).await;
    assert_eq!(rows.len(), 4);
    assert_eq!(
        rows.iter()
            .filter(|row| row.status.as_deref() == Some("failure"))
            .count(),
        3
    );
    let success = rows
        .iter()
        .find(|row| row.status.as_deref() == Some("success"))
        .unwrap();
    assert!((budget_spend - success.cost.unwrap()).abs() < 1e-12);
    let info: Value = client
        .get(proxy.url("/key/info"))
        .bearer_auth(MASTER_KEY)
        .query(&[("key_id", keys[2]["key_id"].as_str().unwrap())])
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!((info["spend"].as_f64().unwrap() - success.cost.unwrap()).abs() < 1e-12);
    let listed: Value = client
        .get(proxy.url("/key/list"))
        .bearer_auth(MASTER_KEY)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    for key in listed["keys"].as_array().unwrap() {
        let expected = if key["key_id"] == keys[2]["key_id"] {
            success.cost.unwrap()
        } else {
            0.0
        };
        assert!((key["spend"].as_f64().unwrap() - expected).abs() < 1e-12);
    }
    failed.assert_hits_async(1).await;
    healthy.assert_hits_async(1).await;
    let db = proxy.db.clone();
    proxy.shutdown().await;
    assert_eq!(db.query_spend(&Default::default()).await.unwrap().len(), 4);
}

#[tokio::test]
async fn malformed_top_level_tags_never_reach_the_provider_or_arm_cooldown() {
    let upstream = MockServer::start_async().await;
    let wire = upstream
        .mock_async(|when, then| {
            when.method(POST);
            then.status(200).json_body(reply());
        })
        .await;
    let proxy = start(config(&upstream)).await;
    for tags in [
        json!("one"),
        json!(1),
        json!(["one", 1]),
        json!({"one":true}),
    ] {
        let response = common::http_client().post(proxy.url("/v1/chat/completions")).bearer_auth(MASTER_KEY)
            .json(&json!({"model":"chat", "messages":[{"role":"user", "content":"hello"}], "tags":tags}))
            .send().await.unwrap();
        assert_eq!(response.status(), 400);
    }
    wire.assert_hits_async(0).await;
    let state = proxy.shared.load_full();
    let key = state.router.cooldown_key(&state.deployments[0]);
    assert!(!state.router.cooldown_store().is_cooling(&key));
    proxy.shutdown().await;
}

#[tokio::test]
async fn review_embedding_cache_hits_report_settled_key_spend() {
    let upstream = MockServer::start_async().await;
    let wire = upstream
        .mock_async(|when, then| {
            when.method(POST).path("/embeddings");
            then.status(200)
                .json_body(json!({"object":"list", "model":"text-embedding-3-small",
            "data":[{"object":"embedding", "index":0, "embedding":[0.1,0.2]}],
            "usage":{"prompt_tokens":5, "total_tokens":5}}));
        })
        .await;
    let mut config = config(&upstream);
    config.model_list[0].oxllm_params.model = "openai/text-embedding-3-small".into();
    config.model_list[0].oxllm_params.input_cost_per_token = Some(2e-6);
    config.oxllm_settings.cache = Some(serde_json::from_value(json!({"type":"memory"})).unwrap());
    let proxy = start(config).await;
    let key = generate_key(&proxy, json!({"max_budget":1.0})).await;
    let client = common::http_client();
    for cached in [false, true] {
        let response = client
            .post(proxy.url("/v1/embeddings"))
            .bearer_auth(key["key"].as_str().unwrap())
            .json(&json!({"model":"chat", "input":"hello", "cache":{"use_cache":true}}))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 200, "{}", response.text().await.unwrap());
        assert_eq!(response.headers().contains_key("x-oxllm-cache-hit"), cached);
        let spend: f64 = response
            .headers()
            .get("x-oxllm-key-spend")
            .expect("tracked embeddings, including cache hits, expose key spend")
            .to_str()
            .unwrap()
            .parse()
            .unwrap();
        assert!((spend - 1e-5).abs() < 1e-12);
        response.bytes().await.unwrap();
    }
    let updated = client
        .post(proxy.url("/key/update"))
        .bearer_auth(MASTER_KEY)
        .json(&json!({"key_id":key["key_id"], "clear_max_budget":true}))
        .send()
        .await
        .unwrap();
    assert_eq!(updated.status(), 200);
    let response = client
        .post(proxy.url("/v1/embeddings"))
        .bearer_auth(key["key"].as_str().unwrap())
        .json(&json!({"model":"chat", "input":"hello", "cache":{"use_cache":true}}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    assert!(response.headers().contains_key("x-oxllm-cache-hit"));
    assert!(
        response.headers().get("x-oxllm-key-spend").is_none(),
        "a retained counter must not masquerade as active budget tracking"
    );
    assert!(response.headers().get("x-litellm-key-spend").is_none());
    response.bytes().await.unwrap();
    wire.assert_hits_async(1).await;
    let db = proxy.db.clone();
    proxy.shutdown().await;
    let rows = db.query_spend(&Default::default()).await.unwrap();
    assert_eq!(rows.len(), 1, "cache hits must not create a second bill");
    assert!((rows[0].cost.unwrap() - 1e-5).abs() < 1e-12);
}

#[tokio::test]
async fn review_corrupt_model_policies_remain_manageable_and_fail_closed() {
    let upstream = MockServer::start_async().await;
    let wire = upstream
        .mock_async(|when, then| {
            when.method(POST).path("/chat/completions");
            then.status(200).json_body(reply());
        })
        .await;
    let mut config = config(&upstream);
    config.router_settings.routing.routing_strategy =
        Some(serde_json::from_value(json!("complexity_router")).unwrap());
    config.router_settings.complexity_router_config =
        Some(oxllm_core::router::ComplexityRouterConfig {
            tiers: std::collections::HashMap::from([("simple".into(), vec!["chat".into()])]),
            default_model: Some("chat".into()),
            ..Default::default()
        });
    let proxy = start(config).await;
    let client = common::http_client();
    for raw in ["{not json}", "gpt-4o, claude-*", "\"gpt-4o\"", "[1,2]"] {
        let key = generate_key(&proxy, json!({"models":["chat"]})).await;
        let id = key["key_id"].as_str().unwrap();
        let mut record = proxy.db.get_key(id).await.unwrap().unwrap();
        record.models = Some(raw.into());
        assert!(proxy.db.update_key(&record).await.unwrap());

        let info = client
            .get(proxy.url("/key/info"))
            .bearer_auth(MASTER_KEY)
            .query(&[("key_id", id)])
            .send()
            .await
            .unwrap();
        assert_eq!(info.status(), 200, "{raw}: {}", info.text().await.unwrap());
        let info: Value = info.json().await.unwrap();
        assert!(
            !info["models"]
                .as_array()
                .expect("typed deny-all policy")
                .is_empty()
        );
        assert!(info.get("key_hash").is_none());

        let listed = client
            .get(proxy.url("/key/list"))
            .bearer_auth(MASTER_KEY)
            .send()
            .await
            .unwrap();
        assert_eq!(
            listed.status(),
            200,
            "one corrupt row must not break the key list"
        );
        let listed: Value = listed.json().await.unwrap();
        let listed_key = listed["keys"]
            .as_array()
            .unwrap()
            .iter()
            .find(|record| record["key_id"] == id)
            .unwrap();
        assert_eq!(listed_key["models"], info["models"]);

        let updated = client
            .post(proxy.url("/key/update"))
            .bearer_auth(MASTER_KEY)
            .json(&json!({"key_id":id, "rpm_limit":7}))
            .send()
            .await
            .unwrap();
        assert_eq!(
            updated.status(),
            200,
            "an unrelated edit must return its committed result"
        );
        let updated: Value = updated.json().await.unwrap();
        assert_eq!(updated["models"], info["models"]);
        assert_eq!(updated["rpm"], 7);
        assert_eq!(
            proxy
                .db
                .get_key(id)
                .await
                .unwrap()
                .unwrap()
                .models
                .as_deref(),
            Some(raw)
        );

        for requested_model in ["chat", info["models"][0].as_str().unwrap()] {
            let denied = client.post(proxy.url("/v1/chat/completions"))
                .bearer_auth(key["key"].as_str().unwrap())
                .json(&json!({"model":requested_model, "messages":[{"role":"user","content":"hello"}]}))
                .send().await.unwrap();
            assert_eq!(
                denied.status(),
                403,
                "deny-all must reject its own marker before complexity routing"
            );
        }
        let repaired = client
            .post(proxy.url("/key/update"))
            .bearer_auth(MASTER_KEY)
            .json(&json!({"key_id":id, "models":["chat"]}))
            .send()
            .await
            .unwrap();
        assert_eq!(repaired.status(), 200);
        let allowed = client
            .post(proxy.url("/v1/chat/completions"))
            .bearer_auth(key["key"].as_str().unwrap())
            .json(&json!({"model":"chat", "messages":[{"role":"user","content":"hello"}]}))
            .send()
            .await
            .unwrap();
        assert_eq!(
            allowed.status(),
            200,
            "an explicit repair restores the intended ACL"
        );
    }
    wire.assert_hits_async(4).await;
    proxy.shutdown().await;
}

#[tokio::test]
async fn review_early_denials_keep_header_tags_and_request_identity() {
    let upstream = MockServer::start_async().await;
    let wire = upstream
        .mock_async(|when, then| {
            when.method(POST);
            then.status(200).json_body(reply());
        })
        .await;
    let proxy = start(config(&upstream)).await;
    let client = common::http_client();
    for (path, body) in [
        ("/customer/new", json!({"end_user_id":"blocked-customer"})),
        (
            "/customer/update",
            json!({"end_user_id":"blocked-customer", "blocked":true}),
        ),
    ] {
        let response = client
            .post(proxy.url(path))
            .bearer_auth(MASTER_KEY)
            .json(&body)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
    }
    let restricted = generate_key(&proxy, json!({"models":["other"]})).await;
    let mut expected_ids = Vec::new();
    for (secret, user, paths) in [
        (
            MASTER_KEY,
            Some("blocked-customer"),
            vec![
                "/v1/chat/completions",
                "/v1/embeddings",
                "/v1/images/generations",
                "/openai/v1/chat/completions",
            ],
        ),
        (
            restricted["key"].as_str().unwrap(),
            None,
            vec![
                "/v1/images/generations",
                "/v1/rerank",
                "/openai/v1/chat/completions",
            ],
        ),
    ] {
        for path in paths {
            let response = client
                .post(proxy.url(path))
                .bearer_auth(secret)
                .header("x-oxllm-tag", "header,body")
                .json(
                    &json!({"model":"chat", "user":user, "input":"hello", "prompt":"hello",
                    "messages":[{"role":"user","content":"hello"}], "tags":["body"]}),
                )
                .send()
                .await
                .unwrap();
            assert_eq!(
                response.status(),
                403,
                "{path}: {}",
                response.text().await.unwrap()
            );
            expected_ids.push(
                response.headers()["x-request-id"]
                    .to_str()
                    .unwrap()
                    .to_owned(),
            );
        }
    }
    let allowed = generate_key(&proxy, json!({"models":["chat"]})).await;
    for path in ["/v1/batches", "/openai/v1/batches"] {
        let response = client
            .post(proxy.url(path))
            .bearer_auth(allowed["key"].as_str().unwrap())
            .header("x-oxllm-tag", "header")
            .json(&json!({"model":"chat", "tags":["body"],
                "requests":[{"params":{"model":"other"}}]}))
            .send()
            .await
            .unwrap();
        assert_eq!(
            response.status(),
            403,
            "nested model ACLs: {path}: {}",
            response.text().await.unwrap()
        );
        expected_ids.push(
            response.headers()["x-request-id"]
                .to_str()
                .unwrap()
                .to_owned(),
        );
    }
    wire.assert_hits_async(0).await;
    let db = proxy.db.clone();
    proxy.shutdown().await;
    let rows = db.query_spend(&Default::default()).await.unwrap();
    assert_eq!(rows.len(), expected_ids.len());
    for id in expected_ids {
        let matching = rows
            .iter()
            .filter(|row| row.request_id.as_deref() == Some(id.as_str()))
            .collect::<Vec<_>>();
        assert_eq!(matching.len(), 1, "each denial is attributed exactly once");
        let row = matching[0];
        assert_eq!(row.status.as_deref(), Some("failure"));
        assert_eq!(row.cost, Some(0.0));
        assert_eq!(
            serde_json::from_str::<Value>(row.tags.as_deref().unwrap()).unwrap(),
            json!(["body", "header"])
        );
    }
}

#[tokio::test]
async fn review_rerank_and_raw_json_validate_and_strip_gateway_tags() {
    let upstream = MockServer::start_async().await;
    let wire = upstream
        .mock_async(|when, then| {
            when.method(POST).path("/v1/rerank").matches(|request| {
                request
                    .body
                    .as_ref()
                    .and_then(|body| serde_json::from_slice::<Value>(body).ok())
                    .is_some_and(|body| {
                        body.get("tags").is_none()
                            && body.pointer("/metadata/tags").is_none()
                            && body.get("native_extension").is_none()
                    })
            });
            then.status(200)
                .json_body(json!({"id":"ranked", "results":[{"index":0,"relevance_score":0.9}]}));
        })
        .await;
    let raw_wire = upstream
        .mock_async(|when, then| {
            when.method(POST).path("/v1/rerank").matches(|request| {
                request
                    .body
                    .as_ref()
                    .and_then(|body| serde_json::from_slice::<Value>(body).ok())
                    .is_some_and(|body| {
                        body.get("tags").is_none()
                            && body.pointer("/metadata/tags").is_none()
                            && body["native_extension"] == "kept"
                            && body["metadata"]["user_id"] == "kept"
                    })
            });
            then.status(200).json_body(
                json!({"id":"raw-ranked", "results":[{"index":0,"relevance_score":0.9}]}),
            );
        })
        .await;
    let mut config = config(&upstream);
    config.model_list[0].oxllm_params.model = "cohere/rerank-english-v3.0".into();
    let proxy = start(config).await;
    let client = common::http_client();
    for path in ["/v1/rerank", "/cohere/v1/rerank"] {
        for tags in [json!("one"), json!(["one", 1]), json!({"one":true})] {
            let response = client
                .post(proxy.url(path))
                .bearer_auth(MASTER_KEY)
                .json(&json!({"model":"chat", "query":"hello", "documents":["one"], "tags":tags}))
                .send()
                .await
                .unwrap();
            assert_eq!(
                response.status(),
                400,
                "{path}: {}",
                response.text().await.unwrap()
            );
        }
        let response = client
            .post(proxy.url(path))
            .bearer_auth(MASTER_KEY)
            .header("x-oxllm-tag", "header")
            .json(
                &json!({"model":"chat", "query":"hello", "documents":["one"],
                "native_extension":"kept", "tags":["body"], "metadata":{"tags":["meta"],"user_id":"kept"}}),
            )
            .send()
            .await
            .unwrap();
        assert_eq!(
            response.status(),
            200,
            "{path}: {}",
            response.text().await.unwrap()
        );
    }
    wire.assert_hits_async(1).await;
    raw_wire.assert_hits_async(1).await;
    let db = proxy.db.clone();
    proxy.shutdown().await;
    let rows = db.query_spend(&Default::default()).await.unwrap();
    assert_eq!(rows.len(), 2);
    for row in rows {
        assert_eq!(
            serde_json::from_str::<Value>(row.tags.as_deref().unwrap()).unwrap(),
            json!(["body", "meta", "header"])
        );
    }
}

#[tokio::test]
async fn review_key_json_validation_preserves_transport_statuses() {
    let upstream = MockServer::start_async().await;
    let proxy = start(config(&upstream)).await;
    let client = common::http_client();
    let created = generate_key(&proxy, json!({"rpm":5})).await;
    for path in ["/key/generate", "/key/update"] {
        let response = client
            .post(proxy.url(path))
            .bearer_auth(MASTER_KEY)
            .body("{}")
            .send()
            .await
            .unwrap();
        assert_eq!(
            response.status(),
            415,
            "unsupported content type remains a transport error"
        );
        let invalid_data = if path == "/key/update" {
            json!({"key_id":created["key_id"], "rpm":"bad"})
        } else {
            json!({"rpm":"bad"})
        };
        for raw in ["{".to_owned(), invalid_data.to_string()] {
            let response = client
                .post(proxy.url(path))
                .bearer_auth(MASTER_KEY)
                .header("content-type", "application/json")
                .body(raw)
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), 400);
        }
    }
    assert_eq!(proxy.db.list_keys(None).await.unwrap().len(), 1);
    assert_eq!(
        proxy
            .db
            .get_key(created["key_id"].as_str().unwrap())
            .await
            .unwrap()
            .unwrap()
            .rpm,
        Some(5)
    );
    proxy.shutdown().await;
}

#[tokio::test]
async fn review_explicit_client_status_policies_still_count_and_cool() {
    for (path, native_path) in [
        ("/v1/chat/completions", "/chat/completions"),
        ("/v1/embeddings", "/embeddings"),
    ] {
        let upstream = MockServer::start_async().await;
        let wire = upstream
            .mock_async(|when, then| {
                when.method(POST).path(native_path);
                then.status(400)
                    .json_body(json!({"error":{"message":"operator opted into this status"}}));
            })
            .await;
        let mut config = config(&upstream);
        config.router_settings.routing.allowed_fails_policy =
            Some(std::collections::HashMap::from([("400".into(), 1)]));
        config.router_settings.routing.cooldown_time = Some(60);
        let proxy = start(config).await;
        let state = proxy.shared.load_full();
        let deployment = &state.deployments[0];
        let cooldown_key = state.router.cooldown_key(deployment);
        let body = json!({"model":"chat", "input":"hello", "messages":[{"role":"user","content":"hello"}]});
        for failures in 1..=2 {
            let response = common::http_client()
                .post(proxy.url(path))
                .bearer_auth(MASTER_KEY)
                .json(&body)
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), 400);
            assert_eq!(
                state
                    .router
                    .cooldown_store()
                    .failure_count(&cooldown_key, 400),
                failures
            );
            assert_eq!(
                state.router.cooldown_store().is_cooling(&cooldown_key),
                failures > 1
            );
            assert_eq!(state.router.usage_store().in_flight(&cooldown_key), 0);
        }
        let response = common::http_client()
            .post(proxy.url(path))
            .bearer_auth(MASTER_KEY)
            .json(&body)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 429);
        wire.assert_hits_async(2).await;
        proxy.shutdown().await;
    }
}

#[tokio::test]
async fn review_key_spend_headers_cover_bridges_cache_and_omission() {
    let upstream = MockServer::start_async().await;
    let unary = upstream
        .mock_async(|when, then| {
            when.method(POST)
                .path("/chat/completions")
                .matches(|request| {
                    request
                        .body
                        .as_ref()
                        .and_then(|body| serde_json::from_slice::<Value>(body).ok())
                        .is_some_and(|body| {
                            body.get("stream").and_then(Value::as_bool) != Some(true)
                        })
                });
            then.status(200).json_body(reply());
        })
        .await;
    let streamed = upstream.mock_async(|when, then| {
        when.method(POST).path("/chat/completions")
            .json_body_partial(json!({"stream":true}).to_string());
        then.status(200).header("content-type", "text/event-stream").body(format!(
            "data: {}\n\ndata: {}\n\ndata: [DONE]\n\n",
            json!({"model":"gpt-4o", "choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":null}]}),
            json!({"model":"gpt-4o", "choices":[], "usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}})
        ));
    }).await;
    let proxy = start(config(&upstream)).await;
    let tracked = generate_key(&proxy, json!({"max_budget":1.0})).await;
    let unlimited = generate_key(&proxy, json!({})).await;
    let client = common::http_client();
    let mut cumulative = 0.0;
    for path in [
        "/v1/chat/completions",
        "/v1/messages",
        "/v1/responses",
        "/v1/completions",
    ] {
        let body = json!({"model":"chat", "max_tokens":5, "cache":{"use_cache":true},
            "messages":[{"role":"user","content":path}], "input":path, "prompt":path});
        let mut settled = None;
        for cached in [false, true] {
            let response = client
                .post(proxy.url(path))
                .bearer_auth(tracked["key"].as_str().unwrap())
                .json(&body)
                .send()
                .await
                .unwrap();
            assert_eq!(
                response.status(),
                200,
                "{path}: {}",
                response.text().await.unwrap()
            );
            assert_eq!(response.headers().contains_key("x-oxllm-cache-hit"), cached);
            let spend: f64 = response.headers()["x-oxllm-key-spend"]
                .to_str()
                .unwrap()
                .parse()
                .unwrap();
            assert!(spend > 0.0);
            if cached {
                assert_eq!(
                    Some(spend),
                    settled,
                    "cached replies preserve the tracked counter"
                );
            } else {
                let cost: f64 = response.headers()["x-oxllm-response-cost"]
                    .to_str()
                    .unwrap()
                    .parse()
                    .unwrap();
                cumulative += cost;
                assert!(
                    (spend - cumulative).abs() < 1e-12,
                    "the snapshot includes every settled request in the window"
                );
                settled = Some(spend);
            }
            assert_eq!(
                response.headers().get("x-litellm-key-spend"),
                response.headers().get("x-oxllm-key-spend")
            );
            response.bytes().await.unwrap();
        }
    }
    let body =
        json!({"model":"chat", "max_tokens":5, "messages":[{"role":"user","content":"hello"}]});
    for secret in [MASTER_KEY, unlimited["key"].as_str().unwrap()] {
        let response = client
            .post(proxy.url("/v1/chat/completions"))
            .bearer_auth(secret)
            .json(&body)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        assert!(response.headers().get("x-oxllm-key-spend").is_none());
        assert!(response.headers().get("x-litellm-key-spend").is_none());
        response.bytes().await.unwrap();
    }
    let mut stream_body = body;
    stream_body["stream"] = json!(true);
    stream_body["stream_options"] = json!({"include_usage":true});
    let response = client
        .post(proxy.url("/v1/chat/completions"))
        .bearer_auth(tracked["key"].as_str().unwrap())
        .json(&stream_body)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    assert!(response.headers().get("x-oxllm-key-spend").is_none());
    assert!(response.headers().get("x-litellm-key-spend").is_none());
    assert!(response.text().await.unwrap().contains("[DONE]"));
    let global: Value = client
        .get(proxy.url("/global/spend"))
        .bearer_auth(MASTER_KEY)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(global.get("max_budget"), Some(&Value::Null));
    unary.assert_hits_async(6).await;
    streamed.assert_hits_async(1).await;
    let db = proxy.db.clone();
    proxy.shutdown().await;
    assert_eq!(db.query_spend(&Default::default()).await.unwrap().len(), 7);
}

#[tokio::test]
async fn review_team_model_deltas_do_not_repair_corrupt_policies_implicitly() {
    let upstream = MockServer::start_async().await;
    let wire = upstream
        .mock_async(|when, then| {
            when.method(POST).path("/chat/completions");
            then.status(200).json_body(reply());
        })
        .await;
    let proxy = start(config(&upstream)).await;
    let client = common::http_client();
    for raw in [
        "not-json",
        "[1, *]",
        "[1,2]",
        "[\"\\u0000oxllm-acl-deny-all\"]",
    ] {
        let response = client
            .post(proxy.url("/team/add"))
            .bearer_auth(MASTER_KEY)
            .json(&json!({"team_name":"protected", "models":["chat"]}))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let created: Value = response.json().await.unwrap();
        let id = created["team_id"].as_str().unwrap();
        let mut team = proxy.db.get_team(id).await.unwrap().unwrap();
        team.models = Some(raw.into());
        proxy.db.update_team(&team).await.unwrap();
        let key = generate_key(&proxy, json!({"team_id":id})).await;
        for path in ["/team/model/add", "/team/model/delete"] {
            let response = client
                .post(proxy.url(path))
                .bearer_auth(MASTER_KEY)
                .json(&json!({"team_id":id, "models":["chat"]}))
                .send()
                .await
                .unwrap();
            assert_eq!(
                response.status(),
                400,
                "{path} must require an explicit policy replacement"
            );
            assert_eq!(
                proxy
                    .db
                    .get_team(id)
                    .await
                    .unwrap()
                    .unwrap()
                    .models
                    .as_deref(),
                Some(raw)
            );
            let denied = client
                .post(proxy.url("/v1/chat/completions"))
                .bearer_auth(key["key"].as_str().unwrap())
                .json(&json!({"model":"chat", "messages":[{"role":"user","content":"hello"}]}))
                .send()
                .await
                .unwrap();
            assert_eq!(
                denied.status(),
                403,
                "incremental edits must not weaken deny-all"
            );
        }
        let repaired = client
            .post(proxy.url("/team/update"))
            .bearer_auth(MASTER_KEY)
            .json(&json!({"team_id":id, "models":["chat"]}))
            .send()
            .await
            .unwrap();
        assert_eq!(repaired.status(), 200);
        let allowed = client
            .post(proxy.url("/v1/chat/completions"))
            .bearer_auth(key["key"].as_str().unwrap())
            .json(&json!({"model":"chat", "messages":[{"role":"user","content":"hello"}]}))
            .send()
            .await
            .unwrap();
        assert_eq!(allowed.status(), 200);
    }
    wire.assert_hits_async(4).await;
    proxy.shutdown().await;
}

#[tokio::test]
async fn review_corrupt_customer_blocklists_block_all_until_replaced() {
    let upstream = MockServer::start_async().await;
    let wire = upstream
        .mock_async(|when, then| {
            when.method(POST).path("/chat/completions");
            then.status(200).json_body(reply());
        })
        .await;
    let proxy = start(config(&upstream)).await;
    let client = common::http_client();
    for (index, raw) in ["not-json", "[1, *]", "[1,2]"].into_iter().enumerate() {
        let id = format!("customer-{index}");
        let created = client
            .post(proxy.url("/customer/new"))
            .bearer_auth(MASTER_KEY)
            .json(&json!({"end_user_id":id}))
            .send()
            .await
            .unwrap();
        assert_eq!(created.status(), 200);
        let mut record = proxy.db.get_customer(&id).await.unwrap().unwrap();
        record.model_blocklist = Some(raw.into());
        proxy.db.update_customer(&record).await.unwrap();
        let body = json!({"model":"chat", "user":format!("  {id}  "),
            "messages":[{"role":"user","content":"hello"}]});
        let denied = client
            .post(proxy.url("/v1/chat/completions"))
            .bearer_auth(MASTER_KEY)
            .json(&body)
            .send()
            .await
            .unwrap();
        assert_eq!(
            denied.status(),
            403,
            "an unreadable customer abuse policy must fail closed"
        );
        let repaired = client
            .post(proxy.url("/customer/update"))
            .bearer_auth(MASTER_KEY)
            .json(&json!({"end_user_id":id, "model_blocklist":["other"]}))
            .send()
            .await
            .unwrap();
        assert_eq!(repaired.status(), 200);
        let allowed = client
            .post(proxy.url("/v1/chat/completions"))
            .bearer_auth(MASTER_KEY)
            .json(&body)
            .send()
            .await
            .unwrap();
        assert_eq!(allowed.status(), 200);
    }
    wire.assert_hits_async(3).await;
    proxy.shutdown().await;
}

#[tokio::test]
async fn review_end_user_attribution_is_trimmed_before_success_or_acl_denial() {
    let upstream = MockServer::start_async().await;
    upstream
        .mock_async(|when, then| {
            when.method(POST).path("/chat/completions");
            then.status(200).json_body(reply());
        })
        .await;
    let proxy = start(config(&upstream)).await;
    let denied = generate_key(&proxy, json!({"models":["other"]})).await;
    let client = common::http_client();
    let mut expected = Vec::new();
    for (secret, status) in [(MASTER_KEY, 200), (denied["key"].as_str().unwrap(), 403)] {
        for (user, normalized) in [("  alice  ", Some("alice")), (" \t ", None)] {
            let response = client
                .post(proxy.url("/v1/chat/completions"))
                .bearer_auth(secret)
                .json(&json!({"model":"chat", "user":user,
                    "messages":[{"role":"user","content":"hello"}]}))
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), status);
            expected.push((
                response.headers()["x-request-id"]
                    .to_str()
                    .unwrap()
                    .to_owned(),
                normalized,
            ));
            response.bytes().await.unwrap();
        }
    }
    let db = proxy.db.clone();
    proxy.shutdown().await;
    let rows = db.query_spend(&Default::default()).await.unwrap();
    assert_eq!(rows.len(), expected.len());
    for (request_id, user) in expected {
        let row = rows
            .iter()
            .find(|row| row.request_id.as_deref() == Some(request_id.as_str()))
            .unwrap();
        assert_eq!(row.end_user_id.as_deref(), user);
    }
}

#[tokio::test]
async fn review_realtime_preupgrade_rejections_are_audited_without_upstream_calls() {
    use tokio_tungstenite::tungstenite::client::IntoClientRequest;
    for wildcard in [false, true] {
        let upstream = MockServer::start_async().await;
        let wire = upstream
            .mock_async(|_, then| {
                then.status(500);
            })
            .await;
        let mut config = config(&upstream);
        if wildcard {
            config.model_list[0].model_name = "*".into();
            config.model_list[0].oxllm_params.model = "openai/*".into();
        } else {
            config.router_settings.routing.routing_strategy =
                Some(serde_json::from_value(json!("tag_based")).unwrap());
        }
        let proxy = start(config).await;
        let cases = if wildcard {
            vec![("*", None, 400)]
        } else {
            vec![("missing", None, 400), ("chat", Some("&missing"), 400)]
        };
        let mut ids = Vec::new();
        for (model, tag, status) in cases {
            let mut url = reqwest::Url::parse(&proxy.url("/v1/realtime")).unwrap();
            url.set_scheme("ws").unwrap();
            url.query_pairs_mut().append_pair("model", model);
            let mut request = url.as_str().into_client_request().unwrap();
            request.headers_mut().insert(
                "authorization",
                format!("Bearer {MASTER_KEY}").parse().unwrap(),
            );
            request
                .headers_mut()
                .insert("x-litellm-session-id", "setup-session".parse().unwrap());
            if let Some(tag) = tag {
                request
                    .headers_mut()
                    .insert("x-oxllm-tag", tag.parse().unwrap());
            }
            let error = tokio_tungstenite::connect_async(request)
                .await
                .expect_err("pre-upgrade rejection");
            let tokio_tungstenite::tungstenite::Error::Http(response) = error else {
                panic!("expected an HTTP rejection: {error}")
            };
            assert_eq!(response.status().as_u16(), status);
            assert_eq!(
                response.headers().get("x-litellm-call-id"),
                response.headers().get("x-request-id")
            );
            ids.push(
                response.headers()["x-request-id"]
                    .to_str()
                    .unwrap()
                    .to_owned(),
            );
        }
        wire.assert_hits_async(0).await;
        let state = proxy.shared.load_full();
        for deployment in &state.deployments {
            assert_eq!(
                state
                    .router
                    .usage_store()
                    .in_flight(&state.router.cooldown_key(deployment)),
                0
            );
        }
        let db = proxy.db.clone();
        proxy.shutdown().await;
        let rows = db.query_spend(&Default::default()).await.unwrap();
        assert_eq!(rows.len(), ids.len());
        for id in ids {
            let row = rows
                .iter()
                .find(|row| row.request_id.as_deref() == Some(id.as_str()))
                .unwrap();
            assert_eq!(row.status.as_deref(), Some("failure"));
            assert_eq!(row.session_id.as_deref(), Some("setup-session"));
            assert_eq!(row.cost, Some(0.0));
            assert!(row.total_tokens.is_none());
        }
    }
}

#[tokio::test]
async fn review_admin_policy_replays_do_not_open_corrupt_team_or_user_scopes() {
    let upstream = MockServer::start_async().await;
    let wire = upstream
        .mock_async(|when, then| {
            when.method(POST).path("/chat/completions");
            then.status(200).json_body(reply());
        })
        .await;
    let proxy = start(config(&upstream)).await;
    let client = common::http_client();
    for (scope, id_field, name_field) in [
        ("team", "team_id", "team_name"),
        ("user", "user_id", "user_name"),
    ] {
        let mut body = json!({"models":["chat"]});
        body[name_field] = json!("protected");
        let response = client
            .post(proxy.url(&format!("/{scope}/add")))
            .bearer_auth(MASTER_KEY)
            .json(&body)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let created: Value = response.json().await.unwrap();
        let id = created[id_field].as_str().unwrap();
        if scope == "team" {
            let mut row = proxy.db.get_team(id).await.unwrap().unwrap();
            row.models = Some("[1, *]".into());
            proxy.db.update_team(&row).await.unwrap();
        } else {
            let mut row = proxy.db.get_user(id).await.unwrap().unwrap();
            row.models = Some("[1, *]".into());
            proxy.db.update_user(&row).await.unwrap();
        }
        let mut policy = json!({});
        policy[id_field] = created[id_field].clone();
        let key = generate_key(&proxy, policy).await;
        let mut update = json!({"max_budget":0.5, "models":["\0oxllm-acl-deny-all"]});
        update[id_field] = created[id_field].clone();
        let updated = client
            .post(proxy.url(&format!("/{scope}/update")))
            .bearer_auth(MASTER_KEY)
            .json(&update)
            .send()
            .await
            .unwrap();
        assert_eq!(updated.status(), 200);
        let request = json!({"model":"chat", "messages":[{"role":"user","content":"hello"}]});
        let denied = client
            .post(proxy.url("/v1/chat/completions"))
            .bearer_auth(key["key"].as_str().unwrap())
            .json(&request)
            .send()
            .await
            .unwrap();
        assert_eq!(
            denied.status(),
            403,
            "an unrelated admin edit preserves effective deny-all"
        );
        update["models"] = json!(["chat"]);
        let repaired = client
            .post(proxy.url(&format!("/{scope}/update")))
            .bearer_auth(MASTER_KEY)
            .json(&update)
            .send()
            .await
            .unwrap();
        assert_eq!(repaired.status(), 200);
        let allowed = client
            .post(proxy.url("/v1/chat/completions"))
            .bearer_auth(key["key"].as_str().unwrap())
            .json(&request)
            .send()
            .await
            .unwrap();
        assert_eq!(allowed.status(), 200);
    }
    wire.assert_hits_async(2).await;
    proxy.shutdown().await;
}

#[tokio::test]
async fn review_unchanged_literal_policies_are_not_trimmed_into_wildcards_on_save() {
    let upstream = MockServer::start_async().await;
    let wire = upstream
        .mock_async(|when, then| {
            when.method(POST).path("/chat/completions");
            then.status(200).json_body(reply());
        })
        .await;
    let proxy = start(config(&upstream)).await;
    let client = common::http_client();
    for (scope, id_field, name_field) in [
        ("key", "key_id", ""),
        ("team", "team_id", "team_name"),
        ("user", "user_id", "user_name"),
    ] {
        let key;
        let id;
        let mut update = json!({"max_budget":0.5, "models":[" * "]});
        if scope == "key" {
            key = generate_key(&proxy, json!({})).await;
            id = key[id_field].as_str().unwrap().to_owned();
            let mut row = proxy.db.get_key(&id).await.unwrap().unwrap();
            row.models = Some("[\" * \"]".into());
            proxy.db.update_key(&row).await.unwrap();
        } else {
            let mut create = json!({});
            create[name_field] = json!("literal-policy");
            let response = client
                .post(proxy.url(&format!("/{scope}/add")))
                .bearer_auth(MASTER_KEY)
                .json(&create)
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), 200);
            let created: Value = response.json().await.unwrap();
            id = created[id_field].as_str().unwrap().to_owned();
            if scope == "team" {
                let mut row = proxy.db.get_team(&id).await.unwrap().unwrap();
                row.models = Some("[\" * \"]".into());
                proxy.db.update_team(&row).await.unwrap();
            } else {
                let mut row = proxy.db.get_user(&id).await.unwrap().unwrap();
                row.models = Some("[\" * \"]".into());
                proxy.db.update_user(&row).await.unwrap();
            }
            let mut policy = json!({});
            policy[id_field] = json!(id);
            key = generate_key(&proxy, policy).await;
        }
        update[id_field] = json!(id);
        let request = json!({"model":"chat", "messages":[{"role":"user","content":"hello"}]});
        for saved in [false, true] {
            if saved {
                let response = client
                    .post(proxy.url(&format!("/{scope}/update")))
                    .bearer_auth(MASTER_KEY)
                    .json(&update)
                    .send()
                    .await
                    .unwrap();
                assert_eq!(response.status(), 200);
            }
            let denied = client
                .post(proxy.url("/v1/chat/completions"))
                .bearer_auth(key["key"].as_str().unwrap())
                .json(&request)
                .send()
                .await
                .unwrap();
            assert_eq!(
                denied.status(),
                403,
                "{scope}: saving an unchanged literal policy must not broaden it"
            );
        }
        update["models"] = json!(["chat"]);
        let repaired = client
            .post(proxy.url(&format!("/{scope}/update")))
            .bearer_auth(MASTER_KEY)
            .json(&update)
            .send()
            .await
            .unwrap();
        assert_eq!(repaired.status(), 200);
        let allowed = client
            .post(proxy.url("/v1/chat/completions"))
            .bearer_auth(key["key"].as_str().unwrap())
            .json(&request)
            .send()
            .await
            .unwrap();
        assert_eq!(allowed.status(), 200);
    }
    wire.assert_hits_async(3).await;
    proxy.shutdown().await;
}
