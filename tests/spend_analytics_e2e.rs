//! Integration tests: spend logs pagination and global spend/activity analytics.

use std::sync::Arc;

use httpmock::MockServer;
use oxllm_server::app::{build_router, build_shared_state_with_db};
use oxllm_server::config::types::{ConfigModel, GeneralSettings, OxllmSettings, ProxyConfig};
use oxllm_server::spend_writer;
use oxllm_server::{Database, OxllmParams, SharedState};
use serde_json::{Value, json};

const MASTER_KEY: &str = "sk-master-key-spend-test";

struct OxllmHandle {
    addr: std::net::SocketAddr,
    shutdown_tx: tokio::sync::oneshot::Sender<()>,
    db: Arc<Database>,
}

impl OxllmHandle {
    fn url(&self, path: &str) -> String {
        format!("http://{}{path}", self.addr)
    }
}

fn proxy_config(mock_base_url: &str) -> ProxyConfig {
    ProxyConfig {
        model_list: vec![ConfigModel {
            model_name: "gpt-4o".to_string(),
            oxllm_params: OxllmParams {
                model: "openai/gpt-4o".to_string(),
                api_key: Some("sk-upstream".to_string()),
                api_base: Some(mock_base_url.to_string()),
                timeout: Some(30.0),
                stream_timeout: Some(30),
                ..Default::default()
            },
            ..Default::default()
        }],
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

async fn fresh_db() -> Arc<Database> {
    let db = Database::connect("sqlite::memory:")
        .await
        .expect("connect sqlite memory");
    db.migrate().await.expect("migrate");
    db
}

async fn start_oxllm(config: ProxyConfig) -> OxllmHandle {
    let db = fresh_db().await;
    let writer = spend_writer::spawn(db.clone(), 1);
    let shared: SharedState =
        build_shared_state_with_db(config, Some(writer.sender()), Some(db.clone()))
            .expect("build state");
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let addr = listener.local_addr().expect("addr");
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
    OxllmHandle {
        addr,
        shutdown_tx,
        db,
    }
}

fn mock_chat(mock: &MockServer) -> httpmock::Mock<'_> {
    mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .header("authorization", "Bearer sk-upstream");
        then.status(200)
            .header("content-type", "application/json")
            .json_body(json!({
            "id": "chatcmpl-7",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-4o",
            "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "hi"},
            "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
            }));
    })
}

fn mock_chat_never(mock: &MockServer) -> httpmock::Mock<'_> {
    mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions");
        then.status(500);
    })
}

async fn shutdown(handle: OxllmHandle) {
    let _ = handle.shutdown_tx.send(());
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
}

async fn wait_for_spend_count(db: &Database, target: usize) {
    for _ in 0..60 {
        let logs = db
            .query_spend(&oxllm_server::SpendFilter::default())
            .await
            .expect("query");
        if logs.len() >= target {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    let logs = db
        .query_spend(&oxllm_server::SpendFilter::default())
        .await
        .expect("final query");
    panic!(
        "spend log count never reached {target}; have {} logs",
        logs.len()
    );
}

fn seeded_log(timestamp: &str, cost: f64, tags: Option<String>) -> oxllm_server::SpendLog {
    oxllm_server::SpendLog {
        deployment_id: None,
        requested_model: None,
        model_group: None,
        id: None,
        timestamp: timestamp.into(),
        request_id: None,
        model: "gpt-4o".into(),
        provider: "openai".into(),
        prompt_tokens: Some(10),
        completion_tokens: Some(5),
        total_tokens: Some(15),
        cost: Some(cost),
        key_id: None,
        team_id: None,
        user_id: None,
        end_user_id: None,
        tag_name: None,
        tags,
        status: Some("success".into()),
        latency_ms: Some(1),
    }
}

#[tokio::test]
async fn azure_model_router_prices_the_served_model_plus_one_fee_for_json_and_sse() {
    let fee_rate = oxllm_core::cost::price_table()["azure_ai/model-router"].input_cost_per_token;
    let usage = oxllm_core::UsageSummary {
        prompt_tokens: 100,
        completion_tokens: 20,
        total_tokens: 120,
        cache_read_input_tokens: Some(40),
        ..Default::default()
    };
    for provider in ["azure", "azure_ai"] {
        for stream in [false, true] {
            let mock = MockServer::start_async().await;
            let wire = mock.mock_async(|when, then| {
                when.method(httpmock::Method::POST)
                    .path(if provider == "azure" { "/openai/v1/chat/completions" } else { "/chat/completions" })
                    .json_body_partial(json!({"model":"production-router"}).to_string());
                if stream {
                    then.status(200).header("content-type", "text/event-stream").body(format!(
                        "data: {}\n\ndata: {}\n\ndata: [DONE]\n\n",
                        json!({"id":"routed", "object":"chat.completion.chunk", "model":"gpt-6.1-sol", "choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":null}]}),
                        json!({"model":"gpt-6.1-sol", "choices":[], "usage":{"prompt_tokens":100,"completion_tokens":20,"total_tokens":120,"prompt_tokens_details":{"cached_tokens":40}}})
                    ));
                } else {
                    then.status(200).json_body(json!({
                        "id":"routed", "object":"chat.completion", "model":"gpt-6.1-sol",
                        "choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],
                        "usage":{"prompt_tokens":100,"completion_tokens":20,"total_tokens":120,"prompt_tokens_details":{"cached_tokens":40}}
                    }));
                }
            }).await;
            let mut config = proxy_config(&mock.base_url());
            config.model_list[0].model_name = "public-router".into();
            let params = &mut config.model_list[0].oxllm_params;
            params.model = format!("{provider}/model_router/production-router");
            params.api_version = (provider == "azure").then(|| "v1".into());
            params
                .extra
                .insert("base_model".into(), json!("model-router"));
            let handle = start_oxllm(config).await;
            let expected =
                oxllm_core::calculate_cost_with_provider("gpt-6.1-sol", &usage, provider).unwrap()
                    + usage.prompt_tokens as f64 * fee_rate;
            let response = reqwest::Client::new()
                .post(handle.url("/v1/chat/completions"))
                .bearer_auth(MASTER_KEY)
                .json(&json!({
                    "model":"public-router", "stream":stream, "max_tokens":32,
                    "messages":[{"role":"user","content":"hello"}]
                }))
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
                assert!((cost - expected).abs() < 1e-12);
            }
            let text = response.text().await.unwrap();
            assert!(text.contains("public-router"), "{text}");
            wait_for_spend_count(&handle.db, 1).await;
            let rows = handle.db.query_spend(&Default::default()).await.unwrap();
            assert_eq!(rows.len(), 1);
            assert_eq!(rows[0].provider, provider);
            assert!((rows[0].cost.unwrap() - expected).abs() < 1e-12);
            wire.assert_hits_async(1).await;
            shutdown(handle).await;
        }
    }
}

#[tokio::test]
async fn every_spend_endpoint_includes_the_final_nanosecond_of_a_bare_end_date() {
    let mock = MockServer::start();
    let handle = start_oxllm(proxy_config(&mock.base_url())).await;
    for (timestamp, cost) in [
        ("2026-07-31T12:00:00Z", 1.0),
        ("2026-07-31T23:59:59.999000001Z", 2.0),
        ("2026-07-31T23:59:59.999999999Z", 4.0),
        ("2026-08-01T00:00:00Z", 8.0),
    ] {
        handle
            .db
            .insert_spend_log(&seeded_log(timestamp, cost, None))
            .await
            .unwrap();
    }
    let client = reqwest::Client::new();
    for endpoint in [
        "/spend/logs",
        "/spend/logs/v2",
        "/global/spend",
        "/spend/keys",
        "/spend/by-model",
        "/spend/users",
        "/spend/teams",
        "/spend/organizations",
        "/spend/tags",
    ] {
        let response = client
            .get(handle.url(endpoint))
            .bearer_auth(MASTER_KEY)
            .query(&[("start_date", "2026-07-31"), ("end_date", "2026-07-31")])
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 200, "{endpoint}");
        let body: Value = response.json().await.unwrap();
        if let Some(total) = body["total_cost"].as_f64() {
            assert_eq!(total, 7.0, "{endpoint}: {body}");
            assert_eq!(body["total_requests"], 3, "{endpoint}: {body}");
        } else {
            let rows = body["data"].as_array().unwrap();
            assert_eq!(
                rows.iter()
                    .map(|row| row["total_spend"].as_f64().unwrap())
                    .sum::<f64>(),
                7.0,
                "{endpoint}: {body}"
            );
            assert_eq!(
                rows.iter()
                    .map(|row| row["total_requests"].as_i64().unwrap())
                    .sum::<i64>(),
                3
            );
        }
    }
    // A full timestamp is an exact upper bound, not an instruction to include
    // the rest of its day. Offsets must be normalized to the same UTC instant.
    for (end, expected) in [
        ("2026-07-31T23:59:59.999Z", 1.0),
        ("2026-08-01T01:59:59.999999999+02:00", 7.0),
    ] {
        let body: Value = client
            .get(handle.url("/global/spend"))
            .bearer_auth(MASTER_KEY)
            .query(&[("start_date", "2026-07-31"), ("end_date", end)])
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(body["total_cost"].as_f64(), Some(expected));
    }
    shutdown(handle).await;
}

#[tokio::test]
async fn deployment_spend_separates_siblings_for_unary_and_streaming_calls() {
    let first = MockServer::start_async().await;
    let second = MockServer::start_async().await;
    let mut config = proxy_config(&first.base_url());
    config.model_list[0].model_name = "external-chat".into();
    config.model_list[0].model_info = Some(oxllm_core::deployment::ModelInfo {
        id: Some("leg-a".into()),
        ..Default::default()
    });
    config.model_list[0].oxllm_params.input_cost_per_token = Some(0.001);
    config.model_list[0].oxllm_params.output_cost_per_token = Some(0.002);
    let mut sibling = config.model_list[0].clone();
    sibling.oxllm_params.api_base = Some(second.base_url());
    sibling.oxllm_params.input_cost_per_token = Some(0.003);
    sibling.oxllm_params.output_cost_per_token = Some(0.004);
    sibling.model_info.as_mut().unwrap().id = Some("leg-b".into());
    config.model_list.push(sibling);
    config.router_settings.routing.routing_strategy = Some("round_robin".into());
    let first_wire = mock_chat(&first);
    let second_wire = second.mock_async(|when, then| {
        when.method(httpmock::Method::POST).path("/chat/completions");
        then.status(200).header("content-type", "text/event-stream").body(concat!(
            "data: {\"id\":\"stream\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"stream\",\"model\":\"gpt-4o\",\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5,\"total_tokens\":15}}\n\n",
            "data: [DONE]\n\n"
        ));
    }).await;
    let handle = start_oxllm(config).await;
    let client = reqwest::Client::new();
    for (stream, id) in [(false, "leg-a"), (true, "leg-b")] {
        let response = client
            .post(handle.url("/v1/chat/completions"))
            .bearer_auth(MASTER_KEY)
            .json(&json!({
                "model": "external-chat", "messages": [{"role": "user", "content": "hi"}],
                "stream": stream, "stream_options": {"include_usage": true}
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        assert_eq!(response.headers()["x-oxllm-deployment-id"], id);
        let body = response.text().await.unwrap();
        assert!(body.contains("external-chat"), "{body}");
    }
    wait_for_spend_count(&handle.db, 2).await;
    let all = handle
        .db
        .query_spend(&oxllm_server::SpendFilter::default())
        .await
        .unwrap();
    assert_eq!(all.len(), 2);
    assert!(all.iter().all(
        |row| row.requested_model.as_deref() == Some("external-chat")
            && row.model_group.as_deref() == Some("external-chat")
    ));
    assert!(
        all.iter()
            .all(|row| row.model == "gpt-4o" && row.provider == "openai")
    );
    assert!((all.iter().filter_map(|row| row.cost).sum::<f64>() - 0.07).abs() < 1e-9);
    for (id, cost) in [("leg-a", 0.02), ("leg-b", 0.05)] {
        for endpoint in ["/spend/logs", "/spend/logs/v2"] {
            let response = client
                .get(handle.url(endpoint))
                .bearer_auth(MASTER_KEY)
                .query(&[("deployment_id", id)])
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), 200);
            let body = response.json::<Value>().await.unwrap();
            assert_eq!(body["data"].as_array().unwrap().len(), 1, "{body}");
            assert_eq!(body["data"][0]["deployment_id"], id);
            assert!((body["data"][0]["cost"].as_f64().unwrap() - cost).abs() < 1e-9);
        }
    }
    first_wire.assert_hits_async(1).await;
    second_wire.assert_hits_async(1).await;
    shutdown(handle).await;
}

#[tokio::test]
async fn fallback_spend_filters_separate_requested_alias_from_serving_group() {
    let primary = MockServer::start_async().await;
    let serving = MockServer::start_async().await;
    let mut config = proxy_config(&primary.base_url());
    config.model_list[0].model_name = "requested-alias".into();
    let mut fallback = config.model_list[0].clone();
    fallback.model_name = "serving-alias".into();
    fallback.oxllm_params.api_base = Some(serving.base_url());
    fallback.oxllm_params.input_cost_per_token = Some(0.002);
    fallback.oxllm_params.output_cost_per_token = Some(0.004);
    fallback.model_info = Some(oxllm_core::ModelInfo {
        id: Some("billed-leg".into()),
        ..Default::default()
    });
    config.model_list.push(fallback);
    config.router_settings.routing.num_retries = Some(0);
    config.router_settings.routing.fallbacks =
        Some(json!([{"requested-alias": ["serving-alias"]}]));
    let failed = primary
        .mock_async(|when, then| {
            when.method(httpmock::Method::POST);
            then.status(503)
                .json_body(json!({"error": {"message": "unavailable"}}));
        })
        .await;
    let healthy = mock_chat(&serving);
    let handle = start_oxllm(config).await;
    let client = reqwest::Client::new();
    let response = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth(MASTER_KEY)
        .json(&json!({"model": "requested-alias", "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    assert_eq!(response.headers()["x-oxllm-model-group"], "serving-alias");
    assert_eq!(
        response.headers()["x-oxllm-requested-model"],
        "requested-alias"
    );
    assert_eq!(
        response.json::<Value>().await.unwrap()["model"],
        "serving-alias"
    );
    wait_for_spend_count(&handle.db, 1).await;
    for endpoint in ["/spend/logs", "/spend/logs/v2"] {
        for filter in [
            ("requested_model", "requested-alias"),
            ("model_group", "serving-alias"),
        ] {
            let response = client
                .get(handle.url(endpoint))
                .bearer_auth(MASTER_KEY)
                .query(&[filter])
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), 200);
            let body = response.json::<Value>().await.unwrap();
            assert_eq!(body["data"].as_array().unwrap().len(), 1);
            let row = &body["data"][0];
            assert_eq!(row["requested_model"], "requested-alias");
            assert_eq!(row["model_group"], "serving-alias");
            assert_eq!(row["deployment_id"], "billed-leg");
            assert_eq!(row["model"], "gpt-4o");
            assert!((row["cost"].as_f64().unwrap() - 0.04).abs() < 1e-9);
        }
    }
    failed.assert_hits_async(1).await;
    healthy.assert_hits_async(1).await;
    shutdown(handle).await;
}

#[tokio::test]
async fn spend_tags_http_pages_are_bounded_and_report_complete_totals() {
    let mock = MockServer::start();
    let handle = start_oxllm(proxy_config(&mock.base_url())).await;
    let rows: Vec<_> = (0..600)
        .map(|i| {
            seeded_log(
                "2026-07-31T12:00:00Z",
                0.125,
                Some(
                    serde_json::to_string(&["shared".to_string(), format!("tag-{i:04}")]).unwrap(),
                ),
            )
        })
        .collect();
    handle.db.insert_spend_logs_batch(&rows).await.unwrap();
    let client = reqwest::Client::new();
    let first: Value = client
        .get(handle.url("/spend/tags?limit=1"))
        .bearer_auth(MASTER_KEY)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(first["data"].as_array().unwrap().len(), 1);
    assert_eq!(first["data"][0]["tag_name"], "shared");
    assert_eq!(first["data"][0]["total_spend"], 75.0);
    assert_eq!(first["data"][0]["total_requests"], 600);
    assert_eq!(first["next_offset"], 1);
    let capped: Value = client
        .get(handle.url("/spend/tags?limit=999999&offset=-5"))
        .bearer_auth(MASTER_KEY)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(capped["data"].as_array().unwrap().len(), 500);
    assert_eq!(
        (capped["limit"].as_i64(), capped["offset"].as_i64()),
        (Some(500), Some(0))
    );
    assert_eq!(capped["next_offset"], 500);
    let last: Value = client
        .get(handle.url("/spend/tags?limit=500&offset=500"))
        .bearer_auth(MASTER_KEY)
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(last["data"].as_array().unwrap().len(), 101);
    assert!(last["next_offset"].is_null());
    let seen: std::collections::BTreeSet<_> = capped["data"]
        .as_array()
        .unwrap()
        .iter()
        .chain(last["data"].as_array().unwrap())
        .map(|row| row["tag_name"].as_str().unwrap())
        .collect();
    assert_eq!(seen.len(), 601);
    shutdown(handle).await;
}

#[tokio::test]
async fn spend_logs_v2_paginates_with_cursor() {
    let mock = MockServer::start();
    let handle = start_oxllm(proxy_config(&mock.base_url())).await;
    let _m = mock_chat(&mock);

    let client = reqwest::Client::new();
    for _ in 0..3 {
        let _ = client
            .post(handle.url("/v1/chat/completions"))
            .bearer_auth(MASTER_KEY)
            .json(&json!({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}))
            .send()
            .await
            .expect("chat");
    }

    wait_for_spend_count(&handle.db, 3).await;

    let page1 = client
        .get(handle.url("/spend/logs/v2?limit=2"))
        .bearer_auth(MASTER_KEY)
        .send()
        .await
        .expect("p1");
    assert_eq!(page1.status(), 200);
    let p1: Value = page1.json().await.expect("json");
    let data1 = p1["data"].as_array().expect("data array");
    assert_eq!(data1.len(), 2, "first page should have 2 rows");
    let next_cursor = p1["next_cursor"]
        .as_i64()
        .expect("next_cursor should be set");

    let page2 = client
        .get(handle.url(&format!("/spend/logs/v2?limit=2&cursor={next_cursor}")))
        .bearer_auth(MASTER_KEY)
        .send()
        .await
        .expect("p2");
    assert_eq!(page2.status(), 200);
    let p2: Value = page2.json().await.expect("json");
    let data2 = p2["data"].as_array().expect("data array");
    assert_eq!(data2.len(), 1, "second page should have the remaining row");

    let seen_ids: std::collections::HashSet<i64> = data1
        .iter()
        .chain(data2.iter())
        .filter_map(|r| r["id"].as_i64())
        .collect();
    assert_eq!(seen_ids.len(), 3, "all three ids should be unique");

    shutdown(handle).await;
}

#[tokio::test]
async fn global_spend_matches_sum_of_logs() {
    let mock = MockServer::start();
    let handle = start_oxllm(proxy_config(&mock.base_url())).await;
    let _m = mock_chat(&mock);

    let client = reqwest::Client::new();
    for _ in 0..2 {
        let resp = client
            .post(handle.url("/v1/chat/completions"))
            .bearer_auth(MASTER_KEY)
            .json(&json!({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}))
            .send()
            .await
            .expect("chat");
        assert_eq!(resp.status(), 200);
    }
    wait_for_spend_count(&handle.db, 2).await;

    let logs_sum: f64 = handle
        .db
        .query_spend(&oxllm_server::SpendFilter::default())
        .await
        .expect("query")
        .iter()
        .filter_map(|l| l.cost)
        .sum();

    let resp = client
        .get(handle.url("/global/spend"))
        .bearer_auth(MASTER_KEY)
        .send()
        .await
        .expect("global");
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("json");
    let total = body["total_cost"].as_f64().expect("total_cost present");
    assert!(
        (total - logs_sum).abs() < 1e-6,
        "global/spend {total} should match sum of logs {logs_sum}"
    );
    assert!(total > 0.0);

    shutdown(handle).await;
}

#[tokio::test]
async fn global_activity_returns_daily_buckets() {
    let mock = MockServer::start();
    let handle = start_oxllm(proxy_config(&mock.base_url())).await;
    let _m = mock_chat(&mock);

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/v1/chat/completions"))
        .bearer_auth(MASTER_KEY)
        .json(&json!({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}))
        .send()
        .await
        .expect("chat");
    assert_eq!(resp.status(), 200);
    wait_for_spend_count(&handle.db, 1).await;

    let resp = client
        .get(handle.url("/global/activity?days=7"))
        .bearer_auth(MASTER_KEY)
        .send()
        .await
        .expect("activity");
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("json");
    let results = body["results"].as_array().expect("results array");
    assert!(!results.is_empty(), "should have at least one bucket");
    let total_spend = body["total_spend"].as_f64().unwrap_or(0.0);
    assert!(total_spend > 0.0, "total_spend should be positive");
    let days = body["days"].as_i64().expect("days");
    assert_eq!(days, 7);

    shutdown(handle).await;
}

#[tokio::test]
async fn spend_calculate_returns_known_cost_without_db() {
    let mock = MockServer::start();
    let handle = start_oxllm(proxy_config(&mock.base_url())).await;
    let never_hit = mock_chat_never(&mock);

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/spend/calculate"))
        .bearer_auth(MASTER_KEY)
        .json(&json!({
        "model": "gpt-4o",
        "usage": {
        "prompt_tokens": 1000,
        "completion_tokens": 500,
        "total_tokens": 1500
        }
        }))
        .send()
        .await
        .expect("calculate");
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("json");
    assert_eq!(body["model"].as_str(), Some("gpt-4o"));
    let cost = body["cost"].as_f64().expect("cost present");
    assert!(cost > 0.0, "gpt-4o cost should be positive, got {cost}");

    assert_eq!(
        never_hit.hits(),
        0,
        "spend/calculate must not touch upstream"
    );

    shutdown(handle).await;
}

#[tokio::test]
async fn spend_calculate_unknown_model_returns_null_cost() {
    let mock = MockServer::start();
    let handle = start_oxllm(proxy_config(&mock.base_url())).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(handle.url("/spend/calculate"))
        .bearer_auth(MASTER_KEY)
        .json(&json!({
        "model": "made-up-model-xyz",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }))
        .send()
        .await
        .expect("calculate");
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.expect("json");
    assert!(
        body["cost"].is_null(),
        "cost should be null for unknown model"
    );

    shutdown(handle).await;
}

#[tokio::test]
async fn spend_endpoints_require_master_key() {
    let mock = MockServer::start();
    let handle = start_oxllm(proxy_config(&mock.base_url())).await;

    let client = reqwest::Client::new();
    let endpoints = ["/spend/logs/v2", "/global/spend", "/global/activity"];
    for ep in endpoints {
        let resp = client
            .get(handle.url(ep))
            .bearer_auth("sk-wrong")
            .send()
            .await
            .expect("req");
        assert_eq!(
            resp.status(),
            401,
            "{ep} should reject non-master key with 401"
        );
    }
    let resp = client
 .post(handle.url("/spend/calculate"))
 .bearer_auth("sk-wrong")
 .json(&json!({"model": "gpt-4o", "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}))
 .send()
 .await
 .expect("req");
    assert_eq!(
        resp.status(),
        401,
        "spend/calculate should require master key"
    );

    shutdown(handle).await;
}
