//! HTTP regressions for native routing, spend previews/receipts/rollups, and alias claims.

mod common;

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use common::OxllmHandle;
use httpmock::prelude::HttpMockRequest;
use httpmock::{Mock, MockServer};
use oxllm_core::deployment::ModelInfo;
use oxllm_server::app::{build_router, build_shared_state_with_db};
use oxllm_server::config::types::{ConfigModel, ProxyConfig};
use oxllm_server::spend_writer::{self, SpendWriterHandle};
use oxllm_server::{Database, ModelConfigRecord, SpendLog};
use reqwest::Client;
use serde_json::{Value, json};

// Published dummy keys used only by the local integration harness.
const MASTER_KEY: &str = "sk-master-test-key";
const UPSTREAM_KEY: &str = "sk-test-key";
const DAY: &str = "2026-08-01";
const LAST_IN_DAY: &str = "2026-08-01T23:59:59.999999999Z";
const ROLLUPS: [(&str, &str); 5] = [
    ("/spend/keys", "key_id"),
    ("/spend/by-model", "model"),
    ("/spend/users", "user_id"),
    ("/spend/teams", "team_id"),
    ("/spend/organizations", "org_id"),
];

fn authenticated_config(base: &str) -> ProxyConfig {
    let mut config = common::proxy_config(base);
    config.general_settings.master_key = Some(MASTER_KEY.into());
    config
}

fn model_config(base: &str, alias: &str, model: &str) -> ConfigModel {
    let mut deployment = common::proxy_config(base).model_list.remove(0);
    deployment.model_name = alias.into();
    deployment.oxllm_params.model = model.into();
    deployment
}

async fn response_json(response: reqwest::Response, expected_status: u16) -> Value {
    let status = response.status().as_u16();
    let body = response.text().await.expect("response body");
    assert_eq!(status, expected_status, "HTTP response: {body}");
    serde_json::from_str(&body).unwrap_or_else(|error| panic!("JSON response: {error}: {body}"))
}

async fn post(client: &Client, url: String, body: &Value) -> reqwest::Response {
    client
        .post(url)
        .bearer_auth(MASTER_KEY)
        .json(body)
        .send()
        .await
        .expect("POST request")
}

async fn post_json(client: &Client, url: String, body: &Value) -> Value {
    response_json(post(client, url, body).await, 200).await
}

async fn get_json(client: &Client, url: String) -> Value {
    response_json(
        client
            .get(url)
            .bearer_auth(MASTER_KEY)
            .send()
            .await
            .expect("GET request"),
        200,
    )
    .await
}

async fn billed_json(response: reqwest::Response) -> (Value, f64) {
    let cost = response
        .headers()
        .get("x-oxllm-response-cost")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<f64>().ok());
    let body = response_json(response, 200).await;
    let cost = cost.unwrap_or_else(|| panic!("missing billed cost for {body}"));
    (body, cost)
}

async fn billed_post(client: &Client, url: String, body: &Value) -> (Value, f64) {
    billed_json(post(client, url, body).await).await
}

fn assert_cost(actual: f64, expected: f64, context: &str) {
    assert!(
        actual.is_finite() && (actual - expected).abs() < 1e-12,
        "{context}: cost {actual}, expected {expected}"
    );
}

async fn preview(client: &Client, handle: &OxllmHandle, body: Value) -> f64 {
    let model = body["model"].clone();
    let result = post_json(client, handle.url("/spend/calculate"), &body).await;
    assert_eq!(
        result["model"], model,
        "preview must retain the public selector"
    );
    result["cost"]
        .as_f64()
        .unwrap_or_else(|| panic!("expected priced preview: {result}"))
}

fn chat_response(model: &str, usage: Value, content: &str) -> Value {
    json!({
        "id": "chatcmpl-review-local",
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop"
        }],
        "usage": usage
    })
}

struct ManagementHandle {
    inner: OxllmHandle,
    db: Arc<Database>,
    server: tokio::task::JoinHandle<()>,
    writer: SpendWriterHandle,
}

impl ManagementHandle {
    fn url(&self, path: &str) -> String {
        self.inner.url(path)
    }

    async fn shutdown(self) {
        let _ = self.inner.shutdown_tx.send(());
        tokio::time::timeout(Duration::from_secs(5), self.server)
            .await
            .expect("server shutdown deadline")
            .expect("server task");
        self.writer.shutdown().await;
    }
}

// The management suite's DB-aware harness, with deterministic writer shutdown.
async fn start_oxllm_with_db(config: ProxyConfig) -> ManagementHandle {
    let db = Database::connect("sqlite::memory:")
        .await
        .expect("connect SQLite");
    db.migrate().await.expect("migrate SQLite");
    let (writer, writer_handle) = spend_writer::spawn_with_shutdown(db.clone(), 1, 0);
    let shared = build_shared_state_with_db(config, Some(writer.sender()), Some(db.clone()))
        .expect("DB-backed state");
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind proxy");
    let addr = listener.local_addr().expect("proxy address");
    let router = build_router(shared);
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
    let server = tokio::spawn(async move {
        axum::serve(listener, router)
            .with_graceful_shutdown(async move {
                let _ = shutdown_rx.await;
            })
            .await
            .expect("serve proxy");
    });
    ManagementHandle {
        inner: OxllmHandle { addr, shutdown_tx },
        db,
        server,
        writer: writer_handle,
    }
}

fn azure_config(base: &str, version: &str, alias: &str, model: &str) -> ProxyConfig {
    let mut config = authenticated_config(base);
    config.model_list = vec![model_config(base, alias, model)];
    config.model_list[0].oxllm_params.api_version = Some(version.into());
    config
}

#[tokio::test]
async fn perplexity_dynamic_receipts_match_headers_streams_previews_ledger_and_budgets() {
    let client = common::http_client();
    for custom in [false, true] {
        let mock = MockServer::start();
        let receipt = json!({
            "input_tokens_cost": 0.000005, "output_tokens_cost": 0.000001,
            "request_cost": 0.00987, "total_cost": 0.009876,
        });
        let usage =
            json!({"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6, "cost": receipt});
        let unary = mock.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/chat/completions")
                .json_body_partial(json!({"model":"sonar", "stream":false}).to_string());
            then.status(200)
                .json_body(chat_response("sonar", usage.clone(), "OK"));
        });
        let streamed = mock.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/chat/completions")
                .json_body_partial(json!({"model":"sonar", "stream":true}).to_string());
            let content = json!({"model":"sonar", "choices":[{"index":0,"delta":{"content":"OK"},"finish_reason":null}]});
            let final_usage = json!({"model":"sonar", "choices":[], "usage":usage});
            then.status(200).header("content-type", "text/event-stream")
                .body(format!("data: {content}\n\ndata: {final_usage}\n\ndata: [DONE]\n\n"));
        });
        let mut config = authenticated_config(&mock.base_url());
        config.model_list = vec![model_config(
            &mock.base_url(),
            "sonar-public",
            "perplexity/sonar",
        )];
        config.oxllm_settings.max_budget = Some(0.025);
        if custom {
            let parameters = &mut config.model_list[0].oxllm_params;
            parameters.input_cost_per_token = Some(2e-6);
            parameters.output_cost_per_token = Some(3e-6);
        }
        let expected = if custom { 13e-6 } else { 0.009876 };
        let handle = start_oxllm_with_db(config).await;
        // Master keys intentionally bypass budget enforcement. Use a virtual
        // key so this regression exercises real reservation/settlement gates.
        let generated = post_json(
            &client,
            handle.url("/key/generate"),
            &json!({"models":["sonar-public"], "max_budget":0.025}),
        )
        .await;
        let key = generated["key"].as_str().expect("virtual key");
        let mut request_ids = Vec::new();
        for stream in [false, true] {
            let response = client.post(handle.url("/v1/chat/completions"))
                .bearer_auth(key).header("x-oxllm-tag", "billing-header")
                .json(&json!({"model":"sonar-public", "stream":stream, "stream_options":{"include_usage":true},
                    "max_tokens":16, "messages":[{"role":"user","content":"OK"}],
                    "metadata":{"tags":["billing-body","billing-header"]}}))
                .send().await.expect("chat request");
            assert_eq!(response.status(), 200);
            request_ids.push(
                response.headers()["x-oxllm-request-id"]
                    .to_str()
                    .unwrap()
                    .to_string(),
            );
            if stream {
                assert!(response.headers().get("x-oxllm-response-cost").is_none());
                let text = response.text().await.unwrap();
                assert!(text.contains("[DONE]"));
                let final_usage = text
                    .lines()
                    .filter_map(|line| line.strip_prefix("data: "))
                    .filter_map(|data| serde_json::from_str::<Value>(data).ok())
                    .find_map(|chunk| chunk["usage"].is_object().then(|| chunk["usage"].clone()))
                    .expect("stream usage receipt");
                let result = post_json(
                    &client,
                    handle.url("/spend/calculate"),
                    &json!({"model":"sonar-public", "provider":"perplexity", "usage":final_usage}),
                )
                .await;
                assert_cost(
                    result["cost"].as_f64().unwrap(),
                    expected,
                    "stream receipt preview",
                );
            } else {
                let (body, cost) = billed_json(response).await;
                assert_cost(cost, expected, "receipt header");
                let result = post_json(&client, handle.url("/spend/calculate"),
                    &json!({"model":"sonar-public", "provider":"perplexity", "usage":body["usage"]})).await;
                assert_cost(
                    result["cost"].as_f64().unwrap(),
                    expected,
                    "unary receipt preview",
                );
            }
        }
        if !custom {
            // Token-only prices would leave this budget virtually untouched;
            // the complete receipts must settle both dynamic request fees.
            // A committed spend row proves post-success (including the earlier
            // budget hook) finished, rather than racing the stream's [DONE].
            tokio::time::timeout(Duration::from_secs(3), async {
                loop {
                    if handle
                        .db
                        .query_spend(&Default::default())
                        .await
                        .unwrap()
                        .len()
                        == 2
                    {
                        break;
                    }
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
            })
            .await
            .expect("both settled rows flushed");
            let denied = client
                .post(handle.url("/v1/chat/completions"))
                .bearer_auth(key)
                .json(&json!({"model":"sonar-public", "max_tokens":16,
                    "messages":[{"role":"user","content":"OK"}]}))
                .send()
                .await
                .unwrap();
            assert_eq!(denied.status(), 429, "dynamic fees must debit the budget");
            for selector in ["sonar-public", "sonar", "perplexity/sonar"] {
                let result = post_json(&client, handle.url("/spend/calculate"), &json!({
                    "model":selector, "provider":"perplexity", "usage":{"prompt_tokens":5,"completion_tokens":1}
                })).await;
                assert_cost(
                    result["cost"].as_f64().unwrap(),
                    6e-6,
                    "no receipt: canonical native rate card",
                );
            }
        }
        unary.assert_hits(1);
        streamed.assert_hits(1);
        let db = handle.db.clone();
        handle.shutdown().await;
        let rows = db.query_spend(&Default::default()).await.unwrap();
        assert_eq!(rows.len(), 2);
        for row in rows {
            assert!(request_ids.contains(row.request_id.as_ref().unwrap()));
            assert_eq!(row.model, "sonar");
            assert_eq!(row.provider, "perplexity");
            assert_eq!(row.prompt_tokens, Some(5));
            assert_eq!(row.completion_tokens, Some(1));
            assert_cost(row.cost.unwrap(), expected, "complete durable receipt");
            assert_eq!(row.tag_name.as_deref(), Some("billing-body"));
            assert_eq!(
                serde_json::from_str::<Value>(row.tags.as_deref().unwrap()).unwrap(),
                json!(["billing-body", "billing-header"])
            );
        }
    }
}

#[tokio::test]
async fn native_responses_usage_and_served_snapshot_match_headers_previews_and_ledger() {
    let mock = MockServer::start();
    let served = "gpt-6.1-sol-2026-09-29";
    let usage = json!({"input_tokens":1000,"output_tokens":80,"total_tokens":1080,
        "input_tokens_details":{"cached_tokens":300,"cache_write_tokens":200},
        "output_tokens_details":{"reasoning_tokens":60}});
    let native = json!({"id":"resp-native-billing","object":"response","status":"completed",
        "model":served,"output":[],"usage":usage});
    let unary = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/v1/responses")
            .json_body_partial(json!({"model":"gpt-6.1-sol","stream":false}).to_string());
        then.status(200).json_body(native.clone());
    });
    let terminal = json!({"type":"response.completed","response":native});
    let wire = format!("event: response.completed\ndata: {terminal}\n\n");
    let streamed = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/v1/responses")
            .json_body_partial(json!({"model":"gpt-6.1-sol","stream":true}).to_string());
        then.status(200)
            .header("content-type", "text/event-stream")
            .body(wire.clone());
    });
    let mut config = authenticated_config(&mock.base_url());
    let mut deployment = model_config(&mock.base_url(), "native-public", "openai/gpt-6.1-sol");
    deployment.oxllm_params.input_cost_per_token = Some(4e-6);
    deployment.oxllm_params.output_cost_per_token = Some(10e-6);
    deployment.oxllm_params.output_cost_per_reasoning_token = Some(2e-6);
    deployment.oxllm_params.cache_read_input_token_cost = Some(1e-6);
    deployment.oxllm_params.cache_creation_input_token_cost = Some(5e-6);
    config.model_list = vec![deployment];
    let handle = start_oxllm_with_db(config).await;
    let client = common::http_client();
    let expected = 500.0 * 4e-6 + 300.0 * 1e-6 + 200.0 * 5e-6 + 20.0 * 10e-6 + 60.0 * 2e-6;
    for stream in [false, true] {
        let response = post(&client, handle.url("/openai/v1/responses"),
            &json!({"model":"native-public","input":"OK","stream":stream,"metadata":{"tags":["native-body"]}})).await;
        assert_eq!(response.status(), 200);
        if stream {
            assert!(response.headers().get("x-oxllm-response-cost").is_none());
            assert_eq!(
                response.text().await.unwrap(),
                wire,
                "native stream stays byte-preserving"
            );
        } else {
            let (body, cost) = billed_json(response).await;
            assert_eq!(body, native);
            assert_cost(cost, expected, "native cache/reasoning cost");
        }
        let result = post_json(
            &client,
            handle.url("/spend/calculate"),
            &json!({"model":"native-public","provider":"openai","usage":usage}),
        )
        .await;
        assert_cost(
            result["cost"].as_f64().unwrap(),
            expected,
            "native detail-bucket preview",
        );
    }
    unary.assert_hits(1);
    streamed.assert_hits(1);
    let db = handle.db.clone();
    handle.shutdown().await;
    let rows = db.query_spend(&Default::default()).await.unwrap();
    assert_eq!(rows.len(), 2);
    for row in rows {
        assert_eq!(
            row.model, served,
            "ledger records the actual served snapshot"
        );
        assert_eq!(row.prompt_tokens, Some(800));
        assert_eq!(row.completion_tokens, Some(80));
        assert_eq!(row.total_tokens, Some(880));
        assert_cost(row.cost.unwrap(), expected, "native durable billing");
        assert_eq!(row.tag_name.as_deref(), Some("native-body"));
    }
}

fn one_azure_version_without_routing_model(request: &HttpMockRequest) -> bool {
    let pairs = request.query_params.as_deref().unwrap_or_default();
    pairs
        .iter()
        .filter(|(name, _)| name.eq_ignore_ascii_case("api-version"))
        .count()
        == 1
        && pairs.iter().all(|(name, _)| name != "model")
}

#[tokio::test]
async fn azure_hooked_images_and_speech_normalize_classic_and_v1_bases_on_wire() {
    let client = common::http_client();
    for (version, wire_version, prefix) in [
        (
            "2025-02-01-preview",
            "2025-02-01-preview",
            "/openai/deployments/actual-deployment",
        ),
        ("v1", "v1", "/openai/v1"),
        ("preview", "preview", "/openai/v1"),
        ("latest", "v1", "/openai/v1"),
    ] {
        for suffix in ["", "/openai/", "/openai/v1/"] {
            let mock = MockServer::start();
            let base = format!(
                "{}{suffix}?api-version=old&%61pi-version=older&API-VERSION=oldest&configured=keep%2Fbase%2Bvalue",
                mock.base_url()
            );
            let image = mock.mock(|when, then| {
                when.method(httpmock::Method::POST)
                    .path(format!("{prefix}/images/generations"))
                    .header("api-key", UPSTREAM_KEY)
                    .query_param("api-version", wire_version)
                    .query_param("configured", "keep/base+value")
                    .matches(one_azure_version_without_routing_model)
                    .json_body(json!({
                        "model": "actual-deployment", "prompt": "A copper lighthouse", "n": 1
                    }));
                then.status(200)
                    .json_body(json!({"created": 1, "data": [{"b64_json": "AAE="}]}));
            });
            let audio_bytes = b"RIFF\x00\xff\xfe\x80review-audio";
            let speech = mock.mock(|when, then| {
                when.method(httpmock::Method::POST)
                    .path(format!("{prefix}/audio/speech"))
                    .header("api-key", UPSTREAM_KEY)
                    .query_param("api-version", wire_version)
                    .query_param("configured", "keep/base+value")
                    .matches(one_azure_version_without_routing_model)
                    .json_body(json!({
                        "model": "actual-deployment", "input": "Hello", "voice": "alloy",
                        "response_format": "wav"
                    }));
                then.status(200)
                    .header("content-type", "audio/wav")
                    .body(audio_bytes);
            });
            let (handle, _) = common::start_oxllm(azure_config(
                &base,
                version,
                "public-alias",
                "azure/actual-deployment",
            ))
            .await
            .expect("Azure proxy");
            let body = post_json(
                &client,
                handle.url("/v1/images/generations"),
                &json!({"model": "public-alias", "prompt": "A copper lighthouse", "n": 1}),
            )
            .await;
            assert_eq!(body["data"][0]["b64_json"], "AAE=", "{version} {suffix}");
            let response = post(
                &client,
                handle.url("/v1/audio/speech"),
                &json!({"model": "public-alias", "input": "Hello", "voice": "alloy", "response_format": "wav"}),
            )
            .await;
            assert_eq!(response.status(), 200, "{version} {suffix}");
            assert_eq!(response.headers()["content-type"], "audio/wav");
            assert_eq!(response.bytes().await.unwrap().as_ref(), audio_bytes);
            image.assert_hits(1);
            speech.assert_hits(1);
            handle.shutdown();
        }
    }
}

#[tokio::test]
async fn azure_provider_prefixed_native_paths_keep_configured_version_and_query_values() {
    let client = common::http_client();
    for (version, prefix) in [
        (
            "2025-02-01-preview",
            "/openai/deployments/actual-deployment",
        ),
        ("v1", "/openai/v1"),
    ] {
        for suffix in ["", "/openai", "/openai/v1"] {
            let mock = MockServer::start();
            let base = format!(
                "{}{suffix}?api-version=old&%61pi-version=older&configured=keep%2Fbase%2Bvalue",
                mock.base_url()
            );
            let upstream = mock.mock(|when, then| {
                when.method(httpmock::Method::POST)
                    .path(format!("{prefix}/images/generations"))
                    .header("api-key", UPSTREAM_KEY)
                    .query_param("api-version", version)
                    .query_param("configured", "keep/base+value")
                    .query_param("client", "hello+world")
                    .matches(one_azure_version_without_routing_model)
                    .json_body(json!({"model": "actual-deployment", "prompt": "Keep every option", "size": "1024x1024"}));
                then.status(200).json_body(json!({"data": [{"b64_json": "AQI="}]}));
            });
            let speech_bytes = b"RIFF\x00\xff\xfe\x80native-speech";
            let speech = mock.mock(|when, then| {
                when.method(httpmock::Method::POST)
                    .path(format!("{prefix}/audio/speech"))
                    .header("api-key", UPSTREAM_KEY)
                    .query_param("api-version", version)
                    .query_param("configured", "keep/base+value")
                    .query_param("client", "hello+world")
                    .matches(one_azure_version_without_routing_model)
                    .json_body(json!({
                        "model": "actual-deployment", "input": "Native speech",
                        "voice": "alloy", "response_format": "wav"
                    }));
                then.status(200)
                    .header("content-type", "audio/wav")
                    .body(speech_bytes);
            });
            let (handle, _) = common::start_oxllm(azure_config(
                &base,
                version,
                "public-alias",
                "azure/actual-deployment",
            ))
            .await
            .unwrap();
            let result = post_json(
                &client,
                handle.url("/azure/v1/images/generations?model=public-alias&api-version=inbound-old&api-version=inbound-older&configured=replace-me&client=hello%2Bworld"),
                &json!({"model": "public-alias", "prompt": "Keep every option", "size": "1024x1024"}),
            )
            .await;
            assert_eq!(result["data"][0]["b64_json"], "AQI=");
            let response = post(
                &client,
                handle.url("/azure/v1/audio/speech?model=public-alias&api-version=inbound-old&api-version=inbound-older&configured=replace-me&client=hello%2Bworld"),
                &json!({
                    "model": "public-alias", "input": "Native speech",
                    "voice": "alloy", "response_format": "wav"
                }),
            )
            .await;
            assert_eq!(response.status(), 200);
            assert_eq!(response.headers()["content-type"], "audio/wav");
            assert_eq!(response.bytes().await.unwrap().as_ref(), speech_bytes);
            upstream.assert_hits(1);
            speech.assert_hits(1);
            handle.shutdown();
        }
    }
}

#[tokio::test]
async fn azure_wildcard_prefixed_selector_and_explicit_deployment_override_reach_wire() {
    let client = common::http_client();
    for version in ["2025-02-01-preview", "v1"] {
        for explicit in [false, true] {
            let mock = MockServer::start();
            let base = format!("{}/openai/v1/", mock.base_url());
            let actual = if explicit {
                "actual-deployment"
            } else {
                "client-selected"
            };
            let prefix = if version == "v1" {
                "/openai/v1".to_string()
            } else {
                format!("/openai/deployments/{actual}")
            };
            let upstream = mock.mock(|when, then| {
                when.method(httpmock::Method::POST)
                    .path(format!("{prefix}/images/generations"))
                    .header("api-key", UPSTREAM_KEY)
                    .query_param("api-version", version)
                    .json_body(json!({"model": actual, "prompt": "Wildcard selector"}));
                then.status(200)
                    .json_body(json!({"data": [{"b64_json": "AwQ="}]}));
            });
            let mut config = azure_config(&base, version, "*", "azure/*");
            if explicit {
                config.model_list[0]
                    .oxllm_params
                    .extra
                    .insert("azure_deployment".into(), json!(actual));
            }
            let (handle, _) = common::start_oxllm(config).await.unwrap();
            for path in ["/v1/images/generations", "/azure/v1/images/generations"] {
                post_json(
                    &client,
                    handle.url(path),
                    &json!({"model": "azure/client-selected", "prompt": "Wildcard selector"}),
                )
                .await;
            }
            upstream.assert_hits(2);
            handle.shutdown();
        }
    }
}

#[tokio::test]
async fn azure_query_only_wildcard_selector_is_inserted_into_native_json() {
    let client = common::http_client();
    for version in ["2025-02-01-preview", "v1"] {
        let mock = MockServer::start();
        let prefix = if version == "v1" {
            "/openai/v1"
        } else {
            "/openai/deployments/client-selected"
        };
        let upstream = mock.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path(format!("{prefix}/images/generations"))
                .header("api-key", UPSTREAM_KEY)
                .query_param("api-version", version)
                .matches(one_azure_version_without_routing_model)
                .json_body(
                    json!({"model": "client-selected", "prompt": "Query-selected deployment"}),
                );
            then.status(200).json_body(json!({"data": []}));
        });
        let (handle, _) =
            common::start_oxllm(azure_config(&mock.base_url(), version, "*", "azure/*"))
                .await
                .unwrap();
        post_json(
            &client,
            handle.url("/azure/v1/images/generations?model=azure%2Fclient-selected&api-version=inbound-old"),
            &json!({"prompt": "Query-selected deployment"}),
        )
        .await;
        upstream.assert_hits(1);
        handle.shutdown();
    }
}

const MULTIPART_BOUNDARY: &str = "review-native-boundary";
const MULTIPART_TYPE: &str = "multipart/form-data; boundary=\"review-native-boundary\"";
const FILE_BYTES: &[u8] = b"\x00\xff\xfe\x80RIFF\r\npublic-alias\x00\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nnot-a-selector";

fn multipart_body(model: &str, file_field: &str) -> Vec<u8> {
    let mut body = format!(
        "--{MULTIPART_BOUNDARY}\r\nContent-Disposition: form-data; name=\"{file_field}\"; filename=\"review.bin\"\r\nContent-Type: application/octet-stream\r\n\r\n"
    )
    .into_bytes();
    body.extend_from_slice(FILE_BYTES);
    body.extend_from_slice(format!(
        "\r\n--{MULTIPART_BOUNDARY}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n{model}\r\n--{MULTIPART_BOUNDARY}\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\nKeep public-alias verbatim here\r\n"
    ).as_bytes());
    if file_field == "image" {
        body.extend_from_slice(format!(
            "--{MULTIPART_BOUNDARY}\r\nContent-Disposition: form-data; name=\"mask\"; filename=\"mask.bin\"\r\nContent-Type: application/octet-stream\r\n\r\n"
        ).as_bytes());
        body.extend_from_slice(b"mask\x00\xff\xfe\x80\r\n");
    }
    body.extend_from_slice(format!("--{MULTIPART_BOUNDARY}--\r\n").as_bytes());
    body
}

fn exact_azure_multipart(request: &HttpMockRequest) -> bool {
    let field = if request.path.ends_with("/images/edits") {
        "image"
    } else {
        "file"
    };
    request.body.as_deref() == Some(multipart_body("actual-deployment", field).as_slice())
}

#[tokio::test]
async fn azure_transcription_and_edit_rewrite_only_model_and_preserve_binary_multipart() {
    let client = common::http_client();
    for version in ["2025-02-01-preview", "v1"] {
        for provider_prefixed in [false, true] {
            for explicit in [false, true] {
                let mock = MockServer::start();
                let base = format!("{}/openai/v1/", mock.base_url());
                let prefix = if version == "v1" {
                    "/openai/v1"
                } else {
                    "/openai/deployments/actual-deployment"
                };
                let mut mocks = Vec::new();
                for operation in ["audio/transcriptions", "images/edits"] {
                    mocks.push(mock.mock(|when, then| {
                        when.method(httpmock::Method::POST)
                            .path(format!("{prefix}/{operation}"))
                            .header("api-key", UPSTREAM_KEY)
                            .header("content-type", MULTIPART_TYPE)
                            .query_param("api-version", version)
                            .matches(exact_azure_multipart);
                        then.status(200)
                            .json_body(json!({"text": "binary intact", "data": []}));
                    }));
                }
                let (alias, model, requested) = if explicit {
                    ("*", "azure/*", "azure/client-selected")
                } else {
                    ("public-alias", "azure/actual-deployment", "public-alias")
                };
                let mut config = azure_config(&base, version, alias, model);
                if explicit {
                    config.model_list[0]
                        .oxllm_params
                        .extra
                        .insert("azure_deployment".into(), json!("actual-deployment"));
                }
                let (handle, _) = common::start_oxllm(config).await.unwrap();
                for (operation, field) in
                    [("audio/transcriptions", "file"), ("images/edits", "image")]
                {
                    let route = if provider_prefixed {
                        "/azure/v1"
                    } else {
                        "/v1"
                    };
                    let response = client
                        .post(handle.url(&format!("{route}/{operation}")))
                        .bearer_auth(MASTER_KEY)
                        .header("content-type", MULTIPART_TYPE)
                        .body(multipart_body(requested, field))
                        .send()
                        .await
                        .unwrap();
                    let body = response_json(response, 200).await;
                    assert_eq!(body["text"], "binary intact");
                }
                for upstream in mocks {
                    upstream.assert_hits(1);
                }
                handle.shutdown();
            }
        }
    }
}

fn seeded_log(timestamp: &str, model: &str, provider: &str, cost: f64) -> SpendLog {
    SpendLog {
        deployment_id: None,
        requested_model: None,
        model_group: None,
        id: None,
        timestamp: timestamp.into(),
        request_id: None,
        model: model.into(),
        provider: provider.into(),
        prompt_tokens: Some(10),
        completion_tokens: Some(5),
        total_tokens: Some(15),
        cost: Some(cost),
        key_id: None,
        team_id: None,
        user_id: None,
        end_user_id: None,
        tag_name: None,
        tags: None,
        status: Some("success".into()),
        latency_ms: Some(1),
    }
}

async fn rollup_page(
    client: &Client,
    handle: &ManagementHandle,
    endpoint: &str,
    limit: i64,
    offset: i64,
) -> Value {
    response_json(
        client
            .get(handle.url(endpoint))
            .bearer_auth(MASTER_KEY)
            .query(&[
                ("start_date", DAY.to_string()),
                ("end_date", DAY.to_string()),
                ("limit", limit.to_string()),
                ("offset", offset.to_string()),
            ])
            .send()
            .await
            .unwrap(),
        200,
    )
    .await
}

fn assert_page(page: &Value, limit: i64, offset: i64, count: usize, has_more: bool) {
    assert_eq!(page["data"].as_array().unwrap().len(), count, "{page}");
    assert_eq!(page["limit"], limit, "{page}");
    assert_eq!(page["offset"], offset, "{page}");
    assert_eq!(page["has_more"], has_more, "{page}");
    assert_eq!(
        page["next_offset"],
        if has_more {
            json!(offset + limit)
        } else {
            Value::Null
        },
        "{page}"
    );
}

#[tokio::test]
async fn spend_rollups_page_tied_groups_stably_without_clamping_date_scope_totals() {
    let mock = MockServer::start();
    let handle = start_oxllm_with_db(authenticated_config(&mock.base_url())).await;
    let client = common::http_client();
    let mut key_aliases = BTreeMap::new();
    let mut scopes = vec![(None, None, None)];
    // Intentionally create in non-sort order; IDs exercise bytewise ties.
    for suffix in ["a", "Z", "A"] {
        let team_id = format!("team-{suffix}");
        let user_id = format!("user-{suffix}");
        let org_id = format!("org-{suffix}");
        post_json(
            &client,
            handle.url("/team/add"),
            &json!({"team_id": team_id, "team_name": format!("Team {suffix}")}),
        )
        .await;
        post_json(
            &client,
            handle.url("/user/add"),
            &json!({"user_id": user_id, "user_name": format!("User {suffix}")}),
        )
        .await;
        post_json(
            &client,
            handle.url("/organization/add"),
            &json!({"org_id": org_id, "org_name": format!("Org {suffix}")}),
        )
        .await;
        let key = post_json(&client, handle.url("/key/generate"), &json!({
            "key_alias": format!("Key {suffix}"), "team_id": team_id, "user_id": user_id, "org_id": org_id
        })).await;
        let key_id = key["key_id"].as_str().unwrap().to_string();
        key_aliases.insert(key_id.clone(), format!("Key {suffix}"));
        scopes.push((Some(key_id), Some(team_id), Some(user_id)));
    }
    let models = [
        ("Same", "openai"),
        ("a", "openai"),
        ("Z", "openai"),
        ("Same", "azure"),
    ];
    let mut logs = Vec::new();
    for ((key_id, team_id, user_id), (model, provider)) in scopes.iter().zip(models) {
        // More than the spend-log row cap in EACH group. A limit of two must
        // page groups, not truncate the input rows feeding SUM/COUNT.
        for i in 0..1_100 {
            let timestamp = if i % 2 == 0 {
                "2026-08-01T03:00:00+02:00"
            } else {
                "2026-08-02T01:59:59.999999999+02:00"
            };
            let mut log = seeded_log(timestamp, model, provider, 0.125);
            log.key_id = key_id.clone();
            log.team_id = team_id.clone();
            log.user_id = user_id.clone();
            logs.push(log);
        }
        for timestamp in [
            "2026-07-31T23:59:59.999999999Z",
            "2026-08-02T00:00:00Z",
            "2026-08-01T00:30:00+02:00",
            "2026-08-01T23:30:00-02:00",
        ] {
            let mut outside = seeded_log(timestamp, model, provider, 1_024.0);
            outside.key_id = key_id.clone();
            outside.team_id = team_id.clone();
            outside.user_id = user_id.clone();
            logs.push(outside);
        }
    }
    handle.db.insert_spend_logs_batch(&logs).await.unwrap();
    for (endpoint, field) in ROLLUPS {
        let first = rollup_page(&client, &handle, endpoint, 2, 0).await;
        let second = rollup_page(&client, &handle, endpoint, 2, 2).await;
        assert_page(&first, 2, 0, 2, true);
        // An exactly full final page still has no next page.
        assert_page(&second, 2, 2, 2, false);
        assert_eq!(
            rollup_page(&client, &handle, endpoint, 2, 0).await,
            first,
            "{endpoint}: unstable ties"
        );
        let offset_bounds = response_json(
            client
                .get(handle.url(endpoint))
                .bearer_auth(MASTER_KEY)
                .query(&[
                    ("start_date", "2026-08-01T02:00:00+02:00"),
                    ("end_date", "2026-08-02T01:59:59.999999999+02:00"),
                    ("limit", "2"),
                    ("offset", "0"),
                ])
                .send()
                .await
                .unwrap(),
            200,
        )
        .await;
        assert_eq!(
            offset_bounds, first,
            "{endpoint}: UTC-equivalent filter scope"
        );
        let rows: Vec<_> = first["data"]
            .as_array()
            .unwrap()
            .iter()
            .chain(second["data"].as_array().unwrap())
            .collect();
        for row in &rows {
            assert_eq!(row["total_spend"], 137.5, "{endpoint}: {row}");
            assert_eq!(row["total_requests"], 1_100, "{endpoint}: {row}");
        }
        if endpoint == "/spend/by-model" {
            let identities: Vec<_> = rows
                .iter()
                .map(|row| {
                    (
                        row["model"].as_str().unwrap(),
                        row["provider"].as_str().unwrap(),
                    )
                })
                .collect();
            assert_eq!(
                identities,
                [
                    ("Same", "azure"),
                    ("Same", "openai"),
                    ("Z", "openai"),
                    ("a", "openai")
                ]
            );
        } else {
            assert!(
                rows[0][field].is_null(),
                "{endpoint}: NULL must win tied ordering"
            );
            let identities: Vec<_> = rows
                .iter()
                .skip(1)
                .map(|row| row[field].as_str().unwrap().to_string())
                .collect();
            let expected = match field {
                "key_id" => key_aliases.keys().cloned().collect::<Vec<_>>(),
                _ => ["A", "Z", "a"]
                    .map(|suffix| format!("{}-{suffix}", field.trim_end_matches("_id")))
                    .to_vec(),
            };
            assert_eq!(identities, expected, "{endpoint}");
        }
        for row in rows {
            match endpoint {
                "/spend/keys" => {
                    assert_eq!(row["last_used"], LAST_IN_DAY);
                    if let Some(id) = row["key_id"].as_str() {
                        assert_eq!(row["key_alias"], key_aliases[id]);
                    } else {
                        assert!(row["key_alias"].is_null());
                    }
                }
                "/spend/users" | "/spend/teams" => {
                    if let Some(id) = row[field].as_str() {
                        let suffix = id.rsplit('-').next().unwrap();
                        let name_field = if field == "user_id" {
                            "user_name"
                        } else {
                            "team_name"
                        };
                        let name = if field == "user_id" { "User" } else { "Team" };
                        assert_eq!(row[name_field], format!("{name} {suffix}"));
                    }
                }
                _ => {}
            }
        }
    }
    handle.shutdown().await;
}

#[tokio::test]
async fn spend_rollups_bound_negative_huge_and_exhausted_pages_for_every_dimension() {
    let mock = MockServer::start();
    let handle = start_oxllm_with_db(authenticated_config(&mock.base_url())).await;
    let client = common::http_client();
    let now = oxllm_server::now_iso();
    let mut logs = Vec::new();
    for i in (0..1_002).rev() {
        let id = format!("scope-{i:04}");
        // Public repository APIs keep this boundary fixture independent of
        // thousands of HTTP setup requests, as in management_e2e.
        let org = serde_json::from_value(json!({
            "org_id": id, "org_name": id, "created_at": now, "updated_at": now
        }))
        .unwrap();
        handle.db.create_organization(&org).await.unwrap();
        let key = serde_json::from_value(json!({
            "key_id": id, "key_hash": format!("review-boundary-{i:04}"), "key_alias": id,
            "org_id": id, "created_at": now, "updated_at": now, "is_active": true
        }))
        .unwrap();
        handle.db.create_key(&key).await.unwrap();
        let mut log = seeded_log("2026-08-01T12:00:00Z", &id, "openai", 0.125);
        log.key_id = Some(id.clone());
        log.team_id = Some(id.clone());
        log.user_id = Some(id);
        logs.push(log);
    }
    handle.db.insert_spend_logs_batch(&logs).await.unwrap();
    for (endpoint, field) in ROLLUPS {
        for bad_limit in [-1, 0, i64::MIN] {
            let page = rollup_page(&client, &handle, endpoint, bad_limit, -5).await;
            assert_page(&page, 1, 0, 1, true);
            assert_eq!(page["data"][0][field], "scope-0000", "{endpoint}");
        }
        let first = rollup_page(&client, &handle, endpoint, i64::MAX, -5).await;
        let last = rollup_page(&client, &handle, endpoint, i64::MAX, 1_000).await;
        assert_page(&first, 1_000, 0, 1_000, true);
        assert_page(&last, 1_000, 1_000, 2, false);
        let rows = first["data"]
            .as_array()
            .unwrap()
            .iter()
            .chain(last["data"].as_array().unwrap());
        for (i, row) in rows.enumerate() {
            assert_eq!(
                row[field],
                format!("scope-{i:04}"),
                "{endpoint}: missing/duplicate tied group"
            );
            assert_eq!(row["total_spend"], 0.125);
            assert_eq!(row["total_requests"], 1);
        }
        let empty = rollup_page(&client, &handle, endpoint, i64::MAX, i64::MAX).await;
        assert_page(&empty, 1_000, i64::MAX, 0, false);
    }
    handle.shutdown().await;
}

#[tokio::test]
async fn spend_calculate_alias_matches_chat_override_and_inherited_cache_output_prices() {
    let mock = MockServer::start();
    let usage = json!({
        "prompt_tokens": 1_000, "completion_tokens": 500, "total_tokens": 1_500,
        "prompt_tokens_details": {"cached_tokens": 200},
        "completion_tokens_details": {"reasoning_tokens": 120}
    });
    let mut config = authenticated_config(&mock.base_url());
    config.model_list[0].model_name = "priced-alias".into();
    config.model_list[0].oxllm_params.input_cost_per_token = Some(3e-6);
    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .header("authorization", format!("Bearer {UPSTREAM_KEY}"))
            .json_body_partial(json!({"model": "gpt-4o", "messages": [{"role": "user", "content": "Preview my bill"}], "service_tier": "priority"}).to_string());
        let mut response = chat_response("gpt-4o", usage.clone(), "priced reply");
        response["service_tier"] = json!("priority");
        then.status(200).json_body(response);
    });
    let (handle, _) = common::start_oxllm(config).await.unwrap();
    let client = common::http_client();
    let (body, billed) = billed_post(
        &client,
        handle.url("/v1/chat/completions"),
        &json!({
            "model": "priced-alias",
            "messages": [{"role": "user", "content": "Preview my bill"}],
            "service_tier": "priority"
        }),
    )
    .await;
    assert_eq!(body["usage"], usage);
    // Only input is overridden. Cache reads/output inherit their own table
    // prices; custom pricing remains flat even for a priority response.
    let expected = 800.0 * 3e-6 + 200.0 * 1.25e-6 + 500.0 * 1e-5;
    assert_cost(billed, expected, "chat configured override");
    for request in [
        json!({"model": "priced-alias", "usage": usage}),
        json!({"model": "priced-alias", "usage": usage, "service_tier": "priority"}),
        json!({"model": "gpt-4o", "provider": "openai", "usage": usage, "pricing_override": {"input_cost_per_token": 3e-6}}),
    ] {
        assert_cost(
            preview(&client, &handle, request).await,
            billed,
            "preview/inference parity",
        );
    }
    upstream.assert_hits(1);
    handle.shutdown();
}

#[tokio::test]
async fn spend_calculate_service_tier_prices_match_served_chat_costs() {
    let mock = MockServer::start();
    let client = common::http_client();
    let (handle, _) = common::start_oxllm(authenticated_config(&mock.base_url()))
        .await
        .unwrap();
    let usage = json!({"prompt_tokens": 1_000, "completion_tokens": 500, "prompt_tokens_details": {"cached_tokens": 200}});
    for (requested, served, preview_tier, expected) in [
        (
            "default",
            "default",
            "standard",
            800.0 * 2.5e-6 + 200.0 * 1.25e-6 + 500.0 * 1e-5,
        ),
        (
            "priority",
            "priority",
            " PRIORITY ",
            800.0 * 4.25e-6 + 200.0 * 2.125e-6 + 500.0 * 1.7e-5,
        ),
        (
            "priority",
            "default",
            "default",
            800.0 * 2.5e-6 + 200.0 * 1.25e-6 + 500.0 * 1e-5,
        ),
    ] {
        let upstream = mock.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/chat/completions")
                .header("authorization", format!("Bearer {UPSTREAM_KEY}"))
                .json_body_partial(
                    json!({"model": "gpt-4o", "service_tier": requested}).to_string(),
                );
            let mut body = chat_response("gpt-4o", usage.clone(), "tier reply");
            body["service_tier"] = json!(served);
            then.status(200).json_body(body);
        });
        let (body, billed) = billed_post(
            &client,
            handle.url("/v1/chat/completions"),
            &json!({
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "Service price"}],
                "service_tier": requested
            }),
        )
        .await;
        assert_eq!(body["service_tier"], served);
        assert_cost(billed, expected, "served chat tier");
        assert_cost(
            preview(
                &client,
                &handle,
                json!({"model": "gpt-4o", "usage": usage, "service_tier": preview_tier}),
            )
            .await,
            billed,
            "served-tier preview",
        );
        upstream.assert_hits(1);
        upstream.delete_async().await;
    }
    for tier in ["batch", "batches"] {
        assert_cost(
            preview(
                &client,
                &handle,
                json!({"model": "gpt-4o", "usage": usage, "service_tier": tier}),
            )
            .await,
            800.0 * 1.25e-6 + 200.0 * 1.25e-6 + 500.0 * 5e-6,
            "published batch token rates with inherited cached-read price",
        );
    }
    handle.shutdown();
}

#[tokio::test]
async fn spend_calculate_provider_specific_prices_match_native_bills_for_the_same_model() {
    let mock = MockServer::start();
    let client = common::http_client();
    let mut config = authenticated_config(&mock.base_url());
    config.model_list = ["groq", "openrouter"]
        .map(|provider| {
            model_config(
                &mock.base_url(),
                &format!("{provider}-alias"),
                &format!("{provider}/openai/gpt-oss-20b"),
            )
        })
        .to_vec();
    let (handle, _) = common::start_oxllm(config).await.unwrap();
    let usage = json!({"prompt_tokens": 1_000, "completion_tokens": 500, "prompt_tokens_details": {"cached_tokens": 200}});
    let mut costs = Vec::new();
    for (provider, expected) in [
        ("groq", 800.0 * 7.5e-8 + 200.0 * 3.75e-8 + 500.0 * 3e-7),
        ("openrouter", 800.0 * 1.8e-8 + 200.0 * 9e-9 + 500.0 * 9e-8),
    ] {
        let upstream = mock.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/v1/chat/completions")
                .header("authorization", format!("Bearer {UPSTREAM_KEY}"))
                .json_body(json!({
                    "model": "openai/gpt-oss-20b", "messages": [{"role": "user", "content": "Provider rate"}],
                    "metadata": {"review_case": provider}
                }));
            then.status(200).json_body(chat_response("openai/gpt-oss-20b", usage.clone(), provider));
        });
        let (_, billed) = billed_post(
            &client,
            handle.url(&format!("/{provider}/v1/chat/completions")),
            &json!({
                "model": format!("{provider}-alias"),
                "messages": [{"role": "user", "content": "Provider rate"}],
                "metadata": {"review_case": provider}
            }),
        )
        .await;
        assert_cost(billed, expected, provider);
        for request in [
            json!({"model": "openai/gpt-oss-20b", "provider": provider, "usage": usage}),
            json!({"model": format!("{provider}-alias"), "usage": usage}),
        ] {
            assert_cost(
                preview(&client, &handle, request).await,
                billed,
                "provider preview",
            );
        }
        costs.push(billed);
        upstream.assert_hits(1);
        upstream.delete_async().await;
    }
    assert!(
        costs[0] > costs[1] * 2.0,
        "provider hint must select distinct rate cards"
    );
    handle.shutdown();
}

#[tokio::test]
async fn spend_calculate_native_cache_one_hour_and_thinking_usage_match_native_bill() {
    let mock = MockServer::start();
    let usage = json!({
        "input_tokens": 100, "output_tokens": 50, "cache_read_input_tokens": 200,
        "cache_creation_input_tokens": 80,
        "cache_creation": {"ephemeral_5m_input_tokens": 50, "ephemeral_1h_input_tokens": 30},
        "output_tokens_details": {"thinking_tokens": 20}
    });
    let mut deployment = model_config(
        &mock.base_url(),
        "native-priced-alias",
        "anthropic/claude-sonnet-4-20250514",
    );
    deployment.oxllm_params.input_cost_per_token = Some(4e-6);
    deployment.oxllm_params.output_cost_per_reasoning_token = Some(3e-5);
    let mut config = authenticated_config(&mock.base_url());
    config.model_list = vec![deployment];
    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/messages")
            .header("x-api-key", UPSTREAM_KEY)
            .json_body(json!({"model": "claude-sonnet-4-20250514", "max_tokens": 128, "messages": [{"role": "user", "content": "Cache accounting"}]}));
        then.status(200).json_body(json!({
            "id": "msg-review-cache", "type": "message", "role": "assistant",
            "model": "claude-sonnet-4-20250514", "content": [{"type": "text", "text": "cached reply"}],
            "stop_reason": "end_turn", "usage": usage
        }));
    });
    let (handle, _) = common::start_oxllm(config).await.unwrap();
    let client = common::http_client();
    let (body, billed) = billed_post(
        &client,
        handle.url("/anthropic/v1/messages"),
        &json!({
            "model": "native-priced-alias", "max_tokens": 128,
            "messages": [{"role": "user", "content": "Cache accounting"}]
        }),
    )
    .await;
    assert_eq!(body["usage"], usage);
    let expected =
        100.0 * 4e-6 + 200.0 * 3e-7 + 50.0 * 3.75e-6 + 30.0 * 6e-6 + 30.0 * 1.5e-5 + 20.0 * 3e-5;
    assert_cost(
        billed,
        expected,
        "native additive cache and reasoning buckets",
    );
    for request in [
        json!({"model": "native-priced-alias", "usage": usage}),
        json!({"model": "native-priced-alias", "usage": {
            "prompt_tokens": 300, "completion_tokens": 50, "cache_read_input_tokens": 200,
            "cache_creation_input_tokens": 80, "cache_creation_1h_input_tokens": 30, "reasoning_tokens": 20
        }}),
        json!({"model": "claude-sonnet-4-20250514", "provider": "anthropic", "usage": usage,
            "pricing_override": {"input_cost_per_token": 4e-6, "output_cost_per_reasoning_token": 3e-5}}),
    ] {
        assert_cost(
            preview(&client, &handle, request).await,
            billed,
            "native preview parity",
        );
    }
    upstream.assert_hits(1);
    handle.shutdown();
}

#[tokio::test]
async fn opaque_profile_hint_prices_actual_chat_and_preview_without_changing_wire_arn() {
    let mock = MockServer::start();
    let arn = "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abc123";
    let underlying = "anthropic.claude-3-7-sonnet-20250219-v1:0";
    let mut deployment = model_config(
        &mock.base_url(),
        "priced-profile",
        &format!("bedrock/{arn}"),
    );
    deployment.oxllm_params.aws_access_key_id = Some("AKIDEXAMPLE".into());
    deployment.oxllm_params.aws_secret_access_key = Some("dummy-signing-key".into());
    deployment
        .oxllm_params
        .extra
        .insert("aws_region_name".into(), json!("us-east-1"));
    deployment.model_info = Some(ModelInfo {
        base_model: Some(underlying.into()),
        ..Default::default()
    });
    let mut config = authenticated_config(&mock.base_url());
    config.model_list = vec![deployment];
    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .matches(|request: &HttpMockRequest| {
                request.path.starts_with("/model/arn")
                    && request
                        .path
                        .contains("application-inference-profile%2Fabc123")
                    && request.path.ends_with("/converse")
            });
        then.status(200).json_body(
            json!({"output":{"message":{"role":"assistant","content":[{"text":"hinted"}]}},
            "usage":{"inputTokens":10,"outputTokens":5,"totalTokens":15},"stopReason":"end_turn"}),
        );
    });
    let (handle, _) = common::start_oxllm(config).await.unwrap();
    let client = common::http_client();
    let (_, billed) = billed_post(
        &client,
        handle.url("/v1/chat/completions"),
        &json!({"model":"priced-profile","messages":[{"role":"user","content":"hello"}]}),
    )
    .await;
    let expected = oxllm_core::calculate_cost_for_service_tier(
        underlying,
        &oxllm_core::UsageSummary {
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15,
            ..Default::default()
        },
        None,
        "bedrock",
        None,
    )
    .unwrap();
    assert_cost(billed, expected, "opaque profile serving hint");
    assert_cost(
        preview(
            &client,
            &handle,
            json!({"model":"priced-profile","usage":{"prompt_tokens":10,"completion_tokens":5}}),
        )
        .await,
        billed,
        "opaque profile preview",
    );
    upstream.assert_hits(1);
    handle.shutdown();
}

#[tokio::test]
async fn spend_calculate_nested_audio_reasoning_and_search_counts_match_native_bills() {
    let mock = MockServer::start();
    let mut config = authenticated_config(&mock.base_url());
    config.model_list = vec![
        model_config(
            &mock.base_url(),
            "audio-preview-alias",
            "openai/gpt-4o-audio-preview",
        ),
        model_config(
            &mock.base_url(),
            "search-preview-alias",
            "perplexity/sonar-deep-research",
        ),
    ];
    let (handle, _) = common::start_oxllm(config).await.unwrap();
    let client = common::http_client();
    for (provider, alias, model, usage, equivalent, expected) in [
        (
            "openai",
            "audio-preview-alias",
            "gpt-4o-audio-preview",
            json!({"prompt_tokens": 1_000, "completion_tokens": 200, "prompt_tokens_details": {"audio_tokens": 600}, "completion_tokens_details": {"reasoning_tokens": 80}}),
            json!({"prompt_tokens": 1_000, "completion_tokens": 200, "audio_input_tokens": 600, "reasoning_tokens": 80}),
            400.0 * 2.5e-6 + 600.0 * 4e-5 + 200.0 * 1e-5,
        ),
        (
            "perplexity",
            "search-preview-alias",
            "sonar-deep-research",
            json!({"input_tokens": 100, "output_tokens": 50, "output_tokens_details": {"thinking_tokens": 20}, "citation_tokens": 7, "prompt_tokens_details": {"web_search_requests": 3}}),
            json!({"prompt_tokens": 100, "completion_tokens": 50, "completion_tokens_details": {"reasoning_tokens": 20}, "citation_tokens": 7, "num_search_queries": 3}),
            100.0 * 2e-6 + 30.0 * 8e-6 + 20.0 * 3e-6 + 7.0 * 2e-6 + 3.0 * 0.005,
        ),
    ] {
        let upstream = mock.mock(|when, then| {
            when.method(httpmock::Method::POST)
                .path("/v1/chat/completions")
                .header("authorization", format!("Bearer {UPSTREAM_KEY}"))
                .json_body(json!({"model": model, "messages": [{"role": "user", "content": "Detailed usage"}]}));
            then.status(200).json_body(chat_response(model, usage.clone(), "detailed reply"));
        });
        let (body, billed) = billed_post(
            &client,
            handle.url(&format!("/{provider}/v1/chat/completions")),
            &json!({"model": alias, "messages": [{"role": "user", "content": "Detailed usage"}]}),
        )
        .await;
        assert_eq!(body["usage"], usage);
        assert_cost(billed, expected, "detailed native bill");
        for preview_usage in [usage, equivalent] {
            assert_cost(
                preview(
                    &client,
                    &handle,
                    json!({"model": alias, "usage": preview_usage}),
                )
                .await,
                billed,
                "detailed preview",
            );
        }
        upstream.assert_hits(1);
        upstream.delete_async().await;
    }
    handle.shutdown();
}

#[tokio::test]
async fn spend_calculate_explicit_duration_and_zero_token_precedence_are_unambiguous() {
    let mock = MockServer::start();
    let never = mock.mock(|_, then| {
        then.status(500);
    });
    let mut config = authenticated_config(&mock.base_url());
    let mut elapsed = model_config(
        &mock.base_url(),
        "elapsed-alias",
        "openai/review-unpriced-elapsed",
    );
    elapsed.oxllm_params.cost_per_second = Some(0.4);
    elapsed.oxllm_params.input_cost_per_second = Some(0.8);
    elapsed.oxllm_params.output_cost_per_second = Some(1.2);
    let mut free = model_config(
        &mock.base_url(),
        "free-alias",
        "openai/review-unpriced-free",
    );
    free.oxllm_params.input_cost_per_token = Some(0.0);
    free.oxllm_params.output_cost_per_token = Some(0.0);
    free.oxllm_params.cost_per_second = Some(100.0);
    config.model_list.extend([elapsed, free]);
    let (handle, _) = common::start_oxllm(config).await.unwrap();
    let client = common::http_client();
    for time_field in ["duration_ms", "response_time_ms"] {
        assert_cost(
            preview(
                &client,
                &handle,
                json!({"model": "elapsed-alias", "usage": {}, (time_field): 2_500.0}),
            )
            .await,
            1.0,
            "configured duration, billed once",
        );
    }
    let missing_duration = post_json(
        &client,
        handle.url("/spend/calculate"),
        &json!({"model": "elapsed-alias", "usage": {}}),
    )
    .await;
    assert!(
        missing_duration["cost"].is_null(),
        "elapsed time cannot be invented: {missing_duration}"
    );
    for rates in [
        json!({"cost_per_second": 0.125, "input_cost_per_second": 2.0, "output_cost_per_second": 3.0}),
        json!({"cost_per_second_rate": 0.125}),
        json!({"input_cost_per_second": 0.125, "output_cost_per_second": 3.0}),
        json!({"output_cost_per_second": 0.125}),
    ] {
        assert_cost(preview(&client, &handle, json!({
            "model": "review-unpriced-explicit", "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            "duration_ms": 2_500.0, "pricing_override": rates
        })).await, 0.3125, "explicit elapsed-time price");
    }
    for request in [
        json!({"model": "free-alias", "usage": {"prompt_tokens": 10, "completion_tokens": 5}, "duration_ms": 2_500.0}),
        json!({"model": "review-unpriced-explicit", "usage": {"prompt_tokens": 10, "completion_tokens": 5}, "duration_ms": 2_500.0,
            "pricing_override": {"input_cost_per_token": 0.0, "output_cost_per_token": 0.0, "cost_per_second": 100.0}}),
    ] {
        assert_cost(
            preview(&client, &handle, request).await,
            0.0,
            "explicit zero tokens beat duration",
        );
    }
    assert_cost(preview(&client, &handle, json!({
        "model": "gpt-4o", "usage": {"prompt_tokens": 1_000, "completion_tokens": 500}, "duration_ms": 2_500.0,
        "pricing_override": {"input_cost_per_token": 0.0, "cost_per_second": 100.0}
    })).await, 0.005, "zero input override still inherits output token price");
    never.assert_hits(0);
    handle.shutdown();
}

#[tokio::test]
async fn spend_calculate_rejects_invalid_rates_usage_and_preview_controls() {
    let mock = MockServer::start();
    let never = mock.mock(|_, then| {
        then.status(500);
    });
    let (handle, _) = common::start_oxllm(authenticated_config(&mock.base_url()))
        .await
        .unwrap();
    let client = common::http_client();
    let base = json!({"model": "gpt-4o", "usage": {"prompt_tokens": 10, "completion_tokens": 5}});
    for patch in [
        json!({"model": " "}),
        json!({"provider": " "}),
        json!({"service_tier": "auto"}),
        json!({"duration_ms": -1.0}),
        json!({"response_time_ms": -1.0}),
        json!({"deployment_id": "missing-configured-id"}),
        json!({"pricing_override": []}),
        json!({"pricing_override": {}}),
        json!({"pricing_override": {"input_cost_per_token": null}}),
        json!({"pricing_override": {"input_cost_per_token": "NaN"}}),
        json!({"pricing_override": {"input_cost_per_token": "Infinity"}}),
        json!({"pricing_override": {"input_cost_per_token": true}}),
        json!({"pricing_override": {"api_base": mock.base_url(), "input_cost_per_token": 1.0}}),
        json!({"pricing_override": {"cost_per_second": 1.0, "cost_per_second_rate": 1.0}}),
    ] {
        let mut request = base.clone();
        request
            .as_object_mut()
            .unwrap()
            .extend(patch.as_object().unwrap().clone());
        let body = response_json(
            post(&client, handle.url("/spend/calculate"), &request).await,
            400,
        )
        .await;
        assert!(body["error"]["message"].as_str().is_some(), "{body}");
    }
    for field in [
        "input_cost_per_token",
        "output_cost_per_token",
        "cache_read_input_token_cost",
        "cache_creation_input_token_cost",
        "cache_creation_input_token_cost_above_1hr",
        "output_cost_per_reasoning_token",
        "cost_per_second",
        "input_cost_per_second",
        "output_cost_per_second",
    ] {
        let mut request = base.clone();
        request["pricing_override"] = json!({(field): -0.001});
        response_json(
            post(&client, handle.url("/spend/calculate"), &request).await,
            400,
        )
        .await;
    }
    for usage in [
        json!([]),
        json!({"prompt_tokens": -1}),
        json!({"input_tokens": 1.5}),
        json!({"completion_tokens": "5"}),
        json!({"total_tokens": -1}),
        json!({"cache_read_input_tokens": -1}),
        json!({"cache_creation_input_tokens": -1}),
        json!({"prompt_tokens_details": []}),
        json!({"prompt_tokens_details": {"cached_tokens": -1}}),
        json!({"prompt_tokens_details": {"audio_tokens": -1}}),
        json!({"completion_tokens_details": {"reasoning_tokens": -1}}),
        json!({"output_tokens_details": {"thinking_tokens": -1}}),
        json!({"cache_creation": {"ephemeral_1h_input_tokens": -1}}),
        json!({"num_search_queries": -1}),
    ] {
        // DTO deserialization rejects invalid count shapes with Axum's 422.
        let response = post(
            &client,
            handle.url("/spend/calculate"),
            &json!({"model": "gpt-4o", "usage": usage}),
        )
        .await;
        let status = response.status().as_u16();
        let body = response.text().await.unwrap();
        assert_eq!(status, 422, "invalid usage accepted: {body}");
    }
    never.assert_hits(0);
    handle.shutdown();
}

#[tokio::test]
async fn spend_calculate_rejects_negative_nested_web_search_counts() {
    let mock = MockServer::start();
    let (handle, _) = common::start_oxllm(authenticated_config(&mock.base_url()))
        .await
        .unwrap();
    let client = common::http_client();
    for invalid in [json!(-1), json!(1.5), json!("3")] {
        let response = post(&client, handle.url("/spend/calculate"), &json!({
            "model": "sonar-deep-research", "provider": "perplexity",
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "prompt_tokens_details": {"web_search_requests": invalid}}
        })).await;
        let status = response.status().as_u16();
        let body = response.text().await.unwrap();
        assert_eq!(
            status, 422,
            "invalid nested search usage must not silently become free: {body}"
        );
    }
    handle.shutdown();
}

#[tokio::test]
async fn spend_calculate_ambiguous_alias_requires_provider_or_configured_deployment_selector() {
    let mock = MockServer::start();
    let never = mock.mock(|_, then| {
        then.status(500);
    });
    let mut config = authenticated_config(&mock.base_url());
    config.model_list = [
        ("openai", "cheap-card", 0.001, 0.002),
        ("openai", "premium-card", 0.003, 0.004),
        ("azure", "azure-card", 0.005, 0.006),
    ]
    .map(|(provider, id, input, output)| {
        let mut deployment = model_config(
            &mock.base_url(),
            "multi-card-alias",
            &format!("{provider}/gpt-4o"),
        );
        deployment.oxllm_params.input_cost_per_token = Some(input);
        deployment.oxllm_params.output_cost_per_token = Some(output);
        deployment.model_info = Some(ModelInfo {
            id: Some(id.into()),
            ..Default::default()
        });
        deployment
    })
    .to_vec();
    let (handle, _) = common::start_oxllm(config).await.unwrap();
    let client = common::http_client();
    let base = json!({"model": "multi-card-alias", "usage": {"prompt_tokens": 10, "completion_tokens": 5}});
    for selector in [json!({}), json!({"provider": "openai"})] {
        let mut request = base.clone();
        request
            .as_object_mut()
            .unwrap()
            .extend(selector.as_object().unwrap().clone());
        let error = response_json(
            post(&client, handle.url("/spend/calculate"), &request).await,
            400,
        )
        .await;
        assert!(
            error["error"]["message"]
                .as_str()
                .unwrap()
                .contains("multiple rate cards")
        );
    }
    for (selector, expected) in [
        (json!({"provider": "azure"}), 0.08),
        (json!({"deployment_id": "cheap-card"}), 0.02),
        (
            json!({"deployment_id": "premium-card", "provider": "openai"}),
            0.05,
        ),
        (json!({"deployment_id": "azure-card"}), 0.08),
        (
            json!({"pricing_override": {"input_cost_per_token": 0.007, "output_cost_per_token": 0.009}}),
            0.115,
        ),
    ] {
        let mut request = base.clone();
        request
            .as_object_mut()
            .unwrap()
            .extend(selector.as_object().unwrap().clone());
        assert_cost(
            preview(&client, &handle, request).await,
            expected,
            "disambiguated configured card without DB",
        );
    }
    for selector in [
        json!({"deployment_id": "missing-card"}),
        json!({"deployment_id": "cheap-card", "provider": "azure"}),
    ] {
        let mut request = base.clone();
        request
            .as_object_mut()
            .unwrap()
            .extend(selector.as_object().unwrap().clone());
        response_json(
            post(&client, handle.url("/spend/calculate"), &request).await,
            400,
        )
        .await;
    }
    never.assert_hits(0);
    handle.shutdown();
}

fn candidate_body(mock: &MockServer, alias: &str, owner: usize) -> Value {
    json!({
        "model_name": alias,
        "oxllm_params": {
            "model": format!("openai/review-native-{owner}"),
            "api_key": format!("sk-review-local-upstream-{owner}"),
            "api_base": mock.base_url(), "timeout": 5.0,
            "input_cost_per_token": 0.001 * (owner + 1) as f64,
            "output_cost_per_token": 0.002 * (owner + 1) as f64
        },
        "model_info": {"id": format!("review-owner-{owner}")}
    })
}

fn mock_candidate(mock: &MockServer, owner: usize) -> Mock<'_> {
    mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .header("authorization", format!("Bearer sk-review-local-upstream-{owner}"))
            .json_body_partial(json!({"model": format!("review-native-{owner}"), "messages": [{"role": "user", "content": "Who owns this alias?"}]}).to_string());
        then.status(200).json_body(chat_response(
            &format!("review-native-{owner}"),
            json!({"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}),
            &format!("owner-{owner}"),
        ));
    })
}

async fn race_posts(
    client: &Client,
    handle: &ManagementHandle,
    requests: Vec<(&str, Value)>,
) -> Vec<(u16, Value)> {
    let barrier = Arc::new(tokio::sync::Barrier::new(requests.len()));
    futures_util::future::join_all(requests.into_iter().map(|(path, body)| {
        let client = client.clone();
        let url = handle.url(path);
        let barrier = barrier.clone();
        async move {
            barrier.wait().await;
            let response = post(&client, url, &body).await;
            let status = response.status().as_u16();
            (status, response_json(response, status).await)
        }
    }))
    .await
}

fn single_claim_winner(results: &[(u16, Value)]) -> usize {
    let successes: Vec<_> = results
        .iter()
        .enumerate()
        .filter(|(_, (status, _))| *status == 200)
        .collect();
    assert_eq!(
        successes.len(),
        1,
        "exactly one alias claim succeeds: {results:?}"
    );
    let winner = successes[0].0;
    for (index, (status, body)) in results.iter().enumerate() {
        if index != winner {
            assert_eq!(
                *status, 400,
                "expected name conflicts, not storage failures: {results:?}"
            );
            assert!(
                body["error"]["message"]
                    .as_str()
                    .unwrap()
                    .contains("model_name"),
                "{body}"
            );
        }
    }
    winner
}

async fn assert_inventory_matches_storage(
    client: &Client,
    handle: &ManagementHandle,
) -> Vec<ModelConfigRecord> {
    let records = handle.db.list_all_models_db().await.unwrap();
    let stored: BTreeMap<_, _> = records
        .iter()
        .map(|record| (record.model_name.clone(), record.model_id.clone()))
        .collect();
    assert_eq!(
        stored.len(),
        records.len(),
        "no duplicate stored alias owners"
    );
    let listed = get_json(client, handle.url("/model/list")).await;
    assert_eq!(listed["truncated"], false);
    let data = listed["data"].as_array().unwrap();
    let live: BTreeMap<_, _> = data
        .iter()
        .map(|row| {
            (
                row["id"].as_str().unwrap().to_string(),
                row["model_id"].as_str().unwrap().to_string(),
            )
        })
        .collect();
    assert_eq!(live.len(), data.len(), "no duplicate listed alias owners");
    assert_eq!(
        live, stored,
        "stored claims and the live inventory disagree"
    );
    records
}

async fn assert_alias_routes_to_owner(
    client: &Client,
    handle: &ManagementHandle,
    alias: &str,
    owner: usize,
) {
    let info = get_json(client, handle.url(&format!("/model/info?model={alias}"))).await;
    assert_eq!(
        info["provider_model"],
        format!("openai/review-native-{owner}")
    );
    let (body, cost) = billed_post(
        client,
        handle.url("/v1/chat/completions"),
        &json!({"model": alias, "messages": [{"role": "user", "content": "Who owns this alias?"}]}),
    )
    .await;
    assert_eq!(body["model"], alias);
    assert_eq!(
        body["choices"][0]["message"]["content"],
        format!("owner-{owner}")
    );
    assert_cost(
        cost,
        0.02 * (owner + 1) as f64,
        "stored owner pricing must be routable",
    );
}

fn assert_record_unchanged(current: &ModelConfigRecord, previous: &ModelConfigRecord) {
    assert_eq!(current.model_id, previous.model_id);
    assert_eq!(current.model_name, previous.model_name);
    assert_eq!(current.oxllm_params, previous.oxllm_params);
    assert_eq!(current.model_info, previous.model_info);
    assert_eq!(current.created_at, previous.created_at);
    assert_eq!(current.updated_at, previous.updated_at);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_model_creates_keep_one_stored_routable_owner_and_release_deleted_claim() {
    let mock = MockServer::start();
    let mocks: Vec<_> = (0..8).map(|owner| mock_candidate(&mock, owner)).collect();
    let mut config = authenticated_config(&mock.base_url());
    config.model_list.clear();
    let handle = start_oxllm_with_db(config).await;
    let client = common::http_client();
    let alias = "concurrent-create-alias";
    let results = race_posts(
        &client,
        &handle,
        (0..8)
            .map(|owner| ("/model/new", candidate_body(&mock, alias, owner)))
            .collect(),
    )
    .await;
    let winner = single_claim_winner(&results);
    let winner_id = results[winner].1["model_id"].as_str().unwrap();
    let records = assert_inventory_matches_storage(&client, &handle).await;
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].model_id, winner_id);
    assert_eq!(records[0].model_name, alias);
    assert_eq!(
        records[0].oxllm_params["model"],
        format!("openai/review-native-{winner}")
    );
    assert_alias_routes_to_owner(&client, &handle, alias, winner).await;
    for (index, upstream) in mocks.iter().enumerate() {
        upstream.assert_hits(usize::from(index == winner));
    }
    post_json(
        &client,
        handle.url("/model/delete"),
        &json!({"model_id": winner_id}),
    )
    .await;
    assert!(
        assert_inventory_matches_storage(&client, &handle)
            .await
            .is_empty()
    );
    response_json(post(&client, handle.url("/v1/chat/completions"), &json!({"model": alias, "messages": [{"role": "user", "content": "Who owns this alias?"}]})).await, 404).await;
    let successor = (winner + 1) % mocks.len();
    let created = post_json(
        &client,
        handle.url("/model/new"),
        &candidate_body(&mock, alias, successor),
    )
    .await;
    assert_ne!(created["model_id"].as_str().unwrap(), winner_id);
    let records = assert_inventory_matches_storage(&client, &handle).await;
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].model_id, created["model_id"].as_str().unwrap());
    assert_alias_routes_to_owner(&client, &handle, alias, successor).await;
    for (index, upstream) in mocks.iter().enumerate() {
        upstream.assert_hits(usize::from(index == winner || index == successor));
    }
    let db = handle.db.clone();
    handle.shutdown().await;
    let logs = db.query_spend(&Default::default()).await.unwrap();
    assert_eq!(
        logs.len(),
        2,
        "both successive owners' successful bills persist"
    );
    let mut stored_costs = Vec::new();
    for log in logs {
        assert_eq!(log.provider, "openai");
        assert_eq!(log.status.as_deref(), Some("success"));
        assert_eq!(log.prompt_tokens, Some(10));
        assert_eq!(log.completion_tokens, Some(5));
        assert_eq!(log.total_tokens, Some(15));
        stored_costs.push(log.cost.expect("durable billed cost"));
    }
    stored_costs.sort_by(f64::total_cmp);
    let mut expected_costs = [0.02 * (winner + 1) as f64, 0.02 * (successor + 1) as f64];
    expected_costs.sort_by(f64::total_cmp);
    for (stored, expected) in stored_costs.into_iter().zip(expected_costs) {
        assert_cost(
            stored,
            expected,
            "deleted/reclaimed alias keeps original bills",
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn competing_model_renames_leave_loser_unchanged_and_release_only_winners_old_alias() {
    let mock = MockServer::start();
    let mocks: Vec<_> = (0..3).map(|owner| mock_candidate(&mock, owner)).collect();
    let mut config = authenticated_config(&mock.base_url());
    config.model_list.clear();
    let handle = start_oxllm_with_db(config).await;
    let client = common::http_client();
    let mut previous = Vec::new();
    for (owner, alias) in ["rename-source-a", "rename-source-b"]
        .into_iter()
        .enumerate()
    {
        let created = post_json(
            &client,
            handle.url("/model/new"),
            &candidate_body(&mock, alias, owner),
        )
        .await;
        previous.push(
            handle
                .db
                .get_model(created["model_id"].as_str().unwrap())
                .await
                .unwrap()
                .unwrap(),
        );
    }
    let target = "rename-claimed-alias";
    let results = race_posts(&client, &handle, previous.iter().enumerate().map(|(owner, record)| ("/model/update", json!({
        "model_id": record.model_id, "model_name": target,
        "oxllm_params": {"weight": owner + 5}, "model_info": {"id": format!("renamed-owner-{owner}")}
    }))).collect()).await;
    let winner = single_claim_winner(&results);
    let loser = 1 - winner;
    let records = assert_inventory_matches_storage(&client, &handle).await;
    assert_eq!(records.len(), 2);
    let current_loser = handle
        .db
        .get_model(&previous[loser].model_id)
        .await
        .unwrap()
        .unwrap();
    assert_record_unchanged(&current_loser, &previous[loser]);
    let current_winner = handle
        .db
        .get_model(&previous[winner].model_id)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(current_winner.model_name, target);
    assert_eq!(current_winner.oxllm_params["weight"], winner + 5);
    assert_eq!(
        current_winner.model_info.unwrap()["id"],
        format!("renamed-owner-{winner}")
    );
    assert_alias_routes_to_owner(&client, &handle, target, winner).await;
    assert_alias_routes_to_owner(&client, &handle, &previous[loser].model_name, loser).await;
    response_json(post(&client, handle.url("/v1/chat/completions"), &json!({"model": previous[winner].model_name, "messages": [{"role": "user", "content": "Who owns this alias?"}]})).await, 404).await;
    // The failed rename did not release its source; the successful one did.
    response_json(
        post(
            &client,
            handle.url("/model/new"),
            &candidate_body(&mock, &previous[loser].model_name, 2),
        )
        .await,
        400,
    )
    .await;
    post_json(
        &client,
        handle.url("/model/new"),
        &candidate_body(&mock, &previous[winner].model_name, 2),
    )
    .await;
    let records = assert_inventory_matches_storage(&client, &handle).await;
    assert_eq!(records.len(), 3);
    assert_alias_routes_to_owner(&client, &handle, &previous[winner].model_name, 2).await;
    for upstream in mocks {
        upstream.assert_hits(1);
    }
    handle.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn model_rename_racing_create_keeps_one_alias_owner_and_coherent_stored_routes() {
    let mock = MockServer::start();
    let source_mock = mock_candidate(&mock, 0);
    let create_mock = mock_candidate(&mock, 1);
    let mut config = authenticated_config(&mock.base_url());
    config.model_list.clear();
    let handle = start_oxllm_with_db(config).await;
    let client = common::http_client();
    let created = post_json(
        &client,
        handle.url("/model/new"),
        &candidate_body(&mock, "rename-source", 0),
    )
    .await;
    let previous = handle
        .db
        .get_model(created["model_id"].as_str().unwrap())
        .await
        .unwrap()
        .unwrap();
    let alias = "rename-versus-create";
    let results = race_posts(&client, &handle, vec![
        ("/model/update", json!({"model_id": previous.model_id, "model_name": alias, "oxllm_params": {"weight": 7}, "model_info": {"id": "renamed-source"}})),
        ("/model/new", candidate_body(&mock, alias, 1)),
    ]).await;
    let winner = single_claim_winner(&results);
    let records = assert_inventory_matches_storage(&client, &handle).await;
    let owners: Vec<_> = records
        .iter()
        .filter(|record| record.model_name == alias)
        .collect();
    assert_eq!(owners.len(), 1);
    assert_eq!(
        owners[0].oxllm_params["model"],
        format!("openai/review-native-{winner}")
    );
    let current_source = handle
        .db
        .get_model(&previous.model_id)
        .await
        .unwrap()
        .unwrap();
    if winner == 0 {
        assert_eq!(records.len(), 1);
        assert_eq!(owners[0].model_id, previous.model_id);
        assert_eq!(current_source.oxllm_params["weight"], 7);
        assert_eq!(current_source.model_info.unwrap()["id"], "renamed-source");
        response_json(post(&client, handle.url("/v1/chat/completions"), &json!({"model": "rename-source", "messages": [{"role": "user", "content": "Who owns this alias?"}]})).await, 404).await;
    } else {
        assert_eq!(records.len(), 2);
        assert_eq!(
            owners[0].model_id,
            results[1].1["model_id"].as_str().unwrap()
        );
        assert_record_unchanged(&current_source, &previous);
        assert_alias_routes_to_owner(&client, &handle, "rename-source", 0).await;
    }
    assert_alias_routes_to_owner(&client, &handle, alias, winner).await;
    // Failed contenders cannot overwrite the winner on a subsequent attempt.
    response_json(
        post(
            &client,
            handle.url("/model/new"),
            &candidate_body(&mock, alias, 1),
        )
        .await,
        400,
    )
    .await;
    if winner == 1 {
        response_json(
            post(
                &client,
                handle.url("/model/update"),
                &json!({"model_id": previous.model_id, "model_name": alias}),
            )
            .await,
            400,
        )
        .await;
        assert_record_unchanged(
            &handle
                .db
                .get_model(&previous.model_id)
                .await
                .unwrap()
                .unwrap(),
            &previous,
        );
    }
    let final_records = assert_inventory_matches_storage(&client, &handle).await;
    assert_eq!(
        serde_json::to_value(final_records).unwrap(),
        serde_json::to_value(records).unwrap(),
        "failed claims must not mutate any stored record"
    );
    source_mock.assert_hits(1);
    create_mock.assert_hits(usize::from(winner == 1));
    handle.shutdown().await;
}
