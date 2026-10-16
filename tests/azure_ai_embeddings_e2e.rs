//! Azure model-inference image embedding wire contracts.

mod common;

use std::sync::Arc;
use std::time::Duration;

use httpmock::{Method::POST, MockServer};
use oxllm_server::hooks::{BudgetHook, CacheHook, CostTrackingHook};
use oxllm_server::spend_writer::SpendEntry;
use serde_json::{Value, json};
use tokio::sync::mpsc;

const IMAGE: &str = "data:image/png;base64,aGVsbG8=";

fn config(upstream: &MockServer) -> oxllm_server::ProxyConfig {
    let mut config = common::proxy_config(&upstream.url("/models"));
    config.model_list[0].model_name = "embeddings".into();
    config.model_list[0].oxllm_params.model = "azure_ai/embed-v-4-0".into();
    config.model_list[0].oxllm_params.api_version = Some("2024-05-01-preview".into());
    config.router_settings.routing.num_retries = Some(0);
    config
}

fn capture_spend(state: &oxllm_server::SharedState) -> mpsc::Receiver<SpendEntry> {
    let mut next = (*state.load_full()).clone();
    let model = next.config.model_list[0].model_name.clone();
    let settings = &mut Arc::make_mut(&mut next.config).oxllm_settings;
    settings.max_budget = Some(100.0);
    settings.budget_duration = Some("1h".into());
    let budgets = std::collections::HashMap::from([(model, 100.0)]);
    settings.model_max_budget = Some(budgets.clone());
    let (tx, rx) = mpsc::channel(8);
    next.spend_tx = Some(tx.clone());
    next.hooks = vec![
        Arc::new(CacheHook::new(next.cache.clone(), Duration::from_secs(60))),
        Arc::new(BudgetHook::new(next.cache.clone(), Some(100.0), "1h").with_model_budget(budgets)),
        Arc::new(CostTrackingHook::with_spend_writer(Some(tx))),
    ];
    state.store(Arc::new(next));
    rx
}

fn request_id(response: &reqwest::Response) -> String {
    let id = response.headers()["x-oxllm-request-id"].to_str().unwrap();
    assert_eq!(response.headers()["x-request-id"], id);
    assert_eq!(response.headers()["x-litellm-call-id"], id);
    id.to_string()
}

fn response(vectors: Vec<Value>, tokens: u64) -> Value {
    json!({
        "object":"list", "model":"embed-v4.0", "data":vectors,
        "usage":{"prompt_tokens":tokens, "total_tokens":tokens}
    })
}

fn embedding(index: usize, vector: Value) -> Value {
    json!({"object":"embedding", "index":index, "embedding":vector})
}

#[tokio::test]
async fn embedding_spend_records_the_selected_model_for_text_images_and_mixed_batches() {
    for (configured_model, requested_model) in [
        ("azure_ai/embed-v-4-0", "embeddings"),
        ("azure_ai/model_router/embed-v-4-0", "embeddings"),
        ("azure_ai/*", "azure_ai/embed-v-4-0"),
    ] {
        let upstream = MockServer::start_async().await;
        let text = upstream
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/models/embeddings")
                    .json_body(json!({"model":"embed-v-4-0", "input":["text"]}));
                then.status(200)
                    .json_body(response(vec![embedding(0, json!([1]))], 2));
            })
            .await;
        let image = upstream
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/models/images/embeddings")
                    .json_body(json!({"model":"embed-v-4-0", "input":[{"image":IMAGE}]}));
                then.status(200)
                    .json_body(response(vec![embedding(0, json!([2]))], 3));
            })
            .await;
        let mut config = config(&upstream);
        let params = &mut config.model_list[0].oxllm_params;
        params.model = configured_model.into();
        params.input_cost_per_token = Some(0.001);
        params.output_cost_per_token = Some(0.0);
        if configured_model.contains('*') {
            config.model_list[0].model_name = "*".into();
        }
        let (handle, state) = common::start_oxllm(config).await.unwrap();
        let mut ledger = capture_spend(&state);
        let client = common::http_client();
        let mut ids = std::collections::HashSet::new();
        for (input, tokens, vectors) in [
            (json!("text"), 2, json!([embedding(0, json!([1]))])),
            (json!(IMAGE), 3, json!([embedding(0, json!([2]))])),
            (
                json!([IMAGE, "text"]),
                5,
                json!([embedding(0, json!([2])), embedding(1, json!([1]))]),
            ),
        ] {
            let result = client
                .post(handle.url("/v1/embeddings"))
                .json(&json!({"model":requested_model,"input":input}))
                .send()
                .await
                .unwrap();
            let status = result.status();
            let id = request_id(&result);
            let body: Value = result.json().await.unwrap();
            assert_eq!(status, 200, "{configured_model}: {body}");
            assert_eq!(body["model"], requested_model);
            assert_eq!(body["data"], vectors);
            assert_eq!(body["usage"]["total_tokens"], tokens);
            let entry = tokio::time::timeout(Duration::from_secs(5), ledger.recv())
                .await
                .unwrap()
                .unwrap();
            assert_eq!(entry.request_id.as_deref(), Some(id.as_str()));
            assert_eq!(entry.model, "embed-v-4-0");
            assert_eq!(entry.requested_model.as_deref(), Some(requested_model));
            assert_eq!(entry.provider, "azure_ai");
            assert_eq!(entry.prompt_tokens, Some(tokens));
            assert_eq!(entry.total_tokens, Some(tokens));
            assert!((entry.cost.unwrap() - tokens as f64 * 0.001).abs() < 1e-12);
            assert!(ids.insert(id), "every spend event belongs to a new request");
        }
        assert!(ledger.try_recv().is_err());
        text.assert_hits_async(2).await;
        image.assert_hits_async(2).await;
        handle.shutdown();
    }
}

#[tokio::test]
async fn azure_chat_spend_preserves_dated_upstream_snapshots() {
    let upstream = MockServer::start_async().await;
    let chat = upstream.mock_async(|when, then| {
        when.method(POST).path("/models/chat/completions")
            .json_body(json!({"model":"gpt-5.5","messages":[{"role":"user","content":"Hello"}]}));
        then.status(200).json_body(json!({"id":"chat-snapshot","model":"gpt-5.5-2026-04-23","choices":[{"index":0,"message":{"role":"assistant","content":"Hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5}}));
    }).await;
    let mut config = config(&upstream);
    config.model_list[0].model_name = "chat".into();
    let params = &mut config.model_list[0].oxllm_params;
    params.model = "azure_ai/gpt-5.5".into();
    params.input_cost_per_token = Some(0.001);
    params.output_cost_per_token = Some(0.002);
    let (handle, state) = common::start_oxllm(config).await.unwrap();
    let mut ledger = capture_spend(&state);
    let result = common::http_client()
        .post(handle.url("/v1/chat/completions"))
        .json(&json!({"model":"chat","messages":[{"role":"user","content":"Hello"}]}))
        .send()
        .await
        .unwrap();
    let status = result.status();
    let id = request_id(&result);
    let body: Value = result.json().await.unwrap();
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["model"], "chat");
    let entry = tokio::time::timeout(Duration::from_secs(5), ledger.recv())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(entry.request_id.as_deref(), Some(id.as_str()));
    assert_eq!(entry.model, "gpt-5.5-2026-04-23");
    assert_eq!(entry.requested_model.as_deref(), Some("chat"));
    assert!((entry.cost.unwrap() - 0.008).abs() < 1e-12);
    chat.assert_hits_async(1).await;
    handle.shutdown();
}

#[tokio::test]
async fn azure_image_embeddings_use_the_image_route_model_and_version() {
    let upstream = MockServer::start_async().await;
    let image = upstream
        .mock_async(|when, then| {
            when.method(POST)
                .path("/models/images/embeddings")
                .query_param("api-version", "2024-05-01-preview")
                .header("authorization", "Bearer sk-test-key")
                .json_body(json!({
                    "model":"embed-v-4-0", "input":[{"image":IMAGE}],
                    "dimensions":3, "encoding_format":"float"
                }));
            then.status(200)
                .header("llm_provider-num_tokens", "11")
                .json_body(response(vec![embedding(0, json!([0.1, 0.2, 0.3]))], 2));
        })
        .await;
    let text = upstream
        .mock_async(|when, then| {
            when.method(POST).path("/models/embeddings");
            then.status(400)
                .json_body(json!({"error":{"message":"images must not use text embeddings"}}));
        })
        .await;
    let (handle, _) = common::start_oxllm(config(&upstream)).await.unwrap();
    let result = common::http_client()
        .post(handle.url("/v1/embeddings"))
        .json(&json!({
            "model":"embeddings", "input":[IMAGE], "input_type":"image",
            "dimensions":3, "encoding_format":"float", "user":"test-user",
            "tags":["image-test"], "metadata":{"ignored":true}
        }))
        .send()
        .await
        .unwrap();
    let status = result.status();
    let body: Value = result.json().await.unwrap();
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["data"], json!([embedding(0, json!([0.1, 0.2, 0.3]))]));
    assert_eq!(
        body["usage"],
        json!({"prompt_tokens":11, "total_tokens":11})
    );
    image.assert_hits_async(1).await;
    text.assert_hits_async(0).await;
    handle.shutdown();
}

#[tokio::test]
async fn image_options_cannot_be_restored_by_text_leg_allowlists_or_overrides() {
    let upstream = MockServer::start_async().await;
    let image = upstream.mock_async(|when, then| {
        when.method(POST).path("/models/images/embeddings")
            .header("x-configured-header", "kept")
            .json_body(json!({"model":"embed-v-4-0", "input":[{"image":IMAGE}], "dimensions":7, "encoding_format":"float"}));
        then.status(200).json_body(response(vec![embedding(0, json!([1]))], 2));
    }).await;
    let mut config = config(&upstream);
    let params = &mut config.model_list[0].oxllm_params;
    params.allowed_openai_params = Some(vec!["input_type".into(), "private_option".into()]);
    params.extra_body = Some(serde_json::from_value(json!({"dimensions":7, "input_type":"image", "private_option":true, "model":"forged-model"})).unwrap());
    params.extra_headers = Some(std::collections::HashMap::from([(
        "x-configured-header".into(),
        "kept".into(),
    )]));
    let (handle, _) = common::start_oxllm(config).await.unwrap();
    let result = common::http_client().post(handle.url("/v1/embeddings"))
        .json(&json!({"model":"embeddings", "input":[IMAGE], "input_type":"image", "private_option":true, "dimensions":3, "encoding_format":"float"})).send().await.unwrap();
    let status = result.status();
    let body: Value = result.json().await.unwrap();
    assert_eq!(status, 200, "{body}");
    image.assert_hits_async(1).await;
    handle.shutdown();
}

#[tokio::test]
async fn oversized_image_error_bodies_preserve_provider_status_and_backoff() {
    let upstream = MockServer::start_async().await;
    let image = upstream
        .mock_async(|when, then| {
            when.method(POST).path("/models/images/embeddings");
            then.status(429)
                .header("retry-after", "17")
                .json_body(json!({"error":{"message":"x".repeat(20_000)}}));
        })
        .await;
    let (handle, _) = common::start_oxllm(config(&upstream)).await.unwrap();
    let result = common::http_client()
        .post(handle.url("/v1/embeddings"))
        .json(&json!({"model":"embeddings", "input":[IMAGE]}))
        .send()
        .await
        .unwrap();
    assert_eq!(result.status(), 429);
    assert_eq!(result.headers()["retry-after"], "17");
    request_id(&result);
    assert_eq!(
        result.json::<Value>().await.unwrap()["error"]["code"],
        "upstream_rate_limited"
    );
    image.assert_hits_async(1).await;
    handle.shutdown();
}

#[tokio::test]
async fn failed_text_batches_never_send_images_or_cache_a_response() {
    let upstream = MockServer::start_async().await;
    let text = upstream
        .mock_async(|when, then| {
            when.method(POST)
                .path("/models/embeddings")
                .json_body(json!({"model":"embed-v-4-0", "input":["text"]}));
            then.status(500)
                .json_body(json!({"error":{"message":"text capacity unavailable"}}));
        })
        .await;
    let image = upstream
        .mock_async(|when, then| {
            when.method(POST).path("/models/images/embeddings");
            then.status(200)
                .json_body(response(vec![embedding(0, json!([1]))], 2));
        })
        .await;
    let mut config = config(&upstream);
    config.router_settings.routing.cooldown_time = Some(0);
    let (handle, state) = common::start_oxllm(config).await.unwrap();
    let _ledger = capture_spend(&state);
    for _ in 0..2 {
        let result = common::http_client()
            .post(handle.url("/v1/embeddings"))
            .json(
                &json!({"model":"embeddings", "input":[IMAGE, "text"], "cache":{"use_cache":true}}),
            )
            .send()
            .await
            .unwrap();
        assert_eq!(result.status(), 500);
        assert!(!result.headers().contains_key("x-oxllm-cache-hit"));
        assert_eq!(
            result.json::<Value>().await.unwrap()["error"]["message"],
            "text capacity unavailable"
        );
    }
    text.assert_hits_async(2).await;
    image.assert_hits_async(0).await;
    handle.shutdown();
}

#[tokio::test]
async fn mixed_batches_keep_valid_text_options_and_apply_supported_overrides_per_leg() {
    let upstream = MockServer::start_async().await;
    let text = upstream.mock_async(|when, then| {
        when.method(POST).path("/models/embeddings").header("x-configured-header", "kept")
            .json_body(json!({"model":"embed-v-4-0", "input":["text"], "dimensions":7, "encoding_format":"float", "input_type":"query"}));
        then.status(200).json_body(response(vec![embedding(0, json!([1]))], 2));
    }).await;
    let image = upstream.mock_async(|when, then| {
        when.method(POST).path("/models/images/embeddings").header("x-configured-header", "kept")
            .json_body(json!({"model":"embed-v-4-0", "input":[{"image":IMAGE}], "dimensions":7, "encoding_format":"float"}));
        then.status(200).json_body(response(vec![embedding(0, json!([2]))], 3));
    }).await;
    let mut config = config(&upstream);
    let params = &mut config.model_list[0].oxllm_params;
    params.allowed_openai_params = Some(vec!["input_type".into()]);
    params.extra_body =
        Some(serde_json::from_value(json!({"dimensions":7, "model":"forged-model"})).unwrap());
    params.extra_headers = Some(std::collections::HashMap::from([(
        "x-configured-header".into(),
        "kept".into(),
    )]));
    let (handle, _) = common::start_oxllm(config).await.unwrap();
    let result = common::http_client().post(handle.url("/v1/embeddings"))
        .json(&json!({"model":"embeddings", "input":["text", IMAGE], "input_type":"query", "dimensions":3, "encoding_format":"float"})).send().await.unwrap();
    assert_eq!(result.status(), 200, "{}", result.text().await.unwrap());
    assert_eq!(
        result.json::<Value>().await.unwrap()["usage"]["total_tokens"],
        5
    );
    text.assert_hits_async(1).await;
    image.assert_hits_async(1).await;
    handle.shutdown();
}

#[tokio::test]
async fn mixed_batches_share_one_attempt_timeout_instead_of_resetting_it_per_leg() {
    let upstream = MockServer::start_async().await;
    let text = upstream
        .mock_async(|when, then| {
            when.method(POST).path("/models/embeddings");
            then.status(200)
                .delay(Duration::from_millis(650))
                .json_body(response(vec![embedding(0, json!([1]))], 2));
        })
        .await;
    let image = upstream
        .mock_async(|when, then| {
            when.method(POST).path("/models/images/embeddings");
            then.status(200)
                .delay(Duration::from_millis(650))
                .json_body(response(vec![embedding(0, json!([2]))], 3));
        })
        .await;
    let (handle, _) = common::start_oxllm(config(&upstream)).await.unwrap();
    let result = common::http_client()
        .post(handle.url("/v1/embeddings"))
        .json(&json!({"model":"embeddings", "input":["text", IMAGE], "timeout":1.0}))
        .send()
        .await
        .unwrap();
    assert_eq!(result.status(), 504);
    request_id(&result);
    assert!(result.json::<Value>().await.unwrap().get("data").is_none());
    text.assert_hits_async(1).await;
    image.assert_hits_async(1).await;
    handle.shutdown();
}

#[tokio::test]
async fn wildcard_image_models_are_resolved_before_native_batch_preparation() {
    let upstream = MockServer::start_async().await;
    let image = upstream
        .mock_async(|when, then| {
            when.method(POST)
                .path("/models/images/embeddings")
                .json_body(json!({"model":"embed-v-4-0", "input":[{"image":IMAGE}]}));
            then.status(200)
                .json_body(response(vec![embedding(0, json!([1]))], 2));
        })
        .await;
    let mut config = config(&upstream);
    config.model_list[0].model_name = "*".into();
    config.model_list[0].oxllm_params.model = "azure_ai/*".into();
    let (handle, _) = common::start_oxllm(config).await.unwrap();
    let result = common::http_client()
        .post(handle.url("/v1/embeddings"))
        .json(&json!({"model":"azure_ai/embed-v-4-0", "input":IMAGE}))
        .send()
        .await
        .unwrap();
    assert_eq!(result.status(), 200, "{}", result.text().await.unwrap());
    assert_eq!(
        result.headers()["x-oxllm-model-name"],
        "azure_ai/embed-v-4-0"
    );
    assert_eq!(
        result.json::<Value>().await.unwrap()["model"],
        "azure_ai/embed-v-4-0"
    );
    image.assert_hits_async(1).await;
    handle.shutdown();
}

#[tokio::test]
async fn non_cohere_scalar_inputs_are_normalized_without_changing_the_text_route() {
    let upstream = MockServer::start_async().await;
    let text = upstream
        .mock_async(|when, then| {
            when.method(POST)
                .path("/models/embeddings")
                .json_body(json!({"model":"text-embedding-3-large", "input":["scalar text"]}));
            then.status(200)
                .json_body(response(vec![embedding(0, json!([1]))], 2));
        })
        .await;
    let mut config = config(&upstream);
    config.model_list[0].oxllm_params.model = "azure_ai/text-embedding-3-large".into();
    let (handle, _) = common::start_oxllm(config).await.unwrap();
    let result = common::http_client()
        .post(handle.url("/v1/embeddings"))
        .json(&json!({"model":"embeddings", "input":"scalar text"}))
        .send()
        .await
        .unwrap();
    assert_eq!(result.status(), 200);
    text.assert_hits_async(1).await;
    handle.shutdown();
}

#[tokio::test]
async fn provider_prefixed_embedding_routes_keep_native_payloads_and_require_the_native_image_path()
{
    let upstream = MockServer::start_async().await;
    let text = upstream
        .mock_async(|when, then| {
            when.method(POST).path("/models/embeddings").json_body(
                json!({"model":"embed-v-4-0", "input":[IMAGE, "text"], "input_type":"image"}),
            );
            then.status(400)
                .json_body(json!({"error":{"message":"use the native image endpoint and body"}}));
        })
        .await;
    let image = upstream.mock_async(|when, then| {
        when.method(POST).path("/models/images/embeddings").query_param("api-version", "2024-05-01-preview")
            .json_body(json!({"model":"embed-v-4-0", "input":[{"image":IMAGE}], "encoding_format":"float"}));
        then.status(200).json_body(response(vec![embedding(0, json!([1]))], 2));
    }).await;
    let (handle, _) = common::start_oxllm(config(&upstream)).await.unwrap();
    let client = common::http_client();
    let untranslated = client
        .post(handle.url("/azure_ai/embeddings"))
        .json(&json!({"model":"embeddings", "input":[IMAGE, "text"], "input_type":"image"}))
        .send()
        .await
        .unwrap();
    assert_eq!(untranslated.status(), 400);
    image.assert_hits_async(0).await;
    let missing_model = client
        .post(handle.url("/azure_ai/images/embeddings"))
        .json(&json!({"input":[{"image":IMAGE}]}))
        .send()
        .await
        .unwrap();
    assert_eq!(missing_model.status(), 400);
    image.assert_hits_async(0).await;
    let native = client
        .post(handle.url("/azure_ai/images/embeddings?api-version=2024-05-01-preview"))
        .json(&json!({"model":"embeddings", "input":[{"image":IMAGE}], "encoding_format":"float"}))
        .send()
        .await
        .unwrap();
    assert_eq!(native.status(), 200, "{}", native.text().await.unwrap());
    text.assert_hits_async(1).await;
    image.assert_hits_async(1).await;
    handle.shutdown();
}

#[tokio::test]
async fn concurrent_mixed_requests_keep_independent_vectors_usage_and_spend() {
    let upstream = MockServer::start_async().await;
    let other_image = "data:image/jpeg;base64,/9j/";
    let mut mocks = Vec::new();
    for (image, text, number) in [(IMAGE, "first", 10), (other_image, "second", 20)] {
        mocks.push(
            upstream
                .mock_async(|when, then| {
                    when.method(POST)
                        .path("/models/embeddings")
                        .json_body(json!({"model":"embed-v-4-0", "input":[text]}));
                    then.status(200)
                        .json_body(response(vec![embedding(0, json!([number + 1]))], 2));
                })
                .await,
        );
        mocks.push(
            upstream
                .mock_async(|when, then| {
                    when.method(POST)
                        .path("/models/images/embeddings")
                        .json_body(json!({"model":"embed-v-4-0", "input":[{"image":image}]}));
                    then.status(200)
                        .json_body(response(vec![embedding(0, json!([number]))], 2));
                })
                .await,
        );
    }
    let mut config = config(&upstream);
    config.model_list[0].oxllm_params.input_cost_per_token = Some(0.001);
    config.model_list[0].oxllm_params.output_cost_per_token = Some(0.0);
    let (handle, state) = common::start_oxllm(config).await.unwrap();
    let mut ledger = capture_spend(&state);
    let client = common::http_client();
    let first = client
        .post(handle.url("/v1/embeddings"))
        .json(&json!({"model":"embeddings", "input":[IMAGE, "first"]}));
    let second = client
        .post(handle.url("/v1/embeddings"))
        .json(&json!({"model":"embeddings", "input":[other_image, "second"]}));
    let (first, second) = tokio::join!(first.send(), second.send());
    for (result, number) in [(first.unwrap(), 10), (second.unwrap(), 20)] {
        assert_eq!(result.status(), 200);
        let body = result.json::<Value>().await.unwrap();
        assert_eq!(
            body["data"],
            json!([
                embedding(0, json!([number])),
                embedding(1, json!([number + 1]))
            ])
        );
        assert_eq!(body["usage"]["total_tokens"], 4);
    }
    let mut ids = std::collections::HashSet::new();
    for _ in 0..2 {
        let entry = tokio::time::timeout(Duration::from_secs(5), ledger.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(entry.total_tokens, Some(4));
        assert!((entry.cost.unwrap() - 0.004).abs() < 1e-12);
        ids.insert(entry.request_id);
    }
    assert_eq!(ids.len(), 2);
    assert!(ledger.try_recv().is_err());
    for mock in mocks {
        mock.assert_hits_async(1).await;
    }
    handle.shutdown();
}

#[tokio::test]
async fn mixed_embedding_batches_restore_order_sum_usage_and_cache_one_complete_response() {
    let upstream = MockServer::start_async().await;
    let second_image = "data:image/jpeg;base64,/9j/";
    let image = upstream.mock_async(|when, then| {
        when.method(POST).path("/models/images/embeddings")
            .query_param("api-version", "2024-05-01-preview")
            .json_body(json!({"model":"embed-v-4-0", "input":[{"image":IMAGE}, {"image":second_image}], "encoding_format":"float"}));
        then.status(200).header("llm_provider-num_tokens", "11")
            .json_body(response(vec![embedding(1, json!([3.0])), embedding(0, json!([1.0]))], 1));
    }).await;
    let text = upstream.mock_async(|when, then| {
        when.method(POST).path("/models/embeddings")
            .query_param("api-version", "2024-05-01-preview")
            .json_body(json!({"model":"embed-v-4-0", "input":["first text", "second text"], "encoding_format":"float", "user":"test-user"}));
        then.status(200).header("llm_provider-num_tokens", "7")
            .json_body(response(vec![embedding(1, json!([4.0])), embedding(0, json!([2.0]))], 2));
    }).await;
    let mut config = config(&upstream);
    config.model_list[0].oxllm_params.input_cost_per_token = Some(0.001);
    config.model_list[0].oxllm_params.output_cost_per_token = Some(0.0);
    let (handle, state) = common::start_oxllm(config).await.unwrap();
    let mut ledger = capture_spend(&state);
    let payload = json!({"model":"embeddings", "input":[IMAGE, "first text", second_image, "second text"],
        "input_type":"image", "encoding_format":"float", "user":"test-user", "cache":{"use_cache":true}});
    let client = common::http_client();
    let mut ids = Vec::new();
    for cached in [false, true] {
        let result = client
            .post(handle.url("/v1/embeddings"))
            .header("x-oxllm-session-id", "image-session")
            .header("x-litellm-tags", "product:images")
            .json(&payload)
            .send()
            .await
            .unwrap();
        assert_eq!(result.status(), 200);
        ids.push(request_id(&result));
        assert_eq!(result.headers().contains_key("x-oxllm-cache-hit"), cached);
        let cost = result.headers()["x-oxllm-response-cost"]
            .to_str()
            .unwrap()
            .parse::<f64>()
            .unwrap();
        assert!(
            (cost - 0.018).abs() < 1e-12,
            "sum both batches once, got {cost}"
        );
        let body: Value = result.json().await.unwrap();
        assert_eq!(body["model"], "embeddings");
        assert_eq!(
            body["data"],
            json!([
                embedding(0, json!([1.0])),
                embedding(1, json!([2.0])),
                embedding(2, json!([3.0])),
                embedding(3, json!([4.0]))
            ])
        );
        assert_eq!(
            body["usage"],
            json!({"prompt_tokens":18, "total_tokens":18})
        );
    }
    assert_ne!(ids[0], ids[1]);
    let entry = tokio::time::timeout(Duration::from_secs(5), ledger.recv())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(entry.request_id.as_deref(), Some(ids[0].as_str()));
    assert_eq!(entry.session_id.as_deref(), Some("image-session"));
    assert_eq!(entry.end_user_id.as_deref(), Some("test-user"));
    assert_eq!(entry.request_tags.as_deref(), Some("[\"product:images\"]"));
    assert_eq!(entry.prompt_tokens, Some(18));
    assert_eq!(entry.total_tokens, Some(18));
    assert!((entry.cost.unwrap() - 0.018).abs() < 1e-12);
    assert!(
        ledger.try_recv().is_err(),
        "one ledger event, not one per batch or cache hit"
    );
    let model_spend = state
        .load_full()
        .cache
        .counter_value(&oxllm_server::hooks::spend_keys::model_spend_key(
            "embeddings",
        ))
        .await
        .unwrap();
    assert!((model_spend - 0.018).abs() < 1e-12);
    image.assert_hits_async(1).await;
    text.assert_hits_async(1).await;
    handle.shutdown();
}

#[tokio::test]
async fn bare_image_input_keeps_explicit_url_versions_and_supports_base64_vectors() {
    let upstream = MockServer::start_async().await;
    let image = upstream.mock_async(|when, then| {
        when.method(POST).path("/models/images/embeddings")
            .query_param("api-version", "explicit-version").query_param("trace", "kept")
            .json_body(json!({"model":"embed-v-4-0", "input":[{"image":IMAGE}], "encoding_format":"base64"}));
        then.status(200).json_body(response(vec![embedding(0, json!("AACAPwAAAEA="))], 4));
    }).await;
    let mut config = config(&upstream);
    config.model_list[0].oxllm_params.api_base =
        Some(upstream.url("/models/images/embeddings/?trace=kept&api-version=explicit-version"));
    config.model_list[0].oxllm_params.api_version = Some("ignored-config-version".into());
    let (handle, _) = common::start_oxllm(config).await.unwrap();
    let result = common::http_client().post(handle.url("/v1/embeddings"))
        .json(&json!({"model":"embeddings", "input":IMAGE, "input_type":"image", "encoding_format":"base64"}))
        .send().await.unwrap();
    assert_eq!(result.status(), 200);
    let body: Value = result.json().await.unwrap();
    assert_eq!(body["data"][0]["embedding"], "AACAPwAAAEA=");
    assert_eq!(body["usage"]["total_tokens"], 4);
    image.assert_hits_async(1).await;
    handle.shutdown();
}

#[tokio::test]
async fn text_only_batches_and_scalar_text_preserve_the_text_endpoint_contract() {
    let upstream = MockServer::start_async().await;
    let inputs = json!(["Dog", "aGVsbG8=", "", "data:image/png;base64,invalid!"]);
    let text = upstream.mock_async(|when, then| {
        when.method(POST).path("/models/embeddings")
            .json_body(json!({"model":"embed-v-4-0", "input":inputs, "encoding_format":"float", "dimensions":512, "user":"test-user"}));
        then.status(200).json_body(response((0..4).map(|index| embedding(index, json!([index]))).collect(), 5));
    }).await;
    let scalar = upstream
        .mock_async(|when, then| {
            when.method(POST)
                .path("/models/embeddings")
                .json_body(json!({"model":"embed-v-4-0", "input":["scalar text"]}));
            then.status(200)
                .json_body(response(vec![embedding(0, json!([1]))], 2));
        })
        .await;
    let image = upstream
        .mock_async(|when, then| {
            when.method(POST).path("/models/images/embeddings");
            then.status(400);
        })
        .await;
    let (handle, _) = common::start_oxllm(config(&upstream)).await.unwrap();
    for payload in [
        json!({"model":"embeddings", "input":inputs, "encoding_format":"float", "dimensions":512, "user":"test-user", "input_type":"image"}),
        json!({"model":"embeddings", "input":"scalar text"}),
    ] {
        let result = common::http_client()
            .post(handle.url("/v1/embeddings"))
            .json(&payload)
            .send()
            .await
            .unwrap();
        assert_eq!(result.status(), 200);
    }
    text.assert_hits_async(1).await;
    scalar.assert_hits_async(1).await;
    image.assert_hits_async(0).await;
    handle.shutdown();
}

#[tokio::test]
async fn non_cohere_foundry_models_do_not_enter_the_image_embedding_path() {
    let upstream = MockServer::start_async().await;
    let text = upstream
        .mock_async(|when, then| {
            when.method(POST).path("/models/embeddings").json_body(
                json!({"model":"text-embedding-3-large", "input":[IMAGE], "dimensions":512}),
            );
            then.status(200)
                .json_body(response(vec![embedding(0, json!([1]))], 2));
        })
        .await;
    let mut config = config(&upstream);
    config.model_list[0].oxllm_params.model = "azure_ai/text-embedding-3-large".into();
    let (handle, _) = common::start_oxllm(config).await.unwrap();
    let result = common::http_client()
        .post(handle.url("/v1/embeddings"))
        .json(
            &json!({"model":"embeddings", "input":[IMAGE], "input_type":"image", "dimensions":512}),
        )
        .send()
        .await
        .unwrap();
    assert_eq!(result.status(), 200);
    text.assert_hits_async(1).await;
    handle.shutdown();
}

#[tokio::test]
async fn a_failed_image_batch_is_an_error_not_a_partial_or_cached_embedding_list() {
    let upstream = MockServer::start_async().await;
    let text = upstream
        .mock_async(|when, then| {
            when.method(POST)
                .path("/models/embeddings")
                .json_body(json!({"model":"embed-v-4-0", "input":["text"]}));
            then.status(200)
                .json_body(response(vec![embedding(0, json!([1]))], 2));
        })
        .await;
    let image = upstream
        .mock_async(|when, then| {
            when.method(POST).path("/models/images/embeddings");
            then.status(400)
                .json_body(json!({"error":{"message":"unsupported image format"}}));
        })
        .await;
    let (handle, state) = common::start_oxllm(config(&upstream)).await.unwrap();
    let _ledger = capture_spend(&state);
    for _ in 0..2 {
        let result = common::http_client()
            .post(handle.url("/v1/embeddings"))
            .json(
                &json!({"model":"embeddings", "input":[IMAGE, "text"], "cache":{"use_cache":true}}),
            )
            .send()
            .await
            .unwrap();
        assert_eq!(result.status(), 400);
        request_id(&result);
        assert!(!result.headers().contains_key("x-oxllm-cache-hit"));
        let body: Value = result.json().await.unwrap();
        assert_eq!(body["error"]["message"], "unsupported image format");
        assert!(body.get("data").is_none());
    }
    text.assert_hits_async(2).await;
    image.assert_hits_async(2).await;
    handle.shutdown();
}

#[tokio::test]
async fn image_rate_limits_keep_the_provider_status_backoff_and_request_correlation() {
    let upstream = MockServer::start_async().await;
    let image = upstream
        .mock_async(|when, then| {
            when.method(POST).path("/models/images/embeddings");
            then.status(429)
                .header("retry-after", "13")
                .header("x-ratelimit-limit-requests", "50")
                .json_body(json!({"error":{"message":"image capacity exhausted"}}));
        })
        .await;
    let (handle, _) = common::start_oxllm(config(&upstream)).await.unwrap();
    let result = common::http_client()
        .post(handle.url("/v1/embeddings"))
        .json(&json!({"model":"embeddings", "input":[IMAGE]}))
        .send()
        .await
        .unwrap();
    assert_eq!(result.status(), 429);
    request_id(&result);
    assert_eq!(result.headers()["retry-after"], "13");
    assert_eq!(result.headers()["x-ratelimit-limit-requests"], "50");
    let body: Value = result.json().await.unwrap();
    assert_eq!(body["error"]["message"], "image capacity exhausted");
    assert_eq!(body["error"]["code"], "upstream_rate_limited");
    image.assert_hits_async(1).await;
    handle.shutdown();
}

#[tokio::test]
async fn malformed_image_response_indices_fail_without_returning_misattributed_vectors() {
    let upstream = MockServer::start_async().await;
    let image = upstream
        .mock_async(|when, then| {
            when.method(POST).path("/models/images/embeddings");
            then.status(200).json_body(response(
                vec![json!({"object":"embedding", "embedding":[1.0]})],
                3,
            ));
        })
        .await;
    let (handle, _) = common::start_oxllm(config(&upstream)).await.unwrap();
    let result = common::http_client()
        .post(handle.url("/v1/embeddings"))
        .json(&json!({"model":"embeddings", "input":[IMAGE]}))
        .send()
        .await
        .unwrap();
    assert_eq!(result.status(), 502);
    request_id(&result);
    let body: Value = result.json().await.unwrap();
    assert_eq!(body["error"]["code"], "upstream_error");
    assert_eq!(
        body["error"]["message"],
        "upstream embedding response does not match its input batch"
    );
    assert!(body.get("data").is_none());
    image.assert_hits_async(1).await;
    handle.shutdown();
}

#[tokio::test]
async fn embedding_fallbacks_receive_the_original_input_not_a_previous_routes_subset() {
    let upstream = MockServer::start_async().await;
    let text = upstream
        .mock_async(|when, then| {
            when.method(POST)
                .path("/models/embeddings")
                .json_body(json!({"model":"embed-v-4-0", "input":["text"]}));
            then.status(200)
                .json_body(response(vec![embedding(0, json!([1]))], 2));
        })
        .await;
    let image = upstream
        .mock_async(|when, then| {
            when.method(POST).path("/models/images/embeddings");
            then.status(503)
                .json_body(json!({"error":{"message":"temporarily unavailable"}}));
        })
        .await;
    let fallback = upstream
        .mock_async(|when, then| {
            when.method(POST)
                .path("/backup/embeddings")
                .json_body(json!({"model":"multimodal-embed", "input":[IMAGE, "text"]}));
            then.status(200).json_body(response(
                vec![embedding(0, json!([10])), embedding(1, json!([20]))],
                8,
            ));
        })
        .await;
    let mut config = config(&upstream);
    let mut backup = config.model_list[0].clone();
    backup.model_name = "backup".into();
    backup.oxllm_params.model = "openai/multimodal-embed".into();
    backup.oxllm_params.api_base = Some(upstream.url("/backup"));
    backup.oxllm_params.api_version = None;
    config.model_list.push(backup);
    config.router_settings.routing.fallbacks = Some(json!([{"embeddings":["backup"]}]));
    let (handle, _) = common::start_oxllm(config).await.unwrap();
    let result = common::http_client()
        .post(handle.url("/v1/embeddings"))
        .json(&json!({"model":"embeddings", "input":[IMAGE, "text"]}))
        .send()
        .await
        .unwrap();
    assert_eq!(result.status(), 200);
    assert_eq!(result.headers()["x-oxllm-attempted-fallbacks"], "1");
    let body: Value = result.json().await.unwrap();
    assert_eq!(body["model"], "backup");
    assert_eq!(
        body["data"],
        json!([embedding(0, json!([10])), embedding(1, json!([20]))])
    );
    text.assert_hits_async(1).await;
    image.assert_hits_async(1).await;
    fallback.assert_hits_async(1).await;
    handle.shutdown();
}
