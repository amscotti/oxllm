//! Provider request contracts and billing across the public API surfaces.

mod common;

use std::sync::Arc;
use std::time::Duration;

use httpmock::MockServer;
use oxllm_core::{UsageSummary, calculate_cost_for_service_tier};
use oxllm_server::hooks::{BudgetHook, CacheHook, CostTrackingHook};
use oxllm_server::spend_writer::SpendEntry;
use serde_json::{Value, json};
use tokio::sync::mpsc;

fn config(base: &str, model: &str) -> oxllm_server::ProxyConfig {
    let mut config = common::proxy_config(base);
    config.model_list[0].model_name = "model".into();
    config.model_list[0].oxllm_params.model = model.into();
    config
}

fn capture_spend(state: &oxllm_server::SharedState) -> mpsc::Receiver<SpendEntry> {
    let mut next = (*state.load_full()).clone();
    let settings = &mut Arc::make_mut(&mut next.config).oxllm_settings;
    settings.max_budget = Some(100.0);
    settings.budget_duration = Some("1h".into());
    settings.model_max_budget = Some(std::collections::HashMap::from([("model".into(), 100.0)]));
    let (tx, rx) = mpsc::channel(8);
    next.spend_tx = Some(tx.clone());
    let mut budget = BudgetHook::new(next.cache.clone(), Some(100.0), "1h");
    budget.model_max_budget = Some(std::collections::HashMap::from([("model".into(), 100.0)]));
    next.hooks = vec![
        Arc::new(CacheHook::new(next.cache.clone(), Duration::from_secs(60))),
        Arc::new(budget),
        Arc::new(CostTrackingHook::with_spend_writer(Some(tx))),
    ];
    state.store(Arc::new(next));
    rx
}

async fn spend(rx: &mut mpsc::Receiver<SpendEntry>) -> SpendEntry {
    tokio::time::timeout(Duration::from_secs(5), rx.recv())
        .await
        .unwrap()
        .unwrap()
}

fn usage() -> UsageSummary {
    UsageSummary {
        prompt_tokens: 1000,
        completion_tokens: 100,
        total_tokens: 1100,
        cache_read_input_tokens: Some(300),
        reasoning_tokens: Some(20),
        ..Default::default()
    }
}

fn wire_usage() -> Value {
    json!({
        "prompt_tokens":1000,"completion_tokens":100,"total_tokens":1100,
        "prompt_tokens_details":{"cached_tokens":300},
        "completion_tokens_details":{"reasoning_tokens":20}
    })
}

#[tokio::test]
async fn accelerated_deepseek_backend_rewrites_limits_and_meters_cached_tokens() {
    let mock = MockServer::start();
    let model = "deepseek-ai/DeepSeek-V4.1-Flash-Fast";
    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST).path("/chat/completions")
            .header("authorization", "Bearer sk-test-key")
            .json_body_partial(json!({"model":model,"max_tokens":100}).to_string());
        then.status(200).json_body(json!({
            "id":"chatcmpl-fast","model":model,
            "choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],
            "usage":wire_usage()
        }));
    });
    let (handle, state) =
        common::start_oxllm(config(&mock.base_url(), &format!("baseten/{model}")))
            .await
            .unwrap();
    let mut ledger = capture_spend(&state);
    let response = common::http_client().post(handle.url("/v1/chat/completions"))
        .json(&json!({"model":"model","max_completion_tokens":100,"messages":[{"role":"user","content":"hi"}]}))
        .send().await.unwrap();
    assert_eq!(response.status(), 200);
    let cost: f64 = response.headers()["x-oxllm-response-cost"]
        .to_str()
        .unwrap()
        .parse()
        .unwrap();
    let expected = calculate_cost_for_service_tier(model, &usage(), None, "baseten", None).unwrap();
    assert!((cost - expected).abs() < 1e-9);
    let entry = spend(&mut ledger).await;
    assert_eq!(entry.provider, "baseten");
    assert_eq!(entry.cost, Some(expected));
    upstream.assert_hits(1);
    handle.shutdown();
}

#[tokio::test]
async fn groq_browser_actions_bill_search_and_opens_end_to_end() {
    // Executed compound-tool actions become billable hosted-tool usage: two
    // `browser.search` actions at the model's per-query rate plus three
    // `browser.open` actions at Groq's flat $1/1k provider rate. The delta
    // against an identical response without actions isolates the fees from
    // token pricing.
    let mock = MockServer::start();
    let model = "openai/gpt-oss-120b";
    let response_body = |actions: Value| {
        json!({
            "id":"chatcmpl-groq-browser","model":model,
            "choices":[{"index":0,"message":{"role":"assistant","content":"searched",
                "executed_tools":actions},"finish_reason":"stop"}],
            "usage":{"prompt_tokens":1000,"completion_tokens":100,"total_tokens":1100}
        })
    };
    let with_actions = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .json_body_partial(
                json!({"model":model,"messages":[{"role":"user","content":"search the web"}]})
                    .to_string(),
            );
        then.status(200).json_body(response_body(json!([
            {"name":"browser.search"},{"name":"browser.search"},
            {"name":"browser.open"},{"name":"browser.open"},{"name":"browser.open"},
            {"name":"code_interpreter"}
        ])));
    });
    let without_actions = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions")
            .json_body_partial(
                json!({"model":model,"messages":[{"role":"user","content":"no tools"}]})
                    .to_string(),
            );
        then.status(200).json_body(response_body(json!([])));
    });
    let (handle, state) = common::start_oxllm(config(&mock.base_url(), &format!("groq/{model}")))
        .await
        .unwrap();
    let mut ledger = capture_spend(&state);
    let client = common::http_client();
    let mut costs = Vec::new();
    for (prompt, upstream) in [
        ("search the web", &with_actions),
        ("no tools", &without_actions),
    ] {
        let response = client
            .post(handle.url("/v1/chat/completions"))
            .json(&json!({"model":"model","messages":[{"role":"user","content":prompt}]}))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let cost = response
            .headers()
            .get("x-oxllm-response-cost")
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.parse::<f64>().ok())
            .expect("cost header");
        let body: Value = response.json().await.unwrap();
        if prompt == "search the web" {
            assert_eq!(body["usage"]["server_tool_use"]["web_search_requests"], 2);
            assert_eq!(body["usage"]["server_tool_use"]["browser_open_requests"], 3);
        }
        costs.push(cost);
        upstream.assert_hits(1);
    }
    let fee_delta = costs[0] - costs[1];
    assert!(
        (fee_delta - (2.0 * 5e-3 + 3.0 * 1e-3)).abs() < 1e-9,
        "search+open fees missing: delta {fee_delta}"
    );
    let entry = spend(&mut ledger).await;
    assert_eq!(entry.provider, "groq");
    assert!((entry.cost.unwrap() - costs[0]).abs() < 1e-12);
    handle.shutdown();
}

#[tokio::test]
async fn canonical_duration_rate_is_billed_once_and_persisted_without_tokens() {
    let mock = MockServer::start();
    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST).path("/chat/completions");
        then.status(200).delay(Duration::from_millis(50)).json_body(json!({
            "id":"chatcmpl-duration","model":"private-duration-model",
            "choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}]
        }));
    });
    let mut config = config(&mock.base_url(), "openai/private-duration-model");
    let params = &mut config.model_list[0].oxllm_params;
    params.cost_per_second = Some(0.003);
    params.input_cost_per_second = Some(0.1);
    params.output_cost_per_second = Some(0.1);
    let (handle, state) = common::start_oxllm(config).await.unwrap();
    let mut ledger = capture_spend(&state);
    let response = common::http_client()
        .post(handle.url("/v1/chat/completions"))
        .json(&json!({"model":"model","messages":[{"role":"user","content":"hi"}]}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let cost: f64 = response.headers()["x-oxllm-response-cost"]
        .to_str()
        .unwrap()
        .parse()
        .unwrap();
    let entry = spend(&mut ledger).await;
    assert!(cost > 0.0);
    assert!((cost - entry.cost.unwrap()).abs() < 1e-9);
    let observed_duration_cost = entry.latency_ms.unwrap() as f64 * 0.003 / 1000.0;
    // The ledger observes the whole request; duration billing measures only
    // the upstream attempt, excluding admission and reservation work.
    assert!(cost >= 0.003 * 0.05);
    assert!(cost <= observed_duration_cost + 1e-9);
    assert!(entry.total_tokens.is_none());
    upstream.assert_hits(1);
    handle.shutdown();
}

#[tokio::test]
async fn served_tier_controls_chat_and_responses_cost_headers_budgets_and_spend() {
    let mock = MockServer::start();
    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST).path("/chat/completions")
            .json_body_partial(json!({"model":"gpt-6.1-sol","service_tier":"priority"}).to_string());
        then.status(200).json_body(json!({
            "id":"chatcmpl-sol","object":"chat.completion","created":1,
            "model":"gpt-6.1-sol","service_tier":"flex",
            "choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],
            "usage":wire_usage()
        }));
    });
    let expected =
        calculate_cost_for_service_tier("gpt-6.1-sol", &usage(), None, "openai", Some("flex"))
            .unwrap();
    let standard =
        calculate_cost_for_service_tier("gpt-6.1-sol", &usage(), None, "openai", None).unwrap();
    assert_ne!(expected, standard);
    for (path, request) in [
        (
            "/v1/chat/completions",
            json!({"model":"model","service_tier":"priority","messages":[{"role":"user","content":"hi"}]}),
        ),
        (
            "/v1/responses",
            json!({"model":"model","service_tier":"priority","input":"hi"}),
        ),
    ] {
        let (handle, state) = common::start_oxllm(config(&mock.base_url(), "openai/gpt-6.1-sol"))
            .await
            .unwrap();
        let mut ledger = capture_spend(&state);
        let response = common::http_client()
            .post(handle.url(path))
            .json(&request)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let cost: f64 = response.headers()["x-oxllm-response-cost"]
            .to_str()
            .unwrap()
            .parse()
            .unwrap();
        let body: Value = response.json().await.unwrap();
        assert_eq!(body["service_tier"], "flex");
        assert!((cost - expected).abs() < 1e-9);
        let entry = spend(&mut ledger).await;
        assert!((entry.cost.unwrap() - expected).abs() < 1e-9);
        let debit = state
            .load_full()
            .cache
            .counter_value("spend:model:model")
            .await
            .unwrap();
        assert!((debit - expected).abs() < 1e-9);
        handle.shutdown();
    }
    upstream.assert_hits(2);
}

#[tokio::test]
async fn streamed_served_tier_survives_usage_only_frames_and_response_conversion() {
    for served in ["flex", "default"] {
        for path in ["/v1/chat/completions", "/v1/responses"] {
            let mock = MockServer::start();
            let first = json!({
                "id":"chatcmpl-sol","model":"gpt-6.1-sol","service_tier":"priority",
                "choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]
            });
            let content = json!({
                "id":"chatcmpl-sol","model":"gpt-6.1-sol","service_tier":served,
                "choices":[{"index":0,"delta":{"content":"hello"},"finish_reason":"stop"}]
            });
            let final_usage = json!({"id":"chatcmpl-sol","model":"gpt-6.1-sol","choices":[],"usage":wire_usage()});
            let sse = format!(
                "data: {first}\n\ndata: {content}\n\ndata: {final_usage}\n\ndata: [DONE]\n\n"
            );
            let upstream = mock.mock(|when, then| {
                when.method(httpmock::Method::POST)
                    .path("/chat/completions")
                    .json_body_partial(
                        json!({"model":"gpt-6.1-sol","stream":true,"service_tier":"priority"})
                            .to_string(),
                    );
                then.status(200)
                    .header("content-type", "text/event-stream")
                    .body(sse);
            });
            let (handle, state) =
                common::start_oxllm(config(&mock.base_url(), "openai/gpt-6.1-sol"))
                    .await
                    .unwrap();
            let mut ledger = capture_spend(&state);
            let request = if path.ends_with("responses") {
                json!({"model":"model","stream":true,"service_tier":"priority","input":"hi"})
            } else {
                json!({"model":"model","stream":true,"service_tier":"priority","stream_options":{"include_usage":true},"messages":[{"role":"user","content":"hi"}]})
            };
            let response = common::http_client()
                .post(handle.url(path))
                .json(&request)
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), 200);
            let text = response.text().await.unwrap();
            assert!(text.contains("hello"));
            if path.ends_with("responses") {
                let completed = text
                    .lines()
                    .filter_map(|line| line.strip_prefix("data: "))
                    .filter_map(|data| serde_json::from_str::<Value>(data).ok())
                    .find(|event| event["type"] == "response.completed")
                    .unwrap();
                assert_eq!(completed["response"]["service_tier"], served);
                assert_eq!(completed["response"]["usage"]["total_tokens"], 1100);
            }
            let billable = (served != "default").then_some(served);
            let expected =
                calculate_cost_for_service_tier("gpt-6.1-sol", &usage(), None, "openai", billable)
                    .unwrap();
            let entry = spend(&mut ledger).await;
            assert!(
                (entry.cost.unwrap() - expected).abs() < 1e-9,
                "{path} served {served}"
            );
            let debit = state
                .load_full()
                .cache
                .counter_value("spend:model:model")
                .await
                .unwrap();
            assert!((debit - expected).abs() < 1e-9);
            upstream.assert_hits(1);
            handle.shutdown();
        }
    }
}

#[tokio::test]
async fn per_message_output_config_is_forwarded_with_beta_on_native_and_translated_routes() {
    let mock = MockServer::start();
    let messages = json!([
        {"role":"user","content":"first","output_config":{"effort":"low"}},
        {"role":"assistant","content":"answer","output_config":{"effort":"high"}},
        {"role":"user","content":"second","output_config":{"effort":"medium"}}
    ]);
    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/messages")
            .header(
                "anthropic-beta",
                oxllm_providers::anthropic::BETA_MID_CONVERSATION_OUTPUT_CONFIG,
            )
            .json_body_partial(
                json!({"model":"claude-sonnet-5-5","messages":messages}).to_string(),
            );
        then.status(200).json_body(json!({
            "id":"msg-test","type":"message","role":"assistant","model":"claude-sonnet-5-5",
            "content":[{"type":"text","text":"hello"}],"stop_reason":"end_turn",
            "usage":{"input_tokens":10,"output_tokens":5}
        }));
    });
    let (handle, _) = common::start_oxllm(config(&mock.base_url(), "anthropic/claude-sonnet-5-5"))
        .await
        .unwrap();
    for path in [
        "/v1/chat/completions",
        "/v1/messages",
        "/anthropic/v1/messages",
    ] {
        let response = common::http_client()
            .post(handle.url(path))
            .json(&json!({"model":"model","messages":messages,"max_tokens":100}))
            .send()
            .await
            .unwrap();
        let status = response.status();
        let body: Value = response.json().await.unwrap();
        assert_eq!(status, 200, "{path}: {body}");
    }
    upstream.assert_hits(3);
    handle.shutdown();
}

#[tokio::test]
async fn unpriced_response_alias_and_cache_hit_preserve_cost_without_double_debit() {
    let mock = MockServer::start();
    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST).path("/chat/completions");
        then.status(200).json_body(json!({
            "id":"chat-alias","model":"unpriced-public-alias",
            "choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],
            "usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}
        }));
    });
    let mut cfg = config(&mock.base_url(), "openai/gpt-4o");
    cfg.model_list[0].oxllm_params.input_cost_per_token = Some(0.01);
    cfg.model_list[0].oxllm_params.output_cost_per_token = Some(0.02);
    let (handle, state) = common::start_oxllm(cfg).await.unwrap();
    let mut ledger = capture_spend(&state);
    let request = json!({"model":"model","max_tokens":10,"messages":[{"role":"user","content":"hi"}],"cache":{"insert":true}});
    for cached in [false, true] {
        let response = common::http_client()
            .post(handle.url("/v1/chat/completions"))
            .json(&request)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let cost: f64 = response.headers()["x-oxllm-response-cost"]
            .to_str()
            .unwrap()
            .parse()
            .unwrap();
        assert!((cost - 0.2).abs() < 1e-12);
        assert_eq!(response.headers().contains_key("x-oxllm-cache-hit"), cached);
        let body: Value = response.json().await.unwrap();
        assert_eq!(body["model"], "model");
        assert!(body.get("_oxllm_cache_version").is_none());
    }
    let entry = spend(&mut ledger).await;
    assert!((entry.cost.unwrap() - 0.2).abs() < 1e-12);
    assert_eq!(entry.model, "unpriced-public-alias");
    assert!(ledger.try_recv().is_err());
    let total = state
        .load_full()
        .cache
        .counter_value("spend:model:model")
        .await
        .unwrap();
    assert!((total - 0.2).abs() < 1e-12);
    upstream.assert_hits(1);
    handle.shutdown();
}

#[tokio::test]
async fn null_stream_usage_does_not_make_reasoning_and_tool_output_free() {
    let mock = MockServer::start();
    let content = json!({"model":"unpriced-public-alias","choices":[{"index":0,"delta":{
        "reasoning_content":"Consider the problem carefully and calculate the answer.",
        "tool_calls":[{"index":0,"id":"call-1","type":"function","function":{"name":"calculate","arguments":"{\"operation\":\"sum\",\"values\":[1,2,3]}"}}]
    },"finish_reason":null}],"usage":null});
    let placeholder = json!({"choices":[],"usage":null});
    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions");
        then.status(200)
            .header("content-type", "text/event-stream")
            .body(format!(
                "data: {content}\n\ndata: {placeholder}\n\ndata: [DONE]\n\n"
            ));
    });
    let (handle, state) = common::start_oxllm(config(&mock.base_url(), "openai/gpt-4o"))
        .await
        .unwrap();
    let mut ledger = capture_spend(&state);
    let response = common::http_client().post(handle.url("/v1/chat/completions"))
        .json(&json!({"model":"model","stream":true,"messages":[{"role":"user","content":"calculate"}]}))
        .send().await.unwrap();
    assert_eq!(response.status(), 200);
    assert!(response.text().await.unwrap().contains("calculate"));
    let entry = spend(&mut ledger).await;
    assert!(entry.completion_tokens.unwrap() > 0);
    assert!(entry.cost.unwrap() > 0.0);
    let total = state
        .load_full()
        .cache
        .counter_value("spend:model:model")
        .await
        .unwrap();
    assert!((total - entry.cost.unwrap()).abs() < 1e-9);
    upstream.assert_hits(1);
    handle.shutdown();
}

#[tokio::test]
async fn null_usage_only_frame_cannot_suppress_real_final_usage() {
    let mock = MockServer::start();
    let placeholder = json!({"choices":[],"usage":null});
    let actual = json!({"model":"gpt-4o","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}});
    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions");
        then.status(200)
            .header("content-type", "text/event-stream")
            .body(format!(
                "data: {placeholder}\n\ndata: {actual}\n\ndata: [DONE]\n\n"
            ));
    });
    let (handle, state) = common::start_oxllm(config(&mock.base_url(), "openai/gpt-4o"))
        .await
        .unwrap();
    let mut ledger = capture_spend(&state);
    let response = common::http_client().post(handle.url("/v1/chat/completions"))
        .json(&json!({"model":"model","stream":true,"stream_options":{"include_usage":true},"messages":[{"role":"user","content":"hi"}]}))
        .send().await.unwrap();
    assert_eq!(response.status(), 200);
    let text = response.text().await.unwrap();
    let usage = text
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter_map(|data| serde_json::from_str::<Value>(data).ok())
        .filter(|event| event["usage"].is_object())
        .collect::<Vec<_>>();
    assert_eq!(usage.len(), 1);
    assert_eq!(usage[0]["usage"]["total_tokens"], 15);
    assert!(spend(&mut ledger).await.cost.unwrap() > 0.0);
    upstream.assert_hits(1);
    handle.shutdown();
}

#[tokio::test]
async fn native_image_usage_applies_overrides_and_debits_router_budget() {
    let mock = MockServer::start();
    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST).path("/images/generations")
            .json_body_partial(json!({"model":"gpt-image-1"}).to_string());
        then.status(200).header("content-type", "application/json")
            .json_body(json!({"data":[{"b64_json":"image"}],"usage":{"input_tokens":10,"output_tokens":5}}));
    });
    let mut cfg = config(&mock.base_url(), "openai/gpt-image-1");
    let params = &mut cfg.model_list[0].oxllm_params;
    params.input_cost_per_token = Some(0.001);
    params.output_cost_per_token = Some(0.002);
    params.max_budget = Some(100.0);
    let (handle, state) = common::start_oxllm(cfg).await.unwrap();
    let mut ledger = capture_spend(&state);
    let response = common::http_client()
        .post(handle.url("/v1/images/generations"))
        .json(&json!({"model":"model","prompt":"A cat"}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let cost: f64 = response.headers()["x-oxllm-response-cost"]
        .to_str()
        .unwrap()
        .parse()
        .unwrap();
    assert!((cost - 0.02).abs() < 1e-12);
    assert_eq!(
        response.json::<Value>().await.unwrap()["data"][0]["b64_json"],
        "image"
    );
    assert_eq!(spend(&mut ledger).await.cost, Some(cost));
    let spent = state
        .load_full()
        .cache
        .counter_value("spend:dep:model:openai/gpt-image-1")
        .await
        .expect("deployment spend");
    assert!((spent - cost).abs() < 1e-9);
    upstream.assert_hits(1);
    handle.shutdown();
}

#[tokio::test]
async fn native_message_stream_is_byte_preserving_and_settles_cache_write_usage() {
    let mock = MockServer::start();
    let start = json!({"type":"message_start","message":{
        "id":"msg-native","type":"message","model":"claude-sonnet-4-6","role":"assistant","content":[],
        "usage":{"input_tokens":10,"output_tokens":0,"cache_creation_input_tokens":50,"cache_creation":{"ephemeral_1h_input_tokens":40}}
    }});
    let text = json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello"}});
    let usage = json!({"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":20,"output_tokens_details":{"thinking_tokens":12}}});
    let stop = json!({"type":"message_stop"});
    let wire = format!(
        "event: message_start\ndata: {start}\n\nevent: content_block_delta\ndata: {text}\n\nevent: message_delta\ndata: {usage}\n\nevent: message_stop\ndata: {stop}\n\n"
    );
    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST).path("/messages");
        then.status(200)
            .header("content-type", "text/event-stream")
            .body(&wire);
    });
    let mut cfg = config(&mock.base_url(), "anthropic/claude-sonnet-4-6");
    cfg.model_list[0].oxllm_params.max_budget = Some(100.0);
    let (handle, state) = common::start_oxllm(cfg).await.unwrap();
    let mut ledger = capture_spend(&state);
    let response = common::http_client().post(handle.url("/anthropic/v1/messages"))
        .json(&json!({"model":"model","stream":true,"max_tokens":100,"messages":[{"role":"user","content":"hi"}]}))
        .send().await.unwrap();
    assert_eq!(response.status(), 200);
    assert_eq!(response.text().await.unwrap(), wire);
    let entry = spend(&mut ledger).await;
    let expected = calculate_cost_for_service_tier(
        "claude-sonnet-4-6",
        &UsageSummary {
            prompt_tokens: 10,
            completion_tokens: 20,
            total_tokens: 30,
            cache_creation_input_tokens: Some(50),
            cache_creation_1h_input_tokens: Some(40),
            reasoning_tokens: Some(12),
            ..Default::default()
        },
        None,
        "anthropic",
        None,
    )
    .unwrap();
    assert!((entry.cost.unwrap() - expected).abs() < 1e-12);
    let total = state
        .load_full()
        .cache
        .counter_value("spend:model:model")
        .await
        .unwrap();
    assert!((total - expected).abs() < 1e-9);
    let spent = state
        .load_full()
        .cache
        .counter_value("spend:dep:model:anthropic/claude-sonnet-4-6")
        .await
        .expect("deployment spend");
    assert!((spent - entry.cost.unwrap()).abs() < 1e-9);
    upstream.assert_hits(1);
    handle.shutdown();
}

#[tokio::test]
async fn native_router_budget_reservation_blocks_concurrent_spend_and_settles_actual_cost() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let entered = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let hits = Arc::new(AtomicUsize::new(0));
    let route = axum::routing::post({
        let entered = entered.clone();
        let release = release.clone();
        let hits = hits.clone();
        move || {
            let entered = entered.clone();
            let release = release.clone();
            let hits = hits.clone();
            async move {
                hits.fetch_add(1, Ordering::SeqCst);
                entered.notify_one();
                release.notified().await;
                axum::Json(json!({"type":"message","model":"claude-sonnet-4-6",
                    "content":[{"type":"text","text":"hello"}],"stop_reason":"end_turn",
                    "usage":{"input_tokens":5,"output_tokens":5}}))
            }
        }
    });
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let upstream_task = tokio::spawn(async move {
        axum::serve(listener, axum::Router::new().route("/messages", route))
            .await
            .unwrap();
    });
    let mut cfg = config(&base, "anthropic/claude-sonnet-4-6");
    let params = &mut cfg.model_list[0].oxllm_params;
    params.input_cost_per_token = Some(0.001);
    params.output_cost_per_token = Some(0.002);
    params.max_budget = Some(0.05);
    let (handle, state) = common::start_oxllm(cfg).await.unwrap();
    let url = handle.url("/anthropic/v1/messages");
    let request =
        json!({"model":"model","max_tokens":10,"messages":[{"role":"user","content":"hello"}]});
    let first = tokio::spawn({
        let url = url.clone();
        let request = request.clone();
        async move {
            common::http_client()
                .post(url)
                .json(&request)
                .send()
                .await
                .unwrap()
        }
    });
    tokio::time::timeout(Duration::from_secs(5), entered.notified())
        .await
        .unwrap();
    let reserved = state
        .load_full()
        .cache
        .counter_value("spend:dep:model:anthropic/claude-sonnet-4-6")
        .await
        .unwrap();
    assert!(reserved > 0.025 && reserved < 0.05, "{reserved}");
    let second = common::http_client()
        .post(&url)
        .json(&request)
        .send()
        .await
        .unwrap();
    assert_eq!(second.status(), 429);
    assert_eq!(hits.load(Ordering::SeqCst), 1);
    release.notify_one();
    let first = first.await.unwrap();
    assert_eq!(first.status(), 200);
    let actual: f64 = first.headers()["x-oxllm-response-cost"]
        .to_str()
        .unwrap()
        .parse()
        .unwrap();
    assert!((actual - 0.015).abs() < 1e-12);
    assert_eq!(
        state
            .load_full()
            .cache
            .counter_value("spend:dep:model:anthropic/claude-sonnet-4-6")
            .await,
        Some(actual)
    );
    handle.shutdown();
    upstream_task.abort();
}

#[tokio::test]
async fn native_error_events_preserve_wire_bytes_and_settle_partial_operator_spend() {
    let mock = MockServer::start();
    let start = json!({"type":"message_start","message":{"id":"msg-error","model":"claude-sonnet-4-6",
        "usage":{"input_tokens":10,"output_tokens":0}}});
    let text = json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"partial response"}});
    let error = json!({"type":"error","error":{"type":"overloaded_error","message":"overloaded"}});
    let wire = format!("data: {start}\n\ndata: {text}\n\ndata: {error}\n\n");
    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST).path("/messages");
        then.status(200)
            .header("content-type", "text/event-stream")
            .body(&wire);
    });
    let mut cfg = config(&mock.base_url(), "anthropic/claude-sonnet-4-6");
    cfg.model_list[0].oxllm_params.max_budget = Some(100.0);
    let (handle, state) = common::start_oxllm(cfg).await.unwrap();
    let mut ledger = capture_spend(&state);
    let response = common::http_client().post(handle.url("/anthropic/v1/messages"))
        .json(&json!({"model":"model","stream":true,"max_tokens":10,"messages":[{"role":"user","content":"hello"}]}))
        .send().await.unwrap();
    assert_eq!(response.status(), 200);
    assert_eq!(response.text().await.unwrap(), wire);
    let entry = spend(&mut ledger).await;
    assert_eq!(entry.status.as_deref(), Some("partial"));
    assert!(entry.completion_tokens.unwrap() > 0);
    assert!(entry.cost.unwrap() > 0.0);
    assert_eq!(
        state
            .load_full()
            .cache
            .counter_value("spend:model:model")
            .await
            .unwrap_or(0.0),
        0.0
    );
    let spent = state
        .load_full()
        .cache
        .counter_value("spend:dep:model:anthropic/claude-sonnet-4-6")
        .await
        .expect("deployment spend");
    assert!((spent - entry.cost.unwrap()).abs() < 1e-9);
    upstream.assert_hits(1);
    handle.shutdown();
}

#[tokio::test]
async fn invalid_configured_headers_are_rejected_before_sending_upstream() {
    let mock = MockServer::start();
    let upstream = mock.mock(|when, then| {
        when.method(httpmock::Method::POST)
            .path("/chat/completions");
        then.status(200)
            .json_body(json!({"model":"gpt-4o","choices":[]}));
    });
    let mut cfg = config(&mock.base_url(), "openai/gpt-4o");
    cfg.model_list[0].oxllm_params.extra_headers = Some(std::collections::HashMap::from([(
        "x-backend-key".into(),
        "private-marker\r\nx-injected: true".into(),
    )]));
    cfg.router_settings.routing.num_retries = Some(0);
    let (handle, _) = common::start_oxllm(cfg).await.unwrap();
    let response = common::http_client()
        .post(handle.url("/v1/chat/completions"))
        .json(&json!({"model":"model","messages":[{"role":"user","content":"hello"}]}))
        .send()
        .await
        .unwrap();
    assert!(response.status().is_server_error());
    assert!(!response.text().await.unwrap().contains("private-marker"));
    upstream.assert_hits(0);
    handle.shutdown();
}
