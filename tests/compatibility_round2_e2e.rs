//! Wire regressions for the second compatibility report.

mod common;

use httpmock::{Method::POST, MockServer};
use oxllm_server::config::parse_config_str;
use serde_json::{Value, json};

fn chat_response(model: &str) -> Value {
    json!({"id": "chat-round2", "object": "chat.completion", "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}})
}

#[tokio::test]
async fn vertex_gemini_requires_the_google_publisher_on_the_actual_wire() {
    let upstream = MockServer::start_async().await;
    let config = parse_config_str(&format!(
        r#"
model_list:
  - model_name: external-gemini-3-flash
    oxllm_params:
      model: vertex_ai/gemini-3-flash-preview
      vertex_project: project
      vertex_location: global
      api_base: {}
      api_key: synthetic-access-token
general_settings:
  allow_private_api_base: true
router_settings:
  num_retries: 0
"#,
        upstream.base_url()
    ))
    .unwrap();
    let wire = upstream
        .mock_async(|when, then| {
            when.method(POST)
                .path(
                    "/v1beta1/projects/project/locations/global/endpoints/openapi/chat/completions",
                )
                .header("authorization", "Bearer synthetic-access-token")
                .json_body_partial(json!({"model": "google/gemini-3-flash-preview"}).to_string());
            then.status(200)
                .json_body(chat_response("google/gemini-3-flash-preview"));
        })
        .await;
    let (proxy, _) = common::start_oxllm(config).await.unwrap();
    let response = common::http_client().post(proxy.url("/v1/chat/completions"))
        .json(&json!({"model": "external-gemini-3-flash", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 32})).send().await.unwrap();
    assert_eq!(response.status(), 200, "{}", response.text().await.unwrap());
    assert!(!response.headers().contains_key("x-oxllm-fallback-used"));
    wire.assert_hits_async(1).await;
    proxy.shutdown();
}

#[tokio::test]
async fn foundry_claude_default_key_is_anthropic_not_azure_subscription_auth() {
    let upstream = MockServer::start_async().await;
    let config = parse_config_str(&format!(
        r#"
model_list:
  - model_name: external-claude
    oxllm_params:
      model: azure_ai/claude-haiku-4-5
      api_base: {}/anthropic/v1/messages
      api_key: synthetic-key
general_settings:
  allow_private_api_base: true
router_settings:
  num_retries: 0
"#,
        upstream.base_url()
    ))
    .unwrap();
    let wire = upstream.mock_async(|when, then| {
        when.method(POST).path("/anthropic/v1/messages")
            .header("x-api-key", "synthetic-key").header("anthropic-version", "2023-06-01")
            .matches(|request| request.headers.as_ref().is_some_and(|headers| headers.iter().all(|(name, _)| !name.eq_ignore_ascii_case("api-key") && !name.eq_ignore_ascii_case("authorization"))));
        then.status(200).json_body(json!({"id": "msg-round2", "type": "message", "role": "assistant", "model": "claude-haiku-4-5",
            "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn", "usage": {"input_tokens": 4, "output_tokens": 2}}));
    }).await;
    let (proxy, _) = common::start_oxllm(config).await.unwrap();
    let response = common::http_client().post(proxy.url("/v1/chat/completions"))
        .json(&json!({"model": "external-claude", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 32})).send().await.unwrap();
    assert_eq!(response.status(), 200, "{}", response.text().await.unwrap());
    assert!(!response.headers().contains_key("x-oxllm-fallback-used"));
    wire.assert_hits_async(1).await;
    proxy.shutdown();
}

#[tokio::test]
async fn azure_reasoning_requests_use_completion_limits_without_a_masking_fallback() {
    for version in ["2025-04-01-preview", "v1"] {
        let upstream = MockServer::start_async().await;
        let config = parse_config_str(&format!(
            r#"
model_list:
  - model_name: external-gpt-5.6-luna
    oxllm_params:
      model: azure/gpt-5.6-luna
      api_base: {}
      api_version: {version}
      api_key: synthetic-key
general_settings:
  allow_private_api_base: true
router_settings:
  num_retries: 0
"#,
            upstream.base_url()
        ))
        .unwrap();
        let wire = upstream
            .mock_async(|when, then| {
                when.method(POST)
                    .path(if version == "v1" {
                        "/openai/v1/chat/completions"
                    } else {
                        "/openai/deployments/gpt-5.6-luna/chat/completions"
                    })
                    .header("api-key", "synthetic-key")
                    .query_param("api-version", version)
                    .json_body_partial(json!({"max_completion_tokens": 64}).to_string())
                    .matches(|request| {
                        request
                            .body
                            .as_ref()
                            .and_then(|body| serde_json::from_slice::<Value>(body).ok())
                            .is_some_and(|body| body.get("max_tokens").is_none())
                    });
                then.status(200).json_body(chat_response("gpt-5.6-luna"));
            })
            .await;
        let (proxy, _) = common::start_oxllm(config).await.unwrap();
        let response = common::http_client().post(proxy.url("/v1/chat/completions"))
            .json(&json!({"model": "external-gpt-5.6-luna", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 64})).send().await.unwrap();
        assert_eq!(response.status(), 200, "{}", response.text().await.unwrap());
        assert!(!response.headers().contains_key("x-oxllm-fallback-used"));
        wire.assert_hits_async(1).await;
        proxy.shutdown();
    }
}

#[tokio::test]
async fn public_model_echo_and_deployment_headers_preserve_serving_cost_and_throttle_signals() {
    let upstream = MockServer::start_async().await;
    let config = parse_config_str(&format!(
        r#"
model_list:
  - model_name: external-gpt-5.5
    oxllm_params:
      model: openai/gpt-5.5
      api_base: {}
      api_key: synthetic-key
      input_cost_per_token: 0.001
      output_cost_per_token: 0.002
    model_info:
      id: region-a
general_settings:
  allow_private_api_base: true
router_settings:
  num_retries: 0
"#,
        upstream.base_url()
    ))
    .unwrap();
    let wire = upstream
        .mock_async(|when, then| {
            when.method(POST).path("/chat/completions");
            then.status(200)
                .header("x-ratelimit-remaining-requests", "7")
                .header("x-ratelimit-reset-tokens", "2s")
                .json_body(chat_response("gpt-5.5-2026-04-23"));
        })
        .await;
    let (proxy, _) = common::start_oxllm(config).await.unwrap();
    let payload = json!({"model": "external-gpt-5.5", "messages": [{"role": "user", "content": "hello"}], "cache": {"insert": true}});
    for cached in [false, true] {
        let response = common::http_client()
            .post(proxy.url("/v1/chat/completions"))
            .json(&payload)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let cost = response.headers()["x-oxllm-response-cost"]
            .to_str()
            .unwrap()
            .parse::<f64>()
            .unwrap();
        assert!((cost - 0.008).abs() < 1e-9);
        assert_eq!(
            response.headers()["x-oxllm-model-group"],
            "external-gpt-5.5"
        );
        assert_eq!(
            response.headers()["x-litellm-model-group"],
            "external-gpt-5.5"
        );
        assert_eq!(response.headers()["x-oxllm-attempted-retries"], "0");
        for header in [
            "x-oxllm-response-duration-ms",
            "x-oxllm-overhead-duration-ms",
        ] {
            assert!(
                response.headers()[header]
                    .to_str()
                    .unwrap()
                    .parse::<f64>()
                    .unwrap()
                    >= 0.0
            );
        }
        if !cached {
            assert_eq!(response.headers()["x-oxllm-deployment-id"], "region-a");
            assert_eq!(response.headers()["x-oxllm-model-id"], "region-a");
            assert_eq!(response.headers()["x-ratelimit-remaining-requests"], "7");
            assert_eq!(response.headers()["x-ratelimit-reset-tokens"], "2s");
        } else {
            assert_eq!(response.headers()["x-oxllm-cache-hit"], "true");
        }
        let body = response.json::<Value>().await.unwrap();
        assert_eq!(body["model"], "external-gpt-5.5");
        assert_eq!(
            body["usage"]["completion_tokens_details"]["reasoning_tokens"],
            0
        );
        assert_eq!(body["usage"]["total_tokens"], 6);
    }
    wire.assert_hits_async(1).await;
    proxy.shutdown();
}

#[tokio::test]
async fn sse_chunks_echo_the_requested_model_but_keep_reported_usage() {
    let upstream = MockServer::start_async().await;
    let mut config = common::proxy_config(&upstream.base_url());
    config.model_list[0].model_name = "external-chat".into();
    let wire = upstream.mock_async(|when, then| {
        when.method(POST).path("/chat/completions");
        then.status(200).header("content-type", "text/event-stream").body(concat!(
            "data: {\"id\":\"stream\",\"model\":\"gpt-4o-snapshot\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":null}]}\n\n",
            "data: {\"id\":\"stream\",\"model\":\"gpt-4o-snapshot\",\"choices\":[],\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":2,\"total_tokens\":6,\"completion_tokens_details\":{\"reasoning_tokens\":1}}}\n\n",
            "data: [DONE]\n\n"
        ));
    }).await;
    let (proxy, _) = common::start_oxllm(config).await.unwrap();
    let response = common::http_client().post(proxy.url("/v1/chat/completions"))
        .json(&json!({"model": "external-chat", "messages": [{"role": "user", "content": "hello"}], "stream": true, "stream_options": {"include_usage": true}})).send().await.unwrap();
    assert_eq!(response.status(), 200);
    let text = response.text().await.unwrap();
    let chunks: Vec<Value> = text
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter_map(|data| serde_json::from_str(data).ok())
        .collect();
    assert!(!chunks.is_empty());
    assert!(
        chunks.iter().all(|chunk| chunk["model"] == "external-chat"),
        "{text}"
    );
    assert!(
        chunks
            .iter()
            .any(|chunk| chunk["usage"]["total_tokens"] == 6
                && chunk["usage"]["completion_tokens_details"]["reasoning_tokens"] == 1)
    );
    wire.assert_hits_async(1).await;
    proxy.shutdown();
}

#[tokio::test]
async fn retry_and_fallback_headers_identify_the_final_serving_deployment() {
    let primary = MockServer::start_async().await;
    let healthy = MockServer::start_async().await;
    let config = parse_config_str(&format!(
        r#"
model_list:
  - model_name: external-chat
    oxllm_params:
      model: openai/gpt-4o
      api_base: {primary}
      api_key: synthetic-key
    model_info:
      id: failing-leg
  - model_name: rescue
    oxllm_params:
      model: openai/gpt-4o
      api_base: {healthy}
      api_key: synthetic-key
    model_info:
      id: serving-leg
general_settings:
  allow_private_api_base: true
router_settings:
  num_retries: 1
  allowed_fails: 10
  fallbacks: [{{external-chat: [rescue]}}]
"#,
        primary = primary.base_url(),
        healthy = healthy.base_url()
    ))
    .unwrap();
    let fail = primary
        .mock_async(|when, then| {
            when.method(POST);
            then.status(500)
                .json_body(json!({"error": {"message": "failure"}}));
        })
        .await;
    let success = healthy
        .mock_async(|when, then| {
            when.method(POST);
            then.status(200).json_body(chat_response("gpt-4o-snapshot"));
        })
        .await;
    let (proxy, _) = common::start_oxllm(config).await.unwrap();
    let response = common::http_client()
        .post(proxy.url("/v1/chat/completions"))
        .json(
            &json!({"model": "external-chat", "messages": [{"role": "user", "content": "hello"}]}),
        )
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    assert_eq!(response.headers()["x-oxllm-deployment-id"], "serving-leg");
    assert_eq!(response.headers()["x-oxllm-attempted-retries"], "1");
    assert_eq!(response.headers()["x-litellm-attempted-retries"], "1");
    assert_eq!(response.headers()["x-oxllm-attempted-fallbacks"], "1");
    assert_eq!(response.headers()["x-oxllm-fallback-used"], "rescue");
    assert_eq!(response.headers()["x-oxllm-model-group"], "rescue");
    assert_eq!(response.headers()["x-oxllm-model-name"], "openai/gpt-4o");
    assert_eq!(
        response.headers()["x-oxllm-requested-model"],
        "external-chat"
    );
    assert_eq!(response.json::<Value>().await.unwrap()["model"], "rescue");
    fail.assert_hits_async(2).await;
    success.assert_hits_async(1).await;
    proxy.shutdown();
}

#[tokio::test]
async fn embeddings_echo_alias_and_forward_only_throttle_headers() {
    let upstream = MockServer::start_async().await;
    let mut config = common::proxy_config(&upstream.base_url());
    config.model_list[0].model_name = "external-embed".into();
    config.model_list[0].oxllm_params.model = "openai/text-embedding-3-small".into();
    let wire = upstream
        .mock_async(|when, then| {
            when.method(POST).path("/embeddings");
            then.status(200)
                .header("x-ratelimit-remaining-tokens", "42")
                .header("x-vendor-internal", "private-value")
                .json_body(json!({
                    "model": "text-embedding-3-small", "object": "list",
                    "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
                    "usage": {"prompt_tokens": 4, "total_tokens": 4}
                }));
        })
        .await;
    let (proxy, shared) = common::start_oxllm(config).await.unwrap();
    let response = common::http_client()
        .post(proxy.url("/v1/embeddings"))
        .json(&json!({"model": "external-embed", "input": "hello"}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    assert_eq!(response.headers()["x-ratelimit-remaining-tokens"], "42");
    assert!(!response.headers().contains_key("x-vendor-internal"));
    assert_eq!(
        response.headers()["x-oxllm-deployment-id"]
            .to_str()
            .unwrap(),
        shared.load().deployments[0]
            .model_info
            .as_ref()
            .unwrap()
            .id
            .as_deref()
            .unwrap()
    );
    assert_eq!(
        response.json::<Value>().await.unwrap()["model"],
        "external-embed"
    );
    wire.assert_hits_async(1).await;
    proxy.shutdown();
}

#[tokio::test]
async fn default_fallback_counts_only_the_resolvable_leg_and_echoes_original_alias() {
    let upstream = MockServer::start_async().await;
    let mut config = common::proxy_config(&upstream.base_url());
    config.router_settings.routing.default_fallbacks = Some(json!(["missing-leg", "gpt-4o"]));
    let wire = upstream
        .mock_async(|when, then| {
            when.method(POST).path("/chat/completions");
            then.status(200).json_body(chat_response("gpt-4o"));
        })
        .await;
    let (proxy, _) = common::start_oxllm(config).await.unwrap();
    let response = common::http_client()
        .post(proxy.url("/v1/chat/completions"))
        .json(
            &json!({"model": "unknown-alias", "messages": [{"role": "user", "content": "hello"}]}),
        )
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    assert_eq!(response.headers()["x-oxllm-attempted-fallbacks"], "1");
    assert_eq!(
        response.headers()["x-oxllm-router-decision"],
        "default_fallback"
    );
    assert_eq!(response.json::<Value>().await.unwrap()["model"], "gpt-4o");
    wire.assert_hits_async(1).await;
    proxy.shutdown();
}

#[tokio::test]
async fn session_affinity_pre_call_alias_drives_sticky_wire_routing() {
    let first = MockServer::start_async().await;
    let second = MockServer::start_async().await;
    let mut config = common::proxy_config(&first.base_url());
    let mut sibling = config.model_list[0].clone();
    sibling.oxllm_params.api_base = Some(second.base_url());
    config.model_list.push(sibling);
    config.router_settings.optional_pre_call_checks = Some(vec!["session_affinity".into()]);
    let first_wire = first
        .mock_async(|when, then| {
            when.method(POST);
            then.status(200).json_body(chat_response("gpt-4o"));
        })
        .await;
    let second_wire = second
        .mock_async(|when, then| {
            when.method(POST);
            then.status(200).json_body(chat_response("gpt-4o"));
        })
        .await;
    let (proxy, _) = common::start_oxllm(config).await.unwrap();
    let mut ids = std::collections::HashSet::new();
    for _ in 0..4 {
        let response = common::http_client()
            .post(proxy.url("/v1/chat/completions"))
            .header("x-oxllm-session-id", "stable-session")
            .json(&json!({"model": "gpt-4o", "messages": [{"role": "user", "content": "hello"}]}))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        assert_eq!(
            response.headers()["x-oxllm-router-strategy"],
            "session_affinity"
        );
        ids.insert(
            response.headers()["x-oxllm-deployment-id"]
                .to_str()
                .unwrap()
                .to_string(),
        );
        let _ = response.bytes().await.unwrap();
    }
    assert_eq!(ids.len(), 1);
    assert_eq!(
        first_wire.hits_async().await + second_wire.hits_async().await,
        4
    );
    proxy.shutdown();
}

#[tokio::test]
async fn named_retry_policies_distinguish_same_leg_retries_from_cross_group_failover() {
    for (status, expected_attempts) in [(429, 2), (400, 1), (401, 1)] {
        let primary = MockServer::start_async().await;
        let rescue = MockServer::start_async().await;
        let config = parse_config_str(&format!(r#"
model_list:
  - model_name: primary
    oxllm_params: {{model: openai/gpt-4o, api_key: synthetic-key, api_base: {primary}}}
  - model_name: rescue
    oxllm_params: {{model: openai/gpt-4o, api_key: synthetic-key, api_base: {rescue}}}
general_settings: {{allow_private_api_base: true}}
router_settings:
  num_retries: 0
  allowed_fails: 10
  retry_policy: {{RateLimitErrorRetries: 1, BadRequestErrorRetries: 0, AuthenticationErrorRetries: 0}}
  fallbacks: [{{primary: [rescue]}}]
"#, primary = primary.base_url(), rescue = rescue.base_url())).unwrap();
        let failed_wire = primary
            .mock_async(|when, then| {
                when.method(POST);
                then.status(status)
                    .header("retry-after", "0")
                    .json_body(json!({"error": {"message": "provider error"}}));
            })
            .await;
        let served_wire = rescue
            .mock_async(|when, then| {
                when.method(POST);
                then.status(200).json_body(chat_response("gpt-4o"));
            })
            .await;
        let (proxy, _) = common::start_oxllm(config).await.unwrap();
        let response = common::http_client()
            .post(proxy.url("/v1/chat/completions"))
            .json(&json!({"model": "primary", "messages": [{"role": "user", "content": "hello"}]}))
            .send()
            .await
            .unwrap();
        if status == 400 {
            assert_eq!(response.status(), 400);
            failed_wire.assert_hits_async(1).await;
            served_wire.assert_hits_async(0).await;
            proxy.shutdown();
            continue;
        }
        assert_eq!(response.status(), 200, "status {status}");
        assert_eq!(
            response.headers()["x-oxllm-attempted-retries"],
            (expected_attempts - 1).to_string()
        );
        assert_eq!(response.headers()["x-oxllm-attempted-fallbacks"], "1");
        failed_wire.assert_hits_async(expected_attempts).await;
        served_wire.assert_hits_async(1).await;
        proxy.shutdown();
    }
    let rejected = parse_config_str(
        "model_list: []\nrouter_settings:\n  retry_policy: {BadRequestErrorRetries: 2}\n",
    )
    .unwrap_err();
    assert!(format!("{rejected:#}").contains("BadRequestErrorRetries"));
}

#[tokio::test]
async fn datadog_receives_terminal_provider_failures_with_deployment_attribution() {
    let upstream = MockServer::start_async().await;
    let intake = MockServer::start_async().await;
    let mut config = common::proxy_config(&upstream.base_url());
    config.oxllm_settings.datadog_config = Some(oxllm_server::config::types::DatadogConfig {
        api_key: "synthetic-dd-key".into(),
        endpoint: Some(intake.base_url()),
        service: Some("gateway".into()),
    });
    let upstream_wire = upstream
        .mock_async(|when, then| {
            when.method(POST);
            then.status(401)
                .json_body(json!({"error": {"message": "invalid upstream credential"}}));
        })
        .await;
    let logged = intake
        .mock_async(|when, then| {
            when.method(POST)
                .path("/api/v2/logs")
                .header("DD-API-KEY", "synthetic-dd-key")
                .body_contains("\"http_status\":401")
                .body_contains("\"deployment_id\"")
                .body_contains("\"status\":\"error\"");
            then.status(202);
        })
        .await;
    let (proxy, _) = common::start_oxllm(config).await.unwrap();
    let response = common::http_client()
        .post(proxy.url("/v1/chat/completions"))
        .json(&json!({"model": "gpt-4o", "messages": [{"role": "user", "content": "hello"}]}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 401);
    for _ in 0..80 {
        if logged.hits_async().await >= 1 {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }
    logged.assert_hits_async(1).await;
    upstream_wire.assert_hits_async(1).await;
    proxy.shutdown();
}

#[tokio::test]
async fn native_fallback_contract_is_consistent_for_json_sse_and_cache_hits() {
    for stream in [false, true] {
        let primary = MockServer::start_async().await;
        let serving = MockServer::start_async().await;
        let config = parse_config_str(&format!(
            r#"
model_list:
  - model_name: requested-alias
    oxllm_params: {{model: openai/gpt-4o, api_key: synthetic-key, api_base: {primary}}}
  - model_name: fallback-alias
    oxllm_params:
      model: openai/gpt-4o
      api_key: synthetic-key
      api_base: {serving}
      input_cost_per_token: 0.001
      output_cost_per_token: 0.002
    model_info: {{id: final-leg}}
general_settings: {{allow_private_api_base: true}}
router_settings:
  num_retries: 0
  fallbacks: [{{requested-alias: [fallback-alias]}}]
"#,
            primary = primary.base_url(),
            serving = serving.base_url()
        ))
        .unwrap();
        let failed = primary
            .mock_async(|when, then| {
                when.method(POST);
                then.status(503)
                    .delay(std::time::Duration::from_millis(20))
                    .json_body(json!({"error": {"message": "synthetic-private-error-detail"}}));
            })
            .await;
        let healthy = serving.mock_async(|when, then| {
            when.method(POST).matches(|request| request.body.as_ref().and_then(|body| serde_json::from_slice::<Value>(body).ok()).is_some_and(|body| body.get("include_fallback_errors").is_none()));
            let then = then.status(200).delay(std::time::Duration::from_millis(40));
            if stream {
                then.header("content-type", "text/event-stream").body(concat!(
                    "data: {\"id\":\"stream\",\"model\":\"gpt-4o-snapshot\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\n",
                    "data: {\"model\":\"gpt-4o-snapshot\",\"choices\":[],\"usage\":{\"prompt_tokens\":4,\"completion_tokens\":2,\"total_tokens\":6}}\n\n",
                    "data: [DONE]\n\n"
                ));
            } else { then.json_body(chat_response("gpt-4o-snapshot")); }
        }).await;
        let (proxy, _) = common::start_oxllm(config).await.unwrap();
        let payload = json!({"model": "requested-alias", "messages": [{"role": "user", "content": "hello"}], "stream": stream, "stream_options": {"include_usage": true}, "include_fallback_errors": true, "cache": {"insert": true}});
        let response = common::http_client()
            .post(proxy.url("/v1/chat/completions"))
            .json(&payload)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        for (name, value) in [
            ("x-oxllm-model-group", "fallback-alias"),
            ("x-oxllm-model-name", "openai/gpt-4o"),
            ("x-oxllm-requested-model", "requested-alias"),
            ("x-oxllm-deployment-id", "final-leg"),
            ("x-oxllm-attempted-fallbacks", "1"),
        ] {
            assert_eq!(response.headers()[name], value);
        }
        let errors = response.headers()["x-oxllm-fallback-errors"]
            .to_str()
            .unwrap();
        assert!(!errors.contains("synthetic-private-error-detail"));
        let errors: Value = serde_json::from_str(errors).unwrap();
        assert_eq!(errors.as_array().unwrap().len(), 1);
        assert_eq!(errors[0]["code"], 503);
        let timing = |name| {
            response.headers()[name]
                .to_str()
                .unwrap()
                .parse::<f64>()
                .unwrap()
        };
        let total = timing("x-oxllm-response-duration-ms");
        let upstream = timing("x-oxllm-upstream-duration-ms");
        let overhead = timing("x-oxllm-overhead-duration-ms");
        assert!(upstream >= 60.0, "{upstream}");
        assert!(total >= upstream && (total - upstream - overhead).abs() < 2.0);
        if !stream {
            assert!((timing("x-oxllm-response-cost") - 0.008).abs() < 1e-9);
        }
        let body = response.text().await.unwrap();
        if stream {
            for chunk in body
                .lines()
                .filter_map(|line| line.strip_prefix("data: "))
                .filter_map(|data| serde_json::from_str::<Value>(data).ok())
            {
                assert_eq!(chunk["model"], "fallback-alias");
            }
        } else {
            assert_eq!(
                serde_json::from_str::<Value>(&body).unwrap()["model"],
                "fallback-alias"
            );
            let cached = common::http_client()
                .post(proxy.url("/v1/chat/completions"))
                .json(&payload)
                .send()
                .await
                .unwrap();
            assert_eq!(cached.headers()["x-oxllm-cache-hit"], "true");
            assert_eq!(cached.headers()["x-oxllm-model-group"], "fallback-alias");
            assert_eq!(cached.headers()["x-oxllm-attempted-fallbacks"], "0");
            assert!(!cached.headers().contains_key("x-oxllm-deployment-id"));
            assert_eq!(
                cached.json::<Value>().await.unwrap()["model"],
                "fallback-alias"
            );
        }
        failed.assert_hits_async(1).await;
        healthy.assert_hits_async(1).await;
        proxy.shutdown();
    }
}

#[tokio::test]
async fn azure_chat_limits_are_mapped_without_losing_sampling_settings() {
    for version in ["2025-04-01-preview", "v1"] {
        let upstream = MockServer::start_async().await;
        let config = parse_config_str(&format!("model_list:\n  - model_name: native-chat\n    oxllm_params: {{model: azure/gpt-5-chat-latest, api_base: {}, api_key: synthetic-key, api_version: {version}}}\ngeneral_settings: {{allow_private_api_base: true}}\nrouter_settings: {{num_retries: 0}}\n", upstream.base_url())).unwrap();
        let wire = upstream
            .mock_async(|when, then| {
                when.method(POST)
                    .json_body_partial(
                        json!({"max_completion_tokens": 32, "temperature": 0.7, "top_p": 0.9})
                            .to_string(),
                    )
                    .matches(|request| {
                        request
                            .body
                            .as_ref()
                            .and_then(|body| serde_json::from_slice::<Value>(body).ok())
                            .is_some_and(|body| body.get("max_tokens").is_none())
                    });
                then.status(200)
                    .json_body(chat_response("gpt-5-chat-latest"));
            })
            .await;
        let (proxy, _) = common::start_oxllm(config).await.unwrap();
        let response = common::http_client().post(proxy.url("/v1/chat/completions")).json(&json!({"model": "native-chat", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 32, "temperature": 0.7, "top_p": 0.9})).send().await.unwrap();
        assert_eq!(response.status(), 200);
        assert_eq!(response.headers()["x-oxllm-model-group"], "native-chat");
        wire.assert_hits_async(1).await;
        proxy.shutdown();
    }
}

#[tokio::test]
async fn response_aliases_are_additive_to_the_native_header_contract() {
    let upstream = MockServer::start_async().await;
    let wire = upstream
        .mock_async(|when, then| {
            when.method(POST);
            then.status(200).json_body(chat_response("gpt-4o"));
        })
        .await;
    let mut config = common::proxy_config(&upstream.base_url());
    config.general_settings.proxy_cors_origins = Some(vec!["https://native-client.example".into()]);
    let (proxy, _) = common::start_oxllm(config).await.unwrap();
    let response = common::http_client()
        .post(proxy.url("/v1/chat/completions"))
        .header("x-oxllm-session-id", "native-session")
        .header("origin", "https://native-client.example")
        .json(&json!({"model": "gpt-4o", "messages": [{"role": "user", "content": "hello"}]}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let exposed = response.headers()["access-control-expose-headers"]
        .to_str()
        .unwrap();
    for header in [
        "x-oxllm-model-group",
        "x-oxllm-requested-model",
        "x-oxllm-model-name",
        "x-oxllm-response-duration-ms",
        "x-oxllm-fallback-errors",
    ] {
        assert!(
            exposed.split(',').any(|name| name.trim() == header),
            "{exposed}"
        );
    }
    for suffix in [
        "model-id",
        "model-name",
        "model-group",
        "attempted-retries",
        "attempted-fallbacks",
        "response-duration-ms",
        "overhead-duration-ms",
    ] {
        assert_eq!(
            response.headers()[format!("x-oxllm-{suffix}")],
            response.headers()[format!("x-litellm-{suffix}")]
        );
    }
    assert!(!response.headers().contains_key("x-oxllm-max-retries"));
    wire.assert_hits_async(1).await;
    proxy.shutdown();
}

#[tokio::test]
async fn routing_check_aliases_preserve_native_deployment_enforcement() {
    let upstream = MockServer::start_async().await;
    let mut config = common::proxy_config(&upstream.base_url());
    config.model_list[0].oxllm_params.rpm = Some(0);
    config.model_list[0].oxllm_params.max_budget = Some(100.0);
    config.router_settings.optional_pre_call_checks = Some(vec![
        "router_budget_limiting".into(),
        "enforce_model_rate_limits".into(),
    ]);
    let wire = upstream
        .mock_async(|when, then| {
            when.method(POST);
            then.status(200).json_body(chat_response("gpt-4o"));
        })
        .await;
    let (proxy, state) = common::start_oxllm(config).await.unwrap();
    assert_eq!(
        state.load().router.config().optional_pre_call_checks,
        ["budget_limiter"]
    );
    let response = common::http_client()
        .post(proxy.url("/v1/chat/completions"))
        .json(&json!({"model": "gpt-4o", "messages": [{"role": "user", "content": "hello"}]}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 429);
    wire.assert_hits_async(0).await;
    proxy.shutdown();
}

#[tokio::test]
async fn successful_same_deployment_retry_exposes_the_effective_native_budget() {
    use axum::response::IntoResponse;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    let calls = Arc::new(AtomicUsize::new(0));
    let count = calls.clone();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let router = axum::Router::new().route(
        "/chat/completions",
        axum::routing::post(move || {
            let count = count.clone();
            async move {
                if count.fetch_add(1, Ordering::SeqCst) == 0 {
                    let mut response = (
                        axum::http::StatusCode::TOO_MANY_REQUESTS,
                        axum::Json(json!({"error": {"message": "retry"}})),
                    )
                        .into_response();
                    response
                        .headers_mut()
                        .insert("retry-after", axum::http::HeaderValue::from_static("0"));
                    response
                } else {
                    axum::Json(chat_response("gpt-4o")).into_response()
                }
            }
        }),
    );
    let upstream = tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    let mut config = common::proxy_config(&base);
    config.router_settings.routing.num_retries = Some(1);
    config.router_settings.routing.allowed_fails = Some(10);
    let (proxy, _) = common::start_oxllm(config).await.unwrap();
    let response = common::http_client()
        .post(proxy.url("/v1/chat/completions"))
        .json(&json!({"model": "gpt-4o", "messages": [{"role": "user", "content": "hello"}]}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    assert_eq!(response.headers()["x-oxllm-attempted-retries"], "1");
    assert_eq!(response.headers()["x-oxllm-max-retries"], "1");
    assert_eq!(response.headers()["x-oxllm-attempted-fallbacks"], "0");
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    proxy.shutdown();
    upstream.abort();
}
