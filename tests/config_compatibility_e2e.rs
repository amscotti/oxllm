//! Wire-level regression tests for deployment configuration and billing.

mod common;

use httpmock::{Method::POST, MockServer};
use oxllm_server::config::parse_config_str;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

fn chat_response(prompt_tokens: u64, completion_tokens: u64) -> Value {
    json!({
        "id": "chat-test", "object": "chat.completion", "model": "gpt-4o",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens}
    })
}

fn native_credential_headers_absent(request: &httpmock::prelude::HttpMockRequest) -> bool {
    request.headers.as_ref().is_none_or(|headers| {
        headers.iter().all(|(name, _)| {
            !name.eq_ignore_ascii_case("authorization") && !name.eq_ignore_ascii_case("x-api-key")
        })
    })
}

#[tokio::test]
async fn failed_priority_members_are_not_repicked_with_zero_cooldown() {
    for endpoint in ["/v1/chat/completions", "/v1/embeddings"] {
        for status in [401, 403, 500] {
            let bad = MockServer::start_async().await;
            let good = MockServer::start_async().await;
            let mut config = common::proxy_config(&bad.base_url());
            config.model_list[0].oxllm_params.order = Some(1);
            if endpoint.ends_with("embeddings") {
                config.model_list[0].oxllm_params.model = "openai/text-embedding-3-small".into();
            }
            let mut sibling = config.model_list[0].clone();
            sibling.oxllm_params.api_base = Some(good.base_url());
            sibling.oxllm_params.order = Some(2);
            config.model_list.push(sibling);
            config.router_settings.routing.num_retries = Some(0);
            config.router_settings.routing.cooldown_time = Some(0);
            let failed = bad
                .mock_async(|when, then| {
                    when.method(POST);
                    then.status(status)
                        .json_body(json!({"error": {"message": "deployment failure"}}));
                })
                .await;
            let healthy = good.mock_async(|when, then| {
                when.method(POST);
                let body = if endpoint.ends_with("embeddings") {
                    json!({"data": [{"index": 0, "embedding": [0.1]}], "model": "text-embedding-3-small", "usage": {"prompt_tokens": 2, "total_tokens": 2}})
                } else { chat_response(2, 1) };
                then.status(200).json_body(body);
            }).await;
            let (proxy, _) = common::start_oxllm(config).await.unwrap();
            let payload = if endpoint.ends_with("embeddings") {
                json!({"model": "gpt-4o", "input": "hi"})
            } else {
                json!({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]})
            };
            let response = common::http_client()
                .post(proxy.url(endpoint))
                .json(&payload)
                .send()
                .await
                .unwrap();
            assert_eq!(
                response.status(),
                200,
                "{endpoint}/{status}: {}",
                response.text().await.unwrap()
            );
            failed.assert_hits_async(1).await;
            healthy.assert_hits_async(1).await;
            proxy.shutdown();
        }
    }
}

#[tokio::test]
async fn configured_request_shaping_reaches_the_chat_wire() {
    let upstream = MockServer::start_async().await;
    let config = parse_config_str(&format!(
        r#"
model_list:
  - model_name: shaped
    oxllm_params:
      model: openai/gpt-6-sol
      api_base: {}
      api_key: synthetic-key
      additional_drop_params: [temperature, max_tokens]
      allowed_openai_params: [reasoning_effort]
      extra_body: {{store: false}}
general_settings:
  allow_private_api_base: true
"#,
        upstream.base_url()
    ))
    .unwrap();
    let expected = json!({"model": "gpt-6-sol", "messages": [{"role": "user", "content": "hi"}],
        "reasoning_effort": "none", "store": false});
    let wire = upstream
        .mock_async(|when, then| {
            when.method(POST)
                .path("/chat/completions")
                .json_body(expected);
            then.status(200).json_body(chat_response(2, 1));
        })
        .await;
    let (proxy, _) = common::start_oxllm(config).await.unwrap();
    let response = common::http_client()
        .post(proxy.url("/v1/chat/completions"))
        .json(
            &json!({"model": "shaped", "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "none", "temperature": 0.3, "max_tokens": 40, "store": true}),
        )
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200, "{}", response.text().await.unwrap());
    wire.assert_hits_async(1).await;
    proxy.shutdown();
}

#[tokio::test]
async fn configured_privacy_controls_also_reach_native_responses() {
    let upstream = MockServer::start_async().await;
    let mut config = common::proxy_config(&upstream.base_url());
    let params = &mut config.model_list[0].oxllm_params;
    params.extra_body = Some(serde_json::Map::from_iter([("store".into(), json!(false))]));
    params.additional_drop_params = Some(vec!["temperature".into()]);
    let wire = upstream.mock_async(|when, then| {
        when.method(POST).path("/v1/responses").json_body(json!({"model": "gpt-4o", "input": "hi", "store": false}));
        then.status(200).json_body(json!({"id": "response-test", "object": "response", "model": "gpt-4o",
            "status": "completed", "output": [], "usage": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3}}));
    }).await;
    let (proxy, _) = common::start_oxllm(config).await.unwrap();
    let response = common::http_client()
        .post(proxy.url("/openai/v1/responses"))
        .json(&json!({"model": "gpt-4o", "input": "hi", "store": true, "temperature": 0.3}))
        .send()
        .await
        .unwrap();
    wire.assert_hits_async(1).await;
    assert_eq!(response.status(), 200, "{}", response.text().await.unwrap());
    proxy.shutdown();
}

#[tokio::test]
async fn signed_bedrock_embeddings_keep_headers_and_bill_body_header_or_estimated_usage() {
    let upstream = MockServer::start_async().await;
    let config = parse_config_str(&format!(
        r#"
model_list:
  - model_name: embed
    oxllm_params:
      model: bedrock/us.cohere.embed-v4:0
      api_base: {}
      aws_access_key_id: synthetic-access
      aws_secret_access_key: synthetic-secret
      aws_session_token: synthetic-session
      aws_region_name: us-west-2
      input_cost_per_token: 0.001
      output_cost_per_token: 0.0
      additional_drop_params: [encoding_format]
      allowed_openai_params: [dimensions]
      extra_headers: {{X-Configured: kept}}
general_settings:
  allow_private_api_base: true
router_settings:
  num_retries: 0
"#,
        upstream.base_url()
    ))
    .unwrap();
    let (proxy, _) = common::start_oxllm(config).await.unwrap();
    let client = common::http_client();
    for (text, receipt, expected_tokens) in [
        ("body metering", Some(12), 12),
        ("header metering", None, 7),
        (
            "estimated metering",
            None,
            oxllm_core::tokens::estimate_tokens("estimated metering", "cohere.embed-v4"),
        ),
    ] {
        let request = json!({"texts": [text], "input_type": "search_document", "truncate": "END", "output_dimension": 512});
        let hash = format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(&request).unwrap())
        );
        let mut body = json!({"embeddings": {"float": [[0.1, 0.2]], "int8": [[1, 2]]}});
        if let Some(tokens) = receipt {
            body["meta"] = json!({"billed_units": {"input_tokens": tokens}});
        }
        let wire = upstream
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/model/us.cohere.embed-v4:0/invoke")
                    .header_exists("authorization")
                    .header_exists("x-amz-date")
                    .header("x-amz-content-sha256", hash)
                    .header("x-amz-security-token", "synthetic-session")
                    .header("x-configured", "kept")
                    .json_body(request);
                let then = then.status(200).json_body(body);
                if text != "estimated metering" {
                    then.header("x-amzn-bedrock-input-token-count", "7");
                }
            })
            .await;
        let response = client
            .post(proxy.url("/v1/embeddings"))
            .json(&json!({"model": "embed", "input": [text], "encoding_format": "float", "dimensions": 512}))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 200, "{}", response.text().await.unwrap());
        let cost = response.headers()["x-oxllm-response-cost"]
            .to_str()
            .unwrap()
            .parse::<f64>()
            .unwrap();
        let body: Value = response.json().await.unwrap();
        assert_eq!(body["usage"]["prompt_tokens"], expected_tokens);
        assert_eq!(body["data"][0]["embedding"], json!([0.1, 0.2]));
        assert!((cost - expected_tokens as f64 * 0.001).abs() < 1e-9);
        wire.assert_hits_async(1).await;
        wire.delete_async().await;
    }
    proxy.shutdown();
}

#[tokio::test]
async fn context_tier_boundary_is_applied_to_the_response_cost_header() {
    let upstream = MockServer::start_async().await;
    let mut config = common::proxy_config(&upstream.base_url());
    let params = &mut config.model_list[0].oxllm_params;
    params.input_cost_per_token = Some(0.001);
    params.output_cost_per_token = Some(0.002);
    params.extra.insert(
        "input_cost_per_token_above_272k_tokens".into(),
        json!(0.003),
    );
    params.extra.insert(
        "output_cost_per_token_above_272k_tokens".into(),
        json!(0.004),
    );
    let (proxy, _) = common::start_oxllm(config).await.unwrap();
    for (tokens, input_rate, output_rate) in [(272_000, 0.001, 0.002), (272_001, 0.003, 0.004)] {
        let wire = upstream
            .mock_async(|when, then| {
                when.method(POST).path("/chat/completions");
                then.status(200).json_body(chat_response(tokens, 2));
            })
            .await;
        let response = common::http_client()
            .post(proxy.url("/v1/chat/completions"))
            .json(&json!({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        let cost = response.headers()["x-oxllm-response-cost"]
            .to_str()
            .unwrap()
            .parse::<f64>()
            .unwrap();
        assert!((cost - (tokens as f64 * input_rate + 2.0 * output_rate)).abs() < 1e-9);
        wire.assert_hits_async(1).await;
        wire.delete_async().await;
    }
    proxy.shutdown();
}

#[tokio::test]
async fn named_retry_policy_retries_rate_limits_and_immediately_falls_back_on_explicit_client_faults()
 {
    for (status, expected_attempts) in [(429, 2), (400, 1), (401, 1)] {
        let upstream = MockServer::start_async().await;
        let config = parse_config_str(&format!(
            r#"
model_list:
  - model_name: primary
    oxllm_params:
      model: openai/gpt-4o
      api_base: {base}
      api_key: primary-key
  - model_name: backup
    oxllm_params:
      model: openai/gpt-4o
      api_base: {base}
      api_key: backup-key
general_settings:
  allow_private_api_base: true
router_settings:
  num_retries: 0
  fallbacks: [{{primary: [backup]}}]
  retry_policy:
    TimeoutErrorRetries: 1
    RateLimitErrorRetries: 1
    BadRequestErrorRetries: 0
    AuthenticationErrorRetries: 0
    ContentPolicyViolationErrorRetries: 0
"#,
            base = upstream.base_url()
        ))
        .unwrap();
        let primary = upstream
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/chat/completions")
                    .header("authorization", "Bearer primary-key");
                then.status(status)
                    .json_body(json!({"error": {"message": "injected failure"}}));
            })
            .await;
        let backup = upstream
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/chat/completions")
                    .header("authorization", "Bearer backup-key");
                then.status(200).json_body(chat_response(2, 1));
            })
            .await;
        let (proxy, _) = common::start_oxllm(config).await.unwrap();
        let response = common::http_client()
            .post(proxy.url("/v1/chat/completions"))
            .json(&json!({"model": "primary", "messages": [{"role": "user", "content": "hi"}]}))
            .send()
            .await
            .unwrap();
        assert_eq!(
            response.status(),
            200,
            "status {status}: {}",
            response.text().await.unwrap()
        );
        assert_eq!(response.headers()["x-oxllm-fallback-used"], "backup");
        primary.assert_hits_async(expected_attempts).await;
        backup.assert_hits_async(1).await;
        proxy.shutdown();
    }
}

#[tokio::test]
async fn foundry_style_embeddings_use_an_openai_body_query_and_configured_auth_header() {
    let upstream = MockServer::start_async().await;
    let config = parse_config_str(&format!(
        r#"
model_list:
  - model_name: foundry-embed
    oxllm_params:
      model: openai/cohere-embed-v4
      api_base: {}/models/embeddings?api-version=2024-05-01-preview
      api_key: synthetic-key
      extra_headers: {{api-key: configured-key}}
      input_cost_per_token: 0.001
      output_cost_per_token: 0.0
general_settings:
  allow_private_api_base: true
router_settings:
  num_retries: 0
"#,
        upstream.base_url()
    ))
    .unwrap();
    let wire = upstream.mock_async(|when, then| {
        when.method(POST).path("/models/embeddings").query_param("api-version", "2024-05-01-preview")
            .header("api-key", "configured-key").json_body(json!({"model": "cohere-embed-v4", "input": ["hi"]}));
        then.status(200).json_body(json!({"object": "list", "data": [{"object": "embedding", "index": 0, "embedding": [0.1]}],
            "model": "cohere-embed-v4", "usage": {"prompt_tokens": 2, "total_tokens": 2}}));
    }).await;
    let (proxy, _) = common::start_oxllm(config).await.unwrap();
    let response = common::http_client()
        .post(proxy.url("/v1/embeddings"))
        .json(&json!({"model": "foundry-embed", "input": ["hi"]}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200, "{}", response.text().await.unwrap());
    assert!(response.headers().contains_key("x-oxllm-response-cost"));
    wire.assert_hits_async(1).await;
    proxy.shutdown();
}

#[tokio::test]
async fn authentication_overrides_replace_native_headers_on_all_three_vendor_wires() {
    for (model, embedding) in [
        ("anthropic/claude-haiku-4-5", false),
        ("xai/grok-4-fast-reasoning", false),
        ("cohere/embed-v4", true),
    ] {
        let upstream = MockServer::start_async().await;
        let config = parse_config_str(&format!(
            r#"
model_list:
  - model_name: configured
    oxllm_params:
      model: {model}
      api_base: {base}
      api_key: native-key
      auth_header: API-Key
      auth_scheme: raw
      extra_headers: {{api-key: final-key}}
general_settings:
  allow_private_api_base: true
router_settings:
  num_retries: 0
"#,
            base = upstream.base_url()
        ))
        .unwrap();
        let wire = upstream.mock_async(|when, then| {
            when.method(POST).header("api-key", "final-key")
                .matches(native_credential_headers_absent);
            let response = if embedding {
                json!({"embeddings": [[0.1]], "meta": {"billed_units": {"input_tokens": 2}}})
            } else if model.starts_with("anthropic/") {
                json!({"id": "msg-configured", "type": "message", "role": "assistant", "model": "claude-haiku-4-5", "content": [{"type": "text", "text": "ok"}],
                    "stop_reason": "end_turn", "usage": {"input_tokens": 2, "output_tokens": 1}})
            } else { chat_response(2, 1) };
            then.status(200).json_body(response);
        }).await;
        let (proxy, _) = common::start_oxllm(config).await.unwrap();
        let payload = if embedding {
            json!({"model": "configured", "input": ["hello"]})
        } else {
            json!({"model": "configured", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 32})
        };
        let response = common::http_client()
            .post(proxy.url(if embedding {
                "/v1/embeddings"
            } else {
                "/v1/chat/completions"
            }))
            .json(&payload)
            .send()
            .await
            .unwrap();
        assert_eq!(
            response.status(),
            200,
            "{model}: {}",
            response.text().await.unwrap()
        );
        wire.assert_hits_async(1).await;
        proxy.shutdown();
    }
}

#[tokio::test]
async fn literal_azure_ai_routes_serve_native_claude_compatible_chat_and_embeddings() {
    for (model, path, embedding) in [
        ("azure_ai/claude-haiku-4-5", "/anthropic/v1/messages", false),
        (
            "azure_ai/grok-4-fast-reasoning",
            "/models/chat/completions",
            false,
        ),
        ("azure_ai/cohere-deployment", "/models/embeddings", true),
    ] {
        let upstream = MockServer::start_async().await;
        let config = parse_config_str(&format!(
            r#"
model_list:
  - model_name: restored
    oxllm_params:
      model: {model}
      api_base: {base}{path}?api-version=2024-05-01-preview
      api_key: configured-key
      auth_header: api-key
      input_cost_per_token: 0.001
      output_cost_per_token: 0.002
general_settings:
  allow_private_api_base: true
router_settings:
  num_retries: 0
"#,
            base = upstream.base_url()
        ))
        .unwrap();
        let wire = upstream.mock_async(|when, then| {
            when.method(POST).path(path).query_param("api-version", "2024-05-01-preview")
                .header("api-key", "configured-key").matches(native_credential_headers_absent);
            let response = if embedding {
                json!({"data": [{"index": 0, "embedding": [0.1]}], "model": "cohere-deployment", "usage": {"prompt_tokens": 2, "total_tokens": 2}})
            } else if model.contains("claude") {
                json!({"id": "msg-restored", "type": "message", "role": "assistant", "model": "claude-haiku-4-5", "content": [{"type": "text", "text": "ok"}],
                    "stop_reason": "end_turn", "usage": {"input_tokens": 2, "output_tokens": 1}})
            } else { chat_response(2, 1) };
            then.status(200).json_body(response);
        }).await;
        let (proxy, _) = common::start_oxllm(config).await.unwrap();
        let payload = if embedding {
            json!({"model": "restored", "input": ["hello"]})
        } else {
            json!({"model": "restored", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 32})
        };
        let response = common::http_client()
            .post(proxy.url(if embedding {
                "/v1/embeddings"
            } else {
                "/v1/chat/completions"
            }))
            .json(&payload)
            .send()
            .await
            .unwrap();
        assert_eq!(
            response.status(),
            200,
            "{model}: {}",
            response.text().await.unwrap()
        );
        let cost = response.headers()["x-oxllm-response-cost"]
            .to_str()
            .unwrap()
            .parse::<f64>()
            .unwrap();
        assert!((cost - if embedding { 0.002 } else { 0.004 }).abs() < 1e-9);
        wire.assert_hits_async(1).await;
        proxy.shutdown();
    }
}

#[tokio::test]
async fn native_cohere_foundry_mode_uses_input_embeddings_and_replacement_auth() {
    let upstream = MockServer::start_async().await;
    let config = parse_config_str(&format!(
        r#"
model_list:
  - model_name: embed
    oxllm_params:
      model: cohere/my-embedding-deployment
      api_type: azure_ai
      api_base: {}/models?api-version=2024-05-01-preview
      api_key: configured-key
      input_cost_per_token: 0.001
      output_cost_per_token: 0.0
general_settings:
  allow_private_api_base: true
router_settings:
  num_retries: 0
"#,
        upstream.base_url()
    ))
    .unwrap();
    let wire = upstream.mock_async(|when, then| {
        when.method(POST).path("/models/embeddings").query_param("api-version", "2024-05-01-preview")
            .header("api-key", "configured-key").matches(native_credential_headers_absent)
            .json_body(json!({"model": "my-embedding-deployment", "input": ["hello"], "dimensions": 512}));
        then.status(200).json_body(json!({"data": [{"index": 0, "embedding": [0.1]}], "model": "my-embedding-deployment", "usage": {"prompt_tokens": 2, "total_tokens": 2}}));
    }).await;
    let (proxy, _) = common::start_oxllm(config).await.unwrap();
    let response = common::http_client()
        .post(proxy.url("/v1/embeddings"))
        .json(&json!({"model": "embed", "input": ["hello"], "dimensions": 512}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200, "{}", response.text().await.unwrap());
    assert_eq!(response.headers()["x-oxllm-response-cost"], "0.002");
    assert_eq!(
        response.json::<Value>().await.unwrap()["usage"]["prompt_tokens"],
        2
    );
    wire.assert_hits_async(1).await;
    proxy.shutdown();
}

#[tokio::test]
async fn foundry_embedding_usage_headers_drive_response_usage_and_billing() {
    for model in ["azure_ai/cohere-deployment", "cohere/cohere-deployment"] {
        for body_usage in [None, Some(2)] {
            let upstream = MockServer::start_async().await;
            let config = parse_config_str(&format!(
                r#"
model_list:
  - model_name: embed
    oxllm_params:
      model: {model}
      api_type: azure_ai
      api_base: {base}/models
      api_key: configured-key
      input_cost_per_token: 0.001
      output_cost_per_token: 0.0
general_settings:
  allow_private_api_base: true
router_settings:
  num_retries: 0
"#,
                base = upstream.base_url()
            ))
            .unwrap();
            let mut body =
                json!({"data": [{"index": 0, "embedding": [0.1]}], "model": "cohere-deployment"});
            if let Some(tokens) = body_usage {
                body["usage"] = json!({"prompt_tokens": tokens, "total_tokens": tokens});
            }
            let wire = upstream
                .mock_async(|when, then| {
                    when.method(POST).path("/models/embeddings");
                    then.status(200)
                        .header("llm_provider-num_tokens", "7")
                        .json_body(body);
                })
                .await;
            let (proxy, _) = common::start_oxllm(config).await.unwrap();
            let response = common::http_client()
                .post(proxy.url("/v1/embeddings"))
                .json(&json!({"model": "embed", "input": ["hello"]}))
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), 200);
            let cost = response.headers()["x-oxllm-response-cost"]
                .to_str()
                .unwrap()
                .parse::<f64>()
                .unwrap();
            assert!((cost - 0.007).abs() < 1e-9);
            assert_eq!(
                response.json::<Value>().await.unwrap()["usage"],
                json!({"prompt_tokens": 7, "total_tokens": 7})
            );
            wire.assert_hits_async(1).await;
            proxy.shutdown();
        }
    }
}

#[tokio::test]
async fn converse_grouping_and_global_history_repairs_reach_the_signed_wire() {
    for override_value in [None, Some(false)] {
        let upstream = MockServer::start_async().await;
        let mut config = parse_config_str(&format!(
            r#"
model_list:
  - model_name: normalized
    oxllm_params:
      model: bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0
      api_base: {}
      aws_access_key_id: synthetic-access
      aws_secret_access_key: synthetic-secret
      aws_region_name: us-east-1
      use_converse_api: true
oxllm_settings:
  modify_params: true
general_settings:
  allow_private_api_base: true
router_settings:
  num_retries: 0
"#,
            upstream.base_url()
        ))
        .unwrap();
        config.model_list[0].oxllm_params.modify_params = override_value;
        let messages = json!([
            {"role": "user", "content": "first"}, {"role": "user", "content": "second"},
            {"role": "assistant", "tool_calls": [
                {"id": "call_a", "type": "function", "function": {"name": "echo", "arguments": "{\"text\":\"a\"}"}},
                {"id": "call_b", "type": "function", "function": {"name": "echo", "arguments": "{\"text\":\"b\"}"}}
            ]},
            {"role": "tool", "tool_call_id": "call_a", "content": "a"},
            {"role": "tool", "tool_call_id": "call_b", "content": "b"},
            {"role": "user", "content": "summarize"}
        ]);
        let mut expected = json!({"messages": [
            {"role": "user", "content": [{"text": "first"}, {"text": "second"}]},
            {"role": "assistant", "content": [
                {"toolUse": {"toolUseId": "call_a", "name": "echo", "input": {"text": "a"}}},
                {"toolUse": {"toolUseId": "call_b", "name": "echo", "input": {"text": "b"}}}
            ]},
            {"role": "user", "content": [
                {"toolResult": {"toolUseId": "call_a", "content": [{"text": "a"}]}},
                {"toolResult": {"toolUseId": "call_b", "content": [{"text": "b"}]}}, {"text": "summarize"}
            ]}
        ], "inferenceConfig": {"maxTokens": 64}});
        if override_value.is_none() {
            expected["toolConfig"] = json!({"tools": [{"toolSpec": {"name": "echo", "description": "Historical function declaration.", "inputSchema": {"json": {"type": "object"}}}}]});
            let turns = expected["messages"].as_array_mut().unwrap();
            turns[2]["content"].as_array_mut().unwrap().pop();
            turns.push(json!({"role": "assistant", "content": [{"text": "Continue."}]}));
            turns.push(json!({"role": "user", "content": [{"text": "summarize"}]}));
        }
        let wire = upstream.mock_async(|when, then| {
            when.method(POST).path("/model/us.anthropic.claude-haiku-4-5-20251001-v1:0/converse")
                .header_exists("authorization").header_exists("x-amz-content-sha256").json_body(expected);
            then.status(200).json_body(json!({"output": {"message": {"role": "assistant", "content": [{"text": "ok"}]}}, "stopReason": "end_turn", "usage": {"inputTokens": 2, "outputTokens": 1, "totalTokens": 3}}));
        }).await;
        let (proxy, _) = common::start_oxllm(config).await.unwrap();
        let response = common::http_client()
            .post(proxy.url("/v1/chat/completions"))
            .json(&json!({"model": "normalized", "messages": messages, "max_tokens": 64}))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 200, "{}", response.text().await.unwrap());
        wire.assert_hits_async(1).await;
        proxy.shutdown();
    }
}

#[tokio::test]
async fn foundry_claude_streams_use_native_body_and_produce_openai_usage() {
    let upstream = MockServer::start_async().await;
    let config = parse_config_str(&format!(
        r#"
model_list:
  - model_name: restored
    oxllm_params:
      model: azure_ai/claude-haiku-4-5
      api_base: {}/anthropic/v1/messages
      api_key: configured-key
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
            .header("x-api-key", "configured-key").header("anthropic-version", "2023-06-01")
            .matches(|request| {
                request.body.as_ref().and_then(|body| serde_json::from_slice::<Value>(body).ok())
                    .is_some_and(|body| body["stream"] == true && body.get("stream_options").is_none())
            });
        then.status(200).header("content-type", "text/event-stream").body(concat!(
            "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg-stream\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-haiku-4-5\",\"content\":[],\"usage\":{\"input_tokens\":2,\"output_tokens\":0}}}\n\n",
            "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"ok\"}}\n\n",
            "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":1}}\n\n",
            "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"
        ));
    }).await;
    let (proxy, _) = common::start_oxllm(config).await.unwrap();
    let response = common::http_client().post(proxy.url("/v1/chat/completions"))
        .json(&json!({"model": "restored", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 32,
            "stream": true, "stream_options": {"include_usage": true}})).send().await.unwrap();
    assert_eq!(response.status(), 200);
    let text = response.text().await.unwrap();
    let chunks: Vec<Value> = text
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter_map(|data| serde_json::from_str(data).ok())
        .collect();
    assert!(
        chunks
            .iter()
            .any(|chunk| chunk["choices"][0]["delta"]["content"] == "ok"),
        "{text}"
    );
    assert!(
        chunks
            .iter()
            .any(|chunk| chunk["usage"]["prompt_tokens"] == 2
                && chunk["usage"]["completion_tokens"] == 1),
        "{text}"
    );
    wire.assert_hits_async(1).await;
    proxy.shutdown();
}
