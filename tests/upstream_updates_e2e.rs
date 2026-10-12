//! Wire regressions for upstream provider and router updates.

mod common;

use httpmock::{Method::POST, MockServer};
use oxllm_server::config::parse_config_str;
use serde_json::{Value, json};

fn reply(model: &str) -> Value {
    json!({"id":"updated", "object":"chat.completion", "model":model,
        "choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],
        "usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}})
}

#[tokio::test]
async fn native_pre_content_error_exhaustion_preserves_the_status_and_retry_budget() {
    let upstream = MockServer::start_async().await;
    let config = parse_config_str(&format!(
        r#"
model_list:
  - model_name: limited
    oxllm_params:
      model: anthropic/claude-haiku-5-5
      api_base: {}/v1
      api_key: synthetic-key
general_settings:
  allow_private_api_base: true
router_settings:
  num_retries: 0
  retry_policy: {{"529": 1}}
"#,
        upstream.base_url()
    ))
    .unwrap();
    let wire = upstream.mock_async(|when, then| {
        when.method(POST).path("/v1/messages");
        then.status(200).header("content-type", "text/event-stream")
            .body("event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"overloaded\"}}\n\n");
    }).await;
    let (proxy, _) = common::start_oxllm(config).await.unwrap();
    let response = common::http_client()
        .post(proxy.url("/v1/messages"))
        .json(&json!({
            "model":"limited", "stream":true, "max_tokens":32,
            "messages":[{"role":"user","content":"hello"}]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 529);
    assert!(
        response.json::<Value>().await.unwrap()["error"]["message"]
            .as_str()
            .unwrap()
            .contains("overloaded")
    );
    wire.assert_hits_async(2).await;
    proxy.shutdown();
}

#[tokio::test]
async fn malformed_stream_preludes_use_authorized_fallbacks_without_committing_metadata() {
    let upstream = MockServer::start_async().await;
    let config = parse_config_str(&format!(
        r#"
model_list:
  - model_name: primary
    oxllm_params:
      model: anthropic/claude-haiku-5-5
      api_base: {base}/v1
      api_key: synthetic-key
  - model_name: rescue
    oxllm_params:
      model: openai/gpt-6-luna
      api_base: {base}
      api_key: synthetic-key
general_settings:
  allow_private_api_base: true
router_settings:
  num_retries: 0
  fallbacks: [{{primary: [rescue]}}]
"#,
        base = upstream.base_url()
    ))
    .unwrap();
    let primary = upstream.mock_async(|when, then| {
        when.method(POST).path("/v1/messages");
        then.status(200).header("content-type", "text/event-stream")
            .body("event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"failed-prelude\",\"model\":\"claude-haiku-5-5\"}}\n\nevent: content_block_delta\ndata: invalid-json-private-detail\n\n");
    }).await;
    let rescue = upstream.mock_async(|when, then| {
        when.method(POST).path("/chat/completions");
        then.status(200).header("content-type", "text/event-stream")
            .body("data: {\"id\":\"healthy\",\"model\":\"gpt-6-luna\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"rescued\"},\"finish_reason\":null}]}\n\ndata: [DONE]\n\n");
    }).await;
    let (proxy, _) = common::start_oxllm(config).await.unwrap();
    let response = common::http_client()
        .post(proxy.url("/v1/chat/completions"))
        .json(&json!({
            "model":"primary", "stream":true, "max_tokens":32, "include_fallback_errors":true,
            "messages":[{"role":"user","content":"hello"}]
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200, "{}", response.text().await.unwrap());
    assert_eq!(response.headers()["x-oxllm-fallback-used"], "rescue");
    let errors: Value = serde_json::from_str(
        response.headers()["x-oxllm-fallback-errors"]
            .to_str()
            .unwrap(),
    )
    .unwrap();
    assert_eq!(errors[0]["code"], 502);
    let body = response.text().await.unwrap();
    assert!(body.contains("rescued"), "{body}");
    assert!(
        !body.contains("failed-prelude") && !body.contains("private-detail"),
        "{body}"
    );
    primary.assert_hits_async(1).await;
    rescue.assert_hits_async(1).await;
    proxy.shutdown();
}

#[tokio::test]
async fn thinking_signature_repair_reports_the_effective_retry_budget() {
    let upstream = MockServer::start_async().await;
    let config = parse_config_str(&format!(
        r#"
model_list:
  - model_name: repair
    oxllm_params:
      model: anthropic/claude-haiku-5-5
      api_base: {}/v1
      api_key: synthetic-key
general_settings:
  allow_private_api_base: true
router_settings:
  num_retries: 0
"#,
        upstream.base_url()
    ))
    .unwrap();
    let invalid = upstream.mock_async(|when, then| {
        when.method(POST).path("/v1/messages").matches(|request| {
            request.body.as_ref().and_then(|body| serde_json::from_slice::<Value>(body).ok())
                .is_some_and(|body| body.get("thinking").is_some())
        });
        then.status(400).json_body(json!({"error":{"type":"invalid_request_error", "message":"Invalid thinking signature"}}));
    }).await;
    let repaired = upstream.mock_async(|when, then| {
        when.method(POST).path("/v1/messages").matches(|request| {
            request.body.as_ref().and_then(|body| serde_json::from_slice::<Value>(body).ok())
                .is_some_and(|body| body.get("thinking").is_none())
        });
        then.status(200).json_body(json!({"id":"repaired", "type":"message", "role":"assistant", "model":"claude-haiku-5-5",
            "content":[{"type":"text", "text":"ok"}], "stop_reason":"end_turn", "usage":{"input_tokens":10, "output_tokens":2}}));
    }).await;
    let (proxy, _) = common::start_oxllm(config).await.unwrap();
    let response = common::http_client().post(proxy.url("/v1/chat/completions")).json(&json!({
        "model":"repair", "max_tokens":2048, "thinking":{"type":"enabled", "budget_tokens":1024},
        "messages":[{"role":"user", "content":"hello"}]
    })).send().await.unwrap();
    assert_eq!(response.status(), 200, "{}", response.text().await.unwrap());
    assert_eq!(response.headers()["x-oxllm-attempted-retries"], "1");
    assert_eq!(response.headers()["x-oxllm-max-retries"], "1");
    invalid.assert_hits_async(1).await;
    repaired.assert_hits_async(1).await;
    proxy.shutdown();
}

#[tokio::test]
async fn request_fallback_overrides_disable_controls_and_zero_retries_stay_off_the_wire() {
    let upstream = MockServer::start_async().await;
    let config = parse_config_str(&format!(
        r#"
model_list:
  - model_name: primary
    oxllm_params:
      model: openai/gpt-6.1-sol
      api_base: {base}
      api_key: synthetic-key
  - model_name: rescue
    oxllm_params:
      model: openai/gpt-6-luna
      api_base: {base}
      api_key: synthetic-key
general_settings:
  allow_private_api_base: true
router_settings:
  num_retries: 3
  cooldown_time: 0
  fallbacks: [{{primary: [configured]}}]
  default_fallbacks: [rescue]
"#,
        base = upstream.base_url()
    ))
    .unwrap();
    let primary = upstream
        .mock_async(|when, then| {
            when.method(POST)
                .path("/chat/completions")
                .json_body_partial(json!({"model":"gpt-6.1-sol"}).to_string())
                .matches(|request| {
                    request
                        .body
                        .as_ref()
                        .and_then(|body| serde_json::from_slice::<Value>(body).ok())
                        .is_some_and(|body| {
                            [
                                "fallbacks",
                                "num_retries",
                                "disable_fallbacks",
                                "max_fallbacks",
                                "include_fallback_errors",
                            ]
                            .iter()
                            .all(|field| body.get(*field).is_none())
                        })
                });
            then.status(503)
                .json_body(json!({"error":{"message":"primary unavailable"}}));
        })
        .await;
    let rescue = upstream
        .mock_async(|when, then| {
            when.method(POST)
                .path("/chat/completions")
                .json_body_partial(json!({"model":"gpt-6-luna"}).to_string());
            then.status(200).json_body(reply("gpt-6-luna"));
        })
        .await;
    let (proxy, _) = common::start_oxllm(config).await.unwrap();
    let mut body = json!({"model":"primary", "num_retries":0, "fallbacks":["rescue"], "include_fallback_errors":true,
        "messages":[{"role":"user","content":"hello"}]});
    let response = common::http_client()
        .post(proxy.url("/v1/chat/completions"))
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200, "{}", response.text().await.unwrap());
    assert_eq!(response.headers()["x-oxllm-attempted-retries"], "0");
    assert_eq!(response.headers()["x-oxllm-fallback-used"], "rescue");
    assert!(response.headers().contains_key("x-oxllm-fallback-errors"));
    for control in [
        json!({"fallbacks":[]}),
        json!({"disable_fallbacks":true}),
        json!({"max_fallbacks":0}),
    ] {
        let mut request = body.clone();
        request
            .as_object_mut()
            .unwrap()
            .extend(control.as_object().unwrap().clone());
        let response = common::http_client()
            .post(proxy.url("/v1/chat/completions"))
            .json(&request)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 503);
    }
    body["model"] = json!("unknown");
    body["max_fallbacks"] = json!(0);
    let response = common::http_client()
        .post(proxy.url("/v1/chat/completions"))
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 404);
    primary.assert_hits_async(4).await;
    rescue.assert_hits_async(1).await;
    proxy.shutdown();
}

#[tokio::test]
#[serial_test::serial(token_counter)]
async fn invalid_request_controls_are_rejected_before_any_provider_call() {
    let upstream = MockServer::start_async().await;
    let config = parse_config_str(&format!(
        r#"
model_list:
  - model_name: controlled
    oxllm_params:
      model: openai/gpt-6.1-sol
      api_base: {}
      api_key: synthetic-key
  - model_name: native
    oxllm_params:
      model: anthropic/claude-haiku-5-5
      api_base: {}
      api_key: synthetic-key
general_settings:
  allow_private_api_base: true
"#,
        upstream.base_url(),
        upstream.base_url()
    ))
    .unwrap();
    let wire = upstream
        .mock_async(|when, then| {
            when.method(POST);
            then.status(200).json_body(reply("gpt-6.1-sol"));
        })
        .await;
    let (proxy, _) = common::start_oxllm(config).await.unwrap();
    for control in [
        json!({"timeout":-1}),
        json!({"stream_timeout":"slow"}),
        json!({"num_retries":-1}),
        json!({"num_retries":11}),
        json!({"num_retries":u32::MAX}),
        json!({"fallbacks":[1]}),
        json!({"disable_fallbacks":"true"}),
        json!({"include_fallback_errors":"true"}),
        json!({"anthropic_identity_token":"oidc/env/SECRET"}),
        json!({"anthropic_disable_workload_identity_federation":true}),
        json!({"anthropic_disable_workload_identity_federation":false}),
        json!({"anthropic_federation_rule_id":null}),
    ] {
        for (path, model) in [
            ("/v1/chat/completions", "controlled"),
            ("/v1/messages", "controlled"),
            ("/v1/responses", "controlled"),
            ("/v1/completions", "controlled"),
            ("/v1/embeddings", "controlled"),
            ("/v1/messages/count_tokens", "native"),
            ("/utils/token_counter", "native"),
            ("/anthropic/v1/messages", "native"),
        ] {
            let mut body = json!({"model":model, "max_tokens":32, "input":"hello", "prompt":"hello",
                "messages":[{"role":"user","content":"hello"}]});
            body.as_object_mut()
                .unwrap()
                .extend(control.as_object().unwrap().clone());
            let response = common::http_client()
                .post(proxy.url(path))
                .json(&body)
                .send()
                .await
                .unwrap();
            assert_eq!(
                response.status(),
                400,
                "{path}: {control}: {}",
                response.text().await.unwrap()
            );
        }
    }
    wire.assert_hits_async(0).await;
    proxy.shutdown();
}

#[tokio::test]
async fn converted_api_surfaces_preserve_opt_in_fallback_error_headers() {
    let upstream = MockServer::start_async().await;
    let config = parse_config_str(&format!(
        r#"
model_list:
  - model_name: primary
    oxllm_params:
      model: openai/gpt-6.1-sol
      api_base: {base}
      api_key: synthetic-key
  - model_name: rescue
    oxllm_params:
      model: openai/gpt-6-luna
      api_base: {base}
      api_key: synthetic-key
general_settings:
  allow_private_api_base: true
router_settings:
  num_retries: 0
  cooldown_time: 0
  fallbacks: [{{primary: [rescue]}}]
"#,
        base = upstream.base_url()
    ))
    .unwrap();
    let primary = upstream
        .mock_async(|when, then| {
            when.method(POST)
                .path("/chat/completions")
                .json_body_partial(json!({"model":"gpt-6.1-sol"}).to_string());
            then.status(503)
                .json_body(json!({"error":{"message":"unavailable-private-detail"}}));
        })
        .await;
    let rescue = upstream
        .mock_async(|when, then| {
            when.method(POST)
                .path("/chat/completions")
                .json_body_partial(json!({"model":"gpt-6-luna"}).to_string());
            then.status(200).json_body(reply("gpt-6-luna"));
        })
        .await;
    let (proxy, _) = common::start_oxllm(config).await.unwrap();
    for path in ["/v1/messages", "/v1/responses", "/v1/completions"] {
        let response = common::http_client().post(proxy.url(path)).json(&json!({
            "model":"primary", "max_tokens":32, "input":"hello", "prompt":"hello", "include_fallback_errors":true,
            "messages":[{"role":"user", "content":"hello"}]
        })).send().await.unwrap();
        assert_eq!(
            response.status(),
            200,
            "{path}: {}",
            response.text().await.unwrap()
        );
        assert_eq!(response.headers()["x-oxllm-fallback-used"], "rescue");
        let errors = response.headers()["x-oxllm-fallback-errors"]
            .to_str()
            .unwrap();
        assert_eq!(
            serde_json::from_str::<Value>(errors).unwrap()[0]["code"],
            503
        );
        assert!(!errors.contains("private-detail"));
    }
    primary.assert_hits_async(3).await;
    rescue.assert_hits_async(3).await;
    proxy.shutdown();
}

#[tokio::test]
async fn standalone_gateway_controls_and_final_body_overrides_stay_off_native_wires() {
    let upstream = MockServer::start_async().await;
    let overrides = json!({
        "num_retries":8, "timeout":10, "stream_timeout":10, "fallbacks":["other"],
        "include_fallback_errors":true, "log_client_error_tracebacks":true, "stream_chunk_size":16,
        "cache":{"use_cache":false, "insert":false}, "cache_control_injection_points":[],
        "metadata":{"tags":["private-tag"], "user_id":"kept"},
        "anthropic_identity_token":"should-never-reach-provider", "anthropic_disable_workload_identity_federation":true
    });
    let config = serde_json::from_value(json!({
        "model_list":[
            {"model_name":"native", "oxllm_params":{"model":"anthropic/claude-haiku-5-5", "api_base":format!("{}/v1", upstream.base_url()), "api_key":"synthetic-key"}},
            {"model_name":"shaped", "oxllm_params":{"model":"anthropic/claude-haiku-5-5", "api_base":format!("{}/v1", upstream.base_url()), "api_key":"synthetic-key", "extra_body":overrides}},
            {"model_name":"chat", "oxllm_params":{"model":"openai/gpt-6.1-sol", "api_base":upstream.base_url(), "api_key":"synthetic-key", "extra_body":overrides}},
            {"model_name":"embed", "oxllm_params":{"model":"openai/text-embedding-3-small", "api_base":upstream.base_url(), "api_key":"synthetic-key", "extra_body":overrides}}
        ],
        "general_settings":{"allow_private_api_base":true}, "router_settings":{"num_retries":0}
    })).unwrap();
    fn clean_body(body: Value) -> bool {
        [
            "fallbacks",
            "context_window_fallbacks",
            "content_policy_fallbacks",
            "disable_fallbacks",
            "max_fallbacks",
            "num_retries",
            "timeout",
            "stream_timeout",
            "include_fallback_errors",
            "log_client_error_tracebacks",
            "stream_chunk_size",
            "cache",
            "cache_control_injection_points",
            "anthropic_identity_token",
            "anthropic_disable_workload_identity_federation",
        ]
        .iter()
        .all(|field| body.get(*field).is_none())
            && body.pointer("/metadata/tags").is_none()
    }
    let native = upstream.mock_async(|when, then| {
        when.method(POST).path("/v1/messages").matches(|request| {
            request.body.as_ref().and_then(|body| serde_json::from_slice::<Value>(body).ok()).is_some_and(clean_body)
        });
        then.status(200).json_body(json!({"id":"native", "type":"message", "model":"claude-haiku-5-5", "role":"assistant",
            "content":[{"type":"text", "text":"ok"}], "stop_reason":"end_turn", "usage":{"input_tokens":10,"output_tokens":2}}));
    }).await;
    let chat = upstream
        .mock_async(|when, then| {
            when.method(POST)
                .path("/chat/completions")
                .matches(|request| {
                    request
                        .body
                        .as_ref()
                        .and_then(|body| serde_json::from_slice::<Value>(body).ok())
                        .is_some_and(clean_body)
                })
                .json_body_partial(json!({"metadata":{"user_id":"kept"}}).to_string());
            then.status(200).json_body(reply("gpt-6.1-sol"));
        })
        .await;
    let embeddings = upstream.mock_async(|when, then| {
        when.method(POST).path("/embeddings").matches(|request| {
            request.body.as_ref().and_then(|body| serde_json::from_slice::<Value>(body).ok()).is_some_and(clean_body)
        })
            .json_body_partial(json!({"metadata":{"user_id":"kept"}}).to_string());
        then.status(200).json_body(json!({"object":"list", "model":"text-embedding-3-small", "data":[{"object":"embedding", "index":0, "embedding":[0.1,0.2]}], "usage":{"prompt_tokens":2,"total_tokens":2}}));
    }).await;
    let (proxy, _) = common::start_oxllm(config).await.unwrap();
    for control in [
        json!({"include_fallback_errors":true}),
        json!({"log_client_error_tracebacks":true}),
        json!({"stream_chunk_size":16}),
        json!({"cache":{"use_cache":false,"insert":false}}),
        json!({"cache_control_injection_points":[{"location":"message","index":0}]}),
        json!({"metadata":{"tags":["private-tag"],"user_id":"kept"}}),
    ] {
        let mut body = json!({"model":"native", "max_tokens":32, "messages":[{"role":"user","content":"hello"}]});
        body.as_object_mut()
            .unwrap()
            .extend(control.as_object().unwrap().clone());
        let response = common::http_client()
            .post(proxy.url("/anthropic/v1/messages"))
            .json(&body)
            .send()
            .await
            .unwrap();
        assert_eq!(
            response.status(),
            200,
            "{control}: {}",
            response.text().await.unwrap()
        );
    }
    for (path, model) in [
        ("/anthropic/v1/messages", "shaped"),
        ("/v1/chat/completions", "chat"),
        ("/v1/embeddings", "embed"),
    ] {
        let response = common::http_client().post(proxy.url(path)).json(&json!({
            "model":model, "max_tokens":32, "input":"hello", "messages":[{"role":"user","content":"hello"}]
        })).send().await.unwrap();
        assert_eq!(
            response.status(),
            200,
            "{path}: {}",
            response.text().await.unwrap()
        );
    }
    native.assert_hits_async(7).await;
    chat.assert_hits_async(1).await;
    embeddings.assert_hits_async(1).await;
    proxy.shutdown();
}

#[tokio::test]
async fn caller_trace_headers_win_over_configured_fallbacks_without_overwriting_body_ids() {
    const TRACE: &str = "00-11111111111111111111111111111111-2222222222222222-01";
    let upstream = MockServer::start_async().await;
    let config = parse_config_str(&format!(
        r#"
model_list:
  - model_name: traced
    oxllm_params:
      model: openai/gpt-6.1-sol
      api_base: {}
      api_key: synthetic-key
      allowed_openai_params: [metadata]
      extra_headers:
        traceparent: "00-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bbbbbbbbbbbbbbbb-01"
        baggage: "session_id=stale"
general_settings:
  allow_private_api_base: true
"#,
        upstream.base_url()
    ))
    .unwrap();
    let wire = upstream
        .mock_async(|when, then| {
            when.method(POST)
                .path("/chat/completions")
                .header("traceparent", TRACE)
                .header("baggage", "session_id=header-session")
                .json_body_partial(
                    json!({"metadata":{"trace_id":"chosen-trace", "session_id":"chosen-session"}})
                        .to_string(),
                );
            then.status(200).json_body(reply("gpt-6.1-sol"));
        })
        .await;
    let (proxy, _) = common::start_oxllm(config).await.unwrap();
    let response = common::http_client()
        .post(proxy.url("/v1/chat/completions"))
        .header("traceparent", TRACE)
        .header("baggage", "session_id=header-session")
        .json(&json!({
            "model":"traced", "messages":[{"role":"user","content":"hello"}],
            "metadata":{"trace_id":"chosen-trace", "session_id":"chosen-session"}
        }))
        .send()
        .await
        .unwrap();
    wire.assert_hits_async(1).await;
    assert_eq!(response.status(), 200, "{}", response.text().await.unwrap());
    proxy.shutdown();
}

#[tokio::test]
#[serial_test::serial(token_counter)]
async fn vertex_partner_token_counting_keeps_system_tools_and_unversioned_models() {
    let upstream = MockServer::start_async().await;
    let config = parse_config_str(&format!(
        r#"
model_list:
  - model_name: counted
    oxllm_params:
      model: vertex_ai/claude-haiku-5-5@default
      api_base: {}
      vertex_project: project
      vertex_location: global
      vertex_count_tokens_location: us-east5
      api_key: synthetic-token
general_settings:
  allow_private_api_base: true
router_settings:
  num_retries: 0
"#,
        upstream.base_url()
    ))
    .unwrap();
    let system = json!([{"type":"text","text":"account for this system prefix"}]);
    let tools = json!([{"name":"weather","description":"look up weather","input_schema":{"type":"object","properties":{"city":{"type":"string"}}}}]);
    let wire = upstream.mock_async(|when, then| {
        when.method(POST).path("/v1/projects/project/locations/us-east5/publishers/anthropic/models/count-tokens:rawPredict")
            .header("authorization", "Bearer synthetic-token")
            .json_body_partial(json!({"model":"claude-haiku-5-5", "system":system, "tools":tools}).to_string())
            .matches(|request| request.body.as_ref().and_then(|body| serde_json::from_slice::<Value>(body).ok())
                .is_some_and(|body| body.get("max_tokens").is_none() && body.get("anthropic_version").is_none()));
        then.status(200).json_body(json!({"input_tokens":1234}));
    }).await;
    let (proxy, _) = common::start_oxllm(config).await.unwrap();
    let body = json!({"model":"counted", "messages":[{"role":"user","content":"hello"}], "system":system, "tools":tools});
    let response = common::http_client()
        .post(proxy.url("/v1/messages/count_tokens"))
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200, "{}", response.text().await.unwrap());
    assert_eq!(
        response.json::<Value>().await.unwrap()["input_tokens"],
        1234
    );
    let mut utility = body.clone();
    utility["call_endpoint"] = json!(true);
    let response = common::http_client()
        .post(proxy.url("/utils/token_counter"))
        .json(&utility)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200, "{}", response.text().await.unwrap());
    let count: Value = response.json().await.unwrap();
    assert_eq!(count["total_tokens"], 1234);
    assert_eq!(count["tokenizer_type"], "vertex_ai_partner_models");
    for path in ["/v1/messages/count_tokens", "/utils/token_counter"] {
        utility["tools"] = json!(["malformed"]);
        let response = common::http_client()
            .post(proxy.url(path))
            .json(&utility)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 400);
    }
    wire.assert_hits_async(2).await;
    proxy.shutdown();
}

#[tokio::test]
#[serial_test::serial(token_counter)]
async fn remote_token_count_failures_fall_back_locally_unless_disabled() {
    let upstream = MockServer::start_async().await;
    let config = parse_config_str(&format!(
        r#"
model_list:
  - model_name: counted
    oxllm_params:
      model: vertex_ai/claude-haiku-5-5
      api_base: {}
      vertex_project: project
      vertex_location: us-east5
      api_key: synthetic-token
general_settings:
  allow_private_api_base: true
"#,
        upstream.base_url()
    ))
    .unwrap();
    let wire = upstream.mock_async(|when, then| {
        when.method(POST).path("/v1/projects/project/locations/us-east5/publishers/anthropic/models/count-tokens:rawPredict")
            .matches(|request| request.body.as_ref().and_then(|body| serde_json::from_slice::<Value>(body).ok())
                .is_some_and(|body| body.get("system").is_none() && body.get("tools").is_none()));
        then.status(503).json_body(json!({"error":{"message":"count unavailable"}}));
    }).await;
    let (proxy, _) = common::start_oxllm(config).await.unwrap();
    let mut body = json!({"model":"counted", "call_endpoint":true, "system":null, "tools":null,
        "messages":[{"role":"user","content":"hello there"}]});
    let response = common::http_client()
        .post(proxy.url("/utils/token_counter"))
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let count: Value = response.json().await.unwrap();
    assert_eq!(count["tokenizer_type"], "openai_tokenizer");
    assert!(count["total_tokens"].as_u64().unwrap() > 0);
    body["disable_token_counter"] = json!(true);
    let response = common::http_client()
        .post(proxy.url("/utils/token_counter"))
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 503);
    wire.assert_hits_async(2).await;
    proxy.shutdown();
}

#[tokio::test]
async fn native_token_counting_normalizes_images_in_messages_and_tool_results() {
    let upstream = MockServer::start_async().await;
    let config = parse_config_str(&format!(
        r#"
model_list:
  - model_name: counted
    oxllm_params:
      model: anthropic/claude-haiku-5-5
      api_base: {}/v1
      api_key: synthetic-key
general_settings:
  allow_private_api_base: true
"#,
        upstream.base_url()
    ))
    .unwrap();
    let wire = upstream.mock_async(|when, then| {
        when.method(POST).path("/v1/messages/count_tokens").header("x-api-key", "synthetic-key")
            .json_body_partial(json!({"messages":[{"role":"user","content":[
                {"type":"image","source":{"type":"base64","media_type":"image/png","data":"aGVsbG8="}},
                {"type":"tool_result","tool_use_id":"tool-1","content":[{"type":"image","source":{"type":"url","url":"https://images.example/picture.png"}}]}
            ]}]}).to_string());
        then.status(200).json_body(json!({"input_tokens":87}));
    }).await;
    let (proxy, _) = common::start_oxllm(config).await.unwrap();
    let response = common::http_client().post(proxy.url("/v1/messages/count_tokens")).json(&json!({
        "model":"counted", "messages":[{"role":"user","content":[
            {"type":"image_url","image_url":{"url":"data:image/png;base64,aGVsbG8="}},
            {"type":"tool_result","tool_use_id":"tool-1","content":[{"type":"image_url","image_url":"https://images.example/picture.png"}]}
        ]}]
    })).send().await.unwrap();
    assert_eq!(response.status(), 200, "{}", response.text().await.unwrap());
    assert_eq!(response.json::<Value>().await.unwrap()["input_tokens"], 87);
    wire.assert_hits_async(1).await;
    proxy.shutdown();
}

#[tokio::test]
async fn gemini_chat_drops_foreign_thinking_signatures_and_keeps_native_provider_fields() {
    for provider in ["gemini", "vertex_ai"] {
        let upstream = MockServer::start_async().await;
        let config = parse_config_str(&format!(
            r#"
model_list:
  - model_name: mixed
    oxllm_params:
      model: {provider}/gemini-3.8-flash
      api_base: {}
      api_key: synthetic-token
      vertex_project: project
      vertex_location: us
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
                when.method(POST).matches(|request| {
                    request
                        .body
                        .as_ref()
                        .and_then(|body| serde_json::from_slice::<Value>(body).ok())
                        .is_some_and(|body| {
                            let message = &body["messages"][1];
                            message.get("thinking_blocks").is_none()
                                && message["reasoning_content"] == "one reasoning copy"
                                && message["provider_specific_fields"]["thought_signature"]
                                    == "native-google-signature"
                                && body.to_string().matches("one reasoning copy").count() == 1
                        })
                });
                then.status(200).json_body(reply("gemini-3.8-flash"));
            })
            .await;
        let (proxy, _) = common::start_oxllm(config).await.unwrap();
        let response = common::http_client().post(proxy.url("/v1/chat/completions")).json(&json!({
            "model":"mixed", "messages":[{"role":"user","content":"hello"},
                {"role":"assistant","content":"answer", "reasoning_content":"one reasoning copy",
                    "thinking_blocks":[{"type":"thinking","thinking":"one reasoning copy","signature":"foreign-claude-signature"}],
                    "provider_specific_fields":{"thought_signature":"native-google-signature"}}]
        })).send().await.unwrap();
        assert_eq!(response.status(), 200, "{}", response.text().await.unwrap());
        wire.assert_hits_async(1).await;
        proxy.shutdown();
    }
}

#[tokio::test]
async fn mantle_cache_breakpoints_reach_every_translated_api_surface() {
    let upstream = MockServer::start_async().await;
    let config = parse_config_str(&format!(
        r#"
model_list:
  - model_name: cacheable
    oxllm_params:
      model: bedrock_mantle/openai.gpt-6.1-sol
      api_base: {}
      api_key: synthetic-key
      drop_params: true
      cache_control_injection_points:
        - {{location: message, role: system}}
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
                .path("/chat/completions")
                .matches(|request| {
                    request
                        .body
                        .as_ref()
                        .and_then(|body| serde_json::from_slice::<Value>(body).ok())
                        .is_some_and(|body| {
                            body["messages"][0]["content"][0]["prompt_cache_breakpoint"]["mode"]
                                == "explicit"
                                && body["prompt_cache_options"]["mode"] == "implicit"
                                && body.get("cache_control_injection_points").is_none()
                                && body["messages"][0]["content"][0]
                                    .get("cache_control")
                                    .is_none()
                        })
                });
            then.status(200).json_body(reply("openai.gpt-6.1-sol"));
        })
        .await;
    let (proxy, _) = common::start_oxllm(config).await.unwrap();
    for (path, body) in [
        (
            "/v1/chat/completions",
            json!({"model":"cacheable", "prompt_cache_options":null,
            "messages":[{"role":"system","content":"stable"},{"role":"user","content":"hello"}]}),
        ),
        (
            "/v1/messages",
            json!({"model":"cacheable", "max_tokens":32, "system":"stable", "prompt_cache_options":null,
            "messages":[{"role":"user","content":"hello"}]}),
        ),
        (
            "/v1/responses",
            json!({"model":"cacheable", "instructions":"stable", "input":"hello", "prompt_cache_options":null}),
        ),
    ] {
        let response = common::http_client()
            .post(proxy.url(path))
            .json(&body)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 200, "{}", response.text().await.unwrap());
        assert!(response.text().await.unwrap().contains("ok"));
    }
    wire.assert_hits_async(3).await;
    proxy.shutdown();
}

#[tokio::test]
async fn bedrock_converse_and_invoke_stream_header_wait_honor_request_timeouts() {
    for (model, operation) in [
        ("anthropic.claude-haiku-5-5", "invoke-with-response-stream"),
        ("amazon.nova-pro-v1:0", "converse-stream"),
    ] {
        let upstream = MockServer::start_async().await;
        let config = parse_config_str(&format!(
            r#"
model_list:
  - model_name: slow
    oxllm_params:
      model: bedrock/{model}
      api_base: {}
      aws_access_key_id: synthetic-access-key
      aws_secret_access_key: synthetic-secret-key
      aws_region_name: us-east-1
      timeout: 5
      stream_timeout: 5
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
                    .path(format!("/model/{model}/{operation}"))
                    .matches(|request| {
                        request
                            .body
                            .as_ref()
                            .and_then(|body| serde_json::from_slice::<Value>(body).ok())
                            .is_some_and(|body| body.get("timeout").is_none())
                    });
                then.status(200)
                    .delay(std::time::Duration::from_millis(300))
                    .body("");
            })
            .await;
        let (proxy, state) = common::start_oxllm(config).await.unwrap();
        let response = common::http_client()
            .post(proxy.url("/v1/chat/completions"))
            .json(&json!({
                "model":"slow", "stream":true, "timeout":0.025, "max_tokens":32,
                "messages":[{"role":"user","content":"hello"}]
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(
            response.status(),
            504,
            "{model}: {}",
            response.text().await.unwrap()
        );
        wire.assert_hits_async(1).await;
        let current = state.load_full();
        let key = current.router.cooldown_key(&current.deployments[0]);
        assert_eq!(current.router.usage_store().in_flight(&key), 0);
        proxy.shutdown();
    }
}
