//! Native-only schema and strict-tool forwarding; no synthetic JSON tools.

mod common;

use std::sync::Arc;
use std::time::Duration;

use base64::Engine as _;
use httpmock::{Method::POST, MockServer};
use oxllm_server::hooks::CacheHook;
use oxllm_server::hooks::cache_hook::cache_key_for;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

#[derive(Clone, Copy, Debug)]
enum Route {
    Direct,
    Invoke,
    Converse,
    Vertex,
    Foundry,
}

const ROUTES: [Route; 5] = [
    Route::Direct,
    Route::Invoke,
    Route::Converse,
    Route::Vertex,
    Route::Foundry,
];
const APIS: [&str; 3] = ["/v1/chat/completions", "/v1/responses", "/v1/messages"];

fn schema() -> Value {
    json!({"type":"object", "properties":{
        "entries":{"type":"array", "items":{"type":"string"}},
        "strict":{"type":"boolean"}
    }, "required":["entries"], "additionalProperties":false})
}

fn value() -> Value {
    json!({"entries":["one"], "strict":false})
}

fn tool() -> Value {
    json!({"type":"function", "function":{"name":"extract_record", "description":"Extract record", "strict":true, "parameters":schema()}})
}

fn config(upstream: &MockServer, route: Route) -> oxllm_server::ProxyConfig {
    let mut config = common::proxy_config(&upstream.base_url());
    config.model_list[0].model_name = "model".into();
    config.router_settings.routing.num_retries = Some(0);
    let params = &mut config.model_list[0].oxllm_params;
    match route {
        Route::Direct => {
            params.model = "anthropic/claude-haiku-5-5".into();
            params.api_base = Some(upstream.url("/direct"));
        }
        Route::Foundry => {
            params.model = "azure_ai/claude-haiku-5-5".into();
            params.api_base = Some(upstream.url("/foundry/anthropic/v1"));
        }
        Route::Vertex => {
            params.model = "vertex_ai/claude-haiku-5-5".into();
            params.api_base = Some(upstream.url("/vertex"));
            params.api_key = Some("ya29.synthetic-local-token".into());
            params
                .extra
                .insert("vertex_project".into(), json!("local-project"));
            params
                .extra
                .insert("vertex_location".into(), json!("us-central1"));
        }
        Route::Invoke | Route::Converse => {
            params.model = "bedrock/anthropic.claude-haiku-5-5".into();
            params.api_base = Some(upstream.url("/bedrock"));
            params.api_key = None;
            params.aws_access_key_id = Some("AKIDEXAMPLE".into());
            params.aws_secret_access_key = Some("synthetic-local-signing-only".into());
            params
                .extra
                .insert("aws_region_name".into(), json!("us-east-1"));
            params.extra.insert(
                "use_converse_api".into(),
                json!(matches!(route, Route::Converse)),
            );
        }
    }
    config
}

fn path(route: Route) -> &'static str {
    match route {
        Route::Direct => "/direct/messages",
        Route::Foundry => "/foundry/anthropic/v1/messages",
        Route::Vertex => {
            "/vertex/v1/projects/local-project/locations/us-central1/publishers/anthropic/models/claude-haiku-5-5:rawPredict"
        }
        Route::Invoke => "/bedrock/model/anthropic.claude-haiku-5-5/invoke",
        Route::Converse => "/bedrock/model/anthropic.claude-haiku-5-5/converse",
    }
}

fn request(api: &str, strict_tool: bool) -> Value {
    let mut request = match api {
        "/v1/responses" => {
            json!({"model":"model", "input":"Extract record", "max_output_tokens":128})
        }
        _ => {
            json!({"model":"model", "messages":[{"role":"user", "content":"Extract record"}], "max_tokens":128})
        }
    };
    if strict_tool {
        match api {
            "/v1/responses" => {
                request["tools"] = json!([{"type":"function", "name":"extract_record", "description":"Extract record", "parameters":schema(), "strict":true}]);
                request["tool_choice"] = json!({"type":"function", "name":"extract_record"});
            }
            "/v1/messages" => {
                request["tools"] = json!([{"name":"extract_record", "description":"Extract record", "input_schema":schema(), "strict":true}]);
                request["tool_choice"] = json!({"type":"tool", "name":"extract_record"});
            }
            _ => {
                request["tools"] = json!([tool()]);
                request["tool_choice"] =
                    json!({"type":"function", "function":{"name":"extract_record"}});
            }
        }
    } else {
        match api {
            "/v1/responses" => {
                request["text"] = json!({"format":{"type":"json_schema", "name":"record", "strict":true, "schema":schema()}})
            }
            "/v1/messages" => {
                request["output_config"] =
                    json!({"format":{"type":"json_schema", "schema":schema()}})
            }
            _ => {
                request["response_format"] = json!({"type":"json_schema", "json_schema":{"name":"record", "strict":true, "schema":schema()}})
            }
        }
    }
    request
}

fn wire_body(route: Route, api: &str, strict_tool: bool, schema: Value) -> Value {
    let mut body = if matches!(route, Route::Converse) {
        json!({"messages":[{"role":"user", "content":[{"text":"Extract record"}]}], "inferenceConfig":{"maxTokens":128}})
    } else {
        let mut body =
            json!({"messages":[{"role":"user", "content":"Extract record"}], "max_tokens":128});
        match route {
            Route::Direct | Route::Foundry => body["model"] = json!("claude-haiku-5-5"),
            Route::Invoke => body["anthropic_version"] = json!("bedrock-2023-05-31"),
            Route::Vertex => body["anthropic_version"] = json!("vertex-2023-10-16"),
            Route::Converse => unreachable!(),
        }
        body
    };
    if strict_tool {
        if matches!(route, Route::Converse) {
            body["toolConfig"] = json!({"tools":[{"toolSpec":{"name":"extract_record", "description":"Extract record", "strict":true, "inputSchema":{"json":schema}}}], "toolChoice":{"tool":{"name":"extract_record"}}});
        } else {
            body["tools"] = json!([{"name":"extract_record", "description":"Extract record", "strict":true, "input_schema":schema}]);
            body["tool_choice"] = json!({"type":"tool", "name":"extract_record"});
        }
    } else if matches!(route, Route::Converse) {
        let name = if api == "/v1/messages" {
            "structured_response"
        } else {
            "record"
        };
        body["outputConfig"] = json!({"textFormat":{"type":"json_schema", "structure":{"jsonSchema":{"name":name, "schema":schema.to_string()}}}});
    } else {
        body["output_config"] = json!({"format":{"type":"json_schema", "schema":schema}});
    }
    body
}

fn reply(route: Route, strict_tool: bool) -> Value {
    if matches!(route, Route::Converse) {
        let content = if strict_tool {
            json!({"toolUse":{"toolUseId":"call-native", "name":"extract_record", "input":value()}})
        } else {
            json!({"text":value().to_string()})
        };
        json!({"output":{"message":{"role":"assistant", "content":[content]}}, "stopReason":if strict_tool {"tool_use"} else {"end_turn"}, "usage":{"inputTokens":5,"outputTokens":4,"totalTokens":9}})
    } else {
        let content = if strict_tool {
            json!({"type":"tool_use", "id":"call-native", "name":"extract_record", "input":value()})
        } else {
            json!({"type":"text", "text":value().to_string()})
        };
        json!({"id":"msg-native", "type":"message", "role":"assistant", "model":"claude-haiku-5-5", "content":[content], "stop_reason":if strict_tool {"tool_use"} else {"end_turn"}, "usage":{"input_tokens":5,"output_tokens":4}})
    }
}

fn correlate(response: &reqwest::Response) {
    assert_eq!(
        response.headers()["x-oxllm-request-id"],
        response.headers()["x-request-id"]
    );
    assert_eq!(
        response.headers()["x-litellm-call-id"],
        response.headers()["x-request-id"]
    );
}

fn assert_usage_and_completion(api: &str, body: &Value, strict_tool: bool) {
    match api {
        "/v1/responses" => {
            assert_eq!(body["status"], "completed");
            assert_eq!(body["usage"]["input_tokens"], 5);
            assert_eq!(body["usage"]["output_tokens"], 4);
            assert_eq!(body["usage"]["total_tokens"], 9);
        }
        "/v1/messages" => {
            assert_eq!(
                body["stop_reason"],
                if strict_tool { "tool_use" } else { "end_turn" }
            );
            assert_eq!(body["usage"]["input_tokens"], 5);
            assert_eq!(body["usage"]["output_tokens"], 4);
        }
        _ => {
            assert_eq!(
                body["choices"][0]["finish_reason"],
                if strict_tool { "tool_calls" } else { "stop" }
            );
            assert_eq!(body["usage"]["prompt_tokens"], 5);
            assert_eq!(body["usage"]["completion_tokens"], 4);
            assert_eq!(body["usage"]["total_tokens"], 9);
        }
    }
}

#[tokio::test]
async fn native_json_output_is_not_replaced_with_tools_on_any_bridge_or_claude_route() {
    for route in ROUTES {
        for api in APIS {
            let upstream = MockServer::start_async().await;
            let wire = upstream
                .mock_async(|when, then| {
                    when.method(POST).path(path(route)).json_body(wire_body(
                        route,
                        api,
                        false,
                        schema(),
                    ));
                    then.status(200).json_body(reply(route, false));
                })
                .await;
            let (handle, _) = common::start_oxllm(config(&upstream, route)).await.unwrap();
            let response = common::http_client()
                .post(handle.url(api))
                .json(&request(api, false))
                .send()
                .await
                .unwrap();
            let status = response.status();
            correlate(&response);
            let body: Value = response.json().await.unwrap();
            assert_eq!(status, 200, "{route:?} {api}: {body}");
            assert_usage_and_completion(api, &body, false);
            let text = match api {
                "/v1/responses" => {
                    assert!(
                        body["output"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .all(|item| item["type"] != "function_call")
                    );
                    &body["output"][0]["content"][0]["text"]
                }
                "/v1/messages" => {
                    assert_eq!(body["content"][0]["type"], "text");
                    &body["content"][0]["text"]
                }
                _ => {
                    assert!(body["choices"][0]["message"].get("tool_calls").is_none());
                    &body["choices"][0]["message"]["content"]
                }
            };
            assert_eq!(text.as_str().unwrap(), value().to_string());
            wire.assert_hits_async(1).await;
            handle.shutdown();
        }
    }
}

#[tokio::test]
async fn explicitly_requested_strict_tools_stay_native_tools_on_every_bridge_and_route() {
    for route in ROUTES {
        for api in APIS {
            let upstream = MockServer::start_async().await;
            let wire = upstream
                .mock_async(|when, then| {
                    when.method(POST).path(path(route)).json_body(wire_body(
                        route,
                        api,
                        true,
                        schema(),
                    ));
                    then.status(200).json_body(reply(route, true));
                })
                .await;
            let (handle, _) = common::start_oxllm(config(&upstream, route)).await.unwrap();
            let response = common::http_client()
                .post(handle.url(api))
                .json(&request(api, true))
                .send()
                .await
                .unwrap();
            let status = response.status();
            correlate(&response);
            let body: Value = response.json().await.unwrap();
            assert_eq!(status, 200, "{route:?} {api}: {body}");
            assert_usage_and_completion(api, &body, true);
            let arguments = match api {
                "/v1/responses" => {
                    assert_eq!(body["output"][0]["type"], "function_call");
                    assert_eq!(body["output"][0]["name"], "extract_record");
                    serde_json::from_str(body["output"][0]["arguments"].as_str().unwrap()).unwrap()
                }
                "/v1/messages" => {
                    assert_eq!(body["content"][0]["type"], "tool_use");
                    assert_eq!(body["content"][0]["name"], "extract_record");
                    body["content"][0]["input"].clone()
                }
                _ => {
                    assert_eq!(body["choices"][0]["finish_reason"], "tool_calls");
                    let call = &body["choices"][0]["message"]["tool_calls"][0];
                    assert_eq!(call["function"]["name"], "extract_record");
                    serde_json::from_str(call["function"]["arguments"].as_str().unwrap()).unwrap()
                }
            };
            assert_eq!(arguments, value());
            wire.assert_hits_async(1).await;
            handle.shutdown();
        }
    }
}

#[tokio::test]
async fn provider_schema_and_model_errors_remain_errors_without_a_downgraded_second_call() {
    let unsupported = json!({"type":"object", "additionalProperties":true, "properties":{"entries":{"type":"array", "minItems":3, "uniqueItems":true, "items":{"type":"string", "minLength":5}}}});
    for route in ROUTES {
        for api in APIS {
            for strict_tool in [false, true] {
                let upstream = MockServer::start_async().await;
                let wire=upstream.mock_async(|when, then| {
            when.method(POST).path(path(route)).json_body(wire_body(route, api, strict_tool, unsupported.clone()));
            then.status(400).header("x-ratelimit-limit-requests", "50")
                .json_body(json!({"error":{"type":"invalid_request_error", "message":"native structured output or schema is unsupported by this model"}}));
        }).await;
                let mut config = config(&upstream, route);
                config.router_settings.routing.num_retries = Some(2);
                let (handle, _) = common::start_oxllm(config).await.unwrap();
                let mut payload = request(api, strict_tool);
                if strict_tool {
                    match api {
                        "/v1/responses" => payload["tools"][0]["parameters"] = unsupported.clone(),
                        "/v1/messages" => payload["tools"][0]["input_schema"] = unsupported.clone(),
                        _ => payload["tools"][0]["function"]["parameters"] = unsupported.clone(),
                    }
                } else {
                    match api {
                        "/v1/responses" => {
                            payload["text"]["format"]["schema"] = unsupported.clone()
                        }
                        "/v1/messages" => {
                            payload["output_config"]["format"]["schema"] = unsupported.clone()
                        }
                        _ => {
                            payload["response_format"]["json_schema"]["schema"] =
                                unsupported.clone()
                        }
                    }
                }
                let response = common::http_client()
                    .post(handle.url(api))
                    .json(&payload)
                    .send()
                    .await
                    .unwrap();
                assert_eq!(response.status(), 400, "{route:?}");
                correlate(&response);
                assert_eq!(response.headers()["x-ratelimit-limit-requests"], "50");
                let body: Value = response.json().await.unwrap();
                assert_eq!(
                    body["error"]["message"],
                    "native structured output or schema is unsupported by this model"
                );
                assert!(body.get("choices").is_none());
                wire.assert_hits_async(1).await;
                handle.shutdown();
            }
        }
    }
}

#[tokio::test]
async fn schema_less_json_mode_is_a_local_unsupported_error_not_tool_emulation() {
    for route in ROUTES {
        for api in APIS {
            let upstream = MockServer::start_async().await;
            let wire = upstream
                .mock_async(|when, then| {
                    when.method(POST);
                    then.status(200).json_body(reply(route, false));
                })
                .await;
            let (handle, _) = common::start_oxllm(config(&upstream, route)).await.unwrap();
            let mut payload = request(api, false);
            if api == "/v1/responses" {
                payload["text"] = json!({"format":{"type":"json_object"}});
            } else if api == APIS[2] {
                payload["output_config"] = json!({"format":{"type":"json_object"}});
            } else {
                payload["response_format"] = json!({"type":"json_object"});
            }
            let response = common::http_client()
                .post(handle.url(api))
                .json(&payload)
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), 400);
            assert!(
                response.json::<Value>().await.unwrap()["error"]["message"]
                    .as_str()
                    .unwrap()
                    .contains("no native representation")
            );
            wire.assert_hits_async(0).await;
            handle.shutdown();
        }
    }
}

#[tokio::test]
async fn configuration_cannot_silently_strip_requested_schemas_or_strict_tools() {
    for route in ROUTES {
        for strict_tool in [false, true] {
            for drop_source in [false, true] {
                let upstream = MockServer::start_async().await;
                let wire = upstream
                    .mock_async(|when, then| {
                        when.method(POST);
                        then.status(200).json_body(reply(route, strict_tool));
                    })
                    .await;
                let mut config = config(&upstream, route);
                let params = &mut config.model_list[0].oxllm_params;
                if drop_source {
                    params.additional_drop_params = Some(vec![
                        if strict_tool {
                            "tools"
                        } else {
                            "response_format"
                        }
                        .into(),
                    ]);
                } else {
                    let field = if strict_tool {
                        if matches!(route, Route::Converse) {
                            "toolConfig"
                        } else {
                            "tools"
                        }
                    } else if matches!(route, Route::Converse) {
                        "outputConfig"
                    } else {
                        "output_config"
                    };
                    params.extra_body =
                        Some(serde_json::Map::from_iter([(field.into(), Value::Null)]));
                }
                let (handle, _) = common::start_oxllm(config).await.unwrap();
                let response = common::http_client()
                    .post(handle.url(APIS[0]))
                    .json(&request(APIS[0], strict_tool))
                    .send()
                    .await
                    .unwrap();
                assert_eq!(
                    response.status(),
                    400,
                    "{route:?} strict={strict_tool} drop={drop_source}"
                );
                assert!(
                    response.json::<Value>().await.unwrap()["error"]["message"]
                        .as_str()
                        .unwrap()
                        .contains("deployment parameter policy")
                );
                wire.assert_hits_async(0).await;
                handle.shutdown();
            }
        }
    }
}

#[tokio::test]
async fn ignored_strict_or_emulated_output_cache_entries_cannot_bypass_the_native_policy() {
    let upstream = MockServer::start_async().await;
    let wire = upstream
        .mock_async(|when, then| {
            when.method(POST);
            then.status(200).json_body(reply(Route::Direct, false));
        })
        .await;
    let (handle, state) = common::start_oxllm(config(&upstream, Route::Direct))
        .await
        .unwrap();
    let mut next = (*state.load_full()).clone();
    next.hooks = vec![Arc::new(CacheHook::new(
        next.cache.clone(),
        Duration::from_secs(60),
    ))];
    state.store(Arc::new(next));
    let payload = json!({"model":"model", "messages":[{"role":"user", "content":"Extract record"}], "max_tokens":128, "response_format":{"type":"json_object"}, "cache":{"use_cache":true}});
    let principal = oxllm_server::auth::AuthUser::open().key_hash_hex;
    let mut legacy = payload.clone();
    legacy.as_object_mut().unwrap().remove("cache");
    fn canonicalize(value: &mut Value) {
        match value {
            Value::Object(object) => {
                object.sort_keys();
                for value in object.values_mut() {
                    canonicalize(value);
                }
            }
            Value::Array(items) => {
                for item in items {
                    canonicalize(item);
                }
            }
            _ => {}
        }
    }
    canonicalize(&mut legacy);
    let mut hasher = Sha256::new();
    hasher.update(principal.as_bytes());
    hasher.update(b"\x00");
    hasher.update(serde_json::to_vec(&legacy).unwrap());
    let legacy_key = format!("resp:{:x}", hasher.finalize());
    assert_ne!(legacy_key, cache_key_for(&payload, &principal));
    state.load_full().cache.set(&legacy_key, json!({"id":"old-emulation", "model":"model", "choices":[{"index":0,"message":{"role":"assistant", "content":"{}"},"finish_reason":"stop"}]}).to_string(), Some(Duration::from_secs(60))).await;
    let response = common::http_client()
        .post(handle.url(APIS[0]))
        .json(&payload)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 400);
    assert!(!response.headers().contains_key("x-oxllm-cache-hit"));
    assert!(
        state.load_full().cache.get(&legacy_key).await.is_some(),
        "legacy data is isolated, not deleted"
    );
    wire.assert_hits_async(0).await;
    handle.shutdown();
}

fn native_stream(tool_name: Option<&str>) -> String {
    let mut events = vec![
        json!({"type":"message_start", "message":{"id":"msg-native-stream", "type":"message", "role":"assistant", "model":"claude-haiku-5-5", "content":[], "usage":{"input_tokens":5,"output_tokens":0}}}),
    ];
    events.push(
        json!({"type":"content_block_start", "index":0, "content_block":match tool_name {
            Some(name) => json!({"type":"tool_use", "id":"call-native", "name":name, "input":{}}),
            None => json!({"type":"text", "text":""}),
        }}),
    );
    let text = value().to_string();
    let (first, second) = text.split_at(text.len() / 2);
    for text in [first, second] {
        events.push(
            json!({"type":"content_block_delta", "index":0, "delta":if tool_name.is_some() {
            json!({"type":"input_json_delta", "partial_json":text})
        } else {json!({"type":"text_delta", "text":text})}}),
        );
    }
    events.extend([
        json!({"type":"content_block_stop", "index":0}),
        json!({"type":"message_delta", "delta":{"stop_reason":if tool_name.is_some() {"tool_use"} else {"end_turn"}}, "usage":{"output_tokens":4}}),
        json!({"type":"message_stop"}),
    ]);
    events
        .into_iter()
        .map(|event| {
            format!(
                "event: {}\ndata: {event}\n\n",
                event["type"].as_str().unwrap()
            )
        })
        .collect()
}

fn aws_event_frame(event: &str, payload: &Value) -> Vec<u8> {
    fn crc32(bytes: &[u8]) -> u32 {
        let mut crc = u32::MAX;
        for byte in bytes {
            crc ^= u32::from(*byte);
            for _ in 0..8 {
                crc = (crc >> 1) ^ if crc & 1 == 1 { 0xedb8_8320 } else { 0 };
            }
        }
        !crc
    }
    let mut headers = Vec::new();
    for (name, value) in [
        (":message-type", "event"),
        (":event-type", event),
        (":content-type", "application/json"),
    ] {
        headers.push(u8::try_from(name.len()).unwrap());
        headers.extend_from_slice(name.as_bytes());
        headers.push(7); // AWS event-stream string header.
        headers.extend_from_slice(&u16::try_from(value.len()).unwrap().to_be_bytes());
        headers.extend_from_slice(value.as_bytes());
    }
    let payload = serde_json::to_vec(payload).unwrap();
    let mut frame = Vec::new();
    frame.extend_from_slice(
        &u32::try_from(16 + headers.len() + payload.len())
            .unwrap()
            .to_be_bytes(),
    );
    frame.extend_from_slice(&u32::try_from(headers.len()).unwrap().to_be_bytes());
    frame.extend_from_slice(&crc32(&frame).to_be_bytes());
    frame.extend(headers);
    frame.extend(payload);
    frame.extend_from_slice(&crc32(&frame).to_be_bytes());
    frame
}

fn stream_reply(route: Route, tool_name: Option<&str>) -> (&'static str, Vec<u8>) {
    if matches!(route, Route::Invoke) {
        let frames = native_stream(tool_name)
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .flat_map(|json| {
                aws_event_frame(
                    "chunk",
                    &json!({"bytes":base64::engine::general_purpose::STANDARD.encode(json)}),
                )
            })
            .collect();
        return ("application/vnd.amazon.eventstream", frames);
    }
    if matches!(route, Route::Converse) {
        let mut events = vec![
            ("messageStart", json!({"role":"assistant"})),
            (
                "contentBlockStart",
                json!({"contentBlockIndex":0,"start":match tool_name {Some(name)=>json!({"toolUse":{"toolUseId":"call-native","name":name}}),None=>json!({})}}),
            ),
        ];
        let text = value().to_string();
        let (first, second) = text.split_at(text.len() / 2);
        for fragment in [first, second] {
            events.push(("contentBlockDelta",json!({"contentBlockIndex":0,"delta":if tool_name.is_some() {json!({"toolUse":{"input":fragment}})} else {json!({"text":fragment})}})));
        }
        events.extend([
            ("contentBlockStop", json!({"contentBlockIndex":0})),
            (
                "messageStop",
                json!({"stopReason":if tool_name.is_some() {"tool_use"} else {"end_turn"}}),
            ),
            (
                "metadata",
                json!({"usage":{"inputTokens":5,"outputTokens":4,"totalTokens":9}}),
            ),
        ]);
        return (
            "application/vnd.amazon.eventstream",
            events
                .into_iter()
                .flat_map(|(event, payload)| aws_event_frame(event, &payload))
                .collect(),
        );
    }
    ("text/event-stream", native_stream(tool_name).into_bytes())
}

fn streaming_path(route: Route) -> String {
    match route {
        Route::Invoke => path(route).replace("/invoke", "/invoke-with-response-stream"),
        Route::Converse => path(route).replace("/converse", "/converse-stream"),
        Route::Vertex => path(route).replace(":rawPredict", ":streamRawPredict"),
        _ => path(route).into(),
    }
}

#[tokio::test]
async fn streaming_preserves_native_json_text_and_explicit_strict_tool_calls() {
    for route in ROUTES {
        for api in APIS {
            for name in [None, Some("extract_record"), Some("json_tool_call")] {
                let upstream = MockServer::start_async().await;
                let mut expected = wire_body(route, api, name.is_some(), schema());
                if !matches!(route, Route::Invoke | Route::Converse) {
                    expected["stream"] = json!(true);
                }
                let mut payload = request(api, name.is_some());
                payload["stream"] = json!(true);
                if api == APIS[0] {
                    payload["stream_options"] = json!({"include_usage":true});
                }
                if let Some(name) = name {
                    if matches!(route, Route::Converse) {
                        expected["toolConfig"]["tools"][0]["toolSpec"]["name"] = json!(name);
                        expected["toolConfig"]["toolChoice"]["tool"]["name"] = json!(name);
                    } else {
                        expected["tools"][0]["name"] = json!(name);
                        expected["tool_choice"]["name"] = json!(name);
                    }
                    match api {
                        "/v1/responses" | "/v1/messages" => {
                            payload["tools"][0]["name"] = json!(name);
                            payload["tool_choice"]["name"] = json!(name);
                        }
                        _ => {
                            payload["tools"][0]["function"]["name"] = json!(name);
                            payload["tool_choice"]["function"]["name"] = json!(name);
                        }
                    }
                }
                let (content_type, body) = stream_reply(route, name);
                let wire = upstream
                    .mock_async(|when, then| {
                        when.method(POST)
                            .path(streaming_path(route))
                            .json_body(expected);
                        then.status(200)
                            .header("content-type", content_type)
                            .body(body);
                    })
                    .await;
                let mut deployment = config(&upstream, route);
                if api == APIS[0] && name.is_none() {
                    deployment.model_list[0].model_name = "*".into();
                    let (pattern, model) = match route {
                        Route::Direct => ("anthropic/*", "anthropic/claude-haiku-5-5"),
                        Route::Invoke | Route::Converse => {
                            ("bedrock/*", "bedrock/anthropic.claude-haiku-5-5")
                        }
                        Route::Vertex => ("vertex_ai/*", "vertex_ai/claude-haiku-5-5"),
                        Route::Foundry => ("azure_ai/*", "azure_ai/claude-haiku-5-5"),
                    };
                    deployment.model_list[0].oxllm_params.model = pattern.into();
                    payload["model"] = json!(model);
                }
                let (handle, _) = common::start_oxllm(deployment).await.unwrap();
                let response = common::http_client()
                    .post(handle.url(api))
                    .json(&payload)
                    .send()
                    .await
                    .unwrap();
                assert_eq!(
                    response.status(),
                    200,
                    "{api} {name:?}: {}",
                    response.text().await.unwrap()
                );
                correlate(&response);
                assert!(
                    response.headers()["content-type"]
                        .to_str()
                        .unwrap()
                        .starts_with("text/event-stream")
                );
                let text = response.text().await.unwrap();
                let events = text
                    .lines()
                    .filter_map(|line| line.strip_prefix("data:"))
                    .map(str::trim)
                    .filter(|value| *value != "[DONE]")
                    .map(|value| serde_json::from_str::<Value>(value).unwrap())
                    .collect::<Vec<_>>();
                assert!(!events.is_empty());
                let mut assembled = String::new();
                for event in &events {
                    let fragment = match (api, name) {
                        ("/v1/responses", None) => {
                            assert_ne!(event["item"]["type"], "function_call");
                            (event["type"] == "response.output_text.delta")
                                .then(|| event["delta"].as_str())
                                .flatten()
                        }
                        ("/v1/responses", Some(_)) => (event["type"]
                            == "response.function_call_arguments.delta")
                            .then(|| event["delta"].as_str())
                            .flatten(),
                        ("/v1/messages", None) => {
                            assert_ne!(event["content_block"]["type"], "tool_use");
                            event["delta"]["text"].as_str()
                        }
                        ("/v1/messages", Some(_)) => event["delta"]["partial_json"].as_str(),
                        (_, None) => {
                            assert!(event.pointer("/choices/0/delta/tool_calls").is_none());
                            event
                                .pointer("/choices/0/delta/content")
                                .and_then(Value::as_str)
                        }
                        (_, Some(_)) => event
                            .pointer("/choices/0/delta/tool_calls/0/function/arguments")
                            .and_then(Value::as_str),
                    };
                    if let Some(fragment) = fragment {
                        assembled.push_str(fragment);
                    }
                }
                assert_eq!(
                    serde_json::from_str::<Value>(&assembled).unwrap(),
                    value(),
                    "{api} {name:?}: {text}"
                );
                match api {
                    "/v1/responses" => {
                        let completed = events
                            .iter()
                            .filter(|event| event["type"] == "response.completed")
                            .collect::<Vec<_>>();
                        assert_eq!(completed.len(), 1);
                        assert_usage_and_completion(api, &completed[0]["response"], name.is_some());
                    }
                    "/v1/messages" => {
                        let start = events
                            .iter()
                            .find(|event| event["type"] == "message_start")
                            .unwrap();
                        // Converse reports input usage in its final metadata,
                        // unlike Anthropic's initial message_start usage.
                        assert_eq!(
                            start["message"]["usage"]["input_tokens"],
                            if matches!(route, Route::Converse) {
                                0
                            } else {
                                5
                            },
                            "{route:?}"
                        );
                        let stops = events
                            .iter()
                            .filter(|event| event["type"] == "message_delta")
                            .collect::<Vec<_>>();
                        assert_eq!(stops.len(), 1);
                        assert_eq!(
                            stops[0]["delta"]["stop_reason"],
                            if name.is_some() {
                                "tool_use"
                            } else {
                                "end_turn"
                            }
                        );
                        assert_eq!(stops[0]["usage"]["output_tokens"], 4);
                        assert_eq!(stops[0]["usage"]["input_tokens"], 5);
                        assert_eq!(
                            events
                                .iter()
                                .filter(|event| event["type"] == "message_stop")
                                .count(),
                            1
                        );
                    }
                    _ => {
                        let finish = events
                            .iter()
                            .find(|event| {
                                event
                                    .pointer("/choices/0/finish_reason")
                                    .is_some_and(|value| !value.is_null())
                            })
                            .unwrap();
                        assert_eq!(
                            finish["choices"][0]["finish_reason"],
                            if name.is_some() { "tool_calls" } else { "stop" }
                        );
                        let usage = events
                            .iter()
                            .rev()
                            .find_map(|event| event.get("usage"))
                            .unwrap();
                        assert_eq!(usage["prompt_tokens"], 5);
                        assert_eq!(usage["completion_tokens"], 4);
                    }
                }
                if let Some(name) = name {
                    assert!(events.iter().any(|event| {
                        match api {
                            "/v1/responses" => {
                                event["item"]["type"] == "function_call"
                                    && event["item"]["name"] == name
                            }
                            "/v1/messages" => {
                                event["content_block"]["type"] == "tool_use"
                                    && event["content_block"]["name"] == name
                            }
                            _ => {
                                event
                                    .pointer("/choices/0/delta/tool_calls/0/function/name")
                                    .and_then(Value::as_str)
                                    == Some(name)
                            }
                        }
                    }));
                }
                wire.assert_hits_async(1).await;
                handle.shutdown();
            }
        }
    }
}

#[tokio::test]
async fn absent_false_and_null_strictness_do_not_add_a_native_strict_flag() {
    for route in ROUTES {
        for strict in [None, Some(json!(false)), Some(Value::Null)] {
            let upstream = MockServer::start_async().await;
            let mut expected = wire_body(route, APIS[0], true, schema());
            if matches!(route, Route::Converse) {
                expected["toolConfig"]["tools"][0]["toolSpec"]
                    .as_object_mut()
                    .unwrap()
                    .remove("strict");
            } else {
                expected["tools"][0]
                    .as_object_mut()
                    .unwrap()
                    .remove("strict");
            }
            let wire = upstream
                .mock_async(|when, then| {
                    when.method(POST).path(path(route)).json_body(expected);
                    then.status(200).json_body(reply(route, true));
                })
                .await;
            let (handle, _) = common::start_oxllm(config(&upstream, route)).await.unwrap();
            let mut payload = request(APIS[0], true);
            if let Some(strict) = strict {
                payload["tools"][0]["function"]["strict"] = strict;
            } else {
                payload["tools"][0]["function"]
                    .as_object_mut()
                    .unwrap()
                    .remove("strict");
            }
            let response = common::http_client()
                .post(handle.url(APIS[0]))
                .json(&payload)
                .send()
                .await
                .unwrap();
            assert_eq!(
                response.status(),
                200,
                "{route:?}: {}",
                response.text().await.unwrap()
            );
            wire.assert_hits_async(1).await;
            handle.shutdown();
        }
    }
}

#[tokio::test]
async fn configured_fallbacks_keep_the_original_native_request_and_surface_unsupported_errors() {
    for strict_tool in [false, true] {
        let upstream = MockServer::start_async().await;
        let primary = upstream
            .mock_async(|when, then| {
                when.method(POST)
                    .path(path(Route::Direct))
                    .json_body(wire_body(Route::Direct, APIS[0], strict_tool, schema()));
                then.status(503)
                    .json_body(json!({"error":{"message":"temporarily unavailable"}}));
            })
            .await;
        let fallback=upstream.mock_async(|when,then| {
            when.method(POST).path(path(Route::Converse)).json_body(wire_body(Route::Converse,APIS[0],strict_tool,schema()));
            then.status(400).json_body(json!({"error":{"message":"structured outputs are unsupported for this model"}}));
        }).await;
        let mut config = config(&upstream, Route::Direct);
        let mut backup = self::config(&upstream, Route::Converse)
            .model_list
            .remove(0);
        backup.model_name = "backup".into();
        config.model_list.push(backup);
        config.router_settings.routing.fallbacks = Some(json!([{"model":["backup"]}]));
        let (handle, _) = common::start_oxllm(config).await.unwrap();
        let response = common::http_client()
            .post(handle.url(APIS[0]))
            .json(&request(APIS[0], strict_tool))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 400);
        assert_eq!(
            response.json::<Value>().await.unwrap()["error"]["message"],
            "structured outputs are unsupported for this model"
        );
        primary.assert_hits_async(1).await;
        fallback.assert_hits_async(1).await;
        handle.shutdown();
    }
}

#[tokio::test]
async fn native_refusals_and_truncated_json_are_not_repaired_or_retried() {
    for (stop, finish, text) in [
        ("max_tokens", "length", "{\"entries\":["),
        ("refusal", "content_filter", "I cannot comply"),
    ] {
        for api in APIS {
            let upstream = MockServer::start_async().await;
            let mut output = reply(Route::Direct, false);
            output["content"][0]["text"] = json!(text);
            output["stop_reason"] = json!(stop);
            let wire = upstream
                .mock_async(|when, then| {
                    when.method(POST)
                        .path(path(Route::Direct))
                        .json_body(wire_body(Route::Direct, api, false, schema()));
                    then.status(200).json_body(output);
                })
                .await;
            let mut config = config(&upstream, Route::Direct);
            config.router_settings.routing.num_retries = Some(2);
            let (handle, _) = common::start_oxllm(config).await.unwrap();
            let response = common::http_client()
                .post(handle.url(api))
                .json(&request(api, false))
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), 200);
            let output = response.json::<Value>().await.unwrap();
            match api {
                "/v1/responses" => {
                    assert_eq!(output["status"], "incomplete");
                    assert_eq!(output["output"][0]["content"][0]["text"], text);
                    assert_eq!(output["usage"]["input_tokens"], 5);
                    assert_eq!(output["usage"]["output_tokens"], 4);
                    if stop == "max_tokens" {
                        assert_eq!(output["incomplete_details"]["reason"], "max_output_tokens");
                    }
                }
                "/v1/messages" => {
                    assert_eq!(output["stop_reason"], stop);
                    assert_eq!(output["content"][0]["type"], "text");
                    assert_eq!(output["content"][0]["text"], text);
                    assert_eq!(output["usage"]["input_tokens"], 5);
                    assert_eq!(output["usage"]["output_tokens"], 4);
                }
                _ => {
                    assert_eq!(output["choices"][0]["message"]["content"], text);
                    assert_eq!(output["choices"][0]["finish_reason"], finish);
                    assert!(output["choices"][0]["message"].get("tool_calls").is_none());
                }
            }
            wire.assert_hits_async(1).await;
            handle.shutdown();
        }
    }
}

#[tokio::test]
async fn new_native_responses_are_cacheable_without_reusing_the_old_policy() {
    let upstream = MockServer::start_async().await;
    let wire = upstream
        .mock_async(|when, then| {
            when.method(POST)
                .path(path(Route::Direct))
                .json_body(wire_body(Route::Direct, APIS[0], false, schema()));
            then.status(200).json_body(reply(Route::Direct, false));
        })
        .await;
    let (handle, state) = common::start_oxllm(config(&upstream, Route::Direct))
        .await
        .unwrap();
    let mut next = (*state.load_full()).clone();
    next.hooks = vec![Arc::new(CacheHook::new(
        next.cache.clone(),
        Duration::from_secs(60),
    ))];
    state.store(Arc::new(next));
    let mut payload = request(APIS[0], false);
    payload["cache"] = json!({"use_cache":true});
    for cached in [false, true] {
        let response = common::http_client()
            .post(handle.url(APIS[0]))
            .json(&payload)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        correlate(&response);
        assert_eq!(response.headers().contains_key("x-oxllm-cache-hit"), cached);
        assert_eq!(
            response.json::<Value>().await.unwrap()["choices"][0]["message"]["content"],
            value().to_string()
        );
    }
    wire.assert_hits_async(1).await;
    handle.shutdown();
}

#[tokio::test]
async fn bad_strict_flags_are_request_errors_before_any_upstream_call() {
    let upstream = MockServer::start_async().await;
    let wire = upstream
        .mock_async(|when, then| {
            when.method(POST);
            then.status(200).json_body(reply(Route::Direct, true));
        })
        .await;
    let (handle, _) = common::start_oxllm(config(&upstream, Route::Direct))
        .await
        .unwrap();
    for function in [
        json!({"name":"extract_record", "strict":"true", "parameters":schema()}),
        json!({"name":"extract_record", "parameters":{"type":"object", "strict":true}}),
    ] {
        let mut payload = request(APIS[0], true);
        payload["tools"] = json!([{"type":"function", "function":function}]);
        let response = common::http_client()
            .post(handle.url(APIS[0]))
            .json(&payload)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 400);
    }
    for api in APIS {
        let mut payload = request(api, true);
        payload["tools"] =
            json!({"function":{"name":"extract_record","strict":true,"parameters":schema()}});
        let response = common::http_client()
            .post(handle.url(api))
            .json(&payload)
            .send()
            .await
            .unwrap();
        assert_eq!(
            response.status(),
            400,
            "malformed tools on {api} must not become plain text"
        );
    }
    wire.assert_hits_async(0).await;
    handle.shutdown();
}

#[tokio::test]
async fn preserving_one_output_feature_cannot_hide_another_feature_being_dropped() {
    for (model, native_format) in [("xai/grok-4", false), ("groq/native-test-model", true)] {
        let upstream = MockServer::start_async().await;
        let wire=upstream.mock_async(|when,then| {
            when.method(POST);
            then.status(200).json_body(json!({"id":"unconstrained","model":"model","choices":[{"index":0,"message":{"role":"assistant","content":"{}"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":4,"total_tokens":9}}));
        }).await;
        let mut config = config(&upstream, Route::Direct);
        config.model_list[0].oxllm_params.model = model.into();
        config.model_list[0].oxllm_params.api_base = Some(upstream.base_url());
        let (handle, _) = common::start_oxllm(config).await.unwrap();
        let mut payload = request(APIS[0], true);
        if native_format {
            payload["output_config"] =
                json!({"format":{"type":"json_schema","schema":schema(),"native_option":true}});
        } else {
            payload["response_format"] = request(APIS[0], false)["response_format"].clone();
        }
        let response = common::http_client()
            .post(handle.url(APIS[0]))
            .json(&payload)
            .send()
            .await
            .unwrap();
        assert_eq!(
            response.status(),
            400,
            "{model}: {}",
            response.text().await.unwrap()
        );
        wire.assert_hits_async(0).await;
        handle.shutdown();
    }
}

#[tokio::test]
async fn legacy_schema_formats_survive_drop_params_on_native_routes_and_messages() {
    for route in ROUTES {
        for api in [APIS[0], APIS[2]] {
            let upstream = MockServer::start_async().await;
            let mut expected = wire_body(route, api, false, schema());
            if matches!(route, Route::Converse) && api == APIS[0] {
                expected["outputConfig"]["textFormat"]["structure"]["jsonSchema"]
                    .as_object_mut()
                    .unwrap()
                    .remove("name");
            }
            let wire = upstream
                .mock_async(|when, then| {
                    when.method(POST).path(path(route)).json_body(expected);
                    then.status(200).json_body(reply(route, false));
                })
                .await;
            let mut config = config(&upstream, route);
            config.model_list[0].oxllm_params.drop_params = Some(true);
            let (handle, _) = common::start_oxllm(config).await.unwrap();
            let mut payload = request(api, false);
            payload.as_object_mut().unwrap().remove("response_format");
            payload.as_object_mut().unwrap().remove("output_config");
            payload["output_format"] = json!({"type":"json_schema","schema":schema()});
            let response = common::http_client()
                .post(handle.url(api))
                .json(&payload)
                .send()
                .await
                .unwrap();
            let status = response.status();
            let body: Value = response.json().await.unwrap();
            assert_eq!(status, 200, "{route:?} {api}: {body}");
            assert_usage_and_completion(api, &body, false);
            wire.assert_hits_async(1).await;
            handle.shutdown();
        }
    }
}

#[tokio::test]
async fn drop_params_keeps_native_effort_alone_and_beside_schema_strict_tools_and_thinking() {
    for route in ROUTES {
        for shaped in [false, true] {
            let upstream = MockServer::start_async().await;
            let mut expected = wire_body(route, APIS[0], shaped, schema());
            let mut payload = request(APIS[0], shaped);
            payload["output_config"] = json!({"effort":"low","format":null});
            if shaped {
                payload["response_format"] = request(APIS[0], false)["response_format"].clone();
                payload["thinking"] = json!({"type":"enabled","budget_tokens":1024});
                payload["max_tokens"] = json!(2048);
            } else {
                payload.as_object_mut().unwrap().remove("response_format");
            }
            if matches!(route, Route::Converse) {
                if shaped {
                    expected["outputConfig"] =
                        wire_body(route, APIS[0], false, schema())["outputConfig"].clone();
                    expected["additionalModelRequestFields"] =
                        json!({"thinking":{"type":"enabled","budget_tokens":1024}});
                    expected["inferenceConfig"]["maxTokens"] = json!(2048);
                } else {
                    expected.as_object_mut().unwrap().remove("outputConfig");
                }
                expected["outputConfig"]["effort"] = json!("low");
            } else {
                expected["output_config"] = if shaped {
                    json!({"format":{"type":"json_schema","schema":schema()},"effort":"low"})
                } else {
                    json!({"effort":"low"})
                };
                if shaped {
                    expected["thinking"] = payload["thinking"].clone();
                    expected["max_tokens"] = json!(2048);
                }
            }
            let wire = upstream
                .mock_async(|when, then| {
                    when.method(POST).path(path(route)).json_body(expected);
                    then.status(200).json_body(reply(route, shaped));
                })
                .await;
            let mut config = config(&upstream, route);
            config.model_list[0].oxllm_params.drop_params = Some(true);
            let (handle, _) = common::start_oxllm(config).await.unwrap();
            let response = common::http_client()
                .post(handle.url(APIS[0]))
                .json(&payload)
                .send()
                .await
                .unwrap();
            let status = response.status();
            let body: Value = response.json().await.unwrap();
            assert_eq!(status, 200, "{route:?} shaped={shaped}: {body}");
            assert_usage_and_completion(APIS[0], &body, shaped);
            wire.assert_hits_async(1).await;
            handle.shutdown();
        }
    }
}

#[tokio::test]
async fn unmappable_native_messages_options_have_an_actionable_messages_error() {
    let upstream = MockServer::start_async().await;
    let wire = upstream
        .mock_async(|when, then| {
            when.method(POST);
            then.status(200).json_body(reply(Route::Direct, false));
        })
        .await;
    let mut config = config(&upstream, Route::Direct);
    config.model_list[0].oxllm_params.model = "groq/native-test-model".into();
    config.model_list[0].oxllm_params.api_base = Some(upstream.base_url());
    let (handle, _) = common::start_oxllm(config).await.unwrap();
    let mut payload = request(APIS[2], false);
    payload["output_config"]["format"]["native_option"] = json!(true);
    let response = common::http_client()
        .post(handle.url(APIS[2]))
        .json(&payload)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 400);
    correlate(&response);
    let body: Value = response.json().await.unwrap();
    assert_eq!(body["type"], "error");
    assert_eq!(body["error"]["type"], "invalid_request_error");
    let message = body["error"]["message"].as_str().unwrap();
    assert!(message.contains("/output_config/format"), "{body}");
    assert!(message.contains("native endpoint"), "{body}");
    wire.assert_hits_async(0).await;
    handle.shutdown();
}

#[tokio::test]
async fn configured_retries_resend_the_original_native_requirements_without_downgrading() {
    for route in [Route::Direct, Route::Converse] {
        for strict_tool in [false, true] {
            let upstream = MockServer::start_async().await;
            let wire = upstream
                .mock_async(|when, then| {
                    when.method(POST).path(path(route)).json_body(wire_body(
                        route,
                        APIS[0],
                        strict_tool,
                        schema(),
                    ));
                    then.status(503)
                        .json_body(json!({"error":{"message":"temporarily unavailable"}}));
                })
                .await;
            let mut config = config(&upstream, route);
            config.router_settings.routing.num_retries = Some(1);
            let (handle, _) = common::start_oxllm(config).await.unwrap();
            let response = common::http_client()
                .post(handle.url(APIS[0]))
                .json(&request(APIS[0], strict_tool))
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), 503, "{route:?} strict={strict_tool}");
            wire.assert_hits_async(2).await;
            handle.shutdown();
        }
    }
}

#[tokio::test]
async fn historical_json_object_with_schema_alias_still_has_a_native_schema_mapping() {
    for route in ROUTES {
        let upstream = MockServer::start_async().await;
        let wire = upstream
            .mock_async(|when, then| {
                when.method(POST).path(path(route)).json_body(wire_body(
                    route,
                    APIS[0],
                    false,
                    schema(),
                ));
                then.status(200).json_body(reply(route, false));
            })
            .await;
        let (handle, _) = common::start_oxllm(config(&upstream, route)).await.unwrap();
        let mut payload = request(APIS[0], false);
        payload["response_format"]["type"] = json!("json_object");
        let response = common::http_client()
            .post(handle.url(APIS[0]))
            .json(&payload)
            .send()
            .await
            .unwrap();
        let status = response.status();
        let body: Value = response.json().await.unwrap();
        assert_eq!(status, 200, "{route:?}: {body}");
        assert_usage_and_completion(APIS[0], &body, false);
        wire.assert_hits_async(1).await;
        handle.shutdown();
    }
}

#[tokio::test]
async fn wildcard_deployments_forward_native_schemas_for_the_requested_model() {
    for route in ROUTES {
        let upstream = MockServer::start_async().await;
        let wire = upstream
            .mock_async(|when, then| {
                when.method(POST).path(path(route)).json_body(wire_body(
                    route,
                    APIS[0],
                    false,
                    schema(),
                ));
                then.status(200).json_body(reply(route, false));
            })
            .await;
        let mut config = config(&upstream, route);
        config.model_list[0].model_name = "*".into();
        config.model_list[0].oxllm_params.model = match route {
            Route::Direct => "anthropic/*",
            Route::Invoke | Route::Converse => "bedrock/*",
            Route::Vertex => "vertex_ai/*",
            Route::Foundry => "azure_ai/*",
        }
        .into();
        let (handle, _) = common::start_oxllm(config).await.unwrap();
        let mut payload = request(APIS[0], false);
        payload["model"] = json!(match route {
            Route::Direct => "anthropic/claude-haiku-5-5",
            Route::Invoke | Route::Converse => "bedrock/anthropic.claude-haiku-5-5",
            Route::Vertex => "vertex_ai/claude-haiku-5-5",
            Route::Foundry => "azure_ai/claude-haiku-5-5",
        });
        let response = common::http_client()
            .post(handle.url(APIS[0]))
            .json(&payload)
            .send()
            .await
            .unwrap();
        let status = response.status();
        let body: Value = response.json().await.unwrap();
        assert_eq!(status, 200, "{route:?}: {body}");
        assert_usage_and_completion(APIS[0], &body, false);
        wire.assert_hits_async(1).await;
        handle.shutdown();
    }
}

#[tokio::test]
async fn strict_tool_choices_without_a_lossless_native_mapping_are_not_relaxed_to_auto() {
    for route in [Route::Direct, Route::Converse] {
        for api in APIS {
            let upstream = MockServer::start_async().await;
            let wire = upstream
                .mock_async(|when, then| {
                    when.method(POST);
                    then.status(200).json_body(reply(route, true));
                })
                .await;
            let (handle, _) = common::start_oxllm(config(&upstream, route)).await.unwrap();
            let mut payload = request(api, true);
            payload["tool_choice"] = json!({"type":"allowed_tools","mode":"required","tools":[{"type":"function","name":"extract_record"}]});
            let response = common::http_client()
                .post(handle.url(api))
                .json(&payload)
                .send()
                .await
                .unwrap();
            let status = response.status();
            let body: Value = response.json().await.unwrap();
            assert_eq!(status, 400, "{route:?} {api}: {body}");
            assert!(
                body["error"]["message"]
                    .as_str()
                    .unwrap()
                    .contains("tool choice"),
                "{body}"
            );
            wire.assert_hits_async(0).await;
            handle.shutdown();
        }
    }
}

#[tokio::test]
async fn hosted_root_strictness_survives_a_legacy_function_carrier() {
    for route in ROUTES {
        for api in APIS {
            let rejected = matches!(route, Route::Converse) || api == APIS[1];
            let upstream = MockServer::start_async().await;
            let mut expected = wire_body(route, api, true, schema());
            expected["tools"] = json!([{"type":"computer_20250124","name":"computer","strict":true,"display_width_px":1280,"display_height_px":720}]);
            expected["tool_choice"] = json!({"type":"tool","name":"computer"});
            let wire = upstream
                .mock_async(|when, then| {
                    if rejected {
                        when.method(POST);
                    } else {
                        when.method(POST).path(path(route)).json_body(expected);
                    }
                    then.status(200).json_body(reply(route, false));
                })
                .await;
            let (handle, _) = common::start_oxllm(config(&upstream, route)).await.unwrap();
            let mut payload = request(api, true);
            payload["tools"] = json!([{"type":"computer_20250124","strict":true,"function":{"name":"computer","parameters":{"display_width_px":1280,"display_height_px":720}}}]);
            payload["tool_choice"] = if api == APIS[2] {
                json!({"type":"tool","name":"computer"})
            } else {
                json!({"type":"function","function":{"name":"computer"}})
            };
            let response = common::http_client()
                .post(handle.url(api))
                .json(&payload)
                .send()
                .await
                .unwrap();
            let status = response.status();
            let body: Value = response.json().await.unwrap();
            assert_eq!(
                status,
                if rejected { 400 } else { 200 },
                "{route:?} {api}: {body}"
            );
            wire.assert_hits_async(usize::from(!rejected)).await;
            handle.shutdown();
        }
    }
}

#[tokio::test]
async fn strict_parallel_tool_constraints_survive_every_bridge_and_native_choice_shape() {
    for route in [Route::Direct, Route::Converse] {
        for api in APIS {
            let upstream = MockServer::start_async().await;
            let mut expected = wire_body(route, api, true, schema());
            let mut payload = request(api, true);
            if api == APIS[2] {
                payload["tool_choice"]["disable_parallel_tool_use"] = json!(true);
            } else {
                payload["parallel_tool_calls"] = json!(false);
            }
            if matches!(route, Route::Converse) {
                expected["toolConfig"]
                    .as_object_mut()
                    .unwrap()
                    .remove("toolChoice");
                expected["additionalModelRequestFields"] = json!({"tool_choice":{"type":"tool","name":"extract_record","disable_parallel_tool_use":true}});
            } else {
                expected["tool_choice"]["disable_parallel_tool_use"] = json!(true);
            }
            let wire = upstream
                .mock_async(|when, then| {
                    when.method(POST).path(path(route)).json_body(expected);
                    then.status(200).json_body(reply(route, true));
                })
                .await;
            let (handle, _) = common::start_oxllm(config(&upstream, route)).await.unwrap();
            let response = common::http_client()
                .post(handle.url(api))
                .json(&payload)
                .send()
                .await
                .unwrap();
            let status = response.status();
            let body: Value = response.json().await.unwrap();
            assert_eq!(status, 200, "{route:?} {api}: {body}");
            assert_usage_and_completion(api, &body, true);
            wire.assert_hits_async(1).await;
            handle.shutdown();
        }
    }
}

#[tokio::test]
async fn converse_snake_case_native_text_format_cannot_be_stripped_by_policy() {
    for dropped in [false, true] {
        let upstream = MockServer::start_async().await;
        let expected = wire_body(Route::Converse, APIS[0], false, schema());
        let wire = upstream
            .mock_async(|when, then| {
                if dropped {
                    when.method(POST).path(path(Route::Converse));
                } else {
                    when.method(POST)
                        .path(path(Route::Converse))
                        .json_body(expected);
                }
                then.status(200).json_body(reply(Route::Converse, false));
            })
            .await;
        let mut config = config(&upstream, Route::Converse);
        if dropped {
            config.model_list[0].oxllm_params.additional_drop_params =
                Some(vec!["output_config".into()]);
        }
        let (handle, _) = common::start_oxllm(config).await.unwrap();
        let mut payload = request(APIS[0], false);
        payload.as_object_mut().unwrap().remove("response_format");
        payload["output_config"] =
            wire_body(Route::Converse, APIS[0], false, schema())["outputConfig"].clone();
        let response = common::http_client()
            .post(handle.url(APIS[0]))
            .json(&payload)
            .send()
            .await
            .unwrap();
        let status = response.status();
        let body: Value = response.json().await.unwrap();
        if dropped {
            assert_eq!(status, 400, "{body}");
            assert!(
                body["error"]["message"]
                    .as_str()
                    .unwrap()
                    .contains("deployment parameter policy")
            );
            wire.assert_hits_async(0).await;
        } else {
            assert_eq!(status, 200, "{body}");
            assert_usage_and_completion(APIS[0], &body, false);
            wire.assert_hits_async(1).await;
        }
        handle.shutdown();
    }
}

#[tokio::test]
async fn converse_serial_strict_tool_choice_cannot_be_dropped_after_native_mapping() {
    for api in APIS {
        let upstream = MockServer::start_async().await;
        let wire = upstream
            .mock_async(|when, then| {
                when.method(POST);
                then.status(200).json_body(reply(Route::Converse, true));
            })
            .await;
        let mut config = config(&upstream, Route::Converse);
        config.model_list[0].oxllm_params.additional_drop_params =
            Some(vec!["additionalModelRequestFields".into()]);
        let (handle, _) = common::start_oxllm(config).await.unwrap();
        let mut payload = request(api, true);
        if api == APIS[2] {
            payload["tool_choice"]["disable_parallel_tool_use"] = json!(true);
        } else {
            payload["parallel_tool_calls"] = json!(false);
        }
        let response = common::http_client()
            .post(handle.url(api))
            .json(&payload)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 400);
        let body: Value = response.json().await.unwrap();
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("deployment parameter policy")
        );
        wire.assert_hits_async(0).await;
        handle.shutdown();
    }
}

#[tokio::test]
async fn native_serial_choices_from_model_specific_fields_are_preserved() {
    for alias in [
        "additionalModelRequestFields",
        "additional_model_request_fields",
    ] {
        for choice in [
            json!({"type":"tool","name":"extract_record","native_option":true}),
            json!({"type":"any"}),
        ] {
            let upstream = MockServer::start_async().await;
            let mut expected = wire_body(Route::Converse, APIS[0], true, schema());
            expected["toolConfig"]
                .as_object_mut()
                .unwrap()
                .remove("toolChoice");
            let mut native = choice.clone();
            native["disable_parallel_tool_use"] = json!(true);
            expected["additionalModelRequestFields"] = json!({"tool_choice":native});
            let wire = upstream
                .mock_async(|when, then| {
                    when.method(POST)
                        .path(path(Route::Converse))
                        .json_body(expected);
                    then.status(200).json_body(reply(Route::Converse, true));
                })
                .await;
            let (handle, _) = common::start_oxllm(config(&upstream, Route::Converse))
                .await
                .unwrap();
            let mut payload = request(APIS[0], true);
            payload.as_object_mut().unwrap().remove("tool_choice");
            payload["parallel_tool_calls"] = json!(false);
            payload[alias] = json!({"tool_choice":choice});
            let response = common::http_client()
                .post(handle.url(APIS[0]))
                .json(&payload)
                .send()
                .await
                .unwrap();
            let status = response.status();
            let body: Value = response.json().await.unwrap();
            assert_eq!(status, 200, "{alias} {choice}: {body}");
            assert_usage_and_completion(APIS[0], &body, true);
            wire.assert_hits_async(1).await;
            handle.shutdown();
        }
    }
}

#[tokio::test]
async fn native_serial_none_choice_already_disables_all_tool_calls() {
    for route in ROUTES {
        for api in [APIS[0], APIS[1]] {
            let upstream = MockServer::start_async().await;
            let mut expected = wire_body(route, api, true, schema());
            if matches!(route, Route::Converse) {
                expected.as_object_mut().unwrap().remove("toolConfig");
            } else {
                expected["tool_choice"] = json!({"type":"none"});
            }
            let wire = upstream
                .mock_async(|when, then| {
                    when.method(POST).path(path(route)).json_body(expected);
                    then.status(200).json_body(reply(route, false));
                })
                .await;
            let mut config = config(&upstream, route);
            config.model_list[0].oxllm_params.allowed_openai_params =
                Some(vec!["tools".into(), "tool_choice".into()]);
            let (handle, _) = common::start_oxllm(config).await.unwrap();
            let mut payload = request(api, true);
            payload["tool_choice"] = json!("none");
            payload["parallel_tool_calls"] = json!(false);
            let response = common::http_client()
                .post(handle.url(api))
                .json(&payload)
                .send()
                .await
                .unwrap();
            let status = response.status();
            let body: Value = response.json().await.unwrap();
            assert_eq!(status, 200, "{route:?} {api}: {body}");
            assert_usage_and_completion(api, &body, false);
            wire.assert_hits_async(1).await;
            handle.shutdown();
        }
    }
}

#[tokio::test]
async fn advisory_restricted_choices_fail_before_native_calls_on_every_bridge() {
    for route in [Route::Direct, Route::Converse] {
        for api in APIS {
            let upstream = MockServer::start_async().await;
            let wire = upstream
                .mock_async(|when, then| {
                    when.method(POST);
                    then.status(200).json_body(reply(route, false));
                })
                .await;
            let (handle, _) = common::start_oxllm(config(&upstream, route)).await.unwrap();
            let mut payload = request(api, true);
            let function = if api == APIS[0] {
                &mut payload["tools"][0]["function"]
            } else {
                &mut payload["tools"][0]
            };
            function.as_object_mut().unwrap().remove("strict");
            payload["tool_choice"] = json!({"type":"allowed_tools","mode":"required","tools":[{"type":"function","name":"extract_record"}]});
            let response = common::http_client()
                .post(handle.url(api))
                .json(&payload)
                .send()
                .await
                .unwrap();
            let status = response.status();
            let body: Value = response.json().await.unwrap();
            assert_eq!(status, 400, "{route:?} {api}: {body}");
            assert!(
                body["error"]["message"]
                    .as_str()
                    .unwrap()
                    .contains("tool choice")
            );
            wire.assert_hits_async(0).await;
            handle.shutdown();
        }
    }
}

#[tokio::test]
async fn openai_compatible_adapters_cannot_hide_conflicting_or_dropped_format_requirements() {
    let upstream = MockServer::start_async().await;
    let wire=upstream.mock_async(|when,then| {when.method(POST);then.status(200).json_body(json!({"id":"mock","choices":[{"index":0,"message":{"role":"assistant","content":"{}"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":4,"total_tokens":9}}));}).await;
    let mut config = config(&upstream, Route::Direct);
    config.model_list[0].oxllm_params.model = "groq/native-test-model".into();
    config.model_list[0].oxllm_params.api_base = Some(upstream.base_url());
    let (handle, _) = common::start_oxllm(config).await.unwrap();
    for format in [
        json!({"type":"json_schema","schema":{"type":"object","properties":{"other":{"type":"number"}}}}),
        json!({"type":"json_schema","schema":schema(),"native_option":true}),
    ] {
        let mut payload = request(APIS[0], false);
        payload["output_config"] = json!({"format":format});
        let response = common::http_client()
            .post(handle.url(APIS[0]))
            .json(&payload)
            .send()
            .await
            .unwrap();
        let status = response.status();
        let body: Value = response.json().await.unwrap();
        assert_eq!(status, 400, "{body}");
    }
    wire.assert_hits_async(0).await;
    handle.shutdown();
}

#[tokio::test]
async fn native_messages_policy_cannot_strip_and_cache_output_requirements() {
    for strict_tool in [false, true] {
        for override_body in [false, true] {
            let upstream = MockServer::start_async().await;
            let wire = upstream
                .mock_async(|when, then| {
                    when.method(POST);
                    then.status(200).json_body(reply(Route::Direct, false));
                })
                .await;
            let mut config = config(&upstream, Route::Direct);
            let field = if strict_tool {
                "tools"
            } else {
                "output_config"
            };
            if override_body {
                config.model_list[0].oxllm_params.extra_body =
                    Some(serde_json::Map::from_iter([(field.into(), Value::Null)]));
            } else {
                config.model_list[0].oxllm_params.additional_drop_params = Some(vec![field.into()]);
            }
            let (handle, state) = common::start_oxllm(config).await.unwrap();
            let mut next = (*state.load_full()).clone();
            next.hooks = vec![Arc::new(CacheHook::new(
                next.cache.clone(),
                Duration::from_secs(60),
            ))];
            state.store(Arc::new(next));
            let mut payload = request(APIS[2], strict_tool);
            payload["cache"] = json!({"use_cache":true});
            for _ in 0..2 {
                let response = common::http_client()
                    .post(handle.url("/anthropic/v1/messages"))
                    .json(&payload)
                    .send()
                    .await
                    .unwrap();
                let status = response.status();
                assert!(!response.headers().contains_key("x-oxllm-cache-hit"));
                let body: Value = response.json().await.unwrap();
                assert_eq!(status, 400, "{body}");
                assert!(
                    body["error"]["message"]
                        .as_str()
                        .unwrap()
                        .contains("deployment parameter policy")
                );
            }
            wire.assert_hits_async(0).await;
            handle.shutdown();
        }
    }
}

#[tokio::test]
async fn per_message_output_formats_fail_closed_when_no_native_mapping_exists() {
    for model in [
        "bedrock/anthropic.claude-haiku-5-5",
        "groq/native-test-model",
        "openai/native-test-model",
    ] {
        let upstream = MockServer::start_async().await;
        let wire=upstream.mock_async(|when,then| {when.method(POST);then.status(200).json_body(json!({"id":"mock","choices":[{"index":0,"message":{"role":"assistant","content":"{}"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":4,"total_tokens":9}}));}).await;
        let mut config = config(&upstream, Route::Converse);
        config.model_list[0].oxllm_params.model = model.into();
        if !model.starts_with("bedrock/") {
            config.model_list[0].oxllm_params.api_base = Some(upstream.base_url());
            config.model_list[0].oxllm_params.api_key = Some("local-test-key".into());
        }
        let (handle, _) = common::start_oxllm(config).await.unwrap();
        let mut payload = request(APIS[2], false);
        payload.as_object_mut().unwrap().remove("output_config");
        payload["messages"][0]["output_config"] =
            json!({"format":{"type":"json_schema","schema":schema()}});
        let response = common::http_client()
            .post(handle.url(APIS[2]))
            .json(&payload)
            .send()
            .await
            .unwrap();
        let status = response.status();
        let body: Value = response.json().await.unwrap();
        assert_eq!(status, 400, "{model}: {body}");
        assert_eq!(body["type"], "error");
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("message")
        );
        wire.assert_hits_async(0).await;
        handle.shutdown();
    }
}

#[tokio::test]
async fn allowed_output_params_do_not_resurrect_consumed_native_aliases() {
    for route in [Route::Invoke, Route::Converse] {
        let upstream = MockServer::start_async().await;
        let wire = upstream
            .mock_async(|when, then| {
                when.method(POST).path(path(route)).json_body(wire_body(
                    route,
                    APIS[0],
                    false,
                    schema(),
                ));
                then.status(200).json_body(reply(route, false));
            })
            .await;
        let mut config = config(&upstream, route);
        config.model_list[0].oxllm_params.allowed_openai_params =
            Some(vec!["response_format".into()]);
        let (handle, _) = common::start_oxllm(config).await.unwrap();
        let response = common::http_client()
            .post(handle.url(APIS[0]))
            .json(&request(APIS[0], false))
            .send()
            .await
            .unwrap();
        let status = response.status();
        let body: Value = response.json().await.unwrap();
        assert_eq!(status, 200, "{route:?}: {body}");
        wire.assert_hits_async(1).await;
        handle.shutdown();
    }
}

#[tokio::test]
async fn early_messages_errors_use_the_native_error_envelope() {
    let upstream = MockServer::start_async().await;
    let (handle, _) = common::start_oxllm(config(&upstream, Route::Direct))
        .await
        .unwrap();
    for raw in [
        "{",
        "{}",
        r#"{"model":"model","messages":[],"num_retries":-1}"#,
    ] {
        let response = common::http_client()
            .post(handle.url(APIS[2]))
            .header("content-type", "application/json")
            .body(raw)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 400);
        let body: Value = response.json().await.unwrap();
        assert_eq!(body["type"], "error", "{body}");
        assert_eq!(body["error"]["type"], "invalid_request_error");
    }
    handle.shutdown();
}

#[tokio::test]
async fn native_message_schema_events_survive_system_message_extraction() {
    for route in [Route::Direct, Route::Invoke, Route::Vertex, Route::Foundry] {
        for api in [APIS[0], APIS[2]] {
            let upstream = MockServer::start_async().await;
            let format = json!({"type":"json_schema","schema":schema()});
            let mut expected = wire_body(route, api, false, schema());
            expected.as_object_mut().unwrap().remove("output_config");
            expected["messages"][0]["output_config"] = json!({"format":format});
            expected["system"] = json!("Extract records accurately.");
            if matches!(route, Route::Invoke | Route::Vertex) {
                expected["anthropic_beta"] = json!(["mid-conversation-output-config-2026-07-01"]);
            }
            let wire = upstream
                .mock_async(|when, then| {
                    when.method(POST).path(path(route)).json_body(expected);
                    then.status(200).json_body(reply(route, false));
                })
                .await;
            let (handle, _) = common::start_oxllm(config(&upstream, route)).await.unwrap();
            let mut payload = request(api, false);
            payload.as_object_mut().unwrap().remove("response_format");
            payload.as_object_mut().unwrap().remove("output_config");
            payload["messages"][0]["output_config"] = json!({"format":format});
            if api == APIS[0] {
                payload["messages"].as_array_mut().unwrap().insert(
                    0,
                    json!({"role":"system","content":"Extract records accurately."}),
                );
            } else {
                payload["system"] = json!("Extract records accurately.");
            }
            let response = common::http_client()
                .post(handle.url(api))
                .json(&payload)
                .send()
                .await
                .unwrap();
            let status = response.status();
            let body: Value = response.json().await.unwrap();
            assert_eq!(status, 200, "{route:?} {api}: {body}");
            assert_usage_and_completion(api, &body, false);
            wire.assert_hits_async(1).await;
            handle.shutdown();
        }
    }
}

#[tokio::test]
async fn agreeing_schema_aliases_collapse_without_losing_the_schema() {
    for route in ROUTES {
        let upstream = MockServer::start_async().await;
        let wire = upstream
            .mock_async(|when, then| {
                when.method(POST).path(path(route)).json_body(wire_body(
                    route,
                    APIS[0],
                    false,
                    schema(),
                ));
                then.status(200).json_body(reply(route, false));
            })
            .await;
        let (handle, _) = common::start_oxllm(config(&upstream, route)).await.unwrap();
        let mut payload = request(APIS[0], false);
        payload["output_format"] = json!({"type":"json_schema","schema":schema()});
        payload["output_config"] = json!({"format":{"type":"json_schema","schema":schema()}});
        let response = common::http_client()
            .post(handle.url(APIS[0]))
            .json(&payload)
            .send()
            .await
            .unwrap();
        let status = response.status();
        let body: Value = response.json().await.unwrap();
        assert_eq!(status, 200, "{route:?}: {body}");
        wire.assert_hits_async(1).await;
        handle.shutdown();
    }
}

#[tokio::test]
async fn unrepresentable_responses_options_and_malformed_tools_are_not_discarded() {
    let upstream = MockServer::start_async().await;
    let wire = upstream
        .mock_async(|when, then| {
            when.method(POST);
            then.status(200).json_body(reply(Route::Direct, false));
        })
        .await;
    let (handle, _) = common::start_oxllm(config(&upstream, Route::Direct))
        .await
        .unwrap();
    for stream in [false, true] {
        for malformed in [false, true] {
            let mut payload = request(APIS[1], false);
            payload["stream"] = json!(stream);
            if malformed {
                payload["tools"] = json!([42]);
            } else {
                payload["text"]["format"]["future_option"] = json!(true);
            }
            let response = common::http_client()
                .post(handle.url(APIS[1]))
                .json(&payload)
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), 400);
            assert!(
                response.headers()["content-type"]
                    .to_str()
                    .unwrap()
                    .starts_with("application/json")
            );
            assert!(response.json::<Value>().await.unwrap()["error"]["message"].is_string());
        }
    }
    wire.assert_hits_async(0).await;
    handle.shutdown();
}

#[tokio::test]
async fn deployment_overrides_cannot_reenable_tools_after_none_is_mapped() {
    let upstream = MockServer::start_async().await;
    let wire = upstream
        .mock_async(|when, then| {
            when.method(POST);
            then.status(200).json_body(reply(Route::Converse, true));
        })
        .await;
    let mut config = config(&upstream, Route::Converse);
    config.model_list[0].oxllm_params.extra_body = Some(serde_json::Map::from_iter([(
        "toolConfig".into(),
        json!({"tools":[{"toolSpec":{"name":"injected","inputSchema":{"json":{"type":"object"}}}}],"toolChoice":{"any":{}}}),
    )]));
    let (handle, _) = common::start_oxllm(config).await.unwrap();
    let mut payload = request(APIS[0], true);
    payload["tool_choice"] = json!("none");
    let response = common::http_client()
        .post(handle.url(APIS[0]))
        .json(&payload)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 400);
    assert!(
        response.json::<Value>().await.unwrap()["error"]["message"]
            .as_str()
            .unwrap()
            .contains("deployment parameter policy")
    );
    wire.assert_hits_async(0).await;
    handle.shutdown();
}

#[tokio::test]
async fn strict_tool_replies_are_cached_in_the_guarded_namespace() {
    let upstream = MockServer::start_async().await;
    let wire = upstream
        .mock_async(|when, then| {
            when.method(POST)
                .path(path(Route::Direct))
                .json_body(wire_body(Route::Direct, APIS[0], true, schema()));
            then.status(200).json_body(reply(Route::Direct, true));
        })
        .await;
    let (handle, state) = common::start_oxllm(config(&upstream, Route::Direct))
        .await
        .unwrap();
    let mut next = (*state.load_full()).clone();
    next.hooks = vec![Arc::new(CacheHook::new(
        next.cache.clone(),
        Duration::from_secs(60),
    ))];
    state.store(Arc::new(next));
    let mut payload = request(APIS[0], true);
    payload["cache"] = json!({"use_cache":true});
    for cached in [false, true] {
        let response = common::http_client()
            .post(handle.url(APIS[0]))
            .json(&payload)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        assert_eq!(response.headers().contains_key("x-oxllm-cache-hit"), cached);
        let body: Value = response.json().await.unwrap();
        assert_usage_and_completion(APIS[0], &body, true);
        let arguments = body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
            .as_str()
            .unwrap();
        assert_eq!(serde_json::from_str::<Value>(arguments).unwrap(), value());
    }
    wire.assert_hits_async(1).await;
    handle.shutdown();
}

#[tokio::test]
async fn streaming_retries_and_fallbacks_keep_native_schema_and_tool_requirements() {
    for tool_call in [false, true] {
        for api in APIS {
            let upstream = MockServer::start_async().await;
            let mut first_body = wire_body(Route::Direct, api, tool_call, schema());
            first_body["stream"] = json!(true);
            let primary = upstream
                .mock_async(|when, then| {
                    when.method(POST)
                        .path(streaming_path(Route::Direct))
                        .json_body(first_body);
                    then.status(503)
                        .json_body(json!({"error":{"message":"temporarily unavailable"}}));
                })
                .await;
            let name = tool_call.then_some("extract_record");
            let (content_type, body) = stream_reply(Route::Converse, name);
            let fallback = upstream
                .mock_async(|when, then| {
                    when.method(POST)
                        .path(streaming_path(Route::Converse))
                        .json_body(wire_body(Route::Converse, api, tool_call, schema()));
                    then.status(200)
                        .header("content-type", content_type)
                        .body(body);
                })
                .await;
            let mut config = config(&upstream, Route::Direct);
            let mut backup = self::config(&upstream, Route::Converse)
                .model_list
                .remove(0);
            backup.model_name = "backup".into();
            config.model_list.push(backup);
            config.router_settings.routing.num_retries = Some(1);
            config.router_settings.routing.fallbacks = Some(json!([{"model":["backup"]}]));
            let (handle, _) = common::start_oxllm(config).await.unwrap();
            let mut payload = request(api, tool_call);
            payload["stream"] = json!(true);
            if api == APIS[0] {
                payload["stream_options"] = json!({"include_usage":true});
            }
            let response = common::http_client()
                .post(handle.url(api))
                .json(&payload)
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), 200);
            let text = response.text().await.unwrap();
            let mut assembled = String::new();
            for line in text
                .lines()
                .filter_map(|line| line.strip_prefix("data:"))
                .map(str::trim)
                .filter(|line| *line != "[DONE]")
            {
                let event: Value = serde_json::from_str(line).unwrap();
                let fragment = match (api, tool_call) {
                    ("/v1/responses", false) => (event["type"] == "response.output_text.delta")
                        .then(|| event["delta"].as_str())
                        .flatten(),
                    ("/v1/responses", true) => (event["type"]
                        == "response.function_call_arguments.delta")
                        .then(|| event["delta"].as_str())
                        .flatten(),
                    ("/v1/messages", false) => event["delta"]["text"].as_str(),
                    ("/v1/messages", true) => event["delta"]["partial_json"].as_str(),
                    (_, false) => event
                        .pointer("/choices/0/delta/content")
                        .and_then(Value::as_str),
                    (_, true) => event
                        .pointer("/choices/0/delta/tool_calls/0/function/arguments")
                        .and_then(Value::as_str),
                };
                if let Some(fragment) = fragment {
                    assembled.push_str(fragment);
                }
            }
            assert_eq!(
                serde_json::from_str::<Value>(&assembled).unwrap(),
                value(),
                "{api}: {text}"
            );
            primary.assert_hits_async(2).await;
            fallback.assert_hits_async(1).await;
            handle.shutdown();
        }
    }
}

#[tokio::test]
async fn native_named_choices_use_the_same_encoding_and_reverse_mapping_as_tools() {
    let upstream = MockServer::start_async().await;
    let mut expected = wire_body(Route::Converse, APIS[0], true, schema());
    expected["toolConfig"]["tools"][0]["toolSpec"]["name"] = json!("my_tool");
    expected["toolConfig"]
        .as_object_mut()
        .unwrap()
        .remove("toolChoice");
    expected["additionalModelRequestFields"] =
        json!({"tool_choice":{"type":"tool","name":"my_tool","disable_parallel_tool_use":true}});
    let mut reply = reply(Route::Converse, true);
    reply["output"]["message"]["content"][0]["toolUse"]["name"] = json!("my_tool");
    let wire = upstream
        .mock_async(|when, then| {
            when.method(POST)
                .path(path(Route::Converse))
                .json_body(expected);
            then.status(200).json_body(reply);
        })
        .await;
    let (handle, _) = common::start_oxllm(config(&upstream, Route::Converse))
        .await
        .unwrap();
    let mut payload = request(APIS[0], true);
    payload["tools"][0]["function"]["name"] = json!("my.tool");
    payload["tool_choice"]["function"]["name"] = json!("my.tool");
    payload["additional_model_request_fields"] =
        json!({"tool_choice":{"type":"tool","name":"my.tool"}});
    payload["parallel_tool_calls"] = json!(false);
    let response = common::http_client()
        .post(handle.url(APIS[0]))
        .json(&payload)
        .send()
        .await
        .unwrap();
    let status = response.status();
    let body: Value = response.json().await.unwrap();
    assert_eq!(status, 200, "{body}");
    assert_eq!(
        body["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
        "my.tool"
    );
    assert_usage_and_completion(APIS[0], &body, true);
    wire.assert_hits_async(1).await;
    handle.shutdown();
}

#[tokio::test]
async fn native_responses_format_is_guarded_before_relay_body_shaping() {
    for replace in [false, true] {
        let upstream = MockServer::start_async().await;
        let wire=upstream.mock_async(|when,then| {
        when.method(POST);
        then.status(200).json_body(json!({"id":"resp-native","object":"response","model":"gpt-4o","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"{}","annotations":[]}]}],"usage":{"input_tokens":5,"output_tokens":4,"total_tokens":9},"status":"completed"}));
    }).await;
        let mut config = common::proxy_config(&upstream.base_url());
        config.model_list[0].model_name = "model".into();
        if replace {
            config.model_list[0].oxllm_params.extra_body = Some(serde_json::Map::from_iter([(
                "text".into(),
                json!({"format":{"type":"text"}}),
            )]));
        } else {
            config.model_list[0].oxllm_params.additional_drop_params = Some(vec!["text".into()]);
        }
        let (handle, _) = common::start_oxllm(config).await.unwrap();
        let mut payload = request(APIS[1], false);
        payload["cache"] = json!({"use_cache":true});
        let response = common::http_client()
            .post(handle.url("/openai/v1/responses"))
            .json(&payload)
            .send()
            .await
            .unwrap();
        let status = response.status();
        let body: Value = response.json().await.unwrap();
        assert_eq!(status, 400, "{body}");
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("deployment parameter policy")
        );
        wire.assert_hits_async(0).await;
        handle.shutdown();
    }
}

#[tokio::test]
async fn colliding_names_preserve_the_requested_tool_in_every_native_adapter() {
    for route in ROUTES {
        let upstream = MockServer::start_async().await;
        let mut expected = wire_body(route, APIS[0], true, schema());
        if matches!(route, Route::Converse) {
            expected["toolConfig"]["tools"][0]["toolSpec"]["name"] = json!("my_tool");
            let mut second = expected["toolConfig"]["tools"][0].clone();
            second["toolSpec"]["name"] = json!("my_tool_2");
            expected["toolConfig"]["tools"]
                .as_array_mut()
                .unwrap()
                .push(second);
            expected["toolConfig"]["toolChoice"]["tool"]["name"] = json!("my_tool");
        } else {
            expected["tools"][0]["name"] = json!("my_tool");
            let mut second = expected["tools"][0].clone();
            second["name"] = json!("my_tool_2");
            expected["tools"].as_array_mut().unwrap().push(second);
            expected["tool_choice"]["name"] = json!("my_tool");
        }
        let mut result = reply(route, true);
        if matches!(route, Route::Converse) {
            result["output"]["message"]["content"][0]["toolUse"]["name"] = json!("my_tool");
        } else {
            result["content"][0]["name"] = json!("my_tool");
        }
        let wire = upstream
            .mock_async(|when, then| {
                when.method(POST).path(path(route)).json_body(expected);
                then.status(200).json_body(result);
            })
            .await;
        let (handle, _) = common::start_oxllm(config(&upstream, route)).await.unwrap();
        let mut payload = request(APIS[0], true);
        payload["tools"][0]["function"]["name"] = json!("my.tool");
        let mut second = payload["tools"][0].clone();
        second["function"]["name"] = json!("my_tool");
        payload["tools"].as_array_mut().unwrap().push(second);
        payload["tool_choice"]["function"]["name"] = json!("my.tool");
        let response = common::http_client()
            .post(handle.url(APIS[0]))
            .json(&payload)
            .send()
            .await
            .unwrap();
        let status = response.status();
        let body: Value = response.json().await.unwrap();
        assert_eq!(status, 200, "{route:?}: {body}");
        assert_eq!(
            body["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
            "my.tool"
        );
        wire.assert_hits_async(1).await;
        handle.shutdown();
    }
}

#[tokio::test]
async fn converse_native_schema_aliases_preserve_schema_string_encoding() {
    for generic_json in [false, true] {
        let upstream = MockServer::start_async().await;
        let serialized = serde_json::to_string_pretty(&schema()).unwrap();
        let format = json!({"type":"json_schema","structure":{"jsonSchema":{"schema":serialized}}});
        let mut expected = wire_body(Route::Converse, APIS[0], false, schema());
        expected["outputConfig"]["textFormat"]["structure"]["jsonSchema"]["schema"] =
            json!(serialized);
        if generic_json {
            expected["outputConfig"]["textFormat"]["structure"]["jsonSchema"]
                .as_object_mut()
                .unwrap()
                .remove("name");
        }
        let wire = upstream
            .mock_async(|when, then| {
                when.method(POST)
                    .path(path(Route::Converse))
                    .json_body(expected);
                then.status(200).json_body(reply(Route::Converse, false));
            })
            .await;
        let (handle, _) = common::start_oxllm(config(&upstream, Route::Converse))
            .await
            .unwrap();
        let mut payload = request(APIS[0], false);
        if generic_json {
            payload["response_format"] = json!({"type":"json_object"});
        }
        payload["output_config"] = json!({"textFormat":format});
        let response = common::http_client()
            .post(handle.url(APIS[0]))
            .json(&payload)
            .send()
            .await
            .unwrap();
        let status = response.status();
        let body: Value = response.json().await.unwrap();
        assert_eq!(status, 200, "generic={generic_json}: {body}");
        wire.assert_hits_async(1).await;
        handle.shutdown();
    }
}
