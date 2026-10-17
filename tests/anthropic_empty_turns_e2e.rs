//! Empty-turn/cache-breakpoint normalization through local mock transports.

mod common;

use httpmock::{Method::POST, MockServer};
use serde_json::{Value, json};

#[derive(Clone, Copy, Debug)]
enum Route {
    Direct,
    Foundry,
    Invoke,
}

const ROUTES: [Route; 3] = [Route::Direct, Route::Foundry, Route::Invoke];
const APIS: [&str; 2] = ["/v1/chat/completions", "/v1/messages"];
const PLACEHOLDER: &str = "[System: Empty message content sanitised to satisfy protocol]";
const MESSAGE_CONFIG_BETA: &str = "mid-conversation-output-config-2026-07-01";

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
        Route::Invoke => {
            params.model = "bedrock/anthropic.claude-haiku-5-5".into();
            params.api_base = Some(upstream.url("/bedrock"));
            params.api_key = None;
            params.aws_access_key_id = Some("AKIDEXAMPLE".into());
            params.aws_secret_access_key = Some("synthetic-local-signing-only".into());
            params
                .extra
                .insert("aws_region_name".into(), json!("us-east-1"));
            params.extra.insert("use_converse_api".into(), json!(false));
        }
    }
    config
}

fn path(route: Route) -> &'static str {
    match route {
        Route::Direct => "/direct/messages",
        Route::Foundry => "/foundry/anthropic/v1/messages",
        Route::Invoke => "/bedrock/model/anthropic.claude-haiku-5-5/invoke",
    }
}

fn reply() -> Value {
    json!({"id": "msg-empty-turn", "type": "message", "role": "assistant",
        "model": "claude-haiku-5-5", "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn", "usage": {"input_tokens": 5, "output_tokens": 1},
    })
}

fn control() -> Value {
    json!({"type": "ephemeral", "ttl": "1h"})
}

fn placeholder(marked: bool) -> Value {
    let mut block = json!({"type": "text", "text": PLACEHOLDER});
    if marked {
        block["cache_control"] = control();
    }
    json!({"role": "user", "content": [block]})
}

fn output_config() -> Value {
    json!({"format": {"type": "json_schema", "schema": {
        "type": "object", "properties": {"value": {"type": "string", "minLength": 3}},
        "required": ["value"], "additionalProperties": false,
    }}, "effort": "low", "native_option": {"keep": true}})
}

fn wire_body(route: Route, messages: Value, message_config: bool) -> Value {
    let mut body = json!({"messages": messages, "max_tokens": 64});
    if matches!(route, Route::Invoke) {
        body["anthropic_version"] = json!("bedrock-2023-05-31");
        if message_config {
            body["anthropic_beta"] = json!([MESSAGE_CONFIG_BETA]);
        }
    } else {
        body["model"] = json!("claude-haiku-5-5");
    }
    body
}

async fn assert_wire(route: Route, api: &str, input: Value, expected: Value, message_config: bool) {
    let upstream = MockServer::start_async().await;
    let wire = upstream
        .mock_async(|when, then| {
            let when = when.method(POST).path(path(route)).json_body(wire_body(
                route,
                expected,
                message_config,
            ));
            if message_config && !matches!(route, Route::Invoke) {
                when.header("anthropic-beta", MESSAGE_CONFIG_BETA);
            }
            then.status(200).json_body(reply());
        })
        .await;
    let (handle, _) = common::start_oxllm(config(&upstream, route)).await.unwrap();
    let response = common::http_client()
        .post(handle.url(api))
        .json(&json!({"model": "model", "messages": input, "max_tokens": 64}))
        .send()
        .await
        .unwrap();
    let status = response.status();
    let body: Value = response.json().await.unwrap();
    handle.shutdown();
    assert_eq!(status, 200, "{route:?} {api}: {body}");
    wire.assert_hits_async(1).await;
    if api == "/v1/messages" {
        assert_eq!(body["content"][0]["text"], "ok");
    } else {
        assert_eq!(body["choices"][0]["message"]["content"], "ok");
    }
}

#[tokio::test]
async fn review_root_empty_cache_breakpoints_survive_both_apis_on_all_routes() {
    let text = json!({"role": "user", "content": "real content"});
    let empty = json!({"role": "user", "content": [], "cache_control": control()});
    let call = json!({"role": "assistant", "content": [{
        "type": "tool_use", "id": "call_1", "name": "lookup", "input": {},
    }]});
    let result = json!({"role": "tool", "tool_call_id": "call_1", "content": "found"});
    let native_result = json!({"role": "user", "content": [{
        "type": "tool_result", "tool_use_id": "call_1", "content": "found",
    }]});
    for route in ROUTES {
        for api in APIS {
            let result = if api == APIS[1] {
                native_result.clone()
            } else {
                result.clone()
            };
            for (input, expected) in [
                (json!([empty, text]), json!([placeholder(true), text])),
                (json!([text, empty]), json!([text, placeholder(true)])),
                (
                    json!([empty, {"role": "user", "cache_control": control()},
                    {"role": "user", "content": null, "cache_control": control()},
                    {"role": "user", "content": []}]),
                    json!([placeholder(true), placeholder(true), placeholder(true)]),
                ),
                (
                    json!([call, empty, text, result]),
                    json!([call, native_result, text]),
                ),
                (
                    json!([call, text, result, empty]),
                    json!([call, native_result, text, placeholder(true)]),
                ),
            ] {
                assert_wire(route, api, input, expected, false).await;
            }
        }
    }
}

#[tokio::test]
async fn cached_blank_text_keeps_its_mark_through_chat_and_messages_on_all_routes() {
    for route in ROUTES {
        for api in APIS {
            let block = json!({"type": "text", "text": " \t\r\n", "cache_control": control(),
                "citations": [], "native_option": {"nested": [null, 1, true]},
            });
            let mut expected = block.clone();
            expected["text"] = json!(PLACEHOLDER);
            assert_wire(
                route,
                api,
                json!([{"role": "user", "content": [block]}]),
                json!([{"role": "user", "content": [expected]}]),
                false,
            )
            .await;
        }
    }
}

#[tokio::test]
async fn unmarked_empty_siblings_do_not_invent_text_on_chat_or_messages() {
    for route in ROUTES {
        for api in APIS {
            let text = json!({"role": "user", "content": "real content"});
            assert_wire(
                route,
                api,
                json!([{"role": "user"}, {"role": "user", "content": null},
                    text, {"role": "user", "content": []}]),
                json!([text]),
                false,
            )
            .await;
            assert_wire(
                route,
                api,
                json!([{"role": "user"}, {"role": "user", "content": null},
                    {"role": "user", "content": []}]),
                json!([placeholder(false)]),
                false,
            )
            .await;
        }
    }
}

#[tokio::test]
async fn empty_user_native_output_constraints_survive_chat_and_messages_on_all_routes() {
    for route in ROUTES {
        for api in APIS {
            for empty in [
                json!({"role": "user"}),
                json!({"role": "user", "content": null}),
                json!({"role": "user", "content": []}),
            ] {
                for marked in [false, true] {
                    let mut input = empty.clone();
                    input["output_config"] = output_config();
                    if marked {
                        input["cache_control"] = control();
                    }
                    let mut expected = placeholder(marked);
                    expected["output_config"] = output_config();
                    let text = json!({"role": "user", "content": "real content"});
                    assert_wire(
                        route,
                        api,
                        json!([input, text]),
                        json!([expected, text]),
                        true,
                    )
                    .await;
                }
            }
        }
    }
}

#[tokio::test]
async fn suppressing_a_configured_empty_turn_is_a_local_400_on_chat_and_messages() {
    for route in ROUTES {
        for api in APIS {
            let upstream = MockServer::start_async().await;
            let wire = upstream
                .mock_async(|when, then| {
                    when.method(POST);
                    then.status(200).json_body(reply());
                })
                .await;
            let (handle, _) = common::start_oxllm(config(&upstream, route)).await.unwrap();
            let response = common::http_client()
                .post(handle.url(api))
                .json(&json!({"model": "model", "max_tokens": 64, "messages": [
                    {"role": "assistant", "content": [{"type": "tool_use", "id": "call_1", "name": "lookup", "input": {}}]},
                    {"role": "user", "content": [], "output_config": output_config()},
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "found"}]},
                ]}))
                .send()
                .await
                .unwrap();
            let status = response.status();
            let body: Value = response.json().await.unwrap();
            handle.shutdown();
            assert_eq!(status, 400, "{route:?} {api}: {body}");
            let error = body["error"]["message"].as_str().unwrap();
            assert!(error.contains("output_config"), "{route:?} {api}: {body}");
            assert!(error.contains("tool_result"), "{route:?} {api}: {body}");
            wire.assert_hits_async(0).await;
        }
    }
}

async fn assert_review_local_error(route: Route, api: &str, input: Value, detail: &str) {
    let upstream = MockServer::start_async().await;
    let wire = upstream
        .mock_async(|when, then| {
            when.method(POST);
            then.status(200).json_body(reply());
        })
        .await;
    let (handle, _) = common::start_oxllm(config(&upstream, route)).await.unwrap();
    for stream in [false, true] {
        let request =
            json!({"model": "model", "max_tokens": 64, "messages": input, "stream": stream});
        let original = request.clone();
        let response = common::http_client()
            .post(handle.url(api))
            .json(&request)
            .send()
            .await
            .unwrap();
        let status = response.status();
        let body: Value = response.json().await.unwrap();
        assert_eq!(request, original);
        assert_eq!(status, 400, "{route:?} {api} stream={stream}: {body}");
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("output_config"),
            "{body}"
        );
        assert!(
            body["error"]["message"].as_str().unwrap().contains(detail),
            "{body}"
        );
        if api == APIS[1] {
            assert_eq!(body["type"], "error");
            assert_eq!(body["error"]["type"], "invalid_request_error");
        }
    }
    handle.shutdown();
    wire.assert_hits_async(0).await;
}

#[tokio::test]
async fn review_separate_output_config_events_cannot_be_reordered_on_either_api() {
    let mut low = output_config();
    low["native_option"] = json!({"source": "text"});
    let mut high = low.clone();
    high["native_option"] = json!({"source": "result"});
    for route in ROUTES {
        for api in APIS {
            for (text_config, tool_config) in [
                (json!({"effort": "low"}), json!({"effort": "high"})),
                (low.clone(), high.clone()),
            ] {
                assert_review_local_error(route, api, json!([
                    {"role": "assistant", "content": [{"type": "tool_use", "id": "call_1", "name": "lookup", "input": {}}]},
                    {"role": "user", "content": "before result", "output_config": text_config},
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": "found"}], "output_config": tool_config},
                ]), "reorder").await;
            }
        }
    }
}

#[tokio::test]
async fn review_mixed_source_output_config_is_rejected_before_messages_conversion() {
    for route in ROUTES {
        for api in [APIS[1], APIS[0]] {
            for config in [
                output_config(),
                json!({"effort": "low"}),
                json!({"native_option": {"keep": true}}),
            ] {
                for result_last in [false, true] {
                    let text = json!({"type": "text", "text": "ordinary"});
                    let result =
                        json!({"type": "tool_result", "tool_use_id": "call_1", "content": "found"});
                    let blocks = if result_last {
                        json!([text, result])
                    } else {
                        json!([result, text])
                    };
                    assert_review_local_error(route, api, json!([
                        {"role": "assistant", "content": [{"type": "tool_use", "id": "call_1", "name": "lookup", "input": {}}]},
                        {"role": "user", "content": blocks, "output_config": config},
                    ]), "split").await;
                }
            }
        }
    }
}

#[tokio::test]
async fn review_split_root_cache_mark_stays_at_the_original_endpoint_on_both_apis() {
    for route in ROUTES {
        for api in APIS {
            for result_last in [false, true] {
                for block_mark in [false, true] {
                    let root_cc = control();
                    let block_cc = json!({"type": "ephemeral", "ttl": "5m"});
                    let mut text = json!({"type": "text", "text": "ordinary"});
                    let mut result =
                        json!({"type": "tool_result", "tool_use_id": "call_1", "content": "found"});
                    if block_mark {
                        if result_last {
                            result["cache_control"] = block_cc.clone();
                        } else {
                            text["cache_control"] = block_cc.clone();
                        }
                    }
                    let input_blocks = if result_last {
                        json!([text, result])
                    } else {
                        json!([result, text])
                    };
                    let effective = if block_mark {
                        block_cc
                    } else {
                        root_cc.clone()
                    };
                    let mut expected_result =
                        json!({"type": "tool_result", "tool_use_id": "call_1", "content": "found"});
                    let expected_text = if result_last {
                        expected_result["cache_control"] = effective;
                        if api == APIS[0] {
                            json!({"role": "user", "content": [{"type": "text", "text": "ordinary"}]})
                        } else {
                            json!({"role": "user", "content": "ordinary"})
                        }
                    } else {
                        json!({"role": "user", "content": [{"type": "text", "text": "ordinary", "cache_control": effective}]})
                    };
                    let call = json!({"role": "assistant", "content": [{"type": "tool_use", "id": "call_1", "name": "lookup", "input": {}}]});
                    assert_wire(route, api,
                        json!([call, {"role": "user", "content": input_blocks, "cache_control": root_cc}]),
                        json!([call, {"role": "user", "content": [expected_result]}, expected_text]), false,
                    ).await;
                }
            }
        }
    }
}
