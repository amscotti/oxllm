use super::*;
use oxllm_core::deployment::OxllmParams;

fn deployment() -> Deployment {
    Deployment {
        oxllm_params: OxllmParams {
            model: "anthropic/claude-sonnet-4-5".into(),
            ..Default::default()
        },
        ..Default::default()
    }
}

#[test]
fn output_configuration_preserves_independent_keys_with_and_without_derived_format() {
    for response_format in [
        Value::Null,
        json!({"type":"json_schema","json_schema":{"schema":{"type":"object","properties":{}}}}),
    ] {
        let request = json!({
            "messages":[{"role":"user","content":"json"}],
            "output_config":{"effort":"high","retained":"value"},
            "response_format":response_format
        });
        let body = AnthropicProvider
            .transform_request(&request, &deployment())
            .unwrap();
        assert_eq!(body["output_config"]["effort"], "high");
        assert_eq!(body["output_config"]["retained"], "value");
    }
    let body = AnthropicProvider
        .transform_request(
            &json!({
                "messages":[{"role":"user","content":"json"}],
                "output_config":{"retained":"value"},
        "response_format":{"type":"json_schema","json_schema":{"schema":{"type":"object","properties":{}}}}
            }),
            &deployment(),
        )
        .unwrap();
    assert_eq!(body["output_config"]["retained"], "value");
    assert_eq!(body["output_config"]["format"]["type"], "json_schema");
}

#[test]
fn null_thinking_does_not_disable_requested_reasoning() {
    let mut dep = deployment();
    dep.oxllm_params.model = "anthropic/claude-opus-4-6".into();
    let body = AnthropicProvider
        .transform_request(
            &json!({
                "messages":[{"role":"user","content":"think"}],
                "thinking":null,"reasoning_effort":"high"
            }),
            &dep,
        )
        .unwrap();
    assert_eq!(body["thinking"]["type"], "adaptive");
    assert_eq!(body["output_config"]["effort"], "high");
}

#[test]
fn midturn_developer_messages_use_the_native_system_role() {
    assert_eq!(
        normalize_message(&json!({"role":"developer","content":"new instruction"})).unwrap()["role"],
        "system"
    );
}

#[test]
fn deferred_tools_and_explicit_effort_enable_their_required_betas() {
    let flags = collect_beta_headers(
        &json!({
            "output_config":{"effort":"high"},
            "tools":[{"type":"function","function":{"name":"lookup","defer_loading":true}}]
        }),
        "claude-opus-4-5",
    );
    assert!(flags.iter().any(|flag| flag == BETA_ADVANCED_TOOL_USE));
    assert!(flags.iter().any(|flag| flag == BETA_EFFORT));
}

#[test]
fn native_thinking_usage_and_redacted_stream_blocks_survive_translation() {
    let response = json!({
        "id":"msg-think","type":"message","model":"claude-opus-4-6","speed":"standard",
        "content":[{"type":"text","text":"answer"}],"stop_reason":"end_turn",
        "usage":{"input_tokens":10,"output_tokens":7,"output_tokens_details":{"thinking_tokens":3}}
    });
    let body = AnthropicProvider
        .transform_response(&response, "claude-opus-4-6")
        .unwrap();
    assert_eq!(
        body["usage"]["completion_tokens_details"]["reasoning_tokens"],
        3
    );
    assert_eq!(body["speed"], "standard");
    let mut state = None;
    let start = format!(
        "data: {}",
        json!({"type":"message_start","message":response})
    );
    let first = AnthropicProvider
        .transform_stream_chunk_stateful(&start, &mut state, &json!({}))
        .unwrap()
        .unwrap();
    assert_eq!(first["speed"], "standard");
    let block = json!({"type":"redacted_thinking","data":"opaque-base64"});
    let event = format!(
        "data: {}",
        json!({"type":"content_block_start","index":0,"content_block":block})
    );
    let translated = AnthropicProvider
        .transform_stream_chunk_stateful(&event, &mut state, &json!({}))
        .unwrap()
        .unwrap();
    assert_eq!(
        translated["choices"][0]["delta"]["thinking_blocks"][0],
        block
    );
    let finish = format!(
        "data: {}",
        json!({"type":"message_delta","delta":{"stop_reason":"end_turn"},
        "usage":{"output_tokens":7,"output_tokens_details":{"thinking_tokens":3}}})
    );
    let translated = AnthropicProvider
        .transform_stream_chunk_stateful(&finish, &mut state, &json!({}))
        .unwrap()
        .unwrap();
    assert_eq!(
        translated["usage"]["completion_tokens_details"]["reasoning_tokens"],
        3
    );
    assert_eq!(translated["speed"], "standard");
}

#[test]
fn validation_stop_reasons_match_in_streaming_and_nonstreaming_responses() {
    for (reason, expected) in [
        ("refusal", "content_filter"),
        ("model_context_window_exceeded", "length"),
        ("max_tokens", "length"),
        ("end_turn", "stop"),
    ] {
        let response = json!({
            "id": "msg_stop", "model": "claude-sonnet-4-5",
            "content": [{"type": "text", "text": "partial reply"}],
            "stop_reason": reason,
            "usage": {"input_tokens": 2, "output_tokens": 3},
        });
        let full = AnthropicProvider
            .transform_response(&response, "claude-sonnet-4-5")
            .unwrap();
        let mut state = None;
        let raw = format!(
            "data: {}",
            json!({
                "type": "message_delta", "delta": {"stop_reason": reason},
                "usage": {"output_tokens": 3},
            })
        );
        let streamed = AnthropicProvider
            .transform_stream_chunk_stateful(&raw, &mut state, &json!({}))
            .unwrap()
            .unwrap();
        assert_eq!(full["choices"][0]["finish_reason"], expected);
        assert_eq!(streamed["choices"][0]["finish_reason"], expected);
    }
}

#[test]
fn validation_parallel_tools_flag_uses_only_supported_choice_variants() {
    for (choice, kind) in [
        (json!(null), "auto"),
        (json!("auto"), "auto"),
        (json!("required"), "any"),
        (
            json!({"type": "function", "function": {"name": "lookup"}}),
            "tool",
        ),
        (json!("none"), "none"),
    ] {
        let request = json!({
            "messages": [{"role": "user", "content": "look up"}],
            "tools": [{"type": "function", "function": {
                "name": "lookup", "parameters": {"type": "object", "properties": {}}
            }}],
            "tool_choice": choice,
            "parallel_tool_calls": false,
        });
        let body = AnthropicProvider
            .transform_request(&request, &deployment())
            .unwrap();
        assert!(body.get("disable_parallel_tool_use").is_none());
        assert_eq!(body["tool_choice"]["type"], kind);
        if kind == "none" {
            assert!(
                body["tool_choice"]
                    .get("disable_parallel_tool_use")
                    .is_none()
            );
        } else {
            assert_eq!(body["tool_choice"]["disable_parallel_tool_use"], true);
        }
    }
    let body = AnthropicProvider
        .transform_request(
            &json!({
                "messages": [{"role": "user", "content": "hello"}], "parallel_tool_calls": false,
            }),
            &deployment(),
        )
        .unwrap();
    assert!(body.get("tool_choice").is_none());
    assert!(body.get("disable_parallel_tool_use").is_none());
}

#[test]
fn validation_signed_thinking_survives_response_to_request_roundtrip() {
    let blocks = json!([
        {"type": "thinking", "thinking": "check the data", "signature": "opaque-signed-value"},
        {"type": "redacted_thinking", "data": "AA=="},
        {"type": "text", "text": "I will check."},
        {"type": "tool_use", "id": "call_1", "name": "lookup", "input": {"key": "x"}},
    ]);
    let response = AnthropicProvider
        .transform_response(
            &json!({
                "id": "msg_replay", "model": "claude-sonnet-4-5", "content": blocks,
                "stop_reason": "tool_use", "usage": {"input_tokens": 3, "output_tokens": 5},
            }),
            "claude-sonnet-4-5",
        )
        .unwrap();
    let body = AnthropicProvider
        .transform_request(
            &json!({
                "messages": [
                    {"role": "user", "content": "check x"},
                    response["choices"][0]["message"],
                    {"role": "tool", "tool_call_id": "call_1", "content": "found"},
                ],
            }),
            &deployment(),
        )
        .unwrap();
    assert_eq!(body["messages"][1]["content"], blocks);
    assert_eq!(body["messages"][2]["content"][0]["tool_use_id"], "call_1");
}

#[test]
fn validation_thinking_replay_avoids_duplicates_and_requires_signatures() {
    let thinking = json!({"type": "thinking", "thinking": "unchanged", "signature": "sig"});
    let request = json!({"messages": [{
        "role": "assistant", "content": [thinking, {"type": "text", "text": "ok"}],
        "thinking_blocks": [thinking], "reasoning_content": "unchanged",
    }]});
    let body = AnthropicProvider
        .transform_request(&request, &deployment())
        .unwrap();
    assert_eq!(body["messages"][0]["content"].as_array().unwrap().len(), 2);
    assert_eq!(body["messages"][0]["content"][0], thinking);

    for assistant in [
        json!({"role": "assistant", "content": "ok", "reasoning_content": "unsigned"}),
        json!({"role": "assistant", "content": "ok", "thinking_blocks": [{"type": "thinking", "thinking": "unsigned"}]}),
    ] {
        assert!(matches!(
            AnthropicProvider.transform_request(&json!({"messages": [assistant]}), &deployment()),
            Err(oxllm_core::Error::BadRequest(_))
        ));
    }
    let body = AnthropicProvider
        .transform_request(
            &json!({"messages": [{
                "role": "assistant", "content": "ok", "reasoning_content": "single streamed block",
                "thinking_blocks": [{"type": "thinking", "signature": "sig"}],
            }]}),
            &deployment(),
        )
        .unwrap();
    assert_eq!(
        body["messages"][0]["content"][0],
        json!({
            "type": "thinking", "thinking": "single streamed block", "signature": "sig",
        })
    );
}

#[test]
fn validation_sse_lone_cr_and_multiline_data_keep_anthropic_dispatch() {
    let mut state = None;
    let start = "event: message_start\rdata: {\rdata: \"type\":\"message_start\",\"message\":{\"id\":\"msg_cr\",\"model\":\"claude-sonnet-4-5\",\"usage\":{}}}\r\r";
    let first = AnthropicProvider
        .transform_stream_chunk_stateful(start, &mut state, &json!({}))
        .unwrap()
        .unwrap();
    let next = AnthropicProvider.transform_stream_chunk_stateful(
        "event: content_block_delta\rdata: {\"type\":\"content_block_delta\",\rdata: \"delta\":{\"type\":\"text_delta\",\"text\":\"hello\"}}\r\r",
        &mut state, &json!({}),
    ).unwrap().unwrap();
    assert_eq!(first["id"], "msg_cr");
    assert_eq!(next["id"], first["id"]);
    assert_eq!(next["choices"][0]["delta"]["content"], "hello");
}

#[test]
fn validation_schema_wrappers_and_unsupported_strict_mode_are_rejected() {
    for wrapper in [
        json!({"name": "missing"}),
        json!({"schema": null}),
        json!(false),
    ] {
        assert!(matches!(
            AnthropicProvider.transform_request(
                &json!({
                    "messages": [{"role": "user", "content": "json"}],
                    "response_format": {"type": "json_schema", "json_schema": wrapper},
                }),
                &deployment()
            ),
            Err(oxllm_core::Error::BadRequest(_))
        ));
    }
    let mut legacy = deployment();
    legacy.oxllm_params.model = "anthropic/claude-3-5-sonnet".into();
    assert!(matches!(
        AnthropicProvider.transform_request(
            &json!({
                "messages": [{"role": "user", "content": "json"}],
                "response_format": {"type": "json_schema", "json_schema": {
                    "strict": true, "schema": {"type": "object", "properties": {}}
                }},
            }),
            &legacy
        ),
        Err(oxllm_core::Error::BadRequest(_))
    ));
}

#[test]
fn validation_legacy_functions_correlate_stable_ids_and_keep_definition_name_roundtrips() {
    let request = json!({
        "messages": [
            {"role": "user", "content": "check weather twice"},
            {"role": "assistant", "content": null, "function_call": {"name": "get.weather", "arguments": "{\"city\":\"Paris\"}"}},
            {"role": "function", "name": "get.weather", "content": "sunny"},
            {"role": "assistant", "function_call": {"name": "get.weather", "arguments": {"city": "London"}}},
            {"role": "function", "name": "get.weather", "content": [{"type": "text", "text": "rainy"}]},
        ],
        "functions": [{"name": "get.weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}],
        "function_call": {"name": "get.weather"},
        "tools": []
    });
    let body = AnthropicProvider
        .transform_request(&request, &deployment())
        .unwrap();
    assert_eq!(
        body,
        AnthropicProvider
            .transform_request(&request, &deployment())
            .unwrap()
    );
    let first = &body["messages"][1]["content"][0];
    let second = &body["messages"][3]["content"][0];
    assert_eq!(first["type"], "tool_use");
    assert_eq!(first["name"], "get_weather");
    assert_eq!(first["input"], json!({"city": "Paris"}));
    assert_eq!(second["input"], json!({"city": "London"}));
    assert!(!first["id"].as_str().unwrap().is_empty());
    assert_ne!(first["id"], second["id"]);
    assert_eq!(
        body["messages"][2]["content"][0]["tool_use_id"],
        first["id"]
    );
    assert_eq!(
        body["messages"][4]["content"][0]["tool_use_id"],
        second["id"]
    );
    assert_eq!(body["tools"][0]["name"], "get_weather");
    assert_eq!(
        body["tool_choice"],
        json!({"type": "tool", "name": "get_weather"})
    );
    assert!(body.get("functions").is_none());
    assert!(body.get("function_call").is_none());
    for message in body["messages"].as_array().unwrap() {
        assert!(message.get("function_call").is_none());
        assert_ne!(message["role"], "function");
    }
    let response = AnthropicProvider.transform_response_with_request(
        &json!({"id": "msg-legacy", "model": "claude-sonnet-4-5", "stop_reason": "tool_use", "usage": {},
            "content": [{"type": "tool_use", "id": "new-call", "name": "get_weather", "input": {}}]}),
        "claude-sonnet-4-5", &request,
    ).unwrap();
    assert_eq!(
        response["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
        "get.weather"
    );
    assert!(request["messages"][1]["function_call"].get("id").is_none());
}

#[test]
fn validation_legacy_generated_ids_do_not_collide_with_modern_history() {
    let request = json!({"messages": [
        {"role": "user", "content": "lookup"},
        {"role": "assistant", "function_call": {"name": "legacy", "arguments": "{}"}},
        {"role": "function", "name": "legacy", "content": "first"},
        {"role": "assistant", "tool_calls": [{"id": "toolu_legacy_1", "type": "function", "function": {"name": "modern", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "toolu_legacy_1", "content": "second"},
    ]});
    let body = AnthropicProvider
        .transform_request(&request, &deployment())
        .unwrap();
    let legacy = &body["messages"][1]["content"][0];
    let modern = &body["messages"][3]["content"][0];
    assert_ne!(legacy["id"], modern["id"]);
    assert_eq!(modern["id"], "toolu_legacy_1");
    assert_eq!(
        body["messages"][2]["content"][0]["tool_use_id"],
        legacy["id"]
    );
    assert_eq!(
        body["messages"][4]["content"][0]["tool_use_id"],
        modern["id"]
    );

    let explicit = json!({"messages": [
        {"role": "assistant", "function_call": {"id": "caller-id", "name": "legacy", "arguments": ""}},
        {"role": "function", "name": "legacy", "tool_call_id": "caller-id", "content": "done"},
    ]});
    let body = AnthropicProvider
        .transform_request(&explicit, &deployment())
        .unwrap();
    assert_eq!(body["messages"][0]["content"][0]["id"], "caller-id");
    assert_eq!(body["messages"][0]["content"][0]["input"], json!({}));
    assert_eq!(
        body["messages"][1]["content"][0]["tool_use_id"],
        "caller-id"
    );
}

#[test]
fn validation_legacy_results_can_correlate_to_modern_calls_without_changing_modern_results() {
    for role in ["function", "tool"] {
        let mut result = json!({"role": role, "content": "done"});
        if role == "function" {
            result["name"] = json!("lookup");
        } else {
            result["tool_call_id"] = json!("modern-id");
        }
        let request = json!({"messages": [
            {"role": "assistant", "content": "checking", "tool_calls": [{"id": "modern-id", "type": "function", "function": {"name": "lookup", "arguments": "{\"key\":1}"}}]},
            result,
        ]});
        let body = AnthropicProvider
            .transform_request(&request, &deployment())
            .unwrap();
        assert_eq!(
            body["messages"][0]["content"][1],
            json!({"type": "tool_use", "id": "modern-id", "name": "lookup", "input": {"key": 1}})
        );
        assert_eq!(
            body["messages"][1]["content"][0],
            json!({"type": "tool_result", "tool_use_id": "modern-id", "content": "done"})
        );
    }
}

#[test]
fn validation_legacy_functions_and_selectors_survive_supported_parameter_filtering() {
    let params = AnthropicProvider.supported_params().unwrap();
    for (choice, native) in [
        (json!("auto"), json!({"type": "auto"})),
        (json!("none"), json!({"type": "none"})),
        (
            json!({"name": "lookup"}),
            json!({"type": "tool", "name": "lookup"}),
        ),
    ] {
        let mut request = json!({
            "messages": [{"role": "user", "content": "lookup"}],
            "functions": [{"name": "lookup", "parameters": {"type": "object"}}],
            "function_call": choice, "unsupported": true
        });
        request
            .as_object_mut()
            .unwrap()
            .retain(|key, _| params.contains(&key.as_str()));
        let body = AnthropicProvider
            .transform_request(&request, &deployment())
            .unwrap();
        assert_eq!(body["tools"][0]["name"], "lookup");
        assert_eq!(body["tool_choice"], native);
        assert!(body.get("unsupported").is_none());
    }
}

#[test]
fn validation_malformed_or_unmatched_legacy_function_history_returns_bad_request() {
    let call = json!({"role": "assistant", "function_call": {"name": "lookup", "arguments": "{}"}});
    let result = json!({"role": "function", "name": "lookup", "content": "done"});
    for messages in [
        json!([result]),
        json!([call]),
        json!([call, {"role": "function", "name": "other", "content": "wrong"}]),
        json!([call, {"role": "user", "content": "intervening"}, result]),
        json!([call, result, result]),
        json!([call, {"role": "function", "content": "missing name"}]),
        json!([call, {"role": "function", "name": "lookup", "content": false}]),
        json!([{"role": "user", "function_call": {"name": "lookup", "arguments": "{}"}}]),
        json!([{"role": "assistant", "function_call": true}, result]),
        json!([{"role": "assistant", "function_call": {"name": "", "arguments": "{}"}}, result]),
        json!([{"role": "assistant", "function_call": {"name": "lookup", "arguments": "not-json"}}, result]),
        json!([{"role": "assistant", "function_call": {"name": "lookup", "arguments": "[]"}}, result]),
        json!([{"role": "assistant", "function_call": {"name": "lookup", "arguments": false}}, result]),
        json!([{"role": "assistant", "function_call": {"name": "lookup", "id": ""}}, result]),
        json!([{"role": "assistant", "function_call": {"name": "lookup"}, "tool_calls": [{"id": "mixed", "function": {"name": "lookup"}}]}, result]),
        json!([{"role": "assistant", "tool_calls": [{"id": "a", "function": {"name": "lookup"}}, {"id": "b", "function": {"name": "lookup"}}]}, result]),
    ] {
        assert!(matches!(
            AnthropicProvider.transform_request(&json!({"messages": messages}), &deployment()),
            Err(oxllm_core::Error::BadRequest(_))
        ));
    }
}

#[test]
fn validation_malformed_legacy_definitions_and_selectors_return_bad_request() {
    for request in [
        json!({"functions": true}),
        json!({"functions": [{"name": ""}]}),
        json!({"functions": [{"name": "lookup", "parameters": []}]}),
        json!({"functions": [{"name": "lookup"}], "tools": [{"type": "function", "function": {"name": "other"}}]}),
        json!({"functions": [{"name": "lookup"}], "function_call": {"name": "missing"}}),
        json!({"functions": [{"name": "lookup"}], "function_call": {"name": ""}}),
        json!({"functions": [{"name": "lookup"}], "function_call": "required"}),
        json!({"functions": [{"name": "lookup"}], "function_call": "auto", "tool_choice": "none"}),
    ] {
        assert!(matches!(
            AnthropicProvider.transform_request(&request, &deployment()),
            Err(oxllm_core::Error::BadRequest(_))
        ));
    }
}

#[test]
fn validation_legacy_selectors_without_definitions_omit_native_tool_choice() {
    for selector in ["auto", "none"] {
        let request = json!({
            "messages": [{"role": "user", "content": "hello"}],
            "function_call": selector
        });
        let body = AnthropicProvider
            .transform_request(&request, &deployment())
            .unwrap();
        assert!(body.get("tools").is_none());
        assert!(body.get("tool_choice").is_none());
    }
    let request = json!({"messages": [
        {"role": "assistant", "function_call": {"name": "lookup", "arguments": "{}"}},
        {"role": "function", "name": "lookup", "content": null}
    ]});
    let body = AnthropicProvider
        .transform_request(&request, &deployment())
        .unwrap();
    assert_eq!(body["messages"][1]["content"][0]["content"], "");
    assert_eq!(
        body["messages"][1]["content"][0]["tool_use_id"],
        body["messages"][0]["content"][0]["id"]
    );
}
