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
        normalize_message(&json!({"role":"developer","content":"new instruction"}))
            .unwrap()
            .unwrap()["role"],
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
    let block = json!({"type":"redacted_thinking","data":"opaque-base64","index":0});
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

fn replay_with_inline_thinking(text: &str, reasoning: Option<&str>) -> Value {
    let mut assistant = json!({
        "role": "assistant",
        "content": [{"type": "thinking", "thinking": text, "signature": "original-signature"}],
        "thinking_blocks": [{"type": "thinking", "signature": "original-signature"}],
        "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}],
    });
    if let Some(reasoning) = reasoning {
        assistant["reasoning_content"] = json!(reasoning);
    }
    AnthropicProvider
        .transform_request(
            &json!({"messages": [
                {"role": "user", "content": "check"}, assistant,
                {"role": "tool", "tool_call_id": "call_1", "content": "found"},
            ]}),
            &deployment(),
        )
        .unwrap()
}

#[test]
fn thinking_replay_regression_inline_empty_beats_missing_mirror() {
    let body = replay_with_inline_thinking("", None);
    assert_eq!(
        body["messages"][1]["content"],
        json!([
            {"type": "thinking", "thinking": "", "signature": "original-signature"},
            {"type": "tool_use", "id": "call_1", "name": "lookup", "input": {}},
        ])
    );
    assert_eq!(body["messages"][2]["content"][0]["tool_use_id"], "call_1");
}

#[test]
fn thinking_replay_regression_inline_text_beats_refilled_mirror() {
    let body = replay_with_inline_thinking("plan", Some("plan"));
    assert_eq!(
        body["messages"][1]["content"],
        json!([
            {"type": "thinking", "thinking": "plan", "signature": "original-signature"},
            {"type": "tool_use", "id": "call_1", "name": "lookup", "input": {}},
        ])
    );
    assert_eq!(body["messages"][2]["content"][0]["tool_use_id"], "call_1");
}

#[test]
fn thinking_replay_regression_explicit_signed_blank_is_not_refilled() {
    for text in ["", " \t\r\n", "signed text"] {
        let block =
            json!({"type": "thinking", "thinking": text, "signature": "original-signature"});
        let body = AnthropicProvider
            .transform_request(
                &json!({"messages": [{
                    "role": "assistant", "content": "answer", "thinking_blocks": [block],
                    "reasoning_content": "different unsigned text",
                }]}),
                &deployment(),
            )
            .unwrap();
        assert_eq!(body["messages"][0]["content"][0], block);
    }
}

#[test]
fn thinking_replay_inline_normalization_preserves_block_order_and_malformed_types() {
    for text in [
        None,
        Some(Value::Null),
        Some(json!("")),
        Some(json!(" \t\r\n")),
        Some(json!("plan")),
        Some(json!(42)),
    ] {
        let mut first = json!({"type": "thinking", "signature": "sig"});
        if let Some(text) = &text {
            first["thinking"] = text.clone();
        }
        let redacted = json!({"type": "redacted_thinking", "data": "AA==", "cache_control": {"type": "ephemeral"}});
        let second = json!({"type": "thinking", "thinking": "second", "signature": "second-sig"});
        let mut assistant = json!({
            "role": "assistant", "content": [first, redacted, second],
            "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}],
            "reasoning_content": "not either signed block",
        });
        // A malformed extension must error rather than be hidden by inline
        // authority; malformed native fields still reach provider validation.
        if text
            .as_ref()
            .is_none_or(|text| text.is_null() || text.is_string())
        {
            assistant["thinking_blocks"] = json!([
                {"type": "thinking", "signature": "second-sig"},
                {"type": "redacted_thinking", "data": "AA=="},
                {"type": "thinking", "thinking": "conflicting mirror", "signature": "sig"},
            ]);
        }
        let body = AnthropicProvider
            .transform_request(&json!({"messages": [assistant]}), &deployment())
            .unwrap();
        let expected_text = text.filter(|text| !text.is_null()).unwrap_or(json!(""));
        first["thinking"] = expected_text;
        assert_eq!(
            body["messages"][0]["content"],
            json!([
                first, redacted, second,
                {"type": "tool_use", "id": "call_1", "name": "lookup", "input": {}},
            ])
        );
    }
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

mod empty_user_runs {
    use super::*;

    fn messages(input: Vec<Value>) -> Value {
        AnthropicProvider
            .transform_request(&json!({"messages": input}), &deployment())
            .unwrap()["messages"]
            .clone()
    }

    fn empty_users() -> [Value; 3] {
        [
            json!({"role": "user"}),
            json!({"role": "user", "content": null}),
            json!({"role": "user", "content": []}),
        ]
    }

    fn control(ttl: &str) -> Value {
        json!({"type": "ephemeral", "ttl": ttl})
    }

    fn placeholder(control: Option<Value>) -> Value {
        let mut block = json!({"type": "text", "text": EMPTY_TEXT_PLACEHOLDER});
        if let Some(control) = control {
            block["cache_control"] = control;
        }
        json!({"role": "user", "content": [block]})
    }

    fn call() -> Value {
        json!({"role": "assistant", "tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": "lookup", "arguments": "{}"},
        }]})
    }

    fn native_call() -> Value {
        json!({"role": "assistant", "content": [{
            "type": "tool_use", "id": "call_1", "name": "lookup", "input": {},
        }]})
    }

    fn result() -> Value {
        json!({"role": "tool", "tool_call_id": "call_1", "content": "found"})
    }

    fn native_result() -> Value {
        json!({"role": "user", "content": [{
            "type": "tool_result", "tool_use_id": "call_1", "content": "found",
        }]})
    }

    fn output_config() -> Value {
        json!({"format": {"type": "json_schema", "schema": {
            "type": "object", "properties": {"value": {"type": "string", "minLength": 3}},
            "required": ["value"], "additionalProperties": false,
        }}, "effort": "low", "native_option": {"keep": true}})
    }

    #[test]
    fn marked_missing_null_and_empty_arrays_survive_before_and_after_user_text() {
        for mut empty in empty_users() {
            empty["cache_control"] = control("1h");
            let text = json!({"role": "user", "content": "real content"});
            for before in [true, false] {
                let (input, expected) = if before {
                    (
                        vec![empty.clone(), text.clone()],
                        json!([placeholder(Some(control("1h"))), text]),
                    )
                } else {
                    (
                        vec![text.clone(), empty.clone()],
                        json!([text, placeholder(Some(control("1h")))]),
                    )
                };
                assert_eq!(messages(input), expected, "before={before}, empty={empty}");
            }
        }
    }

    #[test]
    fn all_empty_runs_keep_every_mark_without_an_extra_unmarked_placeholder() {
        let mut input = Vec::new();
        let mut expected = Vec::new();
        for (index, mut empty) in empty_users().into_iter().enumerate() {
            let cc = json!({"type": "ephemeral", "ttl": "1h", "native_option": index});
            empty["cache_control"] = cc.clone();
            input.push(json!({"role": "user", "content": null}));
            input.push(empty);
            expected.push(placeholder(Some(cc)));
        }
        // A trailing unmarked turn must not replace all preceding breakpoints.
        input.push(json!({"role": "user", "content": []}));
        assert_eq!(messages(input), json!(expected));
    }

    #[test]
    fn unmarked_and_null_marked_empty_siblings_add_no_text() {
        let text = json!({"role": "user", "content": "real content"});
        let mut input = empty_users().to_vec();
        input.push(json!({"role": "user", "content": [], "cache_control": null}));
        assert_eq!(messages(input.clone()), json!([placeholder(None)]));
        input.insert(2, text.clone());
        assert_eq!(messages(input), json!([text]));
    }

    #[test]
    fn later_correlated_result_suppresses_marks_but_an_earlier_result_does_not() {
        for mut empty in empty_users() {
            empty["cache_control"] = control("1h");
            assert_eq!(
                messages(vec![call(), empty.clone(), result()]),
                json!([native_call(), native_result()]),
            );
            assert_eq!(
                messages(vec![call(), result(), empty.clone()]),
                json!([
                    native_call(),
                    native_result(),
                    placeholder(Some(control("1h")))
                ]),
            );
            // A native tool_result is the same correlated result, not text.
            assert_eq!(
                messages(vec![native_call(), empty.clone(), native_result()]),
                json!([native_call(), native_result()]),
            );
            let mut orphan = result();
            orphan["tool_call_id"] = json!("unknown_call");
            let mut native_orphan = native_result();
            native_orphan["content"][0]["tool_use_id"] = json!("unknown_call");
            assert_eq!(
                messages(vec![call(), empty, orphan]),
                json!([
                    native_call(),
                    native_orphan,
                    placeholder(Some(control("1h")))
                ]),
                "an unrelated/orphaned result must not erase a caller's breakpoint",
            );
        }
    }

    #[test]
    fn multiple_results_are_stable_and_only_later_unconsumed_ids_suppress_marks() {
        let mut assistant = native_call();
        assistant["content"].as_array_mut().unwrap().push(json!({
            "type": "tool_use", "id": "call_2", "name": "lookup", "input": {},
        }));
        let mut result2 = result();
        result2["tool_call_id"] = json!("call_2");
        let mut native_result2 = native_result();
        native_result2["content"][0]["tool_use_id"] = json!("call_2");
        let marked = json!({"role": "user", "content": [], "cache_control": control("1h")});
        let first = json!({"role": "user", "content": "first"});
        let last = json!({"role": "user", "content": "last"});
        assert_eq!(
            messages(vec![
                assistant.clone(),
                first.clone(),
                result2,
                marked.clone(),
                last.clone(),
                result(),
                marked.clone(),
                result()
            ]),
            json!([
                assistant,
                native_result2,
                native_result(),
                native_result(),
                first,
                last,
                placeholder(Some(control("1h")))
            ]),
            "the duplicate result after the last mark is not a new correlation",
        );
    }

    #[test]
    fn a_later_assistant_even_if_empty_ends_the_suppression_span() {
        for boundary in [json!({"role": "assistant", "content": null}), call()] {
            let marked = json!({"role": "user", "cache_control": control("1h")});
            let mut expected = vec![native_call(), placeholder(Some(control("1h")))];
            if boundary.get("tool_calls").is_some() {
                expected.push(native_call());
                expected.push(native_result());
            } else {
                // The dropped assistant still ends correlation with call_1.
                expected.push(native_result());
                expected.push(placeholder(Some(control("1h"))));
            }
            assert_eq!(
                messages(vec![call(), marked.clone(), boundary, marked, result()]),
                json!(expected),
            );
        }
    }

    #[test]
    fn kept_mid_conversation_system_messages_end_user_runs() {
        let marked = json!({"role": "user", "content": [], "cache_control": control("1h")});
        let system = json!({"role": "system", "content": "new instruction"});
        let mut dep = deployment();
        dep.oxllm_params.model = "anthropic/claude-sonnet-5-5".into();
        let body = AnthropicProvider
            .transform_request(
                &json!({"messages": [call(), marked, system, result()]}),
                &dep,
            )
            .unwrap();
        assert_eq!(
            body["messages"],
            json!([
                native_call(),
                placeholder(Some(control("1h"))),
                system,
                native_result()
            ]),
        );
    }

    #[test]
    fn tool_results_precede_ordinary_text_without_reordering_text_or_signed_thinking() {
        let signed =
            json!({"type": "thinking", "thinking": " \t\r\n", "signature": "opaque\u{0000}sig"});
        let redacted = json!({"type": "redacted_thinking", "data": "AA=="});
        let assistant = json!({"role": "assistant", "content": [signed, redacted,
            {"type": "tool_use", "id": "call_1", "name": "lookup", "input": {}}
        ]});
        let text1 = json!({"role": "user", "content": "before result"});
        let text2 = json!({"role": "user", "content": "after result"});
        let marked = json!({"role": "user", "content": [], "cache_control": control("1h")});
        let body = messages(vec![
            assistant.clone(),
            text1.clone(),
            result(),
            marked,
            text2.clone(),
        ]);
        assert_eq!(
            body,
            json!([
                assistant,
                native_result(),
                text1,
                placeholder(Some(control("1h"))),
                text2
            ]),
        );
        assert_eq!(
            serde_json::to_vec(&body[0]["content"]).unwrap(),
            serde_json::to_vec(&assistant["content"]).unwrap(),
            "signed thinking and redacted blocks remain byte-for-byte identical",
        );
    }

    #[test]
    fn native_mixed_user_blocks_put_all_results_before_ordinary_blocks() {
        let first = json!({"type": "text", "text": "before", "native_option": 1});
        let last = json!({"type": "text", "text": "after", "native_option": 2});
        let mixed =
            json!({"role": "user", "content": [first, native_result()["content"][0], last]});
        assert_eq!(
            messages(vec![call(), mixed]),
            json!([native_call(), native_result(), {"role": "user", "content": [first, last]}]),
        );
    }

    #[test]
    fn empty_turn_output_configuration_survives_every_placeholder_path() {
        for mut empty in empty_users() {
            for marked in [false, true] {
                empty["output_config"] = output_config();
                if marked {
                    empty["cache_control"] = control("1h");
                }
                let mut expected = placeholder(marked.then(|| control("1h")));
                expected["output_config"] = output_config();
                assert_eq!(messages(vec![empty.clone()]), json!([expected]));
                let text = json!({"role": "user", "content": "real content"});
                assert_eq!(
                    messages(vec![empty.clone(), text.clone()]),
                    json!([expected, text]),
                );
                assert_eq!(
                    messages(vec![call(), result(), empty.clone()]),
                    json!([native_call(), native_result(), expected]),
                );
            }
        }
    }

    #[test]
    fn output_configuration_that_cannot_survive_suppression_is_a_clear_request_error() {
        for mut empty in empty_users() {
            empty["output_config"] = output_config();
            for marked in [false, true] {
                if marked {
                    empty["cache_control"] = control("1h");
                }
                let error = AnthropicProvider
                    .transform_request(
                        &json!({"messages": [call(), empty, result()]}),
                        &deployment(),
                    )
                    .expect_err("must not silently erase message-level native output constraints");
                assert!(matches!(error, oxllm_core::Error::BadRequest(_)));
                assert!(error.to_string().contains("output_config"), "{error}");
                assert!(error.to_string().contains("tool_result"), "{error}");
            }
        }
        for role in ["assistant", "system", "developer"] {
            let error = AnthropicProvider
                .transform_request(
                    &json!({"messages": [{"role": role, "content": null, "output_config": output_config()}]}),
                    &deployment(),
                )
                .expect_err("dropping a non-user empty turn must not erase its output_config");
            assert!(matches!(error, oxllm_core::Error::BadRequest(_)));
            assert!(
                error.to_string().contains("output_config"),
                "{role}: {error}"
            );
        }
    }

    #[test]
    fn splitting_mixed_results_and_text_must_not_duplicate_native_output_constraints() {
        let mixed = json!({"role": "user", "output_config": output_config(), "content": [
            {"type": "text", "text": "before"}, native_result()["content"][0],
        ]});
        let error = AnthropicProvider
            .transform_request(&json!({"messages": [call(), mixed]}), &deployment())
            .expect_err(
                "splitting one configured source message cannot silently invent two constraints",
            );
        assert!(matches!(error, oxllm_core::Error::BadRequest(_)));
        assert!(error.to_string().contains("output_config"), "{error}");
    }

    #[test]
    fn review_output_config_events_cannot_cross_a_tool_result_partition() {
        let mut low = output_config();
        low["native_option"] = json!({"source": "text"});
        let mut high = low.clone();
        high["native_option"] = json!({"source": "result"});
        for (text_config, result_config) in [
            (
                Some(json!({"effort": "low"})),
                Some(json!({"effort": "high"})),
            ),
            (
                Some(json!({"effort": "low"})),
                Some(json!({"effort": "low"})),
            ),
            (Some(low), Some(high)),
            (Some(json!({"native_option": {"source": "text"}})), None),
            (None, Some(json!({"native_option": {"source": "result"}}))),
        ] {
            for native in [false, true] {
                let mut text = json!({"role": "user", "content": "before result"});
                let mut tool = if native { native_result() } else { result() };
                if let Some(config) = &text_config {
                    text["output_config"] = config.clone();
                }
                if let Some(config) = &result_config {
                    tool["output_config"] = config.clone();
                }
                let request = json!({"messages": [call(), text, tool]});
                let original = request.clone();
                let error = AnthropicProvider
                    .transform_request(&request, &deployment())
                    .expect_err(
                        "partitioning must not move even a format-less output_config event",
                    );
                assert!(matches!(error, oxllm_core::Error::BadRequest(_)));
                assert!(error.to_string().contains("output_config"), "{error}");
                assert!(error.to_string().contains("reorder"), "{error}");
                assert_eq!(request, original);
            }
        }
    }

    #[test]
    fn review_output_config_events_that_do_not_cross_the_partition_still_survive() {
        let mut text = json!({"role": "user", "content": "after result"});
        text["output_config"] = output_config();
        let mut tool = result();
        tool["output_config"] = json!({"effort": "high", "native_option": true});
        let mut expected_tool = native_result();
        expected_tool["output_config"] = tool["output_config"].clone();
        assert_eq!(
            messages(vec![call(), tool, text.clone()]),
            json!([native_call(), expected_tool, text]),
        );
        let unconfigured = json!({"role": "user", "content": "before result"});
        assert_eq!(
            messages(vec![call(), unconfigured.clone(), result(), text.clone()]),
            json!([native_call(), native_result(), unconfigured, text]),
            "an untouched later configuration does not forbid unconfigured reordering",
        );
    }

    #[test]
    fn rewriting_blank_text_preserves_every_caller_block_key() {
        for text in [Value::Null, json!(""), json!(" \t\r\n"), json!(7)] {
            for block_mark in [false, true] {
                let mut block = json!({"type": "text", "text": text,
                    "citations": [{"type": "char_location", "start_char_index": 0}],
                    "native_option": {"nested": [null, 1, "keep"]},
                });
                let mut message = json!({"role": "user"});
                if block_mark {
                    block["cache_control"] = control("1h");
                } else {
                    message["cache_control"] = control("1h");
                }
                message["content"] = json!([block]);
                block["text"] = json!(EMPTY_TEXT_PLACEHOLDER);
                block["cache_control"] = control("1h");
                assert_eq!(
                    messages(vec![message]),
                    json!([{"role": "user", "content": [block]}])
                );
            }
        }
        // Even without a cache mark, block-owned keys must not be lost by
        // collapsing a rewritten text block to a bare string.
        let input = json!({"role": "user", "content": [{"type": "text", "text": "", "native_option": true}]});
        assert_eq!(
            messages(vec![input]),
            json!([{"role": "user", "content": [{"type": "text", "text": EMPTY_TEXT_PLACEHOLDER, "native_option": true}]}]),
        );
    }

    #[test]
    fn long_runs_keep_directional_suppression_and_all_later_breakpoints() {
        // Exercise both worst-case suffix shapes: many marked empties before
        // a result, and many marked empties with no later result. Normalizing
        // this run must not scan the remaining suffix again for each message.
        const LEN: usize = 16_384;
        let marked = json!({"role": "user", "content": [], "cache_control": control("1h")});
        let mut input = Vec::with_capacity(2 * LEN + 2);
        input.push(call());
        input.extend(std::iter::repeat_n(marked.clone(), LEN));
        input.push(result());
        input.extend(std::iter::repeat_n(marked, LEN));
        let output = messages(input);
        let output = output.as_array().unwrap();
        assert_eq!(output.len(), LEN + 2);
        assert_eq!(output[0], native_call());
        assert_eq!(output[1], native_result());
        for message in &output[2..] {
            assert_eq!(*message, placeholder(Some(control("1h"))));
        }
    }
}

#[test]
fn validation_schema_wrappers_are_rejected_but_model_support_is_upstream_owned() {
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
    let forwarded = AnthropicProvider
        .transform_request(
            &json!({
                "messages": [{"role": "user", "content": "json"}],
                "response_format": {"type": "json_schema", "json_schema": {
                    "strict": true, "schema": {"type": "object", "properties": {}}
                }},
            }),
            &legacy,
        )
        .unwrap();
    assert_eq!(
        forwarded["output_config"]["format"]["schema"],
        json!({"type":"object", "properties":{}})
    );
    assert!(forwarded.get("tools").is_none());
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
