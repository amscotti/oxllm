use super::*;
use oxllm_core::deployment::OxllmParams;

fn deployment(model: &str, converse: bool) -> Deployment {
    let mut extra = Map::new();
    extra.insert("use_converse_api".into(), json!(converse));
    Deployment {
        oxllm_params: OxllmParams {
            model: format!("bedrock/{model}"),
            extra,
            ..Default::default()
        },
        ..Default::default()
    }
}

#[test]
fn strict_tools_are_forwarded_in_invoke_and_converse_without_schema_rewrites() {
    let schema = json!({"type":"object","additionalProperties":true,"properties":{"data":{"type":"array","uniqueItems":true,"items":{"oneOf":[{"type":"string"},{"type":"integer"}]}}}});
    for converse in [false, true] {
        let body=BedrockProvider.transform_request(&json!({"messages":[],"tools":[{"type":"function","function":{"name":"extract","strict":true,"parameters":schema}}]}),&deployment("anthropic.claude-haiku-5-5",converse)).unwrap();
        let tool = if converse {
            &body["toolConfig"]["tools"][0]["toolSpec"]
        } else {
            &body["tools"][0]
        };
        assert_eq!(tool["strict"], true);
        let forwarded = if converse {
            &tool["inputSchema"]["json"]
        } else {
            &tool["input_schema"]
        };
        assert_eq!(forwarded, &schema);
    }
}

#[test]
fn explicit_thinking_rejects_unsatisfiable_limits_on_both_native_apis() {
    for converse in [false, true] {
        for max_tokens in [100, 1000, 1024] {
            for budget in [json!(2000), json!(2000.0)] {
                let request = json!({"messages":[],"max_tokens":max_tokens,"thinking":{"type":"enabled","budget_tokens":budget}});
                let error = BedrockProvider
                    .transform_request(
                        &request,
                        &deployment("anthropic.claude-haiku-4-5", converse),
                    )
                    .unwrap_err();
                assert!(matches!(error, oxllm_core::Error::BadRequest(_)));
                assert!(error.to_string().contains("thinking.budget_tokens"));
                assert!(error.to_string().contains("max_tokens"));
            }
        }
    }
}

#[test]
fn converse_forwards_disabled_thinking_with_a_stale_budget_for_provider_validation() {
    let thinking = json!({"type":"disabled","budget_tokens":2000});
    let body = BedrockProvider
        .transform_request(
            &json!({"messages":[],"max_tokens":100,"thinking":thinking}),
            &deployment("anthropic.claude-haiku-4-5", true),
        )
        .unwrap();
    assert_eq!(body["additionalModelRequestFields"]["thinking"], thinking);
    assert_eq!(body["inferenceConfig"]["maxTokens"], 100);
}

#[test]
fn converse_budget_window_applies_after_native_thinking_aliases_are_merged() {
    for alias in [
        "thinking",
        "additionalModelRequestFields",
        "additional_model_request_fields",
    ] {
        for limit in ["max_tokens", "max_completion_tokens"] {
            for value in [json!(100), json!(100.0)] {
                let mut request = json!({"messages":[],"reasoning_effort":"high"});
                request[limit] = value;
                let thinking = json!({"type":"enabled","budget_tokens":2000});
                request[alias] = if alias == "thinking" {
                    thinking
                } else {
                    json!({"thinking":thinking})
                };
                assert!(matches!(
                    BedrockProvider.transform_request(
                        &request,
                        &deployment("anthropic.claude-haiku-4-5", true)
                    ),
                    Err(oxllm_core::Error::BadRequest(_))
                ));
            }
        }
    }
}

#[test]
fn converse_clamps_a_possible_window_and_omits_only_inferred_impossible_thinking() {
    let mut request =
        json!({"messages":[],"max_tokens":1500,"thinking":{"type":"enabled","budget_tokens":2000}});
    let configured = deployment("anthropic.claude-haiku-4-5", true);
    let body = BedrockProvider
        .transform_request(&request, &configured)
        .unwrap();
    assert_eq!(
        body["additionalModelRequestFields"]["thinking"]["budget_tokens"],
        1499
    );
    assert_eq!(body["inferenceConfig"]["maxTokens"], 1500);
    request.as_object_mut().unwrap().remove("thinking");
    request["max_tokens"] = json!(100);
    request["reasoning_effort"] = json!("high");
    request["additionalModelRequestFields"] = json!({"independent_option":true});
    let body = BedrockProvider
        .transform_request(&request, &configured)
        .unwrap();
    assert!(
        body.pointer("/additionalModelRequestFields/thinking")
            .is_none()
    );
    assert_eq!(
        body["additionalModelRequestFields"]["independent_option"],
        true
    );
    assert_eq!(body["inferenceConfig"]["maxTokens"], 100);
}

#[test]
fn converse_uses_the_effective_native_thinking_after_alias_precedence() {
    let configured = deployment("anthropic.claude-haiku-4-5", true);
    for thinking in [json!({"type":"disabled"}), json!({"type":"adaptive"})] {
        let request = json!({"messages":[],"max_tokens":100,"reasoning_effort":"high",
            "thinking":{"type":"enabled","budget_tokens":2000},
            "additionalModelRequestFields":{"thinking":{"type":"enabled","budget_tokens":3000}},
            "additional_model_request_fields":{"thinking":thinking}});
        let body = BedrockProvider
            .transform_request(&request, &configured)
            .unwrap();
        assert_eq!(body["additionalModelRequestFields"]["thinking"], thinking);
        assert_eq!(body["inferenceConfig"]["maxTokens"], 100);
    }
    let request = json!({"messages":[],"max_tokens":3000,"reasoning_effort":"high",
        "thinking":{"type":"disabled"},
        "additional_model_request_fields":{"thinking":{"type":"enabled","budget_tokens":2000}}});
    let body = BedrockProvider
        .transform_request(&request, &configured)
        .unwrap();
    assert_eq!(
        body["additionalModelRequestFields"]["thinking"]["budget_tokens"],
        2000
    );
    assert_eq!(body["inferenceConfig"]["maxTokens"], 3000);
}

#[test]
fn native_messages_output_schema_maps_to_converse_without_leaking_format_alias() {
    let schema = json!({"type":"object","additionalProperties":false,"properties":{"data":{"type":"string"}}});
    let body=BedrockProvider.transform_request(&json!({"messages":[],"output_config":{"format":{"type":"json_schema","schema":schema},"effort":"low"}}),&deployment("anthropic.claude-haiku-5-5",true)).unwrap();
    assert_eq!(
        body["outputConfig"]["textFormat"]["structure"]["jsonSchema"]["schema"],
        schema.to_string()
    );
    assert!(body["outputConfig"].get("format").is_none());
    assert_eq!(body["outputConfig"]["effort"], "low");
    assert!(body.get("toolConfig").is_none());
}

#[test]
fn converse_native_tool_specs_reject_misplaced_strict_metadata() {
    let request = json!({"messages":[],"tools":[{"toolSpec":{"name":"extract","inputSchema":{"json":{"type":"object","strict":true}}}}]});
    assert!(matches!(
        BedrockProvider
            .transform_request(&request, &deployment("anthropic.claude-haiku-5-5", true)),
        Err(oxllm_core::Error::BadRequest(_))
    ));
}

#[test]
fn converse_native_null_strict_is_omitted_like_canonical_function_null() {
    let body=BedrockProvider.transform_request(&json!({"messages":[],"tools":[{"toolSpec":{"name":"extract","strict":null,"inputSchema":{"json":{"type":"object","additionalProperties":false}}}}]}),&deployment("anthropic.claude-haiku-5-5",true)).unwrap();
    assert!(
        body["toolConfig"]["tools"][0]["toolSpec"]
            .get("strict")
            .is_none()
    );
}

#[test]
fn converse_camel_format_alias_is_translated_without_an_unknown_native_member() {
    let schema = json!({"type":"object","additionalProperties":false});
    let body=BedrockProvider.transform_request(&json!({"messages":[],"outputConfig":{"format":{"type":"json_schema","schema":schema},"effort":"low"}}),&deployment("anthropic.claude-haiku-5-5",true)).unwrap();
    assert_eq!(
        body["outputConfig"]["textFormat"]["structure"]["jsonSchema"]["schema"],
        schema.to_string()
    );
    assert!(body["outputConfig"].get("format").is_none());
    assert_eq!(body["outputConfig"]["effort"], "low");
}

#[test]
fn converse_must_not_discard_unrepresentable_native_format_options() {
    let request = json!({"messages":[],"output_config":{"format":{"type":"json_schema","schema":{"type":"object"},"vendor_option":true}}});
    assert!(matches!(
        BedrockProvider
            .transform_request(&request, &deployment("anthropic.claude-haiku-5-5", true)),
        Err(oxllm_core::Error::BadRequest(_))
    ));
}

#[test]
fn converse_conflicting_format_definitions_are_not_silently_overwritten() {
    let format =
        json!({"type":"json_schema","schema":{"type":"object","additionalProperties":false}});
    for (fields, message) in [
        (
            json!({"response_format":{"type":"text"},"outputConfig":{"format":format}}),
            "text conflicts with outputConfig.format",
        ),
        (
            json!({"response_format":{"type":"json_schema","json_schema":{"schema":{"type":"object"}}},"output_config":{"format":format}}),
            "response_format conflicts with output_config.format",
        ),
        (
            json!({"output_config":{"format":format},"outputConfig":{"format":{"type":"json_schema","schema":{"type":"object"}}}}),
            "output_config.format conflicts with outputConfig.format",
        ),
        (
            json!({"outputConfig":{"format":format,"textFormat":{"type":"json_schema","structure":{"jsonSchema":{"schema":"{}"}}}}}),
            "conflict",
        ),
        (
            json!({"response_format":{"type":"text"},"outputConfig":{"textFormat":{"type":"json_schema","structure":{"jsonSchema":{"schema":"{}"}}}}}),
            "conflict",
        ),
        (json!({"outputConfig":[]}), "outputConfig must be an object"),
    ] {
        let error = BedrockProvider
            .transform_request(&fields, &deployment("anthropic.claude-haiku-5-5", true))
            .unwrap_err();
        assert!(matches!(error, oxllm_core::Error::BadRequest(_)));
        assert!(error.to_string().contains(message), "{fields}: {error}");
    }
}

#[test]
fn converse_null_native_text_format_is_absent_not_a_schema_override() {
    let schema = json!({"type":"object","additionalProperties":false});
    let body=BedrockProvider.transform_request(&json!({"messages":[],"outputConfig":{"format":{"type":"json_schema","schema":schema},"textFormat":null,"effort":"low"}}),&deployment("anthropic.claude-haiku-5-5",true)).unwrap();
    assert_eq!(
        body["outputConfig"]["textFormat"]["structure"]["jsonSchema"]["schema"],
        schema.to_string()
    );
    assert_eq!(body["outputConfig"]["effort"], "low");
}

#[test]
fn converse_unsupported_strict_choices_are_not_replaced_by_parallel_auto_defaults() {
    for choice in [
        json!({"type":"allowed_tools","mode":"required","tools":[{"type":"function","name":"extract"}]}),
        json!({"type":"function","function":{"name":"extract","native_option":true}}),
        json!({"type":"auto","native_option":true}),
    ] {
        let request = json!({"messages":[],"tools":[{"type":"function","function":{"name":"extract","strict":true,"parameters":{"type":"object","additionalProperties":false}}}],"tool_choice":choice,"parallel_tool_calls":false});
        let error = BedrockProvider
            .transform_request(&request, &deployment("anthropic.claude-haiku-5-5", true))
            .unwrap_err();
        assert!(matches!(error, oxllm_core::Error::BadRequest(_)));
        assert!(error.to_string().contains("tool choice"));
    }
}

#[test]
fn converse_native_tool_choices_and_aliases_cannot_disagree() {
    for fields in [
        json!({"additionalModelRequestFields":{"tool_choice":{"type":"any"}},"additional_model_request_fields":{"tool_choice":{"type":"auto"}}}),
        json!({"tool_choice":"required","additionalModelRequestFields":{"tool_choice":{"type":"auto"}}}),
        json!({"parallel_tool_calls":false,"additionalModelRequestFields":{"tool_choice":{"type":"tool","name":"extract","disable_parallel_tool_use":false}}}),
    ] {
        let mut request = json!({"messages":[],"tools":[{"type":"function","function":{"name":"extract","strict":true,"parameters":{"type":"object","additionalProperties":false}}}]});
        request
            .as_object_mut()
            .unwrap()
            .extend(fields.as_object().unwrap().clone());
        let error = BedrockProvider
            .transform_request(&request, &deployment("anthropic.claude-haiku-5-5", true))
            .unwrap_err();
        assert!(matches!(error, oxllm_core::Error::BadRequest(_)));
        assert!(error.to_string().contains("conflict"), "{error}");
    }
}

#[test]
fn converse_equivalent_native_and_canonical_choices_keep_native_options() {
    for serial in [false, true] {
        let choice = json!({"type":"any","native_option":true});
        let mut request = json!({"messages":[],"tools":[{"type":"function","function":{"name":"extract","strict":true,"parameters":{"type":"object","additionalProperties":false}}}],"tool_choice":"required","additionalModelRequestFields":{"tool_choice":choice},"additional_model_request_fields":{"tool_choice":choice}});
        if serial {
            request["parallel_tool_calls"] = json!(false);
        }
        let body = BedrockProvider
            .transform_request(&request, &deployment("anthropic.claude-haiku-5-5", true))
            .unwrap();
        let mut expected = choice.clone();
        if serial {
            expected["disable_parallel_tool_use"] = json!(true);
        }
        assert_eq!(
            body["additionalModelRequestFields"]["tool_choice"],
            expected
        );
        assert!(body["toolConfig"].get("toolChoice").is_none());
    }
}

#[test]
fn converse_native_choices_share_the_tool_name_encoding() {
    for canonical in [false, true] {
        for alias in [
            "additionalModelRequestFields",
            "additional_model_request_fields",
        ] {
            let mut request = json!({"messages":[],"tools":[{"type":"function","function":{"name":"my.tool","strict":true,"parameters":{"type":"object"}}}],"parallel_tool_calls":false});
            request[alias] = json!({"tool_choice":{"type":"tool","name":"my.tool"}});
            if canonical {
                request["tool_choice"] = json!({"type":"function","function":{"name":"my.tool"}});
            }
            let body = BedrockProvider
                .transform_request(&request, &deployment("anthropic.claude-haiku-5-5", true))
                .unwrap();
            assert_eq!(
                body["toolConfig"]["tools"][0]["toolSpec"]["name"],
                "my_tool"
            );
            assert_eq!(
                body["additionalModelRequestFields"]["tool_choice"]["name"],
                "my_tool"
            );
        }
    }
}

#[test]
fn converse_choice_alias_conflicts_are_errors_even_without_tools() {
    for request in [
        json!({"messages":[],"additionalModelRequestFields":{"tool_choice":{"type":"auto"}},"additional_model_request_fields":{"tool_choice":{"type":"none"}}}),
        json!({"messages":[],"tool_choice":"none","additionalModelRequestFields":{"tool_choice":{"type":"any"}}}),
    ] {
        assert!(matches!(
            BedrockProvider
                .transform_request(&request, &deployment("anthropic.claude-haiku-5-5", true)),
            Err(oxllm_core::Error::BadRequest(_))
        ));
    }
}

#[test]
fn converse_direct_schema_annotations_are_not_wrapper_metadata() {
    let schema = json!({"type":"object","description":"Schema annotation","properties":{}});
    let body = BedrockProvider
        .transform_request(
            &json!({"messages":[],"response_format":{"type":"json_schema","json_schema":schema}}),
            &deployment("anthropic.claude-haiku-5-5", true),
        )
        .unwrap();
    let wire = &body["outputConfig"]["textFormat"]["structure"]["jsonSchema"];
    assert_eq!(wire["schema"], schema.to_string());
    assert!(wire.get("description").is_none());
}

#[test]
fn converse_native_schema_strings_can_supply_json_mode_and_preserve_their_encoding() {
    for alias in ["outputConfig", "output_config"] {
        let format = json!({"type":"json_schema","structure":{"jsonSchema":{"schema":"{ \"type\": \"object\" }"}}});
        for generic_mode in [false, true] {
            let mut request = json!({"messages":[],"response_format":if generic_mode {json!({"type":"json_object"})} else {json!({"type":"json_schema","json_schema":{"schema":{"type":"object"}}})}});
            request[alias] = json!({"textFormat":format});
            let body = BedrockProvider
                .transform_request(&request, &deployment("anthropic.claude-haiku-5-5", true))
                .unwrap();
            assert_eq!(body["outputConfig"]["textFormat"], format);
        }
    }
}

#[test]
fn converse_schema_label_defaults_merge_without_reserializing_the_native_schema() {
    let schema = "{ \"type\": \"object\" }";
    let body=BedrockProvider.transform_request(&json!({"messages":[],"response_format":{"type":"json_schema","json_schema":{"name":"record","description":"Label","schema":{"type":"object"}}},"outputConfig":{"textFormat":{"type":"json_schema","structure":{"jsonSchema":{"schema":schema}}}}}),&deployment("anthropic.claude-haiku-5-5",true)).unwrap();
    assert_eq!(
        body["outputConfig"]["textFormat"]["structure"]["jsonSchema"],
        json!({"name":"record","description":"Label","schema":schema})
    );
}

#[test]
fn converse_empty_native_tool_inputs_are_arguments_not_json_response_text() {
    let request = json!({"model":"anthropic.claude-haiku-5-5","tools":[{"type":"function","function":{"name":"json_tool_call","strict":true,"parameters":{"type":"object","properties":{},"additionalProperties":false}}}]});
    let mut state = None;
    let start=BedrockProvider.transform_stream_chunk_stateful(&json!({"contentBlockIndex":0,"start":{"toolUse":{"toolUseId":"call-empty","name":"json_tool_call"}}}).to_string(),&mut state,&request).unwrap().unwrap();
    assert_eq!(
        start["choices"][0]["delta"]["tool_calls"][0]["function"]["name"],
        "json_tool_call"
    );
    let stop = BedrockProvider
        .transform_stream_chunk_stateful(
            &json!({"contentBlockIndex":0}).to_string(),
            &mut state,
            &request,
        )
        .unwrap()
        .unwrap();
    assert_eq!(
        stop["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"],
        "{}"
    );
    assert!(stop["choices"][0]["delta"].get("content").is_none());
}

#[test]
fn validation_converse_native_schema_has_the_aws_wire_shape_for_each_family() {
    for model in ["anthropic.claude-sonnet-4-5", "zai.glm-5", "deepseek.v3.2"] {
        let request = json!({
            "model": "public-alias", "messages": [{"role": "user", "content": "extract"}],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "extraction", "description": "Extract an answer", "strict": true,
                "schema": {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}
            }},
        });
        let body = BedrockProvider
            .transform_request(&request, &deployment(model, true))
            .unwrap();
        assert!(body.get("model").is_none());
        assert!(body.get("response_format").is_none());
        assert!(body.get("toolConfig").is_none());
        let expected_schema = request["response_format"]["json_schema"]["schema"].clone();
        assert_eq!(
            body["outputConfig"],
            json!({"textFormat": {
                "type": "json_schema", "structure": {"jsonSchema": {
                    "name": "extraction", "description": "Extract an answer", "schema": expected_schema.to_string()
                }}
            }})
        );
        assert_eq!(
            request["response_format"]["json_schema"]["schema"].get("additionalProperties"),
            None
        );
    }
}

#[test]
fn validation_converse_schema_survives_effort_alias_merging() {
    let body = BedrockProvider.transform_request(&json!({
        "messages": [{"role": "user", "content": "extract"}],
        "response_format": {"type": "json_schema", "json_schema": {"schema": {"type": "object", "properties": {}}}},
        "output_config": {"effort": "xhigh"},
    }), &deployment("anthropic.claude-opus-4-5", true)).unwrap();
    assert_eq!(body["outputConfig"]["textFormat"]["type"], "json_schema");
    // Effort is a documented Converse OutputConfig member: it must survive
    // verbatim alongside textFormat, not be relocated or clamped.
    assert_eq!(body["outputConfig"]["effort"], "xhigh");
    assert!(
        body.get("additionalModelRequestFields")
            .and_then(|fields| fields.get("output_config"))
            .is_none()
    );
}

#[test]
fn validation_converse_json_schema_is_not_silently_ignored() {
    for model in [
        "meta.llama3-1-70b-instruct-v1:0",
        "amazon.nova-pro-v1:0",
        "anthropic.claude-3-5-sonnet",
    ] {
        let body = BedrockProvider.transform_request(&json!({
            "messages": [{"role": "user", "content": "extract"}],
            "response_format": {"type": "json_schema", "json_schema": {"schema": {"type": "object"}}},
        }), &deployment(model, true)).unwrap();
        assert_eq!(
            body["outputConfig"]["textFormat"]["structure"]["jsonSchema"]["schema"],
            "{\"type\":\"object\"}"
        );
        assert!(body.get("toolConfig").is_none());
    }
    for format in [
        json!({"type": "json_schema"}),
        json!({"type": "json_schema", "json_schema": {"name": "missing"}}),
    ] {
        assert!(matches!(
            BedrockProvider.transform_request(
                &json!({
                    "messages": [{"role": "user", "content": "extract"}], "response_format": format,
                }),
                &deployment("anthropic.claude-sonnet-4-5", true)
            ),
            Err(oxllm_core::Error::BadRequest(_))
        ));
    }
}

#[test]
fn validation_converse_developer_instructions_stay_in_system_blocks() {
    let body = BedrockProvider.transform_request(&json!({"messages": [
        {"role": "system", "content": "system rules"},
        {"role": "developer", "content": [{"type": "text", "text": "developer rules", "cache_control": {"type": "ephemeral"}}]},
        {"role": "user", "content": "question"},
    ]}), &deployment("anthropic.claude-sonnet-4-5", true)).unwrap();
    assert_eq!(
        body["system"],
        json!([
            {"text": "system rules"}, {"text": "developer rules"}, {"cachePoint": {"type": "default"}}
        ])
    );
    assert_eq!(
        body["messages"],
        json!([{ "role": "user", "content": [{"text": "question"}] }])
    );
}

#[test]
fn validation_converse_thinking_roundtrip_preserves_signed_and_redacted_blocks() {
    let content = json!([
        {"reasoningContent": {"reasoningText": {"text": "check", "signature": "exact-signature"}}},
        {"reasoningContent": {"redactedContent": "AA=="}},
        {"text": "Checking."},
        {"toolUse": {"toolUseId": "tool_1", "name": "lookup", "input": {"key": "x"}}},
    ]);
    let response = BedrockProvider
        .transform_response(
            &json!({
                "output": {"message": {"role": "assistant", "content": content}},
                "stopReason": "tool_use", "usage": {"inputTokens": 3, "outputTokens": 8},
            }),
            "anthropic.claude-sonnet-4-5",
        )
        .unwrap();
    let body = BedrockProvider
        .transform_request(
            &json!({"messages": [
                {"role": "user", "content": "look up x"}, response["choices"][0]["message"],
                {"role": "tool", "tool_call_id": "tool_1", "content": "result"},
            ]}),
            &deployment("anthropic.claude-sonnet-4-5", true),
        )
        .unwrap();
    assert_eq!(body["messages"][1]["content"], content);
    assert_eq!(
        body["messages"][2]["content"][0]["toolResult"]["toolUseId"],
        "tool_1"
    );

    let native = BedrockProvider
        .transform_request(
            &json!({"messages": [{
                "role": "assistant", "content": content,
                "thinking_blocks": response["choices"][0]["message"]["thinking_blocks"],
            }]}),
            &deployment("anthropic.claude-sonnet-4-5", true),
        )
        .unwrap();
    assert_eq!(native["messages"][0]["content"], content);
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
    BedrockProvider
        .transform_request(
            &json!({"messages": [
                {"role": "user", "content": "check"}, assistant,
                {"role": "tool", "tool_call_id": "call_1", "content": "found"},
            ]}),
            &deployment("anthropic.claude-sonnet-4-5", true),
        )
        .unwrap()
}

#[test]
fn thinking_replay_regression_inline_empty_beats_missing_mirror() {
    let body = replay_with_inline_thinking("", None);
    assert_eq!(
        body["messages"][1]["content"],
        json!([
            {"reasoningContent": {"reasoningText": {"text": "", "signature": "original-signature"}}},
            {"toolUse": {"toolUseId": "call_1", "name": "lookup", "input": {}}},
        ])
    );
    assert_eq!(
        body["messages"][2]["content"][0]["toolResult"]["toolUseId"],
        "call_1"
    );
}

#[test]
fn thinking_replay_regression_inline_text_beats_refilled_mirror() {
    let body = replay_with_inline_thinking("plan", Some("plan"));
    assert_eq!(
        body["messages"][1]["content"],
        json!([
            {"reasoningContent": {"reasoningText": {"text": "plan", "signature": "original-signature"}}},
            {"toolUse": {"toolUseId": "call_1", "name": "lookup", "input": {}}},
        ])
    );
    assert_eq!(
        body["messages"][2]["content"][0]["toolResult"]["toolUseId"],
        "call_1"
    );
}

#[test]
fn thinking_replay_regression_explicit_signed_blank_is_not_refilled() {
    for text in ["", " \t\r\n", "signed text"] {
        let body = BedrockProvider
            .transform_request(
                &json!({"messages": [{
                    "role": "assistant", "content": "answer",
                    "thinking_blocks": [{"type": "thinking", "thinking": text, "signature": "original-signature"}],
                    "reasoning_content": "different unsigned text",
                }]}),
                &deployment("anthropic.claude-sonnet-4-5", true),
            )
            .unwrap();
        assert_eq!(
            body["messages"][0]["content"][0],
            json!({
                "reasoningContent": {"reasoningText": {"text": text, "signature": "original-signature"}},
            })
        );
    }
}

#[test]
fn thinking_replay_converse_inline_normalization_preserves_block_order_and_malformed_types() {
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
        let mut assistant = json!({
            "role": "assistant", "content": [
                first, {"type": "redacted_thinking", "data": "AA=="},
                {"type": "thinking", "thinking": "second", "signature": "second-sig"},
            ],
            "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}],
            "reasoning_content": "not either signed block",
        });
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
        let body = BedrockProvider
            .transform_request(
                &json!({"messages": [assistant]}),
                &deployment("anthropic.claude-sonnet-4-5", true),
            )
            .unwrap();
        let expected_text = text.filter(|text| !text.is_null()).unwrap_or(json!(""));
        assert_eq!(
            body["messages"][0]["content"],
            json!([
                {"reasoningContent": {"reasoningText": {"text": expected_text, "signature": "sig"}}},
                {"reasoningContent": {"redactedContent": "AA=="}},
                {"reasoningContent": {"reasoningText": {"text": "second", "signature": "second-sig"}}},
                {"toolUse": {"toolUseId": "call_1", "name": "lookup", "input": {}}},
            ])
        );
    }
}

#[test]
fn thinking_replay_converse_native_content_is_authoritative_for_signed_and_redacted_mirrors() {
    for text in ["", " \t\r\n", "plan"] {
        let content = json!([
            {"reasoningContent": {"reasoningText": {"text": text, "signature": "sig"}}},
            {"reasoningContent": {"redactedContent": "AA=="}},
            {"toolUse": {"toolUseId": "call_1", "name": "lookup", "input": {}}},
        ]);
        let body = BedrockProvider.transform_request(&json!({"messages": [{
            "role": "assistant", "content": content, "reasoning_content": "different text",
            "thinking_blocks": [
                {"type": "thinking", "thinking": "conflicting mirror", "signature": "sig"},
                {"type": "redacted_thinking", "data": "AA==", "cache_control": {"type": "ephemeral"}},
            ],
        }]}), &deployment("anthropic.claude-sonnet-4-5", true)).unwrap();
        assert_eq!(body["messages"][0]["content"], content);
    }
}

#[test]
fn validation_invoke_claude_response_restores_tool_names_on_another_thread() {
    let request = json!({
        "model": "alias", "messages": [{"role": "user", "content": "lookup"}],
        "tools": [{"type": "function", "function": {"name": "get.weather", "parameters": {"type": "object"}}}],
    });
    BedrockProvider
        .transform_request(&request, &deployment("anthropic.claude-sonnet-4-5", false))
        .unwrap();
    let response = std::thread::spawn(move || {
        BedrockProvider.transform_response_with_request(&json!({
        "id": "msg_tool", "model": "anthropic.claude-sonnet-4-5", "stop_reason": "tool_use",
        "content": [{"type": "tool_use", "id": "tool_1", "name": "get_weather", "input": {}}],
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }), "anthropic.claude-sonnet-4-5", &request).unwrap()
    })
    .join()
    .unwrap();
    assert_eq!(
        response["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
        "get.weather"
    );
}

#[test]
fn validation_converse_rejects_nonbase64_image_and_document_data_urls() {
    for block in [
        json!({"type": "image_url", "image_url": {"url": "data:image/png,%89PNG"}}),
        json!({"type": "image_url", "image_url": {"url": "data:application/pdf,%25PDF"}}),
        json!({"type": "file", "file": {"file_data": "data:application/pdf,%25PDF"}}),
        json!({"type": "file", "file": {"file_id": "file-unresolvable"}}),
    ] {
        assert!(matches!(
            BedrockProvider.transform_request(
                &json!({"messages": [{
                    "role": "user", "content": [{"type": "text", "text": "read this"}, block]
                }]}),
                &deployment("anthropic.claude-sonnet-4-5", true)
            ),
            Err(oxllm_core::Error::BadRequest(_))
        ));
    }
    let body = BedrockProvider.transform_request(&json!({"messages": [{
        "role": "user", "content": [{"type": "file", "file": {"file_data": "data:application/pdf;charset=utf-8;base64,JVBERi0="}}]
    }]}), &deployment("anthropic.claude-sonnet-4-5", true)).unwrap();
    assert_eq!(
        body["messages"][0]["content"][0]["document"]["source"]["bytes"],
        "JVBERi0="
    );
}

#[test]
fn validation_invoke_prompt_adapters_keep_array_text_and_developer_turns() {
    for (model, field) in [
        ("amazon.titan-text-express-v1", "inputText"),
        ("meta.llama3-8b-instruct-v1:0", "prompt"),
        ("mistral.mistral-7b-instruct-v0:2", "prompt"),
        ("ai21.j2-ultra-v1", "prompt"),
        ("cohere.command-text-v14", "prompt"),
        ("qwen.imported-chat", "prompt"),
    ] {
        let body = BedrockProvider.transform_request(&json!({"messages": [
            {"role": "developer", "content": [{"type": "text", "text": "rules"}]},
            {"role": "user", "content": [{"type": "text", "text": "first "}, {"type": "text", "text": "second"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "reply"}]},
        ]}), &deployment(model, false)).unwrap();
        let prompt = body[field].as_str().unwrap();
        assert!(prompt.contains("rules"), "{model}");
        assert!(prompt.contains("first second"), "{model}");
        assert!(prompt.find("rules").unwrap() < prompt.find("first second").unwrap());
        assert!(prompt.find("first second").unwrap() < prompt.find("reply").unwrap());
    }
}

#[test]
fn validation_invoke_stop_strings_use_vendor_array_wire_fields() {
    for (model, field) in [
        (
            "amazon.titan-text-express-v1",
            "/textGenerationConfig/stopSequences",
        ),
        ("mistral.mistral-7b-instruct-v0:2", "/stop"),
    ] {
        let body = BedrockProvider
            .transform_request(
                &json!({
                    "messages": [{"role": "user", "content": "hello"}], "stop": "END",
                }),
                &deployment(model, false),
            )
            .unwrap();
        assert_eq!(body.pointer(field), Some(&json!(["END"])));
    }
}

#[test]
fn validation_invoke_prompt_adapters_reject_untranslatable_history() {
    for message in [
        json!({"role": "tool", "tool_call_id": "tool_1", "content": "result"}),
        json!({"role": "function", "name": "lookup", "content": "result"}),
        json!({"role": "assistant", "content": null, "tool_calls": [{"id": "tool_1", "function": {"name": "lookup", "arguments": "{}"}}]}),
        json!({"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]}),
    ] {
        for model in [
            "amazon.titan-text-express-v1",
            "meta.llama3-8b-instruct-v1:0",
            "qwen.imported-chat",
        ] {
            assert!(matches!(
                BedrockProvider
                    .transform_request(&json!({"messages": [message]}), &deployment(model, false)),
                Err(oxllm_core::Error::BadRequest(_))
            ));
        }
    }
}

#[test]
fn validation_real_aws_exception_frame_reaches_provider_error_handling() {
    // Independently generated using Python struct/zlib and Smithy's string
    // header layout. This fixture includes both CRCs and exception headers.
    let frame = b"\x00\x00\x00g\x00\x00\x00@\x38\xec\x35\xa4\x0d:message-type\x07\x00\x09exception\x0f:exception-type\x07\x00\x13throttlingException{\"message\":\"slow down\"}\x46\xf2\x43\xd2";
    let (consumed, payload) = decode_eventstream_frame(frame).unwrap();
    assert_eq!(consumed, frame.len());
    assert_eq!(payload, br#"{"message":"slow down"}"#);
    let sse = format!("data: {}\n\n", std::str::from_utf8(payload).unwrap());
    let mut state = None;
    let error = BedrockProvider
        .transform_stream_chunk_stateful(&sse, &mut state, &json!({}))
        .unwrap_err();
    // The host payload-only contract loses the exception subtype; never
    // pretend this message-only payload completed successfully.
    assert!(
        matches!(error, oxllm_core::Error::Upstream { status: 500, message } if message == "slow down")
    );
}

#[test]
fn validation_named_converse_exceptions_preserve_their_status() {
    for (name, status) in [
        ("throttlingException", 429),
        ("validationException", 400),
        ("modelStreamErrorException", 424),
        ("internalServerException", 500),
        ("serviceUnavailableException", 503),
    ] {
        let raw = json!({name: {"message": "upstream failed"}}).to_string();
        let mut state = None;
        for result in [
            BedrockProvider.transform_stream_chunk(&raw),
            BedrockProvider.transform_stream_chunk_stateful(&raw, &mut state, &json!({})),
            converse::transform_stream_chunk_converse(&raw),
        ] {
            assert!(
                matches!(result, Err(oxllm_core::Error::Upstream { status: actual, .. }) if actual == status),
                "{name}"
            );
        }
    }
}

#[test]
fn validation_converse_stream_identity_is_stable_across_interleaving_and_threads() {
    let request = json!({"model": "public-bedrock-alias"});
    let mut state = None;
    let start = BedrockProvider
        .transform_stream_chunk_stateful(r#"{"role":"assistant"}"#, &mut state, &request)
        .unwrap()
        .unwrap();
    let mut other_state = None;
    let other = BedrockProvider
        .transform_stream_chunk_stateful(
            r#"{"role":"assistant"}"#,
            &mut other_state,
            &json!({"model": "other-model"}),
        )
        .unwrap()
        .unwrap();
    assert_ne!(start["id"], other["id"]);
    let (mut state, delta) = std::thread::spawn(move || {
        let delta = BedrockProvider.transform_stream_chunk_stateful(
            "event: delta\rdata: {\"contentBlockIndex\":0,\rdata: \"delta\":{\"text\":\"hello\"}}\r\r",
            &mut state, &request,
        ).unwrap().unwrap();
        (state, delta)
    }).join().unwrap();
    let usage = BedrockProvider
        .transform_stream_chunk_stateful(
            r#"{"metadata":{"usage":{"inputTokens":1,"outputTokens":2,"totalTokens":3}}}"#,
            &mut state,
            &json!({"model": "public-bedrock-alias"}),
        )
        .unwrap()
        .unwrap();
    for chunk in [&delta, &usage] {
        assert_eq!(chunk["id"], start["id"]);
        assert_eq!(chunk["model"], "public-bedrock-alias");
        assert_eq!(chunk["created"], start["created"]);
        assert!(chunk["created"].as_u64().unwrap() > 0);
    }
    assert_eq!(delta["choices"][0]["delta"]["content"], "hello");
    assert_eq!(usage["choices"], json!([]));
}

#[test]
fn validation_converse_explicit_json_named_tool_stream_remains_tool_calls() {
    let request = json!({"model": "amazon.nova-pro-v1:0", "messages": [{"role": "user", "content": "json"}],
        "tools":[{"type":"function", "function":{"name":"json_tool_call", "strict":true, "parameters":{"type":"object", "additionalProperties":false}}}],
        "tool_choice":{"type":"function", "function":{"name":"json_tool_call"}}});
    let body = BedrockProvider
        .transform_request(&request, &deployment("amazon.nova-pro-v1:0", true))
        .unwrap();
    assert_eq!(
        body["toolConfig"]["toolChoice"],
        json!({"tool": {"name": "json_tool_call"}})
    );
    let mut state = None;
    let events = [
        json!({"role": "assistant"}),
        json!({"contentBlockIndex": 0, "start": {"toolUse": {"toolUseId": "json_1", "name": "json_tool_call"}}}),
        json!({"contentBlockIndex": 0, "delta": {"toolUse": {"input": "{\"answer\":"}}}),
        json!({"contentBlockIndex": 0, "delta": {"toolUse": {"input": "42}"}}}),
        json!({"contentBlockIndex": 0}),
        json!({"stopReason": "tool_use"}),
    ];
    let mut text = String::new();
    let mut finish = Value::Null;
    for event in events {
        if let Some(chunk) = BedrockProvider
            .transform_stream_chunk_stateful(&event.to_string(), &mut state, &request)
            .unwrap()
        {
            let delta = &chunk["choices"][0]["delta"];
            assert!(delta.get("content").is_none());
            text.push_str(
                delta["tool_calls"][0]["function"]["arguments"]
                    .as_str()
                    .unwrap_or(""),
            );
            if !chunk["choices"][0]["finish_reason"].is_null() {
                finish = chunk["choices"][0]["finish_reason"].clone();
            }
        }
    }
    assert_eq!(
        serde_json::from_str::<Value>(&text).unwrap(),
        json!({"answer": 42})
    );
    assert_eq!(finish, "tool_calls");
}

#[test]
fn validation_real_tools_named_json_tool_call_are_not_consumed_as_json_mode() {
    let request = json!({"model": "amazon.nova-pro-v1:0", "tools": [{
        "type": "function", "function": {"name": "json_tool_call", "parameters": {"type": "object"}}
    }]});
    let mut state = None;
    let chunk = BedrockProvider.transform_stream_chunk_stateful(&json!({
        "contentBlockIndex": 0, "start": {"toolUse": {"toolUseId": "real_tool", "name": "json_tool_call"}}
    }).to_string(), &mut state, &request).unwrap().unwrap();
    assert_eq!(
        chunk["choices"][0]["delta"]["tool_calls"][0]["id"],
        "real_tool"
    );
    let response = BedrockProvider
        .transform_response_with_request(
            &json!({
                "output": {"message": {"role": "assistant", "content": [{"toolUse": {
                    "toolUseId": "real_tool", "name": "json_tool_call", "input": {"key": "value"}
                }}]}}, "stopReason": "tool_use", "usage": {},
            }),
            "amazon.nova-pro-v1:0",
            &request,
        )
        .unwrap();
    assert_eq!(
        response["choices"][0]["message"]["tool_calls"][0]["id"],
        "real_tool"
    );
    assert_eq!(response["choices"][0]["finish_reason"], "tool_calls");
}

#[test]
fn validation_converse_stream_thinking_block_is_replayable_after_stop() {
    let request = json!({"model": "anthropic.claude-sonnet-4-5"});
    let mut state = None;
    for event in [
        json!({"role": "assistant"}),
        json!({"contentBlockIndex": 0, "delta": {"reasoningContent": {"text": "check "}}}),
        json!({"contentBlockIndex": 0, "delta": {"reasoningContent": {"text": "data", "signature": "sig-"}}}),
        json!({"contentBlockIndex": 0, "delta": {"reasoningContent": {"signature": "value"}}}),
    ] {
        BedrockProvider
            .transform_stream_chunk_stateful(&event.to_string(), &mut state, &request)
            .unwrap();
    }
    let stop = BedrockProvider
        .transform_stream_chunk_stateful(r#"{"contentBlockIndex":0}"#, &mut state, &request)
        .unwrap()
        .unwrap();
    let thinking = &stop["choices"][0]["delta"]["thinking_blocks"];
    let body = BedrockProvider
        .transform_request(
            &json!({"messages": [{
                "role": "assistant", "content": "answer", "thinking_blocks": thinking,
            }]}),
            &deployment("anthropic.claude-sonnet-4-5", true),
        )
        .unwrap();
    assert_eq!(
        body["messages"][0]["content"][0],
        json!({"reasoningContent": {"reasoningText": {"text": "check data", "signature": "sig-value"}}})
    );
}

#[test]
fn validation_fragmented_redacted_thinking_errors_instead_of_losing_bytes() {
    let mut state = None;
    let request = json!({"model": "anthropic.claude-sonnet-4-5"});
    BedrockProvider
        .transform_stream_chunk_stateful(
            r#"{"contentBlockIndex":0,"delta":{"reasoningContent":{"redactedContent":"AA=="}}}"#,
            &mut state,
            &request,
        )
        .unwrap();
    assert!(matches!(BedrockProvider.transform_stream_chunk_stateful(
        r#"{"contentBlockIndex":0,"delta":{"reasoningContent":{"redactedContent":"AQ=="}}}"#,
        &mut state, &request,
    ), Err(oxllm_core::Error::Transform(message)) if message.contains("redactedContent")));
}

#[test]
fn validation_converse_top_k_uses_family_specific_native_fields() {
    for model in [
        "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-5-sonnet-20241022-v2:0",
    ] {
        let body = BedrockProvider
            .transform_request(
                &json!({"messages": [{"role": "user", "content": "hi"}], "top_k": 20}),
                &deployment(model, true),
            )
            .unwrap();
        assert_eq!(body["additionalModelRequestFields"], json!({"top_k": 20}));
        assert!(body["inferenceConfig"].get("topK").is_none());
    }
    for model in [
        "amazon.nova-micro-v1:0",
        "amazon.nova-lite-v1:0",
        "amazon.nova-pro-v1:0",
        "amazon.nova-premier-v1:0",
        "global.amazon.nova-pro-v1:0",
        "arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.amazon.nova-pro-v1:0",
    ] {
        let body = BedrockProvider
            .transform_request(
                &json!({"messages": [{"role": "user", "content": "hi"}], "top_k": 20}),
                &deployment(model, true),
            )
            .unwrap();
        assert_eq!(
            body["additionalModelRequestFields"],
            json!({"inferenceConfig": {"topK": 20}})
        );
        assert!(body["inferenceConfig"].get("topK").is_none());
        assert!(body["additionalModelRequestFields"].get("top_k").is_none());
    }
}

#[test]
fn validation_converse_top_k_preserves_explicit_native_values_and_siblings() {
    for alias in [
        "additionalModelRequestFields",
        "additional_model_request_fields",
    ] {
        for (model, native, expected) in [
            (
                "anthropic.claude-3-5-sonnet",
                json!({"top_k": 7, "metadata": {"user_id": "caller"}}),
                json!({"top_k": 7, "metadata": {"user_id": "caller"}}),
            ),
            (
                "amazon.nova-pro-v1:0",
                json!({"inferenceConfig": {"topK": 7, "maxTokens": 64}}),
                json!({"inferenceConfig": {"topK": 7, "maxTokens": 64}}),
            ),
            (
                "amazon.nova-pro-v1:0",
                json!({"inferenceConfig": {"maxTokens": 64}}),
                json!({"inferenceConfig": {"topK": 20, "maxTokens": 64}}),
            ),
        ] {
            let mut request = json!({"messages": [], "top_k": 20});
            request[alias] = native;
            let body = BedrockProvider
                .transform_request(&request, &deployment(model, true))
                .unwrap();
            assert_eq!(body["additionalModelRequestFields"], expected);
        }
    }
    assert!(matches!(
        BedrockProvider.transform_request(
            &json!({"messages": [], "top_k": 20, "additionalModelRequestFields": {"inferenceConfig": false}}),
            &deployment("amazon.nova-pro-v1:0", true),
        ),
        Err(oxllm_core::Error::BadRequest(_))
    ));
}

#[test]
fn validation_converse_top_k_rejects_unverified_models_and_malformed_values() {
    for model in [
        "meta.llama3-1-70b-instruct-v1:0",
        "amazon.titan-text-express-v1",
        "amazon.nova-2-lite-v1:0",
        "amazon.nova-canvas-v1:0",
        "unknown.claude-model",
        "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/anthropic.claude-profile",
    ] {
        assert!(matches!(
            BedrockProvider.transform_request(&json!({"messages": [], "top_k": 20}), &deployment(model, true)),
            Err(oxllm_core::Error::BadRequest(message)) if message.contains("top_k")
        ));
        assert!(
            BedrockProvider
                .transform_request(
                    &json!({"messages": [], "top_k": null}),
                    &deployment(model, true)
                )
                .is_ok()
        );
    }
    for value in [json!(-1), json!(0.5), json!("20"), json!({}), json!([])] {
        assert!(matches!(
            BedrockProvider.transform_request(
                &json!({"messages": [], "top_k": value}),
                &deployment("anthropic.claude-3-5-sonnet", true)
            ),
            Err(oxllm_core::Error::BadRequest(_))
        ));
    }
}

#[test]
fn validation_serial_tools_share_known_family_detection_for_ids_and_arns() {
    let request = json!({
        "messages": [], "parallel_tool_calls": false,
        "tools": [{"type": "function", "function": {"name": "get.weather", "parameters": {"type": "object"}}}],
        "tool_choice": {"type": "function", "function": {"name": "get.weather"}}
    });
    for model in [
        "anthropic.claude-3-5-sonnet",
        "bedrock/us-east-1.anthropic.claude-3-5-sonnet",
        "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-5-sonnet-20241022-v2:0",
        "arn:aws-us-gov:bedrock:us-gov-west-1:123456789012:inference-profile/us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    ] {
        assert_eq!(detect_bedrock_provider(model), BedrockVendor::Anthropic);
        let body = converse::transform_request_converse(&request, model, &Map::new()).unwrap();
        assert_eq!(
            body["additionalModelRequestFields"]["tool_choice"],
            json!({
                "type": "tool", "name": "get_weather", "disable_parallel_tool_use": true
            })
        );
        assert!(body["toolConfig"].get("toolChoice").is_none());
    }
    for model in [
        "claude-production",
        "unknown.anthropic",
        "amazon.nova-pro-v1:0",
        "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/anthropic.claude-profile",
        "arn:aws:bedrock:us-east-1:123456789012:inference-profile/claude-production",
        "arn:aws:bedrock:us-east-1:123456789012:provisioned-model/anthropic.claude-production",
        "arn:aws:bedrock:ap-southeast-2:123456789012:application-inference-profile/au.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "arn:aws:bedrock:ap-northeast-1:123456789012:custom-model/jp.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "arn:aws:bedrock:ap-southeast-2:123456789012:provisioned-model/au.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "arn:aws:bedrock:ap-northeast-1:123456789012:imported-model/jp.anthropic.claude-sonnet-4-5-20250929-v1:0",
    ] {
        assert_ne!(detect_bedrock_provider(model), BedrockVendor::Anthropic);
        assert!(matches!(
            converse::transform_request_converse(&request, model, &Map::new()),
            Err(oxllm_core::Error::BadRequest(message)) if message.contains("parallel_tool_calls")
        ));
    }
}

#[test]
fn validation_opaque_profiles_reject_reasoning_without_an_underlying_model_hint() {
    let model = "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/claude-name-is-not-evidence";
    for request in [
        json!({"messages": [], "reasoning_effort": "high"}),
        json!({"messages": [], "thinking": {"type": "enabled", "budget_tokens": 1024}}),
        json!({"messages": [], "additionalModelRequestFields": {"thinking": {"type": "enabled", "budget_tokens": 1024}}}),
    ] {
        assert!(matches!(
            BedrockProvider.transform_request(&request, &deployment(model, true)),
            Err(oxllm_core::Error::BadRequest(message)) if message.contains("base_model")
        ));
    }
    let body = BedrockProvider
        .transform_request(
            &json!({"messages": [{"role": "user", "content": "hi"}], "temperature": 0.4}),
            &deployment(model, true),
        )
        .unwrap();
    assert!(body.get("additionalModelRequestFields").is_none());
    assert_eq!(body["inferenceConfig"]["temperature"], 0.4);
}

#[test]
fn validation_converse_hosted_tools_are_rejected_instead_of_becoming_functions() {
    for kind in [
        "computer_use_preview",
        "computer_20250124",
        "bash_20250124",
        "text_editor_20250124",
        "web_search_20250305",
        "tool_search_tool_regex_20251119",
        "advisor_20260301",
    ] {
        for tool in [
            json!({"type": kind, "name": "hosted"}),
            json!({"type": kind, "function": {"name": "hosted", "parameters": {"type": "object"}}}),
            json!({"toolSpec": {"type": kind, "name": "hosted", "inputSchema": {"json": {"type": "object"}}}}),
        ] {
            assert!(matches!(
                BedrockProvider.transform_request(&json!({"messages": [], "tools": [tool]}), &deployment("anthropic.claude-sonnet-4-5", true)),
                Err(oxllm_core::Error::BadRequest(message)) if message.contains(kind) && message.contains("InvokeModel")
            ));
        }
    }
}

#[test]
fn validation_converse_native_tool_union_keeps_valid_shapes_and_rejects_missing_schema() {
    let tools = json!([
        {"toolSpec": {"name": "lookup", "inputSchema": {"json": {"type": "object"}}}},
        {"systemTool": {"name": "nova_grounding"}},
        {"cachePoint": {"type": "default"}}
    ]);
    let body = BedrockProvider
        .transform_request(
            &json!({"messages": [{"role": "user", "content": "look up a fact"}], "tools": tools}),
            &deployment("amazon.nova-2-lite-v1:0", true),
        )
        .unwrap();
    assert_eq!(body["toolConfig"]["tools"], tools);
    for tool in [
        json!({"toolSpec": {"name": "lookup"}}),
        json!({"toolSpec": {"name": "lookup", "inputSchema": false}}),
        json!({"toolSpec": {"name": "lookup", "inputSchema": {"json": []}}}),
        json!({"toolSpec": {"name": "lookup", "inputSchema": {"json": {"type": "array"}}}}),
        json!({"type": "function", "function": {"name": "lookup", "parameters": {"type": "array"}}}),
        json!({"toolSpec": {"name": "lookup", "inputSchema": {"json": {}}, "display_width_px": 1024}}),
        json!({"toolSpec": {"name": "x".repeat(65), "inputSchema": {"json": {}}}}),
        json!({"systemTool": {"name": "nova_grounding"}, "cachePoint": {"type": "default"}}),
    ] {
        assert!(matches!(
            BedrockProvider.transform_request(
                &json!({"messages": [], "tools": [tool]}),
                &deployment("amazon.nova-2-lite-v1:0", true)
            ),
            Err(oxllm_core::Error::BadRequest(_))
        ));
    }
}

#[test]
fn validation_hosted_tools_keep_native_invoke_shapes_and_explicit_beta_selectors() {
    let request = json!({
        "messages": [{"role": "user", "content": "use these tools"}],
        "anthropic_beta": ["computer-use-2025-01-24"],
        "tools": [
            {"type": "computer_20250124", "function": {"name": "computer", "parameters": {
                "display_width_px": 1024, "display_height_px": 768
            }}},
            {"type": "bash_20250124", "name": "bash"},
            {"type": "text_editor_20250124", "name": "str_replace_editor"}
        ]
    });
    let configured = deployment("anthropic.claude-3-7-sonnet-20250219-v1:0", false);
    let body = BedrockProvider
        .transform_request(&request, &configured)
        .unwrap();
    assert_eq!(body["anthropic_beta"], request["anthropic_beta"]);
    assert_eq!(
        body["tools"],
        json!([
            {"type": "computer_20250124", "name": "computer", "display_width_px": 1024, "display_height_px": 768},
            {"type": "bash_20250124", "name": "bash"},
            {"type": "text_editor_20250124", "name": "str_replace_editor"}
        ])
    );
    assert!(body.get("toolConfig").is_none());
    assert!(body.get("model").is_none());
    assert!(
        BedrockProvider
            .endpoint_url(&configured)
            .ends_with("/invoke")
    );
    for beta in [
        json!(true),
        json!("computer-use-2025-01-24"),
        json!([false]),
        json!([""]),
    ] {
        let mut malformed = request.clone();
        malformed["anthropic_beta"] = beta;
        assert!(matches!(
            BedrockProvider.transform_request(&malformed, &configured),
            Err(oxllm_core::Error::BadRequest(_))
        ));
    }
}

fn serial_profile_request() -> Value {
    json!({
        "model": "public-alias", "messages": [{"role": "user", "content": "look up the weather"}],
        "tools": [{"type": "function", "function": {"name": "get.weather", "parameters": {"type": "object"}}}],
        "tool_choice": {"type": "function", "function": {"name": "get.weather"}},
        "parallel_tool_calls": false
    })
}

fn set_profile_hint(deployment: &mut Deployment, hint: &str, metadata: bool) {
    if metadata {
        deployment.model_info = Some(oxllm_core::deployment::ModelInfo {
            base_model: Some(hint.into()),
            ..Default::default()
        });
    } else {
        deployment
            .oxllm_params
            .extra
            .insert("base_model".into(), json!(hint));
    }
}

#[test]
fn validation_opaque_claude_profiles_use_hints_for_serial_tools_without_replacing_the_arn() {
    for arn in [
        "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/profile123",
        "arn:aws:bedrock:us-east-1:123456789012:inference-profile/opaque123",
    ] {
        for metadata in [false, true] {
            for hint in [
                "anthropic.claude-3-7-sonnet-20250219-v1:0",
                "bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
                "bedrock/jp.anthropic.claude-3-7-sonnet-20250219-v1:0",
                "bedrock/au.anthropic.claude-3-7-sonnet-20250219-v1:0",
                "anthropic/claude-sonnet-4-5",
                "claude-sonnet-4-5",
            ] {
                let mut configured = deployment(arn, true);
                set_profile_hint(&mut configured, hint, metadata);
                let original = serde_json::to_value(&configured).unwrap();
                let request = serial_profile_request();
                let url = BedrockProvider.endpoint_url_for_request(&configured, &request);
                let body = BedrockProvider
                    .transform_request(&request, &configured)
                    .unwrap();
                assert_eq!(
                    body["additionalModelRequestFields"]["tool_choice"],
                    json!({
                        "type": "tool", "name": "get_weather", "disable_parallel_tool_use": true
                    })
                );
                assert!(body["toolConfig"].get("toolChoice").is_none());
                assert_eq!(
                    body["toolConfig"]["tools"][0]["toolSpec"]["inputSchema"]["json"]["type"],
                    "object"
                );
                assert!(body.get("model").is_none());
                assert!(body.get("base_model").is_none());
                assert_eq!(serde_json::to_value(&configured).unwrap(), original);
                assert_eq!(
                    BedrockProvider.endpoint_url_for_request(&configured, &request),
                    url
                );
                let segment = url
                    .split_once("/model/")
                    .unwrap()
                    .1
                    .strip_suffix("/converse")
                    .unwrap();
                assert_eq!(
                    percent_encoding::percent_decode_str(segment)
                        .decode_utf8()
                        .unwrap(),
                    arn
                );
            }
        }
    }
}

#[test]
fn validation_profile_hint_metadata_precedes_extra_and_does_not_change_auto_routing() {
    let arn = "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/profile123";
    let mut configured = deployment(arn, false);
    set_profile_hint(&mut configured, "anthropic/claude-sonnet-4-5", true);
    set_profile_hint(&mut configured, "amazon.nova-pro-v1:0", false);
    let body = BedrockProvider
        .transform_request(&serial_profile_request(), &configured)
        .unwrap();
    assert_eq!(
        body["additionalModelRequestFields"]["tool_choice"]["disable_parallel_tool_use"],
        true
    );
    assert!(
        BedrockProvider
            .endpoint_url(&configured)
            .ends_with("/converse")
    );

    configured.model_info.as_mut().unwrap().base_model = Some("public-alias".into());
    assert!(matches!(
        BedrockProvider.transform_request(&serial_profile_request(), &configured),
        Err(oxllm_core::Error::Config(message)) if message.contains("base_model")
    ));
}

#[test]
fn validation_claude_profile_hints_enable_top_k_thinking_and_signed_replay() {
    let arn = "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/profile123";
    for metadata in [false, true] {
        let mut configured = deployment(arn, true);
        set_profile_hint(
            &mut configured,
            "anthropic.claude-3-7-sonnet-20250219-v1:0",
            metadata,
        );
        let top_k = BedrockProvider
            .transform_request(
                &json!({"messages": [{"role": "user", "content": "hello"}], "top_k": 20}),
                &configured,
            )
            .unwrap();
        assert_eq!(top_k["additionalModelRequestFields"], json!({"top_k": 20}));
        let thinking = BedrockProvider.transform_request(
            &json!({"messages": [{"role": "user", "content": "think"}], "reasoning_effort": "high", "thinking": null, "temperature": 0.4}),
            &configured,
        ).unwrap();
        assert_eq!(
            thinking["additionalModelRequestFields"]["thinking"],
            json!({"type": "enabled", "budget_tokens": 10000})
        );
        assert_eq!(thinking["inferenceConfig"]["maxTokens"], 14096);
        assert!(thinking["inferenceConfig"].get("temperature").is_none());
        assert!(matches!(
            BedrockProvider.transform_request(&json!({"messages": [{"role": "assistant", "content": "reply", "reasoning_content": "unsigned"}]}), &configured),
            Err(oxllm_core::Error::BadRequest(_))
        ));
    }
}

#[test]
fn validation_nova_profile_hints_select_nova_fields_and_reject_claude_only_options() {
    let arn = "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/profile123";
    let mut configured = deployment(arn, true);
    set_profile_hint(&mut configured, "bedrock/au.amazon.nova-pro-v1:0", true);
    let body = BedrockProvider
        .transform_request(
            &json!({"messages": [{"role": "user", "content": "hello"}], "top_k": 20}),
            &configured,
        )
        .unwrap();
    assert_eq!(
        body["additionalModelRequestFields"],
        json!({"inferenceConfig": {"topK": 20}})
    );
    assert!(matches!(
        BedrockProvider.transform_request(&serial_profile_request(), &configured),
        Err(oxllm_core::Error::BadRequest(message)) if message.contains("parallel_tool_calls")
    ));
    assert!(matches!(
        BedrockProvider.transform_request(&json!({"messages": [], "thinking": {"type": "enabled", "budget_tokens": 1024}}), &configured),
        Err(oxllm_core::Error::BadRequest(message)) if message.contains("thinking")
    ));

    configured.model_info = None;
    set_profile_hint(&mut configured, "bedrock/jp.amazon.nova-2-lite-v1:0", false);
    let body = BedrockProvider.transform_request(
        &json!({"messages": [{"role": "user", "content": "think"}], "reasoning_effort": "medium"}), &configured,
    ).unwrap();
    assert_eq!(
        body["additionalModelRequestFields"],
        json!({"reasoningConfig": {"type": "enabled", "maxReasoningEffort": "medium"}})
    );
    let url = BedrockProvider.endpoint_url(&configured);
    let segment = url
        .split_once("/model/")
        .unwrap()
        .1
        .strip_suffix("/converse")
        .unwrap();
    assert_eq!(
        percent_encoding::percent_decode_str(segment)
            .decode_utf8()
            .unwrap(),
        arn
    );
}

#[test]
fn validation_opaque_profiles_reject_missing_invalid_and_recursive_model_hints() {
    let arn = "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/profile123";
    for request in [
        serial_profile_request(),
        json!({"messages": [], "top_k": 20}),
    ] {
        assert!(
            matches!(BedrockProvider.transform_request(&request, &deployment(arn, true)),
            Err(oxllm_core::Error::BadRequest(message)) if message.contains("base_model"))
        );
        for metadata in [false, true] {
            for hint in [
                "",
                "*",
                "anthropic/*",
                "public-alias",
                "claude-production",
                "amazon.nova",
                arn,
            ] {
                let mut configured = deployment(arn, true);
                set_profile_hint(&mut configured, hint, metadata);
                assert!(
                    matches!(BedrockProvider.transform_request(&request, &configured),
                    Err(oxllm_core::Error::Config(message)) if message.contains("base_model"))
                );
            }
        }
        for hint in [
            json!(true),
            json!(123),
            json!([]),
            json!({"model": "anthropic.claude-3-7-sonnet"}),
        ] {
            let mut configured = deployment(arn, true);
            configured
                .oxllm_params
                .extra
                .insert("base_model".into(), hint);
            assert!(
                matches!(BedrockProvider.transform_request(&request, &configured),
                Err(oxllm_core::Error::Config(message)) if message.contains("base_model"))
            );
        }
    }
}

#[test]
fn validation_known_runtime_models_and_arns_ignore_conflicting_profile_hints() {
    for runtime in [
        "anthropic.claude-3-7-sonnet-20250219-v1:0",
        "au.anthropic.claude-3-7-sonnet-20250219-v1:0",
        "jp.anthropic.claude-3-7-sonnet-20250219-v1:0",
        "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-7-sonnet-20250219-v1:0",
        "arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        "arn:aws:bedrock:ap-southeast-2::foundation-model/au.anthropic.claude-3-7-sonnet-20250219-v1:0",
        "arn:aws:bedrock:ap-northeast-1:123456789012:inference-profile/jp.anthropic.claude-3-7-sonnet-20250219-v1:0",
    ] {
        let mut configured = deployment(runtime, true);
        set_profile_hint(&mut configured, "amazon.nova-2-lite-v1:0", true);
        configured
            .oxllm_params
            .extra
            .insert("base_model".into(), json!(false));
        let body = BedrockProvider
            .transform_request(&serial_profile_request(), &configured)
            .unwrap();
        assert_eq!(
            body["additionalModelRequestFields"]["tool_choice"]["disable_parallel_tool_use"],
            true
        );
        assert_eq!(
            BedrockProvider.endpoint_url(&configured),
            BedrockProvider.endpoint_url(&deployment(runtime, true))
        );
    }
}

fn geo_prefixed_model_ids(base: &str) -> Vec<String> {
    ["us", "eu", "apac", "global", "au", "jp"]
        .into_iter()
        .flat_map(|geo| {
            let model = format!("{geo}.{base}");
            [
                model.clone(),
                format!("arn:aws:bedrock:ap-northeast-1::foundation-model/{model}"),
                format!("arn:aws:bedrock:ap-southeast-2:123456789012:inference-profile/{model}"),
            ]
        })
        .collect()
}

fn assert_geo_prefix_url(configured: &Deployment, request: &Value, model: &str, api: &str) {
    let expected = format!(
        "https://bedrock-runtime.us-east-1.amazonaws.com/model/{}/{api}",
        model.replace('/', "%2F")
    );
    assert_eq!(BedrockProvider.endpoint_url(configured), expected);
    assert_eq!(
        BedrockProvider.endpoint_url_for_request(configured, request),
        expected
    );
    assert_eq!(configured.oxllm_params.model, format!("bedrock/{model}"));
}

fn assert_geo_prefix_nova2_request(explicit_converse: bool) {
    let request = json!({
        "model": "public-alias", "messages": [{"role": "user", "content": "think and look up the weather"}],
        "max_tokens": 2048, "reasoning_effort": "medium", "web_search_options": {},
        "tools": [{"type": "function", "function": {"name": "get.weather", "parameters": {"type": "object"}}}],
        "tool_choice": "auto"
    });
    for model in geo_prefixed_model_ids("amazon.nova-2-lite-v1:0") {
        let configured = deployment(&model, explicit_converse);
        let body = BedrockProvider
            .transform_request(&request, &configured)
            .unwrap_or_else(|error| panic!("{model}: {error}"));
        assert_geo_prefix_url(&configured, &request, &model, "converse");
        assert_eq!(
            body["messages"][0]["content"][0]["text"],
            request["messages"][0]["content"]
        );
        assert_eq!(body["inferenceConfig"]["maxTokens"], 2048);
        assert_eq!(
            body["additionalModelRequestFields"],
            json!({"reasoningConfig": {"type": "enabled", "maxReasoningEffort": "medium"}}),
            "{model}"
        );
        assert_eq!(
            body["toolConfig"]["tools"][0]["toolSpec"]["name"],
            "get_weather"
        );
        assert_eq!(
            body["toolConfig"]["tools"][1],
            json!({"systemTool": {"name": "nova_grounding"}})
        );
        assert_eq!(body["toolConfig"]["toolChoice"], json!({"auto": {}}));
        assert!(body.get("model").is_none());
        assert!(body.get("anthropic_version").is_none());
        assert!(body.get("reasoning_effort").is_none());
        assert_eq!(detect_bedrock_provider(&model), BedrockVendor::Nova);
    }
}

#[test]
fn geo_prefix_nova2_default_route_maps_reasoning_and_tools() {
    assert_geo_prefix_nova2_request(false);
}

#[test]
fn geo_prefix_nova2_explicit_converse_maps_reasoning_and_tools() {
    assert_geo_prefix_nova2_request(true);
}

fn assert_geo_prefix_claude_request(explicit_converse: bool) {
    let request = json!({
        "model": "public-alias", "messages": [
            {"role": "system", "content": [{"type": "text", "text": "instructions", "cache_control": {"type": "ephemeral", "ttl": "1h"}}]},
            {"role": "user", "content": "look up the weather"}
        ],
        "max_tokens": 4096, "temperature": 0.4, "reasoning_effort": "low", "top_k": 20,
        "tools": [{"type": "function", "function": {"name": "get.weather", "parameters": {"type": "object"}}, "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
        "tool_choice": "auto", "parallel_tool_calls": false
    });
    for model in geo_prefixed_model_ids("anthropic.claude-sonnet-4-5-20250929-v1:0") {
        let configured = deployment(&model, explicit_converse);
        let body = BedrockProvider
            .transform_request(&request, &configured)
            .unwrap_or_else(|error| panic!("{model}: {error}"));
        assert_geo_prefix_url(
            &configured,
            &request,
            &model,
            if explicit_converse {
                "converse"
            } else {
                "invoke"
            },
        );
        let fields = if explicit_converse {
            assert_eq!(body["inferenceConfig"]["maxTokens"], 4096);
            assert!(body["inferenceConfig"].get("temperature").is_none());
            assert!(body["inferenceConfig"].get("topK").is_none());
            assert_eq!(
                body["toolConfig"]["tools"][0]["toolSpec"]["name"],
                "get_weather"
            );
            assert!(body["toolConfig"].get("toolChoice").is_none());
            assert_eq!(
                body["system"][1],
                json!({"cachePoint": {"type": "default", "ttl": "1h"}})
            );
            // OpenAI tool cache_control is not a Converse-native tool cache point.
            assert_eq!(body["toolConfig"]["tools"].as_array().unwrap().len(), 1);
            assert!(
                body["toolConfig"]["tools"][0]
                    .get("cache_control")
                    .is_none()
            );
            assert!(body.get("anthropic_version").is_none());
            &body["additionalModelRequestFields"]
        } else {
            assert_eq!(body["anthropic_version"], ANTHROPIC_VERSION);
            assert_eq!(body["max_tokens"], 4096);
            assert!(body.get("temperature").is_none());
            assert_eq!(body["tools"][0]["name"], "get_weather");
            assert_eq!(
                body["tools"][0]["cache_control"],
                request["tools"][0]["cache_control"]
            );
            assert_eq!(
                body["system"][0]["cache_control"],
                request["messages"][0]["content"][0]["cache_control"]
            );
            assert!(body.get("toolConfig").is_none());
            &body
        };
        assert_eq!(
            fields["thinking"],
            json!({"type": "enabled", "budget_tokens": 1024}),
            "{model}"
        );
        assert_eq!(fields["top_k"], 20);
        assert_eq!(
            fields["tool_choice"],
            json!({"type": "auto", "disable_parallel_tool_use": true})
        );
        assert!(body.get("model").is_none());
        assert_eq!(detect_bedrock_provider(&model), BedrockVendor::Anthropic);
        assert_eq!(
            BedrockProvider::uses_anthropic_messages_api(&request, &configured),
            !explicit_converse
        );
    }
}

#[test]
fn geo_prefix_claude_default_invoke_maps_reasoning_serial_tools_and_cache() {
    assert_geo_prefix_claude_request(false);
}

#[test]
fn geo_prefix_claude_explicit_converse_maps_reasoning_serial_tools_and_cache() {
    assert_geo_prefix_claude_request(true);
}

#[test]
fn geo_prefix_nova_top_k_keeps_version_specific_support_with_and_without_tools() {
    for explicit_converse in [false, true] {
        for tools in [json!([]), serial_profile_request()["tools"].clone()] {
            let request = json!({"messages": [{"role": "user", "content": "hello"}], "top_k": 20, "tools": tools});
            for model in geo_prefixed_model_ids("amazon.nova-pro-v1:0") {
                let configured = deployment(&model, explicit_converse);
                let body = BedrockProvider
                    .transform_request(&request, &configured)
                    .unwrap();
                assert_geo_prefix_url(&configured, &request, &model, "converse");
                assert_eq!(
                    body["additionalModelRequestFields"],
                    json!({"inferenceConfig": {"topK": 20}})
                );
                assert!(body["inferenceConfig"].get("topK").is_none());
            }
            for model in geo_prefixed_model_ids("amazon.nova-2-lite-v1:0") {
                let configured = deployment(&model, explicit_converse);
                let result = BedrockProvider.transform_request(&request, &configured);
                assert_geo_prefix_url(&configured, &request, &model, "converse");
                assert!(
                    matches!(result, Err(oxllm_core::Error::BadRequest(message)) if message.contains("top_k")),
                    "{model}"
                );
            }
        }
    }
}

#[test]
fn geo_prefix_nova2_reasoning_preserves_tool_none_and_serial_tool_rejection() {
    for model in geo_prefixed_model_ids("amazon.nova-2-lite-v1:0") {
        let configured = deployment(&model, false);
        let mut request = serial_profile_request();
        request["reasoning_effort"] = json!("low");
        assert!(
            matches!(
                BedrockProvider.transform_request(&request, &configured),
                Err(oxllm_core::Error::BadRequest(message)) if message.contains("parallel_tool_calls")
            ),
            "{model}"
        );
        for choice in [json!("none"), json!({"type": "none"})] {
            request["tool_choice"] = choice;
            let body = BedrockProvider
                .transform_request(&request, &configured)
                .unwrap();
            assert_geo_prefix_url(&configured, &request, &model, "converse");
            assert_eq!(
                body["additionalModelRequestFields"],
                json!({"reasoningConfig": {"type": "enabled", "maxReasoningEffort": "low"}})
            );
            assert!(body.get("toolConfig").is_none());
        }
    }
}

#[test]
fn geo_prefix_wildcards_resolve_original_model_ids_for_both_apis() {
    for model in [
        "jp.amazon.nova-2-lite-v1:0",
        "au.anthropic.claude-sonnet-4-5-20250929-v1:0",
    ] {
        for explicit_converse in [false, true] {
            let mut request = json!({
                "model": format!("bedrock/{model}"), "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 4096, "reasoning_effort": "low", "tools": serial_profile_request()["tools"]
            });
            let direct = deployment(model, explicit_converse);
            let expected_body = BedrockProvider
                .transform_request(&request, &direct)
                .unwrap();
            let expected_url = BedrockProvider.endpoint_url(&direct);
            for requested in [model.to_string(), format!("bedrock/{model}")] {
                request["model"] = json!(requested);
                let original = request.clone();
                for wildcard in ["*", "bedrock/*"] {
                    let mut configured = deployment("*", explicit_converse);
                    configured.oxllm_params.model = wildcard.into();
                    assert_eq!(
                        BedrockProvider.endpoint_url_for_request(&configured, &request),
                        expected_url
                    );
                    assert_eq!(
                        BedrockProvider
                            .transform_request(&request, &configured)
                            .unwrap(),
                        expected_body
                    );
                    assert_eq!(configured.oxllm_params.model, wildcard);
                }
                assert_eq!(request, original);
            }
        }
    }
}

#[test]
fn geo_prefix_unknown_custom_ids_remain_passthrough_and_do_not_gain_family_features() {
    let request = json!({
        "model": "public-alias", "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 4096, "reasoning_effort": "low", "web_search_options": {},
        "tools": serial_profile_request()["tools"], "custom_field": {"preserve": true}
    });
    for model in [
        "zz.amazon.nova-2-lite-v1:0",
        "zz.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "japan.amazon.nova-2-lite-v1:0",
        "aus.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "custom.jp.amazon.nova-2-lite-v1:0",
        "custom.au.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "jp.custom.vendor-chat",
        "au.custom.vendor-chat",
    ] {
        let mut configured = deployment(model, false);
        set_profile_hint(
            &mut configured,
            "bedrock/au.anthropic.claude-sonnet-4-5-20250929-v1:0",
            true,
        );
        let body = BedrockProvider
            .transform_request(&request, &configured)
            .unwrap();
        assert_geo_prefix_url(&configured, &request, model, "invoke");
        let mut expected = request.clone();
        expected["model"] = json!(model);
        assert_eq!(body, expected);
        assert_eq!(detect_bedrock_provider(model), BedrockVendor::Unknown);

        configured
            .oxllm_params
            .extra
            .insert("use_converse_api".into(), json!(true));
        let body = BedrockProvider
            .transform_request(&request, &configured)
            .unwrap();
        assert_geo_prefix_url(&configured, &request, model, "converse");
        assert!(body.get("additionalModelRequestFields").is_none());
        assert_eq!(body["toolConfig"]["tools"].as_array().unwrap().len(), 1);
        for (key, value, error_field) in [
            ("top_k", json!(20), "top_k"),
            ("parallel_tool_calls", json!(false), "parallel_tool_calls"),
        ] {
            let mut unsupported = request.clone();
            unsupported[key] = value;
            assert!(
                matches!(
                    BedrockProvider.transform_request(&unsupported, &configured),
                    Err(oxllm_core::Error::BadRequest(message)) if message.contains(error_field)
                ),
                "{model}"
            );
        }
    }
}

fn assert_noncanonical_geo_model_stays_unknown(id: &str) {
    for model in [
        id.to_string(),
        format!("arn:aws:bedrock:ap-northeast-1::foundation-model/{id}"),
        format!("arn:aws:bedrock:ap-southeast-2:123456789012:inference-profile/{id}"),
    ] {
        let request = json!({
            "model": "public-alias", "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 32, "custom_field": {"preserve": true}, "web_search_options": {},
            "tools": serial_profile_request()["tools"]
        });
        let mut configured = deployment(&model, false);
        let body = BedrockProvider
            .transform_request(&request, &configured)
            .unwrap();
        assert_geo_prefix_url(&configured, &request, &model, "invoke");
        let mut expected = request.clone();
        expected["model"] = json!(model);
        assert_eq!(
            body, expected,
            "{model}: custom InvokeModel fields must survive"
        );
        assert_eq!(
            detect_bedrock_provider(&model),
            BedrockVendor::Unknown,
            "{model}"
        );
        assert!(!should_use_converse(&model, false), "{model}");

        let mut wildcard_request = request.clone();
        wildcard_request["model"] = json!(format!("bedrock/{model}"));
        let wildcard = deployment("*", false);
        assert_eq!(
            BedrockProvider.endpoint_url_for_request(&wildcard, &wildcard_request),
            BedrockProvider.endpoint_url(&configured)
        );
        assert_eq!(
            BedrockProvider
                .transform_request(&wildcard_request, &wildcard)
                .unwrap(),
            expected
        );

        configured
            .oxllm_params
            .extra
            .insert("use_converse_api".into(), json!(true));
        let body = BedrockProvider
            .transform_request(&request, &configured)
            .unwrap();
        assert_geo_prefix_url(&configured, &request, &model, "converse");
        assert_eq!(body["messages"][0]["content"][0]["text"], "hello");
        assert_eq!(body["inferenceConfig"]["maxTokens"], 32);
        assert_eq!(body["toolConfig"]["tools"].as_array().unwrap().len(), 1);
        assert!(body.get("model").is_none());
        assert!(body.get("custom_field").is_none());
        assert!(body.get("additionalModelRequestFields").is_none());
        for (key, value, error_field) in [
            ("top_k", json!(20), "top_k"),
            ("parallel_tool_calls", json!(false), "parallel_tool_calls"),
        ] {
            let mut unsupported = request.clone();
            unsupported[key] = value;
            assert!(
                matches!(
                    BedrockProvider.transform_request(&unsupported, &configured),
                    Err(oxllm_core::Error::BadRequest(message)) if message.contains(error_field)
                ),
                "{model}"
            );
        }
    }
}

#[test]
fn geo_prefix_noncanonical_doubled_and_cross_pairs_stay_unknown() {
    for base in [
        "amazon.nova-2-lite-v1:0",
        "anthropic.claude-sonnet-4-5-20250929-v1:0",
    ] {
        for outer in ["jp", "au", "us", "eu", "apac", "global"] {
            for inner in ["jp", "au", "us", "eu", "apac", "global"] {
                assert_noncanonical_geo_model_stays_unknown(&format!("{outer}.{inner}.{base}"));
            }
        }
    }
}

#[test]
fn geo_prefix_noncanonical_malformed_and_nested_region_heads_stay_unknown() {
    for base in [
        "amazon.nova-2-lite-v1:0",
        "anthropic.claude-sonnet-4-5-20250929-v1:0",
    ] {
        for geo in ["jp", "au", "us", "eu", "apac", "global"] {
            for model in [
                format!("{geo}.us-east-1.{base}"),
                format!("us-east-1.{geo}.{base}"),
                format!("{geo}.{geo}-.{base}"),
                format!("{geo}..{base}"),
                format!("{geo}.-{base}"),
                format!(".{geo}.{base}"),
                format!("{geo}x.{base}"),
            ] {
                assert_noncanonical_geo_model_stays_unknown(&model);
            }
        }
    }
}

#[test]
fn geo_prefix_must_use_converse_families_need_only_one_prefix_normalization() {
    // One representative for every MUST_USE_CONVERSE entry, without deriving
    // expectations from the production allowlist itself.
    for base in [
        "ai21.jamba-1-5-large-v1:0",
        "qwen.qwen3-32b-v1:0",
        "amazon.nova-2-lite-v1:0",
        "deepseek.r1-v1:0",
        "openai.gpt-5.6-sol",
        "openai.gpt-6.1-sol",
        "mistral.ministral-3-8b-instruct",
        "mistral.mistral-large-3-675b-instruct",
        "mistral.devstral-2-123b",
        "mistral.magistral-small-2509",
    ] {
        for model in geo_prefixed_model_ids(base) {
            let request =
                json!({"messages": [{"role": "user", "content": "hello"}], "max_tokens": 32});
            let configured = deployment(&model, false);
            let body = BedrockProvider
                .transform_request(&request, &configured)
                .unwrap();
            assert_geo_prefix_url(&configured, &request, &model, "converse");
            assert_eq!(body["messages"][0]["content"][0]["text"], "hello");
            assert_eq!(body["inferenceConfig"]["maxTokens"], 32);
            assert!(body.get("model").is_none());
        }
    }
}
