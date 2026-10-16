use super::*;
use oxllm_core::deployment::OxllmParams;

fn deployment(model: &str) -> Deployment {
    Deployment {
        oxllm_params: OxllmParams {
            model: format!("anthropic/{model}"),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn schema() -> Value {
    json!({"type":"object", "properties":{"entries":{"type":"array", "items":{"type":"string"}}}, "required":["entries"], "additionalProperties":false})
}

#[test]
fn native_schema_forwarding_is_not_gated_by_a_model_name_allowlist() {
    for model in [
        "claude-haiku-5-5",
        "claude-3-5-sonnet",
        "private-deployment",
    ] {
        let request = json!({"messages":[{"role":"user", "content":"Extract entries"}], "response_format":{"type":"json_schema", "json_schema":{"name":"record", "strict":true, "schema":schema()}}});
        let body = AnthropicProvider
            .transform_request(&request, &deployment(model))
            .unwrap();
        assert_eq!(
            body["output_config"]["format"],
            json!({"type":"json_schema", "schema":schema()})
        );
        assert!(body.get("tools").is_none());
        assert!(body.get("tool_choice").is_none());
    }
}

#[test]
fn native_output_schemas_are_forwarded_without_repair_or_constraint_removal() {
    let schema = json!({"type":"object", "additionalProperties":true,
        "properties":{"entries":{"type":"array", "uniqueItems":true, "minItems":3,
            "items":{"oneOf":[{"type":"string", "minLength":5}, {"type":"integer", "minimum":10}]}}}});
    let request = json!({"messages":[], "response_format":{"type":"json_schema", "json_schema":{"name":"record", "schema":schema}}});
    let body = AnthropicProvider
        .transform_request(&request, &deployment("claude-haiku-5-5"))
        .unwrap();
    assert_eq!(body["output_config"]["format"]["schema"], schema);
}

#[test]
fn schema_less_json_mode_is_rejected_instead_of_emulated_with_a_tool() {
    for model in ["claude-haiku-5-5", "claude-sonnet-4-5", "claude-3-5-sonnet"] {
        let request = json!({"messages":[], "response_format":{"type":"json_object"}});
        assert!(matches!(
            AnthropicProvider.transform_request(&request, &deployment(model)),
            Err(oxllm_core::Error::BadRequest(_))
        ));
    }
}

#[test]
fn explicit_strict_tools_preserve_schemas_and_the_callers_tool_choice() {
    let schema = json!({"type":"object", "additionalProperties":false, "properties":{
        "strict":{"type":"boolean"}, "entries":{"type":"array", "uniqueItems":true, "items":{"oneOf":[{"type":"string"}, {"type":"integer"}]}}
    }, "required":["entries"]});
    for model in ["claude-haiku-5-5", "claude-3-5-sonnet"] {
        let request = json!({"messages":[], "tools":[{"type":"function", "function":{"name":"extract_record", "strict":true, "parameters":schema}}], "tool_choice":{"type":"function", "function":{"name":"extract_record"}}});
        let body = AnthropicProvider
            .transform_request(&request, &deployment(model))
            .unwrap();
        assert_eq!(body["tools"][0]["strict"], true);
        assert_eq!(body["tools"][0]["input_schema"], schema);
        assert_eq!(
            body["tool_choice"],
            json!({"type":"tool", "name":"extract_record"})
        );
        assert!(body.get("output_config").is_none());
    }
}

#[test]
fn absent_false_and_null_tool_strictness_do_not_enable_strict_mode() {
    for strict in [None, Some(json!(false)), Some(Value::Null)] {
        let mut function = json!({"name":"extract_record", "parameters":schema()});
        if let Some(strict) = strict {
            function["strict"] = strict;
        }
        let body = AnthropicProvider
            .transform_request(
                &json!({"messages":[], "tools":[{"type":"function", "function":function}]}),
                &deployment("claude-haiku-5-5"),
            )
            .unwrap();
        assert!(body["tools"][0].get("strict").is_none());
    }
}

#[test]
fn misplaced_or_non_boolean_strict_flags_are_clear_request_errors() {
    for function in [
        json!({"name":"extract_record", "parameters":{"type":"object", "strict":true}}),
        json!({"name":"extract_record", "strict":"true", "parameters":schema()}),
    ] {
        assert!(matches!(
            AnthropicProvider.transform_request(
                &json!({"messages":[], "tools":[{"type":"function", "function":function}]}),
                &deployment("claude-haiku-5-5")
            ),
            Err(oxllm_core::Error::BadRequest(_))
        ));
    }
}

#[test]
fn explicitly_named_legacy_json_helper_tools_remain_visible_tool_calls() {
    let request = json!({"messages":[], "tools":[{"type":"function", "function":{"name":"json_tool_call", "strict":true, "parameters":schema()}}]});
    let response = json!({"id":"msg-native", "model":"claude-haiku-5-5", "type":"message", "content":[{"type":"tool_use", "id":"call-native", "name":"json_tool_call", "input":{"entries":["one"]}}], "stop_reason":"tool_use", "usage":{"input_tokens":2,"output_tokens":3}});
    let body = AnthropicProvider
        .transform_response_with_request(&response, "claude-haiku-5-5", &request)
        .unwrap();
    assert_eq!(body["choices"][0]["finish_reason"], "tool_calls");
    assert_eq!(
        body["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
        "json_tool_call"
    );
}

#[test]
fn explicit_strict_flags_on_known_native_hosted_tools_are_provider_validated() {
    let body=AnthropicProvider.transform_request(&json!({"messages":[],"tools":[{"type":"web_search_20250305","name":"web_search","strict":true}]}),&deployment("claude-haiku-5-5")).unwrap();
    assert_eq!(
        body["tools"][0],
        json!({"type":"web_search_20250305","name":"web_search","strict":true})
    );
}

#[test]
fn legacy_output_format_is_a_lossless_stable_native_alias() {
    let format = json!({"type":"json_schema","schema":schema()});
    let request = json!({"messages":[],"output_format":format});
    let body = AnthropicProvider
        .transform_request(&request, &deployment("claude-haiku-5-5"))
        .unwrap();
    assert_eq!(body["output_config"]["format"], format);
    assert!(body.get("output_format").is_none());
    assert!(
        !collect_beta_headers(&request, "claude-haiku-5-5")
            .contains(&BETA_STRUCTURED_OUTPUTS.to_string())
    );
}

#[test]
fn implemented_native_output_fields_are_advertised_for_drop_params() {
    let fields = AnthropicProvider.supported_params().unwrap();
    assert!(fields.contains(&"output_config"));
    assert!(fields.contains(&"output_format"));
}

#[test]
fn null_native_format_does_not_conflict_with_a_requested_schema() {
    let body=AnthropicProvider.transform_request(&json!({"messages":[],"response_format":{"type":"json_schema","json_schema":{"schema":schema()}},"output_config":{"format":null,"effort":"low"}}),&deployment("claude-haiku-5-5")).unwrap();
    assert_eq!(body["output_config"]["format"]["schema"], schema());
    assert_eq!(body["output_config"]["effort"], "low");
}

#[test]
fn hosted_native_fields_are_not_lost_when_the_legacy_carrier_is_mapped() {
    let tool = json!({"type":"computer_20250124","name":"computer","display_width_px":1280,"display_height_px":720,"display_number":2,"cache_control":{"type":"ephemeral"},"strict":true});
    let body = AnthropicProvider
        .transform_request(
            &json!({"messages":[],"tools":[tool]}),
            &deployment("claude-haiku-5-5"),
        )
        .unwrap();
    assert_eq!(body["tools"][0], tool);
}

#[test]
fn equivalent_format_aliases_agree_and_keep_independent_native_options() {
    let format = json!({"type":"json_schema","schema":schema()});
    let body=AnthropicProvider.transform_request(&json!({
        "messages":[],
        "response_format":{"type":"json_schema","json_schema":{"name":"record","schema":schema()}},
        "output_config":{"format":format,"effort":"low","native_option":true},
        "output_format":format
    }),&deployment("claude-haiku-5-5")).unwrap();
    assert_eq!(
        body["output_config"],
        json!({"format":format,"effort":"low","native_option":true})
    );
    assert!(body.get("output_format").is_none());
}

#[test]
fn contradictory_or_malformed_formats_are_actionable_local_errors() {
    let format = json!({"type":"json_schema","schema":schema()});
    for (fields, message) in [
        (
            json!({"response_format":{"type":"text"},"output_config":{"format":format}}),
            "text conflicts with output_config.format",
        ),
        (
            json!({"response_format":{"type":"json_schema","json_schema":{"schema":schema()}},"output_config":{"format":{"type":"json_schema","schema":{"type":"object"}}}}),
            "response_format conflicts with output_config.format",
        ),
        (
            json!({"output_config":{"format":format},"output_format":{"type":"json_schema","schema":{"type":"object"}}}),
            "output_config.format conflicts with output_format",
        ),
        (
            json!({"output_config":[]}),
            "output_config must be an object",
        ),
        (
            json!({"output_config":{"format":true}}),
            "output_config.format must be an object",
        ),
        (
            json!({"output_format":{"type":"json_object"}}),
            "output_format requires json_schema",
        ),
        (
            json!({"output_format":{"type":"json_schema","schema":schema(),"native_option":true}}),
            "use output_config.format for native options",
        ),
    ] {
        let error = AnthropicProvider
            .transform_request(&fields, &deployment("claude-haiku-5-5"))
            .unwrap_err();
        assert!(matches!(error, oxllm_core::Error::BadRequest(_)));
        assert!(error.to_string().contains(message), "{fields}: {error}");
    }
}

#[test]
fn hosted_tool_mapping_keeps_root_options_and_legacy_computer_dimensions() {
    for kind in [
        "bash_20250124",
        "text_editor_20250728",
        "tool_search_tool_regex_20251119",
        "advisor_20260301",
    ] {
        let tool = json!({"type":kind,"name":"native_tool","model":"claude-opus-4-8","cache_control":{"type":"ephemeral"},"native_option":{"enabled":true},"strict":true});
        let body = AnthropicProvider
            .transform_request(
                &json!({"messages":[],"tools":[tool]}),
                &deployment("claude-haiku-5-5"),
            )
            .unwrap();
        assert_eq!(body["tools"][0], tool);
    }
    let body=AnthropicProvider.transform_request(&json!({"messages":[],"tools":[{
        "type":"computer_20250124","display_width_px":1280,
        "function":{"name":"computer","parameters":{"display_width_px":640,"display_height_px":720},"cache_control":{"type":"ephemeral"}}
    }]}),&deployment("claude-haiku-5-5")).unwrap();
    assert_eq!(
        body["tools"][0],
        json!({"type":"computer_20250124","name":"computer","display_width_px":1280,"display_height_px":720,"cache_control":{"type":"ephemeral"}})
    );
}

#[test]
fn unsupported_strict_tool_choices_are_not_replaced_by_parallel_auto_defaults() {
    for choice in [
        json!({"type":"allowed_tools","mode":"required","tools":[{"type":"function","name":"extract"}]}),
        json!("unknown_choice"),
        json!({"type":"function","function":{"name":"extract","native_option":true}}),
    ] {
        let request = json!({"messages":[],"tools":[{"type":"function","function":{"name":"extract","strict":true,"parameters":schema()}}],"tool_choice":choice,"parallel_tool_calls":false});
        let error = AnthropicProvider
            .transform_request(&request, &deployment("claude-haiku-5-5"))
            .unwrap_err();
        assert!(matches!(error, oxllm_core::Error::BadRequest(_)));
        assert!(error.to_string().contains("tool choice"));
    }
}

#[test]
fn native_strict_tool_choices_keep_provider_owned_options() {
    let choice = json!({"type":"any","disable_parallel_tool_use":true,"native_option":true});
    let body=AnthropicProvider.transform_request(&json!({"messages":[],"tools":[{"type":"function","function":{"name":"extract","strict":true,"parameters":schema()}}],"tool_choice":choice}),&deployment("claude-haiku-5-5")).unwrap();
    assert_eq!(body["tool_choice"], choice);
}

#[test]
fn contradictory_hosted_strict_carriers_are_local_errors() {
    for (root, nested) in [(true, false), (false, true)] {
        let tool = json!({"type":"computer_20250124","strict":root,"function":{"name":"computer","strict":nested,"parameters":{"display_width_px":1280,"display_height_px":720}}});
        let error = AnthropicProvider
            .transform_request(
                &json!({"messages":[],"tools":[tool]}),
                &deployment("claude-haiku-5-5"),
            )
            .unwrap_err();
        assert!(matches!(error, oxllm_core::Error::BadRequest(_)));
        assert!(
            error
                .to_string()
                .contains("strict conflicts with function.strict")
        );
    }
}

#[test]
fn advisory_tool_choices_without_native_mapping_are_request_errors() {
    for choice in [
        json!({"type":"allowed_tools","mode":"required","tools":[{"type":"function","name":"extract"}]}),
        json!("unknown_choice"),
    ] {
        let request = json!({"messages":[],"tools":[{"type":"function","function":{"name":"extract","parameters":schema()}}],"tool_choice":choice,"parallel_tool_calls":false});
        assert!(matches!(
            AnthropicProvider.transform_request(&request, &deployment("claude-haiku-5-5")),
            Err(oxllm_core::Error::BadRequest(_))
        ));
    }
}

#[test]
fn root_strict_on_function_carriers_is_not_discarded() {
    let input = json!({"type":"object","properties":{"entries":{"type":"array","uniqueItems":true}},"additionalProperties":false});
    let request = json!({"messages":[],"tools":[{"type":"function","strict":true,"function":{"name":"extract","parameters":input}}]});
    let body = AnthropicProvider
        .transform_request(&request, &deployment("claude-haiku-5-5"))
        .unwrap();
    assert_eq!(body["tools"][0]["strict"], true);
    assert_eq!(body["tools"][0]["input_schema"], input);
}

#[test]
fn misplaced_strict_flags_in_secondary_schema_carriers_are_errors() {
    for carrier in ["input_schema", "inputSchema"] {
        let mut function = json!({"name":"extract","parameters":{"type":"object"}});
        function[carrier] = if carrier == "inputSchema" {
            json!({"json":{"type":"object","strict":true}})
        } else {
            json!({"type":"object","strict":true})
        };
        let request = json!({"messages":[],"tools":[{"type":"function","function":function}]});
        assert!(matches!(
            AnthropicProvider.transform_request(&request, &deployment("claude-haiku-5-5")),
            Err(oxllm_core::Error::BadRequest(_))
        ));
    }
}

#[test]
fn null_native_format_does_not_leave_a_wire_format_member() {
    let body = AnthropicProvider
        .transform_request(
            &json!({"messages":[],"output_config":{"format":null,"effort":"low"}}),
            &deployment("claude-haiku-5-5"),
        )
        .unwrap();
    assert_eq!(body["output_config"], json!({"effort":"low"}));
}

#[test]
fn json_object_intent_can_use_a_caller_supplied_native_schema() {
    for alias in ["output_config", "output_format"] {
        let mut request = json!({"messages":[],"response_format":{"type":"json_object"}});
        let format = json!({"type":"json_schema","schema":schema()});
        request[alias] = if alias == "output_config" {
            json!({"format":format})
        } else {
            format
        };
        let body = AnthropicProvider
            .transform_request(&request, &deployment("claude-haiku-5-5"))
            .unwrap();
        assert_eq!(body["output_config"]["format"]["schema"], schema());
        assert!(body.get("tools").is_none());
    }
}

#[test]
fn hosted_legacy_carriers_do_not_leak_into_native_web_search_tools() {
    let body=AnthropicProvider.transform_request(&json!({"messages":[],"tools":[{"type":"web_search_20250305","name":"web_search","cache_control":null,"function":{"strict":true,"cache_control":{"type":"ephemeral"}}}]}),&deployment("claude-haiku-5-5")).unwrap();
    assert_eq!(
        body["tools"][0],
        json!({"type":"web_search_20250305","name":"web_search","strict":true})
    );
}

#[test]
fn colliding_tool_names_do_not_remap_a_forced_choice_twice() {
    for native in [false, true] {
        let choice = if native {
            json!({"type":"tool","name":"my.tool"})
        } else {
            json!({"type":"function","function":{"name":"my.tool"}})
        };
        let request = json!({"messages":[],"tools":[
            {"type":"function","function":{"name":"my.tool","strict":true,"parameters":schema()}},
            {"type":"function","function":{"name":"my_tool","strict":true,"parameters":schema()}}
        ],"tool_choice":choice});
        let body = AnthropicProvider
            .transform_request(&request, &deployment("claude-haiku-5-5"))
            .unwrap();
        assert_eq!(body["tools"][0]["name"], "my_tool");
        assert_eq!(body["tools"][1]["name"], "my_tool_2");
        assert_eq!(body["tool_choice"]["name"], body["tools"][0]["name"]);
    }
}
