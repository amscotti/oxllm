//! Host-side preservation and cache compatibility for native output requests.

use serde_json::{Map, Value};

const FORMATS: &[&str] = &[
    "/response_format",
    "/text/format",
    "/output_config/format",
    "/output_config/textFormat",
    "/outputConfig/textFormat",
    "/outputConfig/format",
    "/output_format",
];
const TOOLS: &[&str] = &["/tools", "/functions", "/toolConfig/tools"];
const CHOICES: &[&str] = &[
    "/tool_choice",
    "/function_call",
    "/toolConfig/toolChoice",
    "/additionalModelRequestFields/tool_choice",
    "/additional_model_request_fields/tool_choice",
];
// Internal contract state, never a request/wire field. Omitting all tool
// definitions is a native representation of an explicit no-tools selection.
const NO_TOOLS: &str = "$tools_disabled";

fn none_choice(choice: &Value) -> bool {
    choice.as_str() == Some("none")
        || choice.as_object().is_some_and(|object| {
            object.len() == 1 && object.get("type").and_then(Value::as_str) == Some("none")
        })
}

fn explicit_none(body: &Value) -> bool {
    let mut choices = CHOICES
        .iter()
        .filter_map(|path| body.pointer(path).filter(|value| !value.is_null()))
        .peekable();
    choices.peek().is_some() && choices.all(none_choice)
}

fn has_tool_sources(body: &Value) -> bool {
    ["/tools", "/functions", "/toolConfig/tools", "/mcp_servers"]
        .iter()
        .any(|path| {
            body.pointer(path).is_some_and(|value| {
                !value.is_null() && !value.as_array().is_some_and(Vec::is_empty)
            })
        })
        || body
            .get("web_search_options")
            .is_some_and(|value| !value.is_null())
}

fn tools_disabled(body: &Value) -> bool {
    explicit_none(body)
        || (!has_tool_sources(body)
            && !CHOICES
                .iter()
                .any(|path| body.pointer(path).is_some_and(|value| !value.is_null())))
}

fn format_entries(body: &Value) -> Vec<(String, &Value)> {
    let mut formats = FORMATS
        .iter()
        .filter_map(|path| {
            body.pointer(path)
                .filter(|value| shaping_format(path, value))
                .map(|value| ((*path).to_string(), value))
        })
        .collect::<Vec<_>>();
    if let Some(messages) = body.get("messages").and_then(Value::as_array) {
        for (index, message) in messages.iter().enumerate() {
            if let Some(format) = message
                .pointer("/output_config/format")
                .filter(|value| !value.is_null())
            {
                formats.push((format!("/messages/{index}/output_config/format"), format));
            }
        }
    }
    formats
}

#[derive(Debug, PartialEq)]
enum FormatRequirement {
    JsonSchema(Value),
    Json,
    Text,
    Opaque(String, Value),
}

fn only_keys(value: &Value, keys: &[&str]) -> bool {
    value
        .as_object()
        .is_some_and(|object| object.keys().all(|key| keys.contains(&key.as_str())))
}

/// Compare supported API envelopes, not schemas or their constraints. Unknown
/// native options remain opaque and must survive in their original dialect.
fn format_requirement(path: &str, format: &Value) -> FormatRequirement {
    if path == "/text/format" {
        match format.get("type").and_then(Value::as_str) {
            Some("text") if only_keys(format, &["type"]) => return FormatRequirement::Text,
            Some("json_object") if only_keys(format, &["type"]) => return FormatRequirement::Json,
            Some("json_schema")
                if only_keys(format, &["type", "name", "description", "strict", "schema"]) =>
            {
                if let Some(schema) = format.get("schema").filter(|schema| schema.is_object()) {
                    return FormatRequirement::JsonSchema(schema.clone());
                }
            }
            _ => {}
        }
    } else if path == "/response_format" && only_keys(format, &["type", "json_schema"]) {
        match format.get("type").and_then(Value::as_str) {
            Some("text") if only_keys(format, &["type"]) => return FormatRequirement::Text,
            Some("json_object") if format.get("json_schema").is_none() => {
                return FormatRequirement::Json;
            }
            Some("json_object" | "json_schema") => {
                if let Some(wrapper) = format.get("json_schema").filter(|value| value.is_object()) {
                    if let Some(schema) = wrapper.get("schema") {
                        if schema.is_object()
                            && only_keys(wrapper, &["name", "description", "strict", "schema"])
                        {
                            return FormatRequirement::JsonSchema(schema.clone());
                        }
                    } else if wrapper.get("name").is_none() && wrapper.get("strict").is_none() {
                        return FormatRequirement::JsonSchema(wrapper.clone());
                    }
                }
            }
            _ => {}
        }
    } else if path.ends_with("/textFormat") {
        if format.get("type").and_then(Value::as_str) == Some("json_schema")
            && only_keys(format, &["type", "structure"])
            && let Some(structure) = format
                .get("structure")
                .filter(|value| only_keys(value, &["jsonSchema"]))
            && let Some(schema) = structure
                .get("jsonSchema")
                .filter(|value| only_keys(value, &["name", "description", "schema"]))
            && let Some(serialized) = schema.get("schema").and_then(Value::as_str)
            && let Ok(schema) = serde_json::from_str(serialized)
        {
            return FormatRequirement::JsonSchema(schema);
        }
    } else if format.get("type").and_then(Value::as_str) == Some("json_schema")
        && only_keys(format, &["type", "schema"])
        && let Some(schema) = format.get("schema").filter(|value| value.is_object())
    {
        return FormatRequirement::JsonSchema(schema.clone());
    }
    let dialect = if path == "/text/format" {
        "responses-format"
    } else if path.starts_with("/messages/") {
        "message-output-config"
    } else if path.ends_with("/textFormat") {
        "converse-text-format"
    } else if path != "/response_format" {
        "anthropic-format"
    } else {
        path
    };
    FormatRequirement::Opaque(dialect.into(), format.clone())
}

fn satisfies(expected: &FormatRequirement, actual: &FormatRequirement) -> bool {
    expected == actual
        || matches!(
            (expected, actual),
            (FormatRequirement::Json, FormatRequirement::JsonSchema(_))
        )
}

pub(crate) fn validate_formats(body: &Value) -> oxllm_core::Result<()> {
    let formats = FORMATS
        .iter()
        .filter_map(|path| {
            body.pointer(path)
                .filter(|value| !value.is_null())
                .map(|value| (*path, format_requirement(path, value)))
        })
        .collect::<Vec<_>>();
    for (index, (path, requirement)) in formats.iter().enumerate() {
        for (other, other_requirement) in &formats[index + 1..] {
            if !satisfies(requirement, other_requirement)
                && !satisfies(other_requirement, requirement)
            {
                return Err(oxllm_core::Error::BadRequest(format!(
                    "{} conflicts with {}; send one format definition or equivalent lossless definitions",
                    path.trim_start_matches('/'),
                    other.trim_start_matches('/')
                )));
            }
        }
    }
    Ok(())
}

fn shaping_format(path: &str, format: &Value) -> bool {
    !(format.is_null()
        || matches!(path, "/response_format" | "/text/format")
            && format.get("type").and_then(Value::as_str) == Some("text")
            && only_keys(format, &["type"]))
}

fn strict_tool(tool: &Value) -> bool {
    [
        "/strict",
        "/function/strict",
        "/toolSpec/strict",
        "/parameters/strict",
        "/function/parameters/strict",
        "/function/input_schema/strict",
        "/function/inputSchema/json/strict",
        "/input_schema/strict",
        "/toolSpec/inputSchema/json/strict",
    ]
    .iter()
    .any(|path| {
        tool.pointer(path)
            .is_some_and(|value| !value.is_null() && value.as_bool() != Some(false))
    })
}

/// Capture only response-shaping requirements, never prompts or credentials.
pub(crate) fn contract(body: &Value) -> Option<Value> {
    let mut contract = Map::new();
    for (path, format) in format_entries(body) {
        contract.insert(path, format.clone());
    }
    if explicit_none(body) {
        contract.insert(NO_TOOLS.into(), Value::Bool(true));
        return Some(Value::Object(contract));
    }
    let has_choice = CHOICES
        .iter()
        .any(|path| body.pointer(path).is_some_and(|value| !value.is_null()));
    let serial = body.get("parallel_tool_calls").and_then(Value::as_bool) == Some(false)
        || CHOICES.iter().any(|path| {
            body.pointer(path)
                .and_then(|choice| choice.get("disable_parallel_tool_use"))
                .and_then(Value::as_bool)
                == Some(true)
        });
    for &path in TOOLS {
        if let Some(tools) = body.pointer(path).and_then(Value::as_array)
            && (tools.iter().any(strict_tool) || has_choice || serial)
        {
            contract.insert(path.into(), Value::Array(tools.clone()));
        }
    }
    for &choice in CHOICES {
        if let Some(value) = body.pointer(choice).filter(|value| !value.is_null()) {
            contract.insert(choice.into(), value.clone());
        }
    }
    if serial && (has_choice || has_tool_sources(body)) {
        contract.insert("/parallel_tool_calls".into(), Value::Bool(false));
    }
    (!contract.is_empty()).then_some(Value::Object(contract))
}

/// Check each kind of output requirement after API-envelope conversion. A
/// surviving format must not hide dropped strict tools, nor the reverse.
/// Values/names differ across APIs; subsequent policy shaping is checked
/// exactly against the returned native contract instead.
pub(crate) fn native_contract(
    expected: &Option<Value>,
    body: &Value,
) -> oxllm_core::Result<Option<Value>> {
    let Some(expected) = expected else {
        return Ok(None);
    };
    let mut native = contract(body);
    let mut no_tools = expected.get(NO_TOOLS).and_then(Value::as_bool) == Some(true);
    if no_tools {
        if !tools_disabled(body) {
            return Err(oxllm_core::Error::BadRequest(
                "provider adapter must preserve tool_choice none without enabling tool use".into(),
            ));
        }
        native.get_or_insert_with(|| Value::Object(Map::new()))[NO_TOOLS] = Value::Bool(true);
    }
    let requested_formats = FORMATS
        .iter()
        .filter(|path| expected.get(**path).is_some())
        .copied()
        .collect::<Vec<_>>();
    for path in &requested_formats {
        let requirement = format_requirement(path, &expected[*path]);
        if !FORMATS.iter().any(|actual_path| {
            native
                .as_ref()
                .and_then(|native| native.get(*actual_path))
                .is_some_and(|format| {
                    satisfies(&requirement, &format_requirement(actual_path, format))
                })
        }) {
            return Err(oxllm_core::Error::BadRequest(format!(
                "provider adapter has no native representation for requested output format ({path}); use a format supported by this adapter or the provider's native endpoint for additional format options"
            )));
        }
    }
    let message_formats = |contract: &Value| {
        contract
            .as_object()
            .into_iter()
            .flat_map(|contract| contract.iter())
            .filter(|(path, _)| path.starts_with("/messages/"))
            .map(|(path, format)| (path.clone(), format_requirement(path, format)))
            .collect::<Vec<_>>()
    };
    let mut requested_messages = message_formats(expected);
    let mut actual_messages = native.as_ref().map(message_formats).unwrap_or_default();
    let index = |(path, _): &(String, FormatRequirement)| {
        path.split('/')
            .nth(2)
            .and_then(|index| index.parse::<usize>().ok())
            .unwrap_or_default()
    };
    requested_messages.sort_by_key(index);
    actual_messages.sort_by_key(index);
    if requested_messages.len() != actual_messages.len()
        || requested_messages
            .iter()
            .zip(&actual_messages)
            .any(|((_, expected), (_, actual))| !satisfies(expected, actual))
    {
        return Err(oxllm_core::Error::BadRequest("provider adapter has no lossless native representation for message-level output_config.format; use a native Anthropic Messages route".into()));
    }
    let strict_count = |contract: &Value| {
        TOOLS
            .iter()
            .filter_map(|path| contract.get(*path).and_then(Value::as_array))
            .flatten()
            .filter(|tool| strict_tool(tool))
            .count()
    };
    if !no_tools && strict_count(expected) > native.as_ref().map(strict_count).unwrap_or_default() {
        return Err(oxllm_core::Error::BadRequest("provider adapter has no native representation for all requested strict tools; use a route with native strict-tool support".into()));
    }
    if !no_tools
        && CHOICES.iter().any(|path| expected.get(*path).is_some())
        && !CHOICES.iter().any(|path| {
            native
                .as_ref()
                .is_some_and(|native| native.get(*path).is_some())
        })
    {
        let auto_without_tools = !TOOLS.iter().any(|path| {
            expected
                .get(*path)
                .and_then(Value::as_array)
                .is_some_and(|tools| !tools.is_empty())
        }) && CHOICES.iter().filter_map(|path| expected.get(*path)).all(
            |choice| {
                choice.as_str() == Some("auto")
                    || choice.get("type").and_then(Value::as_str) == Some("auto")
            },
        ) && tools_disabled(body);
        if !auto_without_tools {
            return Err(oxllm_core::Error::BadRequest("provider adapter has no native representation for the requested tool choice; use a supported explicit tool choice or the provider's native endpoint".into()));
        }
        native.get_or_insert_with(|| Value::Object(Map::new()))[NO_TOOLS] = Value::Bool(true);
        no_tools = true;
    }
    if !no_tools
        && expected
            .get("/parallel_tool_calls")
            .and_then(Value::as_bool)
            == Some(false)
        && !CHOICES.iter().any(|path| {
            body.pointer(path).is_some_and(|choice| {
                choice.as_str() == Some("none")
                    || choice.get("type").and_then(Value::as_str) == Some("none")
            })
        })
        && body.get("parallel_tool_calls").and_then(Value::as_bool) != Some(false)
        && ![
            "/tool_choice/disable_parallel_tool_use",
            "/additionalModelRequestFields/tool_choice/disable_parallel_tool_use",
            "/additional_model_request_fields/tool_choice/disable_parallel_tool_use",
        ]
        .iter()
        .any(|path| body.pointer(path).and_then(Value::as_bool) == Some(true))
    {
        return Err(oxllm_core::Error::BadRequest(
            "provider adapter cannot preserve parallel_tool_calls false for the requested tools"
                .into(),
        ));
    }
    Ok(native)
}

pub(crate) fn ensure_preserved(expected: &Option<Value>, body: &Value) -> oxllm_core::Result<()> {
    let mut actual = contract(body);
    if expected
        .as_ref()
        .is_some_and(|contract| contract.get(NO_TOOLS).and_then(Value::as_bool) == Some(true))
        && tools_disabled(body)
    {
        actual.get_or_insert_with(|| Value::Object(Map::new()))[NO_TOOLS] = Value::Bool(true);
    }
    if expected.is_some() && (expected != &actual || validate_formats(body).is_err()) {
        return Err(oxllm_core::Error::BadRequest("deployment parameter policy must not remove or alter requested structured outputs, strict tools, or their tool choice".into()));
    }
    Ok(())
}

/// A validated adapter has already translated these source fields. An allow
/// policy must not restore a second, foreign encoding into the native body.
pub(crate) fn controls_param(contract: Option<&Value>, key: &str) -> bool {
    contract.and_then(Value::as_object).is_some_and(|contract| {
        contract.keys().any(|path| {
            path.strip_prefix('/')
                .and_then(|path| path.split('/').next())
                == Some(key)
        }) || contract.get(NO_TOOLS).and_then(Value::as_bool) == Some(true)
            && TOOLS.iter().chain(CHOICES).any(|path| {
                path.strip_prefix('/')
                    .and_then(|path| path.split('/').next())
                    == Some(key)
            })
    })
}

pub(crate) fn has_message_formats(body: &Value) -> bool {
    body.get("messages")
        .and_then(Value::as_array)
        .is_some_and(|messages| {
            messages.iter().any(|message| {
                message
                    .pointer("/output_config/format")
                    .is_some_and(|format| !format.is_null())
            })
        })
}

/// Do not reuse responses produced by the old ignored-strict/tool-emulation
/// policy. Tool requests also cover explicitly named historical helper tools.
pub(crate) fn cache_version(body: &Value) -> Option<&'static str> {
    let has_tools = ["tools", "functions", "toolConfig"]
        .iter()
        .any(|field| body.get(field).is_some_and(|value| !value.is_null()));
    (has_tools
        || CHOICES
            .iter()
            .any(|path| body.pointer(path).is_some_and(|value| !value.is_null()))
        || has_message_formats(body)
        || FORMATS.iter().any(|path| {
            body.pointer(path)
                .is_some_and(|value| shaping_format(path, value))
        }))
    .then_some("native-output-policy-v3")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn output_policy_never_confuses_a_schema_property_with_a_strict_flag() {
        let body = json!({"tools":[{"type":"function", "function":{"name":"test", "parameters":{"type":"object", "properties":{"strict":{"type":"boolean"}}}}}]});
        assert!(contract(&body).is_none());
        assert!(cache_version(&body).is_some());
        assert!(cache_version(&json!({"messages":[]})).is_none());
    }

    #[test]
    fn explicit_text_without_tools_keeps_the_legacy_cache_namespace() {
        let body = json!({"messages":[],"response_format":{"type":"text"}});
        assert!(contract(&body).is_none());
        assert!(cache_version(&body).is_none());
        let native = serde_json::json!({"input":"text","text":{"format":{"type":"text"}}});
        assert!(contract(&native).is_none());
        assert!(cache_version(&native).is_none());
    }

    #[test]
    fn deployment_shaping_must_not_remove_native_formats_or_strictness() {
        for body in [
            json!({"response_format":{"type":"json_schema", "json_schema":{"schema":{"type":"object"}}}}),
            json!({"text":{"format":{"type":"json_schema","name":"record","strict":true,"schema":{"type":"object"}}}}),
            json!({"output_config":{"format":{"type":"json_schema", "schema":{"type":"object"}}}}),
            json!({"output_config":{"textFormat":{"type":"json_schema","structure":{"jsonSchema":{"schema":"{\"type\":\"object\"}"}}}}}),
            json!({"tools":[{"name":"test", "strict":true, "input_schema":{"type":"object"}}], "tool_choice":{"type":"tool", "name":"test"}}),
            json!({"toolConfig":{"tools":[{"toolSpec":{"name":"test", "strict":true, "inputSchema":{"json":{"type":"object"}}}}]}}),
        ] {
            let expected = contract(&body);
            assert!(expected.is_some());
            ensure_preserved(&expected, &body).unwrap();
            assert!(ensure_preserved(&expected, &json!({})).is_err());
        }
    }

    #[test]
    fn preserving_one_requirement_cannot_hide_another_being_dropped() {
        let format = json!({"type":"json_schema","json_schema":{"schema":{"type":"object"}}});
        let tools = json!([{"type":"function","function":{"name":"extract","strict":true,"parameters":{"type":"object"}}}]);
        let expected = contract(&json!({"response_format":format,"tools":tools}));
        for body in [
            json!({"response_format":format}),
            json!({"tools":tools}),
            json!({}),
        ] {
            assert!(matches!(
                native_contract(&expected, &body),
                Err(oxllm_core::Error::BadRequest(_))
            ));
        }
        let native = json!({"output_config":{"format":{"type":"json_schema","schema":{"type":"object"}}},"tools":[{"name":"extract","strict":true,"input_schema":{"type":"object"}}]});
        assert_eq!(
            native_contract(&expected, &native).unwrap(),
            contract(&native)
        );
        assert!(native_contract(&None, &native).unwrap().is_none());
    }

    #[test]
    fn native_choices_and_parallel_limits_are_required_and_survive_body_shaping() {
        let tools = json!([{"type":"function","function":{"name":"extract","strict":true,"parameters":{"type":"object"}}}]);
        let expected =
            contract(&json!({"tools":tools,"tool_choice":"required","parallel_tool_calls":false}));
        assert!(native_contract(&expected, &json!({"tools":tools})).is_err());
        assert!(
            native_contract(&expected, &json!({"tools":tools,"tool_choice":"required"})).is_err()
        );
        let mut native = json!({"toolConfig":{"tools":[{"toolSpec":{"name":"extract","strict":true,"inputSchema":{"json":{"type":"object"}}}}]},"additionalModelRequestFields":{"tool_choice":{"type":"any","disable_parallel_tool_use":true}}});
        let native_contract = native_contract(&expected, &native).unwrap();
        ensure_preserved(&native_contract, &native).unwrap();
        native["additionalModelRequestFields"]["tool_choice"]["disable_parallel_tool_use"] =
            json!(false);
        assert!(ensure_preserved(&native_contract, &native).is_err());
    }

    #[test]
    fn explicit_none_choice_satisfies_the_serial_tool_limit_without_an_invalid_flag() {
        let tools = json!([{"name":"extract","strict":true,"input_schema":{"type":"object"}}]);
        let expected =
            contract(&json!({"tools":tools,"tool_choice":"none","parallel_tool_calls":false}));
        for native in [
            json!({"tools":tools,"tool_choice":{"type":"none"}}),
            json!({"toolConfig":{"tools":[{"toolSpec":{"name":"extract","strict":true,"inputSchema":{"json":{"type":"object"}}}}]},"additionalModelRequestFields":{"tool_choice":{"type":"none"}}}),
        ] {
            native_contract(&expected, &native).unwrap();
        }
    }

    #[test]
    fn one_format_cannot_hide_a_dropped_different_schema_or_native_option() {
        let format = json!({"type":"json_schema","json_schema":{"name":"record","schema":{"type":"object","properties":{"a":{"type":"string"}}}}});
        let native = json!({"response_format":format});
        for extra in [
            json!({"type":"json_schema","schema":{"type":"object","properties":{"b":{"type":"number"}}}}),
            json!({"type":"json_schema","schema":format["json_schema"]["schema"],"native_option":true}),
        ] {
            let expected =
                contract(&json!({"response_format":format,"output_config":{"format":extra}}));
            assert!(native_contract(&expected, &native).is_err());
        }
    }

    #[test]
    fn message_level_schemas_and_advisory_choices_are_guarded() {
        for body in [
            json!({"messages":[{"role":"user","content":"extract","output_config":{"format":{"type":"json_schema","schema":{"type":"object"}}}}]}),
            json!({"tools":[{"type":"function","function":{"name":"extract","parameters":{"type":"object"}}}],"tool_choice":"required"}),
        ] {
            let expected = contract(&body);
            assert!(expected.is_some());
            assert!(cache_version(&body).is_some());
            assert!(ensure_preserved(&expected, &json!({})).is_err());
            assert!(native_contract(&expected, &json!({"messages":[]})).is_err());
        }
    }

    #[test]
    fn native_omission_of_tools_satisfies_none_but_policy_cannot_reenable_tools() {
        let expected = contract(
            &json!({"tools":[{"type":"function","function":{"name":"extract","strict":true,"parameters":{"type":"object"}}}],"tool_choice":"none","parallel_tool_calls":false}),
        );
        let native = native_contract(&expected, &json!({"messages":[]})).unwrap();
        ensure_preserved(&native, &json!({"messages":[]})).unwrap();
        assert!(ensure_preserved(&native,&json!({"toolConfig":{"tools":[{"toolSpec":{"name":"injected","inputSchema":{"json":{"type":"object"}}}}],"toolChoice":{"any":{}}}})).is_err());
    }

    #[test]
    fn invalid_schema_wrapper_is_not_reinterpreted_as_a_direct_schema() {
        let expected = contract(
            &json!({"response_format":{"type":"json_schema","json_schema":{"schema":false}}}),
        );
        let different =
            json!({"output_config":{"format":{"type":"json_schema","schema":{"schema":false}}}});
        assert!(native_contract(&expected, &different).is_err());
    }

    #[test]
    fn omitted_auto_with_no_tools_satisfies_serial_control_but_remains_guarded() {
        for tools in [None, Some(serde_json::json!([]))] {
            let mut request = serde_json::json!({"tool_choice":"auto","parallel_tool_calls":false});
            if let Some(tools) = tools {
                request["tools"] = tools;
            }
            let expected = contract(&request);
            let native = native_contract(&expected, &serde_json::json!({"messages":[]})).unwrap();
            ensure_preserved(&native, &serde_json::json!({"messages":[]})).unwrap();
            assert!(ensure_preserved(&native,&serde_json::json!({"tools":[{"type":"function","function":{"name":"injected"}}]})).is_err());
        }
    }
}
