//! Explicit OpenAI cache markers on deployments that advertise the dialect.

use std::borrow::Cow;

use oxllm_core::deployment::{CacheControlInjectionPoint, Deployment};
use serde_json::{Value, json};

pub(crate) fn apply(body: &mut Value, request: &Value, deployment: &Deployment) {
    let Some(object) = body.as_object_mut() else {
        return;
    };
    object.remove("cache_control_injection_points");
    if deployment
        .oxllm_params
        .cache_control_injection_points
        .as_ref()
        .is_none_or(Vec::is_empty)
        && request
            .get("cache_control_injection_points")
            .is_none_or(Value::is_null)
        && request
            .get("prompt_cache_options")
            .is_none_or(Value::is_null)
    {
        return;
    }
    if !eligible(body, request, deployment) {
        return;
    }
    if let Some(options) = request
        .get("prompt_cache_options")
        .filter(|value| !value.is_null())
    {
        body["prompt_cache_options"] = options.clone();
    }
    let points: Cow<'_, [CacheControlInjectionPoint]> =
        match request.get("cache_control_injection_points") {
            Some(value) if !value.is_null() => Cow::Owned(
                value
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(|point| {
                        // Request points may carry Anthropic TTL controls or
                        // internal stamps. Only the selector is relevant to
                        // this dialect; deployment deserialization stays strict.
                        serde_json::from_value::<CacheControlInjectionPoint>(json!({
                            "location": point.get("location")?,
                            "role": point.get("role"),
                            "index": point.get("index"),
                        }))
                        .ok()
                    })
                    .filter(|point| point.validate().is_ok())
                    .collect(),
            ),
            _ => Cow::Borrowed(
                deployment
                    .oxllm_params
                    .cache_control_injection_points
                    .as_deref()
                    .unwrap_or_default(),
            ),
        };
    if points.is_empty() {
        return;
    }
    let Some(messages) = body.get_mut("messages").and_then(Value::as_array_mut) else {
        return;
    };
    let mut used = messages.iter().map(count_marks).sum::<usize>();
    let mut inserted = false;
    for point in points.iter() {
        for index in 0..messages.len() {
            if used >= 4 {
                break;
            }
            let selected = point
                .role
                .as_deref()
                .is_some_and(|role| messages[index]["role"] == role)
                || point.index.is_some_and(|position| {
                    let position = if position < 0 {
                        messages.len() as i64 + position
                    } else {
                        position
                    };
                    position >= 0 && usize::try_from(position).ok() == Some(index)
                });
            if !selected
                || messages[index]["role"] == "assistant"
                || count_marks(&messages[index]) > 0
            {
                continue;
            }
            if insert_mark(&mut messages[index]) {
                used += 1;
                inserted = true;
            }
        }
    }
    if inserted && body.get("prompt_cache_options").is_none_or(Value::is_null) {
        body["prompt_cache_options"] = json!({"mode":"implicit"});
    }
}

fn eligible(body: &Value, request: &Value, deployment: &Deployment) -> bool {
    let provider = deployment.provider_slug();
    let override_flag = deployment
        .oxllm_params
        .extra
        .get("supports_prompt_cache_breakpoint")
        .filter(|flag| !flag.is_null())
        .map(|flag| flag.as_bool() == Some(true));
    let model = body
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or(&deployment.oxllm_params.model);
    let prefix = format!("{provider}/");
    let configured = &deployment.oxllm_params.model;
    let hint = deployment
        .model_info
        .as_ref()
        .and_then(|info| info.base_model.as_deref())
        .or_else(|| {
            deployment
                .oxllm_params
                .extra
                .get("base_model")
                .and_then(Value::as_str)
        });
    let candidates = [configured.as_str(), model, hint.unwrap_or("")];
    let flag = override_flag.or_else(|| {
        candidates
            .iter()
            .filter(|model| !model.is_empty() && !model.contains('*'))
            .find_map(|model| {
                let model = model.strip_prefix(&prefix).unwrap_or(model);
                let scoped = format!("{prefix}{model}");
                let table = oxllm_core::cost::price_table();
                if let Some(price) = table.get(&scoped) {
                    // A hosted deployment's own card is authoritative, even
                    // without an opt-in flag (notably Mantle GovCloud cards).
                    return if provider == "openai" {
                        price.supports_prompt_cache_breakpoint
                    } else {
                        Some(price.supports_prompt_cache_breakpoint == Some(true))
                    };
                }
                (provider == "bedrock_mantle")
                    .then(|| {
                        let regional =
                            format!("{prefix}{}", model.rsplit('/').next().unwrap_or(model));
                        table
                            .get(&regional)
                            .map(|price| price.supports_prompt_cache_breakpoint == Some(true))
                    })
                    .flatten()
            })
    });
    if provider != "openai" {
        return flag == Some(true);
    }
    let supported = flag.unwrap_or_else(|| {
        oxllm_core::lookup_pricing_with_provider(model, provider)
            .and_then(|price| price.supports_prompt_cache_breakpoint)
            .unwrap_or_else(|| crate::openai::supports_explicit_prompt_cache(model))
    });
    supported
        && (request
            .get("prompt_cache_options")
            .is_some_and(|value| !value.is_null())
            || deployment
                .oxllm_params
                .api_base
                .as_deref()
                .is_none_or(|base| {
                    url::Url::parse(base)
                        .ok()
                        .and_then(|url| url.host_str().map(str::to_owned))
                        .is_some_and(|host| {
                            host == "api.openai.com" || host.ends_with(".api.openai.com")
                        })
                }))
}

fn carries_mark(value: &Value) -> bool {
    ["cache_control", "prompt_cache_breakpoint"]
        .iter()
        .any(|field| value.get(*field).is_some_and(|value| !value.is_null()))
}

fn count_marks(message: &Value) -> usize {
    // Only message/block annotations count. Schema properties, tool arguments
    // and provider metadata with the same key names are ordinary prompt data.
    // External Anthropic automatic/tool controls do not reserve OpenAI slots;
    // a bridged system prompt is already one of these messages.
    usize::from(carries_mark(message))
        + ["content", "tool_calls"]
            .iter()
            .filter_map(|field| message.get(*field).and_then(Value::as_array))
            .flatten()
            .filter(|block| carries_mark(block))
            .count()
}

fn insert_mark(message: &mut Value) -> bool {
    let Some(content) = message.get_mut("content") else {
        return false;
    };
    if let Value::String(text) = content {
        *content =
            json!([{"type":"text", "text":text, "prompt_cache_breakpoint":{"mode":"explicit"}}]);
        return true;
    }
    let target = content.as_array_mut().and_then(|blocks| {
        blocks.iter_mut().rev().find(|block| {
            matches!(
                block.get("type").and_then(Value::as_str),
                Some(
                    "text"
                        | "image"
                        | "image_url"
                        | "file"
                        | "input_audio"
                        | "input_text"
                        | "input_image"
                        | "input_file"
                )
            )
        })
    });
    if let Some(block) = target {
        block["prompt_cache_breakpoint"] = json!({"mode":"explicit"});
        return true;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn deployment(model: &str) -> Deployment {
        let mut deployment = Deployment::default();
        deployment.oxllm_params.model = model.into();
        deployment.oxllm_params.cache_control_injection_points =
            Some(vec![CacheControlInjectionPoint {
                location: "message".into(),
                role: Some("system".into()),
                index: None,
            }]);
        deployment
    }

    #[test]
    fn hosted_flags_are_provider_scoped_strict_and_support_regional_mantle_ids() {
        let request = json!({"messages":[{"role":"system","content":"stable prefix"}], "prompt_cache_options":null});
        for model in [
            "bedrock_mantle/openai.gpt-6.1-sol",
            "bedrock_mantle/us-east-1/openai.gpt-5.6-sol",
            "azure_ai/gpt-6-sol",
        ] {
            let mut dep = deployment(model);
            let mut body = request.clone();
            apply(&mut body, &request, &dep);
            assert_eq!(
                body["messages"][0]["content"][0]["prompt_cache_breakpoint"]["mode"], "explicit",
                "{model}: {body}"
            );
            assert_eq!(body["prompt_cache_options"]["mode"], "implicit");
            for flag in [json!(false), json!("true"), json!(1), json!("false")] {
                dep.oxllm_params
                    .extra
                    .insert("supports_prompt_cache_breakpoint".into(), flag);
                let mut body = request.clone();
                apply(&mut body, &request, &dep);
                assert!(body["messages"][0]["content"].is_string());
            }
        }
        let mut body = request.clone();
        apply(&mut body, &request, &deployment("groq/gpt-6-sol"));
        assert!(
            body["messages"][0]["content"].is_string(),
            "another provider's row cannot opt Groq in"
        );
    }

    #[test]
    fn marks_preserve_client_options_skip_assistants_and_respect_the_four_block_limit() {
        let request = json!({"cache_control_injection_points":["junk", {"location":"message","role":"user"}, 1],
        "prompt_cache_options":{"mode":"explicit"}, "messages":[
            {"role":"user","content":[{"type":"text","text":"one","prompt_cache_breakpoint":{"mode":"explicit","custom":"kept"}}]},
            {"role":"assistant","content":"assistant"},
            {"role":"user","content":"two"}, {"role":"user","content":"three"},
            {"role":"user","content":"four"}, {"role":"user","content":"five"}
        ]});
        let mut body = request.clone();
        apply(
            &mut body,
            &request,
            &deployment("bedrock_mantle/openai.gpt-6.1-sol"),
        );
        assert_eq!(
            body["messages"]
                .as_array()
                .unwrap()
                .iter()
                .map(count_marks)
                .sum::<usize>(),
            4
        );
        assert_eq!(
            body["messages"][0]["content"][0]["prompt_cache_breakpoint"]["custom"],
            "kept"
        );
        assert!(body["messages"][1]["content"].is_string());
        assert!(body["messages"][5]["content"].is_string());
        assert_eq!(body["prompt_cache_options"]["mode"], "explicit");
        assert!(body.get("cache_control_injection_points").is_none());
    }

    #[test]
    fn an_openai_compatible_host_needs_explicit_options_to_accept_the_dialect() {
        let mut dep = deployment("openai/gpt-6.1-sol");
        dep.oxllm_params.api_base = Some("https://api.openai.com@other.example/v1".into());
        let mut request =
            json!({"model":"gpt-6.1-sol", "messages":[{"role":"system","content":"stable"}]});
        let mut body = request.clone();
        apply(&mut body, &request, &dep);
        assert!(body["messages"][0]["content"].is_string());
        request["prompt_cache_options"] = json!({"mode":"implicit"});
        let mut body = request.clone();
        apply(&mut body, &request, &dep);
        assert_eq!(
            body["messages"][0]["content"][0]["prompt_cache_breakpoint"]["mode"],
            "explicit"
        );
    }

    #[test]
    fn request_points_ignore_decorations_without_relaxing_deployment_validation() {
        let request = json!({
            "cache_control_injection_points": [
                {"location":"message", "role":"system", "control":{"type":"ephemeral", "ttl":"1h"},
                    "_litellm_openai_dialect":true, "_litellm_external_breakpoints":4},
                {"location":"message", "index":-1, "control":"unsupported", "future_field":true}
            ],
            "messages":[{"role":"system", "content":"stable"}, {"role":"user", "content":"question"}]
        });
        assert!(
            serde_json::from_value::<CacheControlInjectionPoint>(
                request["cache_control_injection_points"][0].clone()
            )
            .is_err(),
            "deployment selectors must still reject unknown fields"
        );
        for model in [
            "openai/gpt-6.1-sol",
            "bedrock_mantle/openai.gpt-6.1-sol",
            "azure_ai/gpt-6-sol",
        ] {
            let mut body = request.clone();
            apply(&mut body, &request, &deployment(model));
            for message in body["messages"].as_array().unwrap() {
                assert_eq!(
                    message["content"][0]["prompt_cache_breakpoint"],
                    json!({"mode":"explicit"}),
                    "{model}: Anthropic TTL decorations must not change the OpenAI dialect"
                );
            }
            assert_eq!(body["prompt_cache_options"], json!({"mode":"implicit"}));
        }
    }

    #[test]
    fn request_point_validation_and_precedence_survive_lenient_decorations() {
        let dep = deployment("bedrock_mantle/openai.gpt-6.1-sol");
        let messages = json!([
            {"role":"system", "content":"configured target"},
            {"role":"user", "content":"request target"}
        ]);
        let invalid_points = json!([
            null, false, 1, "junk", [], {},
            {"location":"tool", "index":0},
            {"location":"message", "role":"developer"},
            {"location":"message", "role":1},
            {"location":"message", "index":1.5},
            {"location":"message", "index":"1"},
            {"location":"message", "role":"system", "index":0},
            {"location":"message", "role":null, "index":null}
        ]);
        for points in [
            json!(false),
            json!("junk"),
            json!({}),
            json!([]),
            invalid_points.clone(),
        ] {
            let request = json!({"messages":messages, "cache_control_injection_points":points});
            let mut body = request.clone();
            apply(&mut body, &request, &dep);
            assert_eq!(
                body["messages"], messages,
                "malformed request lists must not inherit deployment points"
            );
            assert!(body.get("prompt_cache_options").is_none());
            assert!(body.get("cache_control_injection_points").is_none());
        }
        let mut points = invalid_points.as_array().unwrap().clone();
        points.push(json!({"location":"message", "index":1, "control":{"ttl":"1h"}}));
        let request = json!({"messages":messages, "cache_control_injection_points":points});
        let mut body = request.clone();
        apply(&mut body, &request, &dep);
        assert_eq!(body["messages"][0], messages[0]);
        assert_eq!(
            body["messages"][1]["content"][0]["prompt_cache_breakpoint"],
            json!({"mode":"explicit"})
        );

        for request in [
            json!({"messages":messages}),
            json!({"messages":messages, "cache_control_injection_points":null}),
        ] {
            let mut body = request.clone();
            apply(&mut body, &request, &dep);
            assert_eq!(
                body["messages"][0]["content"][0]["prompt_cache_breakpoint"],
                json!({"mode":"explicit"})
            );
            assert_eq!(body["messages"][1], messages[1]);
        }
    }

    #[test]
    fn extreme_indexes_and_malformed_message_shapes_are_safe_noops() {
        let dep = deployment("bedrock_mantle/openai.gpt-6.1-sol");
        let messages = json!([
            null, "user", 42, true, [],
            {"role":"assistant", "content":"reply"},
            {"role":"user"}, {"role":"user", "content":null},
            {"role":"user", "content":13},
            {"role":"user", "content":[null, [], {"type":"thinking"}, {"type":{}}, {"type":"text", "text":"valid"}, 7]}
        ]);
        for index in [i64::MIN, i64::MAX, 0, 1, 2, 3, 4, 5, 6, 7, 8] {
            let request = json!({"messages":messages, "cache_control_injection_points":[{"location":"message", "index":index}]});
            let mut body = request.clone();
            apply(&mut body, &request, &dep);
            assert_eq!(body["messages"], messages, "index {index}");
            assert!(body.get("prompt_cache_options").is_none());
        }
        let request = json!({"messages":messages, "cache_control_injection_points":[{"location":"message", "index":-1}]});
        let mut body = request.clone();
        apply(&mut body, &request, &dep);
        assert_eq!(
            body["messages"][9]["content"][4]["prompt_cache_breakpoint"],
            json!({"mode":"explicit"})
        );
        assert_eq!(body["messages"][9]["content"][5], 7);
        for messages in [json!(null), json!("junk"), json!({})] {
            let request = json!({"messages":messages});
            let mut body = request.clone();
            apply(&mut body, &request, &dep);
            assert_eq!(body, request);
        }
    }

    #[test]
    fn exact_unflagged_regional_cards_outrank_commercial_flags_and_model_hints() {
        for leaf in ["openai.gpt-5.6-luna", "openai.gpt-5.6-terra"] {
            let model = format!("bedrock_mantle/us-gov-west-1/{leaf}");
            assert_eq!(
                oxllm_core::cost::price_table()[&model].supports_prompt_cache_breakpoint,
                None
            );
            let mut dep = deployment(&model);
            dep.oxllm_params
                .extra
                .insert("base_model".into(), json!(format!("bedrock_mantle/{leaf}")));
            let request = json!({"messages":[{"role":"system", "content":"stable"}]});
            let mut body = request.clone();
            apply(&mut body, &request, &dep);
            assert_eq!(body, request, "{model} must not inherit a commercial flag");
            dep.oxllm_params
                .extra
                .insert("supports_prompt_cache_breakpoint".into(), json!(true));
            apply(&mut body, &request, &dep);
            assert_eq!(
                body["messages"][0]["content"][0]["prompt_cache_breakpoint"],
                json!({"mode":"explicit"})
            );
        }
    }

    #[test]
    fn azure_pricing_regions_do_not_imply_region_prefixed_gpt_routing() {
        assert!(oxllm_core::cost::price_table().contains_key("azure_ai/global/grok-3"));
        let request = json!({"messages":[{"role":"system", "content":"stable"}]});
        for model in [
            "azure_ai/global/grok-3",
            "azure_ai/global/gpt-6-sol",
            "azure_ai/eastus/gpt-6-sol",
            "groq/eastus/gpt-6-sol",
        ] {
            let mut body = request.clone();
            apply(&mut body, &request, &deployment(model));
            assert_eq!(body, request, "{model}");
        }
    }

    #[test]
    fn census_ignores_schema_and_metadata_properties_named_like_controls() {
        let request = json!({
            "cache_control_injection_points":[{"location":"message", "role":"user"}],
            "messages":[
                {"role":"user", "content":"one", "provider_specific_fields":{
                    "cache_control":{}, "nested":{"prompt_cache_breakpoint":{}},
                    "schema":{"properties":{"cache_control":{"type":"string"}, "prompt_cache_breakpoint":{"type":"object"}}}
                }},
                {"role":"user", "content":"two"}, {"role":"user", "content":"three"},
                {"role":"user", "content":"four"}, {"role":"user", "content":"five"}
            ],
            "tools":[{"type":"function", "function":{"name":"lookup", "parameters":{"type":"object",
                "properties":{"cache_control":{"type":"string"}, "prompt_cache_breakpoint":{"type":"object"}}
            }}}]
        });
        let mut body = request.clone();
        apply(
            &mut body,
            &request,
            &deployment("bedrock_mantle/openai.gpt-6.1-sol"),
        );
        for message in &body["messages"].as_array().unwrap()[..4] {
            assert_eq!(
                message["content"][0]["prompt_cache_breakpoint"],
                json!({"mode":"explicit"})
            );
        }
        assert!(body["messages"][4]["content"].is_string());
        assert_eq!(
            body["messages"][0]["provider_specific_fields"],
            request["messages"][0]["provider_specific_fields"]
        );
        assert_eq!(body["tools"], request["tools"]);
    }

    #[test]
    fn openai_cap_counts_system_message_blocks_without_reserving_anthropic_external_controls() {
        use oxllm_core::provider::ChatProvider;
        let provider = crate::openai_compat::OpenAiCompatProvider::new(
            "bedrock_mantle",
            "https://example.test/v1",
            "bedrock_mantle",
            false,
            true,
            true,
        );
        let request = json!({
            "cache_control_injection_points":[{"location":"message", "role":"user"}],
            "cache_control":{"type":"ephemeral"},
            "system":[{"type":"text", "text":"native system", "cache_control":{"type":"ephemeral"}}],
            "tools":[{"type":"function", "cache_control":{"type":"ephemeral"},
                "function":{"name":"lookup", "parameters":{"type":"object", "properties":{}}, "cache_control":{"type":"ephemeral"}}}],
            "messages":[
                {"role":"system", "content":[{"type":"text", "text":"system", "prompt_cache_breakpoint":{"mode":"explicit", "ttl":"30m"}}]},
                {"role":"user", "content":"one"}, {"role":"user", "content":"two"},
                {"role":"user", "content":"three"}, {"role":"user", "content":"four"}
            ]
        });
        let body = provider
            .transform_request(&request, &deployment("bedrock_mantle/openai.gpt-6.1-sol"))
            .unwrap();
        assert!(body.get("system").is_none());
        assert!(body.get("cache_control").is_none());
        assert_eq!(body["messages"][0], request["messages"][0]);
        for message in &body["messages"].as_array().unwrap()[1..4] {
            assert_eq!(
                message["content"][0]["prompt_cache_breakpoint"],
                json!({"mode":"explicit"})
            );
        }
        assert!(body["messages"][4]["content"].is_string());
    }

    #[test]
    fn caller_only_options_and_existing_marks_do_not_synthesize_defaults() {
        use oxllm_core::provider::ChatProvider;
        let provider = crate::openai_compat::OpenAiCompatProvider::new(
            "bedrock_mantle",
            "https://example.test/v1",
            "bedrock_mantle",
            false,
            true,
            true,
        );
        let mut dep = deployment("bedrock_mantle/openai.gpt-6.1-sol");
        dep.oxllm_params.cache_control_injection_points = None;
        for mode in ["implicit", "explicit"] {
            let request = json!({"prompt_cache_options":{"mode":mode, "ttl":"30m"}, "messages":[{"role":"system", "content":"stable"}]});
            let body = provider.transform_request(&request, &dep).unwrap();
            assert_eq!(
                body["prompt_cache_options"],
                request["prompt_cache_options"]
            );
            assert_eq!(body["messages"], request["messages"]);
        }
        dep.oxllm_params.cache_control_injection_points = deployment(&dep.oxllm_params.model)
            .oxllm_params
            .cache_control_injection_points;
        for options in [None, Some(Value::Null)] {
            let mut request = json!({"messages":[{"role":"system", "content":[{"type":"text", "text":"stable", "prompt_cache_breakpoint":{"mode":"explicit", "ttl":"30m"}}]}]});
            if let Some(options) = options {
                request["prompt_cache_options"] = options;
            }
            let body = provider.transform_request(&request, &dep).unwrap();
            assert!(body.get("prompt_cache_options").is_none());
            assert_eq!(body["messages"], request["messages"]);
        }
    }
}
