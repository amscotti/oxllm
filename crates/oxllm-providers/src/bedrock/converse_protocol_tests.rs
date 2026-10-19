use super::bedrock_deployment;
use crate::bedrock::{BedrockProvider, converse};
use oxllm_core::Error;
use oxllm_core::provider::ChatProvider;
use serde_json::{Map, Value, json};

fn transform(model: &str, request: &Value) -> Value {
    converse::transform_request_converse(request, model, &Map::new()).unwrap()
}

fn assert_system_cache_ttls(first: &str, second: &str) {
    let request = json!({"messages": [
        {"role": "system", "content": [
            {"type": "text", "text": "first", "cache_control": {"type": "ephemeral", "ttl": first}},
            {"type": "text", "text": "unmarked"},
            {"type": "text", "text": "second", "cache_control": {"type": "ephemeral", "ttl": second}},
        ]},
        {"role": "user", "content": "hello"},
    ]});
    let original = request.clone();
    for (model, explicit_ttl) in [
        ("anthropic.claude-sonnet-4-5-20250929-v1:0", true),
        ("anthropic.claude-sonnet-4-20250514-v1:0", false),
    ] {
        let body = transform(model, &request);
        let point = |ttl: &str| {
            if explicit_ttl {
                json!({"cachePoint": {"type": "default", "ttl": ttl}})
            } else {
                json!({"cachePoint": {"type": "default"}})
            }
        };
        assert_eq!(
            body["system"],
            json!([
                {"text": "first"}, point(first), {"text": "unmarked"},
                {"text": "second"}, point(second),
            ]),
            "{model}: each breakpoint must retain its own TTL in source order",
        );
    }
    assert_eq!(request, original);
}

#[test]
fn converse_system_cache_ttls_preserve_5m_then_1h() {
    assert_system_cache_ttls("5m", "1h");
}

#[test]
fn converse_system_cache_ttls_preserve_1h_then_5m() {
    assert_system_cache_ttls("1h", "5m");
}

#[test]
fn stop_suppression_recognizes_versions_regions_and_resource_ids() {
    let cases = [
        ("xai.grok-4.6", true),
        ("bedrock/global.xai.grok-4.7", true),
        ("eu.xai.grok-4.6:0", true),
        ("openai.gpt-5.6-sol", true),
        ("bedrock/us.openai.gpt-5.6-terra", true),
        ("apac.openai.gpt-5.10-sol", true),
        ("openai.gpt-6", true),
        ("openai.gpt-6.1-sol:0", true),
        ("us-east-1.openai.gpt-10.2-sol", true),
        ("BEDROCK/GLOBAL.OPENAI.GPT-6.1-SOL", true),
        (
            "arn:aws:bedrock:us-east-1::foundation-model/openai.gpt-6.1-sol:0",
            true,
        ),
        (
            "bedrock/arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.xai.grok-4.7",
            true,
        ),
        ("openai.gpt-5.5", false),
        ("openai.gpt-5.4", false),
        ("openai.gpt-5", false),
        ("openai.gpt-4.10", false),
        ("openai.gpt-oss-120b", false),
        ("anthropic.claude-sonnet-5-5", false),
        ("amazon.nova-pro-v1:0", false),
        ("mistral.mistral-large-3", false),
        ("custom.openai.gpt-6.1-sol", false),
        ("custom.xai.grok-4.7", false),
        ("notopenai.gpt-6.1-sol", false),
        ("openai.gpt-5.6oops", false),
        ("openai.gpt-6oops", false),
        ("openai.gpt-5..6", false),
        ("openai.gpt-5.6.1", false),
        ("openai.gpt-+6", false),
        ("openai.gpt-999999999999999999999999999999", false),
        (
            "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/openai.gpt-6.1-sol",
            false,
        ),
        (
            "arn:aws:bedrock:us-east-1:123456789012:custom-model/openai.gpt-6.1-sol",
            false,
        ),
    ];
    for (model, suppress) in cases {
        for stop in [json!("END"), json!(["END", "DONE"])] {
            let request = json!({
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 32,
                "stop": stop,
            });
            let original = request.clone();
            let body = transform(model, &request);
            let sequences = body["inferenceConfig"].get("stopSequences");
            if suppress {
                assert!(sequences.is_none(), "{model}: {body}");
            } else {
                let expected = if stop.is_string() {
                    json!([stop])
                } else {
                    stop
                };
                assert_eq!(sequences, Some(&expected), "{model}: {body}");
            }
            assert_eq!(body["inferenceConfig"]["maxTokens"], 32, "{model}");
            assert_eq!(request, original);
        }
    }
}

#[test]
fn provider_converse_path_suppresses_stop_for_default_gpt_and_explicit_grok() {
    for model in ["bedrock/us.openai.gpt-6.1-sol", "bedrock/xai.grok-4.7"] {
        let mut deployment = bedrock_deployment(model);
        if model.contains("grok") {
            deployment
                .oxllm_params
                .extra
                .insert("use_converse_api".into(), json!(true));
        }
        let request = json!({
            "messages": [{"role": "user", "content": "hello"}],
            "stop": ["END"],
        });
        assert!(
            BedrockProvider
                .endpoint_url(&deployment)
                .ends_with("/converse")
        );
        let body = BedrockProvider
            .transform_request(&request, &deployment)
            .unwrap();
        assert!(body["inferenceConfig"].get("stopSequences").is_none());
    }
}

fn tool_message(id: &str, content: Value) -> Value {
    json!({"role": "tool", "tool_call_id": id, "content": content})
}

fn image(format: &str, bytes: &str) -> Value {
    json!({"type": "image_url", "image_url": {"url": format!("data:image/{format};base64,{bytes}")}})
}

#[test]
fn supported_families_keep_tool_images_nested_in_content_order() {
    let parts = json!([
        {"type": "text", "text": "first"},
        image("png", "QQ=="),
        {"type": "tool_reference", "tool_name": "lookup.details"},
        {"type": "text", "text": "second"},
        image("jpeg", "Qg=="),
    ]);
    let request = json!({"messages": [tool_message("call_original", parts)]});
    let original = request.clone();
    for model in [
        "bedrock/us.anthropic.claude-sonnet-5-5",
        "amazon.nova-pro-v1:0",
        "mistral.mistral-large-3",
        "openai.gpt-oss-120b",
        "custom.vendor-model",
    ] {
        let body = transform(model, &request);
        assert_eq!(
            body["messages"],
            json!([{"role": "user", "content": [{"toolResult": {
                "toolUseId": "call_original",
                "content": [
                    {"text": "first"},
                    {"image": {"format": "png", "source": {"bytes": "QQ=="}}},
                    {"text": "lookup.details"},
                    {"text": "second"},
                    {"image": {"format": "jpeg", "source": {"bytes": "Qg=="}}},
                ],
            }}]}]),
            "{model}",
        );
    }
    assert_eq!(request, original);
}

#[test]
fn gpt_images_follow_their_result_without_reordering_parallel_results_or_cache() {
    let mut first = tool_message(
        "call_first",
        json!([
            {"type": "text", "text": "first"},
            image("png", "QQ=="),
            {"type": "text", "text": "second"},
            image("jpeg", "Qg=="),
            {"type": "tool_reference", "tool_name": "lookup.more"},
        ]),
    );
    first["status"] = json!("error");
    first["cache_control"] = json!({"type": "ephemeral"});
    let request = json!({"messages": [
        first,
        tool_message("call_second", json!("a separate result")),
    ]});
    for model in [
        "openai.gpt-5.4",
        "us.openai.gpt-5.5",
        "bedrock/global.openai.gpt-5.6-sol",
        "global.openai.gpt-6.1-sol",
    ] {
        let body = transform(model, &request);
        assert_eq!(
            body["messages"],
            json!([{"role": "user", "content": [
                {"toolResult": {
                    "toolUseId": "call_first", "status": "error",
                    "content": [{"text": "first"}, {"text": "second"}, {"text": "lookup.more"}],
                }},
                {"image": {"format": "png", "source": {"bytes": "QQ=="}}},
                {"image": {"format": "jpeg", "source": {"bytes": "Qg=="}}},
                {"cachePoint": {"type": "default"}},
                {"toolResult": {"toolUseId": "call_second", "content": [{"text": "a separate result"}]}},
            ]}]),
            "{model}"
        );
    }
}

#[test]
fn image_only_tool_results_are_valid_for_nested_and_sibling_placement() {
    let request = json!({"messages": [tool_message("call_image", json!([
        image("png", "QQ=="),
        {"type": "image_url", "image_url": "data:image/webp;base64,Qg=="},
    ]))]});
    for (model, siblings) in [
        ("anthropic.claude-sonnet-5-5", false),
        ("openai.gpt-5.5", true),
        ("global.openai.gpt-6.1-sol", true),
    ] {
        let body = transform(model, &request);
        let blocks = body["messages"][0]["content"].as_array().unwrap();
        let result = &blocks[0]["toolResult"];
        assert_eq!(result["toolUseId"], "call_image");
        let expected_images = json!([
            {"image": {"format": "png", "source": {"bytes": "QQ=="}}},
            {"image": {"format": "webp", "source": {"bytes": "Qg=="}}},
        ]);
        if siblings {
            assert_eq!(result["content"].as_array().unwrap().len(), 1);
            assert!(
                result["content"][0]["text"]
                    .as_str()
                    .is_some_and(|text| !text.is_empty())
            );
            assert_eq!(&blocks[1..], expected_images.as_array().unwrap());
        } else {
            assert_eq!(blocks.len(), 1);
            assert_eq!(result["content"], expected_images);
        }
    }
}

#[test]
fn text_only_and_reference_only_tool_results_keep_their_text_and_ids() {
    for (input, expected) in [
        (json!("plain result"), json!([{"text": "plain result"}])),
        (
            json!([{"type": "text", "text": "one"}, {"type": "text", "text": "two"}]),
            json!([{"text": "one"}, {"text": "two"}]),
        ),
        (
            json!([{"type": "tool_reference", "tool_name": "lookup.one"}, {"type": "tool_reference", "tool_name": "lookup.two"}]),
            json!([{"text": "lookup.one"}, {"text": "lookup.two"}]),
        ),
    ] {
        for model in ["anthropic.claude-sonnet-5-5", "openai.gpt-6.1-sol"] {
            let request = json!({"messages": [tool_message("call_text", input.clone())]});
            let body = transform(model, &request);
            assert_eq!(
                body["messages"][0]["content"],
                json!([{"toolResult": {
                    "toolUseId": "call_text", "content": expected,
                }}])
            );
        }
    }
}

#[test]
fn empty_or_unsupported_tool_content_keeps_a_nonempty_result() {
    for content in [
        Value::Null,
        json!(""),
        json!([]),
        json!([{"type": "text", "text": ""}]),
        json!([{"type": "unknown"}, {"type": "tool_reference", "tool_name": 42}]),
    ] {
        for model in ["anthropic.claude-sonnet-5-5", "openai.gpt-6.1-sol"] {
            let body = transform(
                model,
                &json!({"messages": [tool_message("call_empty", content.clone())]}),
            );
            let blocks = body["messages"][0]["content"].as_array().unwrap();
            assert_eq!(blocks.len(), 1);
            let result = &blocks[0]["toolResult"];
            assert_eq!(result["toolUseId"], "call_empty");
            assert_eq!(result["content"].as_array().unwrap().len(), 1);
            assert!(
                result["content"][0]["text"]
                    .as_str()
                    .is_some_and(|text| !text.is_empty())
            );
        }
    }
}

#[test]
fn malformed_tool_images_are_rejected_instead_of_being_dropped() {
    for image in [
        json!({"type": "image_url"}),
        json!({"type": "image_url", "image_url": null}),
        json!({"type": "image_url", "image_url": {"url": 1}}),
        json!({"type": "image_url", "image_url": {"url": "https://images.example/image.png"}}),
        json!({"type": "image_url", "image_url": "data:image/png,percent-encoded"}),
        json!({"type": "image_url", "image_url": "data:image/png;base64,"}),
        json!({"type": "image_url", "image_url": "data:;base64,QQ=="}),
        json!({"type": "image_url", "image_url": "data:application/octet-stream;base64,QQ=="}),
    ] {
        let request = json!({"messages": [tool_message("call_bad", json!([
            {"type": "text", "text": "keep this"}, image,
        ]))]});
        let original = request.clone();
        for model in ["anthropic.claude-sonnet-5-5", "openai.gpt-6.1-sol"] {
            assert!(
                matches!(
                    converse::transform_request_converse(&request, model, &Map::new()),
                    Err(Error::BadRequest(_))
                ),
                "{model}: {request}"
            );
        }
        assert_eq!(request, original);
    }
}

#[test]
fn opaque_profiles_use_declared_family_hints_for_stop_and_image_placement() {
    let model = "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/profile";
    let request = json!({"stop": "END", "messages": [tool_message("call_profile", json!([image("png", "QQ==")]))]});
    for hint in [
        None,
        Some("bedrock/global.openai.gpt-6.1-sol"),
        Some("anthropic.claude-sonnet-5-5"),
    ] {
        let mut extra = Map::new();
        if let Some(hint) = hint {
            extra.insert("base_model".into(), json!(hint));
        }
        let body = converse::transform_request_converse(&request, model, &extra).unwrap();
        let gpt = hint.is_some_and(|hint| hint.contains("openai.gpt-"));
        assert_eq!(body["inferenceConfig"].get("stopSequences").is_none(), gpt);
        let blocks = body["messages"][0]["content"].as_array().unwrap();
        assert_eq!(blocks.len(), if gpt { 2 } else { 1 });
        if !gpt {
            assert!(blocks[0]["toolResult"]["content"][0].get("image").is_some());
        }
        assert!(body.get("model").is_none());
    }
}

#[test]
fn tool_documents_reuse_media_parsing_and_stay_nested_on_gpt() {
    let request = json!({"messages": [tool_message("call_pdf", json!([
        {"type": "image_url", "image_url": {"url": "data:application/pdf;base64,JVBERi0="}},
        {"type": "file", "file": {"file_data": "data:application/pdf;base64,JVBERi0=", "filename": "report"}},
    ]))]});
    let body = transform("openai.gpt-6.1-sol", &request);
    assert_eq!(body["messages"][0]["content"].as_array().unwrap().len(), 1);
    let parts = body["messages"][0]["content"][0]["toolResult"]["content"]
        .as_array()
        .unwrap();
    assert_eq!(parts.len(), 2);
    for part in parts {
        assert_eq!(part["document"]["format"], "pdf");
        assert_eq!(part["document"]["source"]["bytes"], "JVBERi0=");
        assert!(part.get("image").is_none());
    }
}

#[test]
fn tool_files_reject_remote_and_malformed_base64_without_dropping_content() {
    for data in [
        "https://documents.example/report.pdf",
        "file-remote-id",
        "not base64!",
        "",
        "A",
        "SGVsbG8===",
        "data:application/pdf;base64,not base64!",
    ] {
        for model in ["anthropic.claude-sonnet-5-5", "openai.gpt-6.1-sol"] {
            let request = json!({"messages":[tool_message("call_file", json!([
                {"type":"text","text":"keep this"},
                {"type":"file","file":{"file_data":data,"filename":"report"}}
            ]))]});
            assert!(
                matches!(
                    converse::transform_request_converse(&request, model, &Map::new()),
                    Err(Error::BadRequest(_))
                ),
                "{model}: malformed file data must fail locally: {data:?}"
            );
        }
    }
    for data in ["JVBERi0=", "data:application/pdf;base64,JVBERi0="] {
        let body = transform(
            "openai.gpt-6.1-sol",
            &json!({"messages":[tool_message("call_file", json!([
                {"type":"file","file":{"file_data":data,"filename":"report"}}
            ]))]}),
        );
        assert_eq!(
            body["messages"][0]["content"][0]["toolResult"]["content"][0]["document"]["source"]["bytes"],
            "JVBERi0="
        );
    }
}

#[test]
fn post_4_6_claude_sampling_and_forced_tool_choice_are_gated() {
    // Claude models released after Opus 4.6 removed sampling control:
    // only temperature 1 / top_p >= 0.99 may reach inferenceConfig.
    let request = json!({
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.7,
        "top_p": 0.5,
        "tools": [{
            "type": "function",
            "function": {"name": "lookup", "parameters": {"type": "object", "properties": {}}}
        }],
        "tool_choice": "required"
    });
    let body = transform("us.anthropic.claude-sonnet-5-5", &request);
    assert!(
        body["inferenceConfig"].get("temperature").is_none(),
        "{body}"
    );
    assert!(body["inferenceConfig"].get("topP").is_none(), "{body}");
    // Sonnet 5.5 also rejects forced tool use: downgraded to auto.
    assert_eq!(
        body["toolConfig"]["toolChoice"],
        json!({"auto": {}}),
        "{body}"
    );

    let defaults = json!({
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 1.0,
        "top_p": 0.995
    });
    let body = transform("us.anthropic.claude-sonnet-5-5", &defaults);
    assert_eq!(body["inferenceConfig"]["temperature"], 1.0);
    assert_eq!(body["inferenceConfig"]["topP"], 0.995);

    // Pre-4.7 Claude keeps sampling + forced tool use verbatim.
    let body = transform("anthropic.claude-sonnet-4-20250514-v1:0", &request);
    assert_eq!(body["inferenceConfig"]["temperature"], 0.7);
    assert_eq!(body["inferenceConfig"]["topP"], 0.5);
    assert_eq!(body["toolConfig"]["toolChoice"], json!({"any": {}}));

    // Haiku 5.5 keeps forced tool use but still rejects sampling.
    let body = transform("us.anthropic.claude-haiku-5-5", &request);
    assert!(
        body["inferenceConfig"].get("temperature").is_none(),
        "{body}"
    );
    assert_eq!(body["toolConfig"]["toolChoice"], json!({"any": {}}));

    // top_k rides additionalModelRequestFields on Converse; it must be
    // gated for the same post-4.6 models, and a caller-supplied native
    // forced tool_choice in additionalModelRequestFields must not
    // bypass the downgrade.
    let top_k_request = json!({
        "messages": [{"role": "user", "content": "hi"}],
        "top_k": 3
    });
    let body = transform("us.anthropic.claude-sonnet-5-5", &top_k_request);
    assert!(
        body.pointer("/additionalModelRequestFields/top_k")
            .is_none(),
        "top_k must be omitted post-4.6: {body}"
    );
    let body = transform("anthropic.claude-sonnet-4-20250514-v1:0", &top_k_request);
    assert_eq!(body["additionalModelRequestFields"]["top_k"], 3);

    let native_forced = json!({
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{
            "type": "function",
            "function": {"name": "lookup", "parameters": {"type": "object", "properties": {}}}
        }],
        "additionalModelRequestFields": {"tool_choice": {"type": "any"}}
    });
    let body = transform("us.anthropic.claude-sonnet-5-5", &native_forced);
    assert_eq!(
        body["additionalModelRequestFields"]["tool_choice"],
        json!({"type": "auto"}),
        "native forced choice must downgrade: {body}"
    );
    let body = transform("anthropic.claude-sonnet-4-20250514-v1:0", &native_forced);
    assert_eq!(
        body["additionalModelRequestFields"]["tool_choice"],
        json!({"type": "any"}),
        "pre-4.7 models keep native forced choice: {body}"
    );
}
