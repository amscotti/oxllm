use super::*;
use oxllm_core::Endpoint;
use oxllm_core::deployment::{Deployment, OxllmParams};
use oxllm_core::provider::{ChatProvider, EmbeddingProvider, Provider, ResolvedCredentials};
use serde_json::json;

fn deployment() -> Deployment {
    Deployment {
        model_name: "dbrx-instruct-071224".to_string(),
        oxllm_params: OxllmParams {
            model: "databricks/dbrx-instruct-071224".to_string(),
            api_key: Some("databricks-key".to_string()),
            ..Default::default()
        },
        ..Default::default()
    }
}

#[test]
fn endpoint_url_uses_default_databricks_base() {
    let p = DatabricksProvider;
    assert_eq!(
        p.endpoint_url(&deployment()),
        "https://api.databricks.com/serving-endpoints/chat/completions"
    );
}

#[test]
fn embedding_url_uses_default_databricks_base() {
    let p = DatabricksProvider;
    assert_eq!(
        p.embedding_url(&deployment()),
        "https://api.databricks.com/serving-endpoints/embeddings"
    );
}

#[test]
fn endpoint_url_respects_workspace_base() {
    // Databricks requires the per-workspace URL; it is supplied via api_base.
    let p = DatabricksProvider;
    let mut dep = deployment();
    dep.oxllm_params.api_base =
        Some("https://my.workspace.cloud.databricks.com/serving-endpoints".to_string());
    assert_eq!(
        p.endpoint_url(&dep),
        "https://my.workspace.cloud.databricks.com/serving-endpoints/chat/completions"
    );
}

#[test]
fn endpoint_url_strips_trailing_slash_from_custom_base() {
    let p = DatabricksProvider;
    let mut dep = deployment();
    dep.oxllm_params.api_base =
        Some("https://my.workspace.cloud.databricks.com/serving-endpoints/".to_string());
    assert_eq!(
        p.endpoint_url(&dep),
        "https://my.workspace.cloud.databricks.com/serving-endpoints/chat/completions"
    );
}

#[test]
fn apply_auth_uses_bearer_token() {
    let p = DatabricksProvider;
    let creds = ResolvedCredentials::from_key("databricks-secret");
    let mut headers = std::collections::HashMap::new();
    ChatProvider::apply_auth(&p, &deployment(), &creds, &mut headers);
    assert_eq!(
        headers.get("authorization").map(String::as_str),
        Some("Bearer databricks-secret")
    );
}

#[test]
fn request_strips_databricks_prefix_from_model() {
    let p = DatabricksProvider;
    let req = json!({
        "model": "dbrx-instruct-071224",
        "messages": [{"role": "user", "content": "How are you?"}],
        "temperature": 0.5,
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["model"], "dbrx-instruct-071224");
    assert_eq!(out["messages"][0]["content"], "How are you?");
    assert_eq!(out["temperature"], 0.5);
}

#[test]
fn request_drops_thinking_only_assistant_turn_after_stripping_internal_fields() {
    // The Messages bridge emits null content for a thinking-only assistant.
    let req = json!({
        "messages": [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": null,
                "thinking_blocks": [{"type": "thinking", "thinking": "hmm", "signature": "sig_1"}],
                "reasoning_content": "hmm"
            },
            {"role": "user", "content": "again"}
        ]
    });
    let original = req.clone();

    let out = ChatProvider::transform_request(&DatabricksProvider, &req, &deployment()).unwrap();

    assert_eq!(req, original, "the caller's history must not be mutated");
    assert_eq!(
        out["messages"],
        json!([
            {"role": "user", "content": "hi"},
            {"role": "user", "content": "again"}
        ])
    );
}

#[test]
fn request_drops_reasoning_only_assistants_for_each_empty_content_shape() {
    for content in [None, Some(json!(null)), Some(json!("")), Some(json!([]))] {
        for mut assistant in [
            json!({"role": "assistant", "reasoning_content": "hmm"}),
            json!({"role": "assistant", "reasoning_content": "hmm", "tool_calls": null, "function_call": null}),
            json!({"role": "assistant", "reasoning_content": "hmm", "tool_calls": [], "function_call": {}}),
        ] {
            if let Some(content) = &content {
                assistant["content"] = content.clone();
            }
            let req = json!({"messages": [
                {"role": "user", "content": "hi"},
                assistant,
                {"role": "user", "content": "again"}
            ]});
            let out =
                ChatProvider::transform_request(&DatabricksProvider, &req, &deployment()).unwrap();
            assert_eq!(
                out["messages"],
                json!([
                    {"role": "user", "content": "hi"},
                    {"role": "user", "content": "again"}
                ]),
                "request: {req}"
            );
        }
    }
}

fn assert_malformed_assistant_field_preserved(field: &str, values: &[Value]) {
    for value in values {
        let mut expected = json!({
            "role": "assistant", "content": null, "tool_calls": [], "function_call": {}
        });
        expected[field] = value.clone();
        let mut assistant = expected.clone();
        assistant["reasoning_content"] = json!("hmm");
        assistant["thinking_blocks"] =
            json!([{"type": "thinking", "thinking": "hmm", "signature": "sig_1"}]);
        let req = json!({"messages": [
            {"role": "user", "content": "hi"},
            assistant,
            {"role": "user", "content": "again"}
        ]});
        let original = req.clone();
        let out =
            ChatProvider::transform_request(&DatabricksProvider, &req, &deployment()).unwrap();
        assert_eq!(req, original, "the caller's history must not be mutated");
        assert_eq!(
            out["messages"],
            json!([
                {"role": "user", "content": "hi"},
                expected,
                {"role": "user", "content": "again"}
            ]),
            "malformed {field}: {value} must survive for upstream validation"
        );
    }
}

#[test]
fn request_preserves_malformed_tool_calls_for_validation() {
    assert_malformed_assistant_field_preserved(
        "tool_calls",
        &[
            json!({}),
            json!(""),
            json!({"id": "call_invalid"}),
            json!("invalid"),
            json!(false),
            json!(true),
            json!(0),
        ],
    );
}

#[test]
fn request_preserves_malformed_function_call_for_validation() {
    assert_malformed_assistant_field_preserved(
        "function_call",
        &[
            json!([]),
            json!(""),
            json!([{"name": "f"}]),
            json!("invalid"),
            json!(false),
            json!(true),
            json!(0),
        ],
    );
}

#[test]
fn request_preserves_malformed_content_for_validation() {
    assert_malformed_assistant_field_preserved(
        "content",
        &[
            json!({}),
            json!({"invalid": "content"}),
            json!(false),
            json!(true),
            json!(0),
        ],
    );
}

#[test]
fn request_strips_all_internal_fields_and_preserves_tool_call_payloads() {
    let arguments = r#"{ "city": "San Francisco", "options": {"days": 2} }"#;
    let req = json!({"messages": [
        {
            "role": "assistant", "content": null,
            "thinking_blocks": [{"type": "thinking", "thinking": "call lookup", "signature": "sig_2"}],
            "reasoning_content": "call lookup",
            "reasoning_items": [{"type": "reasoning", "id": "reasoning_1"}],
            "provider_specific_fields": {"message_metadata": "drop"},
            "tool_calls": [{
                "id": "call_lookup", "type": "function",
                "provider_specific_fields": {"call_metadata": "drop"},
                "function": {
                    "name": "lookup", "arguments": arguments,
                    "provider_specific_fields": {"function_metadata": "drop"}
                }
            }]
        },
        {"role": "tool", "tool_call_id": "call_lookup", "content": "sunny"}
    ]});
    let original = req.clone();
    let out = ChatProvider::transform_request(&DatabricksProvider, &req, &deployment()).unwrap();

    assert_eq!(req, original, "nested caller metadata must not be mutated");
    assert_eq!(
        out["messages"],
        json!([
            {
                "role": "assistant", "content": null,
                "tool_calls": [{"id": "call_lookup", "type": "function", "function": {"name": "lookup", "arguments": arguments}}]
            },
            {"role": "tool", "tool_call_id": "call_lookup", "content": "sunny"}
        ])
    );
}

#[test]
fn request_keeps_tool_and_function_calls_for_each_empty_content_shape() {
    for calls in [
        json!({"role": "assistant", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]}),
        json!({"role": "assistant", "function_call": {"name": "f", "arguments": "{}"}}),
    ] {
        for content in [None, Some(json!(null)), Some(json!("")), Some(json!([]))] {
            let mut expected = calls.clone();
            if let Some(content) = content {
                expected["content"] = content;
            }
            let mut assistant = expected.clone();
            assistant["reasoning_content"] = json!("call f");
            assistant["thinking_blocks"] =
                json!([{"type": "thinking", "thinking": "call f", "signature": "sig_2"}]);
            let req = json!({"messages": [assistant]});
            let out =
                ChatProvider::transform_request(&DatabricksProvider, &req, &deployment()).unwrap();
            assert_eq!(out["messages"], json!([expected]), "request: {req}");
        }
    }
}

#[test]
fn request_preserves_assistant_text_and_multimodal_content() {
    let expected = json!([
        {"role": "assistant", "name": "writer", "content": "Hello!"},
        {"role": "assistant", "content": [{"type": "text", "text": "Hello!", "cache_control": {"type": "ephemeral"}}]},
        {"role": "assistant", "content": [
            {"type": "text", "text": "Look at this"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}}
        ]},
        {"role": "assistant", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}}
        ]}
    ]);
    let mut req = json!({"messages": expected});
    for message in req["messages"].as_array_mut().unwrap() {
        message["reasoning_content"] = json!("hidden reasoning");
    }
    let out = ChatProvider::transform_request(&DatabricksProvider, &req, &deployment()).unwrap();
    assert_eq!(out["messages"], expected);
}

#[test]
fn request_preserves_non_assistant_roles_after_known_metadata_strip() {
    let mut expected = Vec::new();
    for role in ["user", "system", "tool", "developer", "function"] {
        for content in [
            None,
            Some(json!(null)),
            Some(json!("")),
            Some(json!([])),
            Some(json!("keep this text")),
        ] {
            let mut message = json!({"role": role, "name": "keep-name"});
            if let Some(content) = content {
                message["content"] = content;
            }
            if role == "tool" {
                message["tool_call_id"] = json!("call_lookup");
            }
            expected.push(message);
        }
    }
    let mut req = json!({"messages": expected});
    for message in req["messages"].as_array_mut().unwrap() {
        message["thinking_blocks"] = json!([{"type": "thinking", "thinking": "drop"}]);
        message["reasoning_content"] = json!("drop");
        message["reasoning_items"] = json!([{"type": "reasoning", "id": "drop"}]);
        message["provider_specific_fields"] = json!({"metadata": "drop"});
    }
    let original = req.clone();
    let out = ChatProvider::transform_request(&DatabricksProvider, &req, &deployment()).unwrap();
    assert_eq!(out["messages"], json!(expected));
    assert_eq!(req, original);
}

#[test]
fn shared_compat_does_not_strip_fields_or_drop_assistant_turns() {
    let req = json!({"messages": [{
        "role": "assistant", "content": null,
        "thinking_blocks": [{"type": "thinking", "thinking": "hmm", "signature": "sig_1"}],
        "reasoning_content": "hmm"
    }]});
    let out = ChatProvider::transform_request(&COMPAT, &req, &deployment()).unwrap();
    assert_eq!(out["messages"], req["messages"]);
}

#[test]
fn gpt56_tools_without_effort_send_reasoning_none() {
    let p = DatabricksProvider;
    let mut dep = deployment();
    dep.oxllm_params.model = "databricks/gpt-5.6-sol".to_string();
    let req = json!({
        "model": "gpt-5.6-sol",
        "messages": [{"role": "user", "content": "Hi"}],
        "temperature": 0.2,
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(out["model"], "gpt-5.6-sol");
    assert_eq!(out["reasoning_effort"], "none");
    assert_eq!(out["temperature"], 0.2);
}

#[test]
fn embedding_request_strips_databricks_prefix_from_model() {
    let p = DatabricksProvider;
    let mut dep = deployment();
    dep.oxllm_params.model = "databricks/bge-large-en-v1.5".to_string();
    let req = json!({
        "model": "bge-large-en-v1.5",
        "input": ["Hello", "World"],
    });
    let out = EmbeddingProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(out["model"], "bge-large-en-v1.5");
    assert_eq!(out["input"][0], "Hello");
    assert_eq!(out["input"][1], "World");
}

#[test]
fn request_preserves_stream_flag() {
    let p = DatabricksProvider;
    let req = json!({
        "model": "dbrx-instruct-071224",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": true,
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["stream"], true);
}

#[test]
fn response_is_identity() {
    let p = DatabricksProvider;
    let resp = json!({
        "id": "chatcmpl_3f78f09a",
        "object": "chat.completion",
        "model": "dbrx-instruct-071224",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    });
    let out = ChatProvider::transform_response(&p, &resp, "dbrx-instruct-071224").expect("ok");
    assert_eq!(out, resp);
}

#[test]
fn embedding_response_is_identity() {
    let p = DatabricksProvider;
    let resp = json!({
        "object": "list",
        "model": "bge-large-en-v1.5",
        "data": [{"index": 0, "object": "embedding", "embedding": [0.1, -0.2]}],
        "usage": {"prompt_tokens": 8, "total_tokens": 8},
    });
    let out = EmbeddingProvider::transform_response(&p, &resp, "bge-large-en-v1.5").expect("ok");
    assert_eq!(out, resp);
}

#[test]
fn stream_chunk_parses_openai_format() {
    let p = DatabricksProvider;
    let raw = "data: {\"id\":\"chatcmpl\",\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}";
    let out = p
        .transform_stream_chunk(raw)
        .expect("ok")
        .expect("should produce value");
    assert_eq!(out["id"], "chatcmpl");
    assert_eq!(out["choices"][0]["delta"]["content"], "hi");
}

#[test]
fn stream_done_returns_none() {
    let p = DatabricksProvider;
    assert!(
        p.transform_stream_chunk("data: [DONE]")
            .expect("ok")
            .is_none()
    );
}

#[test]
fn provider_id_and_supports() {
    let p = DatabricksProvider;
    assert_eq!(p.id(), "databricks");
    assert!(p.supports(Endpoint::ChatCompletions));
    assert!(p.supports(Endpoint::Embeddings));
}

#[test]
fn strip_databricks_prefix_handles_unknown() {
    assert_eq!(
        strip_databricks_prefix("databricks/dbrx-instruct-071224"),
        "dbrx-instruct-071224"
    );
    assert_eq!(
        strip_databricks_prefix("dbrx-instruct-071224"),
        "dbrx-instruct-071224"
    );
}

#[test]
fn supported_params_published() {
    let p = DatabricksProvider;
    let params = p.supported_params().expect("should publish params");
    assert!(params.contains(&"temperature"));
    assert!(params.contains(&"tools"));
}
