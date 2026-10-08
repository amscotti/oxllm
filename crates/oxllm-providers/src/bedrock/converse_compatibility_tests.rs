use super::*;

const MODEL: &str = "us.anthropic.claude-haiku-4-5-20251001-v1:0";

#[test]
fn adjacent_role_runs_preserve_content_and_system_boundaries() {
    let request = json!({"messages": [
        {"role": "user", "content": "first"},
        {"role": "system", "content": "system instruction"},
        {"role": "user", "content": [{"type": "text", "text": "second"}]},
        {"role": "assistant", "content": "third"},
        {"role": "assistant", "content": "fourth"},
        {"role": "user", "content": "last"}
    ]});
    let body = transform_request_converse(&request, MODEL, &Map::new()).unwrap();
    assert_eq!(body["system"], json!([{"text": "system instruction"}]));
    assert_eq!(
        body["messages"],
        json!([
            {"role": "user", "content": [{"text": "first"}, {"text": "second"}]},
            {"role": "assistant", "content": [{"text": "third"}, {"text": "fourth"}]},
            {"role": "user", "content": [{"text": "last"}]}
        ])
    );
    assert_eq!(request["messages"].as_array().unwrap().len(), 6);
}

#[test]
fn parallel_tool_results_share_a_turn_and_keep_cache_points_and_followup_text() {
    let request = json!({"messages": [
        {"role": "user", "content": "run both tools"},
        {"role": "assistant", "tool_calls": [
            {"id": "call_a", "type": "function", "function": {"name": "echo", "arguments": "{\"text\":\"a\"}"}},
            {"id": "call_b", "type": "function", "function": {"name": "echo", "arguments": "{\"text\":\"b\"}"}}
        ]},
        {"role": "tool", "tool_call_id": "call_a", "content": "a", "cache_control": {"type": "ephemeral"}},
        {"role": "tool", "tool_call_id": "call_b", "content": "b"},
        {"role": "user", "content": "summarize"}
    ]});
    let body = transform_request_converse(&request, MODEL, &Map::new()).unwrap();
    let messages = body["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[1]["content"].as_array().unwrap().len(), 2);
    assert_eq!(messages[2]["role"], "user");
    assert_eq!(
        messages[2]["content"][0]["toolResult"]["toolUseId"],
        "call_a"
    );
    assert_eq!(
        messages[2]["content"][1],
        json!({"cachePoint": {"type": "default"}})
    );
    assert_eq!(
        messages[2]["content"][2]["toolResult"]["toolUseId"],
        "call_b"
    );
    assert_eq!(messages[2]["content"][3], json!({"text": "summarize"}));
}

#[test]
fn merged_user_turns_keep_guardrail_content_and_image_blocks() {
    let extra = Map::from_iter([
        ("guardrail_id".into(), json!("guardrail")),
        ("guardrail_version".into(), json!("1")),
    ]);
    let request = json!({"messages": [
        {"role": "user", "content": "first"},
        {"role": "user", "content": [
            {"type": "text", "text": "second"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}}
        ]}
    ]});
    let body = transform_request_converse(&request, MODEL, &extra).unwrap();
    assert_eq!(body["messages"].as_array().unwrap().len(), 1);
    let blocks = body["messages"][0]["content"].as_array().unwrap();
    assert_eq!(blocks[0]["guardContent"]["text"]["text"], "first");
    assert_eq!(blocks[1]["guardContent"]["text"]["text"], "second");
    assert_eq!(blocks[2]["image"]["source"]["bytes"], "aGVsbG8=");
}

#[test]
fn modify_params_repairs_undeclared_tools_and_unsigned_thinking_history() {
    let request = json!({"messages": [
        {"role": "user", "content": "echo"},
        {"role": "assistant", "tool_calls": [{"id": "call_a", "type": "function", "function": {"name": "echo", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_a", "content": "done"}
    ], "thinking": {"type": "enabled", "budget_tokens": 1024}, "max_tokens": 2048});
    let original = transform_request_converse(&request, MODEL, &Map::new()).unwrap();
    assert!(original.get("toolConfig").is_none());
    assert_eq!(
        original["additionalModelRequestFields"]["thinking"]["type"],
        "enabled"
    );
    let extra = Map::from_iter([("modify_params".into(), json!(true))]);
    let fixed = transform_request_converse(&request, MODEL, &extra).unwrap();
    assert_eq!(fixed["toolConfig"]["tools"][0]["toolSpec"]["name"], "echo");
    assert_eq!(
        fixed["additionalModelRequestFields"]["thinking"],
        json!({"type": "disabled"})
    );
    assert_eq!(fixed["messages"], original["messages"]);
}

#[test]
fn modify_params_separates_tool_results_from_a_followup_conversation_turn() {
    let request = json!({"messages": [
        {"role": "user", "content": "echo"},
        {"role": "assistant", "tool_calls": [{"id": "call_a", "type": "function", "function": {"name": "echo", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_a", "content": "done"},
        {"role": "user", "content": "summarize"}
    ]});
    let fixed = transform_request_converse(
        &request,
        MODEL,
        &Map::from_iter([("modify_params".into(), json!(true))]),
    )
    .unwrap();
    assert_eq!(fixed["messages"].as_array().unwrap().len(), 5);
    assert!(
        fixed["messages"][2]["content"][0]
            .get("toolResult")
            .is_some()
    );
    assert_eq!(
        fixed["messages"][3],
        json!({"role": "assistant", "content": [{"text": "Continue."}]})
    );
    assert_eq!(
        fixed["messages"][4],
        json!({"role": "user", "content": [{"text": "summarize"}]})
    );
}

#[test]
fn merged_assistant_blocks_keep_original_reasoning_signatures() {
    let request = json!({"messages": [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "first"},
        {"role": "assistant", "content": "second", "thinking_blocks": [{"type": "thinking", "thinking": "thought", "signature": "original-signature"}]},
        {"role": "user", "content": "continue"}
    ]});
    let body = transform_request_converse(&request, MODEL, &Map::new()).unwrap();
    assert_eq!(body["messages"].as_array().unwrap().len(), 3);
    assert_eq!(
        body["messages"][1]["content"],
        json!([
            {"reasoningContent": {"reasoningText": {"text": "thought", "signature": "original-signature"}}},
            {"text": "first"}, {"text": "second"}
        ])
    );
}
