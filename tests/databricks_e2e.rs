//! Databricks request cleanup through the Anthropic Messages HTTP bridge.

mod common;

use httpmock::{Method::POST, MockServer};
use serde_json::{Value, json};

fn config(upstream: &MockServer) -> oxllm_server::ProxyConfig {
    let mut config = common::proxy_config(&upstream.url("/serving-endpoints"));
    config.model_list[0].model_name = "databricks-chat".into();
    config.model_list[0].oxllm_params.model = "databricks/claude-sonnet-4-5".into();
    config.router_settings.routing.num_retries = Some(0);
    config
}

fn reply() -> Value {
    json!({
        "id": "chatcmpl-databricks", "object": "chat.completion", "model": "claude-sonnet-4-5",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello!"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}
    })
}

#[tokio::test]
async fn messages_thinking_only_history_drops_bare_assistant_turns() {
    let upstream = MockServer::start_async().await;
    let accepted = upstream
        .mock_async(|when, then| {
            when.method(POST)
                .path("/serving-endpoints/chat/completions")
                .header("authorization", "Bearer sk-test-key")
                .json_body(json!({
                    "model": "claude-sonnet-4-5",
                    "messages": [
                        {"role": "system", "content": "Be brief."},
                        {"role": "user", "content": "hi"},
                        {"role": "user", "content": "again"}
                    ],
                    "max_tokens": 100
                }));
            then.status(200).json_body(reply());
        })
        .await;
    let rejected = upstream
        .mock_async(|when, then| {
            when.method(POST)
                .path("/serving-endpoints/chat/completions");
            then.status(400).json_body(json!({"error": {
                "message": "assistant messages must have content or tool calls",
                "type": "invalid_request_error"
            }}));
        })
        .await;
    let (handle, _) = common::start_oxllm(config(&upstream)).await.unwrap();
    let response = common::http_client()
        .post(handle.url("/v1/messages"))
        .json(&json!({
            "model": "databricks-chat", "system": "Be brief.", "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "hmm", "signature": "sig_1"}
                ]},
                {"role": "user", "content": "again"}
            ]
        }))
        .send()
        .await
        .unwrap();
    let status = response.status();
    let body: Value = response.json().await.unwrap();

    assert_eq!(status, 200, "{body}");
    assert_eq!(body["type"], "message");
    assert_eq!(body["content"], json!([{"type": "text", "text": "Hello!"}]));
    assert_eq!(body["stop_reason"], "end_turn");
    accepted.assert_hits_async(1).await;
    rejected.assert_hits_async(0).await;
    handle.shutdown();
}

#[tokio::test]
async fn messages_thinking_only_history_keeps_tool_continuation() {
    let upstream = MockServer::start_async().await;
    let input = json!({"city": "San Francisco", "options": {"days": 2}});
    let schema = json!({"type": "object", "properties": {"city": {"type": "string"}, "options": {"type": "object"}}});
    let accepted = upstream
        .mock_async(|when, then| {
            when.method(POST)
                .path("/serving-endpoints/chat/completions")
                .header("authorization", "Bearer sk-test-key")
                .json_body(json!({
                    "model": "claude-sonnet-4-5", "max_tokens": 100,
                    "tools": [{"type": "function", "function": {"name": "lookup", "parameters": schema}}],
                    "messages": [
                        {"role": "user", "content": "What is the weather?"},
                        {"role": "user", "content": "Please continue"},
                        {"role": "assistant", "content": null, "tool_calls": [{
                            "id": "toolu_lookup", "type": "function",
                            "function": {"name": "lookup", "arguments": serde_json::to_string(&input).unwrap()}
                        }]},
                        {"role": "tool", "tool_call_id": "toolu_lookup", "content": "sunny"},
                        {"role": "user", "content": "Summarize the result"}
                    ]
                }));
            then.status(200).json_body(reply());
        })
        .await;
    let rejected = upstream
        .mock_async(|when, then| {
            when.method(POST)
                .path("/serving-endpoints/chat/completions");
            then.status(400).json_body(json!({"error": {
                "message": "history must not contain bare assistant turns or lose tool calls",
                "type": "invalid_request_error"
            }}));
        })
        .await;
    let (handle, _) = common::start_oxllm(config(&upstream)).await.unwrap();
    let response = common::http_client()
        .post(handle.url("/v1/messages"))
        .json(&json!({
            "model": "databricks-chat", "max_tokens": 100,
            "tools": [{"name": "lookup", "input_schema": schema}],
            "messages": [
                {"role": "user", "content": "What is the weather?"},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "hmm", "signature": "sig_1"}
                ]},
                {"role": "user", "content": "Please continue"},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "call lookup", "signature": "sig_2"},
                    {"type": "tool_use", "id": "toolu_lookup", "name": "lookup", "input": input}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_lookup", "content": "sunny"},
                    {"type": "text", "text": "Summarize the result"}
                ]}
            ]
        }))
        .send()
        .await
        .unwrap();
    let status = response.status();
    let body: Value = response.json().await.unwrap();

    assert_eq!(status, 200, "{body}");
    assert_eq!(body["type"], "message");
    assert_eq!(body["content"], json!([{"type": "text", "text": "Hello!"}]));
    assert_eq!(body["stop_reason"], "end_turn");
    accepted.assert_hits_async(1).await;
    rejected.assert_hits_async(0).await;
    handle.shutdown();
}
