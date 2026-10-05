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
