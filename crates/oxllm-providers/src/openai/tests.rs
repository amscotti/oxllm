use super::*;

#[test]
fn versioned_chat_models_keep_sampling_and_sol_one_accepts_explicit_none() {
    for model in ["gpt-5.6-chat", "gpt-6.1-sol-chat-2026-09-24"] {
        let mut body = json!({"temperature":0.3,"top_p":0.5});
        apply_openai_reasoning_limits(&mut body, model);
        assert_eq!(body["temperature"], 0.3);
        assert_eq!(body["top_p"], 0.5);
    }
    let mut body = json!({"reasoning_effort":"none","temperature":0.3,"top_p":0.5});
    apply_openai_reasoning_limits(&mut body, "gpt-6.1-sol");
    assert_eq!(body["temperature"], 0.3);
    assert_eq!(body["top_p"], 0.5);
}
use oxllm_core::Endpoint;
use oxllm_core::deployment::{Deployment, OxllmParams};
use oxllm_core::provider::{ChatProvider, EmbeddingProvider, Provider, ResolvedCredentials};
use serde_json::json;
fn test_deployment() -> Deployment {
    Deployment {
        model_name: "gpt-4o".to_string(),
        oxllm_params: OxllmParams {
            model: "openai/gpt-4o".to_string(),
            api_base: Some("https://api.openai.com/v1".to_string()),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn test_deployment_no_base() -> Deployment {
    let mut dep = test_deployment();
    dep.oxllm_params.api_base = None;
    dep
}

fn normalize(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut sorted = serde_json::Map::new();
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            for key in keys {
                let v = &map[key];
                if v.is_null() {
                    continue;
                }
                sorted.insert(key.clone(), normalize(v));
            }
            Value::Object(sorted)
        }
        Value::Array(items) => Value::Array(items.iter().map(normalize).collect()),
        other => other.clone(),
    }
}

fn assert_semantically_equal(a: &Value, b: &Value) {
    assert_eq!(normalize(a), normalize(b));
}

#[test]
fn test_identity_request() {
    let provider = OpenAiProvider;
    let deployment = test_deployment();
    let request = json!({
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": false
    });
    let result = ChatProvider::transform_request(&provider, &request, &deployment)
        .expect("identity transform should succeed");
    assert_eq!(result, request);
}

#[test]
fn wildcard_deployment_preserves_requested_model() {
    let provider = OpenAiProvider;
    let mut deployment = test_deployment();
    deployment.oxllm_params.model = "openai/*".to_string();
    let request = json!({
        "model": "openai/gpt-5.6",
        "messages": [{"role": "user", "content": "Hello"}]
    });
    let result = ChatProvider::transform_request(&provider, &request, &deployment)
        .expect("wildcard transform should succeed");
    assert_eq!(result["model"], "gpt-5.6");
}

#[test]
fn wildcard_deployment_with_unmatched_prefix_preserves_requested_model() {
    // The documented `provider/*` pattern must work even when the configured
    // model's prefix is not this provider's own strip prefix. Configure
    // `groq/*` on an OpenAI deployment: the `groq/` is not stripped, so the
    // `ends_with("/*")` branch is what catches it as a wildcard.
    let provider = OpenAiProvider;
    let mut deployment = test_deployment();
    deployment.oxllm_params.model = "groq/*".to_string();
    let request = json!({
        "model": "groq/llama-3.1-70b",
        "messages": [{"role": "user", "content": "Hello"}]
    });
    let result = ChatProvider::transform_request(&provider, &request, &deployment)
        .expect("wildcard transform should succeed");
    // strip_openai_prefix does not strip `groq/`, so the client's full id
    // (minus nothing) is forwarded.
    assert_eq!(result["model"], "groq/llama-3.1-70b");
}

#[test]
fn test_identity_response() {
    let provider = OpenAiProvider;
    let response = json!({
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "model": "gpt-4o",
    "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "Hi there!"},
    "finish_reason": "stop"
    }],
    "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21}
    });
    let result = ChatProvider::transform_response(&provider, &response, "gpt-4o")
        .expect("identity response should succeed");
    assert_eq!(result, response);
}

#[test]
fn test_stream_chunk_parsing() {
    let provider = OpenAiProvider;
    let chunk = "data: {\"id\":\"chatcmpl-1\",\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}\n\n";
    let result = provider
        .transform_stream_chunk(chunk)
        .expect("chunk parsing should succeed");
    let value = result.expect("chunk should produce a value");
    assert_eq!(value["id"], "chatcmpl-1");
    assert_eq!(value["choices"][0]["delta"]["content"], "Hi");
}

#[test]
fn test_stream_chunk_done() {
    let provider = OpenAiProvider;
    let result = provider
        .transform_stream_chunk("data: [DONE]\n\n")
        .expect("DONE chunk should not error");
    assert!(result.is_none(), "data: [DONE] must return None");
}

#[test]
fn test_stream_chunk_empty() {
    let provider = OpenAiProvider;
    assert!(
        provider
            .transform_stream_chunk("")
            .expect("empty chunk should not error")
            .is_none()
    );
    assert!(
        provider
            .transform_stream_chunk(" \n\n \t ")
            .expect("whitespace chunk should not error")
            .is_none()
    );
}

#[test]
fn test_endpoint_url_with_explicit_base() {
    let provider = OpenAiProvider;
    let deployment = test_deployment();
    assert_eq!(
        provider.endpoint_url(&deployment),
        "https://api.openai.com/v1/chat/completions"
    );
}

#[test]
fn test_endpoint_url_default_base_when_none() {
    let provider = OpenAiProvider;
    let deployment = test_deployment_no_base();
    assert_eq!(
        provider.endpoint_url(&deployment),
        "https://api.openai.com/v1/chat/completions"
    );
}

#[test]
fn test_endpoint_url_trims_trailing_slash() {
    let provider = OpenAiProvider;
    let mut deployment = test_deployment();
    deployment.oxllm_params.api_base = Some("https://api.openai.com/v1/".to_string());
    assert_eq!(
        provider.endpoint_url(&deployment),
        "https://api.openai.com/v1/chat/completions"
    );
}

#[test]
fn test_apply_auth() {
    let provider = OpenAiProvider;
    let deployment = test_deployment();
    let creds = ResolvedCredentials::from_key("sk-test-key");
    let mut headers = std::collections::HashMap::new();
    ChatProvider::apply_auth(&provider, &deployment, &creds, &mut headers);
    assert_eq!(
        headers.get("authorization").map(String::as_str),
        Some("Bearer sk-test-key")
    );
}

#[test]
fn test_bad_json_chunk() {
    let provider = OpenAiProvider;
    let result = provider.transform_stream_chunk("data: {not valid json}");
    assert!(
        result.is_err(),
        "invalid JSON in stream chunk must return an error"
    );
    match result {
        Err(oxllm_core::Error::Transform(msg)) => {
            assert!(
                msg.contains("invalid stream chunk"),
                "unexpected message: {msg}"
            );
        }
        other => panic!("expected Transform error, got {other:?}"),
    }
}

#[test]
fn test_provider_id_and_supports() {
    let provider = OpenAiProvider;
    assert_eq!(provider.id(), "openai");
    assert!(provider.supports(Endpoint::ChatCompletions));
    assert!(provider.supports(Endpoint::Embeddings));
    assert!(provider.embeddings().is_some());
}

#[test]
fn test_provider_chat_accessor() {
    let provider = OpenAiProvider;
    let deployment = test_deployment();
    let chat = provider
        .chat()
        .expect("OpenAI provider should expose a chat transform");
    let request = json!({"model": "gpt-4o", "messages": []});
    let out = chat
        .transform_request(&request, &deployment)
        .expect("request transform should succeed");
    assert_eq!(out, request);
}

// --- Ported twin: model override ---
//
// When the deployment's model differs from the request's model, the provider's
// `transform_request` MUST override the body's `model` field with the
// deployment's configured upstream model (with the `openai/` prefix stripped).
// Ported from the source `twin_openai_provider_injects_model` test.

/// The API rejects `metadata` unless `store` is explicitly true, so the
/// native OpenAI transform drops the whole object otherwise — including the
/// gateway's `metadata.tags` list. Kept when `store: true` (the caller
/// opted into storage, so the object is valid) or when absent.
#[test]
fn transform_request_drops_metadata_unless_store_enabled() {
    let p = OpenAiProvider;
    let mut dep = test_deployment();
    dep.oxllm_params.model = "openai/gpt-4o-mini".to_string();
    let req = json!({
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "Hi"}],
        "metadata": {"tags": ["app-a"], "user_id": "u-1"},
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert!(
        out.get("metadata").is_none(),
        "metadata must be dropped without store: {out}"
    );
    let stored = json!({
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "Hi"}],
        "store": true,
        "metadata": {"tags": ["app-a"], "user_id": "u-1"},
    });
    let out2 = ChatProvider::transform_request(&p, &stored, &dep).expect("ok");
    assert!(
        out2.get("metadata").is_some(),
        "metadata survives with store=true: {out2}"
    );
}

#[test]
fn transform_request_overrides_request_model_with_deployment_model() {
    let p = OpenAiProvider;
    let mut dep = test_deployment();
    dep.oxllm_params.model = "openai/gpt-4o-mini".to_string();
    let req = json!({
        "model": "caller-supplied-wrong-model",
        "messages": [{"role": "user", "content": "Hi"}],
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(out["model"], "gpt-4o-mini");
    assert_eq!(out["messages"][0]["content"], "Hi");
}

// --- Ported twin: stream flag is preserved ---
//
// Ported from the source's streaming tests — the provider must forward
// the `stream` flag in the request body unchanged.

#[test]
fn transform_request_preserves_stream_flag() {
    let p = OpenAiProvider;
    let req = json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": true,
    });
    let out = ChatProvider::transform_request(&p, &req, &test_deployment()).expect("ok");
    assert_eq!(out["stream"], true);
}

// --- Ported twin: provider accepts model with no provider prefix ---
//
// When the deployment model doesn't start with `openai/`, the provider
// should still forward it unchanged (no accidental stripping).

#[test]
fn transform_request_handles_bare_model() {
    let p = OpenAiProvider;
    let mut dep = test_deployment();
    dep.oxllm_params.model = "gpt-4o".to_string();
    let req = json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hi"}],
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(out["model"], "gpt-4o");
}

const CANONICAL_GOLDEN_REQUEST: &str =
    include_str!("../openai_compat/goldens/chat_request_001.json");
const GOLDEN_REQUEST_002: &str = include_str!("goldens/chat_request_002.json");
const CANONICAL_GOLDEN_RESPONSE: &str =
    include_str!("../openai_compat/goldens/chat_response_001.json");

#[test]
fn test_golden_file_parity() {
    let provider = OpenAiProvider;
    let deployment = test_deployment();
    let model = "gpt-4o";

    let mut request: Value = serde_json::from_str(CANONICAL_GOLDEN_REQUEST)
        .expect("canonical golden request should be valid JSON");
    request["model"] = json!(model);
    let out = ChatProvider::transform_request(&provider, &request, &deployment)
        .expect("golden request transform should succeed");
    assert_semantically_equal(&out, &request);

    // OpenAI-specific tools golden (no canonical equivalent).
    let tools_request: Value =
        serde_json::from_str(GOLDEN_REQUEST_002).expect("tools golden should be valid JSON");
    let out = ChatProvider::transform_request(&provider, &tools_request, &deployment)
        .expect("tools golden request transform should succeed");
    assert_semantically_equal(&out, &tools_request);

    let mut response: Value = serde_json::from_str(CANONICAL_GOLDEN_RESPONSE)
        .expect("canonical golden response should be valid JSON");
    response["model"] = json!(model);
    let out = ChatProvider::transform_response(&provider, &response, model)
        .expect("golden response transform should succeed");
    assert_semantically_equal(&out, &response);
}

// --- Ported from the reference test_openai.py ---
//
// The OpenAI provider is a near-identity transform (only the `model` field is
// rewritten).  These tests verify that all API parameters, complex message
// content (vision, files), tool definitions, and streaming-chunk fields
// survive the pass-through unchanged.

// Reference test: test_openai_prediction_param / test_openai_prediction_param_mock
#[test]
fn passthrough_preserves_prediction() {
    let p = OpenAiProvider;
    let code = "public class User { public string FirstName { get; set; } }";
    let req = json!({
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": "Replace the Username property with an Email property."},
            {"role": "user", "content": code}
        ],
        "prediction": {"type": "content", "content": code}
    });
    let out = ChatProvider::transform_request(&p, &req, &test_deployment()).expect("ok");
    assert_eq!(out["prediction"]["type"], "content");
    assert_eq!(out["prediction"]["content"], code);
}

// Reference test: test_vision_with_custom_model
#[test]
fn passthrough_preserves_vision_content() {
    let p = OpenAiProvider;
    let req = json!({
        "model": "gpt-4o",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAA"}}
            ]
        }],
        "max_tokens": 10
    });
    let out = ChatProvider::transform_request(&p, &req, &test_deployment()).expect("ok");
    assert_eq!(out["messages"][0]["content"][0]["type"], "text");
    assert_eq!(out["messages"][0]["content"][1]["type"], "image_url");
    let url = out["messages"][0]["content"][1]["image_url"]["url"]
        .as_str()
        .unwrap();
    assert!(url.starts_with("data:image/png;base64,"));
}

// Reference test: test_openai_tool_calling
#[test]
fn passthrough_preserves_tool_calls() {
    let p = OpenAiProvider;
    let req = json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "What is TSLA stock price?"}],
        "temperature": 0.5,
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_stock_price",
                "description": "Get current stock price",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string", "description": "The stock ticker symbol"}
                    },
                    "required": ["ticker"]
                }
            }
        }]
    });
    let out = ChatProvider::transform_request(&p, &req, &test_deployment()).expect("ok");
    assert_eq!(out["tools"][0]["type"], "function");
    assert_eq!(out["tools"][0]["function"]["name"], "get_stock_price");
    assert_eq!(out["temperature"], 0.5);
}

// Reference test: test_openai_service_tier_parameter
#[test]
fn passthrough_preserves_service_tier() {
    let p = OpenAiProvider;
    let req = json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hello"}],
        "service_tier": "priority"
    });
    let out = ChatProvider::transform_request(&p, &req, &test_deployment()).expect("ok");
    assert_eq!(out["service_tier"], "priority");
}

// Reference test: test_openai_safety_identifier_parameter
#[test]
fn passthrough_preserves_safety_identifier() {
    let p = OpenAiProvider;
    let req = json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hello"}],
        "safety_identifier": "user_code_123456"
    });
    let out = ChatProvider::transform_request(&p, &req, &test_deployment()).expect("ok");
    assert_eq!(out["safety_identifier"], "user_code_123456");
}

// Reference test: test_openai_gpt5_reasoning
#[test]
fn passthrough_preserves_reasoning_effort() {
    let p = OpenAiProvider;
    let req = json!({
        "model": "gpt-5-mini",
        "messages": [{"role": "user", "content": "Capital of France?"}],
        "reasoning_effort": "minimal"
    });
    let out = ChatProvider::transform_request(&p, &req, &test_deployment()).expect("ok");
    assert_eq!(out["reasoning_effort"], "minimal");
}

// Reference test: test_openai_pdf_url (file type content in messages)
#[test]
fn passthrough_preserves_file_content() {
    let p = OpenAiProvider;
    let req = json!({
        "model": "gpt-4o",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this PDF?"},
                {"type": "file", "file": {"file_id": "https://arxiv.org/pdf/2303.08774"}}
            ]
        }]
    });
    let out = ChatProvider::transform_request(&p, &req, &test_deployment()).expect("ok");
    assert_eq!(out["messages"][0]["content"][1]["type"], "file");
    assert_eq!(
        out["messages"][0]["content"][1]["file"]["file_id"],
        "https://arxiv.org/pdf/2303.08774"
    );
}

// Reference test: test_openai_chat_completion_streaming_handler_reasoning_content
#[test]
fn stream_chunk_with_reasoning_content() {
    let p = OpenAiProvider;
    let chunk = concat!(
        "data: {\"id\":\"e89b6501\",\"object\":\"chat.completion.chunk\",",
        "\"created\":1741037890,\"model\":\"deepseek-reasoner\",",
        "\"choices\":[{\"index\":0,\"delta\":{\"content\":null,\"reasoning_content\":\".\"},",
        "\"finish_reason\":null}]}\n\n"
    );
    let out = p
        .transform_stream_chunk(chunk)
        .expect("ok")
        .expect("should produce value");
    assert_eq!(out["choices"][0]["delta"]["reasoning_content"], ".");
    assert!(out["choices"][0]["delta"]["content"].is_null());
}

// Reference test: test_streaming_tool_calls_with_n_greater_than_1 (chunk parsing)
#[test]
fn stream_chunk_with_tool_calls() {
    let p = OpenAiProvider;
    let chunk = concat!(
        "data: {\"id\":\"x\",\"choices\":[{\"index\":0,",
        "\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",",
        "\"function\":{\"name\":\"get_weather\",\"arguments\":\"{\\\"location\\\":\\\"SF\\\"}\"}}]}}]}\n\n"
    );
    let out = p
        .transform_stream_chunk(chunk)
        .expect("ok")
        .expect("should produce value");
    assert_eq!(
        out["choices"][0]["delta"]["tool_calls"][0]["function"]["name"],
        "get_weather"
    );
    assert_eq!(out["choices"][0]["delta"]["tool_calls"][0]["index"], 0);
    assert_eq!(
        out["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"],
        "{\"location\":\"SF\"}"
    );
}

// Reference test: test_streaming_content_with_n_greater_than_1 (choice index in chunks)
#[test]
fn stream_chunk_with_choice_index() {
    let p = OpenAiProvider;
    let chunk =
        "data: {\"id\":\"x\",\"choices\":[{\"index\":2,\"delta\":{\"content\":\"Hello\"}}]}\n\n";
    let out = p
        .transform_stream_chunk(chunk)
        .expect("ok")
        .expect("should produce value");
    assert_eq!(out["choices"][0]["index"], 2);
    assert_eq!(out["choices"][0]["delta"]["content"], "Hello");
}

// Reference test: test_openai_web_search (response annotations passthrough)
#[test]
fn response_preserves_annotations() {
    let p = OpenAiProvider;
    let resp = json!({
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "model": "gpt-4o-search-preview",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Here is a news story.",
                "annotations": [{
                    "type": "url_citation",
                    "url_citation": {
                        "end_index": 50,
                        "start_index": 10,
                        "url": "https://example.com/news"
                    }
                }]
            },
            "finish_reason": "stop"
        }]
    });
    let out = ChatProvider::transform_response(&p, &resp, "gpt-4o-search-preview").expect("ok");
    assert_eq!(
        out["choices"][0]["message"]["annotations"][0]["type"],
        "url_citation"
    );
    assert_eq!(
        out["choices"][0]["message"]["annotations"][0]["url_citation"]["url"],
        "https://example.com/news"
    );
}

// Reference test: test_openai_prediction_param (completion_tokens_details passthrough)
#[test]
fn response_preserves_usage_details() {
    let p = OpenAiProvider;
    let resp = json!({
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "model": "gpt-4o-mini",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "code"},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 50,
            "completion_tokens": 30,
            "total_tokens": 80,
            "completion_tokens_details": {
                "accepted_prediction_tokens": 10,
                "rejected_prediction_tokens": 2
            }
        }
    });
    let out = ChatProvider::transform_response(&p, &resp, "gpt-4o-mini").expect("ok");
    assert_eq!(
        out["usage"]["completion_tokens_details"]["accepted_prediction_tokens"],
        10
    );
    assert_eq!(
        out["usage"]["completion_tokens_details"]["rejected_prediction_tokens"],
        2
    );
}

// =================================================================
// Reasoning-model temperature stripping (#34301)
// =================================================================
//
// OpenAI reasoning models (o1, o3, o4, gpt-5-*) reject the `temperature`
// parameter unless it is exactly 1.0. The provider must drop it before
// forwarding the request, otherwise the upstream returns a 400.

fn deployment_with(model: &str) -> Deployment {
    Deployment {
        model_name: model.to_string(),
        oxllm_params: OxllmParams {
            model: format!("openai/{model}"),
            api_base: Some("https://api.openai.com/v1".to_string()),
            ..Default::default()
        },
        ..Default::default()
    }
}

#[test]
fn temperature_stripped_for_o1_model() {
    let p = OpenAiProvider;
    let dep = deployment_with("o1-mini");
    let req = json!({
        "model": "o1-mini",
        "messages": [{"role": "user", "content": "Hi"}],
        "temperature": 0.7,
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert!(
        out.get("temperature").is_none(),
        "temperature must be dropped for o1 models"
    );
}

#[test]
fn temperature_preserved_for_gpt4o() {
    let p = OpenAiProvider;
    let dep = deployment_with("gpt-4o");
    let req = json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hi"}],
        "temperature": 0.7,
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(out["temperature"], 0.7);
}

#[test]
fn temperature_1_0_preserved_for_o1() {
    let p = OpenAiProvider;
    let dep = deployment_with("o1");
    let req = json!({
        "model": "o1",
        "messages": [{"role": "user", "content": "Hi"}],
        "temperature": 1.0,
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(
        out["temperature"], 1.0,
        "temperature of exactly 1.0 must be preserved for reasoning models"
    );
}

// =================================================================
// H1: max_tokens → max_completion_tokens mapping for reasoning models
// =================================================================
//
// OpenAI o-series and gpt-5 only accept `max_completion_tokens`. The provider
// must rewrite a user-supplied `max_tokens` so callers can use the canonical
// field name without tripping an HTTP 400.

#[test]
fn max_tokens_mapped_to_max_completion_tokens_for_o3() {
    let p = OpenAiProvider;
    let dep = deployment_with("o3");
    let req = json!({
        "model": "o3",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 256,
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert!(
        out.get("max_tokens").is_none(),
        "max_tokens must be removed for o3"
    );
    assert_eq!(out["max_completion_tokens"], 256);
}

#[test]
fn max_tokens_mapped_to_max_completion_tokens_for_gpt5() {
    let p = OpenAiProvider;
    let dep = deployment_with("gpt-5-mini");
    let req = json!({
        "model": "gpt-5-mini",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 512,
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert!(
        out.get("max_tokens").is_none(),
        "max_tokens must be removed for gpt-5"
    );
    assert_eq!(out["max_completion_tokens"], 512);
}

#[test]
fn max_tokens_not_mapped_for_gpt4o() {
    let p = OpenAiProvider;
    let dep = deployment_with("gpt-4o");
    let req = json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 100,
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(out["max_tokens"], 100);
    assert!(
        out.get("max_completion_tokens").is_none(),
        "must not synthesize max_completion_tokens for non-reasoning models"
    );
}

// H5: null token values in usage are normalized to 0.
// =================================================================
//
// Some OpenAI-compatible upstreams return `"prompt_tokens": null` (or
// `completion_tokens`/`total_tokens`/nested details) on degraded responses.
// Downstream `extract_usage` already substitutes 0 for missing fields, but
// other consumers (proxies, billing hooks, logging) read the raw body —
// `transform_response` rewrites nulls to explicit zeros so the wire shape
// stays consistent.

fn usage_response_with_null(field: &str) -> Value {
    let mut usage = serde_json::Map::new();
    usage.insert("prompt_tokens".to_string(), json!(10));
    usage.insert("completion_tokens".to_string(), json!(5));
    usage.insert("total_tokens".to_string(), json!(15));
    usage.insert(field.to_string(), Value::Null);
    json!({
        "id": "chatcmpl-null",
        "object": "chat.completion",
        "model": "gpt-4o",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop"
        }],
        "usage": Value::Object(usage),
    })
}

#[test]
fn transform_response_normalizes_null_prompt_tokens_to_zero() {
    let p = OpenAiProvider;
    let resp = usage_response_with_null("prompt_tokens");
    let out = ChatProvider::transform_response(&p, &resp, "gpt-4o").expect("ok");
    assert_eq!(out["usage"]["prompt_tokens"], 0, "null prompt_tokens -> 0");
    assert_eq!(
        out["usage"]["completion_tokens"], 5,
        "non-null fields preserved"
    );
    assert_eq!(out["usage"]["total_tokens"], 15);
}

#[test]
fn transform_response_normalizes_null_completion_tokens_to_zero() {
    let p = OpenAiProvider;
    let resp = usage_response_with_null("completion_tokens");
    let out = ChatProvider::transform_response(&p, &resp, "gpt-4o").expect("ok");
    assert_eq!(
        out["usage"]["completion_tokens"], 0,
        "null completion_tokens -> 0"
    );
    assert_eq!(out["usage"]["prompt_tokens"], 10);
}

#[test]
fn transform_response_normalizes_null_total_tokens_to_zero() {
    let p = OpenAiProvider;
    let resp = usage_response_with_null("total_tokens");
    let out = ChatProvider::transform_response(&p, &resp, "gpt-4o").expect("ok");
    assert_eq!(out["usage"]["total_tokens"], 0, "null total_tokens -> 0");
    assert_eq!(out["usage"]["prompt_tokens"], 10);
}

#[test]
fn transform_response_normalizes_null_reasoning_tokens_in_details() {
    let p = OpenAiProvider;
    let mut resp = usage_response_with_null("ignored");
    resp["usage"]["completion_tokens_details"] = json!({
        "reasoning_tokens": null,
        "accepted_prediction_tokens": 3
    });
    let out = ChatProvider::transform_response(&p, &resp, "gpt-4o").expect("ok");
    assert_eq!(
        out["usage"]["completion_tokens_details"]["reasoning_tokens"], 0,
        "null reasoning_tokens in details -> 0"
    );
    // Non-null sibling fields are preserved.
    assert_eq!(
        out["usage"]["completion_tokens_details"]["accepted_prediction_tokens"],
        3
    );
}

#[test]
fn max_completion_tokens_already_present_is_not_duplicated() {
    let p = OpenAiProvider;
    let dep = deployment_with("o3");
    let req = json!({
        "model": "o3",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 100,
        "max_completion_tokens": 200,
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    // Explicit max_completion_tokens wins; max_tokens is dropped silently
    // rather than overwriting the explicit value.
    assert!(
        out.get("max_tokens").is_none(),
        "max_tokens must still be stripped when max_completion_tokens is present"
    );
    assert_eq!(out["max_completion_tokens"], 200);
}

// =================================================================
// H2: strip unsupported params for reasoning models
// =================================================================
//
// o-series models (o1/o3/o4) reject `top_p`, `frequency_penalty`,
// `presence_penalty`, `logprobs`, `top_logprobs` with HTTP 400. gpt-5 / gpt-6
// models that do not accept `reasoning_effort: "none"` reject sampling params
// even when the caller sends `"none"`. Models that accept `"none"` re-enable
// `top_p` and logprobs only while effort resolves to `"none"`.

#[test]
fn top_p_stripped_for_o3() {
    let p = OpenAiProvider;
    let dep = deployment_with("o3");
    let req = json!({
        "model": "o3",
        "messages": [{"role": "user", "content": "Hi"}],
        "top_p": 0.9,
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert!(out.get("top_p").is_none(), "top_p must be stripped for o3");
}

#[test]
fn frequency_penalty_stripped_for_o3() {
    let p = OpenAiProvider;
    let dep = deployment_with("o3");
    let req = json!({
        "model": "o3",
        "messages": [{"role": "user", "content": "Hi"}],
        "frequency_penalty": 0.5,
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert!(
        out.get("frequency_penalty").is_none(),
        "frequency_penalty must be stripped for o3"
    );
}

#[test]
fn logprobs_stripped_for_o3() {
    let p = OpenAiProvider;
    let dep = deployment_with("o3");
    let req = json!({
        "model": "o3",
        "messages": [{"role": "user", "content": "Hi"}],
        "logprobs": true,
        "top_logprobs": 5,
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert!(
        out.get("logprobs").is_none(),
        "logprobs must be stripped for o3"
    );
    assert!(
        out.get("top_logprobs").is_none(),
        "top_logprobs must be stripped for o3"
    );
}

#[test]
fn top_p_not_stripped_for_gpt4o() {
    let p = OpenAiProvider;
    let dep = deployment_with("gpt-4o");
    let req = json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hi"}],
        "top_p": 0.8,
        "frequency_penalty": 0.1,
        "presence_penalty": 0.2,
        "logprobs": true,
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(out["top_p"], 0.8);
    assert_eq!(out["frequency_penalty"], 0.1);
    assert_eq!(out["presence_penalty"], 0.2);
    assert_eq!(out["logprobs"], true);
}

#[test]
fn top_p_stripped_for_gpt5_mini_even_with_reasoning_effort_none() {
    // gpt-5-mini does not accept reasoning_effort "none", so top_p stays illegal.
    let p = OpenAiProvider;
    let dep = deployment_with("gpt-5-mini");
    let req = json!({
        "model": "gpt-5-mini",
        "messages": [{"role": "user", "content": "Hi"}],
        "reasoning_effort": "none",
        "top_p": 0.8,
        "logprobs": true,
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert!(out.get("top_p").is_none());
    assert!(out.get("logprobs").is_none());
}

#[test]
fn top_p_stripped_for_gpt5_with_reasoning_effort_low() {
    // Sanity check: the moment reasoning is on (any non-"none" value), the
    // unsupported sampling surface must be stripped again.
    let p = OpenAiProvider;
    let dep = deployment_with("gpt-5-mini");
    let req = json!({
        "model": "gpt-5-mini",
        "messages": [{"role": "user", "content": "Hi"}],
        "reasoning_effort": "low",
        "top_p": 0.8,
        "logprobs": true,
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert!(
        out.get("top_p").is_none(),
        "top_p must be stripped when reasoning is active"
    );
    assert!(
        out.get("logprobs").is_none(),
        "logprobs must be stripped when reasoning is active"
    );
}

// =================================================================
// H7: system → developer role conversion
// =================================================================
//
// Per OpenAI docs (April 2025+): "When you use a system message with o4-mini,
// o3, o3-mini, and o1 it will be treated as a developer message." Convert the
// role so the upstream sees the modern canonical form.

#[test]
fn system_role_converted_to_developer_for_o3() {
    let p = OpenAiProvider;
    let dep = deployment_with("o3");
    let req = json!({
        "model": "o3",
        "messages": [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hi"},
        ],
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(out["messages"][0]["role"], "developer");
    assert_eq!(out["messages"][0]["content"], "Be concise.");
    assert_eq!(out["messages"][1]["role"], "user");
}

#[test]
fn system_role_preserved_for_gpt5() {
    let p = OpenAiProvider;
    let dep = deployment_with("gpt-5");
    let req = json!({
        "model": "gpt-5",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hi"},
        ],
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(
        out["messages"][0]["role"], "system",
        "gpt-5 supports system messages natively; the developer rewrite is o-series only"
    );
}

#[test]
fn system_role_not_changed_for_gpt4o() {
    let p = OpenAiProvider;
    let dep = deployment_with("gpt-4o");
    let req = json!({
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hi"},
        ],
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(out["messages"][0]["role"], "system");
}

#[test]
fn no_system_message_does_not_crash_for_o3() {
    let p = OpenAiProvider;
    let dep = deployment_with("o3");
    let req = json!({
        "model": "o3",
        "messages": [
            {"role": "user", "content": "Hi"},
        ],
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(out["messages"][0]["role"], "user");
    assert_eq!(out["messages"].as_array().unwrap().len(), 1);
}

// =================================================================
// H9: organization / project header support
// =================================================================
//
// Optional OpenAI multi-organization / project routing, read from
// `oxllm_params.extra` (which captures the `organization` and `project`
// fields via `#[serde(flatten)]`).

fn deployment_with_extra(model: &str, key: &str, value: &str) -> Deployment {
    let mut dep = deployment_with(model);
    dep.oxllm_params.extra.insert(key.to_string(), json!(value));
    dep
}

#[test]
fn organization_header_set_when_configured() {
    let p = OpenAiProvider;
    let dep = deployment_with_extra("gpt-4o", "organization", "org-abc123");
    let creds = ResolvedCredentials::from_key("sk-test");
    let mut headers = std::collections::HashMap::new();
    ChatProvider::apply_auth(&p, &dep, &creds, &mut headers);
    assert_eq!(
        headers.get("openai-organization").map(String::as_str),
        Some("org-abc123")
    );
    // Authorization must still be set.
    assert_eq!(
        headers.get("authorization").map(String::as_str),
        Some("Bearer sk-test")
    );
}

#[test]
fn project_header_set_when_configured() {
    let p = OpenAiProvider;
    let dep = deployment_with_extra("gpt-4o", "project", "proj_xyz");
    let creds = ResolvedCredentials::from_key("sk-test");
    let mut headers = std::collections::HashMap::new();
    ChatProvider::apply_auth(&p, &dep, &creds, &mut headers);
    assert_eq!(
        headers.get("openai-project").map(String::as_str),
        Some("proj_xyz")
    );
    assert!(
        !headers.contains_key("openai-organization"),
        "no organization header when only project is configured"
    );
}

#[test]
fn org_and_project_headers_absent_when_not_configured() {
    let p = OpenAiProvider;
    let dep = deployment_with("gpt-4o");
    let creds = ResolvedCredentials::from_key("sk-test");
    let mut headers = std::collections::HashMap::new();
    ChatProvider::apply_auth(&p, &dep, &creds, &mut headers);
    assert!(!headers.contains_key("openai-organization"));
    assert!(!headers.contains_key("openai-project"));
}

#[test]
fn transform_response_leaves_valid_usage_unchanged() {
    let p = OpenAiProvider;
    let resp = json!({
        "id": "chatcmpl-ok",
        "object": "chat.completion",
        "model": "gpt-4o",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 9,
            "completion_tokens": 12,
            "total_tokens": 21,
            "completion_tokens_details": {"reasoning_tokens": 5}
        }
    });
    let out = ChatProvider::transform_response(&p, &resp, "gpt-4o").expect("ok");
    assert_eq!(out["usage"]["prompt_tokens"], 9);
    assert_eq!(out["usage"]["completion_tokens"], 12);
    assert_eq!(out["usage"]["total_tokens"], 21);
    assert_eq!(
        out["usage"]["completion_tokens_details"]["reasoning_tokens"],
        5
    );
}

#[test]
fn transform_response_noop_when_usage_absent() {
    let p = OpenAiProvider;
    let resp = json!({
        "id": "no-usage",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop"
        }]
    });
    let out = ChatProvider::transform_response(&p, &resp, "gpt-4o").expect("ok");
    assert!(
        out.get("usage").is_none(),
        "no usage field should remain absent"
    );
}

// =================================================================
// Prefix/model helpers
// =================================================================

#[test]
fn strip_openai_prefix_with_various_inputs() {
    assert_eq!(strip_openai_prefix("openai/gpt-4o"), "gpt-4o");
    assert_eq!(strip_openai_prefix("openai/o3-mini"), "o3-mini");
    assert_eq!(
        strip_openai_prefix("openai/sub/provider/model"),
        "sub/provider/model"
    );
    assert_eq!(
        strip_openai_prefix("gpt-4o"),
        "gpt-4o",
        "bare model ids without a provider prefix must be unchanged"
    );
}

#[test]
fn strip_openai_prefix_edge_cases() {
    assert_eq!(
        strip_openai_prefix(""),
        "",
        "empty string must be unchanged"
    );
    assert_eq!(
        strip_openai_prefix("openai"),
        "openai",
        "a bare provider name with no slash must be unchanged"
    );
    assert_eq!(
        strip_openai_prefix("/"),
        "/",
        "a bare slash must be unchanged"
    );
    assert_eq!(
        strip_openai_prefix("OPENAI/gpt-4o"),
        "OPENAI/gpt-4o",
        "stripping must be case-sensitive"
    );
}

#[test]
fn wildcard_bare_star_deployment_preserves_requested_model() {
    let p = OpenAiProvider;
    let mut dep = test_deployment();
    dep.oxllm_params.model = "*".to_string();
    let req = json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hello"}]
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(out["model"], "gpt-4o");
}

#[test]
fn wildcard_deployment_without_request_model_errors() {
    let p = OpenAiProvider;
    let mut dep = test_deployment();
    dep.oxllm_params.model = "openai/*".to_string();
    let req = json!({
        "messages": [{"role": "user", "content": "Hello"}]
    });
    let res = ChatProvider::transform_request(&p, &req, &dep);
    assert!(
        matches!(res, Err(oxllm_core::Error::BadRequest(_))),
        "wildcard deployment with no body model must return BadRequest, got {res:?}"
    );
}

#[test]
fn wildcard_bare_star_without_request_model_errors() {
    let p = OpenAiProvider;
    let mut dep = test_deployment();
    dep.oxllm_params.model = "*".to_string();
    let req = json!({
        "messages": [{"role": "user", "content": "Hello"}]
    });
    let res = ChatProvider::transform_request(&p, &req, &dep);
    assert!(
        matches!(res, Err(oxllm_core::Error::BadRequest(_))),
        "bare-star wildcard with no body model must return BadRequest, got {res:?}"
    );
}

#[test]
fn wildcard_deployment_with_non_string_model_errors() {
    let p = OpenAiProvider;
    let mut dep = test_deployment();
    dep.oxllm_params.model = "*".to_string();
    let req = json!({
        "model": 42,
        "messages": [{"role": "user", "content": "Hello"}]
    });
    let res = ChatProvider::transform_request(&p, &req, &dep);
    assert!(
        matches!(res, Err(oxllm_core::Error::BadRequest(_))),
        "wildcard with non-string body model must return BadRequest, got {res:?}"
    );
}

#[test]
fn transform_request_handles_non_object_body() {
    let p = OpenAiProvider;
    let req = json!(["not", "an", "object"]);
    let out = ChatProvider::transform_request(&p, &req, &test_deployment()).expect("ok");
    assert_eq!(
        out, req,
        "non-object request bodies must pass through unchanged"
    );
}

// =================================================================
// Reasoning classifiers
// =================================================================

#[test]
fn is_reasoning_model_classification() {
    for model in [
        "o1",
        "o1-mini",
        "o1-preview",
        "o1-pro",
        "o3",
        "o3-mini",
        "o3-custom-finetune",
        "o4",
        "o4-mini",
        "gpt-5",
        "gpt-5-mini",
        "gpt-6-luna",
        "gpt-6-sol",
        "gpt-6-astra",
        "gpt-6-luna-2026-09-22",
        "foo-reasoning",
        "gpt-4o-reasoning",
    ] {
        assert!(
            is_reasoning_model(model),
            "{model} should be classified as a reasoning model"
        );
    }
    for model in [
        "gpt-4o",
        "gpt-4.1",
        "gpt-4",
        "gpt-3.5-turbo",
        "text-embedding-3-small",
        "o1foo",
        "o3bar",
        "o4baz",
        "o",
        "o100",
        "gpt-5-chat",
        "gpt-5-chat-latest",
        "gpt-5-chat-2025-08-07",
    ] {
        assert!(
            !is_reasoning_model(model),
            "{model} should not be classified as a reasoning model"
        );
    }
}

#[test]
fn is_o_series_classification() {
    for model in [
        "o1",
        "o1-mini",
        "o1-preview",
        "o1-pro",
        "o3",
        "o3-mini",
        "o3-custom",
        "o4",
        "o4-mini",
    ] {
        assert!(
            is_o_series(model),
            "{model} should be classified as an o-series model"
        );
    }
    for model in ["gpt-5", "gpt-4o", "o1foo", "o3bar", "o4baz", "o100"] {
        assert!(
            !is_o_series(model),
            "{model} must not be classified as o-series"
        );
    }
}

#[test]
fn should_strip_reasoning_unsupported_matrix() {
    let none = json!({"reasoning_effort": "none"});
    let low = json!({"reasoning_effort": "low"});
    let absent = json!({});
    for model in ["o1", "o3", "o4"] {
        assert!(
            should_strip_reasoning_unsupported(model, &none),
            "o-series must strip regardless of reasoning_effort"
        );
        assert!(should_strip_reasoning_unsupported(model, &low));
        assert!(should_strip_reasoning_unsupported(model, &absent));
    }
    assert!(
        should_strip_reasoning_unsupported("gpt-5", &none),
        "gpt-5 does not accept reasoning_effort none, so sampling params stay illegal"
    );
    assert!(should_strip_reasoning_unsupported("gpt-5", &low));
    assert!(
        should_strip_reasoning_unsupported("gpt-5", &absent),
        "gpt-5 defaults to reasoning ON (medium) so unsupported params must strip"
    );
    assert!(!should_strip_reasoning_unsupported("gpt-4o", &low));
    assert!(!should_strip_reasoning_unsupported("gpt-4o", &absent));
}

// =================================================================
// Reasoning-model gaps
// =================================================================

#[test]
fn temperature_stripped_for_gpt5_with_reasoning_effort() {
    let p = OpenAiProvider;
    let dep = deployment_with("gpt-5");
    let req = json!({
        "model": "gpt-5",
        "messages": [{"role": "user", "content": "Hi"}],
        "reasoning_effort": "low",
        "temperature": 0.7,
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert!(
        out.get("temperature").is_none(),
        "temperature must be dropped for gpt-5 with active reasoning"
    );
    assert_eq!(out["reasoning_effort"], "low");
}

#[test]
fn temperature_stripped_for_gpt5_even_with_reasoning_effort_none() {
    let p = OpenAiProvider;
    let dep = deployment_with("gpt-5");
    let req = json!({
        "model": "gpt-5",
        "messages": [{"role": "user", "content": "Hi"}],
        "reasoning_effort": "none",
        "temperature": 0.7,
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert!(
        out.get("temperature").is_none(),
        "gpt-5 does not accept reasoning_effort none, so a non-default temperature is dropped"
    );
}

#[test]
fn temperature_integer_1_0_preserved_for_o1() {
    let p = OpenAiProvider;
    let dep = deployment_with("o1");
    let req = json!({
        "model": "o1",
        "messages": [{"role": "user", "content": "Hi"}],
        "temperature": 1,
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(
        out["temperature"].as_i64(),
        Some(1),
        "an integer temperature of exactly 1.0 must be preserved"
    );
}

#[test]
fn max_tokens_mapped_to_max_completion_tokens_for_o4() {
    let p = OpenAiProvider;
    let dep = deployment_with("o4-mini");
    let req = json!({
        "model": "o4-mini",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 100,
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert!(
        out.get("max_tokens").is_none(),
        "max_tokens must be removed for o4"
    );
    assert_eq!(out["max_completion_tokens"], 100);
}

#[test]
fn developer_role_already_present_not_double_converted() {
    let p = OpenAiProvider;
    let dep = deployment_with("o3");
    let req = json!({
        "model": "o3",
        "messages": [
            {"role": "developer", "content": "Be concise."},
            {"role": "user", "content": "Hi"},
        ],
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(out["messages"][0]["role"], "developer");
    assert_eq!(out["messages"][0]["content"], "Be concise.");
    assert_eq!(out["messages"][1]["role"], "user");
}

#[test]
fn system_message_with_array_content_converted_for_o3() {
    let p = OpenAiProvider;
    let dep = deployment_with("o3");
    let req = json!({
        "model": "o3",
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": "Be concise."}]
            },
            {"role": "user", "content": "Hi"},
        ],
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(out["messages"][0]["role"], "developer");
    assert_eq!(out["messages"][0]["content"][0]["type"], "text");
    assert_eq!(out["messages"][0]["content"][0]["text"], "Be concise.");
}

#[test]
fn non_system_messages_not_converted_for_reasoning_model() {
    let p = OpenAiProvider;
    let dep = deployment_with("o1");
    let req = json!({
        "model": "o1",
        "messages": [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "tool", "tool_call_id": "call_1", "content": "42"},
        ],
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(out["messages"][0]["role"], "developer");
    assert_eq!(out["messages"][1]["role"], "user");
    assert_eq!(
        out["messages"][2]["role"], "assistant",
        "assistant messages must not be converted"
    );
    assert_eq!(
        out["messages"][3]["role"], "tool",
        "tool messages must not be converted"
    );
}

#[test]
fn gpt5_reasoning_strips_all_unsupported_params_at_once() {
    let p = OpenAiProvider;
    let dep = deployment_with("gpt-5-mini");
    let req = json!({
        "model": "gpt-5-mini",
        "messages": [{"role": "user", "content": "Hi"}],
        "reasoning_effort": "low",
        "logprobs": true,
        "top_logprobs": 5,
        "top_p": 0.9,
        "frequency_penalty": 0.5,
        "presence_penalty": 0.5,
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    for key in [
        "logprobs",
        "top_logprobs",
        "top_p",
        "frequency_penalty",
        "presence_penalty",
    ] {
        assert!(
            out.get(key).is_none(),
            "{key} must be stripped for gpt-5 with active reasoning"
        );
    }
    assert_eq!(out["reasoning_effort"], "low");
}

#[test]
fn gpt5_strips_unsupported_params_with_absent_reasoning_effort() {
    let p = OpenAiProvider;
    let dep = deployment_with("gpt-5");
    let req = json!({
        "model": "gpt-5",
        "messages": [{"role": "user", "content": "Hi"}],
        "logprobs": true,
        "top_p": 0.9,
        "frequency_penalty": 0.5,
        "presence_penalty": 0.5,
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    for key in ["logprobs", "top_p", "frequency_penalty", "presence_penalty"] {
        assert!(
            out.get(key).is_none(),
            "{key} must be stripped for gpt-5 with absent reasoning_effort (defaults to medium)"
        );
    }
}

#[test]
fn gpt5_with_reasoning_effort_none_still_strips_sampling_params() {
    let p = OpenAiProvider;
    let dep = deployment_with("gpt-5");
    let req = json!({
        "model": "gpt-5",
        "messages": [{"role": "user", "content": "Hi"}],
        "reasoning_effort": "none",
        "top_p": 0.9,
        "logprobs": true,
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert!(out.get("top_p").is_none());
    assert!(out.get("logprobs").is_none());
}

// =================================================================
// gpt-5-chat family is NOT a reasoning model
// =================================================================
//
// The `gpt-5-chat*` family (gpt-5-chat, gpt-5-chat-latest,
// gpt-5-chat-2025-08-07) are regular chat models: they support
// `temperature`, `top_p`, and the standard sampling surface, but NOT
// `reasoning_effort`. They must NOT be routed through the GPT-5 reasoning
// path (the reference GPT-5-family capability check).

#[test]
fn gpt5_chat_family_not_classified_as_reasoning() {
    assert!(!is_reasoning_model("gpt-5-chat"));
    assert!(!is_reasoning_model("gpt-5-chat-latest"));
    assert!(!is_reasoning_model("gpt-5-chat-2025-08-07"));
    // Sanity: true gpt-5 reasoning models are still classified correctly.
    assert!(is_reasoning_model("gpt-5"));
    assert!(is_reasoning_model("gpt-5-mini"));
}

#[test]
fn gpt5_chat_preserves_sampling_params() {
    let p = OpenAiProvider;
    let dep = deployment_with("gpt-5-chat");
    let req = json!({
        "model": "gpt-5-chat",
        "messages": [{"role": "user", "content": "Hi"}],
        "temperature": 0.7,
        "top_p": 0.9,
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(
        out["temperature"], 0.7,
        "gpt-5-chat supports temperature; it must survive the transform"
    );
    assert_eq!(
        out["top_p"], 0.9,
        "gpt-5-chat supports top_p; it must survive the transform"
    );
    // `max_tokens` is NOT remapped for gpt-5-chat (it's not a reasoning model).
    assert!(
        out.get("max_completion_tokens").is_none(),
        "max_tokens must not be remapped for gpt-5-chat"
    );
}

#[test]
fn gpt5_chat_latest_preserves_max_tokens() {
    let p = OpenAiProvider;
    let dep = deployment_with("gpt-5-chat-latest");
    let req = json!({
        "model": "gpt-5-chat-latest",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 128,
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(
        out["max_tokens"], 128,
        "gpt-5-chat-latest is a regular chat model; max_tokens is forwarded as-is"
    );
}

// =================================================================
// Dict-form reasoning_effort (Responses-API shape)
// =================================================================
//
// A raw dict `reasoning_effort` like {"effort": "none", "summary": "detailed"}
// (valid in the Responses API) must be recognized as effort="none" and
// normalized to a string. gpt-6-luna accepts "none" but does not default to
// it, so this is what keeps temperature.

#[test]
fn gpt6_luna_dict_reasoning_effort_none_preserves_sampling_params() {
    let p = OpenAiProvider;
    let dep = deployment_with("gpt-6-luna");
    let req = json!({
        "model": "gpt-6-luna",
        "messages": [{"role": "user", "content": "Hi"}],
        "reasoning_effort": {"effort": "none", "summary": "detailed"},
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 32,
        "frequency_penalty": 0.2,
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(out["temperature"], 0.7);
    assert_eq!(out["top_p"], 0.9);
    assert_eq!(out["reasoning_effort"], "none");
    assert_eq!(out["max_completion_tokens"], 32);
    assert!(out.get("max_tokens").is_none());
    assert!(out.get("frequency_penalty").is_none());
}

#[test]
fn gpt6_luna_absent_effort_strips_sampling_params() {
    let p = OpenAiProvider;
    let dep = deployment_with("gpt-6-luna");
    let req = json!({
        "model": "gpt-6-luna",
        "messages": [{"role": "user", "content": "Hi"}],
        "temperature": 0.2,
        "top_p": 0.5,
        "max_tokens": 16,
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert!(out.get("temperature").is_none());
    assert!(out.get("top_p").is_none());
    assert_eq!(out["max_completion_tokens"], 16);
}

#[test]
fn gpt6_astra_strips_temperature_even_with_explicit_none() {
    let p = OpenAiProvider;
    let dep = deployment_with("gpt-6-astra");
    let req = json!({
        "model": "gpt-6-astra",
        "messages": [{"role": "user", "content": "Hi"}],
        "reasoning_effort": "none",
        "temperature": 0.2,
        "top_p": 0.5,
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert!(out.get("temperature").is_none());
    assert!(out.get("top_p").is_none());
    assert_eq!(out["reasoning_effort"], "none");
}

#[test]
fn gpt55_absent_effort_strips_temperature_explicit_none_keeps_it() {
    let p = OpenAiProvider;
    let dep = deployment_with("gpt-5.5");
    let absent = json!({
        "model": "gpt-5.5",
        "messages": [{"role": "user", "content": "Hi"}],
        "temperature": 0.2,
    });
    let out = ChatProvider::transform_request(&p, &absent, &dep).expect("ok");
    assert!(out.get("temperature").is_none());
    let explicit = json!({
        "model": "gpt-5.5",
        "messages": [{"role": "user", "content": "Hi"}],
        "reasoning_effort": "none",
        "temperature": 0.2,
        "top_p": 0.4,
    });
    let out = ChatProvider::transform_request(&p, &explicit, &dep).expect("ok");
    assert_eq!(out["temperature"], 0.2);
    assert_eq!(out["top_p"], 0.4);
}

#[test]
fn effort_level_extracts_string_and_dict_forms() {
    assert_eq!(effort_level(&json!({})), None);
    assert_eq!(
        effort_level(&json!({"reasoning_effort": "none"})),
        Some("none")
    );
    assert_eq!(
        effort_level(&json!({"reasoning_effort": {"effort": "low"}})),
        Some("low")
    );
    assert_eq!(
        effort_level(&json!({"reasoning_effort": {"effort": "high", "summary": "detailed"}})),
        Some("high")
    );
    assert_eq!(
        effort_level(&json!({"reasoning_effort": {"summary": "detailed"}})),
        None,
        "dict without 'effort' key yields None"
    );
    assert_eq!(
        effort_level(&json!({"reasoning_effort": 42})),
        None,
        "non-string/non-object value yields None"
    );
}

// =================================================================
// Embeddings
// =================================================================

#[test]
fn embedding_url_uses_default_base() {
    let p = OpenAiProvider;
    let dep = test_deployment_no_base();
    assert_eq!(
        p.embedding_url(&dep),
        "https://api.openai.com/v1/embeddings"
    );
}

#[test]
fn embedding_url_respects_custom_base_and_trims_trailing_slash() {
    let p = OpenAiProvider;
    let mut dep = deployment_with("text-embedding-3-small");
    dep.oxllm_params.api_base = Some("https://api.example.com/v1/".to_string());
    assert_eq!(
        p.embedding_url(&dep),
        "https://api.example.com/v1/embeddings"
    );
}

#[test]
fn embedding_transform_rewrites_model() {
    let p = OpenAiProvider;
    let dep = deployment_with("text-embedding-3-small");
    let req = json!({
        "model": "caller-supplied-model",
        "input": "hello world"
    });
    let out = EmbeddingProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(out["model"], "text-embedding-3-small");
    assert_eq!(out["input"], "hello world");
}

#[test]
fn embedding_transform_preserves_encoding_format_and_dimensions() {
    let p = OpenAiProvider;
    let dep = deployment_with("text-embedding-3-small");
    let req = json!({
        "model": "text-embedding-3-small",
        "input": ["a", "b"],
        "encoding_format": "float",
        "dimensions": 256
    });
    let out = EmbeddingProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(out["encoding_format"], "float");
    assert_eq!(out["dimensions"], 256);
    assert_eq!(out["input"], json!(["a", "b"]));
}

#[test]
fn embedding_wildcard_preserves_requested_model() {
    let p = OpenAiProvider;
    let dep = deployment_with("*");
    let req = json!({
        "model": "text-embedding-3-large",
        "input": "hi"
    });
    let out = EmbeddingProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(out["model"], "text-embedding-3-large");
}

#[test]
fn embedding_transform_response_sanitizes_null_tokens() {
    let p = OpenAiProvider;
    let resp = json!({
        "object": "list",
        "data": [{
            "object": "embedding",
            "index": 0,
            "embedding": [0.1, 0.2, 0.3]
        }],
        "model": "text-embedding-3-small",
        "usage": {"prompt_tokens": null, "total_tokens": null}
    });
    let out =
        EmbeddingProvider::transform_response(&p, &resp, "text-embedding-3-small").expect("ok");
    assert_eq!(out["usage"]["prompt_tokens"], 0);
    assert_eq!(out["usage"]["total_tokens"], 0);
    assert_eq!(out["data"][0]["embedding"][0], 0.1);
}

#[test]
fn provider_supports_embeddings_endpoint() {
    let p = OpenAiProvider;
    assert!(p.supports(Endpoint::Embeddings));
    assert!(
        p.embeddings().is_some(),
        "OpenAI provider must expose an embedding transform"
    );
}

// =================================================================
// Auth
// =================================================================

#[test]
fn org_and_project_headers_set_together() {
    let p = OpenAiProvider;
    let mut dep = deployment_with("gpt-4o");
    dep.oxllm_params
        .extra
        .insert("organization".to_string(), json!("org-abc123"));
    dep.oxllm_params
        .extra
        .insert("project".to_string(), json!("proj_xyz"));
    let creds = ResolvedCredentials::from_key("sk-test");
    let mut headers = std::collections::HashMap::new();
    ChatProvider::apply_auth(&p, &dep, &creds, &mut headers);
    assert_eq!(
        headers.get("openai-organization").map(String::as_str),
        Some("org-abc123")
    );
    assert_eq!(
        headers.get("openai-project").map(String::as_str),
        Some("proj_xyz")
    );
    assert_eq!(
        headers.get("authorization").map(String::as_str),
        Some("Bearer sk-test")
    );
}

#[test]
fn org_header_ignored_when_extra_value_not_string() {
    let p = OpenAiProvider;
    let mut dep = deployment_with("gpt-4o");
    dep.oxllm_params
        .extra
        .insert("organization".to_string(), json!(12345));
    let creds = ResolvedCredentials::from_key("sk-test");
    let mut headers = std::collections::HashMap::new();
    ChatProvider::apply_auth(&p, &dep, &creds, &mut headers);
    assert!(
        !headers.contains_key("openai-organization"),
        "non-string organization values must be ignored"
    );
    assert_eq!(
        headers.get("authorization").map(String::as_str),
        Some("Bearer sk-test")
    );
}

#[test]
fn apply_auth_uses_best_effort_bearer_for_non_api_key_creds() {
    let p = OpenAiProvider;
    let dep = deployment_with("gpt-4o");
    let creds = ResolvedCredentials::GcpBearer {
        token: "ya29.test-token".to_string(),
    };
    let mut headers = std::collections::HashMap::new();
    ChatProvider::apply_auth(&p, &dep, &creds, &mut headers);
    assert_eq!(
        headers.get("authorization").map(String::as_str),
        Some("Bearer ya29.test-token")
    );
}

// =================================================================
// Passthrough
// =================================================================

#[test]
fn supported_params_returns_none() {
    let p = OpenAiProvider;
    assert!(
        p.supported_params().is_none(),
        "the OpenAI provider is a pass-through and must not declare supported params"
    );
}

#[test]
fn transform_request_preserves_unknown_params() {
    let p = OpenAiProvider;
    let req = json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hi"}],
        "foo_bar": 42,
        "some_unknown": {"nested": true}
    });
    let out = ChatProvider::transform_request(&p, &req, &test_deployment()).expect("ok");
    assert_eq!(out["foo_bar"], 42);
    assert_eq!(out["some_unknown"]["nested"], true);
}

#[test]
fn transform_response_preserves_system_fingerprint_and_logprobs() {
    let p = OpenAiProvider;
    let resp = json!({
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "model": "gpt-4o",
        "system_fingerprint": "fp_44709d6fcb",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "logprobs": {
                "content": [{
                    "token": "ok",
                    "logprob": -0.03,
                    "bytes": [111, 107],
                    "top_logprobs": []
                }]
            },
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21}
    });
    let out = ChatProvider::transform_response(&p, &resp, "gpt-4o").expect("ok");
    assert_eq!(out["system_fingerprint"], "fp_44709d6fcb");
    assert_eq!(out["choices"][0]["logprobs"]["content"][0]["token"], "ok");
    assert_eq!(
        out["choices"][0]["logprobs"]["content"][0]["logprob"],
        -0.03
    );
}

// =================================================================
// Null-token edges
// =================================================================

#[test]
fn transform_response_noop_when_usage_not_object() {
    let p = OpenAiProvider;
    let resp = json!({
        "id": "chatcmpl-weird",
        "object": "chat.completion",
        "model": "gpt-4o",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop"
        }],
        "usage": "string usage"
    });
    let out = ChatProvider::transform_response(&p, &resp, "gpt-4o").expect("ok");
    assert_eq!(out, resp, "non-object usage must pass through unchanged");
}

// =================================================================
// supports-none reasoning models (gpt-5.1 / 5.4 / 5.5 / 5.6 / gpt-6)
// =================================================================
//
// gpt-5.1, gpt-5.2, and gpt-5.4 default to reasoning OFF, so temperature /
// top_p / logprobs survive when `reasoning_effort` is omitted. gpt-5.5,
// gpt-5.6, gpt-6-luna, and gpt-6-sol accept `"none"` but default to reasoning
// ON, so those params are stripped unless the caller sets `"none"`.
// gpt-6-astra does not accept `"none"`.

#[test]
fn supports_none_classification() {
    for model in [
        "gpt-5.1",
        "gpt-5.2",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.4-nano",
        "gpt-5.5",
        "gpt-5.6",
        "gpt-5.6-luna",
        "gpt-5.6-terra",
        "gpt-5.6-sol",
        "gpt-6-luna",
        "gpt-6-sol",
        "gpt-6-luna-2026-09-22",
        "gpt-5.1-2025-11-13",
        "gpt-5.4-2026-03-05",
    ] {
        assert!(
            supports_none_reasoning_effort(model),
            "{model} should accept reasoning_effort none"
        );
    }
    for model in [
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-nano",
        "gpt-5-codex",
        "gpt-5-pro",
        "gpt-5.1-codex",
        "gpt-5.1-codex-mini",
        "gpt-5.1-codex-max",
        "gpt-5.2-pro",
        "gpt-5.4-pro",
        "gpt-5.5-pro",
        "gpt-5.1-chat",
        "gpt-5-chat",
        "gpt-4o",
        "gpt-6-astra",
        "gpt-6-astra-2026-09-03",
    ] {
        assert!(
            !supports_none_reasoning_effort(model),
            "{model} should NOT accept reasoning_effort none"
        );
    }
    for model in [
        "gpt-5.1",
        "gpt-5.2",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.4-nano",
    ] {
        assert!(
            defaults_to_none_effort(model),
            "{model} should default to reasoning off"
        );
    }
    for model in [
        "gpt-5.5",
        "gpt-5.6",
        "gpt-5.6-luna",
        "gpt-6-luna",
        "gpt-6-sol",
        "gpt-6-astra",
        "gpt-5",
    ] {
        assert!(
            !defaults_to_none_effort(model),
            "{model} should not default to reasoning off"
        );
    }
}

#[test]
fn gpt5_1_absent_reasoning_effort_preserves_sampling_params() {
    let p = OpenAiProvider;
    let dep = deployment_with("gpt-5.1");
    let req = json!({
        "model": "gpt-5.1",
        "messages": [{"role": "user", "content": "Hi"}],
        "temperature": 0.7,
        "top_p": 0.9,
        "logprobs": true,
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(
        out["temperature"], 0.7,
        "gpt-5.1 defaults to reasoning OFF; temperature must survive"
    );
    assert_eq!(out["top_p"], 0.9);
    assert_eq!(out["logprobs"], true);
}

#[test]
fn gpt5_1_explicit_reasoning_effort_strips_sampling_params() {
    let p = OpenAiProvider;
    let dep = deployment_with("gpt-5.1");
    let req = json!({
        "model": "gpt-5.1",
        "messages": [{"role": "user", "content": "Hi"}],
        "reasoning_effort": "low",
        "temperature": 0.7,
        "top_p": 0.9,
        "logprobs": true,
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert!(
        out.get("temperature").is_none(),
        "gpt-5.1 with active reasoning must drop temperature"
    );
    assert!(out.get("top_p").is_none());
    assert!(out.get("logprobs").is_none());
    assert_eq!(out["reasoning_effort"], "low");
}

#[test]
fn gpt56_tools_without_effort_send_reasoning_none_and_keep_temperature() {
    let p = OpenAiProvider;
    let dep = deployment_with("gpt-5.6");
    let req = json!({
        "model": "gpt-5.6",
        "messages": [{"role": "user", "content": "Hi"}],
        "temperature": 0.2,
        "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}],
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(out["reasoning_effort"], "none");
    assert_eq!(out["temperature"], 0.2);
    assert_eq!(out["tools"][0]["function"]["name"], "lookup");
}

#[test]
fn gpt56_tools_with_explicit_effort_keep_effort_and_drop_temperature() {
    let p = OpenAiProvider;
    let dep = deployment_with("gpt-5.6");
    let req = json!({
        "model": "gpt-5.6",
        "messages": [{"role": "user", "content": "Hi"}],
        "reasoning_effort": "high",
        "temperature": 0.2,
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(out["reasoning_effort"], "high");
    assert!(out.get("temperature").is_none());
}

#[test]
fn gpt56_without_tools_does_not_insert_reasoning_none() {
    let p = OpenAiProvider;
    let dep = deployment_with("gpt-5.6");
    let req = json!({
        "model": "gpt-5.6",
        "messages": [{"role": "user", "content": "Hi"}],
        "temperature": 0.2,
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert!(out.get("reasoning_effort").is_none());
    assert!(out.get("temperature").is_none());
}

#[test]
fn gpt56_empty_tools_does_not_insert_reasoning_none() {
    let p = OpenAiProvider;
    let dep = deployment_with("gpt-5.6");
    let req = json!({
        "model": "gpt-5.6",
        "messages": [{"role": "user", "content": "Hi"}],
        "temperature": 0.2,
        "tools": [],
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert!(out.get("reasoning_effort").is_none());
    assert!(out.get("temperature").is_none());
}

#[test]
fn gpt6_luna_tools_do_not_force_reasoning_none() {
    let p = OpenAiProvider;
    let dep = deployment_with("gpt-6-luna");
    let req = json!({
        "model": "gpt-6-luna",
        "messages": [{"role": "user", "content": "Hi"}],
        "temperature": 0.2,
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert!(out.get("reasoning_effort").is_none());
    assert!(out.get("temperature").is_none());
}

#[test]
fn gpt5_1_reasoning_effort_none_preserves_sampling_params() {
    let p = OpenAiProvider;
    let dep = deployment_with("gpt-5.1");
    let req = json!({
        "model": "gpt-5.1",
        "messages": [{"role": "user", "content": "Hi"}],
        "reasoning_effort": "none",
        "temperature": 0.7,
        "top_p": 0.9,
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(out["temperature"], 0.7);
    assert_eq!(out["top_p"], 0.9);
}

#[test]
fn gpt5_4_absent_reasoning_effort_preserves_sampling_params() {
    let p = OpenAiProvider;
    let dep = deployment_with("gpt-5.4");
    let req = json!({
        "model": "gpt-5.4",
        "messages": [{"role": "user", "content": "Hi"}],
        "temperature": 0.7,
        "top_p": 0.9,
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(out["temperature"], 0.7);
    assert_eq!(out["top_p"], 0.9);
}

#[test]
fn gpt5_base_absent_reasoning_effort_strips_sampling_params() {
    // Negative: the non-supports-none base gpt-5 keeps the historic behaviour
    // (absent effort => reasoning ON => strip).
    let p = OpenAiProvider;
    let dep = deployment_with("gpt-5");
    let req = json!({
        "model": "gpt-5",
        "messages": [{"role": "user", "content": "Hi"}],
        "temperature": 0.7,
        "top_p": 0.9,
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert!(
        out.get("temperature").is_none(),
        "base gpt-5 defaults to reasoning ON; temperature must strip"
    );
    assert!(out.get("top_p").is_none());
}

#[test]
fn gpt5_mini_absent_reasoning_effort_strips_sampling_params() {
    let p = OpenAiProvider;
    let dep = deployment_with("gpt-5-mini");
    let req = json!({
        "model": "gpt-5-mini",
        "messages": [{"role": "user", "content": "Hi"}],
        "temperature": 0.7,
        "top_p": 0.9,
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert!(out.get("temperature").is_none());
    assert!(out.get("top_p").is_none());
}

#[test]
fn gpt5_1_codex_absent_reasoning_effort_strips_sampling_params() {
    // The -codex variant does NOT support none, so it keeps the strip-on-absent
    // behaviour even though it is a versioned gpt-5.1 model.
    let p = OpenAiProvider;
    let dep = deployment_with("gpt-5.1-codex");
    let req = json!({
        "model": "gpt-5.1-codex",
        "messages": [{"role": "user", "content": "Hi"}],
        "temperature": 0.7,
        "top_p": 0.9,
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert!(
        out.get("temperature").is_none(),
        "gpt-5.1-codex is not supports-none; temperature must strip"
    );
    assert!(out.get("top_p").is_none());
}

// =================================================================
// MEDIUM 2: endpoint_url / embedding_url double-append guard
// =================================================================

#[test]
fn endpoint_url_with_full_chat_completions_base_is_not_doubled() {
    let p = OpenAiProvider;
    let mut dep = test_deployment();
    dep.oxllm_params.api_base = Some("https://gateway.example.com/v1/chat/completions".to_string());
    assert_eq!(
        p.endpoint_url(&dep),
        "https://gateway.example.com/v1/chat/completions"
    );
}

#[test]
fn embedding_url_with_full_embeddings_base_is_not_doubled() {
    let p = OpenAiProvider;
    let mut dep = test_deployment();
    dep.oxllm_params.api_base = Some("https://gateway.example.com/v1/embeddings".to_string());
    assert_eq!(
        p.embedding_url(&dep),
        "https://gateway.example.com/v1/embeddings"
    );
}

#[test]
fn endpoint_url_full_base_with_trailing_slash_not_doubled() {
    let p = OpenAiProvider;
    let mut dep = test_deployment();
    dep.oxllm_params.api_base =
        Some("https://gateway.example.com/v1/chat/completions/".to_string());
    assert_eq!(
        p.endpoint_url(&dep),
        "https://gateway.example.com/v1/chat/completions"
    );
}

#[test]
fn endpoint_urls_preserve_queries_on_full_operation_paths() {
    let p = OpenAiProvider;
    let mut dep = test_deployment();
    dep.oxllm_params.api_base =
        Some("https://example.test/models/chat/completions?api-version=test".into());
    assert_eq!(
        ChatProvider::endpoint_url(&p, &dep),
        "https://example.test/models/chat/completions?api-version=test"
    );
    dep.oxllm_params.api_base =
        Some("https://example.test/models/embeddings?api-version=test".into());
    assert_eq!(
        EmbeddingProvider::embedding_url(&p, &dep),
        "https://example.test/models/embeddings?api-version=test"
    );
}

// =================================================================
// MEDIUM 3: dict-form reasoning_effort normalization
// =================================================================

#[test]
fn dict_reasoning_effort_normalized_to_string_in_output() {
    let p = OpenAiProvider;
    let dep = deployment_with("gpt-5");
    let req = json!({
        "model": "gpt-5",
        "messages": [{"role": "user", "content": "Hi"}],
        "reasoning_effort": {"effort": "low", "summary": "detailed"},
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(
        out["reasoning_effort"], "low",
        "dict reasoning_effort must be normalized to its effort string"
    );
    assert!(
        out["reasoning_effort"].is_string(),
        "reasoning_effort must be a string on the wire"
    );
}

#[test]
fn dict_reasoning_effort_drops_summary_key_in_output() {
    let p = OpenAiProvider;
    let dep = deployment_with("gpt-5.1");
    let req = json!({
        "model": "gpt-5.1",
        "messages": [{"role": "user", "content": "Hi"}],
        "reasoning_effort": {"effort": "high", "summary": "detailed"},
        "temperature": 0.7,
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(out["reasoning_effort"], "high");
    // The Responses-only `summary` key must not leak to the Chat Completions body.
    if let Some(obj) = out.as_object() {
        assert!(
            !obj.contains_key("summary"),
            "summary must not appear as a top-level field"
        );
    }
    assert_eq!(
        out["reasoning_effort"].as_str(),
        Some("high"),
        "no nested object should remain under reasoning_effort"
    );
}

#[test]
fn dict_reasoning_effort_without_effort_key_left_untouched() {
    // A dict with no "effort" key is genuinely malformed; leave it so the
    // upstream rejects it (matching historic behaviour).
    let p = OpenAiProvider;
    let dep = deployment_with("gpt-5");
    let req = json!({
        "model": "gpt-5",
        "messages": [{"role": "user", "content": "Hi"}],
        "reasoning_effort": {"summary": "detailed"},
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert!(
        out["reasoning_effort"].is_object(),
        "dict without 'effort' must be forwarded unchanged"
    );
}

#[test]
fn string_reasoning_effort_unchanged_by_normalization() {
    let p = OpenAiProvider;
    let dep = deployment_with("gpt-5");
    let req = json!({
        "model": "gpt-5",
        "messages": [{"role": "user", "content": "Hi"}],
        "reasoning_effort": "low",
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(out["reasoning_effort"], "low");
}

#[test]
fn provider_supports_raw_openai_surface_endpoints() {
    // The raw passthrough gate (`validate_raw_endpoint`) rejects an endpoint
    // the provider does not declare, so the OpenAI adapter must advertise
    // every endpoint it forwards byte-for-byte.
    let p = OpenAiProvider;
    assert!(p.supports(Endpoint::ChatCompletions));
    assert!(p.supports(Endpoint::Embeddings));
    assert!(p.supports(Endpoint::ImagesGenerations));
    assert!(p.supports(Endpoint::AudioSpeech));
    assert!(p.supports(Endpoint::AudioTranscriptions));
    assert!(p.supports(Endpoint::Moderations));
    assert!(!p.supports(Endpoint::Rerank));
}
