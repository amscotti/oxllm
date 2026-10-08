//! Unit tests for the Meta provider.
//!
//! These cover the OpenAI-compatible identity transform, prefix stripping,
//! request/response passthrough, and streaming chunk handling.

use super::*;
use oxllm_core::Endpoint;
use oxllm_core::deployment::{Deployment, OxllmParams};
use oxllm_core::provider::{ChatProvider, Provider};
use serde_json::json;

fn deployment() -> Deployment {
    Deployment {
        model_name: "muse-spark-1.1".to_string(),
        oxllm_params: OxllmParams {
            model: "meta/muse-spark-1.1".to_string(),
            api_key: Some("meta-key".to_string()),
            ..Default::default()
        },
        ..Default::default()
    }
}

#[test]
fn endpoint_url_uses_default_meta_base() {
    let p = PROVIDER;
    assert_eq!(
        p.endpoint_url(&deployment()),
        "https://api.meta.ai/v1/chat/completions"
    );
}

#[test]
fn endpoint_url_respects_custom_base() {
    let p = PROVIDER;
    let mut dep = deployment();
    dep.oxllm_params.api_base = Some("https://custom.example.com/v1/".to_string());
    assert_eq!(
        p.endpoint_url(&dep),
        "https://custom.example.com/v1/chat/completions"
    );
}

#[test]
fn endpoint_url_does_not_double_append_chat_completions() {
    // The reference URL builder guards with `endswith("/chat/completions")`
    // before appending; a caller who configures `api_base` as the full endpoint
    // URL must not get `/chat/completions` appended a second time.
    let p = PROVIDER;
    let mut dep = deployment();
    dep.oxllm_params.api_base = Some("https://api.meta.ai/v1/chat/completions".to_string());
    assert_eq!(
        p.endpoint_url(&dep),
        "https://api.meta.ai/v1/chat/completions"
    );
}

#[test]
fn strip_meta_prefix_basic() {
    assert_eq!(strip_meta_prefix("meta/muse-spark-1.1"), "muse-spark-1.1");
    assert_eq!(strip_meta_prefix("muse-spark-1.1"), "muse-spark-1.1");
    assert_eq!(strip_meta_prefix(""), "");
}

#[test]
fn supports_embeddings_returns_false() {
    assert!(!PROVIDER.supports(Endpoint::Embeddings));
    assert!(PROVIDER.embeddings().is_none());
}

// --- Wildcard deployments ---

#[test]
fn provider_wildcard_with_unmatched_prefix_preserves_requested_model() {
    // The documented `provider/*` pattern must work even when the configured
    // model's prefix doesn't match this provider's strip_prefix. Before the
    // fix, a configured model like `openai/*` on a Meta deployment was
    // routed as the literal string `openai/*`; now the body's `model` field
    // wins.
    let p = PROVIDER;
    let mut dep = deployment();
    dep.oxllm_params.model = "openai/*".to_string();
    let req = json!({
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "Hi"}]
    });
    let out = p.transform_request(&req, &dep).expect("ok");
    assert_eq!(out["model"], "gpt-4o-mini");
}

#[test]
fn provider_wildcard_preserves_full_path_when_unmatched() {
    // When the client sends a model whose prefix doesn't match the provider's
    // strip_prefix either, the wildcard still kicks in (the whole client
    // string is forwarded after the configured `/*` is treated as wildcard).
    let p = PROVIDER;
    let mut dep = deployment();
    dep.oxllm_params.model = "openai/*".to_string();
    let req = json!({
        "model": "openai/gpt-4o",
        "messages": [{"role": "user", "content": "Hi"}]
    });
    let out = p.transform_request(&req, &dep).expect("ok");
    // strip_model only strips `meta/`, so `openai/gpt-4o` survives intact.
    assert_eq!(out["model"], "openai/gpt-4o");
}

#[test]
fn request_deployment_with_custom_model_respects_strip() {
    let p = PROVIDER;
    let mut dep = deployment();
    dep.oxllm_params.model = "meta/custom-model-v1".to_string();
    dep.model_name = "custom-model-v1".to_string();
    let req = json!({
        "model": "meta/custom-model-v1",
        "messages": [{"role": "user", "content": "Hi"}],
    });
    let out = p.transform_request(&req, &dep).expect("ok");
    assert_eq!(out["model"], "custom-model-v1");
}

// --- Supported params ---

#[test]
fn supported_params_includes_newer_openai_params() {
    // `drop_params: true` strips anything not in supported_params BEFORE the
    // transform runs. The newer OpenAI params must be listed so they survive
    // the strip and reach upstreams that accept them.
    let p = PROVIDER;
    let params = p.supported_params().expect("should publish params");
    for key in [
        "logit_bias",
        "store",
        "metadata",
        "modalities",
        "audio",
        "service_tier",
    ] {
        assert!(
            params.contains(&key),
            "`{key}` must be in SUPPORTED_CHAT_PARAMS so drop_params keeps it"
        );
    }
}

#[test]
fn transform_request_preserves_newer_openai_params() {
    // End-to-end check that the build_chat_request path actually forwards the
    // newly-added params to the upstream body.
    let p = PROVIDER;
    let req = json!({
        "model": "meta/muse-spark-1.1",
        "messages": [{"role": "user", "content": "Hi"}],
        "logit_bias": {"50256": -100},
        "store": true,
        "metadata": {"user_id": "u_123"},
        "modalities": ["text"],
        "audio": {"voice": "alloy", "format": "wav"},
        "service_tier": "default",
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["logit_bias"]["50256"], -100);
    assert_eq!(out["store"], true);
    assert_eq!(out["metadata"]["user_id"], "u_123");
    assert_eq!(out["modalities"][0], "text");
    assert_eq!(out["audio"]["voice"], "alloy");
    assert_eq!(out["service_tier"], "default");
}

// --- Additional request parameter tests (ported from the reference optional-param tests) ---

#[test]
fn request_preserves_response_format_json_object() {
    let p = PROVIDER;
    let req = json!({
        "model": "meta/muse-spark-1.1",
        "messages": [{"role": "user", "content": "Hi"}],
        "response_format": {"type": "json_object"},
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["response_format"]["type"], "json_object");
}

#[test]
fn request_preserves_response_format_json_schema() {
    let p = PROVIDER;
    let req = json!({
        "model": "meta/muse-spark-1.1",
        "messages": [{"role": "user", "content": "Hi"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "weather",
                "schema": {
                    "type": "object",
                    "properties": {"temp": {"type": "number"}},
                },
            },
        },
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["response_format"]["type"], "json_schema");
    assert_eq!(out["response_format"]["json_schema"]["name"], "weather");
}

#[test]
fn request_preserves_seed() {
    let p = PROVIDER;
    let req = json!({
        "model": "meta/muse-spark-1.1",
        "messages": [{"role": "user", "content": "Hi"}],
        "seed": 42,
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["seed"], 42);
}

#[test]
fn request_preserves_user() {
    let p = PROVIDER;
    let req = json!({
        "model": "meta/muse-spark-1.1",
        "messages": [{"role": "user", "content": "Hi"}],
        "user": "test_user",
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["user"], "test_user");
}

#[test]
fn request_preserves_stream_options() {
    let p = PROVIDER;
    let req = json!({
        "model": "meta/muse-spark-1.1",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": true,
        "stream_options": {"include_usage": true},
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["stream_options"]["include_usage"], true);
}

#[test]
fn request_preserves_max_completion_tokens() {
    let p = PROVIDER;
    let req = json!({
        "model": "meta/muse-spark-1.1",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_completion_tokens": 2048,
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["max_completion_tokens"], 2048);
}

#[test]
fn request_preserves_max_tokens() {
    let p = PROVIDER;
    let req = json!({
        "model": "meta/muse-spark-1.1",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 500,
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["max_tokens"], 500);
}

#[test]
fn request_preserves_tools_and_tool_choice() {
    let p = PROVIDER;
    let req = json!({
        "model": "meta/muse-spark-1.1",
        "messages": [{"role": "user", "content": "Hi"}],
        "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}],
        "tool_choice": "auto",
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["tools"][0]["function"]["name"], "get_weather");
    assert_eq!(out["tool_choice"], "auto");
}

#[test]
fn request_preserves_parallel_tool_calls() {
    let p = PROVIDER;
    let req = json!({
        "model": "meta/muse-spark-1.1",
        "messages": [{"role": "user", "content": "Hi"}],
        "parallel_tool_calls": true,
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["parallel_tool_calls"], true);
}

#[test]
fn request_preserves_stop_sequence_array() {
    let p = PROVIDER;
    let req = json!({
        "model": "meta/muse-spark-1.1",
        "messages": [{"role": "user", "content": "Hi"}],
        "stop": ["stop1", "stop2"],
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["stop"][0], "stop1");
    assert_eq!(out["stop"][1], "stop2");
}

#[test]
fn request_preserves_stop_sequence_string() {
    let p = PROVIDER;
    let req = json!({
        "model": "meta/muse-spark-1.1",
        "messages": [{"role": "user", "content": "Hi"}],
        "stop": "\n",
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["stop"], "\n");
}

#[test]
fn request_preserves_sampling_params() {
    let p = PROVIDER;
    let req = json!({
        "model": "meta/muse-spark-1.1",
        "messages": [{"role": "user", "content": "Hi"}],
        "temperature": 0.7,
        "top_p": 0.9,
        "presence_penalty": 0.5,
        "frequency_penalty": 0.2,
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["temperature"], 0.7);
    assert_eq!(out["top_p"], 0.9);
    assert_eq!(out["presence_penalty"], 0.5);
    assert_eq!(out["frequency_penalty"], 0.2);
}

#[test]
fn request_preserves_logprobs() {
    let p = PROVIDER;
    let req = json!({
        "model": "meta/muse-spark-1.1",
        "messages": [{"role": "user", "content": "Hi"}],
        "logprobs": true,
        "top_logprobs": 3,
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["logprobs"], true);
    assert_eq!(out["top_logprobs"], 3);
}

#[test]
fn request_preserves_n() {
    let p = PROVIDER;
    let req = json!({
        "model": "meta/muse-spark-1.1",
        "messages": [{"role": "user", "content": "Hi"}],
        "n": 3,
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["n"], 3);
}

#[test]
fn request_drops_null_params() {
    let p = PROVIDER;
    let req = json!({
        "model": "meta/muse-spark-1.1",
        "messages": [{"role": "user", "content": "Hi"}],
        "temperature": null,
        "max_tokens": 5,
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert!(
        out.get("temperature").is_none(),
        "null temperature should be dropped"
    );
    assert_eq!(out["max_tokens"], 5);
}

// --- Response transforms ---

#[test]
fn transform_response_preserves_tool_calls() {
    let p = PROVIDER;
    let resp = json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}]
            },
            "finish_reason": "stop"
        }],
    });
    let out = p.transform_response(&resp, "muse-spark-1.1").expect("ok");
    assert_eq!(
        out["choices"][0]["message"]["tool_calls"][0]["id"],
        "call_1"
    );
    assert_eq!(
        out["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
        "get_weather"
    );
}

#[test]
fn transform_response_normalizes_null_tokens() {
    let p = PROVIDER;
    let resp = json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": null, "completion_tokens": null, "total_tokens": null},
    });
    let out = p.transform_response(&resp, "muse-spark-1.1").expect("ok");
    assert_eq!(out["usage"]["prompt_tokens"], 0);
    assert_eq!(out["usage"]["completion_tokens"], 0);
    assert_eq!(out["usage"]["total_tokens"], 0);
}

#[test]
fn transform_response_normalizes_null_token_details() {
    let p = PROVIDER;
    let resp = json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8, "completion_tokens_details": {"reasoning_tokens": null}},
    });
    let out = p.transform_response(&resp, "muse-spark-1.1").expect("ok");
    assert_eq!(
        out["usage"]["completion_tokens_details"]["reasoning_tokens"],
        0
    );
}

#[test]
fn transform_response_preserves_full_structure() {
    // The response transform must be a near-identity: metadata like
    // `system_fingerprint` and per-choice `logprobs` survive untouched.
    let p = PROVIDER;
    let resp = json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1729459200,
        "model": "muse-spark-1.1",
        "system_fingerprint": "fp_abc",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hi",
                "logprobs": {"content": [{"token": "Hi", "logprob": -0.5}]}
            },
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    });
    let out = p.transform_response(&resp, "muse-spark-1.1").expect("ok");
    assert_eq!(out["system_fingerprint"], "fp_abc");
    assert_eq!(
        out["choices"][0]["message"]["logprobs"]["content"][0]["token"],
        "Hi"
    );
}

#[test]
fn transform_response_with_empty_choices() {
    let p = PROVIDER;
    let resp = json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    });
    let out = p.transform_response(&resp, "muse-spark-1.1").expect("ok");
    assert_eq!(out, resp);
    assert!(out["choices"].as_array().unwrap().is_empty());
}

#[test]
fn response_preserves_reasoning_tokens_usage() {
    // Meta-specific: reasoning models (e.g. muse-spark-1.1) report
    // `usage.completion_tokens_details.reasoning_tokens`; the near-identity
    // transform must keep it untouched.
    let p = PROVIDER;
    let resp = json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "The answer is 42."
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 9,
            "completion_tokens": 27,
            "total_tokens": 36,
            "completion_tokens_details": {"reasoning_tokens": 27}
        },
    });
    let out = p.transform_response(&resp, "muse-spark-1.1").expect("ok");
    assert_eq!(out, resp);
    assert_eq!(
        out["usage"]["completion_tokens_details"]["reasoning_tokens"],
        27
    );
}

#[test]
fn transform_response_preserves_message_reasoning_content() {
    // Meta's reasoning models (e.g. muse-spark-1.1) may emit
    // `message.reasoning_content` on non-streaming responses (analogous to
    // DeepSeek/Perplexity sonar-reasoning). The near-identity transform must
    // surface it unchanged.
    let p = PROVIDER;
    let resp = json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "The answer is 42.",
                "reasoning_content": "step by step..."
            },
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 9, "completion_tokens": 27, "total_tokens": 36},
    });
    let out = p.transform_response(&resp, "muse-spark-1.1").expect("ok");
    assert_eq!(
        out["choices"][0]["message"]["reasoning_content"],
        "step by step..."
    );
    assert_eq!(out["choices"][0]["message"]["content"], "The answer is 42.");
}

#[test]
fn transform_response_preserves_null_content_on_max_tokens_exhaustion() {
    // Documented quirk (module docs): a reasoning model may leave `content`
    // null and report `finish_reason: "length"` when `max_tokens` is exhausted
    // by reasoning. The near-identity transform must forward that shape
    // verbatim so the client sees the upstream's literal response.
    let p = PROVIDER;
    let resp = json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": null
            },
            "finish_reason": "length"
        }],
        "usage": {
            "prompt_tokens": 9,
            "completion_tokens": 500,
            "total_tokens": 509,
            "completion_tokens_details": {"reasoning_tokens": 500}
        },
    });
    let out = p.transform_response(&resp, "muse-spark-1.1").expect("ok");
    assert!(out["choices"][0]["message"]["content"].is_null());
    assert_eq!(out["choices"][0]["finish_reason"], "length");
    assert_eq!(
        out["usage"]["completion_tokens_details"]["reasoning_tokens"],
        500
    );
}

// --- Stream chunks ---

#[test]
fn stream_chunk_with_finish_reason() {
    let p = PROVIDER;
    let raw = "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"},\"finish_reason\":\"stop\"}]}";
    let out = p
        .transform_stream_chunk(raw)
        .expect("ok")
        .expect("should produce value");
    assert_eq!(out["choices"][0]["delta"]["content"], "Hello");
    assert_eq!(out["choices"][0]["finish_reason"], "stop");
}

#[test]
fn stream_chunk_with_usage() {
    let p = PROVIDER;
    let raw = "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":3,\"total_tokens\":8}}";
    let out = p
        .transform_stream_chunk(raw)
        .expect("ok")
        .expect("should produce value");
    assert_eq!(out["usage"]["prompt_tokens"], 5);
    assert_eq!(out["usage"]["completion_tokens"], 3);
    assert_eq!(out["usage"]["total_tokens"], 8);
}

#[test]
fn stream_chunk_with_malformed_json() {
    let p = PROVIDER;
    assert!(
        p.transform_stream_chunk("data: {not valid").is_err(),
        "malformed JSON must surface as an error"
    );
}

#[test]
fn stream_chunk_with_reasoning_tokens_usage() {
    // Meta-specific: the final stream chunk of a reasoning model carries
    // `usage.completion_tokens_details.reasoning_tokens`; it must pass
    // through untouched alongside the delta.
    let p = PROVIDER;
    let raw = "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":9,\"completion_tokens\":27,\"total_tokens\":36,\"completion_tokens_details\":{\"reasoning_tokens\":27}}}";
    let out = p
        .transform_stream_chunk(raw)
        .expect("ok")
        .expect("should produce value");
    assert_eq!(out["choices"][0]["delta"]["content"], "hi");
    assert_eq!(
        out["usage"]["completion_tokens_details"]["reasoning_tokens"],
        27
    );
}

#[test]
fn stream_chunk_preserves_reasoning_content_delta() {
    // If Meta emits reasoning text under `delta.reasoning_content` during
    // streaming, the passthrough `parse_openai_stream_chunk` must forward it
    // untouched.
    let p = PROVIDER;
    let raw = "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"thinking...\"},\"finish_reason\":null}]}";
    let out = p
        .transform_stream_chunk(raw)
        .expect("ok")
        .expect("should produce value");
    assert_eq!(
        out["choices"][0]["delta"]["reasoning_content"],
        "thinking..."
    );
}

#[test]
fn stream_chunk_normalizes_null_token_counters() {
    // Wire-shape parity with the non-streaming path: a stream usage chunk
    // carrying null token counters must be rewritten to `0` just like
    // `transform_response` does via `sanitize_null_tokens`, rather than
    // forwarding `null` to the client.
    let p = PROVIDER;
    let raw = "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":null,\"completion_tokens\":null,\"total_tokens\":null,\"completion_tokens_details\":{\"reasoning_tokens\":null}}}";
    let out = p
        .transform_stream_chunk(raw)
        .expect("ok")
        .expect("should produce value");
    assert_eq!(out["usage"]["prompt_tokens"], 0);
    assert_eq!(out["usage"]["completion_tokens"], 0);
    assert_eq!(out["usage"]["total_tokens"], 0);
    assert_eq!(
        out["usage"]["completion_tokens_details"]["reasoning_tokens"],
        0
    );
}

// --- Prefix / misc ---

#[test]
fn strip_meta_prefix_edge_cases() {
    assert_eq!(strip_meta_prefix(""), "");
    assert_eq!(strip_meta_prefix("meta"), "meta");
    assert_eq!(strip_meta_prefix("/"), "/");
    assert_eq!(
        strip_meta_prefix("meta/sub/provider/model"),
        "sub/provider/model"
    );
}

// --- Golden file parity ---

const CANONICAL_GOLDEN_REQUEST: &str =
    include_str!("../openai_compat/goldens/chat_request_001.json");
const CANONICAL_GOLDEN_RESPONSE: &str =
    include_str!("../openai_compat/goldens/chat_response_001.json");

#[test]
fn test_golden_file_parity() {
    let provider = PROVIDER;
    let dep = deployment();
    let model = "muse-spark-1.1";

    let mut request: serde_json::Value = serde_json::from_str(CANONICAL_GOLDEN_REQUEST)
        .expect("canonical golden request should be valid JSON");
    request["model"] = json!(model);
    let out = ChatProvider::transform_request(&provider, &request, &dep)
        .expect("golden request transform should succeed");
    assert_eq!(out, request);

    let mut response: serde_json::Value = serde_json::from_str(CANONICAL_GOLDEN_RESPONSE)
        .expect("canonical golden response should be valid JSON");
    response["model"] = json!(model);
    let out = ChatProvider::transform_response(&provider, &response, model)
        .expect("golden response transform should succeed");
    assert_eq!(out, response);
}
