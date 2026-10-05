use super::*;
use oxllm_core::Endpoint;
use oxllm_core::deployment::{Deployment, OxllmParams};
use oxllm_core::provider::{ChatProvider, Provider, ResolvedCredentials};
use serde_json::{Value, json};

fn deployment() -> Deployment {
    Deployment {
        model_name: "sonar-reasoning".to_string(),
        oxllm_params: OxllmParams {
            model: "perplexity/sonar-reasoning".to_string(),
            api_key: Some("perplexity-key".to_string()),
            ..Default::default()
        },
        ..Default::default()
    }
}

#[test]
fn endpoint_url_uses_default_perplexity_base() {
    let p = PerplexityProvider;
    assert_eq!(
        p.endpoint_url(&deployment()),
        "https://api.perplexity.ai/chat/completions"
    );
}

#[test]
fn endpoint_url_respects_custom_base() {
    let p = PerplexityProvider;
    let mut dep = deployment();
    dep.oxllm_params.api_base = Some("https://custom.example.com/v1/".to_string());
    assert_eq!(
        p.endpoint_url(&dep),
        "https://custom.example.com/v1/chat/completions"
    );
}

#[test]
fn apply_auth_uses_bearer_token() {
    let p = PerplexityProvider;
    let creds = ResolvedCredentials::from_key("perplexity-secret");
    let mut headers = std::collections::HashMap::new();
    p.apply_auth(&deployment(), &creds, &mut headers);
    assert_eq!(
        headers.get("authorization").map(String::as_str),
        Some("Bearer perplexity-secret")
    );
}

#[test]
fn request_strips_perplexity_prefix_from_model() {
    let p = PerplexityProvider;
    let req = json!({
        "model": "sonar-reasoning",
        "messages": [{"role": "user", "content": "Think about this carefully."}],
        "reasoning_effort": "high",
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["model"], "sonar-reasoning");
    assert_eq!(out["messages"][0]["content"], "Think about this carefully.");
    assert_eq!(out["reasoning_effort"], "high");
}

#[test]
fn request_preserves_reasoning_effort() {
    let p = PerplexityProvider;
    let req = json!({
        "model": "perplexity/sonar-reasoning-pro",
        "messages": [{"role": "user", "content": "Hi"}],
        "reasoning_effort": "medium",
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["reasoning_effort"], "medium");
}

#[test]
fn request_preserves_stream_flag() {
    let p = PerplexityProvider;
    let req = json!({
        "model": "sonar-reasoning",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": true,
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["stream"], true);
}

#[test]
fn response_is_identity() {
    let p = PerplexityProvider;
    let resp = json!({
        "id": "cmpl-test",
        "object": "chat.completion",
        "model": "sonar-reasoning",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "This is a test response.",
                "reasoning_content": "Let me think about this step by step..."
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 9,
            "completion_tokens": 20,
            "total_tokens": 29,
            "completion_tokens_details": {"reasoning_tokens": 15}
        },
    });
    let out = p.transform_response(&resp, "sonar-reasoning").expect("ok");
    assert_eq!(out, resp);
}

#[test]
fn stream_chunk_parses_openai_format() {
    let p = PerplexityProvider;
    let raw = "data: {\"id\":\"x\",\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}";
    let out = p
        .transform_stream_chunk(raw)
        .expect("ok")
        .expect("should produce value");
    assert_eq!(out["id"], "x");
    assert_eq!(out["choices"][0]["delta"]["content"], "hi");
}

#[test]
fn stream_done_returns_none() {
    let p = PerplexityProvider;
    assert!(
        p.transform_stream_chunk("data: [DONE]")
            .expect("ok")
            .is_none()
    );
}

#[test]
fn provider_id_and_supports() {
    let p = PerplexityProvider;
    assert_eq!(p.id(), "perplexity");
    assert!(p.supports(Endpoint::ChatCompletions));
    assert!(!p.supports(Endpoint::Embeddings));
}

#[test]
fn strip_perplexity_prefix_handles_unknown() {
    assert_eq!(
        strip_perplexity_prefix("perplexity/sonar-reasoning"),
        "sonar-reasoning"
    );
    assert_eq!(
        strip_perplexity_prefix("sonar-reasoning"),
        "sonar-reasoning"
    );
}

#[test]
fn supported_params_published() {
    let p = PerplexityProvider;
    let params = p.supported_params().expect("should publish params");
    assert!(params.contains(&"temperature"));
    assert!(params.contains(&"reasoning_effort"));
}

// --- Wildcard deployments ---

#[test]
fn provider_wildcard_with_unmatched_prefix_preserves_requested_model() {
    // The documented `provider/*` pattern must work even when the configured
    // model's prefix doesn't match this provider's strip_prefix. Before the
    // fix, a configured model like `openai/*` on a Perplexity deployment was
    // routed as the literal string `openai/*`; now the body's `model` field
    // wins.
    let p = PerplexityProvider;
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
    let p = PerplexityProvider;
    let mut dep = deployment();
    dep.oxllm_params.model = "openai/*".to_string();
    let req = json!({
        "model": "openai/gpt-4o",
        "messages": [{"role": "user", "content": "Hi"}]
    });
    let out = p.transform_request(&req, &dep).expect("ok");
    // strip_model only strips `perplexity/`, so `openai/gpt-4o` survives intact.
    assert_eq!(out["model"], "openai/gpt-4o");
}

#[test]
fn request_deployment_with_custom_model_respects_strip() {
    let p = PerplexityProvider;
    let mut dep = deployment();
    dep.oxllm_params.model = "perplexity/custom-model-v1".to_string();
    dep.model_name = "custom-model-v1".to_string();
    let req = json!({
        "model": "perplexity/custom-model-v1",
        "messages": [{"role": "user", "content": "Hi"}],
    });
    let out = p.transform_request(&req, &dep).expect("ok");
    assert_eq!(out["model"], "custom-model-v1");
}

// --- Supported params ---

#[test]
fn supported_params_excludes_perplexity_unsupported() {
    // Perplexity's Sonar/pplx models reject these OpenAI-platform fields with a
    // 400; the published supported_params list must exclude them so the host's
    // `drop_params` mechanism strips them before they reach the upstream
    // (parity with LiteLLM's PerplexityChatConfig.get_supported_openai_params).
    // `tools`/`tool_choice` are NOT in this exclusion set: sonar-pro supports
    // function calling (verified live 2026-08).
    let p = PerplexityProvider;
    let params = p.supported_params().expect("should publish params");
    for key in [
        "parallel_tool_calls",
        "seed",
        "logprobs",
        "top_logprobs",
        "logit_bias",
        "store",
        "metadata",
        "modalities",
        "audio",
        "service_tier",
    ] {
        assert!(
            !params.contains(&key),
            "`{key}` must NOT be in Perplexity's supported_params (upstream rejects it)"
        );
    }
    // The narrow surface Perplexity does accept must still be advertised.
    for key in [
        "temperature",
        "top_p",
        "max_tokens",
        "response_format",
        "reasoning_effort",
    ] {
        assert!(
            params.contains(&key),
            "`{key}` must be in Perplexity's supported_params"
        );
    }
}

#[test]
fn drop_params_strips_perplexity_unsupported_before_transform() {
    // Simulates the host's `drop_params = true` flow: top-level keys not in the
    // provider's `supported_params` are stripped before `transform_request`
    // runs. Perplexity rejects `seed` with a 400, so the narrowing must ensure
    // it is removed here (while supported params like `temperature` and — for
    // sonar-pro, `tools`/`tool_choice` — survive).
    let p = PerplexityProvider;
    let supported: std::collections::HashSet<&str> = p
        .supported_params()
        .expect("should publish params")
        .into_iter()
        .collect();
    let req = json!({
        "model": "perplexity/sonar-reasoning",
        "messages": [{"role": "user", "content": "Hi"}],
        "temperature": 0.7,
        "tools": [{"type": "function", "function": {"name": "get_weather"}}],
        "tool_choice": "auto",
        "seed": 42,
    });
    let mut stripped = serde_json::Map::new();
    for (key, value) in req.as_object().expect("object") {
        if supported.contains(key.as_str()) {
            stripped.insert(key.clone(), value.clone());
        }
    }
    let stripped_req = Value::Object(stripped);
    let out = p
        .transform_request(&stripped_req, &deployment())
        .expect("ok");
    assert!(
        out.get("tools").is_some(),
        "tools must survive drop_params (sonar-pro function calling)"
    );
    assert!(
        out.get("tool_choice").is_some(),
        "tool_choice must survive drop_params (sonar-pro function calling)"
    );
    assert!(out.get("seed").is_none(), "seed must be stripped");
    assert_eq!(out["temperature"], 0.7);
}

#[test]
fn transform_request_drops_perplexity_unsupported_newer_params() {
    // Perplexity's narrowed `supported_params` is honored by
    // `build_chat_request_with_params`, so these OpenAI-platform fields are
    // dropped at transform time (parity with LiteLLM's
    // `PerplexityChatConfig.get_supported_openai_params`, which excludes them).
    let p = PerplexityProvider;
    let req = json!({
        "model": "perplexity/sonar-reasoning",
        "messages": [{"role": "user", "content": "Hi"}],
        "logit_bias": {"50256": -100},
        "store": true,
        "metadata": {"user_id": "u_123"},
        "modalities": ["text"],
        "audio": {"voice": "alloy", "format": "wav"},
        "service_tier": "default",
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert!(
        out.get("logit_bias").is_none(),
        "logit_bias must be dropped"
    );
    assert!(out.get("store").is_none(), "store must be dropped");
    assert!(out.get("metadata").is_none(), "metadata must be dropped");
    assert!(
        out.get("modalities").is_none(),
        "modalities must be dropped"
    );
    assert!(out.get("audio").is_none(), "audio must be dropped");
    assert!(
        out.get("service_tier").is_none(),
        "service_tier must be dropped"
    );
    assert_eq!(out["messages"][0]["content"], "Hi");
    assert_eq!(out["model"], "sonar-reasoning");
}

// --- Additional request parameter tests (ported from LiteLLM test_optional_params.py) ---

#[test]
fn request_preserves_response_format_json_object() {
    let p = PerplexityProvider;
    let req = json!({
        "model": "perplexity/sonar-reasoning",
        "messages": [{"role": "user", "content": "Hi"}],
        "response_format": {"type": "json_object"},
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["response_format"]["type"], "json_object");
}

#[test]
fn request_preserves_response_format_json_schema() {
    let p = PerplexityProvider;
    let req = json!({
        "model": "perplexity/sonar-reasoning",
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
fn request_drops_seed() {
    // `seed` is not in Perplexity's supported_params, so the transform must
    // drop it (parity with LiteLLM's PerplexityChatConfig).
    let p = PerplexityProvider;
    let req = json!({
        "model": "perplexity/sonar-reasoning",
        "messages": [{"role": "user", "content": "Hi"}],
        "seed": 42,
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert!(
        out.get("seed").is_none(),
        "seed must be dropped by Perplexity transform"
    );
}

#[test]
fn request_preserves_user() {
    let p = PerplexityProvider;
    let req = json!({
        "model": "perplexity/sonar-reasoning",
        "messages": [{"role": "user", "content": "Hi"}],
        "user": "test_user",
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["user"], "test_user");
}

#[test]
fn request_preserves_stream_options() {
    let p = PerplexityProvider;
    let req = json!({
        "model": "perplexity/sonar-reasoning",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": true,
        "stream_options": {"include_usage": true},
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["stream_options"]["include_usage"], true);
}

#[test]
fn request_preserves_max_completion_tokens() {
    let p = PerplexityProvider;
    let req = json!({
        "model": "perplexity/sonar-reasoning",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_completion_tokens": 2048,
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["max_completion_tokens"], 2048);
}

#[test]
fn request_preserves_max_tokens() {
    let p = PerplexityProvider;
    let req = json!({
        "model": "perplexity/sonar-reasoning",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 500,
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["max_tokens"], 500);
}

#[test]
fn request_keeps_tools_and_tool_choice() {
    // Verified live 2026-08: sonar-pro performs function calling (plain
    // `sonar` answers a clear upstream 400), so the transform must PRESERVE
    // `tools`/`tool_choice` instead of dropping them.
    let p = PerplexityProvider;
    let req = json!({
        "model": "perplexity/sonar-pro",
        "messages": [{"role": "user", "content": "Hi"}],
        "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}],
        "tool_choice": "auto",
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert!(
        out.get("tools").is_some(),
        "tools must be forwarded to Perplexity (sonar-pro supports function calling)"
    );
    assert!(
        out.get("tool_choice").is_some(),
        "tool_choice must be forwarded"
    );
}

#[test]
fn request_drops_parallel_tool_calls() {
    let p = PerplexityProvider;
    let req = json!({
        "model": "perplexity/sonar-reasoning",
        "messages": [{"role": "user", "content": "Hi"}],
        "parallel_tool_calls": true,
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert!(
        out.get("parallel_tool_calls").is_none(),
        "parallel_tool_calls must be dropped by Perplexity transform"
    );
}

#[test]
fn request_drops_stop_sequence_array() {
    // `stop` is not in Perplexity's supported_params, so the transform must
    // drop it (parity with LiteLLM's PerplexityChatConfig).
    let p = PerplexityProvider;
    let req = json!({
        "model": "perplexity/sonar-reasoning",
        "messages": [{"role": "user", "content": "Hi"}],
        "stop": ["stop1", "stop2"],
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert!(
        out.get("stop").is_none(),
        "stop must be dropped by Perplexity transform"
    );
}

#[test]
fn request_drops_stop_sequence_string() {
    let p = PerplexityProvider;
    let req = json!({
        "model": "perplexity/sonar-reasoning",
        "messages": [{"role": "user", "content": "Hi"}],
        "stop": "\n",
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert!(
        out.get("stop").is_none(),
        "stop must be dropped by Perplexity transform"
    );
}

#[test]
fn request_preserves_sampling_params() {
    let p = PerplexityProvider;
    let req = json!({
        "model": "perplexity/sonar-reasoning",
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
fn request_drops_logprobs() {
    // `logprobs`/`top_logprobs` are not in Perplexity's supported_params, so
    // the transform must drop them.
    let p = PerplexityProvider;
    let req = json!({
        "model": "perplexity/sonar-reasoning",
        "messages": [{"role": "user", "content": "Hi"}],
        "logprobs": true,
        "top_logprobs": 3,
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert!(
        out.get("logprobs").is_none(),
        "logprobs must be dropped by Perplexity transform"
    );
    assert!(
        out.get("top_logprobs").is_none(),
        "top_logprobs must be dropped"
    );
}

#[test]
fn request_drops_n() {
    // `n` is not in Perplexity's supported_params, so the transform must
    // drop it.
    let p = PerplexityProvider;
    let req = json!({
        "model": "perplexity/sonar-reasoning",
        "messages": [{"role": "user", "content": "Hi"}],
        "n": 3,
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert!(
        out.get("n").is_none(),
        "n must be dropped by Perplexity transform"
    );
}

#[test]
fn request_drops_null_params() {
    let p = PerplexityProvider;
    let req = json!({
        "model": "perplexity/sonar-reasoning",
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
    let p = PerplexityProvider;
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
    let out = p.transform_response(&resp, "sonar-reasoning").expect("ok");
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
    let p = PerplexityProvider;
    let resp = json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": null, "completion_tokens": null, "total_tokens": null},
    });
    let out = p.transform_response(&resp, "sonar-reasoning").expect("ok");
    assert_eq!(out["usage"]["prompt_tokens"], 0);
    assert_eq!(out["usage"]["completion_tokens"], 0);
    assert_eq!(out["usage"]["total_tokens"], 0);
}

#[test]
fn transform_response_normalizes_null_token_details() {
    let p = PerplexityProvider;
    let resp = json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8, "completion_tokens_details": {"reasoning_tokens": null}},
    });
    let out = p.transform_response(&resp, "sonar-reasoning").expect("ok");
    assert_eq!(
        out["usage"]["completion_tokens_details"]["reasoning_tokens"],
        0
    );
}

#[test]
fn transform_response_preserves_full_structure() {
    // The response transform must be a near-identity: metadata like
    // `system_fingerprint` and per-choice `logprobs` survive untouched.
    let p = PerplexityProvider;
    let resp = json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1729459200,
        "model": "sonar-reasoning",
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
    let out = p.transform_response(&resp, "sonar-reasoning").expect("ok");
    assert_eq!(out["system_fingerprint"], "fp_abc");
    assert_eq!(
        out["choices"][0]["message"]["logprobs"]["content"][0]["token"],
        "Hi"
    );
}

#[test]
fn transform_response_with_empty_choices() {
    let p = PerplexityProvider;
    let resp = json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    });
    let out = p.transform_response(&resp, "sonar-reasoning").expect("ok");
    assert_eq!(out, resp);
    assert!(out["choices"].as_array().unwrap().is_empty());
}

#[test]
fn response_preserves_reasoning_content() {
    // Perplexity-specific: sonar-reasoner models surface their chain of
    // thought in `message.reasoning_content`; the near-identity transform must
    // keep it.
    let p = PerplexityProvider;
    let resp = json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "The answer is 42.",
                "reasoning_content": "Let me think step by step..."
            },
            "finish_reason": "stop"
        }],
    });
    let out = p.transform_response(&resp, "sonar-reasoning").expect("ok");
    assert_eq!(
        out["choices"][0]["message"]["reasoning_content"],
        "Let me think step by step..."
    );
}

// --- Stream chunks ---

#[test]
fn stream_chunk_with_finish_reason() {
    let p = PerplexityProvider;
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
    let p = PerplexityProvider;
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
    let p = PerplexityProvider;
    assert!(
        p.transform_stream_chunk("data: {not valid").is_err(),
        "malformed JSON must surface as an error"
    );
}

#[test]
fn stream_chunk_with_reasoning_content() {
    // Perplexity-specific: streaming chunks carry `delta.reasoning_content` for
    // sonar-reasoner; it must pass through untouched.
    let p = PerplexityProvider;
    let raw = "data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"content\":null,\"reasoning_content\":\"thinking step by step\"}}]}";
    let out = p
        .transform_stream_chunk(raw)
        .expect("ok")
        .expect("should produce value");
    assert_eq!(
        out["choices"][0]["delta"]["reasoning_content"],
        "thinking step by step"
    );
}

// --- Prefix / misc ---

#[test]
fn strip_perplexity_prefix_edge_cases() {
    assert_eq!(strip_perplexity_prefix(""), "");
    assert_eq!(strip_perplexity_prefix("perplexity"), "perplexity");
    assert_eq!(strip_perplexity_prefix("/"), "/");
    assert_eq!(
        strip_perplexity_prefix("perplexity/sub/provider/model"),
        "sub/provider/model"
    );
}

#[test]
fn supports_embeddings_returns_false() {
    let p = PerplexityProvider;
    assert!(!p.supports(Endpoint::Embeddings));
    assert!(p.embeddings().is_none());
}

// --- Golden file parity ---

const CANONICAL_GOLDEN_REQUEST: &str =
    include_str!("../openai_compat/goldens/chat_request_001.json");
const CANONICAL_GOLDEN_RESPONSE: &str =
    include_str!("../openai_compat/goldens/chat_response_001.json");

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
fn test_golden_file_parity() {
    let provider = PerplexityProvider;
    let mut dep = deployment();
    dep.oxllm_params.model = "perplexity/sonar-reasoning".to_string();
    let model = "sonar-reasoning";

    let mut request: Value = serde_json::from_str(CANONICAL_GOLDEN_REQUEST)
        .expect("canonical golden request should be valid JSON");
    request["model"] = json!(model);
    let out = ChatProvider::transform_request(&provider, &request, &dep)
        .expect("golden request transform should succeed");
    assert_semantically_equal(&out, &request);

    let mut response: Value = serde_json::from_str(CANONICAL_GOLDEN_RESPONSE)
        .expect("canonical golden response should be valid JSON");
    response["model"] = json!(model);
    let out = ChatProvider::transform_response(&provider, &response, model)
        .expect("golden response transform should succeed");
    assert_semantically_equal(&out, &response);
}

// --- Perplexity citation / search-query billing ---

#[test]
fn supported_params_includes_web_search_options() {
    let p = PerplexityProvider;
    let params = p.supported_params().expect("should publish params");
    assert!(
        params.contains(&"web_search_options"),
        "web_search_options must be in Perplexity's supported_params"
    );
}

#[test]
fn request_preserves_web_search_options() {
    let p = PerplexityProvider;
    let req = json!({
        "model": "perplexity/sonar",
        "messages": [{"role": "user", "content": "Search the web"}],
        "web_search_options": {"search_context_size": "low"},
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["web_search_options"]["search_context_size"], "low");
}

#[test]
fn transform_response_surfaces_citation_tokens_from_citations_array() {
    let p = PerplexityProvider;
    let resp = json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Paris [1]"},
            "finish_reason": "stop",
        }],
        "citations": ["https://example.com/a", "https://example.com/b", "https://example.com/c"],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    });
    let out = p.transform_response(&resp, "sonar").expect("ok");
    assert_eq!(out["usage"]["citation_tokens"], 3);
}

#[test]
fn transform_response_surfaces_num_search_queries_in_usage() {
    let p = PerplexityProvider;
    let resp = json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Paris"},
            "finish_reason": "stop",
        }],
        "citations": ["https://example.com/a"],
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 2,
            "total_tokens": 7,
            "num_search_queries": 5,
        },
    });
    let out = p.transform_response(&resp, "sonar").expect("ok");
    assert_eq!(out["usage"]["num_search_queries"], 5);
    assert_eq!(
        out["usage"]["prompt_tokens_details"]["web_search_requests"],
        5
    );
}

#[test]
fn transform_response_surfaces_top_level_num_search_queries() {
    let p = PerplexityProvider;
    let resp = json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Paris"},
            "finish_reason": "stop",
        }],
        "num_search_queries": 2,
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    });
    let out = p.transform_response(&resp, "sonar").expect("ok");
    assert_eq!(out["usage"]["num_search_queries"], 2);
}

#[test]
fn transform_response_no_op_without_citations_or_searches() {
    let p = PerplexityProvider;
    let resp = json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Hi"},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    });
    let out = p.transform_response(&resp, "sonar").expect("ok");
    assert!(out["usage"].get("citation_tokens").is_none());
    assert!(out["usage"].get("num_search_queries").is_none());
    assert!(
        out["usage"].get("prompt_tokens_details").is_none(),
        "no prompt_tokens_details added when no search queries"
    );
}

#[test]
fn transform_response_no_op_with_empty_citations() {
    let p = PerplexityProvider;
    let resp = json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Hi"},
            "finish_reason": "stop",
        }],
        "citations": [],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    });
    let out = p.transform_response(&resp, "sonar").expect("ok");
    // Empty citations array → no citation tokens derived.
    assert!(out["usage"].get("citation_tokens").is_none());
}

#[test]
fn response_transforms_non_object_body_does_not_panic() {
    // A non-object upstream body (e.g. a bare JSON string/array error
    // payload) must be left untouched instead of panicking on
    // `out["usage"]` via serde_json IndexMut.
    for body in [
        serde_json::Value::Array(vec![serde_json::Value::from("err")]),
        serde_json::Value::from("upstream error"),
    ] {
        let mut out = body.clone();
        // Directly exercise the internal helper; it is pub(crate) for tests.
        crate::perplexity::apply_perplexity_response_transforms_for_test(&mut out);
        assert_eq!(out, body, "non-object body must be left untouched");
    }
}
