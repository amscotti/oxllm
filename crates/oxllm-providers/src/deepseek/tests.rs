use super::*;
use oxllm_core::Endpoint;
use oxllm_core::deployment::{Deployment, OxllmParams};
use oxllm_core::provider::{ChatProvider, Provider, ResolvedCredentials};
use serde_json::{Value, json};

fn deployment() -> Deployment {
    Deployment {
        model_name: "deepseek-reasoner".to_string(),
        oxllm_params: OxllmParams {
            model: "deepseek/deepseek-reasoner".to_string(),
            api_key: Some("deepseek-key".to_string()),
            ..Default::default()
        },
        ..Default::default()
    }
}

#[test]
fn endpoint_url_uses_default_deepseek_base() {
    let p = DeepSeekProvider;
    assert_eq!(
        p.endpoint_url(&deployment()),
        "https://api.deepseek.com/beta/chat/completions"
    );
}

#[test]
fn default_base_supports_thinking_mode() {
    // The `/beta` default matches the reference and is the endpoint that accepts
    // the native `thinking` parameter mapped from `reasoning_effort`.
    assert_eq!(DEFAULT_DEEPSEEK_BASE, "https://api.deepseek.com/beta");
}

#[test]
fn endpoint_url_respects_custom_base() {
    let p = DeepSeekProvider;
    let mut dep = deployment();
    dep.oxllm_params.api_base = Some("https://custom.example.com/v1/".to_string());
    assert_eq!(
        p.endpoint_url(&dep),
        "https://custom.example.com/v1/chat/completions"
    );
}

#[test]
fn apply_auth_uses_bearer_token() {
    let p = DeepSeekProvider;
    let creds = ResolvedCredentials::from_key("deepseek-secret");
    let mut headers = std::collections::HashMap::new();
    p.apply_auth(&deployment(), &creds, &mut headers);
    assert_eq!(
        headers.get("authorization").map(String::as_str),
        Some("Bearer deepseek-secret")
    );
}

// --- Ported from the reference DeepSeek async-completion tests ---

#[test]
fn request_strips_deepseek_prefix_from_model() {
    let p = DeepSeekProvider;
    let req = json!({
        "model": "deepseek-reasoner",
        "messages": [{"role": "user", "content": "Hello, world!"}],
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["model"], "deepseek-reasoner");
    assert_eq!(out["messages"][0]["content"], "Hello, world!");
}

#[test]
fn request_preserves_stream_flag_false() {
    let p = DeepSeekProvider;
    let req = json!({
        "model": "deepseek-reasoner",
        "messages": [{"role": "user", "content": "Hello, world!"}],
        "stream": false,
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["stream"], false);
}

#[test]
fn request_preserves_stream_flag_true() {
    let p = DeepSeekProvider;
    let req = json!({
        "model": "deepseek-reasoner",
        "messages": [{"role": "user", "content": "Hello, world!"}],
        "stream": true,
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["stream"], true);
}

#[test]
fn response_is_identity() {
    let p = DeepSeekProvider;
    let resp = json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    });
    let out = p
        .transform_response(&resp, "deepseek-reasoner")
        .expect("ok");
    assert_eq!(out, resp);
}

#[test]
fn stream_chunk_parses_openai_format() {
    let p = DeepSeekProvider;
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
    let p = DeepSeekProvider;
    assert!(
        p.transform_stream_chunk("data: [DONE]")
            .expect("ok")
            .is_none()
    );
}

#[test]
fn provider_id_and_supports() {
    let p = DeepSeekProvider;
    assert_eq!(p.id(), "deepseek");
    assert!(p.supports(Endpoint::ChatCompletions));
    assert!(!p.supports(Endpoint::Embeddings));
}

#[test]
fn strip_deepseek_prefix_handles_unknown() {
    assert_eq!(
        strip_deepseek_prefix("deepseek/deepseek-reasoner"),
        "deepseek-reasoner"
    );
    assert_eq!(
        strip_deepseek_prefix("deepseek-reasoner"),
        "deepseek-reasoner"
    );
}

#[test]
fn supported_params_published() {
    let p = DeepSeekProvider;
    let params = p.supported_params().expect("should publish params");
    assert!(params.contains(&"temperature"));
    assert!(params.contains(&"tools"));
}

#[test]
fn request_passes_known_params_and_drops_unknown() {
    let p = DeepSeekProvider;
    let req = json!({
        "model": "deepseek-reasoner",
        "messages": [{"role": "user", "content": "Hi"}],
        "temperature": 0.5,
        "max_tokens": 100,
        "stream": false,
        "n": 2,
        "some_unknown_param": "ignored",
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["temperature"], 0.5);
    assert_eq!(out["max_tokens"], 100);
    assert_eq!(out["n"], 2);
    assert!(out.get("some_unknown_param").is_none());
}

#[test]
fn supported_params_includes_reasoning_effort() {
    let p = DeepSeekProvider;
    let params = p.supported_params().expect("should publish params");
    assert!(params.contains(&"reasoning_effort"));
}

#[test]
fn request_maps_reasoning_effort_to_thinking() {
    // DeepSeek's native thinking control is `{"type": "enabled"|"disabled"}`.
    // An OpenAI-style `reasoning_effort` is mapped to it ("none" → disabled,
    // anything else → enabled) and consumed — never forwarded verbatim (parity
    // with the reference DeepSeek parameter mapping).
    let p = DeepSeekProvider;
    let req = json!({
        "model": "deepseek/deepseek-reasoner",
        "messages": [{"role": "user", "content": "Hi"}],
        "reasoning_effort": "high",
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["thinking"]["type"], "enabled");
    assert!(
        out.get("reasoning_effort").is_none(),
        "reasoning_effort must be consumed, not forwarded"
    );
}

#[test]
fn request_maps_reasoning_effort_none_to_thinking_disabled() {
    let p = DeepSeekProvider;
    let req = json!({
        "model": "deepseek/deepseek-reasoner",
        "messages": [{"role": "user", "content": "Hi"}],
        "reasoning_effort": "none",
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["thinking"]["type"], "disabled");
    assert!(out.get("reasoning_effort").is_none());
}

#[test]
fn request_explicit_thinking_wins_over_reasoning_effort() {
    // An explicit `thinking` dict always takes precedence over reasoning_effort.
    let p = DeepSeekProvider;
    let req = json!({
        "model": "deepseek/deepseek-v3.2",
        "messages": [{"role": "user", "content": "Hi"}],
        "reasoning_effort": "high",
        "thinking": {"type": "disabled"},
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["thinking"]["type"], "disabled");
    assert!(out.get("reasoning_effort").is_none());
}

#[test]
fn request_thinking_param_survives_transform() {
    // The native `thinking` parameter is not on the shared SUPPORTED_CHAT_PARAMS
    // list (so build_chat_request drops it); the DeepSeek transform must recover
    // it from the original request (issue: thinking was silently dropped).
    let p = DeepSeekProvider;
    let req = json!({
        "model": "deepseek/deepseek-v3.2",
        "messages": [{"role": "user", "content": "Hi"}],
        "thinking": {"type": "enabled"},
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["thinking"]["type"], "enabled");
}

#[test]
fn wildcard_deployment_preserves_requested_model() {
    let p = DeepSeekProvider;
    let mut dep = deployment();
    dep.oxllm_params.model = "deepseek/*".to_string();
    let req = json!({
        "model": "deepseek/deepseek-chat",
        "messages": [{"role": "user", "content": "Hi"}]
    });
    let out = p.transform_request(&req, &dep).expect("ok");
    assert_eq!(out["model"], "deepseek-chat");
}

#[test]
fn strip_deepseek_prefix_with_various_inputs() {
    assert_eq!(
        strip_deepseek_prefix("deepseek/deepseek-chat"),
        "deepseek-chat"
    );
    assert_eq!(
        strip_deepseek_prefix("deepseek/deepseek-reasoner"),
        "deepseek-reasoner"
    );
    assert_eq!(strip_deepseek_prefix("deepseek-chat"), "deepseek-chat");
}

// --- Wildcard deployments ---

#[test]
fn provider_wildcard_with_unmatched_prefix_preserves_requested_model() {
    // The documented `provider/*` pattern must work even when the configured
    // model's prefix doesn't match this provider's strip_prefix. Before the
    // fix, a configured model like `openai/*` on a DeepSeek deployment was
    // routed as the literal string `openai/*`; now the body's `model` field
    // wins.
    let p = DeepSeekProvider;
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
    let p = DeepSeekProvider;
    let mut dep = deployment();
    dep.oxllm_params.model = "openai/*".to_string();
    let req = json!({
        "model": "openai/gpt-4o",
        "messages": [{"role": "user", "content": "Hi"}]
    });
    let out = p.transform_request(&req, &dep).expect("ok");
    // strip_model only strips `deepseek/`, so `openai/gpt-4o` survives intact.
    assert_eq!(out["model"], "openai/gpt-4o");
}

#[test]
fn request_deployment_with_custom_model_respects_strip() {
    let p = DeepSeekProvider;
    let mut dep = deployment();
    dep.oxllm_params.model = "deepseek/custom-model-v1".to_string();
    dep.model_name = "custom-model-v1".to_string();
    let req = json!({
        "model": "deepseek/custom-model-v1",
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
    let p = DeepSeekProvider;
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
    let p = DeepSeekProvider;
    let req = json!({
        "model": "deepseek/deepseek-reasoner",
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
    let p = DeepSeekProvider;
    let req = json!({
        "model": "deepseek/deepseek-reasoner",
        "messages": [{"role": "user", "content": "Hi"}],
        "response_format": {"type": "json_object"},
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["response_format"]["type"], "json_object");
}

#[test]
fn request_preserves_response_format_json_schema() {
    let p = DeepSeekProvider;
    let req = json!({
        "model": "deepseek/deepseek-reasoner",
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
    let p = DeepSeekProvider;
    let req = json!({
        "model": "deepseek/deepseek-reasoner",
        "messages": [{"role": "user", "content": "Hi"}],
        "seed": 42,
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["seed"], 42);
}

#[test]
fn request_preserves_user() {
    let p = DeepSeekProvider;
    let req = json!({
        "model": "deepseek/deepseek-reasoner",
        "messages": [{"role": "user", "content": "Hi"}],
        "user": "test_user",
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["user"], "test_user");
}

#[test]
fn request_preserves_stream_options() {
    let p = DeepSeekProvider;
    let req = json!({
        "model": "deepseek/deepseek-reasoner",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": true,
        "stream_options": {"include_usage": true},
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["stream_options"]["include_usage"], true);
}

#[test]
fn request_preserves_max_completion_tokens() {
    let p = DeepSeekProvider;
    let req = json!({
        "model": "deepseek/deepseek-reasoner",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_completion_tokens": 2048,
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["max_completion_tokens"], 2048);
}

#[test]
fn request_preserves_max_tokens() {
    let p = DeepSeekProvider;
    let req = json!({
        "model": "deepseek/deepseek-reasoner",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 500,
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["max_tokens"], 500);
}

#[test]
fn request_preserves_tools_and_tool_choice() {
    let p = DeepSeekProvider;
    let req = json!({
        "model": "deepseek/deepseek-reasoner",
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
    let p = DeepSeekProvider;
    let req = json!({
        "model": "deepseek/deepseek-reasoner",
        "messages": [{"role": "user", "content": "Hi"}],
        "parallel_tool_calls": true,
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["parallel_tool_calls"], true);
}

#[test]
fn request_preserves_stop_sequence_array() {
    let p = DeepSeekProvider;
    let req = json!({
        "model": "deepseek/deepseek-reasoner",
        "messages": [{"role": "user", "content": "Hi"}],
        "stop": ["stop1", "stop2"],
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["stop"][0], "stop1");
    assert_eq!(out["stop"][1], "stop2");
}

#[test]
fn request_preserves_stop_sequence_string() {
    let p = DeepSeekProvider;
    let req = json!({
        "model": "deepseek/deepseek-reasoner",
        "messages": [{"role": "user", "content": "Hi"}],
        "stop": "\n",
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["stop"], "\n");
}

#[test]
fn request_preserves_sampling_params() {
    let p = DeepSeekProvider;
    let req = json!({
        "model": "deepseek/deepseek-reasoner",
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
    let p = DeepSeekProvider;
    let req = json!({
        "model": "deepseek/deepseek-reasoner",
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
    let p = DeepSeekProvider;
    let req = json!({
        "model": "deepseek/deepseek-reasoner",
        "messages": [{"role": "user", "content": "Hi"}],
        "n": 3,
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["n"], 3);
}

#[test]
fn request_drops_null_params() {
    let p = DeepSeekProvider;
    let req = json!({
        "model": "deepseek/deepseek-reasoner",
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

// --- DeepSeek thinking-mode multi-turn reasoning_content (issue #2) ---

#[test]
fn thinking_mode_injects_reasoning_content_on_prior_assistant_turn() {
    // deepseek-reasoner has always-on reasoning; the API requires every prior
    // assistant message to carry `reasoning_content`. A client that did not
    // round-trip it must get a single-space placeholder injected.
    let p = DeepSeekProvider;
    let req = json!({
        "model": "deepseek/deepseek-reasoner",
        "messages": [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "And 3+3?"},
        ],
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(
        out["messages"][1]["reasoning_content"], " ",
        "prior assistant turn missing reasoning_content must get a placeholder"
    );
}

#[test]
fn thinking_mode_preserves_existing_reasoning_content() {
    // An assistant turn that already carries non-empty reasoning_content is left
    // untouched.
    let p = DeepSeekProvider;
    let req = json!({
        "model": "deepseek/deepseek-reasoner",
        "messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello", "reasoning_content": "I should be friendly."},
            {"role": "user", "content": "Bye"},
        ],
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(
        out["messages"][1]["reasoning_content"],
        "I should be friendly."
    );
}

#[test]
fn thinking_mode_promotes_provider_specific_fields_reasoning_content() {
    // When reasoning_content was saved in provider_specific_fields (the reference
    // convention), it is promoted onto the assistant message.
    let p = DeepSeekProvider;
    let req = json!({
        "model": "deepseek/deepseek-reasoner",
        "messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello", "provider_specific_fields": {"reasoning_content": "promoted chain"}},
            {"role": "user", "content": "Bye"},
        ],
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["messages"][1]["reasoning_content"], "promoted chain");
}

#[test]
fn thinking_mode_via_explicit_thinking_enables_injection() {
    // A non-always-on model (deepseek-v3.2) opts into thinking via the native
    // `thinking` param; prior assistant turns must then also be patched.
    let p = DeepSeekProvider;
    let mut dep = deployment();
    dep.oxllm_params.model = "deepseek/deepseek-v3.2".to_string();
    dep.model_name = "deepseek-v3.2".to_string();
    let req = json!({
        "model": "deepseek/deepseek-v3.2",
        "messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "Bye"},
        ],
        "thinking": {"type": "enabled"},
    });
    let out = p.transform_request(&req, &dep).expect("ok");
    assert_eq!(out["thinking"]["type"], "enabled");
    assert_eq!(out["messages"][1]["reasoning_content"], " ");
}

#[test]
fn non_thinking_model_does_not_inject_reasoning_content() {
    // deepseek-chat has no reasoning; assistant messages must NOT be patched.
    let p = DeepSeekProvider;
    let mut dep = deployment();
    dep.oxllm_params.model = "deepseek/deepseek-chat".to_string();
    dep.model_name = "deepseek-chat".to_string();
    let req = json!({
        "model": "deepseek/deepseek-chat",
        "messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "Bye"},
        ],
    });
    let out = p.transform_request(&req, &dep).expect("ok");
    assert!(
        out["messages"][1].get("reasoning_content").is_none(),
        "non-thinking model must not get reasoning_content injected"
    );
}

#[test]
fn thinking_mode_only_patches_assistant_messages() {
    // user/system/tool messages are never touched.
    let p = DeepSeekProvider;
    let req = json!({
        "model": "deepseek/deepseek-reasoner",
        "messages": [
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "Hi"},
        ],
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert!(out["messages"][0].get("reasoning_content").is_none());
    assert!(out["messages"][1].get("reasoning_content").is_none());
}

// --- DeepSeek non-function tool filtering (issue #4) ---

#[test]
fn drops_non_function_tool_types() {
    // DeepSeek rejects any tool variant other than `type: "function"` with a
    // 400. A request bridged from /v1/responses carrying a non-function tool
    // must have it filtered out, keeping the function entries.
    let p = DeepSeekProvider;
    let req = json!({
        "model": "deepseek/deepseek-chat",
        "messages": [{"role": "user", "content": "Hi"}],
        "tools": [
            {"type": "namespace", "namespace": {"name": "shell"}},
            {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}},
        ],
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    let tools = out["tools"].as_array().expect("tools should remain");
    assert_eq!(tools.len(), 1, "only the function tool should survive");
    assert_eq!(tools[0]["type"], "function");
    assert_eq!(tools[0]["function"]["name"], "get_weather");
}

#[test]
fn drops_all_tools_when_none_are_function() {
    let p = DeepSeekProvider;
    let req = json!({
        "model": "deepseek/deepseek-chat",
        "messages": [{"role": "user", "content": "Hi"}],
        "tools": [{"type": "namespace", "namespace": {"name": "shell"}}],
        "tool_choice": "auto",
        "parallel_tool_calls": true,
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert!(out.get("tools").is_none(), "tools must be dropped entirely");
    assert!(
        out.get("tool_choice").is_none(),
        "tool_choice must be dropped with no function tools"
    );
    assert!(
        out.get("parallel_tool_calls").is_none(),
        "parallel_tool_calls must be dropped with no function tools"
    );
}

#[test]
fn keeps_all_function_tools_unchanged() {
    // When every tool is already type:function, nothing is filtered.
    let p = DeepSeekProvider;
    let req = json!({
        "model": "deepseek/deepseek-chat",
        "messages": [{"role": "user", "content": "Hi"}],
        "tools": [
            {"type": "function", "function": {"name": "get_weather"}},
            {"type": "function", "function": {"name": "get_time"}},
        ],
        "tool_choice": "auto",
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["tools"].as_array().unwrap().len(), 2);
    assert_eq!(out["tool_choice"], "auto");
}

// --- Response transforms ---

#[test]
fn transform_response_preserves_tool_calls() {
    let p = DeepSeekProvider;
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
    let out = p
        .transform_response(&resp, "deepseek-reasoner")
        .expect("ok");
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
    let p = DeepSeekProvider;
    let resp = json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": null, "completion_tokens": null, "total_tokens": null},
    });
    let out = p
        .transform_response(&resp, "deepseek-reasoner")
        .expect("ok");
    assert_eq!(out["usage"]["prompt_tokens"], 0);
    assert_eq!(out["usage"]["completion_tokens"], 0);
    assert_eq!(out["usage"]["total_tokens"], 0);
}

#[test]
fn transform_response_normalizes_null_token_details() {
    let p = DeepSeekProvider;
    let resp = json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8, "completion_tokens_details": {"reasoning_tokens": null}},
    });
    let out = p
        .transform_response(&resp, "deepseek-reasoner")
        .expect("ok");
    assert_eq!(
        out["usage"]["completion_tokens_details"]["reasoning_tokens"],
        0
    );
}

#[test]
fn transform_response_preserves_full_structure() {
    // The response transform must be a near-identity: metadata like
    // `system_fingerprint` and per-choice `logprobs` survive untouched.
    let p = DeepSeekProvider;
    let resp = json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1729459200,
        "model": "deepseek-chat",
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
    let out = p.transform_response(&resp, "deepseek-chat").expect("ok");
    assert_eq!(out["system_fingerprint"], "fp_abc");
    assert_eq!(
        out["choices"][0]["message"]["logprobs"]["content"][0]["token"],
        "Hi"
    );
}

#[test]
fn transform_response_with_empty_choices() {
    let p = DeepSeekProvider;
    let resp = json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    });
    let out = p.transform_response(&resp, "deepseek-chat").expect("ok");
    assert_eq!(out, resp);
    assert!(out["choices"].as_array().unwrap().is_empty());
}

#[test]
fn response_preserves_reasoning_content() {
    // DeepSeek-specific: reasoning models surface their chain of thought in
    // `message.reasoning_content`; the near-identity transform must keep it.
    let p = DeepSeekProvider;
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
    let out = p
        .transform_response(&resp, "deepseek-reasoner")
        .expect("ok");
    assert_eq!(
        out["choices"][0]["message"]["reasoning_content"],
        "Let me think step by step..."
    );
}

// --- Stream chunks ---

#[test]
fn stream_chunk_with_finish_reason() {
    let p = DeepSeekProvider;
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
    let p = DeepSeekProvider;
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
    let p = DeepSeekProvider;
    assert!(
        p.transform_stream_chunk("data: {not valid").is_err(),
        "malformed JSON must surface as an error"
    );
}

#[test]
fn stream_chunk_with_reasoning_content() {
    // DeepSeek-specific: streaming chunks carry `delta.reasoning_content` for
    // reasoning models; it must pass through untouched.
    let p = DeepSeekProvider;
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
fn strip_deepseek_prefix_edge_cases() {
    assert_eq!(strip_deepseek_prefix(""), "");
    assert_eq!(strip_deepseek_prefix("deepseek"), "deepseek");
    assert_eq!(strip_deepseek_prefix("/"), "/");
    assert_eq!(
        strip_deepseek_prefix("deepseek/sub/provider/model"),
        "sub/provider/model"
    );
}

#[test]
fn supports_embeddings_returns_false() {
    let p = DeepSeekProvider;
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
    let provider = DeepSeekProvider;
    let mut dep = deployment();
    dep.oxllm_params.model = "deepseek/deepseek-chat".to_string();
    let model = "deepseek-chat";

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
