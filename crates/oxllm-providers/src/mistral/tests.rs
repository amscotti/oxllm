use super::*;
use crate::openai_compat::SUPPORTED_CHAT_PARAMS;
use oxllm_core::Endpoint;
use oxllm_core::deployment::{Deployment, OxllmParams};
use oxllm_core::provider::{ChatProvider, EmbeddingProvider, Provider, ResolvedCredentials};
use serde_json::json;

fn deployment() -> Deployment {
    Deployment {
        model_name: "mistral-large-latest".to_string(),
        oxllm_params: OxllmParams {
            model: "mistral/mistral-large-latest".to_string(),
            api_key: Some("mistral-key".to_string()),
            ..Default::default()
        },
        ..Default::default()
    }
}

#[test]
fn endpoint_url_uses_default_mistral_base() {
    let p = MistralProvider;
    assert_eq!(
        p.endpoint_url(&deployment()),
        "https://api.mistral.ai/v1/chat/completions"
    );
}

#[test]
fn embedding_url_uses_default_mistral_base() {
    let p = MistralProvider;
    assert_eq!(
        p.embedding_url(&deployment()),
        "https://api.mistral.ai/v1/embeddings"
    );
}

#[test]
fn endpoint_url_respects_custom_base() {
    let p = MistralProvider;
    let mut dep = deployment();
    dep.oxllm_params.api_base = Some("https://custom.example.com/v1/".to_string());
    assert_eq!(
        p.endpoint_url(&dep),
        "https://custom.example.com/v1/chat/completions"
    );
}

#[test]
fn apply_auth_uses_bearer_token() {
    let p = MistralProvider;
    let creds = ResolvedCredentials::from_key("mistral-secret");
    let mut headers = std::collections::HashMap::new();
    ChatProvider::apply_auth(&p, &deployment(), &creds, &mut headers);
    assert_eq!(
        headers.get("authorization").map(String::as_str),
        Some("Bearer mistral-secret")
    );
}

#[test]
fn request_strips_mistral_prefix_from_model() {
    let p = MistralProvider;
    let req = json!({
    "model": "mistral-large-latest",
    "messages": [{"role": "user", "content": "Hi"}],
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["model"], "mistral-large-latest");
    assert_eq!(out["messages"][0]["content"], "Hi");
}

#[test]
fn request_preserves_stream_flag() {
    let p = MistralProvider;
    let req = json!({
    "model": "mistral-large-latest",
    "messages": [{"role": "user", "content": "Hi"}],
    "stream": true,
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["stream"], true);
}

#[test]
fn embedding_request_strips_mistral_prefix_from_model() {
    let p = MistralProvider;
    let mut dep = deployment();
    dep.oxllm_params.model = "mistral/mistral-embed".to_string();
    let req = json!({
    "model": "mistral-embed",
    "input": "hello world",
    });
    let out = EmbeddingProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(out["model"], "mistral-embed");
    assert_eq!(out["input"], "hello world");
}

#[test]
fn response_is_identity() {
    let p = MistralProvider;
    let resp = json!({
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    });
    let out = ChatProvider::transform_response(&p, &resp, "mistral-large-latest").expect("ok");
    assert_eq!(out, resp);
}

#[test]
fn stream_chunk_parses_openai_format() {
    let p = MistralProvider;
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
    let p = MistralProvider;
    assert!(
        p.transform_stream_chunk("data: [DONE]")
            .expect("ok")
            .is_none()
    );
}

#[test]
fn provider_id_and_supports() {
    let p = MistralProvider;
    assert_eq!(p.id(), "mistral");
    assert!(p.supports(Endpoint::ChatCompletions));
    assert!(p.supports(Endpoint::Embeddings));
    assert!(p.supports(Endpoint::AudioTranscriptions));
    assert!(p.embeddings().is_some());
}

#[test]
fn strip_mistral_prefix_handles_unknown() {
    assert_eq!(
        strip_mistral_prefix("mistral/mistral-large-latest"),
        "mistral-large-latest"
    );
    assert_eq!(
        strip_mistral_prefix("mistral-large-latest"),
        "mistral-large-latest"
    );
}

#[test]
fn supported_params_published() {
    let p = MistralProvider;
    let params = ChatProvider::supported_params(&p).expect("should publish params");
    assert!(params.contains(&"temperature"));
    assert!(params.contains(&"tools"));
}

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
    let provider = MistralProvider;
    let deployment = deployment();
    let model = "mistral-large-latest";

    let mut request: Value = serde_json::from_str(CANONICAL_GOLDEN_REQUEST)
        .expect("canonical golden request should be valid JSON");
    request["model"] = json!(model);
    let out = ChatProvider::transform_request(&provider, &request, &deployment)
        .expect("golden request transform should succeed");
    assert_semantically_equal(&out, &request);

    let mut response: Value = serde_json::from_str(CANONICAL_GOLDEN_RESPONSE)
        .expect("canonical golden response should be valid JSON");
    response["model"] = json!(model);
    let out = ChatProvider::transform_response(&provider, &response, model)
        .expect("golden response transform should succeed");
    assert_semantically_equal(&out, &response);
}

// --- Additional request parameter tests (ported from LiteLLM test_optional_params.py and test_mistral_api.py) ---

#[test]
fn request_preserves_mistral_accepted_extra_params() {
    // Mistral accepts `logit_bias` (on the narrowed
    // `MISTRAL_SUPPORTED_CHAT_PARAMS` list). The OpenAI-platform-only fields
    // `store`, `modalities`, `audio`, `service_tier`, `metadata` are NOT on
    // the list and are stripped by the host's `drop_params` path — see
    // `drop_params_strips_openai_only_field_from_mistral_request`. `metadata`
    // is not in Mistral's documented `/chat/completions` surface and is
    // excluded to avoid a 422 (parity with LiteLLM's
    // `get_supported_openai_params`, which also omits it).
    let p = MistralProvider;
    let req = json!({
        "model": "mistral/mistral-large-latest",
        "messages": [{"role": "user", "content": "Hi"}],
        "logit_bias": {"50256": -100},
        "metadata": {"user_id": "u_123"},
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["logit_bias"]["50256"], -100);
    assert!(
        out.get("metadata").is_none(),
        "metadata must not be forwarded to Mistral"
    );
}

#[test]
fn request_preserves_stream_options() {
    let p = MistralProvider;
    let req = json!({
        "model": "mistral/mistral-large-latest",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": true,
        "stream_options": {"include_usage": true},
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["stream_options"]["include_usage"], true);
}

#[test]
fn request_maps_seed_to_random_seed_and_preserves_user() {
    // M1: Mistral reads `random_seed`, not `seed`.
    let p = MistralProvider;
    let req = json!({
        "model": "mistral/mistral-large-latest",
        "messages": [{"role": "user", "content": "Hi"}],
        "seed": 123,
        "user": "test_user",
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert!(out.get("seed").is_none(), "`seed` must be renamed away");
    assert_eq!(out["random_seed"], 123);
    assert_eq!(out["user"], "test_user");
}

#[test]
fn request_preserves_response_format() {
    let p = MistralProvider;
    let req = json!({
        "model": "mistral/mistral-large-latest",
        "messages": [{"role": "user", "content": "Hi"}],
        "response_format": {"type": "json_object"},
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["response_format"]["type"], "json_object");
}

#[test]
fn request_preserves_stop_sequences() {
    let p = MistralProvider;
    let req = json!({
        "model": "mistral/mistral-large-latest",
        "messages": [{"role": "user", "content": "Hi"}],
        "stop": ["word1", "word2"],
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["stop"][0], "word1");
    assert_eq!(out["stop"][1], "word2");
}

#[test]
fn non_magistral_drops_reasoning_effort() {
    // Mistral does not accept `reasoning_effort` on the wire for non-magistral
    // models; the transform consumes it and never forwards it.
    let p = MistralProvider;
    let req = json!({
        "model": "mistral/mistral-large-latest",
        "messages": [{"role": "user", "content": "Hi"}],
        "reasoning_effort": "low",
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert!(
        out.get("reasoning_effort").is_none(),
        "reasoning_effort must not be forwarded to Mistral"
    );
}

#[test]
fn request_preserves_parallel_tool_calls() {
    let p = MistralProvider;
    let req = json!({
        "model": "mistral/mistral-large-latest",
        "messages": [{"role": "user", "content": "Hi"}],
        "parallel_tool_calls": true,
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["parallel_tool_calls"], true);
}

#[test]
fn request_maps_max_completion_tokens_to_max_tokens() {
    // M2: Mistral reads `max_tokens`, not `max_completion_tokens`.
    let p = MistralProvider;
    let req = json!({
        "model": "mistral/mistral-large-latest",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_completion_tokens": 4096,
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert!(
        out.get("max_completion_tokens").is_none(),
        "`max_completion_tokens` must be renamed away"
    );
    assert_eq!(out["max_tokens"], 4096);
}

#[test]
fn request_drops_unknown_params() {
    let p = MistralProvider;
    let req = json!({
        "model": "mistral/mistral-large-latest",
        "messages": [{"role": "user", "content": "Hi"}],
        "temperature": 0.7,
        "some_unknown_param": "should_be_dropped",
        "another_unknown": 42,
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["temperature"], 0.7);
    assert!(out.get("some_unknown_param").is_none());
    assert!(out.get("another_unknown").is_none());
}

// --- Embedding parameter tests ---

#[test]
fn embedding_request_with_encoding_format_and_dimensions() {
    let p = MistralProvider;
    let mut dep = deployment();
    dep.oxllm_params.model = "mistral/mistral-embed".to_string();
    let req = json!({
        "model": "mistral-embed",
        "input": "hello world",
        "encoding_format": "float",
        "dimensions": 256,
    });
    let out = EmbeddingProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(out["encoding_format"], "float");
    assert_eq!(out["dimensions"], 256);
}

#[test]
fn embedding_request_with_user_param() {
    let p = MistralProvider;
    let mut dep = deployment();
    dep.oxllm_params.model = "mistral/mistral-embed".to_string();
    let req = json!({
        "model": "mistral-embed",
        "input": "hello world",
        "user": "embed_user",
    });
    let out = EmbeddingProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(out["user"], "embed_user");
}

#[test]
fn embedding_request_drops_unknown_params() {
    let p = MistralProvider;
    let mut dep = deployment();
    dep.oxllm_params.model = "mistral/mistral-embed".to_string();
    let req = json!({
        "model": "mistral-embed",
        "input": "hello world",
        "temperature": 0.5,
        "unknown_embed_param": "x",
    });
    let out = EmbeddingProvider::transform_request(&p, &req, &dep).expect("ok");
    assert!(out.get("temperature").is_none());
    assert!(out.get("unknown_embed_param").is_none());
}

#[test]
fn strip_mistral_prefix_with_various_inputs() {
    assert_eq!(
        strip_mistral_prefix("mistral/mistral-large-latest"),
        "mistral-large-latest"
    );
    assert_eq!(
        strip_mistral_prefix("mistral/mistral-small-latest"),
        "mistral-small-latest"
    );
    assert_eq!(
        strip_mistral_prefix("mistral/codestral-latest"),
        "codestral-latest"
    );
    assert_eq!(
        strip_mistral_prefix("mistral-large-latest"),
        "mistral-large-latest"
    );
    assert_eq!(strip_mistral_prefix("openai/gpt-4o"), "openai/gpt-4o");
}

#[test]
fn supported_params_includes_reasoning_effort() {
    let p = MistralProvider;
    let params = ChatProvider::supported_params(&p).expect("should publish params");
    assert!(params.contains(&"reasoning_effort"));
}

// --- Swarm review fixes (H3 / H4 / M1 / M2) ---

#[test]
fn mistral_maps_tool_choice_required_to_any() {
    // H3: Mistral rejects `tool_choice: "required"`; it expects `"any"`.
    let p = MistralProvider;
    let req = json!({
        "model": "mistral/mistral-large-latest",
        "messages": [{"role": "user", "content": "Hi"}],
        "tools": [{"type": "function", "function": {"name": "f"}}],
        "tool_choice": "required",
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["tool_choice"], "any");
}

#[test]
fn mistral_leaves_tool_choice_other_values_untouched() {
    // H3 complement: only `"required"` is remapped; `"auto"` / `"none"` and
    // object forms pass through unchanged.
    let p = MistralProvider;
    for value in ["auto", "none"] {
        let req = json!({
            "model": "mistral/mistral-large-latest",
            "messages": [{"role": "user", "content": "Hi"}],
            "tool_choice": value,
        });
        let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
        assert_eq!(out["tool_choice"], value);
    }
    let req = json!({
        "model": "mistral/mistral-large-latest",
        "messages": [{"role": "user", "content": "Hi"}],
        "tool_choice": {"type": "function", "function": {"name": "f"}},
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["tool_choice"]["function"]["name"], "f");
}

#[test]
fn mistral_strips_reasoning_content_from_assistant_messages() {
    // H4: prior-turn reasoning fields on assistant messages cause Mistral 422.
    let p = MistralProvider;
    let req = json!({
        "model": "mistral/mistral-large-latest",
        "messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello", "reasoning_content": "secret thoughts"},
        ],
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert!(
        out["messages"][1].get("reasoning_content").is_none(),
        "reasoning_content must be stripped from assistant messages"
    );
    assert_eq!(out["messages"][1]["content"], "Hello");
    assert_eq!(out["messages"][1]["role"], "assistant");
}

#[test]
fn mistral_strips_thinking_blocks_from_assistant_messages() {
    // H4: same treatment for `thinking_blocks`.
    let p = MistralProvider;
    let req = json!({
        "model": "mistral/mistral-large-latest",
        "messages": [
            {"role": "assistant", "content": "Hello", "thinking_blocks": [{"type": "thinking", "thinking": "..."}]},
        ],
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert!(out["messages"][0].get("thinking_blocks").is_none());
}

#[test]
fn mistral_leaves_non_assistant_reasoning_fields_untouched() {
    // H4 scope: only `assistant` messages are walked. A `user`/`system`/`tool`
    // message carrying these keys (unusual, but not ours to rewrite) is left
    // alone so we never drop fields Mistral did not object to.
    let p = MistralProvider;
    let req = json!({
        "model": "mistral/mistral-large-latest",
        "messages": [
            {"role": "system", "reasoning_content": "keep-me"},
            {"role": "tool", "tool_call_id": "x", "thinking_blocks": [{"t": 1}]},
        ],
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["messages"][0]["reasoning_content"], "keep-me");
    assert!(out["messages"][1]["thinking_blocks"][0]["t"].is_i64());
}

#[test]
fn mistral_maps_seed_and_max_completion_tokens_together() {
    // M1 + M2 in one body.
    let p = MistralProvider;
    let req = json!({
        "model": "mistral/mistral-large-latest",
        "messages": [{"role": "user", "content": "Hi"}],
        "seed": 42,
        "max_completion_tokens": 128,
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert!(out.get("seed").is_none());
    assert_eq!(out["random_seed"], 42);
    assert!(out.get("max_completion_tokens").is_none());
    assert_eq!(out["max_tokens"], 128);
}

#[test]
fn mistral_explicit_max_tokens_is_overridden_by_max_completion_tokens() {
    // M2 precedence: when both are sent, `max_completion_tokens` is the
    // newer OpenAI field and wins for the mapped `max_tokens`.
    let p = MistralProvider;
    let req = json!({
        "model": "mistral/mistral-large-latest",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 10,
        "max_completion_tokens": 99,
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert!(out.get("max_completion_tokens").is_none());
    assert_eq!(out["max_tokens"], 99);
}

// --- Swarm review fixes (supported_params narrowing + random_seed direct-send) ---

#[test]
fn supported_params_excludes_openai_only_fields() {
    // Mistral rejects these OpenAI-platform-only fields with a 422. With
    // `drop_params = true` the host strips anything not in the published list,
    // so they must be absent here (otherwise they survive stripping → upstream
    // 422); with `drop_params = false` the host warns about them.
    let p = MistralProvider;
    let params = ChatProvider::supported_params(&p).expect("mistral publishes params");
    for excluded in ["store", "modalities", "audio", "service_tier", "metadata"] {
        assert!(
            !params.contains(&excluded),
            "Mistral must not advertise OpenAI-only / undocumented param '{excluded}'"
        );
    }
    // Mistral does accept `logit_bias` — it must remain.
    assert!(
        params.contains(&"logit_bias"),
        "Mistral must advertise accepted param 'logit_bias'"
    );
    // `random_seed` (Mistral's native name) is advertised alongside `seed`.
    assert!(params.contains(&"random_seed"));
    assert!(params.contains(&"seed"));
    // `reasoning_effort` and `thinking` are consumed internally for magistral
    // reasoning-prompt injection (never forwarded).
    assert!(params.contains(&"reasoning_effort"));
    assert!(params.contains(&"thinking"));
    // Core params the endpoint accepts must still be present.
    for required in ["model", "messages", "temperature", "max_tokens", "tools"] {
        assert!(
            params.contains(&required),
            "Mistral must advertise core param '{required}'"
        );
    }
}

#[test]
fn supported_params_is_shared_list_minus_excluded_plus_random_seed() {
    // The Mistral list must be exactly the shared list minus the OpenAI-only
    // (and undocumented-on-Mistral) fields, plus `random_seed`, `safe_prompt`
    // and `thinking`, so a future edit to either side cannot drift.
    let p = MistralProvider;
    let params = ChatProvider::supported_params(&p).expect("mistral publishes params");
    let mut expected: Vec<&str> = SUPPORTED_CHAT_PARAMS
        .iter()
        .copied()
        .filter(|p| !params_excluded_from_mistral(p))
        .chain(["random_seed", "safe_prompt", "thinking"])
        .collect();
    expected.sort_unstable();
    let mut got: Vec<&str> = params.clone();
    got.sort_unstable();
    assert_eq!(got, expected, "Mistral param list drifted from shared list");
}

fn params_excluded_from_mistral(p: &str) -> bool {
    matches!(
        p,
        "store" | "modalities" | "audio" | "service_tier" | "metadata"
    )
}

#[test]
fn drop_params_strips_openai_only_field_from_mistral_request() {
    // Models the production `drop_params = true` path: the host first strips
    // top-level keys not in the provider's published `supported_params`, THEN
    // hands the filtered body to `transform_request` (whose `build_chat_request`
    // only copies listed keys). `store` is OpenAI-only and must be absent from
    // the final upstream body even though the client sent it.
    let p = MistralProvider;
    let supported = ChatProvider::supported_params(&p).expect("mistral publishes params");
    let req = json!({
        "model": "mistral/mistral-large-latest",
        "messages": [{"role": "user", "content": "Hi"}],
        "store": true,
        "temperature": 0.5,
    });
    let mut stripped = serde_json::Map::new();
    for (k, v) in req.as_object().expect("obj") {
        if supported.contains(&k.as_str()) {
            stripped.insert(k.clone(), v.clone());
        }
    }
    let stripped_req = Value::Object(stripped);
    let out = ChatProvider::transform_request(&p, &stripped_req, &deployment()).expect("ok");
    assert!(
        out.get("store").is_none(),
        "store must not be forwarded to Mistral"
    );
    assert_eq!(out["temperature"], 0.5);
}

// --- random_seed direct-send (LOW) ---

#[test]
fn random_seed_direct_send_passes_through() {
    // A client sending Mistral's native `random_seed` (no `seed`) gets it
    // forwarded unchanged. The shared body builder drops `random_seed` (it is
    // not on the shared list), so the transform must recover it from the
    // original request.
    let p = MistralProvider;
    let req = json!({
        "model": "mistral/mistral-large-latest",
        "messages": [{"role": "user", "content": "Hi"}],
        "random_seed": 7,
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["random_seed"], 7);
    assert!(out.get("seed").is_none());
}

#[test]
fn random_seed_direct_send_wins_over_seed() {
    // When a client sends both `seed` and `random_seed`, the explicit native
    // `random_seed` wins and `seed` is not forwarded.
    let p = MistralProvider;
    let req = json!({
        "model": "mistral/mistral-large-latest",
        "messages": [{"role": "user", "content": "Hi"}],
        "seed": 1,
        "random_seed": 2,
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["random_seed"], 2);
    assert!(out.get("seed").is_none());
}

#[test]
fn random_seed_null_is_dropped() {
    // A null `random_seed` (and no `seed`) must not be forwarded.
    let p = MistralProvider;
    let req = json!({
        "model": "mistral/mistral-large-latest",
        "messages": [{"role": "user", "content": "Hi"}],
        "random_seed": null,
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert!(out.get("random_seed").is_none());
    assert!(out.get("seed").is_none());
}

// --- Phase 2 review fixes ---

fn magistral_deployment() -> Deployment {
    Deployment {
        model_name: "magistral-medium-latest".to_string(),
        oxllm_params: OxllmParams {
            model: "mistral/magistral-medium-latest".to_string(),
            api_key: Some("mistral-key".to_string()),
            ..Default::default()
        },
        ..Default::default()
    }
}

#[test]
fn request_strips_dollar_schema_from_tool_parameters() {
    // HIGH: `$id`/`$schema` (emitted by Pydantic / many OpenAI SDKs) cause a
    // Mistral 422. They must be stripped at every nesting level while the rest
    // of the schema is preserved.
    let p = MistralProvider;
    let req = json!({
        "model": "mistral/mistral-large-latest",
        "messages": [{"role": "user", "content": "Hi"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "$id": "https://example.com/weather.json",
                    "type": "object",
                    "properties": {
                        "city": {
                            "$schema": "http://json-schema.org/draft-07/schema#",
                            "$id": "nested",
                            "type": "string"
                        }
                    },
                    "required": ["city"]
                }
            }
        }],
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    let params = &out["tools"][0]["function"]["parameters"];
    assert!(
        params.get("$schema").is_none(),
        "top-level $schema must be stripped"
    );
    assert!(
        params.get("$id").is_none(),
        "top-level $id must be stripped"
    );
    assert_eq!(params["type"], "object");
    let city = &params["properties"]["city"];
    assert!(
        city.get("$schema").is_none(),
        "nested $schema must be stripped"
    );
    assert!(city.get("$id").is_none(), "nested $id must be stripped");
    assert_eq!(city["type"], "string");
    assert_eq!(params["required"][0], "city");
}

#[test]
fn request_strips_dollar_schema_from_array_items() {
    // The recursive walk must also descend into `items` arrays.
    let p = MistralProvider;
    let req = json!({
        "model": "mistral/mistral-large-latest",
        "messages": [{"role": "user", "content": "Hi"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "f",
                "parameters": {
                    "type": "array",
                    "items": {"$schema": "http://x", "type": "string"}
                }
            }
        }],
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    let items = &out["tools"][0]["function"]["parameters"]["items"];
    assert!(items.get("$schema").is_none());
    assert_eq!(items["type"], "string");
}

#[test]
fn response_normalizes_magistral_content_list() {
    // MEDIUM: magistral returns `content` as a list of content blocks; clients
    // must receive OpenAI-standard string `content` + `reasoning_content` +
    // `thinking_blocks`.
    let p = MistralProvider;
    let resp = json!({
        "id": "x",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": [
                        {"type": "text", "text": "let me think"},
                        {"type": "text", "text": " more"}
                    ]},
                    {"type": "text", "text": "The answer is 42."}
                ]
            },
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    });
    let out = ChatProvider::transform_response(&p, &resp, "magistral-medium-latest").expect("ok");
    let msg = &out["choices"][0]["message"];
    assert_eq!(msg["content"], "The answer is 42.");
    assert_eq!(msg["reasoning_content"], "let me think more");
    let blocks = msg["thinking_blocks"]
        .as_array()
        .expect("thinking_blocks present");
    assert_eq!(blocks.len(), 1);
    assert_eq!(blocks[0]["type"], "thinking");
    assert_eq!(blocks[0]["thinking"], "let me think more");
    assert_eq!(blocks[0]["signature"], "mistral");
}

#[test]
fn stream_chunk_normalizes_magistral_delta_content_list() {
    // MEDIUM: streaming deltas with list-form `delta.content` are normalized
    // the same way as the non-streaming path.
    let p = MistralProvider;
    let raw = r#"data: {"id":"x","choices":[{"index":0,"delta":{"content":[{"type":"thinking","thinking":[{"type":"text","text":"draft"}]},{"type":"text","text":"hi"}]}}]}"#;
    let out = p
        .transform_stream_chunk(raw)
        .expect("ok")
        .expect("should produce value");
    let delta = &out["choices"][0]["delta"];
    assert_eq!(delta["content"], "hi");
    assert_eq!(delta["reasoning_content"], "draft");
    assert_eq!(delta["thinking_blocks"][0]["type"], "thinking");
}

#[test]
fn response_leaves_string_content_untouched() {
    // Non-magistral responses (string content) pass through unchanged.
    let p = MistralProvider;
    let resp = json!({
        "id": "x",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Hi"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    });
    let out = ChatProvider::transform_response(&p, &resp, "mistral-large-latest").expect("ok");
    assert_eq!(out["choices"][0]["message"]["content"], "Hi");
    assert!(
        out["choices"][0]["message"]
            .get("reasoning_content")
            .is_none()
    );
    assert!(
        out["choices"][0]["message"]
            .get("thinking_blocks")
            .is_none()
    );
}

#[test]
fn response_content_list_without_text_yields_null_content() {
    // A content list with only thinking blocks yields null content + reasoning.
    let p = MistralProvider;
    let resp = json!({
        "id": "x",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": [{"type": "thinking", "thinking": [{"type": "text", "text": "pondering"}]}]
            },
            "finish_reason": "stop"
        }],
    });
    let out = ChatProvider::transform_response(&p, &resp, "magistral-medium-latest").expect("ok");
    let msg = &out["choices"][0]["message"];
    assert!(msg["content"].is_null(), "no text block → null content");
    assert_eq!(msg["reasoning_content"], "pondering");
}

#[test]
fn request_strips_name_from_user_message() {
    // MEDIUM: Mistral only accepts `name` on tool messages.
    let p = MistralProvider;
    let req = json!({
        "model": "mistral/mistral-large-latest",
        "messages": [{"role": "user", "name": "alice", "content": "hi"}],
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert!(out["messages"][0].get("name").is_none());
    assert_eq!(out["messages"][0]["content"], "hi");
}

#[test]
fn request_keeps_name_in_tool_message() {
    // MEDIUM: a non-empty `name` on a tool message is preserved.
    let p = MistralProvider;
    let req = json!({
        "model": "mistral/mistral-large-latest",
        "messages": [
            {"role": "user", "content": "what?"},
            {"role": "tool", "tool_call_id": "t1", "name": "get_weather", "content": "{}"},
        ],
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["messages"][1]["name"], "get_weather");
}

#[test]
fn request_strips_empty_name_from_tool_message() {
    // MEDIUM: an empty/whitespace `name` on a tool message is dropped.
    let p = MistralProvider;
    let req = json!({
        "model": "mistral/mistral-large-latest",
        "messages": [
            {"role": "user", "content": "what?"},
            {"role": "tool", "tool_call_id": "t1", "name": "   ", "content": "{}"},
        ],
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert!(out["messages"][1].get("name").is_none());
    assert_eq!(out["messages"][1]["tool_call_id"], "t1");
}

#[test]
fn magistral_reasoning_effort_injects_system_prompt() {
    // MEDIUM: for magistral models, reasoning_effort triggers injection of
    // Mistral's recommended reasoning system prompt and is itself not
    // forwarded.
    let p = MistralProvider;
    let req = json!({
        "model": "mistral/magistral-medium-latest",
        "messages": [{"role": "user", "content": "Solve this."}],
        "reasoning_effort": "medium",
    });
    let out = ChatProvider::transform_request(&p, &req, &magistral_deployment()).expect("ok");
    assert!(
        out.get("reasoning_effort").is_none(),
        "reasoning_effort must not be forwarded"
    );
    let messages = out["messages"].as_array().expect("messages present");
    assert_eq!(messages[0]["role"], "system");
    let sys = messages[0]["content"]
        .as_str()
        .expect("system content string");
    assert!(sys.contains("[SYSTEM_PROMPT]"));
    assert!(sys.contains("<think>"));
    assert!(
        sys.contains("Solve this.")
            || messages.iter().any(|m| m["content"]
                .as_str()
                .is_some_and(|c| c.contains("Solve this.")))
    );
}

#[test]
fn magistral_reasoning_effort_prepends_to_existing_system_message() {
    // If a system message already exists, the reasoning prompt is prepended to
    // its content rather than inserting a new message.
    let p = MistralProvider;
    let req = json!({
        "model": "mistral/magistral-medium-latest",
        "messages": [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hi"},
        ],
        "reasoning_effort": "high",
    });
    let out = ChatProvider::transform_request(&p, &req, &magistral_deployment()).expect("ok");
    let messages = out["messages"].as_array().expect("messages present");
    assert_eq!(messages.len(), 2, "no new message should be inserted");
    assert_eq!(messages[0]["role"], "system");
    let content = messages[0]["content"].as_str().expect("string content");
    assert!(content.starts_with("<s>[SYSTEM_PROMPT]"));
    assert!(content.contains("Be concise."));
}

#[test]
fn magistral_null_reasoning_effort_does_not_inject_prompt() {
    // A null reasoning_effort is dropped but does not trigger injection.
    let p = MistralProvider;
    let req = json!({
        "model": "mistral/magistral-medium-latest",
        "messages": [{"role": "user", "content": "Hi"}],
        "reasoning_effort": null,
    });
    let out = ChatProvider::transform_request(&p, &req, &magistral_deployment()).expect("ok");
    assert!(out.get("reasoning_effort").is_none());
    let messages = out["messages"].as_array().expect("messages present");
    assert!(
        !messages.iter().any(|m| m["role"] == "system"),
        "no system prompt should be injected for null reasoning_effort"
    );
}

#[test]
fn request_preserves_safe_prompt() {
    // LOW: `safe_prompt` is a documented Mistral-native param and must be
    // forwarded when sent by the client.
    let p = MistralProvider;
    let req = json!({
        "model": "mistral/mistral-large-latest",
        "messages": [{"role": "user", "content": "Hi"}],
        "safe_prompt": true,
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["safe_prompt"], true);
}

#[test]
fn supported_params_includes_safe_prompt() {
    let p = MistralProvider;
    let params = ChatProvider::supported_params(&p).expect("should publish params");
    assert!(params.contains(&"safe_prompt"));
}

#[test]
fn request_drops_empty_assistant_message() {
    // LOW: Mistral rejects an assistant message with `content: ""` and no
    // tool_calls; such messages are dropped.
    let p = MistralProvider;
    let req = json!({
        "model": "mistral/mistral-large-latest",
        "messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "again"},
        ],
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    let messages = out["messages"].as_array().expect("messages present");
    assert_eq!(messages.len(), 2, "empty assistant message must be dropped");
    assert_eq!(messages[0]["content"], "Hi");
    assert_eq!(messages[1]["content"], "again");
}

#[test]
fn request_keeps_empty_assistant_with_tool_calls() {
    // LOW: an assistant message with empty content but with tool_calls is kept.
    let p = MistralProvider;
    let req = json!({
        "model": "mistral/mistral-large-latest",
        "messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "f"}}]},
        ],
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    let messages = out["messages"].as_array().expect("messages present");
    assert_eq!(messages.len(), 2, "assistant with tool_calls must be kept");
    assert_eq!(messages[1]["tool_calls"][0]["id"], "t1");
}

#[test]
fn mistral_maps_unknown_tool_choice_string_to_any() {
    // LOW: any string that is not `auto`/`none` is mapped to `any` (LiteLLM
    // catch-all), not just the exact value `required`.
    let p = MistralProvider;
    let req = json!({
        "model": "mistral/mistral-large-latest",
        "messages": [{"role": "user", "content": "Hi"}],
        "tool_choice": "something",
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["tool_choice"], "any");
}

// --- Phase 2 review fixes (round 2) ---

#[test]
fn magistral_reasoning_effort_preserves_list_content_system_message() {
    // HIGH: a system message whose `content` is the multimodal array form must
    // not be overwritten by the reasoning prompt. Both the prompt and the
    // original text must reach upstream.
    let p = MistralProvider;
    let req = json!({
        "model": "mistral/magistral-medium-latest",
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "Be nice."}]},
            {"role": "user", "content": "Solve this."},
        ],
        "reasoning_effort": "high",
    });
    let out = ChatProvider::transform_request(&p, &req, &magistral_deployment()).expect("ok");
    assert!(
        out.get("reasoning_effort").is_none(),
        "reasoning_effort must not be forwarded"
    );
    let messages = out["messages"].as_array().expect("messages present");
    assert_eq!(messages[0]["role"], "system");
    let content = messages[0]["content"]
        .as_array()
        .expect("array content shape must be preserved");
    // The first block carries the reasoning prompt.
    let first = &content[0];
    assert_eq!(first["type"], "text");
    let prompt_text = first["text"].as_str().expect("prompt text");
    assert!(prompt_text.contains("[SYSTEM_PROMPT]"));
    assert!(prompt_text.ends_with("\n\n"));
    // The original system text is preserved as the second block.
    assert_eq!(content[1]["type"], "text");
    assert_eq!(content[1]["text"], "Be nice.");
    assert_eq!(content.len(), 2);
}

#[test]
fn magistral_thinking_param_injects_system_prompt() {
    // MEDIUM: `thinking` is an alternative trigger for the magistral reasoning
    // system prompt (parity with LiteLLM, which treats both `reasoning_effort`
    // and `thinking` as triggers).
    let p = MistralProvider;
    let req = json!({
        "model": "mistral/magistral-medium-latest",
        "messages": [{"role": "user", "content": "Solve this."}],
        "thinking": true,
    });
    let out = ChatProvider::transform_request(&p, &req, &magistral_deployment()).expect("ok");
    assert!(
        out.get("thinking").is_none(),
        "thinking must not be forwarded"
    );
    let messages = out["messages"].as_array().expect("messages present");
    assert_eq!(messages[0]["role"], "system");
    let sys = messages[0]["content"]
        .as_str()
        .expect("system content string");
    assert!(sys.contains("[SYSTEM_PROMPT]"));
    assert!(sys.contains("<think>"));
}

#[test]
fn magistral_thinking_param_not_forwarded() {
    // MEDIUM complement: even when injection is a no-op (null thinking), the
    // field is stripped from the upstream body.
    let p = MistralProvider;
    let req = json!({
        "model": "mistral/magistral-medium-latest",
        "messages": [{"role": "user", "content": "Hi"}],
        "thinking": null,
    });
    let out = ChatProvider::transform_request(&p, &req, &magistral_deployment()).expect("ok");
    assert!(out.get("thinking").is_none());
    let messages = out["messages"].as_array().expect("messages present");
    assert!(
        !messages.iter().any(|m| m["role"] == "system"),
        "no system prompt should be injected for null thinking"
    );
}

#[test]
fn magistral_thinking_and_reasoning_effort_both_trigger_injection() {
    // MEDIUM: either field triggers injection; sending both does not double-
    // inject.
    let p = MistralProvider;
    let req = json!({
        "model": "mistral/magistral-medium-latest",
        "messages": [{"role": "user", "content": "Hi"}],
        "reasoning_effort": "low",
        "thinking": true,
    });
    let out = ChatProvider::transform_request(&p, &req, &magistral_deployment()).expect("ok");
    assert!(out.get("reasoning_effort").is_none());
    assert!(out.get("thinking").is_none());
    let messages = out["messages"].as_array().expect("messages present");
    let system_count = messages.iter().filter(|m| m["role"] == "system").count();
    assert_eq!(system_count, 1, "exactly one system message expected");
}

#[test]
fn non_magistral_drops_thinking_param() {
    // MEDIUM: `thinking` on a non-magistral model is consumed (dropped) and
    // never forwarded, with no prompt injection.
    let p = MistralProvider;
    let req = json!({
        "model": "mistral/mistral-large-latest",
        "messages": [{"role": "user", "content": "Hi"}],
        "thinking": true,
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert!(
        out.get("thinking").is_none(),
        "thinking must not be forwarded to Mistral"
    );
    let messages = out["messages"].as_array().expect("messages present");
    assert!(
        !messages.iter().any(|m| m["role"] == "system"),
        "no system prompt should be injected for non-magistral"
    );
}

#[test]
fn response_empty_string_content_becomes_null() {
    // LOW: Mistral sometimes returns `content: ""` instead of null; downstream
    // consumers that test `content == null` to mean "no text" need the
    // canonical null form.
    let p = MistralProvider;
    let resp = json!({
        "id": "x",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": ""},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1},
    });
    let out = ChatProvider::transform_response(&p, &resp, "mistral-large-latest").expect("ok");
    let content = &out["choices"][0]["message"]["content"];
    assert!(
        content.is_null(),
        "empty-string content must normalize to null"
    );
}

#[test]
fn response_nonempty_string_content_is_preserved() {
    // LOW complement: non-empty content is not normalized away.
    let p = MistralProvider;
    let resp = json!({
        "id": "x",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "hello"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    });
    let out = ChatProvider::transform_response(&p, &resp, "mistral-large-latest").expect("ok");
    assert_eq!(out["choices"][0]["message"]["content"], "hello");
}

#[test]
fn request_strips_extra_fields_from_tool_calls() {
    // LOW: Mistral only accepts `{id, type:"function", function}` per
    // tool_call entry; extra keys (e.g. OpenAI's `index`, `logprobs`) trigger
    // `extra_forbidden` 422s. They must be stripped on every message.
    let p = MistralProvider;
    let req = json!({
        "model": "mistral/mistral-large-latest",
        "messages": [
            {"role": "user", "content": "what?"},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"},
                "index": 0,
                "logprobs": {"content": [{"token": "x"}]}
            }]},
        ],
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    let messages = out["messages"].as_array().expect("messages present");
    assert_eq!(messages.len(), 2, "assistant with tool_calls must be kept");
    let tool_call = &messages[1]["tool_calls"][0];
    assert_eq!(tool_call["id"], "call_1");
    assert_eq!(tool_call["type"], "function");
    assert_eq!(tool_call["function"]["name"], "get_weather");
    assert!(
        tool_call.get("index").is_none(),
        "extra `index` must be stripped"
    );
    assert!(
        tool_call.get("logprobs").is_none(),
        "extra `logprobs` must be stripped"
    );
}

#[test]
fn request_keeps_assistant_with_function_call_and_empty_content() {
    // LOW: a legacy assistant message carrying `function_call` with empty
    // content is NOT dropped (parity with LiteLLM's typed
    // `_is_empty_assistant_message` which keeps messages where any typed
    // field is non-null).
    let p = MistralProvider;
    let req = json!({
        "model": "mistral/mistral-large-latest",
        "messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "", "function_call": {"name": "f", "arguments": "{}"}},
        ],
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    let messages = out["messages"].as_array().expect("messages present");
    assert_eq!(
        messages.len(),
        2,
        "assistant with function_call must be kept"
    );
    assert_eq!(messages[1]["role"], "assistant");
    assert_eq!(messages[1]["function_call"]["name"], "f");
}
