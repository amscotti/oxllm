use super::*;
use crate::openai_compat::SUPPORTED_CHAT_PARAMS;
use oxllm_core::Endpoint;
use oxllm_core::deployment::{Deployment, OxllmParams};
use oxllm_core::provider::{ChatProvider, Provider, ResolvedCredentials};
use serde_json::json;

#[test]
fn proven_additive_reasoning_usage_is_repaired_for_json_and_sse() {
    let body = json!({"choices":[], "usage":{"prompt_tokens":5, "completion_tokens":1, "total_tokens":11,
        "completion_tokens_details":{"reasoning_tokens":5}}});
    let normalized = GeminiProvider
        .transform_response(&body, "gemini/gemini-3-flash")
        .unwrap();
    assert_eq!(normalized["usage"]["completion_tokens"], 6);
    assert_eq!(normalized["usage"]["total_tokens"], 11);
    assert_eq!(
        normalized["usage"]["completion_tokens_details"]["reasoning_tokens"],
        5
    );
    assert_eq!(
        GeminiProvider
            .transform_response(&normalized, "gemini/gemini-3-flash")
            .unwrap(),
        normalized
    );
    let streamed = GeminiProvider
        .transform_stream_chunk(&format!("data: {body}\n\n"))
        .unwrap()
        .unwrap();
    assert_eq!(streamed["usage"], normalized["usage"]);
    let other = crate::openai_compat::OpenAiCompatProvider::new(
        "other",
        "https://example.test",
        "other",
        true,
        true,
        true,
    );
    assert_eq!(
        other.transform_response(&body, "other/model").unwrap()["usage"],
        body["usage"],
        "Google-specific repairs must not change other compatible providers"
    );
}

#[test]
fn nonadditive_or_unknown_gemini_usage_is_not_reinterpreted() {
    for usage in [
        json!({"prompt_tokens":5,"completion_tokens":6,"total_tokens":11,"completion_tokens_details":{"reasoning_tokens":5}}),
        json!({"prompt_tokens":21,"total_tokens":21,"completion_tokens_details":{"reasoning_tokens":5}}),
        json!({"prompt_tokens":21,"total_tokens":21,"completion_tokens_details":{"reasoning_tokens":"unknown"}}),
        json!({"prompt_tokens":u64::MAX,"completion_tokens":1,"total_tokens":u64::MAX,"completion_tokens_details":{"reasoning_tokens":1}}),
    ] {
        let body = json!({"usage":usage});
        let normalized = GeminiProvider
            .transform_response(&body, "gemini/gemini-3-flash")
            .unwrap();
        assert_eq!(normalized["usage"], usage);
        let streamed = GeminiProvider
            .transform_stream_chunk(&format!("data: {body}\n\n"))
            .unwrap()
            .unwrap();
        assert_eq!(streamed["usage"], usage);
    }
}

#[test]
fn validation_logprobs_forwarding_is_preserved_pending_model_specific_evidence() {
    // Decision (see `GEMINI_SUPPORTED_CHAT_PARAMS`): a stale forum report is
    // not
    // model/account-level evidence, so `logprobs`/`top_logprobs` are
    // forwarded to Gemini's OpenAI-compat surface rather than translated or
    // stripped. This pins the whole forwarding contract end to end, which the
    // previous assertion of the transform's output alone did not:
    //   1. Gemini must *advertise* both params, because the host's
    //      `drop_params = true` path strips keys outside `supported_params`
    //      before `transform_request` ever runs;
    //   2. `transform_request` must then carry them into the upstream body.
    // `logit_bias` is the sentinel proving the strip/rebuild path is actually
    // exercised here — it must not survive either step.
    let p = GeminiProvider;
    let supported =
        ChatProvider::supported_params(&GeminiProvider).expect("gemini publishes params");
    for forwarded in ["logprobs", "top_logprobs"] {
        assert!(
            supported.contains(&forwarded),
            "Gemini must advertise '{forwarded}' so drop_params keeps it"
        );
    }
    let request = json!({
        "model": "alias",
        "messages": [{"role": "user", "content": "hello"}],
        "logprobs": true,
        "top_logprobs": 3,
        "logit_bias": {"1": 100},
    });
    // Host-side `drop_params = true`: keep only advertised keys.
    let mut stripped = serde_json::Map::new();
    for (k, v) in request.as_object().expect("obj") {
        if supported.contains(&k.as_str()) {
            stripped.insert(k.clone(), v.clone());
        }
    }
    let body = p
        .transform_request(&Value::Object(stripped), &deployment())
        .unwrap();
    assert_eq!(
        body["logprobs"], true,
        "logprobs must survive drop_params + transform_request"
    );
    assert_eq!(
        body["top_logprobs"], 3,
        "top_logprobs must survive drop_params + transform_request"
    );
    assert!(
        body.get("logit_bias").is_none(),
        "sentinel: unsupported logit_bias must be stripped, proving the filter runs"
    );
}

fn deployment() -> Deployment {
    Deployment {
        model_name: "gemini-pro".to_string(),
        oxllm_params: OxllmParams {
            model: "gemini/gemini-pro".to_string(),
            api_key: Some("gemini-key".to_string()),
            ..Default::default()
        },
        ..Default::default()
    }
}

#[test]
fn flagged_gemini_models_reject_or_drop_sampling_params() {
    for model in [
        "gemini-3.5-flash-lite",
        "gemini-3.6-flash",
        "gemini-3.7-flash",
        "gemini-3.8-flash",
        "gemini-3.8-flash-cyber",
    ] {
        let mut dep = deployment();
        dep.oxllm_params.model = format!("gemini/{model}");
        let request = json!({
            "messages": [{"role": "user", "content": "Hi"}],
            "temperature": 0.2,
            "top_p": 0.5,
            "top_k": 10
        });
        assert!(
            GeminiProvider.transform_request(&request, &dep).is_err(),
            "{model}"
        );
        dep.oxllm_params.drop_params = Some(true);
        let body = GeminiProvider.transform_request(&request, &dep).unwrap();
        for param in ["temperature", "top_p", "top_k"] {
            assert!(body.get(param).is_none(), "{model}: {param}");
        }
    }
    let mut older = deployment();
    older.oxllm_params.model = "gemini/gemini-2.5-flash".into();
    let body = GeminiProvider
        .transform_request(
            &json!({"messages": [], "temperature": 0.2, "top_p": 0.5}),
            &older,
        )
        .unwrap();
    assert_eq!(body["temperature"], 0.2);
    assert_eq!(body["top_p"], 0.5);
}

#[test]
fn endpoint_url_uses_default_gemini_base() {
    let p = GeminiProvider;
    assert_eq!(
        p.endpoint_url(&deployment()),
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    );
}

#[test]
fn endpoint_url_respects_custom_base() {
    let p = GeminiProvider;
    let mut dep = deployment();
    dep.oxllm_params.api_base = Some("https://custom.example.com/v1/".to_string());
    assert_eq!(
        p.endpoint_url(&dep),
        "https://custom.example.com/v1/chat/completions"
    );
}

#[test]
fn apply_auth_uses_bearer_token() {
    let p = GeminiProvider;
    let creds = ResolvedCredentials::from_key("gemini-secret");
    let mut headers = std::collections::HashMap::new();
    p.apply_auth(&deployment(), &creds, &mut headers);
    assert_eq!(
        headers.get("authorization").map(String::as_str),
        Some("Bearer gemini-secret")
    );
}

#[test]
fn request_strips_gemini_prefix_from_model() {
    let p = GeminiProvider;
    let req = json!({
    "model": "gemini-pro",
    "messages": [{"role": "user", "content": "Hi"}],
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["model"], "gemini-pro");
}

#[test]
fn request_deployment_model_overrides_request_model() {
    // The stripped deployment model always wins for non-wildcard deployments,
    // even when the client body carries a different `model`. The golden
    // parity test uses a deployment whose model matches the request body, so
    // it cannot detect a regression where `resolve_model` started forwarding
    // the client's model instead. This test pins the divergence case.
    let p = GeminiProvider;
    let mut dep = deployment();
    dep.oxllm_params.model = "gemini/gemini-2.5-flash".to_string();
    let req = json!({
    "model": "wrong",
    "messages": [{"role": "user", "content": "Hi"}],
    });
    let out = p.transform_request(&req, &dep).expect("ok");
    assert_eq!(out["model"], "gemini-2.5-flash");
}

#[test]
fn request_passes_known_params_and_drops_unknown() {
    let p = GeminiProvider;
    let req = json!({
    "model": "gemini-pro",
    "messages": [{"role": "user", "content": "Hi"}],
    "temperature": 0.5,
    "max_tokens": 100,
    "stream": false,
    "n": 2,
    "logprobs": true,
    "some_unknown_param": "ignored",
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["temperature"], 0.5);
    assert_eq!(out["max_tokens"], 100);
    assert_eq!(out["n"], 2);
    assert_eq!(out["messages"][0]["content"], "Hi");
    assert_eq!(out["logprobs"], true);
    assert!(out.get("some_unknown_param").is_none());
}

#[test]
fn request_drops_null_params() {
    let p = GeminiProvider;
    let req = json!({
    "model": "gemini-pro",
    "messages": [{"role": "user", "content": "Hi"}],
    "temperature": null,
    "max_tokens": 5,
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert!(out.get("temperature").is_none());
    assert_eq!(out["max_tokens"], 5);
}

#[test]
fn response_is_identity() {
    let p = GeminiProvider;
    let resp = json!({
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
    });
    let out = p.transform_response(&resp, "gemini-pro").expect("ok");
    assert_eq!(out, resp);
}

#[test]
fn stream_chunk_parses_openai_format() {
    let p = GeminiProvider;
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
    let p = GeminiProvider;
    assert!(
        p.transform_stream_chunk("data: [DONE]")
            .expect("ok")
            .is_none()
    );
}

#[test]
fn stream_empty_chunk_returns_none() {
    let p = GeminiProvider;
    assert!(p.transform_stream_chunk("").expect("ok").is_none());
}

#[test]
fn stream_invalid_json_errors() {
    let p = GeminiProvider;
    assert!(p.transform_stream_chunk("data: {bad json").is_err());
}

#[test]
fn provider_id_and_supports() {
    let p = GeminiProvider;
    assert_eq!(p.id(), "gemini");
    assert!(p.supports(Endpoint::ChatCompletions));
    assert!(p.supports(Endpoint::Embeddings));
}

#[test]
fn strip_gemini_prefix_handles_unknown() {
    assert_eq!(strip_gemini_prefix("gemini/gemini-pro"), "gemini-pro");
    assert_eq!(strip_gemini_prefix("gemini-pro"), "gemini-pro");
}

const GOLDEN_REQUEST_001: &str = include_str!("goldens/chat_request_001.json");
const GOLDEN_REQUEST_002: &str = include_str!("goldens/chat_request_002.json");
const GOLDEN_RESPONSE_001: &str = include_str!("goldens/chat_response_001.json");

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
fn request_preserves_reasoning_effort() {
    let p = GeminiProvider;
    let req = json!({
    "model": "gemini-pro",
    "messages": [{"role": "user", "content": "Hi"}],
    "reasoning_effort": "low",
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["reasoning_effort"], "low");
}

#[test]
fn gemini_37_and_38_flash_minimal_reasoning_effort_becomes_low() {
    let p = GeminiProvider;
    for model in [
        "gemini/gemini-3.7-flash",
        "gemini-3.8-flash",
        "gemini/gemini-3.8-flash-preview",
    ] {
        let mut dep = deployment();
        dep.oxllm_params.model = model.to_string();
        let req = json!({
            "model": model,
            "messages": [{"role": "user", "content": "Hi"}],
            "reasoning_effort": "minimal",
        });
        let out = p.transform_request(&req, &dep).expect("ok");
        assert_eq!(out["reasoning_effort"], "low", "{model}");
    }
    let mut dep = deployment();
    dep.oxllm_params.model = "gemini/gemini-3.6-flash".to_string();
    let req = json!({
        "model": "gemini-3.6-flash",
        "messages": [{"role": "user", "content": "Hi"}],
        "reasoning_effort": "minimal",
    });
    let out = p.transform_request(&req, &dep).expect("ok");
    assert_eq!(out["reasoning_effort"], "minimal");
}

#[test]
fn request_preserves_response_format() {
    let p = GeminiProvider;
    let req = json!({
    "model": "gemini-pro",
    "messages": [{"role": "user", "content": "Hi"}],
    "response_format": {"type": "json_object"},
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["response_format"]["type"], "json_object");
}

#[test]
fn request_preserves_tools() {
    let p = GeminiProvider;
    let req = json!({
    "model": "gemini-pro",
    "messages": [{"role": "user", "content": "Hi"}],
    "tools": [{
    "type": "function",
    "function": {
    "name": "get_weather",
    "description": "Get the weather",
    "parameters": {"type": "object", "properties": {"loc": {"type": "string"}}},
    },
    }],
    "tool_choice": "auto",
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    let tools = out["tools"].as_array().expect("tools");
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0]["function"]["name"], "get_weather");
    assert_eq!(out["tool_choice"], "auto");
}

#[test]
fn request_preserves_stream_options() {
    let p = GeminiProvider;
    let req = json!({
    "model": "gemini-pro",
    "messages": [{"role": "user", "content": "Hi"}],
    "stream": true,
    "stream_options": {"include_usage": true},
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["stream"], true);
    assert_eq!(out["stream_options"]["include_usage"], true);
}

#[test]
fn request_preserves_service_tier() {
    // Gemini's OpenAI-compat surface accepts `service_tier` ("priority" /
    // "flex"), so it must survive `transform_request` (which rebuilds the body
    // from the Gemini allow-list). A silent drop would route traffic to the
    // default tier instead of the requested one.
    let p = GeminiProvider;
    let req = json!({
    "model": "gemini-pro",
    "messages": [{"role": "user", "content": "Hi"}],
    "service_tier": "flex",
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["service_tier"], "flex");
}

#[test]
fn request_preserves_modalities_and_audio() {
    // `modalities` and (for TTS/audio models) `audio` are Gemini-supported per
    // the reference supported-param list, so they must survive transform.
    let p = GeminiProvider;
    let req = json!({
    "model": "gemini-2.5-flash-preview-tts",
    "messages": [{"role": "user", "content": "Hi"}],
    "modalities": ["text", "audio"],
    "audio": {"voice": "alloy", "format": "wav"},
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert_eq!(out["modalities"][0], "text");
    assert_eq!(out["audio"]["voice"], "alloy");
}

#[test]
fn response_preserves_thinking_content() {
    let p = GeminiProvider;
    let resp = json!({
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "choices": [{
    "index": 0,
    "message": {
    "role": "assistant",
    "content": "Hello",
    "reasoning_content": "I am thinking about this..."
    },
    "finish_reason": "stop",
    }],
    });
    let out = p.transform_response(&resp, "gemini-pro").expect("ok");
    assert_eq!(out["choices"][0]["message"]["content"], "Hello");
    assert_eq!(
        out["choices"][0]["message"]["reasoning_content"],
        "I am thinking about this..."
    );
}

#[test]
fn response_preserves_finish_reason() {
    let p = GeminiProvider;
    let resp = json!({
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "Hello"},
    "finish_reason": "length",
    }],
    "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
    });
    let out = p.transform_response(&resp, "gemini-pro").expect("ok");
    assert_eq!(out["choices"][0]["finish_reason"], "length");
    assert_eq!(out["usage"]["total_tokens"], 15);
}

#[test]
fn response_with_tool_calls_passthrough() {
    let p = GeminiProvider;
    let resp = json!({
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "choices": [{
    "index": 0,
    "message": {
    "role": "assistant",
    "content": null,
    "tool_calls": [{
    "id": "call_1",
    "type": "function",
    "function": {"name": "get_weather", "arguments": "{\"loc\":\"Paris\"}"},
    }],
    },
    "finish_reason": "tool_calls",
    }],
    });
    let out = p.transform_response(&resp, "gemini-pro").expect("ok");
    let calls = out["choices"][0]["message"]["tool_calls"]
        .as_array()
        .expect("tool_calls");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0]["function"]["name"], "get_weather");
    assert_eq!(out["choices"][0]["finish_reason"], "tool_calls");
}

#[test]
fn stream_chunk_with_content_and_finish_reason() {
    let p = GeminiProvider;
    let raw = "data: {\"id\":\"x\",\"choices\":[{\"delta\":{\"content\":\"world\"},\"finish_reason\":\"stop\"}]}";
    let out = p
        .transform_stream_chunk(raw)
        .expect("ok")
        .expect("should produce value");
    assert_eq!(out["choices"][0]["delta"]["content"], "world");
    assert_eq!(out["choices"][0]["finish_reason"], "stop");
}

#[test]
fn embedding_request_replaces_client_model_with_deployment_model() {
    // The OpenAI-compat embedding transform takes the model from the
    // *deployment* (`gemini/gemini-pro` → stripped `gemini-pro`), never from
    // the client body. A single chat-model deployment therefore cannot serve
    // embeddings; a dedicated `gemini/gemini-embedding-001` deployment is
    // required (see `embedding_deployment_strips_prefix`).
    let p = GeminiProvider;
    let req = json!({
    "model": "gemini-embedding-001",
    "input": "Hello world",
    });
    let out = oxllm_core::provider::EmbeddingProvider::transform_request(&p, &req, &deployment())
        .expect("ok");
    assert_eq!(out["model"], "gemini-pro");
    assert_eq!(out["input"], "Hello world");
}

#[test]
fn embedding_deployment_strips_prefix() {
    // An embedding deployment configured with `gemini/gemini-embedding-001`
    // must strip the `gemini/` prefix so the upstream request carries the bare
    // `gemini-embedding-001`.
    let p = GeminiProvider;
    let mut dep = deployment();
    dep.oxllm_params.model = "gemini/gemini-embedding-001".to_string();
    let req = json!({
    "model": "gemini-embedding-001",
    "input": "embed me",
    });
    let out =
        oxllm_core::provider::EmbeddingProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(out["model"], "gemini-embedding-001");
    assert_eq!(out["input"], "embed me");
}

#[test]
fn supported_params_excludes_openai_only_fields() {
    // The Gemini OpenAI-compat surface rejects these OpenAI-only fields.
    // With `drop_params = true` the host strips anything not in this list, so
    // they must be absent here (otherwise they survive stripping → upstream 400).
    let params = ChatProvider::supported_params(&GeminiProvider).expect("gemini publishes params");
    for excluded in ["logit_bias", "store", "metadata"] {
        assert!(
            !params.contains(&excluded),
            "Gemini must not advertise OpenAI-only param '{excluded}'"
        );
    }
    // `service_tier`, `modalities`, and `audio` (on TTS/audio models) ARE
    // accepted by Gemini's OpenAI-compat surface (the reference lists them in
    // `get_supported_openai_params`), so they must survive here.
    for included in ["service_tier", "modalities", "audio"] {
        assert!(
            params.contains(&included),
            "Gemini must advertise supported param '{included}'"
        );
    }
    // The Gemini list must be exactly the shared list minus the rejected
    // fields, so a future edit to either side cannot silently drift.
    let mut expected: Vec<&str> = SUPPORTED_CHAT_PARAMS
        .iter()
        .copied()
        .filter(|p| !params_excluded_from_gemini(p))
        .collect();
    expected.sort_unstable();
    let mut got: Vec<&str> = params.clone();
    got.sort_unstable();
    assert_eq!(got, expected, "Gemini param list drifted from shared list");
    // Core params the endpoint does accept must still be present.
    for required in ["model", "messages", "temperature", "max_tokens", "tools"] {
        assert!(
            params.contains(&required),
            "Gemini must advertise core param '{required}'"
        );
    }
}

fn params_excluded_from_gemini(p: &str) -> bool {
    matches!(p, "logit_bias" | "store" | "metadata")
}

#[test]
fn drop_params_strips_openai_only_field_from_gemini_request() {
    // Models the production `drop_params = true` path: the host first strips
    // top-level keys not in the provider's published `supported_params`, THEN
    // hands the filtered body to `transform_request` (whose `build_chat_request`
    // only copies listed keys). `logit_bias` is OpenAI-only, so it must be
    // absent from the final upstream body even though the client sent it.
    let p = GeminiProvider;
    let supported =
        ChatProvider::supported_params(&GeminiProvider).expect("gemini publishes params");
    let req = json!({
    "model": "gemini-pro",
    "messages": [{"role": "user", "content": "Hi"}],
    "logit_bias": {"1": 100},
    "temperature": 0.5,
    });
    // Host-side strip: keep only advertised keys.
    let mut stripped = serde_json::Map::new();
    for (k, v) in req.as_object().expect("obj") {
        if supported.contains(&k.as_str()) {
            stripped.insert(k.clone(), v.clone());
        }
    }
    let stripped_req = Value::Object(stripped);
    let out = p
        .transform_request(&stripped_req, &deployment())
        .expect("ok");
    assert!(
        out.get("logit_bias").is_none(),
        "logit_bias must not be forwarded to Gemini"
    );
    assert_eq!(out["temperature"], 0.5);
}

#[test]
fn transform_request_strips_openai_only_fields_via_supported_params_override() {
    // Gemini's `with_supported_params(GEMINI_SUPPORTED_CHAT_PARAMS)` is now
    // honored by `build_chat_request_with_params`, so OpenAI-only fields are
    // dropped at transform time even when host `drop_params` is false. This
    // prevents preventable Gemini 400s from `logit_bias` / `store` / etc.
    let p = GeminiProvider;
    let req = json!({
    "model": "gemini-pro",
    "messages": [{"role": "user", "content": "Hi"}],
    "logit_bias": {"1": 100},
    "temperature": 0.5,
    });
    let out = p.transform_request(&req, &deployment()).expect("ok");
    assert!(
        out.get("logit_bias").is_none(),
        "Gemini transform must drop OpenAI-only fields, got {out}"
    );
    assert_eq!(out["temperature"], 0.5);
    assert_eq!(out["model"], "gemini-pro");
}

#[test]
fn test_golden_file_parity() {
    let provider = GeminiProvider;
    let deployment = deployment();

    let cases: &[(&str, &str, bool)] = &[
        ("chat_request_001", GOLDEN_REQUEST_001, true),
        ("chat_request_002", GOLDEN_REQUEST_002, true),
        ("chat_response_001", GOLDEN_RESPONSE_001, false),
    ];

    for (name, golden, is_request) in cases {
        let input: Value = serde_json::from_str(golden)
            .unwrap_or_else(|e| panic!("golden {name} should be valid JSON: {e}"));
        let output = if *is_request {
            ChatProvider::transform_request(&provider, &input, &deployment)
                .unwrap_or_else(|e| panic!("golden {name} request transform failed: {e}"))
        } else {
            ChatProvider::transform_response(&provider, &input, "gemini-pro")
                .unwrap_or_else(|e| panic!("golden {name} response transform failed: {e}"))
        };
        assert_semantically_equal(&output, &input);
    }
}
