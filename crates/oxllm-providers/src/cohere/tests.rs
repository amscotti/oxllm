use super::*;
use oxllm_core::Endpoint;
use oxllm_core::deployment::{Deployment, OxllmParams};
use oxllm_core::provider::{
    ChatProvider, EmbeddingProvider, Provider, RerankProvider, ResolvedCredentials,
};
use serde_json::json;

fn deployment() -> Deployment {
    Deployment {
        model_name: "command-r-plus".to_string(),
        oxllm_params: OxllmParams {
            model: "cohere/command-r-plus".to_string(),
            api_key: Some("cohere-key".to_string()),
            ..Default::default()
        },
        ..Default::default()
    }
}

#[test]
fn endpoint_url_uses_default_cohere_base() {
    let p = CohereProvider;
    assert_eq!(
        p.endpoint_url(&deployment()),
        "https://api.cohere.com/v2/chat"
    );
}

#[test]
fn embedding_url_uses_default_cohere_base() {
    let p = CohereProvider;
    assert_eq!(
        p.embedding_url(&deployment()),
        "https://api.cohere.com/v2/embed"
    );
}

#[test]
fn rerank_url_uses_v1_default_base() {
    let p = CohereProvider;
    assert_eq!(
        p.rerank_url(&deployment()),
        "https://api.cohere.com/v1/rerank"
    );
}

#[test]
fn endpoint_url_respects_custom_base() {
    let p = CohereProvider;
    let mut dep = deployment();
    dep.oxllm_params.api_base = Some("https://custom.example.com/v2/".to_string());
    assert_eq!(p.endpoint_url(&dep), "https://custom.example.com/v2/chat");
}

#[test]
fn rerank_url_preserves_explicit_v2_base() {
    let p = CohereProvider;
    let mut dep = deployment();
    dep.oxllm_params.api_base = Some("https://custom.example.com/v2".to_string());
    assert_eq!(p.rerank_url(&dep), "https://custom.example.com/v2/rerank");
}

#[test]
fn apply_auth_uses_bearer_token() {
    let p = CohereProvider;
    let creds = ResolvedCredentials::from_key("cohere-secret");
    let mut headers = std::collections::HashMap::new();
    ChatProvider::apply_auth(&p, &deployment(), &creds, &mut headers);
    assert_eq!(
        headers.get("authorization").map(String::as_str),
        Some("Bearer cohere-secret")
    );
}

#[test]
fn request_strips_cohere_prefix_from_model() {
    let p = CohereProvider;
    let req = json!({
    "model": "command-r-plus",
    "messages": [{"role": "user", "content": "Hi"}],
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["model"], "command-r-plus");
}

/// Wildcard deployments (`cohere/*`) must take the model from the request
/// body (prefix-stripped) rather than forwarding the literal `*` upstream as
/// the model id — matching the openai_compat chat providers and the
/// Voyage/Jina rerank helpers. Covers the chat, embeddings, and rerank
/// transforms.
#[test]
fn request_with_wildcard_deployment_uses_request_model() {
    let p = CohereProvider;
    let mut dep = deployment();
    dep.oxllm_params.model = "cohere/*".to_string();

    // Chat
    let req = json!({
    "model": "cohere/command-r7-08-2024",
    "messages": [{"role": "user", "content": "Hi"}],
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(out["model"], "command-r7-08-2024");

    // Embeddings
    let req = json!({
    "model": "embed-v4.0",
    "input": ["hello"],
    });
    let out = EmbeddingProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(out["model"], "embed-v4.0");

    // Rerank
    let req = json!({
    "model": "rerank-v3.5",
    "query": "q",
    "documents": ["a"],
    });
    let out = RerankProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(out["model"], "rerank-v3.5");

    // No request model → wildcard literal forwarded, so the
    // misconfiguration surfaces upstream instead of a silent substitution.
    let req = json!({"messages": [{"role": "user", "content": "Hi"}]});
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(out["model"], "*");
}

#[test]
fn request_keeps_system_message_in_v2_messages() {
    let p = CohereProvider;
    let req = json!({
    "model": "command-r-plus",
    "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hi"},
    ],
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert!(out.get("preamble").is_none());
    let messages = out["messages"].as_array().expect("messages");
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0], req["messages"][0]);
    assert_eq!(messages[1]["role"], "user");
    assert!(out.get("system").is_none());
}

#[test]
fn request_passes_known_params_and_drops_unknown() {
    let p = CohereProvider;
    let req = json!({
    "model": "command-r-plus",
    "messages": [{"role": "user", "content": "Hi"}],
    "temperature": 0.5,
    "max_tokens": 100,
    "stream": false,
    "top_p": 0.9,
    "frequency_penalty": 0.1,
    "presence_penalty": 0.2,
    "seed": 42,
    "stop": ["END"],
    "some_unknown_param": "ignored",
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["temperature"], 0.5);
    assert_eq!(out["max_tokens"], 100);
    assert_eq!(out["p"], 0.9);
    assert!(out.get("top_p").is_none(), "top_p must be renamed to p");
    assert_eq!(out["frequency_penalty"], 0.1);
    assert_eq!(out["presence_penalty"], 0.2);
    assert_eq!(out["seed"], 42);
    assert_eq!(out["stop_sequences"], json!(["END"]));
    assert!(out.get("some_unknown_param").is_none());
}

#[test]
fn request_preserves_stream_flag() {
    let p = CohereProvider;
    let req = json!({
    "model": "command-r-plus",
    "messages": [{"role": "user", "content": "Hi"}],
    "stream": true,
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["stream"], true);
}

#[test]
fn request_maps_tools_to_v2_parameters_schema() {
    let p = CohereProvider;
    let req = json!({
    "model": "command-r-plus",
    "messages": [{"role": "user", "content": "Hi"}],
    "tools": [{
    "type": "function",
    "function": {
    "name": "get_weather",
    "description": "Get the weather",
    "parameters": {
    "type": "object",
    "properties": {
    "location": {
    "type": "string",
    "description": "The city",
    },
    },
    "required": ["location"],
    },
    },
    }],
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    let tools = out["tools"].as_array().expect("tools");
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0]["type"], "function");
    assert_eq!(tools[0]["function"]["name"], "get_weather");
    assert_eq!(tools[0]["function"]["description"], "Get the weather");
    // v2 expects OpenAI-style JSON Schema under `parameters`, not v1
    // `parameter_definitions`.
    let params = &tools[0]["function"]["parameters"];
    assert!(tools[0]["function"].get("parameter_definitions").is_none());
    assert_eq!(params["type"], "object");
    assert_eq!(params["properties"]["location"]["type"], "string");
    assert_eq!(params["properties"]["location"]["description"], "The city");
    assert_eq!(params["required"], json!(["location"]));
}

#[test]
fn request_maps_tool_role_to_v2_tool_message() {
    let p = CohereProvider;
    let req = json!({
    "model": "command-r-plus",
    "messages": [
    {"role": "user", "content": "Weather?"},
    {
    "role": "assistant",
    "content": null,
    "tool_calls": [{
    "id": "tc_1",
    "type": "function",
    "function": {"name": "get_weather", "arguments": "{\"location\":\"Paris\"}"},
    }],
    },
    {
    "role": "tool",
    "tool_call_id": "tc_1",
    "content": "22C and sunny",
    },
    ],
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    let messages = out["messages"].as_array().expect("messages");
    let tool_msg = messages
        .iter()
        .find(|m| m.get("role").and_then(|r| r.as_str()) == Some("tool"))
        .expect("tool message present");
    assert_eq!(tool_msg["tool_call_id"], "tc_1");
    let content = tool_msg["content"].as_array().expect("document content");
    assert_eq!(content[0]["type"], "document");
    assert_eq!(content[0]["document"]["data"], "22C and sunny");
}

#[test]
fn request_maps_tool_choice_string() {
    let p = CohereProvider;
    let req = json!({
    "model": "command-r-plus",
    "messages": [{"role": "user", "content": "Hi"}],
    "tool_choice": "auto",
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert!(out.get("tool_choice").is_none());

    let req = json!({
    "model": "command-r-plus",
    "messages": [{"role": "user", "content": "Hi"}],
    "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
    "tools": [
        {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "other", "parameters": {"type": "object"}}}
    ],
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["tool_choice"], "REQUIRED");
    assert_eq!(out["tools"].as_array().unwrap().len(), 1);
    assert_eq!(out["tools"][0]["function"]["name"], "get_weather");
}

#[test]
fn response_concatenates_content_blocks() {
    let p = CohereProvider;
    let resp = json!({
    "id": "cohere-1",
    "finish_reason": "COMPLETE",
    "message": {
    "role": "assistant",
    "content": [
    {"type": "text", "text": "Hello "},
    {"type": "text", "text": "world"},
    ],
    },
    "usage": {
    "billed_units": {"input_tokens": 5, "output_tokens": 2},
    "tokens": {"input_tokens": 6, "output_tokens": 3},
    },
    });
    let out = ChatProvider::transform_response(&p, &resp, "command-r-plus").expect("ok");
    assert_eq!(out["object"], "chat.completion");
    assert_eq!(out["id"], "cohere-1");
    assert_eq!(out["model"], "command-r-plus");
    let choices = out["choices"].as_array().expect("choices");
    assert_eq!(choices.len(), 1);
    assert_eq!(choices[0]["message"]["role"], "assistant");
    assert_eq!(choices[0]["message"]["content"], "Hello world");
    assert_eq!(choices[0]["finish_reason"], "stop");
    assert_eq!(out["usage"]["prompt_tokens"], 5);
    assert_eq!(out["usage"]["completion_tokens"], 2);
    assert_eq!(out["usage"]["total_tokens"], 7);
}

#[test]
fn response_maps_finish_reasons() {
    let p = CohereProvider;
    for (cohere, openai) in [
        ("COMPLETE", "stop"),
        ("MAX_TOKENS", "length"),
        ("TOOL_CALL", "tool_calls"),
        ("TOOL_CALLS", "tool_calls"),
    ] {
        let resp = json!({
        "id": "x",
        "finish_reason": cohere,
        "message": {"role": "assistant", "content": []},
        "usage": {"billed_units": {"input_tokens": 0, "output_tokens": 0}},
        });
        let out = ChatProvider::transform_response(&p, &resp, "command-r-plus").expect("ok");
        assert_eq!(out["choices"][0]["finish_reason"], openai, "for {cohere}");
    }
}

#[test]
fn response_maps_tool_calls() {
    let p = CohereProvider;
    let resp = json!({
    "id": "x",
    "finish_reason": "TOOL_CALLS",
    "message": {
    "role": "assistant",
    "content": [],
    "tool_calls": [{
    "id": "call_1",
    "type": "function",
    "function": {
    "name": "get_weather",
    "parameters": {"location": "Paris"},
    },
    }],
    },
    "usage": {"billed_units": {"input_tokens": 5, "output_tokens": 3}},
    });
    let out = ChatProvider::transform_response(&p, &resp, "command-r-plus").expect("ok");
    let calls = out["choices"][0]["message"]["tool_calls"]
        .as_array()
        .expect("tool_calls");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0]["id"], "call_1");
    assert_eq!(calls[0]["type"], "function");
    assert_eq!(calls[0]["function"]["name"], "get_weather");
    assert!(calls[0]["function"]["arguments"].is_string());
    assert_eq!(out["choices"][0]["finish_reason"], "tool_calls");
}

/// MEDIUM: a non-streaming response that omits `billed_units` must still meter
/// via `tokens` (mirroring the streaming path's fallback in
/// `build_message_end_chunk`). Previously this returned 0/0/0 silently.
#[test]
fn response_falls_back_to_tokens_without_billed_units() {
    let p = CohereProvider;
    let resp = json!({
    "id": "x",
    "finish_reason": "COMPLETE",
    "message": {
    "role": "assistant",
    "content": [{"type": "text", "text": "hi"}],
    },
    "usage": {"tokens": {"input_tokens": 7, "output_tokens": 4}},
    });
    let out = ChatProvider::transform_response(&p, &resp, "command-r-plus").expect("ok");
    assert_eq!(out["usage"]["prompt_tokens"], 7);
    assert_eq!(out["usage"]["completion_tokens"], 4);
    assert_eq!(out["usage"]["total_tokens"], 11);
}

/// Both response paths prefer billed units over raw token counts when both
/// are present, matching what the provider charges.
#[test]
fn response_prefers_billed_units_over_tokens_for_usage_basis() {
    let p = CohereProvider;
    let resp = json!({
    "id": "x",
    "finish_reason": "COMPLETE",
    "message": {
    "role": "assistant",
    "content": [{"type": "text", "text": "hi"}],
    },
    "usage": {
    "billed_units": {"input_tokens": 12, "output_tokens": 8},
    "tokens": {"input_tokens": 14, "output_tokens": 9},
    },
    });
    let out = ChatProvider::transform_response(&p, &resp, "command-r-plus").expect("ok");
    assert_eq!(out["usage"]["prompt_tokens"], 12);
    assert_eq!(out["usage"]["completion_tokens"], 8);
    assert_eq!(out["usage"]["total_tokens"], 20);
}

/// LOW: a tool-call message with no text must surface `content: null` (not
/// `""`) so strict OpenAI clients can branch on `content.is_null()` to
/// distinguish a tool-call response from a text response, matching
/// OpenAI-compatible conventions.
#[test]
fn response_tool_call_message_content_is_null() {
    let p = CohereProvider;
    let resp = json!({
    "id": "x",
    "finish_reason": "TOOL_CALLS",
    "message": {
    "role": "assistant",
    "content": [],
    "tool_calls": [{
    "id": "call_1",
    "type": "function",
    "function": {"name": "get_weather", "parameters": {"location": "Paris"}},
    }],
    },
    "usage": {"billed_units": {"input_tokens": 5, "output_tokens": 3}},
    });
    let out = ChatProvider::transform_response(&p, &resp, "command-r-plus").expect("ok");
    assert!(
        out["choices"][0]["message"]["content"].is_null(),
        "content must be null for a tool-call-only message, got: {}",
        out["choices"][0]["message"]["content"]
    );
    assert_eq!(
        out["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
        "get_weather"
    );
}

#[test]
fn stream_message_start_yields_role_chunk() {
    let p = CohereProvider;
    let raw = "event: message-start\ndata: {\"type\":\"message-start\"}";
    let out = p
        .transform_stream_chunk(raw)
        .expect("ok")
        .expect("should produce value");
    assert_eq!(out["object"], "chat.completion.chunk");
    assert_eq!(out["choices"][0]["delta"]["role"], "assistant");
}

#[test]
fn stream_content_delta_yields_text_chunk() {
    let p = CohereProvider;
    let raw = "event: content-delta\ndata: {\"type\":\"content-delta\",\"delta\":{\"message\":{\"content\":{\"text\":\"hi\"}}}}";
    let out = p
        .transform_stream_chunk(raw)
        .expect("ok")
        .expect("should produce value");
    assert_eq!(out["choices"][0]["delta"]["content"], "hi");
}

#[test]
fn stream_message_end_yields_finish_chunk() {
    let p = CohereProvider;
    let raw = "event: message-end\ndata: {\"type\":\"message-end\",\"delta\":{\"finish_reason\":\"COMPLETE\",\"usage\":{\"tokens\":{\"input_tokens\":5,\"output_tokens\":3}}}}";
    let out = p
        .transform_stream_chunk(raw)
        .expect("ok")
        .expect("should produce value");
    assert_eq!(out["choices"][0]["finish_reason"], "stop");
    assert_eq!(out["usage"]["prompt_tokens"], 5);
    assert_eq!(out["usage"]["completion_tokens"], 3);
    assert_eq!(out["usage"]["total_tokens"], 8);
}

#[test]
fn stream_done_returns_none() {
    let p = CohereProvider;
    assert!(
        p.transform_stream_chunk("data: [DONE]")
            .expect("ok")
            .is_none()
    );
    assert!(p.transform_stream_chunk("").expect("ok").is_none());
}

#[test]
fn stream_unknown_event_returns_none() {
    let p = CohereProvider;
    assert!(
        p.transform_stream_chunk("data: {\"type\":\"some-other-event\"}")
            .expect("ok")
            .is_none()
    );
}

#[test]
fn stream_message_start_carries_id_and_model() {
    let p = CohereProvider;
    let raw = "event: message-start\ndata: {\"type\":\"message-start\",\"id\":\"chatcmpl-abc\",\"model\":\"command-r-plus\"}";
    let out = p
        .transform_stream_chunk(raw)
        .expect("ok")
        .expect("should produce value");
    assert_eq!(out["id"], "chatcmpl-abc");
    assert_eq!(out["model"], "command-r-plus");
    assert!(out.get("created").is_some(), "created must be present");
}

#[test]
fn stream_subsequent_chunks_inherit_id_and_model_from_message_start() {
    let p = CohereProvider;
    // Use the stateful (production) path so cross-chunk state lives in a
    // host-owned slot instead of the process-wide test cell, which other
    // stream tests mutate concurrently.
    let mut state: Option<Box<dyn std::any::Any + Send + Sync>> = None;
    let request = json!({});
    // Prime the stream meta with a message-start carrying id/model.
    let start = "event: message-start\ndata: {\"type\":\"message-start\",\"id\":\"chatcmpl-xyz\",\"model\":\"command-r-plus\"}";
    let _ = p
        .transform_stream_chunk_stateful(start, &mut state, &request)
        .expect("ok");
    // Now send a content-delta — it must inherit the stashed id/model.
    let delta = "event: content-delta\ndata: {\"type\":\"content-delta\",\"delta\":{\"message\":{\"content\":{\"text\":\"hi\"}}}}";
    let out = p
        .transform_stream_chunk_stateful(delta, &mut state, &request)
        .expect("ok")
        .expect("should produce value");
    assert_eq!(out["id"], "chatcmpl-xyz");
    assert_eq!(out["model"], "command-r-plus");
    assert!(out.get("created").is_some(), "created must be present");
    assert_eq!(out["choices"][0]["delta"]["content"], "hi");
}

#[test]
fn supported_params_includes_top_k() {
    let p = CohereProvider;
    let params = ChatProvider::supported_params(&p).expect("should publish params");
    assert!(
        params.contains(&"top_k"),
        "top_k must be in supported_params so drop_params does not strip it: {params:?}"
    );
    assert!(params.contains(&"top_p"));
}

#[test]
fn embedding_request_maps_input_to_texts() {
    let p = CohereProvider;
    let req = json!({"model": "embed-v3", "input": "hello"});
    let out = EmbeddingProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["texts"], json!(["hello"]));
    assert_eq!(out["input_type"], DEFAULT_EMBED_INPUT_TYPE);
    assert_eq!(out["model"], "command-r-plus");
}

#[test]
fn embedding_request_maps_array_input_to_texts() {
    let p = CohereProvider;
    let req = json!({"model": "embed-v3", "input": ["hello", "world"]});
    let out = EmbeddingProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["texts"], json!(["hello", "world"]));
}

#[test]
fn default_embed_input_type_is_valid_cohere_value() {
    // Cohere v2 /embed only accepts search_document | search_query |
    // classification | clustering | image. The default for embedding
    // documents must be `search_document` — `text` is rejected with 422.
    assert_eq!(DEFAULT_EMBED_INPUT_TYPE, "search_document");
    let valid = [
        "search_document",
        "search_query",
        "classification",
        "clustering",
        "image",
    ];
    assert!(valid.contains(&DEFAULT_EMBED_INPUT_TYPE));
}

#[test]
fn embedding_request_uses_default_input_type() {
    let p = CohereProvider;
    let req = json!({"model": "embed-v3", "input": "hello"});
    let out = EmbeddingProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["input_type"], "search_document");
}

#[test]
fn embedding_response_maps_to_openai_data_array() {
    let p = CohereProvider;
    let resp = json!({
    "id": "emb-1",
    "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
    "meta": {"billed_units": {"input_tokens": 4}},
    });
    let out = EmbeddingProvider::transform_response(&p, &resp, "embed-v3").expect("ok");
    assert_eq!(out["object"], "list");
    let data = out["data"].as_array().expect("data");
    assert_eq!(data.len(), 2);
    assert_eq!(data[0]["index"], 0);
    assert_eq!(data[0]["object"], "embedding");
    assert_eq!(data[0]["embedding"], json!([0.1, 0.2, 0.3]));
    assert_eq!(data[1]["index"], 1);
    assert_eq!(out["usage"]["prompt_tokens"], 4);
    assert_eq!(out["usage"]["total_tokens"], 4);
}

#[test]
fn embedding_response_handles_object_form() {
    let p = CohereProvider;
    let resp = json!({
    "id": "emb-2",
    "embeddings": {"float": [[0.1, 0.2]]},
    "meta": {"billed_units": {"input_tokens": 1}},
    });
    let out = EmbeddingProvider::transform_response(&p, &resp, "embed-v3").expect("ok");
    let data = out["data"].as_array().expect("data");
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["embedding"], json!([0.1, 0.2]));
}

/// Image-embedding calls report tokens under `billed_units.images`, not
/// `input_tokens`. The reference sums both into `prompt_tokens`; an image-only
/// response must meter the image tokens rather than report 0.
#[test]
fn embedding_response_meters_image_tokens_when_no_input_tokens() {
    let p = CohereProvider;
    let resp = json!({
    "id": "emb-img",
    "embeddings": [[0.1, 0.2]],
    "meta": {"billed_units": {"images": 5}},
    });
    let out = EmbeddingProvider::transform_response(&p, &resp, "embed-v3").expect("ok");
    assert_eq!(out["usage"]["prompt_tokens"], 5);
    assert_eq!(out["usage"]["total_tokens"], 5);
}

/// When both text and image tokens are billed, both are summed.
#[test]
fn embedding_response_sums_text_and_image_tokens() {
    let p = CohereProvider;
    let resp = json!({
    "id": "emb-mix",
    "embeddings": [[0.1, 0.2]],
    "meta": {"billed_units": {"input_tokens": 3, "images": 7}},
    });
    let out = EmbeddingProvider::transform_response(&p, &resp, "embed-v3").expect("ok");
    assert_eq!(out["usage"]["prompt_tokens"], 10);
    assert_eq!(out["usage"]["total_tokens"], 10);
}

#[test]
fn rerank_request_maps_to_cohere_shape() {
    let p = CohereProvider;
    let req = json!({
    "model": "rerank-v3",
    "query": "What is AI?",
    "documents": ["AI is ...", "Cats are nice"],
    "top_n": 2,
    });
    let out = RerankProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["model"], "command-r-plus");
    assert_eq!(out["query"], "What is AI?");
    assert_eq!(out["documents"], json!(["AI is ...", "Cats are nice"]));
    assert_eq!(out["top_n"], 2);
}

#[test]
fn rerank_response_preserves_results_and_usage() {
    let p = CohereProvider;
    let resp = json!({
    "id": "rr-1",
    "results": [
    {"index": 0, "relevance_score": 0.9},
    {"index": 1, "relevance_score": 0.2},
    ],
    "meta": {"billed_units": {"search_units": 1}},
    });
    let out = RerankProvider::transform_response(&p, &resp, "rerank-v3").expect("ok");
    assert_eq!(out["id"], "rr-1");
    let results = out["results"].as_array().expect("results");
    assert_eq!(results.len(), 2);
    assert_eq!(out["usage"]["search_units"], 1);
}

#[test]
fn provider_id_and_supports() {
    let p = CohereProvider;
    assert_eq!(p.id(), "cohere");
    assert!(p.supports(Endpoint::ChatCompletions));
    assert!(p.supports(Endpoint::Embeddings));
    assert!(p.supports(Endpoint::Rerank));
    assert!(!p.supports(Endpoint::Completions));
    assert!(p.chat().is_some());
    assert!(p.embeddings().is_some());
    assert!(p.rerank().is_some());
}

#[test]
fn strip_cohere_prefix_handles_unknown() {
    assert_eq!(
        strip_cohere_prefix("cohere/command-r-plus"),
        "command-r-plus"
    );
    assert_eq!(strip_cohere_prefix("command-r-plus"), "command-r-plus");
}

#[test]
fn map_finish_reason_unknown_falls_back_to_stop() {
    assert_eq!(map_finish_reason_to_openai("UNKNOWN"), json!("stop"));
    assert_eq!(map_finish_reason_to_openai("COMPLETE"), json!("stop"));
    assert_eq!(map_finish_reason_to_openai("MAX_TOKENS"), json!("length"));
    assert_eq!(
        map_finish_reason_to_openai("TOOL_CALLS"),
        json!("tool_calls")
    );
    assert_eq!(map_finish_reason_to_openai("ERROR"), json!("stop"));
    assert_eq!(map_finish_reason_to_openai("error"), json!("stop"));
    assert_eq!(
        map_finish_reason_to_openai("ERROR_TOXIC"),
        json!("content_filter")
    );
}

/// Bug #40: Cohere v2 Chat expects `p` and `k`, not `top_p` / `top_k`.
/// The transform must rename these parameters on the way out.
#[test]
fn request_renames_top_p_to_p_and_top_k_to_k() {
    let p = CohereProvider;
    let req = json!({
    "model": "command-r-plus",
    "messages": [{"role": "user", "content": "Hi"}],
    "top_p": 0.9,
    "top_k": 40,
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["p"], 0.9, "top_p must be forwarded as p");
    assert_eq!(out["k"], 40, "top_k must be forwarded as k");
    assert!(
        out.get("top_p").is_none(),
        "top_p must NOT appear in Cohere output"
    );
    assert!(
        out.get("top_k").is_none(),
        "top_k must NOT appear in Cohere output"
    );
}

/// When `top_p` / `top_k` are absent the transform must not synthesise them.
#[test]
fn request_omits_p_and_k_when_not_provided() {
    let p = CohereProvider;
    let req = json!({
    "model": "command-r-plus",
    "messages": [{"role": "user", "content": "Hi"}],
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert!(out.get("p").is_none());
    assert!(out.get("k").is_none());
    assert!(out.get("top_p").is_none());
    assert!(out.get("top_k").is_none());
}

#[test]
fn supported_params_published() {
    let p = CohereProvider;
    let params = ChatProvider::supported_params(&p).expect("should publish params");
    assert!(params.contains(&"temperature"));
    assert!(params.contains(&"tools"));
    assert!(params.contains(&"max_tokens"));
}

const GOLDEN_REQUEST_001: &str = include_str!("goldens/chat_request_001.json");
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
fn golden_request_round_trips() {
    let p = CohereProvider;
    let expected: Value = serde_json::from_str(GOLDEN_REQUEST_001).expect("valid json");
    let input = json!({
    "model": "command-r-plus",
    "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
    ],
    "temperature": 0.7,
    "max_tokens": 256,
    "stream": false,
    });
    let out = ChatProvider::transform_request(&p, &input, &deployment()).expect("transform");
    assert_semantically_equal(&out, &expected);
}

#[test]
fn golden_response_round_trips() {
    let p = CohereProvider;
    let cohere_input: Value = serde_json::from_str(GOLDEN_RESPONSE_001).expect("valid json");
    let out =
        ChatProvider::transform_response(&p, &cohere_input, "command-r-plus").expect("transform");
    assert_eq!(out["object"], "chat.completion");
    assert_eq!(
        out["choices"][0]["message"]["content"],
        "The capital of France is Paris."
    );
    assert_eq!(out["choices"][0]["finish_reason"], "stop");
    assert_eq!(out["usage"]["prompt_tokens"], 12);
    assert_eq!(out["usage"]["completion_tokens"], 8);
    assert_eq!(out["usage"]["total_tokens"], 20);
}

// --- Ported twin: model override ---
//
// The provider must override the request body's `model` with the
// deployment's configured model (source `twin_cohere_provider_success`
// adapted for our API).

#[test]
fn transform_request_overrides_request_model_with_deployment_model() {
    let p = CohereProvider;
    let mut dep = deployment();
    dep.oxllm_params.model = "cohere/command-r".to_string();
    let req = json!({
        "model": "caller-supplied-ignored",
        "messages": [{"role": "user", "content": "Hi"}],
    });
    let out = ChatProvider::transform_request(&p, &req, &dep).expect("ok");
    assert_eq!(out["model"], "command-r");
    assert_eq!(out["messages"][0]["content"], "Hi");
}

// v2 preserves chronological message order, including system instructions.

#[test]
fn request_preserves_system_and_conversation_order() {
    let p = CohereProvider;
    let req = json!({
        "model": "command-r-plus",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "What is 2+2?"},
        ],
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert!(out.get("preamble").is_none());
    let messages = out["messages"].as_array().expect("messages");
    assert_eq!(messages.len(), 4);
    assert_eq!(out["messages"], req["messages"]);
}

#[test]
fn response_maps_tool_calls_with_arguments_as_string() {
    let p = CohereProvider;
    let resp = json!({
    "id": "x",
    "finish_reason": "TOOL_CALLS",
    "message": {
    "role": "assistant",
    "content": [],
    "tool_calls": [{
    "id": "call_1",
    "type": "function",
    "function": {
    "name": "get_weather",
    "parameters": {"location": "Paris"},
    },
    }],
    },
    "usage": {"billed_units": {"input_tokens": 5, "output_tokens": 3}},
    });
    let out = ChatProvider::transform_response(&p, &resp, "command-r-plus").expect("ok");
    let calls = out["choices"][0]["message"]["tool_calls"]
        .as_array()
        .expect("tool_calls");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0]["id"], "call_1");
    assert_eq!(calls[0]["type"], "function");
    assert_eq!(calls[0]["function"]["name"], "get_weather");
    let args: serde_json::Value =
        serde_json::from_str(calls[0]["function"]["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(args["location"], "Paris");
    assert_eq!(out["choices"][0]["finish_reason"], "tool_calls");
}

#[test]
fn request_tool_calls_round_trip_with_tool_results() {
    let p = CohereProvider;
    let req = json!({
    "model": "command-r-plus",
    "messages": [
    {"role": "user", "content": "What's the weather in Paris?"},
    {
    "role": "assistant",
    "content": null,
    "tool_calls": [{
    "id": "call_1",
    "type": "function",
    "function": {"name": "get_weather", "arguments": "{\"location\":\"Paris\"}"},
    }],
    },
    {
    "role": "tool",
    "tool_call_id": "call_1",
    "content": "22C and sunny",
    },
    ],
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    let messages = out["messages"].as_array().expect("messages");
    assert_eq!(messages.len(), 3);
    let tool_msg = messages
        .iter()
        .find(|m| m.get("role").and_then(|r| r.as_str()) == Some("tool"))
        .expect("tool message present");
    assert_eq!(tool_msg["tool_call_id"], "call_1");
    let content = tool_msg["content"].as_array().expect("document content");
    assert_eq!(content[0]["type"], "document");
    assert_eq!(content[0]["document"]["data"], "22C and sunny");
}

#[test]
fn response_with_billed_units_usage() {
    let p = CohereProvider;
    let resp = json!({
    "id": "cohere-1",
    "finish_reason": "COMPLETE",
    "message": {
    "role": "assistant",
    "content": [{"type": "text", "text": "Hello"}],
    },
    "usage": {
    "billed_units": {"input_tokens": 5, "output_tokens": 2},
    },
    });
    let out = ChatProvider::transform_response(&p, &resp, "command-r-plus").expect("ok");
    assert_eq!(out["usage"]["prompt_tokens"], 5);
    assert_eq!(out["usage"]["completion_tokens"], 2);
    assert_eq!(out["usage"]["total_tokens"], 7);
}

#[test]
fn request_keeps_multiple_system_messages() {
    let p = CohereProvider;
    let req = json!({
    "model": "command-r-plus",
    "messages": [
    {"role": "system", "content": "First system."},
    {"role": "system", "content": "Second system."},
    {"role": "user", "content": "Hi"},
    ],
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert!(out.get("preamble").is_none());
    assert_eq!(out["messages"], req["messages"]);
}

#[test]
fn embedding_response_handles_single_embedding_object() {
    let p = CohereProvider;
    let resp = json!({
    "id": "emb-3",
    "embeddings": {"float": [[0.5, 0.6]]},
    "meta": {"billed_units": {"input_tokens": 1}},
    });
    let out = EmbeddingProvider::transform_response(&p, &resp, "embed-v3").expect("ok");
    let data = out["data"].as_array().expect("data");
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["embedding"], json!([0.5, 0.6]));
    assert_eq!(data[0]["index"], 0);
    assert_eq!(data[0]["object"], "embedding");
}

// =============================================================================
// Ported from the reference `tests/llm_translation/test_rerank.py`.
//
// These cover the Cohere rerank transform surface: URL construction for
// custom bases, request payload shape, and response handling. Network/E2E
// tests from the Python suite are omitted (no live API); tests are adapted
// to the oxllm transform API (`transform_request` / `transform_response`).
// =============================================================================

/// Ported from `test_rerank_custom_api_base` (v1 case): a custom base that
/// already ends in `/v1` must be used verbatim with `/rerank` appended.
#[test]
fn rerank_url_custom_v1_base_passes_through() {
    let p = CohereProvider;
    let mut dep = deployment();
    dep.oxllm_params.api_base = Some("https://example.com/v1".to_string());
    assert_eq!(p.rerank_url(&dep), "https://example.com/v1/rerank");
}

/// Ported from `test_rerank_custom_api_base`: a custom base with a trailing
/// slash is trimmed before the `/v1/rerank` suffix is appended.
#[test]
fn rerank_url_custom_base_trims_trailing_slash() {
    let p = CohereProvider;
    let mut dep = deployment();
    dep.oxllm_params.api_base = Some("https://example.com/v1/".to_string());
    assert_eq!(p.rerank_url(&dep), "https://example.com/v1/rerank");
}

/// A configured API version is an explicit operator choice.
#[test]
fn rerank_url_custom_v2_base_is_not_rewritten() {
    let p = CohereProvider;
    let mut dep = deployment();
    dep.oxllm_params.api_base = Some("https://example.com/v2".to_string());
    assert_eq!(p.rerank_url(&dep), "https://example.com/v2/rerank");
}

/// Ported from `test_cohere_rerank_v2_client`: the request payload must
/// carry the resolved model, query, documents, and top_n. The supported
/// Cohere rerank params (`rank_fields`, `return_documents`,
/// `max_chunks_per_doc`) are forwarded; `max_tokens_per_doc` from the
/// Python suite is NOT in oxllm's allow-list and is dropped.
#[test]
fn rerank_request_forwards_supported_params_and_drops_max_tokens_per_doc() {
    let p = CohereProvider;
    let req = json!({
    "model": "rerank-multilingual-v3.0",
    "query": "Hello there!",
    "documents": ["Hello there!", "How are you?", "How do you do?"],
    "top_n": 2,
    "rank_fields": ["title"],
    "return_documents": true,
    "max_chunks_per_doc": 3,
    "max_tokens_per_doc": 3,
    });
    let out = RerankProvider::transform_request(&p, &req, &deployment()).expect("ok");
    // Model is overridden with the deployment's configured model.
    assert_eq!(out["model"], "command-r-plus");
    assert_eq!(out["query"], "Hello there!");
    assert_eq!(
        out["documents"],
        json!(["Hello there!", "How are you?", "How do you do?"])
    );
    assert_eq!(out["top_n"], 2);
    assert_eq!(out["rank_fields"], json!(["title"]));
    assert_eq!(out["return_documents"], true);
    assert_eq!(out["max_chunks_per_doc"], 3);
    assert!(
        out.get("max_tokens_per_doc").is_none(),
        "max_tokens_per_doc is not a supported Cohere rerank param and must be dropped"
    );
}

/// Ported from `test_cohere_rerank_v2_client` (top_n_override fallback path):
/// when `top_n` is absent but `top_n_override` is present, the override is
/// promoted to `top_n`.
#[test]
fn rerank_request_top_n_override_used_when_top_n_absent() {
    let p = CohereProvider;
    let req = json!({
    "model": "rerank-v3",
    "query": "q",
    "documents": ["a"],
    "top_n_override": 5,
    });
    let out = RerankProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["top_n"], 5);
}

/// A null `top_n` is treated as absent (not forwarded), so the output omits
/// the field entirely rather than emitting `"top_n": null`.
#[test]
fn rerank_request_omits_top_n_when_null() {
    let p = CohereProvider;
    let req = json!({
    "model": "rerank-v3",
    "query": "q",
    "documents": ["a"],
    "top_n": null,
    });
    let out = RerankProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert!(
        out.get("top_n").is_none(),
        "null top_n must not be forwarded"
    );
}

// =============================================================================
// Reference feature-gap parity: max_completion_tokens alias, Cohere-native
// pass-through params, citations, and tool-plan / citation streaming.
// =============================================================================

/// Feature 1: OpenAI `max_completion_tokens` maps to Cohere `max_tokens`.
#[test]
fn request_maps_max_completion_tokens_to_max_tokens() {
    let p = CohereProvider;
    let req = json!({
    "model": "command-r-plus",
    "messages": [{"role": "user", "content": "Hi"}],
    "max_completion_tokens": 4096,
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["max_tokens"], 4096);
    assert!(
        out.get("max_completion_tokens").is_none(),
        "max_completion_tokens must be renamed to max_tokens"
    );
}

/// Feature 1 precedence: when both are sent, `max_completion_tokens` wins.
#[test]
fn request_max_completion_tokens_overrides_max_tokens() {
    let p = CohereProvider;
    let req = json!({
    "model": "command-r-plus",
    "messages": [{"role": "user", "content": "Hi"}],
    "max_tokens": 100,
    "max_completion_tokens": 256,
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["max_tokens"], 256);
}

/// v2 has no `num_generations` parameter.
#[test]
fn request_does_not_send_num_generations() {
    let p = CohereProvider;
    let req = json!({
    "model": "command-r-plus",
    "messages": [{"role": "user", "content": "Hi"}],
    "n": 3,
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert!(out.get("num_generations").is_none());
    assert!(out.get("n").is_none());
}

/// Only documented v2 fields are forwarded; v1 RAG fields were removed.
#[test]
fn request_forwards_documents_and_drops_removed_v1_fields() {
    let p = CohereProvider;
    let req = json!({
    "model": "command-r-plus",
    "messages": [{"role": "user", "content": "Hi"}],
    "connectors": [{"id": "web-search"}],
    "search_queries_only": true,
    "documents": [{"id": "doc1"}],
    "conversation_id": "conv-123",
    "prompt_truncation": "AUTO_PRESERVE_ORDER",
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert!(out.get("connectors").is_none());
    assert!(out.get("search_queries_only").is_none());
    assert_eq!(out["documents"], json!([{"id": "doc1"}]));
    assert!(out.get("conversation_id").is_none());
    assert!(out.get("prompt_truncation").is_none());
}

/// Feature 2: native params are also readable from `extra_body` (reference
/// pass-through convention).
#[test]
fn request_native_params_read_from_extra_body() {
    let p = CohereProvider;
    let req = json!({
    "model": "command-r-plus",
    "messages": [{"role": "user", "content": "Hi"}],
    "extra_body": {"documents": ["doc-from-extra"], "conversation_id": "conv-from-extra", "n": 2},
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["documents"], json!(["doc-from-extra"]));
    assert!(out.get("conversation_id").is_none());
    assert!(out.get("num_generations").is_none());
}

/// Feature 2: a top-level value wins over an `extra_body` one.
#[test]
fn request_top_level_native_param_beats_extra_body() {
    let p = CohereProvider;
    let req = json!({
    "model": "command-r-plus",
    "messages": [{"role": "user", "content": "Hi"}],
    "documents": ["top-level"],
    "extra_body": {"documents": ["extra"]},
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["documents"], json!(["top-level"]));
}

/// `force_single_step` is NOT auto-set when omitted, even when the last
/// message is from the user. The auto-set heuristic is a reference **v1**
/// transform behavior; this provider targets the v2 endpoint, which (per
/// the reference `v2_transformation.py`) does not apply it. v2 parity.
#[test]
fn request_force_single_step_not_auto_set_when_last_message_is_user() {
    let p = CohereProvider;
    let req = json!({
    "model": "command-r-plus",
    "messages": [{"role": "user", "content": "Hi"}],
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert!(
        out.get("force_single_step").is_none(),
        "v2 transform must not auto-set force_single_step"
    );
}

/// `force_single_step` is NOT auto-set when the last message is not from
/// the user (e.g. assistant mid-conversation).
#[test]
fn request_force_single_step_omitted_when_last_message_is_assistant() {
    let p = CohereProvider;
    let req = json!({
    "model": "command-r-plus",
    "messages": [
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello"},
    ],
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert!(
        out.get("force_single_step").is_none(),
        "must not auto-set when last message is not user"
    );
}

/// `force_single_step` was removed in v2, even when explicitly supplied.
#[test]
fn request_force_single_step_explicit_is_omitted() {
    let p = CohereProvider;
    let req = json!({
    "model": "command-r-plus",
    "messages": [{"role": "user", "content": "Hi"}],
    "force_single_step": false,
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert!(out.get("force_single_step").is_none());
}

/// Non-streaming citations are surfaced as a provider extension. Per the
/// Cohere v2 Chat API and the reference `v2_transformation.transform_response`,
/// citations live under `message.citations` (not top-level) on the
/// non-streaming path — this is the real API shape.
#[test]
fn response_extracts_citations_from_message() {
    let p = CohereProvider;
    let resp = json!({
    "id": "cohere-1",
    "finish_reason": "COMPLETE",
    "message": {
    "role": "assistant",
    "content": [{"type": "text", "text": "Paris is in France."}],
    "citations": [
    {"start": 0, "end": 5, "text": "Paris", "document_ids": ["doc-a"]},
    {"start": 9, "end": 15, "text": "France", "document_ids": ["doc-b"]},
    ],
    },
    "usage": {"billed_units": {"input_tokens": 5, "output_tokens": 2}},
    });
    let out = ChatProvider::transform_response(&p, &resp, "command-r-plus").expect("ok");
    let citations = out["citations"].as_array().expect("citations present");
    assert_eq!(citations.len(), 2);
    assert_eq!(citations[0]["text"], "Paris");
    assert_eq!(citations[0]["document_ids"], json!(["doc-a"]));
    assert_eq!(citations[0]["start"], 0);
    assert_eq!(citations[0]["end"], 5);
    assert_eq!(citations[1]["text"], "France");
}

/// `message.citations` takes precedence over a top-level `citations`
/// (defensive fallback path) when both are present.
#[test]
fn response_prefers_message_citations_over_top_level() {
    let p = CohereProvider;
    let resp = json!({
    "id": "cohere-1",
    "finish_reason": "COMPLETE",
    "message": {
    "role": "assistant",
    "content": [{"type": "text", "text": "Hi"}],
    "citations": [
    {"start": 0, "end": 2, "text": "from-message", "document_ids": ["m"]},
    ],
    },
    "citations": [
    {"start": 0, "end": 2, "text": "from-top", "document_ids": ["t"]},
    ],
    "usage": {"billed_units": {"input_tokens": 1, "output_tokens": 1}},
    });
    let out = ChatProvider::transform_response(&p, &resp, "command-r-plus").expect("ok");
    let citations = out["citations"].as_array().expect("citations present");
    assert_eq!(citations.len(), 1);
    assert_eq!(citations[0]["text"], "from-message");
}

/// Top-level `body.citations` is still honored as a fallback when
/// `message.citations` is absent (defensive against non-standard upstreams).
#[test]
fn response_falls_back_to_top_level_citations() {
    let p = CohereProvider;
    let resp = json!({
    "id": "cohere-1",
    "finish_reason": "COMPLETE",
    "message": {
    "role": "assistant",
    "content": [{"type": "text", "text": "Hi"}],
    },
    "citations": [
    {"start": 0, "end": 2, "text": "top", "document_ids": ["t"]},
    ],
    "usage": {"billed_units": {"input_tokens": 1, "output_tokens": 1}},
    });
    let out = ChatProvider::transform_response(&p, &resp, "command-r-plus").expect("ok");
    let citations = out["citations"].as_array().expect("citations present");
    assert_eq!(citations.len(), 1);
    assert_eq!(citations[0]["text"], "top");
}

/// Feature 3: when there are no citations, the field is omitted (not `[]`).
#[test]
fn response_omits_citations_when_absent() {
    let p = CohereProvider;
    let resp = json!({
    "id": "cohere-1",
    "finish_reason": "COMPLETE",
    "message": {
    "role": "assistant",
    "content": [{"type": "text", "text": "Hi"}],
    },
    "usage": {"billed_units": {"input_tokens": 5, "output_tokens": 2}},
    });
    let out = ChatProvider::transform_response(&p, &resp, "command-r-plus").expect("ok");
    assert!(
        out.get("citations").is_none(),
        "no citations field should be added when the source has none"
    );
}

/// Feature 4: `tool-plan-delta` events are emitted as `reasoning_content`.
#[test]
fn stream_tool_plan_delta_yields_reasoning_content() {
    let p = CohereProvider;
    let raw = "event: tool-plan-delta\ndata: {\"type\":\"tool-plan-delta\",\"delta\":{\"message\":{\"tool_plan\":\"I will search\"}}}";
    let out = p
        .transform_stream_chunk(raw)
        .expect("ok")
        .expect("should produce value");
    assert_eq!(
        out["choices"][0]["delta"]["reasoning_content"],
        "I will search"
    );
    // Must not leak into content.
    assert!(out["choices"][0]["delta"].get("content").is_none());
}

/// Feature 4: empty tool-plan text yields no chunk (avoids noise).
#[test]
fn stream_tool_plan_delta_empty_yields_none() {
    let p = CohereProvider;
    let raw = "event: tool-plan-delta\ndata: {\"type\":\"tool-plan-delta\",\"delta\":{\"message\":{\"tool_plan\":\"\"}}}";
    assert!(p.transform_stream_chunk(raw).expect("ok").is_none());
}

/// Feature 5: `citation-start` events emit their citations inline as a
/// top-level `citations` array on the chunk itself.
///
/// This is deterministic: each `transform_stream_chunk` call is a pure
/// function of its input, so the assertion cannot be perturbed by a
/// concurrent test's `message-start` (the old accumulate-and-flush-on-end
/// design shared a process-wide cell that another streaming test could wipe
/// mid-sequence).
#[test]
fn stream_citation_start_emits_citations_inline() {
    let p = CohereProvider;
    let raw = "event: citation-start\ndata: {\"type\":\"citation-start\",\"delta\":{\"message\":{\"citations\":[{\"start\":0,\"end\":5,\"text\":\"Paris\",\"document_ids\":[\"d1\"]}]}}}";
    let out = p
        .transform_stream_chunk(raw)
        .expect("ok")
        .expect("citation-start must emit a chunk");
    let citations = out["citations"].as_array().expect("citations present");
    assert_eq!(citations.len(), 1);
    assert_eq!(citations[0]["text"], "Paris");
    assert_eq!(citations[0]["start"], 0);
    assert_eq!(citations[0]["end"], 5);
    assert_eq!(citations[0]["document_ids"], json!(["d1"]));
    assert_eq!(out["object"], "chat.completion.chunk");
}

/// Feature 5: each `citation-start` event carries its own citations, so
/// multiple events in a stream each produce an independent chunk. No
/// cross-event state is required, so this is deterministic under parallel
/// test execution.
#[test]
fn stream_citation_start_emits_each_event_independently() {
    let p = CohereProvider;
    let cite1 = "event: citation-start\ndata: {\"type\":\"citation-start\",\"delta\":{\"message\":{\"citations\":[{\"start\":0,\"end\":5,\"text\":\"Paris\",\"document_ids\":[\"d1\"]}]}}}";
    let out1 = p
        .transform_stream_chunk(cite1)
        .expect("ok")
        .expect("first citation-start emits");
    let cite2 = "event: citation-start\ndata: {\"type\":\"citation-start\",\"delta\":{\"message\":{\"citations\":[{\"start\":9,\"end\":15,\"text\":\"France\",\"document_ids\":[\"d2\"]}]}}}";
    let out2 = p
        .transform_stream_chunk(cite2)
        .expect("ok")
        .expect("second citation-start emits");

    let c1 = out1["citations"].as_array().expect("first citations");
    let c2 = out2["citations"].as_array().expect("second citations");
    assert_eq!(c1.len(), 1);
    assert_eq!(c2.len(), 1);
    assert_eq!(c1[0]["text"], "Paris");
    assert_eq!(c2[0]["text"], "France");
}

/// Feature 5: a `citation-start` with no citations yields no chunk.
#[test]
fn stream_citation_start_empty_yields_none() {
    let p = CohereProvider;
    let raw = "event: citation-start\ndata: {\"type\":\"citation-start\",\"delta\":{\"message\":{\"citations\":[]}}}";
    assert!(p.transform_stream_chunk(raw).expect("ok").is_none());
}

/// Feature 5: `message-end` carries only finish reason / usage — it must
/// never synthesize a `citations` field, regardless of what other events a
/// concurrent test may have produced. Deterministic by construction.
#[test]
fn stream_message_end_never_synthesizes_citations() {
    let p = CohereProvider;
    let end = "event: message-end\ndata: {\"type\":\"message-end\",\"delta\":{\"finish_reason\":\"COMPLETE\",\"usage\":{\"tokens\":{\"input_tokens\":5,\"output_tokens\":3}}}}";
    let out = p
        .transform_stream_chunk(end)
        .expect("ok")
        .expect("message-end chunk");
    assert!(
        out.get("citations").is_none(),
        "message-end must not carry citations"
    );
    assert_eq!(out["choices"][0]["finish_reason"], "stop");
    assert_eq!(out["usage"]["total_tokens"], 8);
}

/// The expanded supported_params list must advertise the new native params so
/// drop_params does not strip them.
#[test]
fn supported_params_includes_cohere_native_and_alias() {
    let p = CohereProvider;
    let params = ChatProvider::supported_params(&p).expect("should publish params");
    for required in ["max_completion_tokens", "documents"] {
        assert!(params.contains(&required), "missing {required}");
    }
    for removed in [
        "n",
        "preamble",
        "connectors",
        "search_queries_only",
        "conversation_id",
        "prompt_truncation",
        "force_single_step",
    ] {
        assert!(
            !params.contains(&removed),
            "{removed} is not a v2 parameter: {params:?}"
        );
    }
}

/// Ported from `test_rerank_response_assertions`: the response shape must
/// preserve the `id`, the ordered `results` array (each with `index` and a
/// float `relevance_score`), and surface `billed_units.search_units` as usage.
#[test]
fn rerank_response_preserves_results_order_and_scores() {
    let p = CohereProvider;
    let resp = json!({
    "id": "ab0fcca0-b617-11ef-b292-0242ac110002",
    "results": [
    {"index": 2, "relevance_score": 0.9958819150924683},
    {"index": 0, "relevance_score": 0.001293411129154265},
    {"index": 1, "relevance_score": 7.641685078851879e-05},
    ],
    "meta": {
    "api_version": {"version": "1.0"},
    "billed_units": {"search_units": 1},
    },
    });
    let out = RerankProvider::transform_response(&p, &resp, "rerank-v3").expect("ok");
    assert_eq!(out["id"], "ab0fcca0-b617-11ef-b292-0242ac110002");
    let results = out["results"].as_array().expect("results");
    assert_eq!(results.len(), 3);
    // Order is preserved verbatim (no re-sort).
    assert_eq!(results[0]["index"], 2);
    assert_eq!(results[0]["relevance_score"], 0.9958819150924683);
    assert_eq!(results[1]["index"], 0);
    assert_eq!(results[2]["index"], 1);
    // `billed_units.search_units` is hoisted into the OpenAI-shaped `usage`.
    assert_eq!(out["usage"]["search_units"], 1);
    // `api_version` is provider metadata, not part of the normalized shape.
    assert!(out.get("meta").is_none());
}

/// Ported from `test_rerank_cohere_api` (`return_documents=True`): when the
/// upstream response includes `document` objects on each result, they pass
/// through unchanged so callers receive the matched text.
#[test]
fn rerank_response_preserves_document_objects_when_returned() {
    let p = CohereProvider;
    let resp = json!({
    "id": "rr-1",
    "results": [
    {"index": 0, "relevance_score": 0.9, "document": {"text": "hello"}},
    {"index": 1, "relevance_score": 0.2, "document": {"text": "world"}},
    ],
    "meta": {"billed_units": {"search_units": 1}},
    });
    let out = RerankProvider::transform_response(&p, &resp, "rerank-v3").expect("ok");
    let results = out["results"].as_array().expect("results");
    assert_eq!(results.len(), 2);
    assert_eq!(results[0]["document"]["text"], "hello");
    assert_eq!(results[1]["document"]["text"], "world");
}

/// Ported from `test_rerank_response_assertions` (meta-less variant): a
/// response without `meta.billed_units` must not synthesize a `usage` block.
#[test]
fn rerank_response_omits_usage_when_no_billed_units() {
    let p = CohereProvider;
    let resp = json!({
    "id": "rr-2",
    "results": [{"index": 0, "relevance_score": 0.5}],
    });
    let out = RerankProvider::transform_response(&p, &resp, "rerank-v3").expect("ok");
    assert_eq!(out["id"], "rr-2");
    assert_eq!(out["results"][0]["relevance_score"], 0.5);
    assert!(
        out.get("usage").is_none(),
        "no billed_units means no usage block"
    );
}

// The OpenAI `developer` role maps to Cohere v2's `system` role.
#[test]
fn request_maps_developer_message_to_system() {
    let p = CohereProvider;
    let req = json!({
    "model": "command-r-plus",
    "messages": [
    {"role": "developer", "content": "You are a coding assistant."},
    {"role": "user", "content": "Hi"},
    ],
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert!(out.get("preamble").is_none());
    let messages = out["messages"].as_array().expect("messages");
    assert_eq!(messages.len(), 2);
    assert_eq!(
        messages[0],
        json!({"role": "system", "content": "You are a coding assistant."})
    );
    assert_eq!(messages[1]["role"], "user");
    assert!(
        messages
            .iter()
            .all(|m| m.get("role").and_then(|r| r.as_str()) != Some("developer")),
        "developer message must not appear in messages"
    );
}

#[test]
fn request_keeps_system_and_developer_order() {
    let p = CohereProvider;
    let req = json!({
    "model": "command-r-plus",
    "messages": [
    {"role": "system", "content": "Be concise."},
    {"role": "developer", "content": "Prefer Rust."},
    {"role": "user", "content": "Hi"},
    ],
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert!(out.get("preamble").is_none());
    assert_eq!(
        out["messages"],
        json!([
            {"role": "system", "content": "Be concise."},
            {"role": "system", "content": "Prefer Rust."},
            {"role": "user", "content": "Hi"},
        ])
    );
}

// M7: the OpenAI legacy `function` role is an older spelling of `tool` (the
// function-call response). It must be normalized to a Cohere v2 tool result,
// same as a modern `tool`-role message.
#[test]
fn request_maps_function_role_to_v2_tool_message() {
    let p = CohereProvider;
    let req = json!({
    "model": "command-r-plus",
    "messages": [
    {"role": "user", "content": "Weather?"},
    {
    "role": "assistant",
    "content": null,
    "tool_calls": [{
    "id": "tc_2",
    "type": "function",
    "function": {"name": "get_weather", "arguments": "{\"location\":\"Paris\"}"},
    }],
    },
    {
    "role": "function",
    "name": "get_weather",
    "tool_call_id": "tc_2",
    "content": "18C and rainy",
    },
    ],
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    let messages = out["messages"].as_array().expect("messages");
    let tool_msg = messages
        .iter()
        .find(|m| m.get("role").and_then(|r| r.as_str()) == Some("tool"))
        .expect("function-role message must be normalized to a tool message");
    assert_eq!(tool_msg["tool_call_id"], "tc_2");
    let content = tool_msg["content"].as_array().expect("document content");
    assert_eq!(content[0]["type"], "document");
    assert_eq!(content[0]["document"]["data"], "18C and rainy");
    assert!(
        messages
            .iter()
            .all(|m| m.get("role").and_then(|r| r.as_str()) != Some("function")),
        "function role must not leak through to Cohere"
    );
}

// =============================================================================
// In-body stream error detection + SSE spec compliance.
// =============================================================================

/// An in-body `error` object on a `data:` payload must surface as an
/// `Upstream` error rather than falling through to the unknown-event arm
/// (which would silently drop it).
#[test]
fn stream_in_body_error_surfaces_upstream_error() {
    let p = CohereProvider;
    let raw = r#"data: {"error":{"message":"capacity exhausted"}}"#;
    let err = p
        .transform_stream_chunk(raw)
        .expect_err("in-body error must surface");
    // "capacity" is an overload signal → retryable 503 (not a hard 500), so
    // the orchestrator's retry policy engages.
    assert!(matches!(
        err,
        oxllm_core::Error::Upstream { status: 503, message }
            if message == "capacity exhausted"
    ));
}

/// Cohere's native delta.error is a string; preserve string error messages.
#[test]
fn stream_in_body_error_without_message_defaults() {
    let p = CohereProvider;
    let raw = r#"data: {"error":"provider failed"}"#;
    let err = p
        .transform_stream_chunk(raw)
        .expect_err("in-body error must surface");
    assert!(matches!(
        err,
        oxllm_core::Error::Upstream { status: 500, message }
            if message == "provider failed"
    ));
}

/// A null `error` field must NOT be treated as an error (the chunk is a
/// legitimate event that happens to carry `error: null`).
#[test]
fn stream_null_error_field_is_not_an_error() {
    let p = CohereProvider;
    let raw = r#"data: {"type":"content-delta","delta":{"message":{"content":{"text":"hi"}}},"error":null}"#;
    let out = p
        .transform_stream_chunk(raw)
        .expect("null error must not surface as Upstream error")
        .expect("should produce value");
    assert_eq!(out["choices"][0]["delta"]["content"], "hi");
}

/// An SSE comment line (`:` prefix) must be ignored, yielding no chunk.
#[test]
fn stream_comment_line_is_ignored() {
    let p = CohereProvider;
    assert!(
        p.transform_stream_chunk(": keepalive\n\n")
            .expect("ok")
            .is_none(),
        "comment line must be ignored"
    );
}

/// `event:`/`id:`/`retry:` field lines (with no `data:` line) must be ignored
/// per the SSE spec.
#[test]
fn stream_non_data_field_lines_are_ignored() {
    let p = CohereProvider;
    assert!(
        p.transform_stream_chunk("event: ping\n\n")
            .expect("ok")
            .is_none()
    );
    assert!(
        p.transform_stream_chunk("id: 42\n\n")
            .expect("ok")
            .is_none()
    );
    assert!(
        p.transform_stream_chunk("retry: 1000\n\n")
            .expect("ok")
            .is_none()
    );
}

/// An unknown SSE field line must be ignored rather than surfaced as an error.
#[test]
fn stream_unknown_field_line_is_ignored() {
    let p = CohereProvider;
    assert!(
        p.transform_stream_chunk("foo: bar\n\n")
            .expect("ok")
            .is_none(),
        "unknown field line must be ignored"
    );
    assert!(
        p.transform_stream_chunk("baz:qux\n\n")
            .expect("ok")
            .is_none(),
        "unknown field line without space must be ignored"
    );
}

/// `data:[DONE]` (no space after the colon) must be recognized as the terminal
/// marker, matching `openai_compat/sse.rs`.
#[test]
fn stream_done_without_space_is_recognized() {
    let p = CohereProvider;
    assert!(
        p.transform_stream_chunk("data:[DONE]")
            .expect("ok")
            .is_none(),
        "data:[DONE] must be recognized as terminal"
    );
}

/// `extract_data_line` skips non-data field lines in a multi-line chunk and
/// returns the `data:` payload; a comment/event-only chunk yields `None`.
#[test]
fn extract_data_line_skips_fields_and_returns_payload() {
    // Multi-line: event line skipped, data payload returned.
    assert_eq!(
        extract_data_line("event: message-start\ndata: {\"x\":1}"),
        Some("{\"x\":1}".to_string())
    );
    // data: without space still yields the payload.
    assert_eq!(
        extract_data_line("data:{\"x\":1}"),
        Some("{\"x\":1}".to_string())
    );
    // Bare JSON object is returned.
    assert_eq!(
        extract_data_line("{\"x\":1}"),
        Some("{\"x\":1}".to_string())
    );
    // Comment/event/id/retry-only chunks yield None.
    assert_eq!(extract_data_line(": keepalive"), None);
    assert_eq!(extract_data_line("event: ping"), None);
    assert_eq!(extract_data_line("id: 42"), None);
    assert_eq!(extract_data_line("retry: 1000"), None);
    assert_eq!(extract_data_line("foo: bar"), None);
}

/// Per the SSE spec, multiple `data:` lines in one event must be joined
/// with `\n` before dispatch. Previously `extract_data_line` returned only
/// the first `data:` line, mis-parsing any multi-line payload.
#[test]
fn extract_data_line_concatenates_multi_line_data() {
    assert_eq!(
        extract_data_line("data: {\"a\":\ndata: 1}"),
        Some("{\"a\":\n1}".to_string())
    );
    assert_eq!(
        extract_data_line("data: hello\ndata: world"),
        Some("hello\nworld".to_string())
    );
}

/// A multi-line chunk carrying an `event:` line followed by a `data:` line with
/// an in-body error must still surface the error (the event line is skipped,
/// the data payload's error is detected).
#[test]
fn stream_multiline_event_with_in_body_error_surfaces_upstream_error() {
    let p = CohereProvider;
    let raw = "event: error\ndata: {\"error\":{\"message\":\"overloaded\"}}";
    let err = p
        .transform_stream_chunk(raw)
        .expect_err("in-body error must surface even with an event: line");
    // "overloaded" → retryable 503.
    assert!(matches!(
        err,
        oxllm_core::Error::Upstream { status: 503, message }
            if message == "overloaded"
    ));
}

// =============================================================================
// C50 review fixes: per-stream state, billing/transform correctness.
// =============================================================================

/// HIGH: concurrent streams must not corrupt each other's `id`/`model`/
/// `created`. Two interleaved chunk sequences (A.start → B.start → A.delta)
/// driven through `transform_stream_chunk_stateful` must keep A.delta carrying
/// A's id/model. The old process-wide `Mutex` cell shared state across all
/// streams, so B.start would overwrite A's stashed meta and leak B's
/// conversation id (a cross-tenant metadata leak) into A's chunks.
#[test]
fn stateful_stream_isolates_meta_across_interleaved_streams() {
    let p = CohereProvider;
    let request =
        json!({"model": "command-r-plus", "messages": [{"role": "user", "content": "hi"}]});

    let mut state_a: Option<Box<dyn std::any::Any + Send + Sync>> = None;
    let mut state_b: Option<Box<dyn std::any::Any + Send + Sync>> = None;

    // Stream A: message-start stashes A's meta.
    let a_start = "event: message-start\ndata: {\"type\":\"message-start\",\"id\":\"chatcmpl-A\",\"model\":\"command-a\"}";
    let a_out = p
        .transform_stream_chunk_stateful(a_start, &mut state_a, &request)
        .expect("ok")
        .expect("A start chunk");
    assert_eq!(a_out["id"], "chatcmpl-A");
    assert_eq!(a_out["model"], "command-a");

    // Stream B: message-start stashes B's meta (independent state slot).
    let b_start = "event: message-start\ndata: {\"type\":\"message-start\",\"id\":\"chatcmpl-B\",\"model\":\"command-b\"}";
    let b_out = p
        .transform_stream_chunk_stateful(b_start, &mut state_b, &request)
        .expect("ok")
        .expect("B start chunk");
    assert_eq!(b_out["id"], "chatcmpl-B");

    // Stream A's subsequent content-delta must carry A's id/model, NOT B's.
    let a_delta = "event: content-delta\ndata: {\"type\":\"content-delta\",\"delta\":{\"message\":{\"content\":{\"text\":\"hi\"}}}}";
    let a_delta_out = p
        .transform_stream_chunk_stateful(a_delta, &mut state_a, &request)
        .expect("ok")
        .expect("A delta chunk");
    assert_eq!(
        a_delta_out["id"], "chatcmpl-A",
        "A's delta must keep A's id, not B's"
    );
    assert_eq!(
        a_delta_out["model"], "command-a",
        "A's delta must keep A's model, not B's"
    );
    assert_eq!(a_delta_out["choices"][0]["delta"]["content"], "hi");
}

/// HIGH: a single state slot threaded across two threads (mirroring tokio's
/// worker migration) carries the stashed meta with the task.
#[test]
fn stateful_stream_meta_survives_thread_boundary() {
    let p = CohereProvider;
    let request =
        json!({"model": "command-r-plus", "messages": [{"role": "user", "content": "hi"}]});
    let mut state: Option<Box<dyn std::any::Any + Send + Sync>> = None;

    let start = "event: message-start\ndata: {\"type\":\"message-start\",\"id\":\"chatcmpl-T\",\"model\":\"command-r-plus\"}";
    let start_out = p
        .transform_stream_chunk_stateful(start, &mut state, &request)
        .expect("ok")
        .expect("start");
    let created = start_out["created"].as_u64().expect("created present");

    let delta = "event: content-delta\ndata: {\"type\":\"content-delta\",\"delta\":{\"message\":{\"content\":{\"text\":\"x\"}}}}".to_string();
    let p_clone = p.clone();
    let handle = std::thread::spawn(move || {
        let out = p_clone
            .transform_stream_chunk_stateful(&delta, &mut state, &request)
            .expect("ok")
            .expect("delta on other thread");
        (out, state)
    });
    let (delta_out, _state) = handle.join().expect("thread joined");
    assert_eq!(delta_out["id"], "chatcmpl-T");
    assert_eq!(delta_out["model"], "command-r-plus");
    assert_eq!(delta_out["created"].as_u64(), Some(created));
}

/// MEDIUM: streaming `message-end` must bill on `billed_units` (matching the
/// non-streaming path), falling back to `tokens` only when `billed_units` is
/// absent. The golden `chat_response_001.json` shows they diverge (12/8 vs
/// 14/9), so reading `tokens` metered the same call differently by mode.
#[test]
fn stream_message_end_prefers_billed_units_over_tokens() {
    let p = CohereProvider;
    super::clear_stream_meta_for_test();
    let raw = "event: message-end\ndata: {\"type\":\"message-end\",\"delta\":{\"finish_reason\":\"COMPLETE\",\"usage\":{\"billed_units\":{\"input_tokens\":12,\"output_tokens\":8},\"tokens\":{\"input_tokens\":14,\"output_tokens\":9}}}}";
    let out = p
        .transform_stream_chunk(raw)
        .expect("ok")
        .expect("message-end chunk");
    assert_eq!(out["usage"]["prompt_tokens"], 12);
    assert_eq!(out["usage"]["completion_tokens"], 8);
    assert_eq!(out["usage"]["total_tokens"], 20);
}

/// MEDIUM: when `billed_units` is absent in the stream, fall back to `tokens`.
#[test]
fn stream_message_end_falls_back_to_tokens_without_billed_units() {
    let p = CohereProvider;
    super::clear_stream_meta_for_test();
    let raw = "event: message-end\ndata: {\"type\":\"message-end\",\"delta\":{\"finish_reason\":\"COMPLETE\",\"usage\":{\"tokens\":{\"input_tokens\":7,\"output_tokens\":4}}}}";
    let out = p
        .transform_stream_chunk(raw)
        .expect("ok")
        .expect("message-end chunk");
    assert_eq!(out["usage"]["prompt_tokens"], 7);
    assert_eq!(out["usage"]["completion_tokens"], 4);
    assert_eq!(out["usage"]["total_tokens"], 11);
}

/// MEDIUM: a multi-type embedding response (`embedding_types` set) must emit
/// one OpenAI `data` entry per input with unique sequential indices. The old
/// loop iterated each kind and re-ran `enumerate()` from 0, producing
/// colliding indices `[0,1,0,1]` and a multiplied array.
#[test]
fn embedding_response_multi_type_emits_sequential_indices() {
    let p = CohereProvider;
    let resp = json!({
    "id": "emb-mt",
    "embeddings": {
        "float": [[0.1, 0.2], [0.3, 0.4]],
        "int8": [[1, 2], [3, 4]],
    },
    "meta": {"billed_units": {"input_tokens": 2}},
    });
    let out = EmbeddingProvider::transform_response(&p, &resp, "embed-v3").expect("ok");
    let data = out["data"].as_array().expect("data");
    assert_eq!(
        data.len(),
        2,
        "one entry per input, not multiplied by type count"
    );
    assert_eq!(data[0]["index"], 0);
    assert_eq!(data[1]["index"], 1);
    // Canonical type is `float`.
    assert_eq!(data[0]["embedding"], json!([0.1, 0.2]));
    assert_eq!(data[1]["embedding"], json!([0.3, 0.4]));
}

/// MEDIUM: when `float` is not among the returned types, pick the first type
/// deterministically and still emit sequential indices.
#[test]
fn embedding_response_multi_type_without_float_picks_first() {
    let p = CohereProvider;
    let resp = json!({
    "id": "emb-mt2",
    "embeddings": {
        "int8": [[1], [2]],
        "binary": [[0], [1]],
    },
    });
    let out = EmbeddingProvider::transform_response(&p, &resp, "embed-v3").expect("ok");
    let data = out["data"].as_array().expect("data");
    assert_eq!(data.len(), 2);
    assert_eq!(data[0]["index"], 0);
    assert_eq!(data[1]["index"], 1);
}

/// Legacy fixture compatibility. Official v2 uses start/delta/end instead.
#[test]
fn stream_tool_call_event_reads_delta_tool_calls() {
    let p = CohereProvider;
    super::clear_stream_meta_for_test();
    let raw = "event: tool-call\ndata: {\"type\":\"tool-call\",\"delta\":{\"tool_calls\":[{\"id\":\"call_9\",\"name\":\"get_weather\",\"arguments\":\"{\\\"location\\\":\\\"Paris\\\"}\"}]}}";
    let out = p
        .transform_stream_chunk(raw)
        .expect("ok")
        .expect("tool-call chunk");
    let tc = &out["choices"][0]["delta"]["tool_calls"][0];
    assert_eq!(tc["id"], "call_9", "id must come from delta, not be empty");
    assert_eq!(tc["function"]["name"], "get_weather");
    assert_eq!(tc["function"]["arguments"], "{\"location\":\"Paris\"}");
}

/// MEDIUM: `input_type` for embeddings is overridable via `extra_body`
/// (native pass-through), validated against Cohere's allow-list.
#[test]
fn embedding_request_input_type_override_from_extra_body() {
    let p = CohereProvider;
    let req = json!({
    "model": "embed-v3",
    "input": "what is ai?",
    "extra_body": {"input_type": "search_query"},
    });
    let out = EmbeddingProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["input_type"], "search_query");
}

/// MEDIUM: a top-level `input_type` override wins and is forwarded.
#[test]
fn embedding_request_input_type_override_top_level() {
    let p = CohereProvider;
    let req = json!({
    "model": "embed-v3",
    "input": "doc",
    "input_type": "classification",
    });
    let out = EmbeddingProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["input_type"], "classification");
}

/// MEDIUM: an invalid `input_type` is dropped (falls back to the default)
/// rather than forwarded, so Cohere does not 422.
#[test]
fn embedding_request_input_type_invalid_dropped() {
    let p = CohereProvider;
    let req = json!({
    "model": "embed-v3",
    "input": "doc",
    "input_type": "not-a-real-type",
    });
    let out = EmbeddingProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["input_type"], DEFAULT_EMBED_INPUT_TYPE);
}

/// LOW: a `tool-call-delta` event carrying more than one tool call must emit
/// all of them (each with its own index), not just the first.
#[test]
fn stream_tool_call_delta_emits_all_entries() {
    let p = CohereProvider;
    super::clear_stream_meta_for_test();
    let raw = "event: tool-call-delta\ndata: {\"type\":\"tool-call-delta\",\"delta\":{\"tool_calls\":[{\"id\":\"c1\",\"name\":\"a\",\"arguments\":\"{}\"},{\"id\":\"c2\",\"name\":\"b\",\"arguments\":\"{}\"}]}}";
    let out = p
        .transform_stream_chunk(raw)
        .expect("ok")
        .expect("tool-call-delta chunk");
    let calls = out["choices"][0]["delta"]["tool_calls"]
        .as_array()
        .expect("tool_calls array");
    assert_eq!(calls.len(), 2, "both tool calls must be emitted");
    assert_eq!(calls[0]["index"], 0);
    assert_eq!(calls[0]["function"]["name"], "a");
    assert_eq!(calls[1]["index"], 1);
    assert_eq!(calls[1]["function"]["name"], "b");
}

/// The documented v2 wire shape nests each call under
/// `delta.message.tool_calls` (with the event-level `index`), and a call's
/// arguments arrive split across a `tool-call-start` plus one or more
/// `tool-call-delta` events. This pins the stateless `transform_stream_chunk`
/// path — the stateful host path's multi-call reassembly is covered by
/// `v2_contract_tests` — and reassembles the fragments the way a client must.
/// Dropping the nested pointer or the event-level index fails the assertions.
#[test]
fn stream_v2_tool_call_arguments_assemble_across_deltas() {
    let p = CohereProvider;
    super::clear_stream_meta_for_test();
    let events = [
        json!({"type": "tool-call-start", "index": 2, "delta": {"message": {"tool_calls": {
            "id": "call_1", "type": "function",
            "function": {"name": "get_weather", "arguments": ""}
        }}}}),
        json!({"type": "tool-call-delta", "index": 2, "delta": {"message": {"tool_calls": {
            "function": {"arguments": "{\"loc\":"}
        }}}}),
        json!({"type": "tool-call-delta", "index": 2, "delta": {"message": {"tool_calls": {
            "function": {"arguments": "\"Paris\"}"}
        }}}}),
    ];
    let mut id = String::new();
    let mut name = String::new();
    let mut arguments = String::new();
    for event in events {
        let raw = format!("event: {}\ndata: {event}", event["type"].as_str().unwrap());
        let out = p
            .transform_stream_chunk(&raw)
            .expect("valid v2 event")
            .expect("tool-call chunk");
        let tc = &out["choices"][0]["delta"]["tool_calls"][0];
        assert_eq!(tc["index"], 2, "event-level index must be preserved");
        if let Some(s) = tc.get("id").and_then(Value::as_str) {
            id.push_str(s);
        }
        if let Some(s) = tc.pointer("/function/name").and_then(Value::as_str) {
            name.push_str(s);
        }
        if let Some(s) = tc.pointer("/function/arguments").and_then(Value::as_str) {
            arguments.push_str(s);
        }
    }
    assert_eq!(id, "call_1");
    assert_eq!(name, "get_weather");
    assert_eq!(arguments, "{\"loc\":\"Paris\"}");
}

/// LOW: standard OpenAI sampling params are read from `extra_body` too (the
/// native pass-through convention), matching the Cohere-native params.
#[test]
fn request_standard_params_read_from_extra_body() {
    let p = CohereProvider;
    let req = json!({
    "model": "command-r-plus",
    "messages": [{"role": "user", "content": "Hi"}],
    "extra_body": {"temperature": 0.5, "top_p": 0.8, "top_k": 40, "stop": ["X"], "seed": 7},
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["temperature"], 0.5);
    assert_eq!(out["p"], 0.8);
    assert_eq!(out["k"], 40);
    assert_eq!(out["stop_sequences"], json!(["X"]));
    assert_eq!(out["seed"], 7);
}

/// LOW: a top-level standard param wins over an `extra_body` one.
#[test]
fn request_standard_param_top_level_beats_extra_body() {
    let p = CohereProvider;
    let req = json!({
    "model": "command-r-plus",
    "messages": [{"role": "user", "content": "Hi"}],
    "temperature": 0.3,
    "extra_body": {"temperature": 0.9},
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    assert_eq!(out["temperature"], 0.3);
}

/// LOW: an in-body error whose code/message signals an overload maps to a
/// retryable 503 (engages the orchestrator's retry policy); a generic error
/// stays 500.
#[test]
fn stream_in_body_error_overload_maps_to_retryable_status() {
    let p = CohereProvider;
    let rate_limited = r#"data: {"error":{"message":"too many requests","code":"rate_limited"}}"#;
    let err = p
        .transform_stream_chunk(rate_limited)
        .expect_err("rate-limit surfaces");
    assert!(matches!(
        err,
        oxllm_core::Error::Upstream { status: 503, .. }
    ));

    let generic = r#"data: {"error":{"message":"bad request","code":"invalid"}}"#;
    let err = p
        .transform_stream_chunk(generic)
        .expect_err("generic surfaces");
    assert!(matches!(
        err,
        oxllm_core::Error::Upstream { status: 500, .. }
    ));
}

/// LOW: request-side tool-call mapping stringifies object-form `arguments`
/// (for symmetry with the response path) instead of dropping them to `"{}"`.
#[test]
fn request_tool_call_stringifies_object_arguments() {
    let p = CohereProvider;
    let req = json!({
    "model": "command-r-plus",
    "messages": [
    {"role": "user", "content": "Weather?"},
    {
    "role": "assistant",
    "content": null,
    "tool_calls": [{
    "id": "call_1",
    "type": "function",
    "function": {"name": "get_weather", "arguments": {"location": "Paris"}},
    }],
    },
    ],
    });
    let out = ChatProvider::transform_request(&p, &req, &deployment()).expect("ok");
    let assistant = out["messages"]
        .as_array()
        .expect("messages")
        .iter()
        .find(|m| m.get("role").and_then(|r| r.as_str()) == Some("assistant"))
        .expect("assistant message");
    let args = &assistant["tool_calls"][0]["function"]["arguments"];
    assert!(args.is_string(), "arguments must be stringified");
    let parsed: Value = serde_json::from_str(args.as_str().expect("str")).expect("valid json");
    assert_eq!(parsed["location"], "Paris");
}
