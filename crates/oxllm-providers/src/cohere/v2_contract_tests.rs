//! Fixtures follow https://docs.cohere.com/v2/docs/streaming and the v2 OpenAPI schema.

use super::*;

#[test]
fn streamed_v2_tool_calls_reassemble_without_missing_ids_or_duplicate_arguments() {
    let mut state = None;
    let request = json!({"model": "command-r7b"});
    let events = [
        json!({"type": "message-start", "id": "stream-1"}),
        json!({"type": "tool-call-start", "index": 0, "delta": {"message": {"tool_calls": {
            "id": "call-weather", "type": "function", "function": {"name": "weather", "arguments": ""}
        }}}}),
        json!({"type": "tool-call-delta", "index": 0, "delta": {"message": {"tool_calls": {
            "function": {"arguments": "{\"city\":"}
        }}}}),
        json!({"type": "tool-call-start", "index": 1, "delta": {"message": {"tool_calls": {
            "id": "call-time", "type": "function", "function": {"name": "time", "arguments": ""}
        }}}}),
        json!({"type": "tool-call-delta", "index": 1, "delta": {"message": {"tool_calls": {
            "function": {"arguments": "{}"}
        }}}}),
        json!({"type": "tool-call-end", "index": 1}),
        json!({"type": "tool-call-delta", "index": 0, "delta": {"message": {"tool_calls": {
            "function": {"arguments": "\"Paris\"}"}
        }}}}),
        json!({"type": "tool-call-end", "index": 0}),
        json!({"type": "message-end", "delta": {"finish_reason": "TOOL_CALL"}}),
    ];
    let mut calls = [
        json!({"id": "", "name": "", "arguments": ""}),
        json!({"id": "", "name": "", "arguments": ""}),
    ];
    for event in events {
        let result = CohereProvider
            .transform_stream_chunk_stateful(
                &format!(
                    "event: {}\ndata: {event}\n\n",
                    event["type"].as_str().unwrap()
                ),
                &mut state,
                &request,
            )
            .expect("valid v2 event");
        if event["type"] == "tool-call-end" {
            assert!(result.is_none(), "end has no payload to append");
            continue;
        }
        let chunk = result.expect("role, tool or finish chunk");
        assert_eq!(chunk["id"], "stream-1");
        assert_eq!(chunk["model"], "command-r7b");
        if event["type"] == "message-end" {
            assert_eq!(chunk["choices"][0]["finish_reason"], "tool_calls");
        }
        if let Some(deltas) = chunk
            .pointer("/choices/0/delta/tool_calls")
            .and_then(Value::as_array)
        {
            for delta in deltas {
                if event["type"] == "tool-call-delta" {
                    assert!(delta.get("id").is_none());
                    assert!(delta["function"].get("name").is_none());
                }
                let call = &mut calls[delta["index"].as_u64().unwrap() as usize];
                for (field, value) in [
                    ("id", delta.get("id")),
                    ("name", delta.pointer("/function/name")),
                    ("arguments", delta.pointer("/function/arguments")),
                ] {
                    if let Some(fragment) = value.and_then(Value::as_str) {
                        call[field] = json!(format!("{}{fragment}", call[field].as_str().unwrap()));
                    }
                }
            }
        }
    }
    assert_eq!(
        calls[0],
        json!({"id": "call-weather", "name": "weather", "arguments": "{\"city\":\"Paris\"}"})
    );
    assert_eq!(
        calls[1],
        json!({"id": "call-time", "name": "time", "arguments": "{}"})
    );
}

#[test]
fn terminal_delta_error_and_failure_reasons_are_not_successful_stops() {
    for (delta, status, text) in [
        (
            json!({"finish_reason": "ERROR", "error": "capacity exhausted"}),
            503,
            "capacity exhausted",
        ),
        (
            json!({"finish_reason": "ERROR", "error": {"message": "failed"}}),
            500,
            "failed",
        ),
        (json!({"finish_reason": "TIMEOUT"}), 504, "TIMEOUT"),
        (
            json!({"finish_reason": "ERROR", "error": null}),
            500,
            "ERROR",
        ),
    ] {
        let event = json!({"type": "message-end", "delta": delta});
        let error = CohereProvider
            .transform_stream_chunk(&format!("data: {event}"))
            .unwrap_err();
        assert!(
            matches!(error, oxllm_core::Error::Upstream { status: got, message } if got == status && message.contains(text))
        );
    }
    let event = r#"data: {"type":"message-end","delta":{"finish_reason":"COMPLETE","error":null}}"#;
    assert!(
        CohereProvider
            .transform_stream_chunk(event)
            .unwrap()
            .is_some()
    );
}

#[test]
fn v2_tool_choice_uses_documented_enum_or_is_omitted() {
    let deployment = Deployment::default();
    for (choice, expected) in [
        ("auto", None),
        ("required", Some("REQUIRED")),
        ("REQUIRED", Some("REQUIRED")),
        ("none", Some("NONE")),
        ("NONE", Some("NONE")),
    ] {
        let request = json!({"messages": [], "tool_choice": choice,
            "tools": [{"type": "function", "function": {"name": "weather", "parameters": {"type": "object"}}}]});
        let out = ChatProvider::transform_request(&CohereProvider, &request, &deployment).unwrap();
        assert_eq!(out.get("tool_choice").and_then(Value::as_str), expected);
    }
    let request = json!({"messages": [], "tool_choice": {"type": "function", "function": {"name": "missing"}}});
    assert!(ChatProvider::transform_request(&CohereProvider, &request, &deployment).is_err());
}

#[test]
fn removed_v1_fields_do_not_reappear_via_extra_body() {
    let removed = json!({
        "preamble": "old", "connectors": [], "search_queries_only": true,
        "conversation_id": "old", "prompt_truncation": "AUTO", "n": 2,
        "num_generations": 2, "force_single_step": true
    });
    for from_extra in [false, true] {
        let mut request = if from_extra {
            json!({"extra_body": removed})
        } else {
            removed.clone()
        };
        request["messages"] = json!([
            {"role": "developer", "content": [{"type": "text", "text": "Use v2."}]},
            {"role": "user", "content": "Hi"}
        ]);
        let out =
            ChatProvider::transform_request(&CohereProvider, &request, &Deployment::default())
                .unwrap();
        for key in removed.as_object().unwrap().keys() {
            assert!(out.get(key).is_none(), "removed parameter: {key}");
        }
        assert_eq!(
            out["messages"][0],
            json!({"role": "system", "content": "Use v2."})
        );
    }
}

#[test]
fn v2_citation_object_preserves_sources() {
    let citation = json!({"start": 0, "end": 5, "text": "Paris", "sources": [{"type": "document", "id": "doc:1"}]});
    let event = json!({"type": "citation-start", "index": 0, "delta": {"message": {"citations": citation}}});
    let out = CohereProvider
        .transform_stream_chunk(&format!("data: {event}"))
        .unwrap()
        .unwrap();
    assert_eq!(out["citations"], json!([citation]));
}

#[test]
fn supported_parameter_filter_keeps_extra_body_without_forwarding_unknown_contents() {
    let mut deployment = Deployment::default();
    deployment.oxllm_params.model = "cohere/*".into();
    let mut request = json!({
        "model": "cohere/command-r7b",
        "messages": [{"role": "user", "content": "look up"}],
        "temperature": 0.4,
        "unsupported": true,
        "extra_body": {
            "temperature": 0.9, "top_k": 8, "top_p": 0.5,
            "documents": [{"id": "doc-1", "data": {"text": "reference"}}],
            "tools": [{"type": "function", "function": {
                "name": "lookup", "parameters": {"type": "object"}
            }}],
            "tool_choice": "required", "preamble": "removed", "unsupported": true
        }
    });
    let params = CohereProvider.supported_params().unwrap();
    request
        .as_object_mut()
        .unwrap()
        .retain(|key, _| params.contains(&key.as_str()));
    let body = ChatProvider::transform_request(&CohereProvider, &request, &deployment).unwrap();
    assert_eq!(body["model"], "command-r7b");
    assert_eq!(body["temperature"], 0.4);
    assert_eq!(body["k"], 8);
    assert_eq!(body["p"], 0.5);
    assert_eq!(body["documents"], request["extra_body"]["documents"]);
    assert_eq!(body["tools"][0]["function"]["name"], "lookup");
    assert_eq!(body["tool_choice"], "REQUIRED");
    for key in ["extra_body", "unsupported", "preamble", "top_k", "top_p"] {
        assert!(body.get(key).is_none(), "{key}");
    }
}

#[test]
fn multifield_sse_uses_all_supported_line_endings_and_preserves_error_dispatch() {
    for ending in ["\r", "\n", "\r\n"] {
        let mut state = None;
        let request = json!({"model": "command-r7b"});
        let start = format!(
            ": keepalive{ending}event: message-start{ending}id: 1{ending}data: {{\"type\":\"message-start\",{ending}data: \"id\":\"stream-cr\"}}{ending}{ending}"
        );
        let first = CohereProvider
            .transform_stream_chunk_stateful(&start, &mut state, &request)
            .unwrap()
            .unwrap();
        let delta = format!(
            "event: content-delta{ending}retry: 1000{ending}data: {{\"type\":\"content-delta\",{ending}data: \"delta\":{{\"message\":{{\"content\":{{\"text\":\"hello\"}}}}}}}}{ending}{ending}"
        );
        let next = CohereProvider
            .transform_stream_chunk_stateful(&delta, &mut state, &request)
            .unwrap()
            .unwrap();
        assert_eq!(next["id"], first["id"]);
        assert_eq!(next["model"], "command-r7b");
        assert_eq!(next["choices"][0]["delta"]["content"], "hello");
        assert!(
            CohereProvider
                .transform_stream_chunk_stateful(
                    &format!("event: done{ending}data: [DONE]{ending}{ending}"),
                    &mut state,
                    &request,
                )
                .unwrap()
                .is_none()
        );
        let error = format!(
            "event: message-end{ending}data: {{\"type\":\"message-end\",{ending}data: \"delta\":{{\"finish_reason\":\"ERROR\",\"error\":\"capacity exhausted\"}}}}{ending}{ending}"
        );
        assert!(matches!(
            CohereProvider.transform_stream_chunk_stateful(&error, &mut state, &request),
            Err(oxllm_core::Error::Upstream { status: 503, .. })
        ));
    }
}

#[test]
fn configured_endpoint_bases_preserve_versions_prefixes_and_queries() {
    for (base, prefix) in [
        (
            "https://proxy.example/v2/chat/?token=a%2Fb",
            "https://proxy.example/v2",
        ),
        (
            "https://proxy.example/v1/embed?token=a%2Fb",
            "https://proxy.example/v1",
        ),
        (
            "https://proxy.example/v2/rerank?token=a%2Fb",
            "https://proxy.example/v2",
        ),
        (
            "https://proxy.example/v1/?token=a%2Fb",
            "https://proxy.example/v1",
        ),
        (
            "https://proxy.example/custom/?token=a%2Fb",
            "https://proxy.example/custom",
        ),
        (
            "https://proxy.example/chat?token=a%2Fb",
            "https://proxy.example",
        ),
        (
            "https://proxy.example/路径/chat?token=a%2Fb",
            "https://proxy.example/路径",
        ),
        (
            "https://proxy.example/chatty?token=a%2Fb",
            "https://proxy.example/chatty",
        ),
    ] {
        let mut deployment = Deployment::default();
        deployment.oxllm_params.api_base = Some(base.into());
        assert_eq!(
            CohereProvider.endpoint_url(&deployment),
            format!("{prefix}/chat?token=a%2Fb")
        );
        assert_eq!(
            CohereProvider.embedding_url(&deployment),
            format!("{prefix}/embed?token=a%2Fb")
        );
        assert_eq!(
            CohereProvider.rerank_url(&deployment),
            format!("{prefix}/rerank?token=a%2Fb")
        );
    }
    let mut deployment = Deployment::default();
    deployment.oxllm_params.api_base = Some("https://proxy.example/?token=1#docs".into());
    assert_eq!(
        CohereProvider.endpoint_url(&deployment),
        "https://proxy.example/chat?token=1#docs"
    );
    assert_eq!(
        CohereProvider.embedding_url(&deployment),
        "https://proxy.example/embed?token=1#docs"
    );
    assert_eq!(
        CohereProvider.rerank_url(&deployment),
        "https://proxy.example/v1/rerank?token=1#docs"
    );
}

#[test]
fn wildcard_custom_root_keeps_reverse_proxy_paths_and_native_request_bodies() {
    let mut deployment = Deployment {
        model_name: "*".into(),
        ..Default::default()
    };
    deployment.oxllm_params.api_base = Some("http://127.0.0.1:49999".into());
    let chat_request = json!({
        "model": "cohere/command-r-plus",
        "messages": [{"role": "user", "content": "hello"}]
    });
    for model in ["cohere/command-r-plus", "cohere/*"] {
        deployment.oxllm_params.model = model.into();
        assert_eq!(
            CohereProvider.endpoint_url(&deployment),
            "http://127.0.0.1:49999/chat"
        );
        let chat =
            ChatProvider::transform_request(&CohereProvider, &chat_request, &deployment).unwrap();
        assert_eq!(chat["model"], "command-r-plus");
        assert_eq!(chat["messages"], chat_request["messages"]);
        assert!(chat.get("message").is_none());
    }
    let embedding = EmbeddingProvider::transform_request(
        &CohereProvider,
        &json!({"model": "cohere/embed-v4.0", "input": ["document"]}),
        &deployment,
    )
    .unwrap();
    assert_eq!(
        CohereProvider.embedding_url(&deployment),
        "http://127.0.0.1:49999/embed"
    );
    assert_eq!(embedding["model"], "embed-v4.0");
    assert_eq!(embedding["texts"], json!(["document"]));
    assert_eq!(embedding["input_type"], "search_document");

    let rerank = RerankProvider::transform_request(
        &CohereProvider,
        &json!({"model": "cohere/rerank-v3.5", "query": "question", "documents": ["answer"]}),
        &deployment,
    )
    .unwrap();
    assert_eq!(
        CohereProvider.rerank_url(&deployment),
        "http://127.0.0.1:49999/v1/rerank"
    );
    assert_eq!(rerank["model"], "rerank-v3.5");
    assert_eq!(rerank["query"], "question");
    assert_eq!(rerank["documents"], json!(["answer"]));
}

#[test]
fn custom_resource_roots_preserve_per_operation_version_defaults() {
    for root in [
        "http://127.0.0.1:49999",
        "http://localhost:49999",
        "http://[::1]:49999",
        "https://proxy.example",
        "https://cohere.com",
        "https://api.cohere.com.proxy.example",
        "https://proxy.api.cohere.com",
        "https://api.cohere.com:8443",
        "http://api.cohere.ai:8080",
        "https://api.cohere.com@proxy.example",
    ] {
        let mut deployment = Deployment::default();
        deployment.oxllm_params.api_base = Some(format!("{root}/?tenant=a%2Fb#docs"));
        assert_eq!(
            CohereProvider.endpoint_url(&deployment),
            format!("{root}/chat?tenant=a%2Fb#docs")
        );
        assert_eq!(
            CohereProvider.embedding_url(&deployment),
            format!("{root}/embed?tenant=a%2Fb#docs")
        );
        assert_eq!(
            CohereProvider.rerank_url(&deployment),
            format!("{root}/v1/rerank?tenant=a%2Fb#docs")
        );
    }
}

#[test]
fn bare_hosted_resource_roots_infer_only_the_canonical_operation_versions() {
    for root in [
        "https://api.cohere.com",
        "https://api.cohere.ai",
        "https://API.COHERE.COM",
        "https://api.cohere.com:443",
        "http://api.cohere.ai:80",
        "https://api.cohere.com.",
    ] {
        let mut deployment = Deployment::default();
        deployment.oxllm_params.api_base = Some(format!("{root}/?tenant=a%2Fb#docs"));
        assert_eq!(
            CohereProvider.endpoint_url(&deployment),
            format!("{root}/v2/chat?tenant=a%2Fb#docs")
        );
        assert_eq!(
            CohereProvider.embedding_url(&deployment),
            format!("{root}/v2/embed?tenant=a%2Fb#docs")
        );
        assert_eq!(
            CohereProvider.rerank_url(&deployment),
            format!("{root}/v1/rerank?tenant=a%2Fb#docs")
        );
    }
}

#[test]
fn explicit_hosted_and_custom_paths_override_version_inference_and_do_not_duplicate_endpoints() {
    for (base, prefix) in [
        ("https://api.cohere.com/v1/", "https://api.cohere.com/v1"),
        (
            "https://api.cohere.com/v2/chat/",
            "https://api.cohere.com/v2",
        ),
        ("https://api.cohere.ai/chat/", "https://api.cohere.ai"),
        ("https://api.cohere.com/embed/", "https://api.cohere.com"),
        ("https://api.cohere.com/rerank/", "https://api.cohere.com"),
        (
            "https://api.cohere.com/tenant/",
            "https://api.cohere.com/tenant",
        ),
        (
            "https://proxy.example/api/v3/embed/",
            "https://proxy.example/api/v3",
        ),
        (
            "https://proxy.example/路径/rerank/",
            "https://proxy.example/路径",
        ),
        (
            "https://proxy.example/chatty/",
            "https://proxy.example/chatty",
        ),
    ] {
        let mut deployment = Deployment::default();
        deployment.oxllm_params.api_base = Some(format!("{base}?tenant=a%2Fb#docs"));
        assert_eq!(
            CohereProvider.endpoint_url(&deployment),
            format!("{prefix}/chat?tenant=a%2Fb#docs")
        );
        assert_eq!(
            CohereProvider.embedding_url(&deployment),
            format!("{prefix}/embed?tenant=a%2Fb#docs")
        );
        assert_eq!(
            CohereProvider.rerank_url(&deployment),
            format!("{prefix}/rerank?tenant=a%2Fb#docs")
        );
    }
}

#[test]
fn endpoint_helper_preserves_historical_custom_root_chat_embed_and_rerank_paths() {
    for root in ["http://127.0.0.1:49999", "https://proxy.example"] {
        for trailing_slash in ["", "/"] {
            let mut deployment = Deployment::default();
            deployment.oxllm_params.api_base = Some(format!("{root}{trailing_slash}?tenant=a%2Fb"));
            for (endpoint, default_base, default_version, suffix) in [
                ("chat", DEFAULT_COHERE_BASE, "v2", "/chat"),
                ("embed", DEFAULT_COHERE_BASE, "v2", "/embed"),
                ("rerank", DEFAULT_COHERE_V1_BASE, "v1", "/v1/rerank"),
            ] {
                assert_eq!(
                    cohere_endpoint_url(&deployment, default_base, default_version, endpoint),
                    format!("{root}{suffix}?tenant=a%2Fb")
                );
            }
        }
    }
}

#[test]
fn endpoint_helper_preserves_explicit_rerank_routes_without_duplicate_versions_or_operations() {
    for (base, expected) in [
        (
            "http://127.0.0.1:49999/rerank",
            "http://127.0.0.1:49999/rerank",
        ),
        (
            "http://127.0.0.1:49999/v1/rerank/",
            "http://127.0.0.1:49999/v1/rerank",
        ),
        (
            "http://127.0.0.1:49999/v2/rerank",
            "http://127.0.0.1:49999/v2/rerank",
        ),
        (
            "http://127.0.0.1:49999/v2/",
            "http://127.0.0.1:49999/v2/rerank",
        ),
        (
            "http://127.0.0.1:49999/proxy/",
            "http://127.0.0.1:49999/proxy/rerank",
        ),
        (
            "http://127.0.0.1:49999/proxy/rerank/",
            "http://127.0.0.1:49999/proxy/rerank",
        ),
        (
            "https://api.cohere.com/rerank",
            "https://api.cohere.com/rerank",
        ),
        (
            "https://api.cohere.com/v2/rerank",
            "https://api.cohere.com/v2/rerank",
        ),
    ] {
        let mut deployment = Deployment::default();
        deployment.oxllm_params.api_base = Some(format!("{base}?tenant=a%2Fb#docs"));
        assert_eq!(
            cohere_endpoint_url(&deployment, DEFAULT_COHERE_V1_BASE, "v1", "rerank"),
            format!("{expected}?tenant=a%2Fb#docs")
        );
    }
}
