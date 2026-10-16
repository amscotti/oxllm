use super::*;
use oxllm_core::deployment::OxllmParams;

fn deployment(model: &str, base: &str) -> Deployment {
    Deployment {
        oxllm_params: OxllmParams {
            model: format!("azure_ai/{model}"),
            api_base: Some(base.into()),
            ..Default::default()
        },
        ..Default::default()
    }
}

#[test]
fn foundry_urls_accept_roots_full_endpoints_and_existing_query_versions() {
    let provider = AzureAiProvider;
    for base in [
        "https://resource.services.ai.azure.com",
        "https://resource.services.ai.azure.com/models",
        "https://resource.services.ai.azure.com/models/chat/completions/",
    ] {
        let deployment = deployment("grok-4-fast-reasoning", base);
        assert_eq!(
            provider.endpoint_url(&deployment),
            "https://resource.services.ai.azure.com/models/chat/completions"
        );
        assert_eq!(
            provider.embedding_url(&deployment),
            "https://resource.services.ai.azure.com/models/embeddings"
        );
    }
    let mut configured = deployment(
        "cohere-deployment",
        "https://resource.services.ai.azure.com/models/embeddings?api-version=existing&trace=kept#fragment",
    );
    configured.oxllm_params.api_version = Some("new-version".into());
    assert_eq!(
        provider.embedding_url(&configured),
        "https://resource.services.ai.azure.com/models/embeddings?api-version=existing&trace=kept#fragment"
    );
    configured.oxllm_params.api_base =
        Some("https://resource.services.ai.azure.com/models?trace=kept".into());
    configured.oxllm_params.api_version = Some("2024-05-01-preview".into());
    assert_eq!(
        provider.endpoint_url(&configured),
        "https://resource.services.ai.azure.com/models/chat/completions?trace=kept&api-version=2024-05-01-preview"
    );
    let legacy = deployment("grok", "https://model.inference.ai.azure.com");
    assert_eq!(
        provider.endpoint_url(&legacy),
        "https://model.inference.ai.azure.com/chat/completions"
    );
    let lookalike = deployment("grok", "https://resource.services.ai.azure.com.evil.test");
    assert_eq!(
        provider.endpoint_url(&lookalike),
        "https://resource.services.ai.azure.com.evil.test/chat/completions"
    );
}

#[test]
fn claude_uses_native_shape_and_anthropic_auth_without_a_duplicate_azure_key() {
    let provider = AzureAiProvider;
    for base in [
        "https://resource.services.ai.azure.com",
        "https://resource.services.ai.azure.com/models",
        "https://resource.services.ai.azure.com/anthropic",
        "https://resource.services.ai.azure.com/anthropic/v1/messages",
    ] {
        let deployment = deployment("claude-haiku-4-5", base);
        assert_eq!(
            provider.endpoint_url(&deployment),
            "https://resource.services.ai.azure.com/anthropic/v1/messages"
        );
        let body = ChatProvider::transform_request(
            &provider,
            &json!({"model": "alias", "messages": [
            {"role": "system", "content": "be brief"}, {"role": "user", "content": "hello"}
        ], "max_tokens": 20, "stream": true}),
            &deployment,
        )
        .unwrap();
        assert_eq!(body["model"], "claude-haiku-4-5");
        assert!(body.get("system").is_some());
        assert_eq!(body["messages"].as_array().unwrap().len(), 1);
        assert_eq!(body["stream"], true);
        let mut headers = HashMap::new();
        ChatProvider::apply_auth(
            &provider,
            &deployment,
            &ResolvedCredentials::from_key("key"),
            &mut headers,
        );
        assert_eq!(headers["x-api-key"], "key");
        assert_eq!(headers["anthropic-version"], "2023-06-01");
        assert!(!headers.contains_key("api-key"));
        assert!(!headers.contains_key("authorization"));
    }
}

#[test]
fn private_claude_deployment_names_can_be_selected_by_native_endpoint() {
    let deployment = deployment(
        "private-deployment",
        "https://resource.services.ai.azure.com/anthropic/v1/messages?api-version=version",
    );
    let body = ChatProvider::transform_request(
        &AzureAiProvider,
        &json!({"messages": [{"role": "user", "content": "hello"}]}),
        &deployment,
    )
    .unwrap();
    assert_eq!(body["model"], "private-deployment");
    assert!(body.get("max_tokens").is_some());
    assert_eq!(
        AzureAiProvider.endpoint_url(&deployment),
        "https://resource.services.ai.azure.com/anthropic/v1/messages?api-version=version"
    );
}

#[test]
fn compatible_messages_drop_native_extensions_without_mutating_fallback_input() {
    let request = json!({"messages": [
        {"role": "user", "content": [{"type": "text", "text": "one"}, {"type": "text", "text": "two"}], "cache_control": {"type": "ephemeral"}},
        {"role": "assistant", "content": "reply", "thinking_blocks": [{"type": "thinking", "thinking": "thought", "signature": "sig"}], "reasoning_content": "thought"}
    ]});
    let body = ChatProvider::transform_request(
        &AzureAiProvider,
        &request,
        &deployment("grok", "https://resource.services.ai.azure.com/models"),
    )
    .unwrap();
    assert_eq!(body["model"], "grok");
    assert_eq!(body["messages"][0]["content"], "one\ntwo");
    assert!(body["messages"][0].get("cache_control").is_none());
    assert!(body["messages"][1].get("thinking_blocks").is_none());
    assert!(request["messages"][1].get("thinking_blocks").is_some());
}

#[test]
fn compatible_text_arrays_strip_anthropic_controls_before_collapsing() {
    let request = json!({"messages":[
        {"role":"user", "content":[
            {"type":"text", "text":"one", "cache_control":{"type":"ephemeral", "ttl":"1h"}},
            {"type":"text", "text":"two", "cache_control":null}
        ]},
        {"role":"user", "content":[{"type":"text", "text":"null marker", "prompt_cache_breakpoint":null}]},
        {"role":"user", "content":[]}
    ]});
    let original = request.clone();
    let body = ChatProvider::transform_request(
        &AzureAiProvider,
        &request,
        &deployment("grok", "https://resource.services.ai.azure.com/models"),
    )
    .unwrap();
    assert_eq!(body["messages"][0]["content"], "one\ntwo");
    assert_eq!(body["messages"][1]["content"], "null marker");
    assert_eq!(body["messages"][2]["content"], json!([]));
    assert_eq!(request, original);
}

#[test]
fn compatible_mixed_and_text_only_content_preserves_native_openai_marks() {
    let marker = json!({"mode":"explicit", "ttl":"30m", "custom":"kept"});
    let request = json!({"prompt_cache_options":{"mode":"explicit", "ttl":"30m"}, "messages":[
        {"role":"user", "content":[
            {"type":"text", "text":"look", "cache_control":{"type":"ephemeral"}, "prompt_cache_breakpoint":marker},
            {"type":"image_url", "image_url":{"url":"https://example.test/image.png"}, "cache_control":{"type":"ephemeral"}}
        ]},
        {"role":"user", "content":[
            {"type":"text", "text":"one", "cache_control":{"type":"ephemeral"}, "prompt_cache_breakpoint":marker},
            {"type":"text", "text":"two", "cache_control":{"type":"ephemeral"}}
        ]}
    ]});
    let original = request.clone();
    let body = ChatProvider::transform_request(
        &AzureAiProvider,
        &request,
        &deployment("gpt-6-sol", "https://resource.services.ai.azure.com/models"),
    )
    .unwrap();
    for message in body["messages"].as_array().unwrap() {
        let blocks = message["content"]
            .as_array()
            .expect("native markers require block content");
        assert_eq!(blocks[0]["prompt_cache_breakpoint"], marker);
        assert!(
            blocks
                .iter()
                .all(|block| block.get("cache_control").is_none())
        );
    }
    assert_eq!(
        body["messages"][0]["content"][1]["image_url"],
        request["messages"][0]["content"][1]["image_url"]
    );
    assert_eq!(
        body["prompt_cache_options"],
        request["prompt_cache_options"]
    );
    assert_eq!(request, original);
}

#[test]
fn foundry_cache_injection_uses_the_sanitized_wire_census() {
    let mut configured = deployment("gpt-6-sol", "https://resource.services.ai.azure.com/models");
    configured.oxllm_params.cache_control_injection_points =
        Some(vec![oxllm_core::deployment::CacheControlInjectionPoint {
            location: "message".into(),
            role: Some("user".into()),
            index: None,
        }]);
    let messages = (0..5).map(|index| json!({"role":"user", "cache_control":{"type":"ephemeral"},
        "content":[{"type":"text", "text":format!("message {index}"), "cache_control":{"type":"ephemeral", "ttl":"1h"}}]
    })).collect::<Vec<_>>();
    let request = json!({"messages":messages, "prompt_cache_options":null});
    let original = request.clone();
    let body = ChatProvider::transform_request(&AzureAiProvider, &request, &configured).unwrap();
    for message in &body["messages"].as_array().unwrap()[..4] {
        assert!(message.get("cache_control").is_none());
        assert!(message["content"][0].get("cache_control").is_none());
        assert_eq!(
            message["content"][0]["prompt_cache_breakpoint"],
            json!({"mode":"explicit"})
        );
    }
    assert_eq!(body["messages"][4]["content"], "message 4");
    assert_eq!(body["prompt_cache_options"], json!({"mode":"implicit"}));
    assert_eq!(request, original);
}

#[test]
fn compatible_tool_controls_are_stripped_without_touching_schema_properties() {
    let request = json!({"messages":[{"role":"user", "content":"hello"}], "tools":[
        {"type":"function", "cache_control":{"type":"ephemeral"}, "function":{
            "name":"lookup", "cache_control":{"type":"ephemeral", "ttl":"1h"},
            "parameters":{"type":"object", "properties":{
                "cache_control":{"type":"string"}, "prompt_cache_breakpoint":{"type":"object"}
            }}
        }}
    ]});
    let original = request.clone();
    for model in ["grok", "gpt-6-sol"] {
        let body = ChatProvider::transform_request(
            &AzureAiProvider,
            &request,
            &deployment(model, "https://resource.services.ai.azure.com/models"),
        )
        .unwrap();
        assert!(body["tools"][0].get("cache_control").is_none());
        assert!(body["tools"][0]["function"].get("cache_control").is_none());
        assert_eq!(
            body["tools"][0]["function"]["parameters"],
            request["tools"][0]["function"]["parameters"]
        );
    }
    assert_eq!(request, original);
}

#[test]
fn embeddings_are_openai_shaped_and_keep_dimensions_and_usage() {
    let deployment = deployment(
        "cohere-deployment",
        "https://resource.services.ai.azure.com/models",
    );
    let request = json!({"model": "alias", "input": ["one", "two"], "dimensions": 512, "encoding_format": "float"});
    let body =
        EmbeddingProvider::transform_request(&AzureAiProvider, &request, &deployment).unwrap();
    assert_eq!(
        body,
        json!({"model": "cohere-deployment", "input": ["one", "two"], "dimensions": 512, "encoding_format": "float"})
    );
    let response = json!({"data": [{"index": 0, "embedding": [0.1]}], "model": "cohere-deployment", "usage": {"prompt_tokens": 5, "total_tokens": 5}});
    assert_eq!(
        EmbeddingProvider::transform_response(
            &AzureAiProvider,
            &response,
            &deployment.oxllm_params.model
        )
        .unwrap(),
        response
    );
}

#[test]
fn embedding_models_use_the_selected_wire_selector_not_the_vendor_echo() {
    let response = json!({"object":"list","model":"embed-v4.0","data":[{"index":0,"embedding":[0.1]}],"usage":{"prompt_tokens":5,"total_tokens":5,"cost":0.01}});
    for (model, expected) in [
        ("azure_ai/embed-v-4-0", "embed-v-4-0"),
        ("embed-v-4-0", "embed-v-4-0"),
        ("azure_ai/private-embedding", "private-embedding"),
        ("azure_ai/model_router/embed-v-4-0", "embed-v-4-0"),
        ("azure_ai/model-router/embed-v-4-0", "embed-v-4-0"),
    ] {
        let body =
            EmbeddingProvider::transform_response(&AzureAiProvider, &response, model).unwrap();
        assert_eq!(body["model"], expected);
        assert_eq!(body["data"], response["data"]);
        assert_eq!(body["usage"], response["usage"]);
    }
}

#[test]
fn embedding_request_and_response_selectors_remove_router_wrappers_exactly_once() {
    for (model, expected) in [
        ("model_router/embed-v-4-0", "embed-v-4-0"),
        ("model-router/private-embedding", "private-embedding"),
        (
            "model_router/model_router/embedding",
            "model_router/embedding",
        ),
        (
            "model-router/model-router/embedding",
            "model-router/embedding",
        ),
    ] {
        let deployment = deployment(model, "https://resource.services.ai.azure.com/models");
        let request = EmbeddingProvider::transform_request(
            &AzureAiProvider,
            &json!({"input":["text"]}),
            &deployment,
        )
        .unwrap();
        let response = EmbeddingProvider::transform_response(
            &AzureAiProvider,
            &json!({"model":"vendor-echo","data":[]}),
            &deployment.oxllm_params.model,
        )
        .unwrap();
        assert_eq!(request["model"], expected);
        assert_eq!(response["model"], request["model"]);
    }
}

#[test]
fn chat_model_snapshots_are_not_rewritten_by_embedding_model_normalization() {
    let response = json!({"id":"chat","model":"gpt-5.5-2026-04-23","choices":[{"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":4,"total_tokens":9}});
    let body =
        ChatProvider::transform_response(&AzureAiProvider, &response, "azure_ai/gpt-5.5").unwrap();
    assert_eq!(body["model"], response["model"]);
}

#[test]
fn image_embeddings_use_a_distinct_route_and_a_native_allowlisted_body() {
    let mut deployment = deployment(
        "embed-v-4-0",
        "https://resource.services.ai.azure.com/models/embeddings/",
    );
    deployment.oxllm_params.api_version = Some("2024-05-01-preview".into());
    let request = json!({
        "model":"alias", "input":["data:image/png;base64,aGVsbG8="],
        "input_type":"image", "dimensions":512, "encoding_format":"float",
        "user":"test-user", "metadata":{"ignored":true}
    });
    assert_eq!(
        AzureAiProvider.embedding_url_for_request(&deployment, &request),
        "https://resource.services.ai.azure.com/models/images/embeddings?api-version=2024-05-01-preview"
    );
    let body =
        EmbeddingProvider::transform_request(&AzureAiProvider, &request, &deployment).unwrap();
    assert_eq!(
        body,
        json!({
            "model":"embed-v-4-0", "input":[{"image":"data:image/png;base64,aGVsbG8="}],
            "dimensions":512, "encoding_format":"float"
        })
    );
}

#[test]
fn image_embedding_urls_normalize_full_paths_and_preserve_query_versions() {
    let request = json!({"input":"data:image/png;base64,YWJj"});
    for base in [
        "https://resource.services.ai.azure.com",
        "https://resource.services.ai.azure.com/models/",
        "https://resource.services.ai.azure.com/models/embeddings",
        "https://resource.services.ai.azure.com/models/images/embeddings/",
        "https://resource.services.ai.azure.com/models/chat/completions",
    ] {
        let configured = deployment("embed-v-4-0", base);
        assert_eq!(
            AzureAiProvider.embedding_url_for_request(&configured, &request),
            "https://resource.services.ai.azure.com/models/images/embeddings?api-version=2024-05-01-preview",
            "{base}"
        );
        assert_eq!(
            AzureAiProvider.embedding_url(&configured),
            "https://resource.services.ai.azure.com/models/embeddings",
            "text defaults must not change for {base}"
        );
    }
    for base in [
        "https://resource.services.ai.azure.com/models/images/embeddings/?api-version=existing&trace=kept#fragment",
        "https://resource.services.ai.azure.com/models/embeddings?api-version=existing&trace=kept#fragment",
    ] {
        let mut configured = deployment("embed-v-4-0", base);
        configured.oxllm_params.api_version = Some("configured".into());
        assert_eq!(
            AzureAiProvider.embedding_url_for_request(&configured, &request),
            "https://resource.services.ai.azure.com/models/images/embeddings?api-version=existing&trace=kept#fragment"
        );
    }
    let mut configured = deployment(
        "embed-v-4-0",
        "https://gateway.example.test/prefix/models/images/embeddings?trace=kept#fragment",
    );
    configured.oxllm_params.api_version = Some("version with +&=#%?".into());
    assert_eq!(
        AzureAiProvider.embedding_url_for_request(&configured, &request),
        "https://gateway.example.test/prefix/models/images/embeddings?trace=kept&api-version=version%20with%20%2B%26%3D%23%25%3F#fragment"
    );
}

#[test]
fn mixed_embedding_batches_are_disjoint_and_move_native_inputs_without_copying() {
    let configured = deployment(
        "embed-v-4-0",
        "https://resource.services.ai.azure.com/models",
    );
    let request = json!({"model":"alias", "input":[
        "data:image/png;base64,YWJj", "first text", "data:image/jpeg;base64,/9j/", "second text"
    ], "dimensions":256, "encoding_format":"base64", "input_type":"image"});
    let image_pointer = request["input"][0].as_str().unwrap().as_ptr();
    let text_pointer = request["input"][1].as_str().unwrap().as_ptr();
    let EmbeddingRequestPlan::Batched(batches) = AzureAiProvider
        .prepare_requests(request, &configured)
        .unwrap()
    else {
        panic!("mapped plan")
    };
    assert_eq!(batches.len(), 2);
    assert_eq!(batches[0].input_indices, vec![1, 3]);
    assert_eq!(
        batches[0].upstream.body["input"],
        json!(["first text", "second text"])
    );
    assert_eq!(batches[1].input_indices, vec![0, 2]);
    let body = &batches[1].upstream.body;
    assert_eq!(
        body["input"][0]["image"].as_str().unwrap().as_ptr(),
        image_pointer
    );
    assert_eq!(
        batches[0].upstream.body["input"][0]
            .as_str()
            .unwrap()
            .as_ptr(),
        text_pointer
    );
    assert_eq!(
        body,
        &json!({"model":"embed-v-4-0", "input":[
        {"image":"data:image/png;base64,YWJj"}, {"image":"data:image/jpeg;base64,/9j/"}
    ], "dimensions":256, "encoding_format":"base64"})
    );
    assert!(matches!(
        AzureAiProvider.prepare_requests(
            json!({"input":["data:image/png;base64,YWJj", 123]}),
            &configured
        ),
        Err(Error::BadRequest(_))
    ));
}

#[test]
fn scalar_and_pure_embedding_inputs_need_only_one_batch() {
    let configured = deployment(
        "embed-v-4-0",
        "https://resource.services.ai.azure.com/models",
    );
    for input in [
        json!("data:image/png;base64,YWJj"),
        json!(["data:image/png;base64,YWJj"]),
        json!(["data:image/png;base64,YWJj", "data:image/jpeg;base64,/9j/"]),
    ] {
        let request = json!({"input":input});
        let EmbeddingRequestPlan::Batched(batches) = AzureAiProvider
            .prepare_requests(request, &configured)
            .unwrap()
        else {
            panic!("mapped plan")
        };
        assert_eq!(batches.len(), 1);
        assert!(!batches[0].input_indices.is_empty());
        let body = &batches[0].upstream.body;
        assert!(
            body["input"]
                .as_array()
                .unwrap()
                .iter()
                .all(|input| input.get("image").is_some())
        );
    }
    let request = json!({"input":"ordinary text"});
    let EmbeddingRequestPlan::Single(prepared) = AzureAiProvider
        .prepare_requests(request, &configured)
        .unwrap()
    else {
        panic!("single plan")
    };
    let body = prepared.body;
    assert_eq!(body["input"], json!(["ordinary text"]));
    let request = json!({"input":["Dog", "aGVsbG8=", "", "data:image/png;base64,invalid!"]});
    assert!(matches!(
        AzureAiProvider
            .prepare_requests(request.clone(), &configured)
            .unwrap(),
        EmbeddingRequestPlan::Single(_)
    ));
    assert_eq!(
        EmbeddingProvider::transform_request(&AzureAiProvider, &request, &configured).unwrap()["input"],
        request["input"]
    );
}

#[test]
fn cohere_image_selection_uses_concrete_models_and_explicit_base_model_hints() {
    let request = json!({"model":"azure_ai/embed-v-4-0", "input":["data:image/png;base64,YWJj"]});
    for model in [
        "embed-v-4-0",
        "Cohere-embed-v4",
        "cohere/embed-v4.0",
        "embed-english-v3.0",
        "*",
    ] {
        let configured = deployment(model, "https://resource.services.ai.azure.com/models");
        assert!(
            AzureAiProvider
                .embedding_url_for_request(&configured, &request)
                .contains("/images/embeddings?"),
            "{model}"
        );
        let EmbeddingRequestPlan::Batched(batches) = AzureAiProvider
            .prepare_requests(request.clone(), &configured)
            .unwrap()
        else {
            panic!("mapped plan")
        };
        assert_eq!(batches.len(), 1);
        assert!(
            !batches[0].upstream.body["model"]
                .as_str()
                .unwrap()
                .contains('*')
        );
    }
    let mut configured = deployment(
        "private-deployment",
        "https://resource.services.ai.azure.com/models",
    );
    configured.model_info = Some(oxllm_core::deployment::ModelInfo {
        base_model: Some("cohere/embed-v4.0".into()),
        ..Default::default()
    });
    assert!(
        AzureAiProvider
            .embedding_url_for_request(&configured, &request)
            .contains("/images/embeddings?")
    );
    assert_eq!(
        EmbeddingProvider::transform_request(&AzureAiProvider, &request, &configured).unwrap()["model"],
        "private-deployment"
    );
    // The response side uses the same selector; the base-model hint only
    // selects the route and never leaks into attribution.
    assert_eq!(
        EmbeddingProvider::transform_response(
            &AzureAiProvider,
            &json!({"model":"embed-v4.0","data":[]}),
            &configured.oxllm_params.model,
        )
        .unwrap()["model"],
        "private-deployment"
    );
    configured.model_info = None;
    configured
        .oxllm_params
        .extra
        .insert("base_model".into(), json!("cohere/embed-v4.0"));
    assert!(
        AzureAiProvider
            .embedding_url_for_request(&configured, &request)
            .contains("/images/embeddings?")
    );
    for model in [
        "text-embedding-3-large",
        "custom-embedding",
        "embed-vision-v1",
    ] {
        let configured = deployment(model, "https://resource.services.ai.azure.com/models");
        assert_eq!(
            AzureAiProvider.embedding_url_for_request(&configured, &request),
            AzureAiProvider.embedding_url(&configured)
        );
        assert!(matches!(
            AzureAiProvider
                .prepare_requests(request.clone(), &configured)
                .unwrap(),
            EmbeddingRequestPlan::Single(_)
        ));
        assert_eq!(
            EmbeddingProvider::transform_request(&AzureAiProvider, &request, &configured).unwrap()
                ["input"],
            request["input"]
        );
    }
}

#[test]
fn image_embeddings_retain_documented_auth_and_explicit_credential_overrides() {
    let mut configured = deployment(
        "embed-v-4-0",
        "https://resource.services.ai.azure.com/models",
    );
    for (creds, expected) in [
        (
            ResolvedCredentials::from_key("key"),
            HashMap::from([("api-key".into(), "key".into())]),
        ),
        (
            ResolvedCredentials::AzureAd {
                token: "entra".into(),
            },
            HashMap::from([("authorization".into(), "Bearer entra".into())]),
        ),
    ] {
        let mut headers = HashMap::new();
        EmbeddingProvider::apply_auth(&AzureAiProvider, &configured, &creds, &mut headers);
        assert_eq!(headers, expected);
    }
    configured.oxllm_params.auth_header = Some("x-custom-key".into());
    configured.oxllm_params.auth_scheme = Some("Token".into());
    let mut headers = HashMap::new();
    EmbeddingProvider::apply_auth(
        &AzureAiProvider,
        &configured,
        &ResolvedCredentials::from_key("key"),
        &mut headers,
    );
    assert_eq!(
        headers,
        HashMap::from([("x-custom-key".into(), "Token key".into())])
    );
}

#[test]
fn entra_tokens_and_deployment_auth_overrides_use_one_credential_header() {
    let mut deployment = deployment("grok", "https://resource.services.ai.azure.com/models");
    let mut headers = HashMap::new();
    ChatProvider::apply_auth(
        &AzureAiProvider,
        &deployment,
        &ResolvedCredentials::AzureAd {
            token: "entra".into(),
        },
        &mut headers,
    );
    assert_eq!(
        headers,
        HashMap::from([("authorization".into(), "Bearer entra".into())])
    );
    deployment.oxllm_params.auth_header = Some("x-custom-key".into());
    deployment.oxllm_params.auth_scheme = Some("Token".into());
    headers.clear();
    ChatProvider::apply_auth(
        &AzureAiProvider,
        &deployment,
        &ResolvedCredentials::from_key("key"),
        &mut headers,
    );
    assert_eq!(
        headers,
        HashMap::from([("x-custom-key".into(), "Token key".into())])
    );
}

#[test]
fn native_and_compatible_streams_dispatch_without_losing_state() {
    let mut state = None;
    let request = json!({"messages": [{"role": "user", "content": "hello"}]});
    let start = "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg\",\"model\":\"claude-haiku-4-5\",\"usage\":{\"input_tokens\":3,\"output_tokens\":0}}}";
    AzureAiProvider
        .transform_stream_chunk_stateful(start, &mut state, &request)
        .unwrap();
    let delta = "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hello\"}}";
    let chunk = AzureAiProvider
        .transform_stream_chunk_stateful(delta, &mut state, &request)
        .unwrap()
        .unwrap();
    assert_eq!(chunk["choices"][0]["delta"]["content"], "hello");
    let compatible = "data: {\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hello\"},\"finish_reason\":null}]}";
    assert_eq!(
        AzureAiProvider
            .transform_stream_chunk(compatible)
            .unwrap()
            .unwrap()["choices"][0]["delta"]["content"],
        "hello"
    );
}

#[test]
fn missing_bases_and_unresolved_wildcards_fail_before_inference() {
    let mut deployment = deployment("*", "https://resource.services.ai.azure.com/models");
    assert!(
        ChatProvider::transform_request(&AzureAiProvider, &json!({"messages": []}), &deployment)
            .is_err()
    );
    let body = ChatProvider::transform_request(
        &AzureAiProvider,
        &json!({"model": "azure_ai/grok", "messages": []}),
        &deployment,
    )
    .unwrap();
    assert_eq!(body["model"], "grok");
    deployment.oxllm_params.api_base = None;
    assert!(
        ChatProvider::transform_request(
            &AzureAiProvider,
            &json!({"model": "grok", "messages": []}),
            &deployment
        )
        .is_err()
    );
}
