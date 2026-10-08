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
fn claude_uses_native_shape_and_azure_auth_without_a_duplicate_native_key() {
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
        assert_eq!(headers["api-key"], "key");
        assert_eq!(headers["anthropic-version"], "2023-06-01");
        assert!(!headers.contains_key("x-api-key"));
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
        EmbeddingProvider::transform_response(&AzureAiProvider, &response, "alias").unwrap(),
        response
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
