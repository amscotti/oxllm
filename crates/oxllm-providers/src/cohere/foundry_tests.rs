use super::*;
use oxllm_core::deployment::OxllmParams;

#[test]
fn foundry_native_cohere_configuration_uses_the_openai_embedding_contract() {
    let mut deployment = Deployment {
        oxllm_params: OxllmParams {
            model: "cohere/my-embedding-deployment".into(),
            api_base: Some("https://resource.services.ai.azure.com/models".into()),
            api_version: Some("2024-05-01-preview".into()),
            ..Default::default()
        },
        ..Default::default()
    };
    for base in [
        "https://resource.services.ai.azure.com",
        "https://resource.services.ai.azure.com/models",
        "https://resource.services.ai.azure.com/models/embed",
        "https://resource.services.ai.azure.com/models/embeddings",
    ] {
        deployment.oxllm_params.api_base = Some(base.into());
        assert_eq!(
            CohereProvider.embedding_url(&deployment),
            "https://resource.services.ai.azure.com/models/embeddings?api-version=2024-05-01-preview"
        );
        let body = EmbeddingProvider::transform_request(
            &CohereProvider,
            &json!({"model": "alias", "input": ["hello"], "dimensions": 512}),
            &deployment,
        )
        .unwrap();
        assert_eq!(
            body,
            json!({"model": "my-embedding-deployment", "input": ["hello"], "dimensions": 512})
        );
        let mut headers = HashMap::new();
        EmbeddingProvider::apply_auth(
            &CohereProvider,
            &deployment,
            &ResolvedCredentials::from_key("key"),
            &mut headers,
        );
        assert_eq!(headers, HashMap::from([("api-key".into(), "key".into())]));
    }
    let response = json!({"data": [{"index": 0, "embedding": [0.1]}], "model": "my-embedding-deployment", "usage": {"prompt_tokens": 7, "total_tokens": 7}});
    assert_eq!(
        EmbeddingProvider::transform_response(&CohereProvider, &response, "alias").unwrap(),
        response
    );
}

#[test]
fn explicit_gateway_mode_does_not_change_unmarked_custom_cohere_endpoints() {
    let mut deployment = Deployment {
        oxllm_params: OxllmParams {
            model: "cohere/embed-v4".into(),
            api_base: Some("http://localhost/models".into()),
            ..Default::default()
        },
        ..Default::default()
    };
    let request = json!({"input": ["hello"]});
    assert_eq!(
        CohereProvider.embedding_url(&deployment),
        "http://localhost/models/embed"
    );
    assert!(
        EmbeddingProvider::transform_request(&CohereProvider, &request, &deployment)
            .unwrap()
            .get("texts")
            .is_some()
    );
    deployment
        .oxllm_params
        .extra
        .insert("api_type".into(), json!("azure_ai"));
    assert_eq!(
        CohereProvider.embedding_url(&deployment),
        "http://localhost/models/embeddings"
    );
    assert!(
        EmbeddingProvider::transform_request(&CohereProvider, &request, &deployment)
            .unwrap()
            .get("input")
            .is_some()
    );
}
