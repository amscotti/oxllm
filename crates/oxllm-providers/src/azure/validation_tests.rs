use super::*;
use oxllm_core::deployment::OxllmParams;
use serde_json::json;

fn deployment(model: &str, version: &str) -> Deployment {
    Deployment {
        model_name: "public-alias".into(),
        oxllm_params: OxllmParams {
            model: model.into(),
            api_base: Some("https://resource.openai.azure.com".into()),
            api_version: Some(version.into()),
            ..Default::default()
        },
        ..Default::default()
    }
}

#[test]
fn validation_v1_chat_and_embeddings_rewrite_alias_to_deployment_name() {
    for version in ["v1", "preview", "latest"] {
        let deployment = deployment("azure/production-deployment", version);
        let chat_request = json!({"model": "public-alias", "messages": [{
            "role": "user", "content": [{"type": "image_url", "image_url": "data:image/png;base64,AA=="}]
        }]});
        let chat =
            ChatProvider::transform_request(&AzureProvider, &chat_request, &deployment).unwrap();
        let embedding = EmbeddingProvider::transform_request(
            &AzureProvider,
            &json!({
                "model": "public-alias", "input": ["text"], "dimensions": 128,
            }),
            &deployment,
        )
        .unwrap();
        assert_eq!(chat["model"], "production-deployment");
        assert_eq!(
            chat["messages"][0]["content"][0]["image_url"],
            json!({"url": "data:image/png;base64,AA=="})
        );
        assert_eq!(embedding["model"], "production-deployment");
        assert_eq!(embedding["input"], json!(["text"]));
        assert_eq!(embedding["dimensions"], 128);
        assert!(
            AzureProvider
                .endpoint_url(&deployment)
                .contains("/openai/v1/chat/completions?")
        );
        assert_eq!(chat_request["model"], "public-alias");
    }
}

#[test]
fn validation_v1_deployment_resolution_honors_extra_and_wildcards() {
    let mut extra = deployment("gpt-4o", "v1");
    extra
        .oxllm_params
        .extra
        .insert("azure_deployment".into(), json!("extra-deployment"));
    let body = ChatProvider::transform_request(
        &AzureProvider,
        &json!({"model": "alias", "messages": []}),
        &extra,
    )
    .unwrap();
    assert_eq!(body["model"], "extra-deployment");

    let wildcard = deployment("azure/*", "v1");
    for request_model in ["azure/customer-deployment", "customer-deployment"] {
        let request = json!({"model": request_model, "messages": [], "input": "text"});
        assert_eq!(
            ChatProvider::transform_request(&AzureProvider, &request, &wildcard).unwrap()["model"],
            "customer-deployment"
        );
        assert_eq!(
            EmbeddingProvider::transform_request(&AzureProvider, &request, &wildcard).unwrap()["model"],
            "customer-deployment"
        );
    }
    assert!(matches!(
        ChatProvider::transform_request(&AzureProvider, &json!({"messages": []}), &wildcard),
        Err(oxllm_core::Error::BadRequest(_))
    ));
    assert!(matches!(
        ChatProvider::transform_request(
            &AzureProvider,
            &json!({"messages": []}),
            &deployment("azure/", "v1")
        ),
        Err(oxllm_core::Error::Config(_))
    ));
}

#[test]
fn validation_classic_azure_remains_deployment_path_routed() {
    let deployment = deployment("azure/production-deployment", "2024-10-21");
    let request = json!({"model": "public-alias", "messages": [{"role": "user", "content": "hi"}]});
    let out = ChatProvider::transform_request(&AzureProvider, &request, &deployment).unwrap();
    // The classic route selects the deployment in the URL path: the body
    // `model` must be left exactly as the caller sent it (a rewrite to
    // `production-deployment` here would mean the v1 body-model rewrite leaked
    // onto the classic path).
    assert_eq!(
        out["model"], "public-alias",
        "classic route must not rewrite the body model"
    );
    assert_eq!(out, request);
    assert!(
        AzureProvider
            .endpoint_url(&deployment)
            .contains("/deployments/production-deployment/chat/completions")
    );
}

#[test]
fn validation_wildcard_reasoning_limits_use_the_concrete_request_model() {
    for version in ["2024-10-21", "v1"] {
        for configured in ["azure/*", "*"] {
            for requested in ["azure/o3-mini", "o3-mini"] {
                let deployment = deployment(configured, version);
                let request = json!({
                    "model": requested,
                    "messages": [{"role": "system", "content": "rules"}, {"role": "user", "content": "hi"}],
                    "max_tokens": 32, "temperature": 0.3, "top_p": 0.5
                });
                let body =
                    ChatProvider::transform_request(&AzureProvider, &request, &deployment).unwrap();
                assert_eq!(body["max_completion_tokens"], 32);
                assert!(body.get("max_tokens").is_none());
                assert!(body.get("temperature").is_none());
                assert!(body.get("top_p").is_none());
                assert_eq!(body["messages"][0]["role"], "developer");
                if version == "v1" {
                    assert_eq!(body["model"], "o3-mini");
                } else {
                    assert_eq!(body["model"], requested);
                    assert_eq!(
                        AzureProvider.endpoint_url_for_request(&deployment, &request),
                        "https://resource.openai.azure.com/openai/deployments/o3-mini/chat/completions?api-version=2024-10-21"
                    );
                }
            }
        }
    }
}

#[test]
fn validation_classic_deployment_aliases_need_explicit_model_capability_hints() {
    let request = json!({
        "model": "gpt-6-luna", "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 32, "temperature": 0.3, "top_p": 0.5
    });
    for alias in [
        "azure/gpt-6-luna",
        "azure/o3-production",
        "azure/custom-reasoning",
    ] {
        let mut deployment = deployment(alias, "2024-10-21");
        assert_eq!(
            ChatProvider::transform_request(&AzureProvider, &request, &deployment).unwrap(),
            request
        );
        deployment.model_info = Some(oxllm_core::deployment::ModelInfo {
            base_model: Some("openai/o3-mini".into()),
            ..Default::default()
        });
        let body = ChatProvider::transform_request(&AzureProvider, &request, &deployment).unwrap();
        assert_eq!(body["max_completion_tokens"], 32);
        assert!(body.get("temperature").is_none());
        assert!(body.get("top_p").is_none());
    }
    let mut deployment = deployment("gpt-6-luna", "2024-10-21");
    deployment
        .oxllm_params
        .extra
        .insert("azure_deployment".into(), json!("production"));
    let body = ChatProvider::transform_request(&AzureProvider, &request, &deployment).unwrap();
    assert_eq!(body["max_completion_tokens"], 32);
    assert!(body.get("temperature").is_none());
    assert!(
        AzureProvider
            .endpoint_url_for_request(&deployment, &request)
            .contains("/deployments/production/")
    );
}

#[test]
fn validation_native_helper_resolves_classic_wildcards_across_standard_bases_and_operations() {
    for base in [
        "https://resource.openai.azure.com",
        "https://resource.openai.azure.com/openai/",
        "https://resource.openai.azure.com/openai/v1/",
        "https://resource.openai.azure.com/openai/v1/chat/completions/",
    ] {
        for endpoint in [
            "images/generations",
            "audio/speech",
            "audio/transcriptions",
            "chat/completions",
            "embeddings",
        ] {
            let mut deployment = deployment("azure/*", "2024-10-21");
            deployment.oxllm_params.api_base = Some(format!(
                "{base}?api-version=old&token=a%2Fb&%61pi-version=also-old"
            ));
            let url = AzureProvider::native_endpoint_url_for_model(
                &deployment,
                Some("azure/native-deployment"),
                endpoint,
            )
            .unwrap();
            assert_eq!(
                url,
                format!(
                    "https://resource.openai.azure.com/openai/deployments/native-deployment/{endpoint}?token=a%2Fb&api-version=2024-10-21"
                )
            );
        }
    }
}

#[test]
fn validation_native_helper_preserves_explicit_deployments_and_v1_version_query_contracts() {
    for version in ["2024-10-21", "v1", "latest", "preview"] {
        let mut deployment = deployment("azure/*", version);
        deployment
            .oxllm_params
            .extra
            .insert("azure_deployment".into(), json!("production"));
        deployment.oxllm_params.api_base = Some(
            "https://resource.services.ai.azure.com/openai/v1/?token=1&api-version=old#docs".into(),
        );
        for model in [None, Some("public-alias"), Some("azure/other")] {
            let url =
                AzureProvider::native_endpoint_url_for_model(&deployment, model, "audio/speech")
                    .unwrap();
            let expected = if version == "2024-10-21" {
                "https://resource.services.ai.azure.com/openai/deployments/production/audio/speech?token=1&api-version=2024-10-21#docs".to_string()
            } else {
                let wire_version = if version == "latest" { "v1" } else { version };
                format!(
                    "https://resource.services.ai.azure.com/openai/v1/audio/speech?token=1&api-version={wire_version}#docs"
                )
            };
            assert_eq!(url, expected);
        }
        let body = ChatProvider::transform_request(
            &AzureProvider,
            &json!({"model": "public-alias", "messages": []}),
            &deployment,
        )
        .unwrap();
        if version != "2024-10-21" {
            assert_eq!(body["model"], "production");
        }
        assert_eq!(
            deployment.oxllm_params.extra["azure_deployment"],
            "production"
        );
    }
}

#[test]
fn validation_classic_wildcard_aliases_and_invalid_native_selectors_return_bad_request() {
    let deployment = deployment("azure/*", "2024-10-21");
    for model in [
        None,
        Some("public-alias"),
        Some("*"),
        Some("azure/*"),
        Some("azure/"),
        Some(""),
        Some(".."),
        Some("azure/../other"),
        Some("azure/back\\slash"),
    ] {
        assert!(matches!(
            AzureProvider::native_endpoint_url_for_model(&deployment, model, "images/generations"),
            Err(oxllm_core::Error::BadRequest(_))
        ));
        let mut request = json!({"messages": []});
        if let Some(model) = model {
            request["model"] = json!(model);
        }
        assert!(matches!(
            ChatProvider::transform_request(&AzureProvider, &request, &deployment),
            Err(oxllm_core::Error::BadRequest(_))
        ));
        assert_eq!(
            AzureProvider.endpoint_url_for_request(&deployment, &request),
            AZURE_DEPLOYMENT_NAME_NOT_CONFIGURED
        );
    }
}

#[test]
fn validation_native_helper_reports_configuration_errors_and_encodes_deployment_segments() {
    let mut configured = deployment("azure/production", "2024-10-21");
    configured.oxllm_params.api_base = None;
    assert!(matches!(
        AzureProvider::native_endpoint_url_for_model(&configured, None, "audio/speech"),
        Err(oxllm_core::Error::Config(_))
    ));
    for base in [
        "",
        "/openai",
        "ftp://resource.example",
        "https:///missing-host",
        "https://resource.example/has space",
    ] {
        configured.oxllm_params.api_base = Some(base.into());
        assert!(matches!(
            AzureProvider::native_endpoint_url_for_model(&configured, None, "audio/speech"),
            Err(oxllm_core::Error::Config(_))
        ));
    }
    configured.oxllm_params.api_base = Some("https://resource.openai.azure.com/openai".into());
    for model in ["azure/", "azure/..", "azure/../other"] {
        configured.oxllm_params.model = model.into();
        assert!(matches!(
            AzureProvider::native_endpoint_url_for_model(&configured, None, "audio/speech"),
            Err(oxllm_core::Error::Config(_))
        ));
    }
    configured.oxllm_params.model = "azure/safe/name?x=1#fragment".into();
    let url =
        AzureProvider::native_endpoint_url_for_model(&configured, None, "audio/speech").unwrap();
    assert_eq!(
        url,
        "https://resource.openai.azure.com/openai/deployments/safe%2Fname%3Fx=1%23fragment/audio/speech?api-version=2024-10-21"
    );
    for endpoint in [
        "",
        "/audio/speech",
        "audio/../speech",
        "audio/speech?injected=1",
        "audio//speech",
    ] {
        assert!(matches!(
            AzureProvider::native_endpoint_url_for_model(&configured, None, endpoint),
            Err(oxllm_core::Error::BadRequest(_))
        ));
    }
    configured.oxllm_params.api_version = Some("2024&evil=1#fragment".into());
    let url =
        AzureProvider::native_endpoint_url_for_model(&configured, None, "audio/speech").unwrap();
    assert!(url.ends_with("?api-version=2024%26evil%3D1%23fragment"));
}

#[test]
fn validation_explicit_wildcard_deployment_configuration_must_be_concrete() {
    for value in [
        json!(""),
        json!("*"),
        json!("production*"),
        json!(true),
        json!([]),
    ] {
        let mut configured = deployment("azure/*", "2024-10-21");
        configured
            .oxllm_params
            .extra
            .insert("azure_deployment".into(), value);
        assert!(matches!(
            AzureProvider::native_endpoint_url_for_model(
                &configured,
                Some("azure/concrete"),
                "audio/speech"
            ),
            Err(oxllm_core::Error::Config(_))
        ));
    }
}

#[test]
fn validation_malformed_azure_function_calls_return_typed_errors_without_panics() {
    let configured = deployment("azure/production", "2024-10-21");
    for value in [
        json!(true),
        json!("lookup"),
        json!([]),
        json!(123),
        json!({"name": "lookup", "arguments": false}),
        json!({"name": "lookup", "arguments": {}}),
    ] {
        let request = json!({"messages": [{"role": "assistant", "function_call": value}]});
        assert!(matches!(
            ChatProvider::transform_request(&AzureProvider, &request, &configured),
            Err(oxllm_core::Error::BadRequest(_))
        ));
    }
    let request =
        json!({"messages": [{"role": "assistant", "content": "text", "function_call": null}]});
    assert_eq!(
        ChatProvider::transform_request(&AzureProvider, &request, &configured).unwrap(),
        request
    );
}

#[test]
fn validation_prefixed_native_selector_disambiguates_a_bare_public_wildcard_alias() {
    let mut configured = deployment("azure/*", "2024-10-21");
    configured.model_name = "o3-mini".into();
    let request = json!({"model": "azure/o3-mini", "messages": [], "max_tokens": 32});
    let body = ChatProvider::transform_request(&AzureProvider, &request, &configured).unwrap();
    assert_eq!(body["max_completion_tokens"], 32);
    assert!(
        AzureProvider
            .endpoint_url_for_request(&configured, &request)
            .contains("/deployments/o3-mini/")
    );
    assert!(
        AzureProvider::native_endpoint_url_for_model(
            &configured,
            Some("azure/o3-mini"),
            "audio/speech"
        )
        .is_ok()
    );
    assert!(matches!(
        AzureProvider::native_endpoint_url_for_model(&configured, Some("o3-mini"), "audio/speech"),
        Err(oxllm_core::Error::BadRequest(_))
    ));
}
