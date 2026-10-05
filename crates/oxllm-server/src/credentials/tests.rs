use super::*;
use oxllm_core::deployment::OxllmParams;
use oxllm_core::provider::ResolvedCredentials;

fn deployment_with_key(key: Option<&str>) -> Deployment {
    Deployment {
        model_name: "test-model".to_string(),
        oxllm_params: OxllmParams {
            model: "openai/gpt-4o".to_string(),
            api_key: key.map(String::from),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn azure_deployment_with_key(key: Option<&str>) -> Deployment {
    let mut dep = deployment_with_key(key);
    dep.oxllm_params.model = "azure/my-deployment".to_string();
    dep
}

fn vertex_deployment_with_key(key: Option<&str>) -> Deployment {
    let mut dep = deployment_with_key(key);
    dep.oxllm_params.model = "vertex_ai/gemini-1.5-pro".to_string();
    dep
}

#[tokio::test]
async fn static_resolver_returns_api_key_variant() {
    let resolver = StaticKeyResolver;
    let dep = deployment_with_key(Some("sk-test-123"));
    let creds = resolver.resolve(&dep).await.expect("ok");
    match creds {
        ResolvedCredentials::ApiKey { bearer } => assert_eq!(bearer, "sk-test-123"),
        other => panic!("expected ApiKey, got {other:?}"),
    }
}

#[tokio::test]
async fn static_resolver_returns_azure_variant_for_azure_slug() {
    let resolver = StaticKeyResolver;
    let dep = azure_deployment_with_key(Some("azure-test-key"));
    let creds = resolver.resolve(&dep).await.expect("ok");
    match creds {
        ResolvedCredentials::Azure { api_key } => assert_eq!(api_key, "azure-test-key"),
        other => panic!("expected Azure, got {other:?}"),
    }
}

#[tokio::test]
async fn static_resolver_errors_on_missing_key() {
    let resolver = StaticKeyResolver;
    let dep = deployment_with_key(None);
    let err = resolver.resolve(&dep).await.expect_err("should error");
    match err {
        Error::Config(msg) => {
            assert!(msg.contains("missing api_key"), "unexpected message: {msg}");
            // L6: error message must NOT leak the deployment model name.
            assert!(
                !msg.contains("openai/gpt-4o"),
                "error message must not leak the model name, got: {msg}"
            );
        }
        other => panic!("expected Config error, got {other:?}"),
    }
}

#[tokio::test]
async fn static_resolver_allows_ollama_without_key() {
    let resolver = StaticKeyResolver;
    let mut dep = deployment_with_key(None);
    dep.oxllm_params.model = "ollama/llama2".to_string();
    let creds = resolver.resolve(&dep).await.expect("ollama should resolve");
    match creds {
        ResolvedCredentials::ApiKey { bearer } => assert_eq!(bearer, ""),
        other => panic!("expected ApiKey, got {other:?}"),
    }
}

#[test]
fn from_key_produces_api_key_variant() {
    let creds = ResolvedCredentials::from_key("sk-xyz");
    match creds {
        ResolvedCredentials::ApiKey { bearer } => assert_eq!(bearer, "sk-xyz"),
        _ => panic!("from_key must yield ApiKey"),
    }
}

#[test]
fn best_effort_bearer_extracts_from_each_variant() {
    let api_key = ResolvedCredentials::ApiKey { bearer: "a".into() };
    assert_eq!(api_key.best_effort_bearer(), Some("a"));

    let azure = ResolvedCredentials::Azure {
        api_key: "b".into(),
    };
    assert_eq!(azure.best_effort_bearer(), Some("b"));

    let aad = ResolvedCredentials::AzureAd {
        token: "aad".into(),
    };
    assert_eq!(aad.best_effort_bearer(), Some("aad"));

    let aws = ResolvedCredentials::AwsSigV4 {
        access_key: "c".into(),
        secret_key: "d".into(),
        session_token: None,
        region: "us-east-1".into(),
    };
    // L1: AwsSigV4 has no bearer equivalent — sending the access key id as
    // `Authorization: Bearer` against Bedrock always 403s. The host's SigV4
    // signer is the only valid auth path for this credential kind.
    assert_eq!(aws.best_effort_bearer(), None);

    let gcp = ResolvedCredentials::GcpBearer { token: "e".into() };
    assert_eq!(gcp.best_effort_bearer(), Some("e"));

    let presigned = ResolvedCredentials::PreSignedHeaders(std::collections::HashMap::new());
    assert_eq!(presigned.best_effort_bearer(), None);
}

// ---------------------------------------------------------------------------
// VertexResolver
// ---------------------------------------------------------------------------

fn env_only(var: &'static str, val: &'static str) -> impl Fn(&'static str) -> Option<String> {
    move |name: &'static str| {
        if name == var {
            Some(val.to_string())
        } else {
            None
        }
    }
}

fn env_both(gcp: &'static str, vertex: &'static str) -> impl Fn(&'static str) -> Option<String> {
    move |name: &'static str| match name {
        "GCP_ACCESS_TOKEN" => Some(gcp.to_string()),
        "VERTEX_AI_ACCESS_TOKEN" => Some(vertex.to_string()),
        _ => None,
    }
}

fn env_none() -> impl Fn(&'static str) -> Option<String> {
    |_| None
}

#[tokio::test]
async fn vertex_resolver_prefers_api_key_when_set() {
    let resolver = VertexResolver::new();
    let dep = vertex_deployment_with_key(Some("ya29.from-config"));
    let creds = resolver.resolve(&dep).await.expect("ok");
    match creds {
        ResolvedCredentials::GcpBearer { token } => assert_eq!(token, "ya29.from-config"),
        other => panic!("expected GcpBearer, got {other:?}"),
    }
}

#[tokio::test]
async fn vertex_resolver_reads_gcp_access_token_env() {
    let resolver = VertexResolver::with_env(env_only(
        oxllm_providers::vertex_ai::ENV_GCP_ACCESS_TOKEN,
        "ya29.from-env-gcp",
    ));
    let dep = vertex_deployment_with_key(None);
    let creds = resolver.resolve(&dep).await.expect("ok");
    match creds {
        ResolvedCredentials::GcpBearer { token } => assert_eq!(token, "ya29.from-env-gcp"),
        other => panic!("expected GcpBearer, got {other:?}"),
    }
}

#[tokio::test]
async fn vertex_resolver_falls_back_to_vertex_ai_access_token_env() {
    let resolver = VertexResolver::with_env(env_only(
        oxllm_providers::vertex_ai::ENV_VERTEX_ACCESS_TOKEN,
        "ya29.from-env-vertex",
    ));
    let dep = vertex_deployment_with_key(None);
    let creds = resolver.resolve(&dep).await.expect("ok");
    match creds {
        ResolvedCredentials::GcpBearer { token } => assert_eq!(token, "ya29.from-env-vertex"),
        other => panic!("expected GcpBearer, got {other:?}"),
    }
}

#[tokio::test]
async fn vertex_resolver_prefers_gcp_env_over_vertex_env() {
    // When both are set, GCP_ACCESS_TOKEN wins (it is checked first).
    let resolver = VertexResolver::with_env(env_both("ya29.from-gcp", "ya29.from-vertex"));
    let dep = vertex_deployment_with_key(None);
    let creds = resolver.resolve(&dep).await.expect("ok");
    match creds {
        ResolvedCredentials::GcpBearer { token } => assert_eq!(token, "ya29.from-gcp"),
        other => panic!("expected GcpBearer, got {other:?}"),
    }
}

#[tokio::test]
async fn vertex_resolver_errors_when_no_token_available() {
    let resolver = VertexResolver::with_env(env_none());
    let dep = vertex_deployment_with_key(None);
    let err = resolver
        .resolve(&dep)
        .await
        .expect_err("should error without a token");
    match err {
        Error::Config(msg) => {
            assert!(
                msg.contains("missing GCP access token"),
                "unexpected message: {msg}"
            );
            // L6: error message must NOT leak the deployment model name.
            assert!(
                !msg.contains("vertex_ai/gemini-1.5-pro"),
                "error message must not leak the model name, got: {msg}"
            );
        }
        other => panic!("expected Config error, got {other:?}"),
    }
}

#[tokio::test]
async fn vertex_resolver_ignores_empty_env_value() {
    // An empty value is filtered at the resolver level (not just the default
    // lookup) so a custom lookup returning `Some("")` cannot yield a
    // `GcpBearer { token: "" }` that Vertex would reject with 401.
    let resolver = VertexResolver::with_env(env_only(
        oxllm_providers::vertex_ai::ENV_GCP_ACCESS_TOKEN,
        "",
    ));
    let dep = vertex_deployment_with_key(None);
    let err = resolver
        .resolve(&dep)
        .await
        .expect_err("empty token must be rejected, not propagated");
    assert!(
        matches!(err, Error::Config(_)),
        "expected Config error, got {err:?}"
    );
}

#[tokio::test]
async fn vertex_resolver_skips_empty_env_value_and_falls_through() {
    // When the primary env var resolves to an empty string, the resolver
    // must skip it and fall through to the next var in the lookup chain.
    let env = move |name: &'static str| match name {
        oxllm_providers::vertex_ai::ENV_GCP_ACCESS_TOKEN => Some(String::new()),
        oxllm_providers::vertex_ai::ENV_VERTEX_ACCESS_TOKEN => Some("ya29.fallback".into()),
        _ => None,
    };
    let resolver = VertexResolver::with_env(env);
    let dep = vertex_deployment_with_key(None);
    let creds = resolver.resolve(&dep).await.expect("ok");
    match creds {
        ResolvedCredentials::GcpBearer { token } => {
            assert_eq!(
                token, "ya29.fallback",
                "empty primary token must fall through to the secondary var"
            );
        }
        other => panic!("expected GcpBearer, got {other:?}"),
    }
}

#[tokio::test]
async fn vertex_resolver_falls_back_to_env_when_config_api_key_is_empty() {
    // An empty config `api_key` (a realistic outcome of an unresolved
    // `os.environ/VAR` template or a YAML typo) must NOT short-circuit; the
    // resolver should fall through to the env lookup — matching the mantle
    // resolver's `.filter(|s| !s.is_empty())` behavior.
    let resolver = VertexResolver::with_env(env_only(
        oxllm_providers::vertex_ai::ENV_GCP_ACCESS_TOKEN,
        "ya29.from-env",
    ));
    let dep = vertex_deployment_with_key(Some(""));
    let creds = resolver.resolve(&dep).await.expect("ok");
    match creds {
        ResolvedCredentials::GcpBearer { token } => {
            assert_eq!(
                token, "ya29.from-env",
                "empty config api_key must fall back to the env token, not \"\""
            );
        }
        other => panic!("expected GcpBearer, got {other:?}"),
    }
}
