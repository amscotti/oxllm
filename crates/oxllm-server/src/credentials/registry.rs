//! A registry-shaped dispatch for credential resolvers.
//!
//! Mirrors the pattern used by [`ProviderRegistry`](oxllm_core::ProviderRegistry):
//! each provider slug maps to a dedicated [`CredentialResolver`], with a default
//! fallback ([`StaticKeyResolver`]) for providers that have no specialized
//! resolver.
//!
//! Providers without a specialized resolver (or those whose slug is not
//! registered) fall back to [`StaticKeyResolver`]; the Bedrock and Vertex
//! resolvers register themselves by slug in
//! `state/build.rs::build_credential_resolver` at startup —
//! A single `register` call per provider, not a growing if-chain.

use std::collections::HashMap;
use std::sync::Arc;

use oxllm_core::Deployment;

use super::{AnthropicResolver, CredentialResolver, StaticKeyResolver, WatsonXResolver};

/// A slug → resolver map with a default fallback.
///
/// Adding a new cloud provider is a single `register("bedrock",
/// Arc::new(BedrockResolver))` call from `AppState::build`; the resolver is
/// then transparently dispatched for every `bedrock/*` deployment.
pub struct CredentialResolverRegistry {
    resolvers: HashMap<&'static str, Arc<dyn CredentialResolver>>,
    default: Arc<dyn CredentialResolver>,
}

impl CredentialResolverRegistry {
    /// Build a registry with the static fallback and IBM IAM for both WatsonX
    /// slugs. Other cloud resolvers are added via [`register`](Self::register).
    pub fn new() -> Self {
        let mut registry = Self::with_default(Arc::new(StaticKeyResolver));
        registry.register("anthropic", Arc::new(AnthropicResolver::new()));
        let watsonx: Arc<dyn CredentialResolver> = Arc::new(WatsonXResolver::new());
        registry.register("watsonx", watsonx.clone());
        registry.register("watsonx_text", watsonx);
        registry
    }

    /// Build a registry with an explicit default resolver (used by tests that
    /// want to substitute a mock without registering slugs).
    pub fn with_default(default: Arc<dyn CredentialResolver>) -> Self {
        Self {
            resolvers: HashMap::new(),
            default,
        }
    }

    /// Register `resolver` for the given provider `slug`. Future calls to
    /// [`resolve_for`](Self::resolve_for) for deployments matching `slug` will
    /// return this resolver in preference to the default. Duplicate
    /// registrations overwrite the previous entry.
    pub fn register(&mut self, slug: &'static str, resolver: Arc<dyn CredentialResolver>) {
        self.resolvers.insert(slug, resolver);
    }

    /// Pick the resolver for `deployment` based on its provider slug, falling
    /// back to the default when no specialized resolver is registered.
    pub fn resolve_for(&self, deployment: &Deployment) -> &Arc<dyn CredentialResolver> {
        self.resolvers
            .get(deployment.provider_slug())
            .unwrap_or(&self.default)
    }
}

impl Default for CredentialResolverRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::credentials::CredentialResolver;
    use async_trait::async_trait;
    use oxllm_core::Error;
    use oxllm_core::deployment::OxllmParams;
    use oxllm_core::provider::ResolvedCredentials;

    struct MarkerResolver(&'static str);

    #[async_trait]
    impl CredentialResolver for MarkerResolver {
        async fn resolve(&self, _: &Deployment) -> Result<ResolvedCredentials, Error> {
            Ok(ResolvedCredentials::from_key(self.0))
        }
    }

    fn deployment_for_slug(slug: &str) -> Deployment {
        Deployment {
            model_name: "m".to_string(),
            oxllm_params: OxllmParams {
                model: format!("{slug}/model"),
                api_key: Some("k".to_string()),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn resolve_for_falls_back_to_default_when_unregistered() {
        let registry = CredentialResolverRegistry::new();
        let dep = deployment_for_slug("openai");
        let resolver = registry.resolve_for(&dep);
        let creds = resolver.resolve(&dep).await.expect("ok");
        match creds {
            ResolvedCredentials::ApiKey { bearer } => assert_eq!(bearer, "k"),
            other => panic!("expected ApiKey, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn register_overrides_default_for_slug() {
        let mut registry = CredentialResolverRegistry::new();
        registry.register("bedrock", Arc::new(MarkerResolver("bedrock-key")));
        let dep = deployment_for_slug("bedrock");
        let resolver = registry.resolve_for(&dep);
        let creds = resolver.resolve(&dep).await.expect("ok");
        match creds {
            ResolvedCredentials::ApiKey { bearer } => assert_eq!(bearer, "bedrock-key"),
            other => panic!("expected ApiKey, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn register_does_not_affect_other_slugs() {
        let mut registry = CredentialResolverRegistry::new();
        registry.register("bedrock", Arc::new(MarkerResolver("bedrock-key")));
        let openai = deployment_for_slug("openai");
        let creds = registry
            .resolve_for(&openai)
            .resolve(&openai)
            .await
            .expect("ok");
        match creds {
            ResolvedCredentials::ApiKey { bearer } => assert_eq!(bearer, "k"),
            other => panic!("expected ApiKey, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn watsonx_aliases_share_the_iam_resolver_and_honor_explicit_token_mode() {
        let registry = CredentialResolverRegistry::new();
        let mut native = deployment_for_slug("watsonx");
        native
            .oxllm_params
            .extra
            .insert("watsonx_token".into(), serde_json::json!("pre-exchanged"));
        let mut alias = native.clone();
        alias.oxllm_params.model = "watsonx_text/model".into();
        assert!(Arc::ptr_eq(
            registry.resolve_for(&native),
            registry.resolve_for(&alias)
        ));
        for dep in [native, alias] {
            let creds = registry.resolve_for(&dep).resolve(&dep).await.unwrap();
            assert_eq!(creds.best_effort_bearer(), Some("pre-exchanged"));
        }
    }
}
