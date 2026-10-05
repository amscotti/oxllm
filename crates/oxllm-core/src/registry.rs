//! The provider registry — maps a provider slug to its [`Provider`] impl.
//!
//! The server builds one of these at startup from the enabled Cargo features
//! and uses [`ProviderRegistry::resolve_for_deployment`] to find the right
//! transform object at request time. This replaces hardcoded provider lookup
//! in handlers.

use std::collections::HashMap;
use std::sync::Arc;

use crate::deployment::Deployment;
use crate::provider::Provider;

/// A registry of providers keyed by slug (e.g. `"openai"`, `"anthropic"`).
#[derive(Default)]
pub struct ProviderRegistry {
    providers: HashMap<String, Arc<dyn Provider>>,
}

impl ProviderRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            providers: HashMap::new(),
        }
    }

    /// Register a provider under the given slug.
    ///
    /// Registering the same slug twice is almost always a feature-flag wiring
    /// mistake (two providers compiled in under the same id). Rather than
    /// silently shadowing ("last wins"), this logs a `warn` so the duplicate
    /// is observable. The new provider still replaces the prior one.
    pub fn register(&mut self, id: &str, provider: Arc<dyn Provider>) {
        if self.providers.contains_key(id) {
            tracing::warn!(
                slug = %id,
                new = %provider.id(),
                "provider slug already registered; replacing (likely a duplicate feature-flag wiring)"
            );
        }
        self.providers.insert(id.to_string(), provider);
    }

    /// Look up a provider by slug.
    pub fn get(&self, id: &str) -> Option<&Arc<dyn Provider>> {
        self.providers.get(id)
    }

    /// Resolve the provider for a deployment using its [`Deployment::provider_slug`].
    pub fn resolve_for_deployment(&self, deployment: &Deployment) -> Option<&Arc<dyn Provider>> {
        self.get(deployment.provider_slug())
    }

    /// Iterate over all registered provider slugs.
    pub fn slugs(&self) -> impl Iterator<Item = &str> {
        self.providers.keys().map(String::as_str)
    }
}

impl std::fmt::Debug for ProviderRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProviderRegistry")
            .field("slugs", &self.providers.keys().collect::<Vec<_>>())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::endpoint::Endpoint;
    use crate::provider::{ProviderId, ResolvedCredentials};
    use std::collections::HashMap;

    struct FakeProvider {
        slug: &'static str,
    }

    impl Provider for FakeProvider {
        fn id(&self) -> ProviderId {
            self.slug
        }

        fn supports(&self, _endpoint: Endpoint) -> bool {
            false
        }
    }

    fn deployment_with_slug(slug: &str) -> Deployment {
        let model = match slug {
            "openai" => "openai/gpt-4o".to_string(),
            "anthropic" => "anthropic/claude-3-5-sonnet".to_string(),
            other => format!("{other}/model"),
        };
        Deployment {
            model_name: "m".to_string(),
            oxllm_params: crate::deployment::OxllmParams {
                model,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[test]
    fn register_and_get_provider() {
        let mut registry = ProviderRegistry::new();
        let provider: Arc<dyn Provider> = Arc::new(FakeProvider { slug: "openai" });
        registry.register("openai", provider);
        assert!(registry.get("openai").is_some());
        assert!(registry.get("anthropic").is_none());
    }

    #[test]
    fn register_duplicate_slug_replaces_and_is_detectable() {
        // Registering the same slug twice logs a warn (a duplicate feature-flag
        // wiring mistake) and replaces the prior provider. The new provider
        // wins so the misconfiguration is at least consistent, not a silent
        // shadow whose winner depends on HashMap iteration order.
        let mut registry = ProviderRegistry::new();
        registry.register("openai", Arc::new(FakeProvider { slug: "openai" }));
        registry.register(
            "openai",
            Arc::new(FakeProvider {
                slug: "replacement",
            }),
        );
        let resolved = registry.get("openai").expect("slug still present");
        assert_eq!(
            resolved.id(),
            "replacement",
            "duplicate registration must replace, last wins"
        );
        assert_eq!(registry.slugs().count(), 1, "no duplicate keys");
    }

    #[test]
    fn resolve_for_deployment_uses_provider_slug() {
        let mut registry = ProviderRegistry::new();
        registry.register("anthropic", Arc::new(FakeProvider { slug: "anthropic" }));
        let dep = deployment_with_slug("anthropic");
        let resolved = registry
            .resolve_for_deployment(&dep)
            .expect("anthropic provider should resolve");
        assert_eq!(resolved.id(), "anthropic");
    }

    #[test]
    fn resolve_for_unknown_provider_returns_none() {
        let registry = ProviderRegistry::new();
        let dep = deployment_with_slug("openai");
        assert!(registry.resolve_for_deployment(&dep).is_none());
    }

    #[test]
    fn default_empty_registry_has_no_slugs() {
        let registry = ProviderRegistry::default();
        assert_eq!(registry.slugs().count(), 0);
    }

    #[test]
    fn slugs_iterates_all_registered() {
        let mut registry = ProviderRegistry::new();
        registry.register("openai", Arc::new(FakeProvider { slug: "openai" }));
        registry.register("anthropic", Arc::new(FakeProvider { slug: "anthropic" }));
        let mut slugs: Vec<&str> = registry.slugs().collect();
        slugs.sort();
        assert_eq!(slugs, vec!["anthropic", "openai"]);
    }

    #[test]
    fn resolved_credentials_from_key_is_pure() {
        let creds = ResolvedCredentials::from_key("sk-test");
        let bearer = match &creds {
            ResolvedCredentials::ApiKey { bearer } => bearer.as_str(),
            _ => panic!("expected ApiKey variant"),
        };
        let mut headers = HashMap::new();
        headers.insert("authorization".to_string(), format!("Bearer {bearer}"));
        assert_eq!(
            headers.get("authorization").map(String::as_str),
            Some("Bearer sk-test")
        );
    }
}
