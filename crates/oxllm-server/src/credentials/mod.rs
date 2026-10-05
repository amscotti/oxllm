//! Credential resolution layer (host only).
//!
//! The host resolves credentials **once per deployment** (potentially cached)
//! into a [`ResolvedCredentials`] value, then passes that pure material into
//! the provider transforms. This keeps all network/STS/OAuth exchange out of
//! `oxllm-providers` (which must remain pure).

mod registry;
mod static_key;
#[cfg(test)]
mod tests;
mod token_exchange;
mod vertex_resolver;
mod watsonx_resolver;

pub(crate) mod bedrock;
mod bedrock_mantle;
mod bedrock_mantle_token;

pub use bedrock::BedrockCredentialResolver;
pub use bedrock_mantle::BedrockMantleCredentialResolver;
pub use registry::CredentialResolverRegistry;
pub use static_key::StaticKeyResolver;
pub use vertex_resolver::VertexResolver;
pub use watsonx_resolver::WatsonXResolver;

use async_trait::async_trait;
use oxllm_core::Deployment;
use oxllm_core::Error;
use oxllm_core::provider::ResolvedCredentials;

/// Resolves [`ResolvedCredentials`] for a given deployment.
///
/// Implementations live in the host crate only. The default implementation
/// ([`StaticKeyResolver`]) reproduces the pre-Phase-9 behavior of reading
/// `oxllm_params.api_key`. [`CredentialResolverRegistry::new`] includes the
/// IBM IAM resolver; startup (`state/build.rs::build_credential_resolver`)
/// additionally registers Bedrock and Vertex by slug.
#[async_trait]
pub trait CredentialResolver: Send + Sync {
    /// Resolve the credentials for `deployment`.
    async fn resolve(&self, deployment: &Deployment) -> Result<ResolvedCredentials, Error>;
}
