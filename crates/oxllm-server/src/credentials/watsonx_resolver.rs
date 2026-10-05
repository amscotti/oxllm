//! IBM Cloud WatsonX credentials. API keys are exchanged with IBM IAM; already
//! exchanged tokens use the explicit `watsonx_token` / `WATSONX_TOKEN` mode.
//! Static tokens require out-of-band rotation. Cloud Pak for Data and custom
//! IAM endpoints are not implemented by this IBM Cloud resolver.

use std::sync::Arc;

use async_trait::async_trait;
use oxllm_core::provider::ResolvedCredentials;
use oxllm_core::{Deployment, Error};

use super::CredentialResolver;
use super::token_exchange::TokenExchange;

const IAM_TOKEN_URL: &str = "https://iam.cloud.ibm.com/identity/token";
type EnvLookup = Arc<dyn Fn(&'static str) -> Option<String> + Send + Sync>;

/// Resolves IBM IAM bearer tokens with coalesced, expiry-aware caching.
///
/// Per-deployment `watsonx_token` wins, followed by a per-deployment API key
/// (`api_key`, then `watsonx_api_key`). Only when neither is configured are
/// `WATSONX_TOKEN`, `WATSONX_APIKEY`, and `WATSONX_API_KEY` consulted, in order.
/// This prevents a process-wide token from overriding an explicit tenant key.
#[derive(Clone)]
pub struct WatsonXResolver {
    env_lookup: EnvLookup,
    exchange: Arc<TokenExchange>,
}

impl std::fmt::Debug for WatsonXResolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WatsonXResolver").finish_non_exhaustive()
    }
}

impl WatsonXResolver {
    /// Create a resolver backed by operator-provided environment variables.
    pub fn new() -> Self {
        Self::with_env(|name| std::env::var(name).ok())
    }

    /// Supply an environment lookup without mutating the process environment.
    pub fn with_env<F: Fn(&'static str) -> Option<String> + Send + Sync + 'static>(
        env_lookup: F,
    ) -> Self {
        Self {
            env_lookup: Arc::new(env_lookup),
            exchange: Arc::new(TokenExchange::new(IAM_TOKEN_URL)),
        }
    }
}

impl Default for WatsonXResolver {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl CredentialResolver for WatsonXResolver {
    async fn resolve(&self, deployment: &Deployment) -> Result<ResolvedCredentials, Error> {
        let params = &deployment.oxllm_params;
        let configured = |name: &str| match params.extra.get(name) {
            None | Some(serde_json::Value::Null) => Ok(None),
            Some(serde_json::Value::String(value)) => {
                Ok((!value.trim().is_empty()).then_some(value.as_str()))
            }
            _ => Err(Error::Config(format!("{name} must be a credential string"))),
        };
        if let Some(token) = configured("watsonx_token")? {
            return Ok(ResolvedCredentials::from_key(token));
        }
        let key = match params.api_key.as_deref().filter(|s| !s.trim().is_empty()) {
            Some(key) => Some(key.to_string()),
            None => configured("watsonx_api_key")?.map(String::from),
        };
        let key = if let Some(key) = key {
            key
        } else {
            if let Some(token) = (self.env_lookup)("WATSONX_TOKEN").filter(|s| !s.trim().is_empty())
            {
                return Ok(ResolvedCredentials::from_key(token));
            }
            ["WATSONX_APIKEY", "WATSONX_API_KEY"].into_iter()
                .find_map(|name| (self.env_lookup)(name).filter(|s| !s.trim().is_empty()))
                .ok_or_else(|| Error::Config("missing WatsonX credentials: configure api_key or watsonx_token, or set WATSONX_APIKEY / WATSONX_TOKEN".into()))?
        };
        let token = self
            .exchange
            .token(key.as_bytes(), || {
                Ok(vec![
                    (
                        "grant_type",
                        "urn:ibm:params:oauth:grant-type:apikey".into(),
                    ),
                    ("apikey", key.clone()),
                ])
            })
            .await?;
        Ok(ResolvedCredentials::from_key(token))
    }
}

#[cfg(test)]
#[path = "watsonx_tests.rs"]
mod tests;
