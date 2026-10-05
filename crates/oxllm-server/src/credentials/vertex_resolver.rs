//! Credential resolver for `vertex_ai/*` deployments.
//!
//! Returns [`ResolvedCredentials::GcpBearer`] using either an explicitly
//! supplied access token or a standard Google credential JSON file.
//!
//! Resolution order for the token:
//! 1. `oxllm_params.api_key` (config-set static token).
//! 2. Deployment `vertex_credentials` (JSON object, JSON string, or file path).
//! 3. `GCP_ACCESS_TOKEN`, then `VERTEX_AI_ACCESS_TOKEN` env vars.
//! 4. A file selected by `GOOGLE_APPLICATION_CREDENTIALS`.
//!
//! `service_account` (RS256 JWT grant) and `authorized_user` (refresh-token
//! grant) files are supported, with expiry-aware token caching. This is a
//! **subset** of ADC: no well-known gcloud file discovery, metadata server,
//! external-account federation, impersonation or non-default universes.
//! Static access-token mode has no refresh material or trustworthy expiry;
//! operators must rotate those tokens through config reload/environment.

use std::io::Read;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use oxllm_core::Deployment;
use oxllm_core::Error;
use oxllm_core::provider::ResolvedCredentials;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use zeroize::{Zeroize, Zeroizing};

use super::CredentialResolver;
use super::token_exchange::{MAX_CREDENTIAL_BYTES, TokenExchange, unix_now};

const GOOGLE_TOKEN_URL: &str = "https://oauth2.googleapis.com/token";
const CLOUD_PLATFORM_SCOPE: &str = "https://www.googleapis.com/auth/cloud-platform";

/// The signature of the env-var lookup the resolver performs.
type EnvLookup = Arc<dyn Fn(&'static str) -> Option<String> + Send + Sync>;

/// Default env-var lookup: delegates to [`std::env::var`].
fn default_env_lookup() -> EnvLookup {
    Arc::new(|name: &'static str| std::env::var(name).ok().filter(|s| !s.is_empty()))
}

/// Resolve Vertex AI credentials from static tokens or renewable Google JSON.
///
/// The env-var lookup is parameterized via [`VertexResolver::with_env`] so
/// tests can inject deterministic values without touching the real process
/// environment (which is `unsafe` on edition 2024).
#[derive(Clone)]
pub struct VertexResolver {
    env_lookup: EnvLookup,
    exchange: Arc<TokenExchange>,
}

impl std::fmt::Debug for VertexResolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VertexResolver").finish()
    }
}

impl VertexResolver {
    /// Create a resolver that reads from the real process environment.
    pub fn new() -> Self {
        Self {
            env_lookup: default_env_lookup(),
            exchange: Arc::new(TokenExchange::new(GOOGLE_TOKEN_URL)),
        }
    }

    /// Create a resolver with a custom env-var lookup function (for tests).
    pub fn with_env<F: Fn(&'static str) -> Option<String> + Send + Sync + 'static>(
        env_lookup: F,
    ) -> Self {
        Self {
            env_lookup: Arc::new(env_lookup),
            exchange: Arc::new(TokenExchange::new(GOOGLE_TOKEN_URL)),
        }
    }
}

impl Default for VertexResolver {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl CredentialResolver for VertexResolver {
    async fn resolve(&self, deployment: &Deployment) -> Result<ResolvedCredentials, Error> {
        if let Some(token) = deployment
            .oxllm_params
            .api_key
            .as_deref()
            .filter(|s| !s.trim().is_empty())
        {
            return Ok(ResolvedCredentials::GcpBearer {
                token: token.to_string(),
            });
        }
        // Deployment-owned credentials outrank ambient process tokens. The
        // resolver has no request-body input; inference clients cannot choose
        // a file or token endpoint through extra_body.
        if let Some(source) = deployment
            .oxllm_params
            .extra
            .get("vertex_credentials")
            .filter(|value| !value.is_null() && value.as_str() != Some(""))
        {
            return self.resolve_source(source).await;
        }
        let lookup = &self.env_lookup;
        for var in [
            oxllm_providers::vertex_ai::ENV_GCP_ACCESS_TOKEN,
            oxllm_providers::vertex_ai::ENV_VERTEX_ACCESS_TOKEN,
        ] {
            // Filter empties at the resolver level (matching the bedrock /
            // mantle resolvers) so a custom lookup that returns `Some("")`
            // does not yield a `GcpBearer { token: "" }` that Vertex would
            // reject with 401. The default lookup already filters, but this
            // is a defense-in-depth contract rather than an assumption.
            if let Some(token) = lookup(var).filter(|s| !s.trim().is_empty()) {
                return Ok(ResolvedCredentials::GcpBearer { token });
            }
        }
        if let Some(path) =
            lookup("GOOGLE_APPLICATION_CREDENTIALS").filter(|s| !s.trim().is_empty())
        {
            return self.resolve_json(read_credential_file(path).await?).await;
        }
        Err(Error::Config(format!(
            "missing GCP access token: set api_key in oxllm_params \
             or the {} (or {}) env var; for automatic refresh configure \
             vertex_credentials or GOOGLE_APPLICATION_CREDENTIALS with a \
             service_account or authorized_user JSON file",
            oxllm_providers::vertex_ai::ENV_GCP_ACCESS_TOKEN,
            oxllm_providers::vertex_ai::ENV_VERTEX_ACCESS_TOKEN,
        )))
    }
}

impl VertexResolver {
    async fn resolve_source(&self, source: &Value) -> Result<ResolvedCredentials, Error> {
        let bytes =
            match source {
                Value::Object(_) => Zeroizing::new(
                    serde_json::to_vec(source)
                        .map_err(|_| Error::Config("invalid vertex_credentials JSON".into()))?,
                ),
                Value::String(value) if value.trim_start().starts_with('{') => {
                    Zeroizing::new(value.as_bytes().to_vec())
                }
                Value::String(path) => read_credential_file(path.clone()).await?,
                _ => return Err(Error::Config(
                    "vertex_credentials must be a JSON object, JSON string or credential-file path"
                        .into(),
                )),
            };
        self.resolve_json(bytes).await
    }

    async fn resolve_json(&self, bytes: Zeroizing<Vec<u8>>) -> Result<ResolvedCredentials, Error> {
        if bytes.len() > MAX_CREDENTIAL_BYTES {
            return Err(Error::Config(
                "Google credential file exceeds 64 KiB".into(),
            ));
        }
        let credentials: GoogleCredentials = serde_json::from_slice(&bytes)
            .map_err(|_| Error::Config("invalid Google credential JSON".into()))?;
        credentials.validate()?;
        let token = self.exchange.token(&bytes, || credentials.form()).await?;
        Ok(ResolvedCredentials::GcpBearer { token })
    }
}

/// Read only regular, bounded files selected by operator configuration. File
/// I/O stays off the async executor; each resolve observes secret rotation.
async fn read_credential_file(path: String) -> Result<Zeroizing<Vec<u8>>, Error> {
    tokio::time::timeout(
        Duration::from_secs(3),
        tokio::task::spawn_blocking(move || {
            let invalid =
                || Error::Config("cannot read Google credential file as a regular file".into());
            let metadata = std::fs::metadata(&path).map_err(|_| invalid())?;
            if !metadata.is_file() {
                return Err(invalid());
            }
            if metadata.len() > MAX_CREDENTIAL_BYTES as u64 {
                return Err(Error::Config(
                    "Google credential file exceeds 64 KiB".into(),
                ));
            }
            let file = std::fs::File::open(&path).map_err(|_| invalid())?;
            if !file.metadata().map_err(|_| invalid())?.is_file() {
                return Err(invalid());
            }
            let mut bytes = Zeroizing::new(Vec::new());
            file.take(MAX_CREDENTIAL_BYTES as u64 + 1)
                .read_to_end(&mut bytes)
                .map_err(|_| invalid())?;
            if bytes.len() > MAX_CREDENTIAL_BYTES {
                return Err(Error::Config(
                    "Google credential file exceeds 64 KiB".into(),
                ));
            }
            Ok(bytes)
        }),
    )
    .await
    .map_err(|_| Error::Config("Google credential file read timed out".into()))?
    .map_err(|_| Error::Config("Google credential file reader failed".into()))?
}

#[derive(Deserialize)]
struct GoogleCredentials {
    #[serde(rename = "type")]
    kind: String,
    token_uri: Option<String>,
    universe_domain: Option<String>,
    client_email: Option<String>,
    private_key: Option<String>,
    private_key_id: Option<String>,
    client_id: Option<String>,
    client_secret: Option<String>,
    refresh_token: Option<String>,
}

impl Drop for GoogleCredentials {
    fn drop(&mut self) {
        self.private_key.zeroize();
        self.client_secret.zeroize();
        self.refresh_token.zeroize();
    }
}

impl GoogleCredentials {
    fn validate(&self) -> Result<(), Error> {
        // Reject unexpected endpoints, including IP literals, userinfo,
        // fragments and lookalike hosts. Accepted legacy Google URIs still
        // exchange at the canonical endpoint; api_base never affects auth.
        if self.token_uri.as_deref().is_some_and(|uri| {
            !matches!(
                uri,
                GOOGLE_TOKEN_URL
                    | "https://accounts.google.com/o/oauth2/token"
                    | "https://www.googleapis.com/oauth2/v4/token"
            )
        }) {
            return Err(Error::Config("unsupported Google credential token_uri; only official Google OAuth endpoints are supported".into()));
        }
        if self
            .universe_domain
            .as_deref()
            .is_some_and(|domain| domain != "googleapis.com")
        {
            return Err(Error::Config(
                "only the googleapis.com credential universe is supported".into(),
            ));
        }
        match self.kind.as_str() {
            "service_account" => {
                required(&self.client_email, "client_email")?;
                required(&self.private_key, "private_key")?;
            }
            "authorized_user" => {
                required(&self.client_id, "client_id")?;
                required(&self.client_secret, "client_secret")?;
                required(&self.refresh_token, "refresh_token")?;
            }
            _ => return Err(Error::Config("unsupported Google credential type: only service_account and authorized_user are implemented (not full ADC)".into())),
        }
        Ok(())
    }

    fn form(&self) -> Result<Vec<(&'static str, String)>, Error> {
        if self.kind == "authorized_user" {
            return Ok(vec![
                ("grant_type", "refresh_token".into()),
                ("client_id", required(&self.client_id, "client_id")?.into()),
                (
                    "client_secret",
                    required(&self.client_secret, "client_secret")?.into(),
                ),
                (
                    "refresh_token",
                    required(&self.refresh_token, "refresh_token")?.into(),
                ),
            ]);
        }
        let key = jsonwebtoken::EncodingKey::from_rsa_pem(
            required(&self.private_key, "private_key")?.as_bytes(),
        )
        .map_err(|_| Error::Config("invalid Google service-account RSA private key".into()))?;
        let now = unix_now()?;
        let claims = Claims {
            iss: required(&self.client_email, "client_email")?,
            scope: CLOUD_PLATFORM_SCOPE,
            aud: GOOGLE_TOKEN_URL,
            iat: now,
            exp: now.checked_add(3600).ok_or_else(|| {
                Error::Config("system clock exceeds Google token lifetime range".into())
            })?,
        };
        let mut header = jsonwebtoken::Header::new(jsonwebtoken::Algorithm::RS256);
        header.kid = self.private_key_id.clone();
        let assertion = jsonwebtoken::encode(&header, &claims, &key)
            .map_err(|_| Error::Config("cannot sign Google service-account assertion".into()))?;
        Ok(vec![
            (
                "grant_type",
                "urn:ietf:params:oauth:grant-type:jwt-bearer".into(),
            ),
            ("assertion", assertion),
        ])
    }
}

fn required<'a>(value: &'a Option<String>, field: &str) -> Result<&'a str, Error> {
    value
        .as_deref()
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| Error::Config(format!("Google credential JSON is missing {field}")))
}

#[derive(Serialize)]
struct Claims<'a> {
    iss: &'a str,
    scope: &'a str,
    aud: &'a str,
    iat: u64,
    exp: u64,
}

#[cfg(test)]
#[path = "vertex_tests.rs"]
mod tests;
