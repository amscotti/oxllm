//! AWS credential resolver for Bedrock deployments.
//!
//! Reads the standard `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` /
//! `AWS_SESSION_TOKEN` / `AWS_REGION` environment variables and packages them
//! into [`ResolvedCredentials::AwsSigV4`]. The host pipeline signs the
//! request with these materials before sending it to Bedrock.

use std::sync::Arc;

use async_trait::async_trait;
use oxllm_core::Deployment;
use oxllm_core::Error;
use oxllm_core::provider::ResolvedCredentials;

use crate::bedrock_sigv4::validate_aws_region;

use super::CredentialResolver;

/// Signature of the env-var lookup the resolver performs. Parameterized so
/// tests can inject deterministic values without mutating the (unsafe-on-2024)
/// process environment.
pub(crate) type EnvLookup = Arc<dyn Fn(&'static str) -> Option<String> + Send + Sync>;

/// Read AWS credentials from the real process environment.
fn default_env_lookup() -> EnvLookup {
    Arc::new(|name: &'static str| std::env::var(name).ok().filter(|s| !s.is_empty()))
}

/// Resolved raw AWS credential material, before it is packaged for a provider.
/// Returned by [`resolve_aws_material`] and consumed by both
/// [`BedrockCredentialResolver`] and the realtime Bedrock adapter
/// (`crate::realtime::bedrock::BedrockRealtimeAdapter`), so the
/// typed → extra → env precedence lives in one place.
pub(crate) struct AwsMaterial {
    pub(crate) access_key: String,
    pub(crate) secret_key: String,
    pub(crate) session_token: Option<String>,
    pub(crate) region: String,
}

/// Resolve AWS credential material for `deployment` using the standard
/// precedence shared by the Bedrock runtime resolver and the realtime
/// adapter:
///
/// 1. Per-deployment typed fields (`aws_access_key_id` /
///    `aws_secret_access_key` / `aws_session_token`) — via
///    [`resolve_access_secret_session`].
/// 2. The serde-flatten `extra` bag (legacy / lite-LLM-compatible aliases).
/// 3. The ambient environment via `env_lookup`.
///
/// Region precedence (matches LiteLLM's `base_aws_llm.py`):
/// `extra.aws_region_name` → `extra.aws_region` → `AWS_REGION_NAME` →
/// `AWS_REGION` → `AWS_DEFAULT_REGION` → `us-east-1`. `aws_region_name` is
/// the LiteLLM-canonical key read from `optional_params`; `aws_region` is an
/// oxllm-native shorthand. Both are accepted so a config migrated verbatim
/// from LiteLLM resolves the region correctly. The region is validated
/// against `^[a-z0-9-]+$` so a typo like `"us-east-1 "` surfaces locally
/// instead of producing an obscure `SignatureDoesNotMatch` from AWS.
///
/// Default-region divergence: LiteLLM's `base_aws_llm.py` falls back to
/// `us-west-2`, while oxllm (and AWS's own Bedrock documentation) uses
/// `us-east-1`. This is an intentional divergence — operators are expected
/// to set a region explicitly or via `AWS_REGION`, and a wrong-region error
/// surfaces immediately for region-scoped models.
pub(crate) fn resolve_aws_material(
    deployment: &Deployment,
    env_lookup: &EnvLookup,
) -> Result<AwsMaterial, Error> {
    let extra = &deployment.oxllm_params.extra;

    let (access_key, secret_key, session_token) = resolve_access_secret_session(
        deployment,
        env_lookup,
        "missing AWS credentials: set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY",
    )?;

    let region = str_from_extra(extra, "aws_region_name")
        .or_else(|| str_from_extra(extra, "aws_region"))
        .or_else(|| env_lookup("AWS_REGION_NAME"))
        .or_else(|| env_lookup("AWS_REGION"))
        .or_else(|| env_lookup("AWS_DEFAULT_REGION"))
        .unwrap_or_else(|| "us-east-1".to_string());
    validate_aws_region(&region).map_err(Error::Config)?;

    Ok(AwsMaterial {
        access_key,
        secret_key,
        session_token,
        region,
    })
}

/// Resolve the access-key / secret-key / session-token triple shared by
/// every AWS-backed resolver (Bedrock native, Bedrock realtime, and Bedrock
/// Mantle). Centralizing the typed → extra → env precedence keeps a future
/// addition (e.g. `aws_profile_name` or web-identity) in one place instead
/// of drifting across resolvers. Region resolution is intentionally NOT
/// shared: the mantle resolver inserts `BEDROCK_MANTLE_REGION` into its
/// precedence chain.
pub(crate) fn resolve_access_secret_session(
    deployment: &Deployment,
    env_lookup: &EnvLookup,
    missing_msg: &str,
) -> Result<(String, String, Option<String>), Error> {
    let params = &deployment.oxllm_params;
    let extra = &params.extra;

    let access_key = params
        .aws_access_key_id
        .clone()
        .filter(|s| !s.is_empty())
        .or_else(|| str_from_extra(extra, "aws_access_key_id"))
        .or_else(|| env_lookup("AWS_ACCESS_KEY_ID"))
        .filter(|s| !s.is_empty());
    let secret_key = params
        .aws_secret_access_key
        .clone()
        .filter(|s| !s.is_empty())
        .or_else(|| str_from_extra(extra, "aws_secret_access_key"))
        .or_else(|| env_lookup("AWS_SECRET_ACCESS_KEY"))
        .filter(|s| !s.is_empty());

    let (access_key, secret_key) = match (access_key, secret_key) {
        (Some(a), Some(s)) => (a, s),
        _ => return Err(Error::Config(missing_msg.to_string())),
    };

    let session_token = params
        .aws_session_token
        .clone()
        .filter(|s| !s.is_empty())
        .or_else(|| str_from_extra(extra, "aws_session_token"))
        .or_else(|| env_lookup("AWS_SESSION_TOKEN"))
        .filter(|s| !s.is_empty());

    Ok((access_key, secret_key, session_token))
}

/// Pull a string value out of the serde-flatten `extra` map.
fn str_from_extra(extra: &serde_json::Map<String, serde_json::Value>, key: &str) -> Option<String> {
    extra
        .get(key)
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .map(String::from)
}

/// Bedrock credential resolver — env-only for now.
///
/// Future improvements (STS AssumeRole, SSO, IMDS) belong here. Production
/// users typically run with a static key env-injected by their secret manager,
/// which is what this resolver supports today.
///
/// AWS env vars are read on each `resolve` (the hot path) so that a credential
/// rotation performed by an external process (secret manager reload, IAM
/// refresh) takes effect without a gateway reload — matching the mantle and
/// vertex resolvers. The lookup is injectable via [`Self::with_env`] so tests
/// are deterministic and never touch the real process environment.
pub struct BedrockCredentialResolver {
    env_lookup: EnvLookup,
}

impl std::fmt::Debug for BedrockCredentialResolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BedrockCredentialResolver").finish()
    }
}

impl Clone for BedrockCredentialResolver {
    fn clone(&self) -> Self {
        Self {
            env_lookup: self.env_lookup.clone(),
        }
    }
}

impl Default for BedrockCredentialResolver {
    fn default() -> Self {
        Self::new()
    }
}

impl BedrockCredentialResolver {
    /// Create a resolver backed by the real process environment.
    pub fn new() -> Self {
        Self {
            env_lookup: default_env_lookup(),
        }
    }

    /// Create a resolver with a custom env-var lookup function (for tests).
    pub fn with_env<F: Fn(&'static str) -> Option<String> + Send + Sync + 'static>(
        env_lookup: F,
    ) -> Self {
        Self {
            env_lookup: Arc::new(env_lookup),
        }
    }
}

#[async_trait]
impl CredentialResolver for BedrockCredentialResolver {
    async fn resolve(&self, deployment: &Deployment) -> Result<ResolvedCredentials, Error> {
        let material = resolve_aws_material(deployment, &self.env_lookup)?;
        Ok(ResolvedCredentials::AwsSigV4 {
            access_key: material.access_key,
            secret_key: material.secret_key,
            session_token: material.session_token,
            region: material.region,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxllm_core::deployment::OxllmParams;

    fn deployment_with_extra(extra: serde_json::Map<String, serde_json::Value>) -> Deployment {
        Deployment {
            model_name: "bedrock-claude".to_string(),
            oxllm_params: OxllmParams {
                model: "bedrock/anthropic.claude-3-5-sonnet".to_string(),
                extra,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn env_with(
        vars: &[(&'static str, &'static str)],
    ) -> impl Fn(&'static str) -> Option<String> + Clone {
        let map: std::collections::HashMap<&'static str, &'static str> =
            vars.iter().copied().collect();
        move |name: &'static str| map.get(name).copied().map(String::from)
    }

    #[tokio::test]
    async fn reads_credentials_from_environment() {
        let resolver = BedrockCredentialResolver::with_env(env_with(&[
            ("AWS_ACCESS_KEY_ID", "AKIATEST"),
            ("AWS_SECRET_ACCESS_KEY", "secret"),
            ("AWS_SESSION_TOKEN", "token"),
            ("AWS_REGION", "eu-west-1"),
        ]));
        let dep = deployment_with_extra(serde_json::Map::new());
        let creds = resolver.resolve(&dep).await.expect("ok");
        match creds {
            ResolvedCredentials::AwsSigV4 {
                access_key,
                secret_key,
                session_token,
                region,
            } => {
                assert_eq!(access_key, "AKIATEST");
                assert_eq!(secret_key, "secret");
                assert_eq!(session_token.as_deref(), Some("token"));
                assert_eq!(region, "eu-west-1");
            }
            other => panic!("expected AwsSigV4, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn errors_when_credentials_missing() {
        let resolver = BedrockCredentialResolver::with_env(env_with(&[]));
        let dep = deployment_with_extra(serde_json::Map::new());
        let err = resolver.resolve(&dep).await.expect_err("missing creds");
        assert!(matches!(err, Error::Config(_)));
    }

    /// Empty-string values in `extra` must not short-circuit the precedence
    /// chain: they fall through to env vars instead of producing a
    /// "missing credentials" / bad-region error.
    #[tokio::test]
    async fn empty_extra_values_fall_through_to_env() {
        let resolver = BedrockCredentialResolver::with_env(env_with(&[
            ("AWS_ACCESS_KEY_ID", "AKIATEST"),
            ("AWS_SECRET_ACCESS_KEY", "secret"),
            ("AWS_REGION", "eu-west-1"),
        ]));
        let mut extra = serde_json::Map::new();
        extra.insert("aws_access_key_id".to_string(), "".into());
        extra.insert("aws_secret_access_key".to_string(), "".into());
        extra.insert("aws_region_name".to_string(), "".into());
        let dep = deployment_with_extra(extra);
        let creds = resolver.resolve(&dep).await.expect("ok");
        match creds {
            ResolvedCredentials::AwsSigV4 {
                access_key, region, ..
            } => {
                assert_eq!(access_key, "AKIATEST");
                assert_eq!(region, "eu-west-1");
            }
            other => panic!("expected AwsSigV4, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn per_deployment_extra_overrides_environment() {
        let resolver = BedrockCredentialResolver::with_env(env_with(&[
            ("AWS_ACCESS_KEY_ID", "ENV-KEY"),
            ("AWS_SECRET_ACCESS_KEY", "ENV-SECRET"),
            ("AWS_REGION", "us-east-1"),
        ]));

        let mut extra = serde_json::Map::new();
        extra.insert(
            "aws_access_key_id".to_string(),
            serde_json::Value::String("DEP-KEY".to_string()),
        );
        extra.insert(
            "aws_region".to_string(),
            serde_json::Value::String("us-west-2".to_string()),
        );
        let dep = deployment_with_extra(extra);
        let creds = resolver.resolve(&dep).await.expect("ok");
        match creds {
            ResolvedCredentials::AwsSigV4 {
                access_key,
                secret_key,
                region,
                ..
            } => {
                // Typed/extra `aws_access_key_id` wins over the env value.
                assert_eq!(access_key, "DEP-KEY");
                // Env secret is used because the deployment did not override it.
                assert_eq!(secret_key, "ENV-SECRET");
                // Extra `aws_region` wins over the env default region.
                assert_eq!(region, "us-west-2");
            }
            other => panic!("expected AwsSigV4, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn defaults_region_to_us_east_1() {
        let resolver = BedrockCredentialResolver::with_env(env_with(&[
            ("AWS_ACCESS_KEY_ID", "AKIATEST"),
            ("AWS_SECRET_ACCESS_KEY", "secret"),
        ]));
        let dep = deployment_with_extra(serde_json::Map::new());
        let creds = resolver.resolve(&dep).await.expect("ok");
        match creds {
            ResolvedCredentials::AwsSigV4 { region, .. } => {
                assert_eq!(region, "us-east-1");
            }
            other => panic!("expected AwsSigV4, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn env_region_overrides_default_region() {
        // When no per-deployment region is set, the env `AWS_REGION` is used
        // instead of the `us-east-1` default.
        let resolver = BedrockCredentialResolver::with_env(env_with(&[
            ("AWS_ACCESS_KEY_ID", "AKIATEST"),
            ("AWS_SECRET_ACCESS_KEY", "secret"),
            ("AWS_REGION", "ap-south-1"),
        ]));
        let dep = deployment_with_extra(serde_json::Map::new());
        let creds = resolver.resolve(&dep).await.expect("ok");
        match creds {
            ResolvedCredentials::AwsSigV4 { region, .. } => {
                assert_eq!(region, "ap-south-1");
            }
            other => panic!("expected AwsSigV4, got {other:?}"),
        }
    }

    /// LiteLLM checks `AWS_REGION_NAME` (a LiteLLM-specific alias) BEFORE the
    /// standard `AWS_REGION` (`base_aws_llm.py:569`). Operators migrating from
    /// LiteLLM who rely on `AWS_REGION_NAME` must not silently fall through to
    /// `us-east-1` and hit `SignatureDoesNotMatch` with no local hint.
    #[tokio::test]
    async fn aws_region_name_env_is_preferred_over_aws_region() {
        let resolver = BedrockCredentialResolver::with_env(env_with(&[
            ("AWS_ACCESS_KEY_ID", "AKIATEST"),
            ("AWS_SECRET_ACCESS_KEY", "secret"),
            ("AWS_REGION_NAME", "eu-central-1"),
            ("AWS_REGION", "us-east-1"),
        ]));
        let dep = deployment_with_extra(serde_json::Map::new());
        let creds = resolver.resolve(&dep).await.expect("ok");
        match creds {
            ResolvedCredentials::AwsSigV4 { region, .. } => {
                assert_eq!(
                    region, "eu-central-1",
                    "AWS_REGION_NAME must win over AWS_REGION (LiteLLM parity)"
                );
            }
            other => panic!("expected AwsSigV4, got {other:?}"),
        }
    }

    /// LiteLLM reads the per-deployment region from `optional_params` under
    /// the canonical key `aws_region_name` (`base_aws_llm.py:_get_aws_region_name`).
    /// A config migrated verbatim from LiteLLM (`aws_region_name: eu-west-2`
    /// in `model_list`) lands in `extra["aws_region_name"]`. The resolver
    /// must honour it even when no `aws_region` shorthand and no env var are
    /// set, otherwise it silently falls through to the default region and
    /// hits a wrong-region / `SignatureDoesNotMatch` error from AWS.
    #[tokio::test]
    async fn per_deployment_aws_region_name_key_is_honoured() {
        let resolver = BedrockCredentialResolver::with_env(env_with(&[
            ("AWS_ACCESS_KEY_ID", "AKIATEST"),
            ("AWS_SECRET_ACCESS_KEY", "secret"),
        ]));
        let mut extra = serde_json::Map::new();
        extra.insert(
            "aws_region_name".to_string(),
            serde_json::Value::String("eu-west-2".to_string()),
        );
        let dep = deployment_with_extra(extra);
        let creds = resolver.resolve(&dep).await.expect("ok");
        match creds {
            ResolvedCredentials::AwsSigV4 { region, .. } => {
                assert_eq!(
                    region, "eu-west-2",
                    "aws_region_name config key must resolve (LiteLLM parity)"
                );
            }
            other => panic!("expected AwsSigV4, got {other:?}"),
        }
    }

    /// The oxllm-native `aws_region` shorthand still works and wins over the
    /// env defaults; `aws_region_name` is a fallback alias, not a replacement.
    #[tokio::test]
    async fn aws_region_shorthand_still_works() {
        let resolver = BedrockCredentialResolver::with_env(env_with(&[
            ("AWS_ACCESS_KEY_ID", "AKIATEST"),
            ("AWS_SECRET_ACCESS_KEY", "secret"),
            ("AWS_REGION", "us-east-1"),
        ]));
        let mut extra = serde_json::Map::new();
        extra.insert(
            "aws_region".to_string(),
            serde_json::Value::String("ap-southeast-2".to_string()),
        );
        let dep = deployment_with_extra(extra);
        let creds = resolver.resolve(&dep).await.expect("ok");
        match creds {
            ResolvedCredentials::AwsSigV4 { region, .. } => {
                assert_eq!(region, "ap-southeast-2");
            }
            other => panic!("expected AwsSigV4, got {other:?}"),
        }
    }

    /// A malformed region (here a trailing space) must be rejected up front
    /// with a typed config error rather than producing an obscure
    /// `SignatureDoesNotMatch` from AWS.
    #[tokio::test]
    async fn errors_on_invalid_aws_region() {
        let resolver = BedrockCredentialResolver::with_env(env_with(&[
            ("AWS_ACCESS_KEY_ID", "AKIATEST"),
            ("AWS_SECRET_ACCESS_KEY", "secret"),
            ("AWS_REGION", "us-east-1 "),
        ]));
        let dep = deployment_with_extra(serde_json::Map::new());
        let err = resolver.resolve(&dep).await.expect_err("invalid region");
        assert!(matches!(err, Error::Config(_)), "got {err:?}");
        // Defense in depth: the raw region must not leak into the message.
        assert!(
            !format!("{err}").contains("us-east-1 "),
            "raw region leaked: {err}"
        );
    }
}
