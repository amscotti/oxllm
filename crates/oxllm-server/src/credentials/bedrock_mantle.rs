//! Credential resolver for `bedrock_mantle/*` deployments.
//!
//! - **Phase 1 (explicit token):** when `oxllm_params.api_key` is set it is
//!   returned verbatim as a [`ResolvedCredentials::ApiKey`]. As a fallback,
//!   the documented env vars `BEDROCK_MANTLE_API_KEY` and
//!   `AWS_BEARER_TOKEN_BEDROCK` are consulted in that order
//!   (`BedrockMantleAuthMixin._resolve_bearer_token`).
//! - **Phase 2 (auto-generated token):** otherwise a Bedrock Mantle bearer
//!   token is minted from AWS credentials and cached. This eliminates the
//!   external `aws-bedrock-token-generator` (Python/JS) dependency.
//!
//! AWS credential resolution order (each level falls through to the next):
//! 1. `oxllm_params` typed fields: `aws_access_key_id`,
//!    `aws_secret_access_key`, `aws_session_token`.
//! 2. The standard environment variables `AWS_ACCESS_KEY_ID`,
//!    `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`.
//! 3. Region (resolved independently): `oxllm_params.extra.aws_region_name`,
//!    `oxllm_params.extra.aws_region`, then `BEDROCK_MANTLE_REGION`,
//!    `AWS_REGION_NAME`, `AWS_REGION`, `AWS_DEFAULT_REGION`, then
//!    `us-east-1`.
//!
//! The generated token is cached and reused until it is within
//! [`REFRESH_THRESHOLD_SECS`] of expiry, at which point it is regenerated
//! before the next request. The cache is keyed on a fingerprint of the
//! resolved credential material so per-deployment credentials never leak
//! across deployments.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use oxllm_core::Deployment;
use oxllm_core::Error;
use oxllm_core::provider::ResolvedCredentials;

use crate::bedrock_sigv4::{hex_encode, sha256, validate_aws_region};

use super::CredentialResolver;
use super::bedrock::resolve_access_secret_session;
use super::bedrock_mantle_token::{MAX_EXPIRES_SECS, generate_bedrock_mantle_token};

/// Regenerate the token when fewer than this many seconds of validity remain.
const REFRESH_THRESHOLD_SECS: u64 = 300; // 5 minutes
/// Default token lifetime (12 hours — the AWS maximum), used when the
/// deployment does not request a shorter expiry.
const DEFAULT_EXPIRES_SECS: u64 = 43_200;

/// Signature of the env-var lookup the resolver performs. Parameterized so
/// tests can inject deterministic values without mutating the (unsafe-on-2024)
/// process environment.
type EnvLookup = Arc<dyn Fn(&'static str) -> Option<String> + Send + Sync>;
/// Signature of the wall-clock used for timestamps + cache expiry.
type Now = Arc<dyn Fn() -> u64 + Send + Sync>;

/// Read AWS credentials from the real process environment.
fn default_env_lookup() -> EnvLookup {
    Arc::new(|name: &'static str| std::env::var(name).ok().filter(|s| !s.is_empty()))
}

/// Read the current Unix time (seconds) from the system clock.
fn default_now() -> Now {
    Arc::new(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    })
}

/// Credential resolver for `bedrock_mantle/*` deployments.
///
/// Construct with [`BedrockMantleCredentialResolver::new`] (real env + clock)
/// or [`BedrockMantleCredentialResolver::with_env`] for deterministic tests.
/// The resolver holds an in-memory token cache shared across every request
/// routed to a `bedrock_mantle` deployment.
pub struct BedrockMantleCredentialResolver {
    env_lookup: EnvLookup,
    now: Now,
    /// Per-fingerprint token cache. Keyed on the credential fingerprint so
    /// distinct `bedrock_mantle/*` deployments do not evict each other's
    /// entries (which would defeat caching and force a token regeneration on
    /// every request). Bounded by the number of distinct deployments, so no
    /// unbounded-growth concern.
    cache: Mutex<HashMap<String, CachedToken>>,
}

impl std::fmt::Debug for BedrockMantleCredentialResolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BedrockMantleCredentialResolver").finish()
    }
}

impl Default for BedrockMantleCredentialResolver {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone)]
struct CachedToken {
    token: String,
    /// Absolute Unix-second instant at which the embedded presigned URL
    /// expires (`now_at_generation + expires_secs`).
    expires_at_unix: u64,
}

impl BedrockMantleCredentialResolver {
    /// Create a resolver backed by the real process environment and clock.
    pub fn new() -> Self {
        Self {
            env_lookup: default_env_lookup(),
            now: default_now(),
            cache: Mutex::new(HashMap::new()),
        }
    }

    /// Create a resolver with a custom env-var lookup and clock. Both are
    /// injectable so unit tests are deterministic and never touch the real
    /// process environment.
    pub fn with_env<F, G>(env_lookup: F, now: G) -> Self
    where
        F: Fn(&'static str) -> Option<String> + Send + Sync + 'static,
        G: Fn() -> u64 + Send + Sync + 'static,
    {
        Self {
            env_lookup: Arc::new(env_lookup),
            now: Arc::new(now),
            cache: Mutex::new(HashMap::new()),
        }
    }
}

/// Resolved raw AWS credential material, before it is turned into either an
/// `AwsSigV4` (Bedrock native) or a Mantle bearer token.
struct AwsMaterial {
    access_key: String,
    secret_key: String,
    session_token: Option<String>,
    region: String,
    expires_secs: u64,
}

impl AwsMaterial {
    /// A stable identity for cache keying. Includes every field that changes
    /// the generated token, so swapping credentials always invalidates the
    /// cache even if the cached token is still time-valid. The access-key id,
    /// secret key, and session token are all included as one-way SHA-256
    /// digests — never in raw form — so a rotation of *only* the secret (e.g.
    /// after a leak) or *only* the session token (the documented
    /// STS/web-identity rotation story) invalidates the cache without
    /// exposing any credential material in logs, error messages, or heap
    /// memory beyond the digests.
    ///
    /// `expires_secs` is clamped to [`MAX_EXPIRES_SECS`] before fingerprinting
    /// so two configs that both request `>= MAX_EXPIRES_SECS` (and therefore
    /// produce identical tokens after the generator's own clamp) share a cache
    /// slot instead of forcing a regeneration on each rotation.
    fn fingerprint(&self) -> String {
        let access_digest = hex_encode(&sha256(self.access_key.as_bytes()));
        let secret_digest = hex_encode(&sha256(self.secret_key.as_bytes()));
        let session_digest = self
            .session_token
            .as_ref()
            .map(|t| hex_encode(&sha256(t.as_bytes())))
            .unwrap_or_default();
        let expires_for_fp = self.expires_secs.min(MAX_EXPIRES_SECS);
        format!(
            "{access_digest}|{region}|{expires}|{session_digest}|{secret_digest}",
            region = self.region,
            expires = expires_for_fp,
        )
    }
}

#[async_trait]
impl CredentialResolver for BedrockMantleCredentialResolver {
    async fn resolve(&self, deployment: &Deployment) -> Result<ResolvedCredentials, Error> {
        // Phase 1: an explicit, pre-built token always wins.
        if let Some(api_key) = deployment
            .oxllm_params
            .api_key
            .as_deref()
            .filter(|s| !s.is_empty())
        {
            return Ok(ResolvedCredentials::ApiKey {
                bearer: api_key.to_string(),
            });
        }

        // Phase 1b: the reference documents `BEDROCK_MANTLE_API_KEY` /
        // `AWS_BEARER_TOKEN_BEDROCK` as env-var fallbacks for a pre-built
        // bearer token (`BedrockMantleAuthMixin._resolve_bearer_token`:
        // `api_key or BEDROCK_MANTLE_API_KEY or AWS_BEARER_TOKEN_BEDROCK`).
        // Honour them before auto-generation so a migrated config that
        // relies on the env var does not silently mint a *different* token
        // from AWS credentials (or fail with "missing AWS credentials").
        let env = &self.env_lookup;
        if let Some(bearer) = env("BEDROCK_MANTLE_API_KEY")
            .or_else(|| env("AWS_BEARER_TOKEN_BEDROCK"))
            .filter(|s| !s.is_empty())
        {
            return Ok(ResolvedCredentials::ApiKey { bearer });
        }

        let material = resolve_aws_material(deployment, &self.env_lookup)?;
        let now = (self.now)();
        let fp = material.fingerprint();

        // Fast path: a cached token for the *same* credential material that
        // is still outside the refresh window. The lock is dropped immediately
        // so concurrent resolves for distinct fingerprints do not serialize on
        // the slow path below.
        let cached = {
            let cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
            cache.get(&fp).cloned()
        };
        if let Some(cached) = cached
            && now.saturating_add(REFRESH_THRESHOLD_SECS) < cached.expires_at_unix
        {
            return Ok(ResolvedCredentials::ApiKey {
                bearer: cached.token,
            });
        }

        // Slow path: mint a fresh token without holding the cache lock. Two
        // concurrent calls for the same fingerprint will both generate and the
        // last writer wins — acceptable for a CPU-only computation that runs
        // at most once per [`REFRESH_THRESHOLD_SECS`] per fingerprint.
        //
        // `effective` clamps the requested expiry to [`MAX_EXPIRES_SECS`] so
        // the cache entry's `expires_at_unix` matches the lifetime AWS
        // actually grants the token (the generator performs the same clamp
        // before embedding `X-Amz-Expires`). Without this, a config requesting
        // `expires > MAX` would store a cache entry that outlives the token,
        // serving stale (AWS-rejected) tokens for hours.
        let effective = material.expires_secs.min(MAX_EXPIRES_SECS);
        let expires_at_unix = now.saturating_add(effective);
        let token = generate_bedrock_mantle_token(
            &material.access_key,
            &material.secret_key,
            material.session_token.as_deref(),
            &material.region,
            effective,
            now,
        )
        .map_err(Error::Config)?;
        {
            let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
            // Sweep expired entries to bound memory under credential rotation
            // (STS/web-identity tokens rotate frequently; without eviction each
            // rotation would leave another stale entry — and its cached bearer
            // token — sitting in the map for the process lifetime). Cheaper
            // than an LRU: a single `retain` over a map whose natural size is
            // O(distinct deployments).
            cache.retain(|_, v| v.expires_at_unix > now);
            cache.insert(
                fp,
                CachedToken {
                    token: token.clone(),
                    expires_at_unix,
                },
            );
        }
        Ok(ResolvedCredentials::ApiKey { bearer: token })
    }
}

/// Resolve the raw AWS credential material for `deployment`.
///
/// The access-key / secret-key / session-token triple is resolved via the
/// shared [`resolve_access_secret_session`] helper (same precedence as the
/// Bedrock native resolver), so future additions land in one place. Region
/// and `expires_secs` are mantle-specific and resolved here: region consults
/// `aws_region_name` (canonical) then `aws_region`, then
/// `BEDROCK_MANTLE_REGION` so a Mantle deployment can target a different
/// region than the ambient AWS config.
fn resolve_aws_material(
    deployment: &Deployment,
    env_lookup: &EnvLookup,
) -> Result<AwsMaterial, Error> {
    let extra = &deployment.oxllm_params.extra;

    let (access_key, secret_key, session_token) = resolve_access_secret_session(
        deployment,
        env_lookup,
        "missing AWS credentials for bedrock_mantle: set aws_access_key_id and \
         aws_secret_access_key (or the AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY env vars), \
         or set api_key (or the BEDROCK_MANTLE_API_KEY / AWS_BEARER_TOKEN_BEDROCK env vars) \
         to a pre-built bedrock-api-key token",
    )?;

    let region = deployment
        .oxllm_params
        .api_base
        .as_deref()
        .and_then(mantle_region_from_api_base)
        .or_else(|| str_from_extra(extra, "aws_region_name"))
        .or_else(|| str_from_extra(extra, "aws_region"))
        .or_else(|| env_lookup("BEDROCK_MANTLE_REGION"))
        .or_else(|| env_lookup("AWS_REGION_NAME"))
        .or_else(|| env_lookup("AWS_REGION"))
        .or_else(|| env_lookup("AWS_DEFAULT_REGION"))
        .unwrap_or_else(|| "us-east-1".to_string());
    validate_aws_region(&region).map_err(Error::Config)?;

    let expires_secs = match extra.get("aws_bedrock_token_expires") {
        None => DEFAULT_EXPIRES_SECS,
        // Reject malformed values (non-numeric strings, negatives, floats)
        // instead of silently falling back to DEFAULT_EXPIRES_SECS — a typo
        // like `"12h"` or `-1` is a config error and should surface as such,
        // matching the zero-rejection path below.
        Some(v) => num_from_value(v).ok_or_else(|| {
            Error::Config("aws_bedrock_token_expires must be a non-negative integer".to_string())
        })?,
    };

    // A zero expiry embeds `X-Amz-Expires=0`, which AWS rejects, and would
    // also force a token regeneration on every request (the token is already
    // inside the refresh window the instant it is minted). Treat it as a
    // misconfiguration rather than minting an unusable token.
    if expires_secs == 0 {
        return Err(Error::Config(
            "aws_bedrock_token_expires must be greater than 0".to_string(),
        ));
    }

    Ok(AwsMaterial {
        access_key,
        secret_key,
        session_token,
        region,
        expires_secs,
    })
}

/// Official Mantle endpoints pin the token scope to their host region.
/// Arbitrary custom bases leave region resolution to deployment/env settings.
fn mantle_region_from_api_base(base: &str) -> Option<String> {
    let url = url::Url::parse(base).ok()?;
    if !matches!(url.scheme(), "http" | "https") {
        return None;
    }
    let region = url
        .host_str()?
        .strip_prefix("bedrock-mantle.")?
        .strip_suffix(".api.aws")?;
    validate_aws_region(region).ok()?;
    Some(region.to_string())
}

/// Pull a string value out of the serde-flatten `extra` map.
fn str_from_extra(extra: &serde_json::Map<String, serde_json::Value>, key: &str) -> Option<String> {
    extra
        .get(key)
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .map(String::from)
}

/// Coerce a JSON value into `u64`, accepting both numeric literals
/// (`aws_bedrock_token_expires: 3600`) and numeric strings
/// (`"3600"`).
fn num_from_value(v: &serde_json::Value) -> Option<u64> {
    v.as_u64()
        .or_else(|| v.as_str().and_then(|s| s.parse::<u64>().ok()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn official_mantle_base_pins_the_scope_region_over_ambient_settings() {
        let mut deployment = Deployment::default();
        deployment.oxllm_params.api_base =
            Some("https://bedrock-mantle.us-west-2.api.aws/openai/v1".into());
        deployment
            .oxllm_params
            .extra
            .insert("aws_region_name".into(), serde_json::json!("us-east-1"));
        let env: EnvLookup = Arc::new(|name| match name {
            "AWS_ACCESS_KEY_ID" => Some("test-access".into()),
            "AWS_SECRET_ACCESS_KEY" => Some("test-secret".into()),
            "AWS_REGION" => Some("eu-west-1".into()),
            _ => None,
        });
        assert_eq!(
            resolve_aws_material(&deployment, &env).unwrap().region,
            "us-west-2"
        );
        assert_eq!(
            mantle_region_from_api_base("https://bedrock-mantle.us-west-2.api.aws.evil.example/v1"),
            None
        );
    }
    use oxllm_core::deployment::OxllmParams;

    /// A fixed env + clock so resolver behavior is fully deterministic.
    fn fixture_env() -> (
        impl Fn(&'static str) -> Option<String> + Clone,
        impl Fn() -> u64 + Clone,
    ) {
        let env: std::collections::HashMap<&'static str, String> = [
            ("AWS_ACCESS_KEY_ID", "ENV-AK"),
            ("AWS_SECRET_ACCESS_KEY", "ENV-SK"),
            ("AWS_SESSION_TOKEN", "ENV-TOKEN"),
            ("AWS_REGION", "eu-west-1"),
        ]
        .into_iter()
        .map(|(k, v)| (k, v.to_string()))
        .collect();
        let env_lookup = move |name: &'static str| env.get(name).cloned();
        let now = move || 1_715_690_696u64;
        (env_lookup, now)
    }

    fn deployment_with(params: OxllmParams) -> Deployment {
        Deployment {
            model_name: "m".to_string(),
            oxllm_params: params,
            ..Default::default()
        }
    }

    fn mantle_params() -> OxllmParams {
        OxllmParams {
            model: "bedrock_mantle/openai.gpt-oss-120b".to_string(),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn explicit_api_key_is_returned_verbatim() {
        // Phase 1 path: a pre-built token short-circuits auto-generation.
        let (env, now) = fixture_env();
        let resolver = BedrockMantleCredentialResolver::with_env(env, now);
        let mut params = mantle_params();
        params.api_key = Some("bedrock-api-key-prebuilt".to_string());
        let creds = resolver
            .resolve(&deployment_with(params))
            .await
            .expect("ok");
        match creds {
            ResolvedCredentials::ApiKey { bearer } => {
                assert_eq!(bearer, "bedrock-api-key-prebuilt");
            }
            other => panic!("expected ApiKey, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn generates_token_from_typed_credentials() {
        let (env, now) = fixture_env();
        let resolver = BedrockMantleCredentialResolver::with_env(env, now);
        let mut params = mantle_params();
        params.aws_access_key_id = Some("AKIAIOSFODNN7EXAMPLE".to_string());
        params.aws_secret_access_key = Some("wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY".to_string());
        params.aws_session_token = Some("session-tok".to_string());
        let creds = resolver
            .resolve(&deployment_with(params))
            .await
            .expect("ok");
        match creds {
            ResolvedCredentials::ApiKey { bearer } => {
                assert!(bearer.starts_with("bedrock-api-key-"), "got {bearer}");
            }
            other => panic!("expected ApiKey, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn falls_back_to_environment_credentials() {
        let (env, now) = fixture_env();
        let resolver = BedrockMantleCredentialResolver::with_env(env, now);
        let creds = resolver
            .resolve(&deployment_with(mantle_params()))
            .await
            .expect("env creds available");
        match creds {
            ResolvedCredentials::ApiKey { bearer } => {
                // The token embeds the env access key in its credential scope.
                let payload = decode(&bearer);
                assert!(
                    payload.contains("ENV-AK"),
                    "expected env access key: {payload}"
                );
                assert!(
                    payload.contains("eu-west-1"),
                    "region should come from AWS_REGION: {payload}"
                );
                assert!(
                    payload.contains("X-Amz-Security-Token=ENV-TOKEN"),
                    "session token from env: {payload}"
                );
            }
            other => panic!("expected ApiKey, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn typed_credentials_override_environment() {
        let (env, now) = fixture_env();
        let resolver = BedrockMantleCredentialResolver::with_env(env, now);
        let mut params = mantle_params();
        params.aws_access_key_id = Some("DEP-AK".to_string());
        params.aws_secret_access_key = Some("dep-secret".to_string());
        let creds = resolver
            .resolve(&deployment_with(params))
            .await
            .expect("ok");
        match creds {
            ResolvedCredentials::ApiKey { bearer } => {
                let payload = decode(&bearer);
                assert!(
                    payload.contains("DEP-AK"),
                    "typed field must win: {payload}"
                );
                assert!(
                    !payload.contains("ENV-AK"),
                    "env value must not leak: {payload}"
                );
            }
            other => panic!("expected ApiKey, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn region_prefers_bedrock_mantle_region_env() {
        let mut env: std::collections::HashMap<&'static str, String> =
            std::collections::HashMap::new();
        env.insert("AWS_ACCESS_KEY_ID", "AK".into());
        env.insert("AWS_SECRET_ACCESS_KEY", "sk".into());
        env.insert("AWS_REGION", "us-east-1".into());
        env.insert("BEDROCK_MANTLE_REGION", "ap-south-1".into());
        let env_lookup = move |name: &'static str| env.get(name).cloned();
        let now = || 1_715_690_696u64;
        let resolver = BedrockMantleCredentialResolver::with_env(env_lookup, now);
        let creds = resolver
            .resolve(&deployment_with(mantle_params()))
            .await
            .expect("ok");
        match creds {
            ResolvedCredentials::ApiKey { bearer } => {
                let payload = decode(&bearer);
                assert!(
                    payload.contains("ap-south-1"),
                    "BEDROCK_MANTLE_REGION must win: {payload}"
                );
            }
            other => panic!("expected ApiKey, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn errors_when_no_credentials_available() {
        let env = |_: &'static str| None::<String>;
        let now = || 1_715_690_696u64;
        let resolver = BedrockMantleCredentialResolver::with_env(env, now);
        let err = resolver
            .resolve(&deployment_with(mantle_params()))
            .await
            .expect_err("no creds");
        assert!(matches!(err, Error::Config(_)), "got {err:?}");
    }

    /// The reference `BedrockMantleAuthMixin._resolve_bearer_token` falls back to
    /// `BEDROCK_MANTLE_API_KEY` then `AWS_BEARER_TOKEN_BEDROCK` when no
    /// `api_key` is configured (`common_utils.py`). A config migrated from
    /// an existing proxy that relies on the env var must resolve to that exact bearer
    /// token rather than silently minting a different one from AWS
    /// credentials (or erroring with "missing AWS credentials").
    #[tokio::test]
    async fn bedrock_mantle_api_key_env_var_is_used_as_bearer() {
        let env: std::collections::HashMap<&'static str, String> =
            [("BEDROCK_MANTLE_API_KEY", "prebuilt-bearer".to_string())]
                .into_iter()
                .collect();
        let env_lookup = move |name: &'static str| env.get(name).cloned();
        let now = move || 1_715_690_696u64;
        let resolver = BedrockMantleCredentialResolver::with_env(env_lookup, now);
        let creds = resolver
            .resolve(&deployment_with(mantle_params()))
            .await
            .expect("env bearer token must resolve");
        match creds {
            ResolvedCredentials::ApiKey { bearer } => {
                assert_eq!(
                    bearer, "prebuilt-bearer",
                    "BEDROCK_MANTLE_API_KEY must be returned verbatim, not auto-generated"
                );
            }
            other => panic!("expected ApiKey, got {other:?}"),
        }
    }

    /// The secondary alias `AWS_BEARER_TOKEN_BEDROCK` is consulted only when
    /// `BEDROCK_MANTLE_API_KEY` is absent (matching the reference `or` chain).
    #[tokio::test]
    async fn aws_bearer_token_bedrock_env_var_is_used_as_bearer() {
        let env: std::collections::HashMap<&'static str, String> =
            [("AWS_BEARER_TOKEN_BEDROCK", "alias-bearer".to_string())]
                .into_iter()
                .collect();
        let env_lookup = move |name: &'static str| env.get(name).cloned();
        let now = move || 1_715_690_696u64;
        let resolver = BedrockMantleCredentialResolver::with_env(env_lookup, now);
        let creds = resolver
            .resolve(&deployment_with(mantle_params()))
            .await
            .expect("alias bearer token must resolve");
        match creds {
            ResolvedCredentials::ApiKey { bearer } => assert_eq!(bearer, "alias-bearer"),
            other => panic!("expected ApiKey, got {other:?}"),
        }
    }

    /// An explicit `api_key` in config wins over the env-var fallbacks.
    #[tokio::test]
    async fn api_key_config_wins_over_env_bearer() {
        let env: std::collections::HashMap<&'static str, String> =
            [("BEDROCK_MANTLE_API_KEY", "env-bearer".to_string())]
                .into_iter()
                .collect();
        let env_lookup = move |name: &'static str| env.get(name).cloned();
        let now = move || 1_715_690_696u64;
        let resolver = BedrockMantleCredentialResolver::with_env(env_lookup, now);
        let mut params = mantle_params();
        params.api_key = Some("config-token".to_string());
        let creds = resolver
            .resolve(&deployment_with(params))
            .await
            .expect("ok");
        match creds {
            ResolvedCredentials::ApiKey { bearer } => assert_eq!(bearer, "config-token"),
            other => panic!("expected ApiKey, got {other:?}"),
        }
    }

    /// The canonical `aws_region_name` config key must resolve the
    /// region for a mantle deployment (parity with the native resolver).
    #[tokio::test]
    async fn per_deployment_aws_region_name_key_is_honoured() {
        let env: std::collections::HashMap<&'static str, String> = [
            ("AWS_ACCESS_KEY_ID", "AK".into()),
            ("AWS_SECRET_ACCESS_KEY", "sk".into()),
            ("AWS_REGION", "us-east-1".into()),
        ]
        .into_iter()
        .collect();
        let env_lookup = move |name: &'static str| env.get(name).cloned();
        let now = move || 1_715_690_696u64;
        let resolver = BedrockMantleCredentialResolver::with_env(env_lookup, now);

        let mut params = mantle_params();
        params.aws_access_key_id = Some("AK".into());
        params.aws_secret_access_key = Some("sk".into());
        let mut extra = serde_json::Map::new();
        extra.insert(
            "aws_region_name".to_string(),
            serde_json::Value::String("eu-west-2".to_string()),
        );
        params.extra = extra;

        let creds = resolver
            .resolve(&deployment_with(params))
            .await
            .expect("ok");
        match creds {
            ResolvedCredentials::ApiKey { bearer } => {
                let payload = decode(&bearer);
                assert!(
                    payload.contains("eu-west-2"),
                    "aws_region_name config key must resolve: {payload}"
                );
            }
            other => panic!("expected ApiKey, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn empty_region_extra_falls_through_to_env() {
        // An explicitly empty `aws_region_name` must NOT short-circuit the
        // precedence chain; we should fall through to AWS_REGION instead of
        // failing region validation.
        let env: std::collections::HashMap<&'static str, String> = [
            ("AWS_ACCESS_KEY_ID", "AK".into()),
            ("AWS_SECRET_ACCESS_KEY", "sk".into()),
            ("AWS_REGION", "us-east-1".into()),
        ]
        .into_iter()
        .collect();
        let env_lookup = move |name: &'static str| env.get(name).cloned();
        let now = move || 1_715_690_696u64;
        let resolver = BedrockMantleCredentialResolver::with_env(env_lookup, now);

        let mut params = mantle_params();
        params.aws_access_key_id = Some("AK".into());
        params.aws_secret_access_key = Some("sk".into());
        let mut extra = serde_json::Map::new();
        extra.insert(
            "aws_region_name".to_string(),
            serde_json::Value::String(String::new()),
        );
        params.extra = extra;

        let creds = resolver
            .resolve(&deployment_with(params))
            .await
            .expect("empty region must fall through to env");
        match creds {
            ResolvedCredentials::ApiKey { bearer } => {
                let payload = decode(&bearer);
                assert!(
                    payload.contains("us-east-1"),
                    "region should come from AWS_REGION: {payload}"
                );
            }
            other => panic!("expected ApiKey, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn caches_token_across_calls() {
        // A non-mutating clock pins `now`, so the second resolve must reuse the
        // cached token (byte-identical) instead of regenerating.
        let (env, now) = fixture_env();
        let resolver = BedrockMantleCredentialResolver::with_env(env, now);
        let first = match resolver
            .resolve(&deployment_with(mantle_params()))
            .await
            .unwrap()
        {
            ResolvedCredentials::ApiKey { bearer } => bearer,
            other => panic!("{other:?}"),
        };
        let second = match resolver
            .resolve(&deployment_with(mantle_params()))
            .await
            .unwrap()
        {
            ResolvedCredentials::ApiKey { bearer } => bearer,
            other => panic!("{other:?}"),
        };
        assert_eq!(first, second, "cached token must be reused");
    }

    #[tokio::test]
    async fn refreshes_when_within_threshold() {
        // Request a short expiry so the token falls inside the refresh window
        // on the very next call (now + 300 == expires_at → not strictly less).
        let (env, _) = fixture_env();
        let now_cell = Arc::new(Mutex::new(1_715_690_696u64));
        let now_clone = now_cell.clone();
        let now_fn = move || *now_clone.lock().unwrap();
        let resolver = BedrockMantleCredentialResolver::with_env(env, now_fn);

        let mut params = mantle_params();
        params.aws_access_key_id = Some("AK".into());
        params.aws_secret_access_key = Some("sk".into());
        let mut extra = serde_json::Map::new();
        extra.insert(
            "aws_bedrock_token_expires".to_string(),
            serde_json::Value::Number(serde_json::Number::from(100u64)),
        );
        params.extra = extra;

        let first = bearer(&resolver, params.clone()).await;
        // Advance the clock past the refresh threshold (token valid for 100s,
        // threshold 300s → now+300 >= expires_at → must regenerate).
        *now_cell.lock().unwrap() += 100;
        let second = bearer(&resolver, params).await;

        assert_ne!(
            first, second,
            "token must be regenerated once inside the refresh window"
        );
        // And the new token is itself cached (third call is stable).
        let third = bearer(
            &resolver,
            OxllmParams {
                model: "bedrock_mantle/m".to_string(),
                aws_access_key_id: Some("AK".into()),
                aws_secret_access_key: Some("sk".into()),
                extra: [(
                    "aws_bedrock_token_expires".to_string(),
                    serde_json::Value::Number(serde_json::Number::from(100u64)),
                )]
                .into_iter()
                .collect(),
                ..Default::default()
            },
        )
        .await;
        assert_eq!(second, third, "refreshed token must be cached");
    }

    #[tokio::test]
    async fn cache_invalidated_when_credentials_change() {
        // Same clock, different credentials → different fingerprint → fresh
        // token (no stale leak across deployments).
        let (env, now) = fixture_env();
        let resolver = BedrockMantleCredentialResolver::with_env(env, now);

        let mut a = mantle_params();
        a.aws_access_key_id = Some("AK-A".into());
        a.aws_secret_access_key = Some("sk".into());
        let mut b = mantle_params();
        b.aws_access_key_id = Some("AK-B".into());
        b.aws_secret_access_key = Some("sk".into());

        let ta = bearer(&resolver, a).await;
        let tb = bearer(&resolver, b).await;
        assert_ne!(ta, tb, "different credentials must yield different tokens");
        assert!(decode(&ta).contains("AK-A"));
        assert!(decode(&tb).contains("AK-B"));
    }

    /// A rotation of ONLY the secret key must invalidate the cache. The
    /// fingerprint includes a non-reversible digest of the secret, so the old
    /// (secret-one) token is never served after the secret changes.
    #[tokio::test]
    async fn cache_invalidates_on_secret_only_rotation() {
        let (env, now) = fixture_env();
        let resolver = BedrockMantleCredentialResolver::with_env(env, now);

        let mk = |secret: &str| {
            let mut p = mantle_params();
            p.aws_access_key_id = Some("AK".into());
            p.aws_secret_access_key = Some(secret.to_string());
            p
        };

        let t1 = bearer(&resolver, mk("secret-one")).await;
        // Same secret → cached (byte-identical) under the fixed clock.
        let t1_again = bearer(&resolver, mk("secret-one")).await;
        assert_eq!(t1, t1_again, "unchanged creds must hit the cache");
        // Rotate ONLY the secret → different fingerprint → fresh token derived
        // from secret-two. The old fingerprint (which excluded the secret)
        // would have served the stale secret-one token here.
        let t2 = bearer(&resolver, mk("secret-two")).await;
        assert_ne!(t1, t2, "a secret-only rotation must invalidate the cache");
        assert_ne!(
            decode(&t1).rsplit("X-Amz-Signature=").next().unwrap_or(""),
            decode(&t2).rsplit("X-Amz-Signature=").next().unwrap_or(""),
            "the two tokens must carry different signatures (different secrets)"
        );
    }

    /// A rotation of ONLY the session token must invalidate the cache. The
    /// fingerprint digests the session token (rather than embedding it raw,
    /// which would leak a bearer secret into heap memory) — verify the digest
    /// still distinguishes distinct tokens.
    #[tokio::test]
    async fn cache_invalidates_on_session_only_rotation() {
        let (env, now) = fixture_env();
        let resolver = BedrockMantleCredentialResolver::with_env(env, now);

        let mk = |session: &str| {
            let mut p = mantle_params();
            p.aws_access_key_id = Some("AK".into());
            p.aws_secret_access_key = Some("sk".into());
            p.aws_session_token = Some(session.to_string());
            p
        };

        let t1 = bearer(&resolver, mk("session-one")).await;
        let t1_again = bearer(&resolver, mk("session-one")).await;
        assert_eq!(t1, t1_again, "unchanged session must hit the cache");
        let t2 = bearer(&resolver, mk("session-two")).await;
        assert_ne!(t1, t2, "a session-only rotation must invalidate the cache");
    }

    /// The fingerprint must never embed raw credential material — the
    /// access-key id (which identifies the account), the secret key, and the
    /// AWS STS session token (a short-lived bearer secret) are all
    /// one-way-digested. A digest distinguishes distinct values without
    /// exposing them in any future `dbg!`/core-dump path.
    #[test]
    fn fingerprint_does_not_contain_raw_credential_material() {
        let material = AwsMaterial {
            access_key: "AKIA-ACCESS-KEY-ID".into(),
            secret_key: "sk".into(),
            session_token: Some("super-secret-session-token".into()),
            region: "us-east-1".into(),
            expires_secs: 3600,
        };
        let fp = material.fingerprint();
        assert!(
            !fp.contains("super-secret-session-token"),
            "raw session token leaked into fingerprint: {fp}"
        );
        let session_digest = hex_encode(&sha256(b"super-secret-session-token"));
        assert!(
            fp.contains(&session_digest),
            "expected the SHA-256 digest of the session token in the fingerprint: {fp}"
        );
        // Sanity: the secret key digest is also present, not the raw secret.
        assert!(!fp.contains("sk"), "raw secret key substring leaked: {fp}");
        // The access-key id is digested too — never embedded verbatim.
        assert!(
            !fp.contains("AKIA-ACCESS-KEY-ID"),
            "raw access-key id leaked into fingerprint: {fp}"
        );
        let access_digest = hex_encode(&sha256(b"AKIA-ACCESS-KEY-ID"));
        assert!(
            fp.contains(&access_digest),
            "expected the SHA-256 digest of the access-key id in the fingerprint: {fp}"
        );
    }

    /// The fingerprint clamps `expires_secs` to [`MAX_EXPIRES_SECS`] before
    /// formatting, so two configs that both request `>= MAX_EXPIRES_SECS`
    /// produce identical tokens AND identical fingerprints. The second
    /// resolve must hit the cache (byte-identical token) instead of
    /// regenerating.
    #[tokio::test]
    async fn two_over_max_expires_share_a_cache_slot() {
        let (env, now) = fixture_env();
        let resolver = BedrockMantleCredentialResolver::with_env(env, now);

        let mk = |expires: u64| {
            let mut p = mantle_params();
            p.aws_access_key_id = Some("AK".into());
            p.aws_secret_access_key = Some("sk".into());
            let mut extra = serde_json::Map::new();
            extra.insert(
                "aws_bedrock_token_expires".to_string(),
                serde_json::Value::Number(serde_json::Number::from(expires)),
            );
            p.extra = extra;
            p
        };

        let t1 = bearer(&resolver, mk(MAX_EXPIRES_SECS + 5_000)).await;
        let t2 = bearer(&resolver, mk(MAX_EXPIRES_SECS + 10_000)).await;
        assert_eq!(
            t1, t2,
            "both expires > MAX must clamp to MAX and share a cache slot"
        );
    }

    /// When `aws_bedrock_token_expires > MAX_EXPIRES_SECS`, the generator
    /// clamps the embedded `X-Amz-Expires` to `MAX_EXPIRES_SECS`. The cache
    /// entry's `expires_at_unix` must reflect that clamped lifetime — not the
    /// raw requested one — or the cache would serve AWS-rejected stale tokens
    /// for the gap between `MAX` and the requested expiry. Advancing the clock
    /// past `MAX_EXPIRES_SECS` must therefore trigger a regeneration.
    #[tokio::test]
    async fn token_refreshes_after_max_when_expires_over_max() {
        let (env, _) = fixture_env();
        let now_cell = Arc::new(Mutex::new(1_715_690_696u64));
        let now_clone = now_cell.clone();
        let now_fn = move || *now_clone.lock().unwrap();
        let resolver = BedrockMantleCredentialResolver::with_env(env, now_fn);

        let mk = || {
            let mut p = mantle_params();
            p.aws_access_key_id = Some("AK".into());
            p.aws_secret_access_key = Some("sk".into());
            let mut extra = serde_json::Map::new();
            extra.insert(
                "aws_bedrock_token_expires".to_string(),
                serde_json::Value::Number(serde_json::Number::from(100_000u64)),
            );
            p.extra = extra;
            p
        };

        let first = bearer(&resolver, mk()).await;
        // The token embeds X-Amz-Expires=MAX_EXPIRES_SECS (clamped from
        // 100_000). Advancing the clock past MAX_EXPIRES_SECS must trigger a
        // regeneration: with the fix, the cache entry's `expires_at_unix`
        // equals `now + MAX`, so the refresh-window check fails and the slow
        // path runs. Pre-fix the entry used the un-clamped 100_000 and the
        // cache would have served the stale token until t=99_700.
        *now_cell.lock().unwrap() += MAX_EXPIRES_SECS + 1;
        let second = bearer(&resolver, mk()).await;

        assert_ne!(
            first, second,
            "token must regenerate after MAX_EXPIRES_SECS even when expires > MAX"
        );
    }

    /// A malformed `aws_bedrock_token_expires` (non-numeric string) must
    /// surface as a config error rather than silently falling back to the
    /// 12-hour default. Matches the existing zero-rejection path.
    #[tokio::test]
    async fn errors_on_non_numeric_expires() {
        let (env, now) = fixture_env();
        let resolver = BedrockMantleCredentialResolver::with_env(env, now);

        let mut params = mantle_params();
        params.aws_access_key_id = Some("AK".into());
        params.aws_secret_access_key = Some("sk".into());
        let mut extra = serde_json::Map::new();
        extra.insert(
            "aws_bedrock_token_expires".to_string(),
            serde_json::Value::String("12h".into()),
        );
        params.extra = extra;

        let err = resolver
            .resolve(&deployment_with(params))
            .await
            .expect_err("non-numeric expires must be rejected");
        assert!(matches!(err, Error::Config(_)), "got {err:?}");
        assert!(
            format!("{err}").contains("aws_bedrock_token_expires"),
            "error should name the offending setting: {err}"
        );
    }

    /// A negative-string `aws_bedrock_token_expires` is rejected on the same
    /// path as the non-numeric case (the parser refuses, surfacing as
    /// `Error::Config`).
    #[tokio::test]
    async fn errors_on_negative_string_expires() {
        let (env, now) = fixture_env();
        let resolver = BedrockMantleCredentialResolver::with_env(env, now);

        let mut params = mantle_params();
        params.aws_access_key_id = Some("AK".into());
        params.aws_secret_access_key = Some("sk".into());
        let mut extra = serde_json::Map::new();
        extra.insert(
            "aws_bedrock_token_expires".to_string(),
            serde_json::Value::String("-1".into()),
        );
        params.extra = extra;

        let err = resolver
            .resolve(&deployment_with(params))
            .await
            .expect_err("negative expires must be rejected");
        assert!(matches!(err, Error::Config(_)), "got {err:?}");
    }

    /// An invalid AWS region must be rejected up front (mirrors the reference
    /// `_validate_aws_region_name`). A typo like `"us-east-1 "` (trailing
    /// space) would otherwise produce an obscure `SignatureDoesNotMatch`.
    #[tokio::test]
    async fn errors_on_invalid_aws_region() {
        let env: std::collections::HashMap<&'static str, String> = [
            ("AWS_ACCESS_KEY_ID", "AK"),
            ("AWS_SECRET_ACCESS_KEY", "sk"),
            ("AWS_REGION", "us-east-1 "),
        ]
        .into_iter()
        .map(|(k, v)| (k, v.to_string()))
        .collect();
        let env_lookup = move |name: &'static str| env.get(name).cloned();
        let now = move || 1_715_690_696u64;
        let resolver = BedrockMantleCredentialResolver::with_env(env_lookup, now);

        let err = resolver
            .resolve(&deployment_with(mantle_params()))
            .await
            .expect_err("invalid region must be rejected");
        assert!(matches!(err, Error::Config(_)), "got {err:?}");
        // Defense in depth: the raw region must not appear in the message.
        assert!(
            !format!("{err}").contains("us-east-1 "),
            "raw region leaked: {err}"
        );
    }

    /// Cache eviction: a stale entry (past expiry) must be swept on the next
    /// slow-path resolve, bounding the cache at O(distinct live tokens).
    #[tokio::test]
    async fn cache_evicts_expired_entries_on_slow_path() {
        let env: std::collections::HashMap<&'static str, String> = [
            ("AWS_ACCESS_KEY_ID", "ENV-AK"),
            ("AWS_SECRET_ACCESS_KEY", "ENV-SK"),
        ]
        .into_iter()
        .map(|(k, v)| (k, v.to_string()))
        .collect();
        let env_lookup = move |name: &'static str| env.get(name).cloned();
        let now_cell = Arc::new(Mutex::new(1_715_690_696u64));
        let now_clone = now_cell.clone();
        let now_fn = move || *now_clone.lock().unwrap();
        let resolver = BedrockMantleCredentialResolver::with_env(env_lookup, now_fn);

        let mk = |ak: &str, sk: &str, expires: u64| {
            let mut p = mantle_params();
            p.aws_access_key_id = Some(ak.into());
            p.aws_secret_access_key = Some(sk.into());
            let mut extra = serde_json::Map::new();
            extra.insert(
                "aws_bedrock_token_expires".to_string(),
                serde_json::Value::Number(serde_json::Number::from(expires)),
            );
            p.extra = extra;
            p
        };

        // Resolve A with a short expiry; advance the clock past expiry; then
        // resolve B (slow path) — A must be evicted by the sweep.
        let _ta = bearer(&resolver, mk("AK-A", "sk-a", 60)).await;
        assert_eq!(resolver.cache.lock().unwrap().len(), 1);

        *now_cell.lock().unwrap() += 120; // A is now expired.
        let _tb = bearer(&resolver, mk("AK-B", "sk-b", 60)).await;

        let cache_size = resolver.cache.lock().unwrap().len();
        assert_eq!(
            cache_size, 1,
            "expired entry must be swept, leaving only B (got {cache_size})"
        );
    }

    /// Concurrent resolves for distinct fingerprints must not panic /
    /// deadlock and must each produce a valid token. Guards against the
    /// lock-hold regression where the slow path used to keep the Mutex locked
    /// across token generation (which would have serialized distinct
    /// fingerprints on the slow path).
    #[tokio::test]
    async fn concurrent_resolves_for_distinct_fingerprints_complete() {
        let env: std::collections::HashMap<&'static str, String> = [
            ("AWS_ACCESS_KEY_ID", "ENV-AK"),
            ("AWS_SECRET_ACCESS_KEY", "ENV-SK"),
        ]
        .into_iter()
        .map(|(k, v)| (k, v.to_string()))
        .collect();
        let env_lookup = move |name: &'static str| env.get(name).cloned();
        let now = move || 1_715_690_696u64;
        let resolver =
            std::sync::Arc::new(BedrockMantleCredentialResolver::with_env(env_lookup, now));

        let mk = |n: usize| {
            let mut p = mantle_params();
            p.aws_access_key_id = Some(format!("AK-{n}"));
            p.aws_secret_access_key = Some(format!("sk-{n}"));
            p
        };

        let handles: Vec<_> = (0..8)
            .map(|n| {
                let resolver = resolver.clone();
                tokio::spawn(async move { bearer(&resolver, mk(n)).await })
            })
            .collect();
        for h in handles {
            let token = h.await.expect("task panicked");
            assert!(token.starts_with("bedrock-api-key-"), "got {token}");
        }
        // All 8 distinct fingerprints cached.
        assert_eq!(resolver.cache.lock().unwrap().len(), 8);
    }

    /// Two distinct deployments resolved in alternation must each hit their
    /// own cache slot. A single-slot cache would evict A when B is resolved,
    /// forcing a regeneration. The monotonic clock makes a regeneration
    /// observable: a cached token keeps its original timestamp, a regenerated
    /// one embeds a later timestamp and therefore differs.
    #[tokio::test]
    async fn cache_serves_multiple_deployments_without_thrash() {
        let env: std::collections::HashMap<&'static str, String> = [
            ("AWS_ACCESS_KEY_ID", "ENV-AK"),
            ("AWS_SECRET_ACCESS_KEY", "ENV-SK"),
        ]
        .into_iter()
        .map(|(k, v)| (k, v.to_string()))
        .collect();
        let env_lookup = move |name: &'static str| env.get(name).cloned();
        let counter = Arc::new(Mutex::new(1_715_690_696u64));
        let counter_clone = counter.clone();
        let now = move || {
            let v = *counter_clone.lock().unwrap();
            *counter_clone.lock().unwrap() += 1;
            v
        };
        let resolver = BedrockMantleCredentialResolver::with_env(env_lookup, now);

        let mk = |ak: &str, sk: &str| {
            let mut p = mantle_params();
            p.aws_access_key_id = Some(ak.to_string());
            p.aws_secret_access_key = Some(sk.to_string());
            p
        };

        let a = mk("AK-A", "sk-a");
        let b = mk("AK-B", "sk-b");

        let a1 = bearer(&resolver, a.clone()).await;
        let b1 = bearer(&resolver, b.clone()).await;
        // A must still be cached after B's resolve: the token is returned
        // byte-identical. A single-slot cache would have evicted A and
        // regenerated it under the now-advanced clock → a different token.
        let a2 = bearer(&resolver, a).await;
        assert_eq!(a1, a2, "deployment A's token must survive a B resolve");
        let b2 = bearer(&resolver, b).await;
        assert_eq!(b1, b2, "deployment B's token must survive an A resolve");
    }

    /// A zero `aws_bedrock_token_expires` is a misconfiguration: it embeds
    /// `X-Amz-Expires=0` (rejected by AWS) and forces a regeneration every
    /// request. Reject it up front.
    #[tokio::test]
    async fn errors_on_zero_token_expiry() {
        let (env, now) = fixture_env();
        let resolver = BedrockMantleCredentialResolver::with_env(env, now);

        let mut params = mantle_params();
        params.aws_access_key_id = Some("AK".into());
        params.aws_secret_access_key = Some("sk".into());
        let mut extra = serde_json::Map::new();
        extra.insert(
            "aws_bedrock_token_expires".to_string(),
            serde_json::Value::Number(serde_json::Number::from(0u64)),
        );
        params.extra = extra;

        let err = resolver
            .resolve(&deployment_with(params))
            .await
            .expect_err("zero expiry must be rejected");
        assert!(matches!(err, Error::Config(_)), "got {err:?}");
        assert!(
            format!("{err}").contains("aws_bedrock_token_expires"),
            "error should name the offending setting: {err}"
        );
    }

    async fn bearer(resolver: &BedrockMantleCredentialResolver, params: OxllmParams) -> String {
        match resolver.resolve(&deployment_with(params)).await.unwrap() {
            ResolvedCredentials::ApiKey { bearer } => bearer,
            other => panic!("{other:?}"),
        }
    }

    fn decode(token: &str) -> String {
        use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64_STANDARD};
        let payload = token.strip_prefix("bedrock-api-key-").unwrap();
        String::from_utf8(BASE64_STANDARD.decode(payload).unwrap()).unwrap()
    }
}
