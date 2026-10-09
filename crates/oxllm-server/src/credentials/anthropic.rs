//! Anthropic federation from deployment-owned OIDC file or environment references.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use moka::future::Cache;
use oxllm_core::provider::ResolvedCredentials;
use oxllm_core::{Deployment, Error};
use serde::Serialize;
use serde_json::{Value, json};
use zeroize::Zeroizing;

use super::token_exchange::{MAX_CREDENTIAL_BYTES, TokenExchange};
use super::{CredentialResolver, StaticKeyResolver};

const TRUSTED_HOSTS_ENV: &str = "OXLLM_ANTHROPIC_WIF_ALLOWED_HOSTS";
const CREDENTIAL_DIRS_ENV: &str = "OXLLM_OIDC_ALLOWED_CREDENTIAL_DIRS";

/// Resolves static Anthropic keys or cached, short-lived federation grants.
/// Custom identity acquisition can be supplied through the resolver registry.
pub struct AnthropicResolver {
    exchanges: Cache<String, Arc<TokenExchange>>,
    trusted_hosts: Vec<(String, Option<u16>)>,
    credential_dirs: Vec<PathBuf>,
    allow_http: bool,
    use_env: bool,
}

impl AnthropicResolver {
    /// Static keys retain precedence. Federation exchanges trust Anthropic's
    /// host and any exact authorities explicitly configured by the server.
    pub fn new() -> Self {
        let mut trusted_hosts = vec![("api.anthropic.com".into(), None)];
        if let Ok(value) = std::env::var(TRUSTED_HOSTS_ENV) {
            for entry in value
                .split(',')
                .map(str::trim)
                .filter(|entry| !entry.is_empty())
            {
                if let Some(authority) = allowlisted_authority(entry) {
                    trusted_hosts.push(authority);
                }
            }
        }
        let credential_dirs = std::env::var_os(CREDENTIAL_DIRS_ENV)
            .map(|value| std::env::split_paths(&value).collect())
            .unwrap_or_else(|| {
                vec![
                    PathBuf::from("/var/run/secrets"),
                    PathBuf::from("/run/secrets"),
                ]
            });
        Self {
            exchanges: Cache::builder()
                .max_capacity(16)
                .time_to_idle(Duration::from_secs(3600))
                .build(),
            trusted_hosts,
            credential_dirs,
            allow_http: false,
            use_env: true,
        }
    }

    fn value(&self, deployment: &Deployment, field: &str, env: &str) -> Option<String> {
        parameter_value(deployment, field)
            .map(str::to_owned)
            .or_else(|| {
                self.use_env
                    .then(|| std::env::var(env).ok())
                    .flatten()
                    .filter(|value| !value.trim().is_empty())
            })
    }

    fn endpoint(&self, deployment: &Deployment) -> Result<String, Error> {
        let mut url = url::Url::parse(
            deployment
                .oxllm_params
                .api_base
                .as_deref()
                .unwrap_or("https://api.anthropic.com/v1"),
        )
        .map_err(|_| denied("invalid Anthropic federation API base"))?;
        if (url.scheme() != "https" && !(self.allow_http && url.scheme() == "http"))
            || !url.username().is_empty()
            || url.password().is_some()
        {
            return Err(denied(
                "Anthropic federation requires an HTTPS API base without embedded credentials",
            ));
        }
        if !self.trusted_hosts.iter().any(|(host, port)| {
            url.host_str() == Some(host.as_str())
                && port.is_none_or(|port| url.port_or_known_default() == Some(port))
        }) {
            return Err(denied(
                "Anthropic federation API host is not in OXLLM_ANTHROPIC_WIF_ALLOWED_HOSTS",
            ));
        }
        let mut path = url.path().trim_end_matches('/').to_string();
        loop {
            let stripped = ["/v1/messages", "/v1"]
                .iter()
                .find_map(|suffix| path.strip_suffix(suffix));
            let Some(stripped) = stripped else {
                break;
            };
            path = stripped.trim_end_matches('/').to_string();
        }
        url.set_path(&format!("{path}/v1/oauth/token"));
        url.set_query(None);
        url.set_fragment(None);
        Ok(url.to_string())
    }
}

impl Default for AnthropicResolver {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl CredentialResolver for AnthropicResolver {
    async fn resolve(&self, deployment: &Deployment) -> Result<ResolvedCredentials, Error> {
        deployment.oxllm_params.validate_auth_overrides()?;
        if deployment.oxllm_params.api_key.is_some()
            || deployment
                .oxllm_params
                .extra
                .get("anthropic_disable_workload_identity_federation")
                .and_then(Value::as_bool)
                == Some(true)
        {
            return StaticKeyResolver.resolve(deployment).await;
        }
        let Some(source) = select_identity_source(deployment, |name| {
            self.use_env.then(|| std::env::var(name).ok()).flatten()
        }) else {
            return StaticKeyResolver.resolve(deployment).await;
        };
        let rule = self.value(
            deployment,
            "anthropic_federation_rule_id",
            "ANTHROPIC_FEDERATION_RULE_ID",
        );
        let organization = self.value(
            deployment,
            "anthropic_organization_id",
            "ANTHROPIC_ORGANIZATION_ID",
        );
        let missing: Vec<_> = [
            ("anthropic_federation_rule_id", rule.is_none()),
            ("anthropic_organization_id", organization.is_none()),
        ]
        .into_iter()
        .filter_map(|(field, missing)| missing.then_some(field))
        .collect();
        if !missing.is_empty() {
            return Err(denied(format!(
                "Anthropic federation requires {}",
                missing.join(" and ")
            )));
        }
        let reference = match source {
            IdentitySource::Reference(reference) => reference,
            IdentitySource::Custom => {
                return Err(denied(
                    "anthropic_identity_source requires a custom registered credential resolver; the built-in resolver accepts OIDC file/env references",
                ));
            }
        };
        let identity = Identity::parse(&reference)?;
        let endpoint = self.endpoint(deployment)?;
        let mut body = json!({"grant_type":"urn:ietf:params:oauth:grant-type:jwt-bearer",
            "federation_rule_id":rule, "organization_id":organization});
        for (field, env, key) in [
            (
                "anthropic_service_account_id",
                "ANTHROPIC_SERVICE_ACCOUNT_ID",
                "service_account_id",
            ),
            (
                "anthropic_federation_workspace_id",
                "ANTHROPIC_FEDERATION_WORKSPACE_ID",
                "workspace_id",
            ),
        ] {
            if let Some(value) = self.value(deployment, field, env) {
                body[key] = json!(value);
            }
        }
        let material = serde_json::to_vec(&json!([endpoint, reference, body]))
            .map_err(|_| denied("cannot construct Anthropic federation grant"))?;
        let exchange = self
            .exchanges
            .get_with(endpoint.clone(), async {
                Arc::new(TokenExchange::at(endpoint))
            })
            .await;
        let token = exchange
            .token_json(&material, async {
                let assertion = identity.read(&self.credential_dirs).await?;
                Ok(FederationGrant {
                    fields: body,
                    assertion,
                })
            })
            .await?;
        Ok(ResolvedCredentials::PreSignedHeaders(HashMap::from([
            ("authorization".into(), format!("Bearer {token}")),
            ("anthropic-beta".into(), "oauth-2025-04-20".into()),
        ])))
    }
}

fn parameter_value<'a>(deployment: &'a Deployment, field: &str) -> Option<&'a str> {
    deployment
        .oxllm_params
        .extra
        .get(field)
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
}

#[cfg_attr(test, derive(Debug, PartialEq, Eq))]
enum IdentitySource {
    Reference(String),
    Custom,
}

fn select_identity_source(
    deployment: &Deployment,
    env: impl Fn(&str) -> Option<String>,
) -> Option<IdentitySource> {
    if deployment
        .oxllm_params
        .extra
        .get("anthropic_identity_source")
        .and_then(Value::as_str)
        .is_some_and(|kind| !kind.is_empty())
    {
        return Some(IdentitySource::Custom);
    }
    // Choose deployment-owned sources before consulting any fleet defaults.
    if let Some(file) = parameter_value(deployment, "anthropic_identity_token_file") {
        return Some(IdentitySource::Reference(format!("oidc/file/{file}")));
    }
    if let Some(reference) = parameter_value(deployment, "anthropic_identity_token") {
        return Some(IdentitySource::Reference(reference.into()));
    }
    let explicit_reference = ["anthropic_identity_token_file", "anthropic_identity_token"]
        .iter()
        .any(|field| {
            deployment
                .oxllm_params
                .extra
                .get(*field)
                .and_then(Value::as_str)
                .is_some_and(|value| !value.is_empty())
        });
    if !explicit_reference && env("ANTHROPIC_IDENTITY_SOURCE").is_some() {
        return Some(IdentitySource::Custom);
    }
    if let Some(file) =
        env("ANTHROPIC_IDENTITY_TOKEN_FILE").filter(|value| !value.trim().is_empty())
    {
        return Some(IdentitySource::Reference(format!("oidc/file/{file}")));
    }
    if let Some(reference) =
        env("OXLLM_ANTHROPIC_IDENTITY_TOKEN_REF").filter(|value| !value.trim().is_empty())
    {
        return Some(IdentitySource::Reference(reference));
    }
    env("ANTHROPIC_IDENTITY_TOKEN")
        .map(Zeroizing::new)
        .map(|_| IdentitySource::Reference("oidc/env/ANTHROPIC_IDENTITY_TOKEN".into()))
}

fn allowlisted_authority(entry: &str) -> Option<(String, Option<u16>)> {
    let url = if entry.contains("://") {
        entry.to_string()
    } else {
        format!("https://{entry}")
    };
    let url = url::Url::parse(&url).ok()?;
    let host = url.host_str()?;
    let authority = entry
        .split_once("://")
        .map_or(entry, |(_, rest)| rest)
        .split(['/', '?', '#'])
        .next()
        .unwrap_or(entry);
    // URL parsing elides an explicit default port, but allowlist pins retain it.
    let explicit_port = authority
        .rsplit_once(':')
        .and_then(|(_, port)| port.parse::<u16>().ok());
    Some((host.into(), explicit_port.or(url.port())))
}

#[derive(Serialize)]
struct FederationGrant {
    #[serde(flatten)]
    fields: Value,
    #[serde(serialize_with = "serialize_assertion")]
    assertion: Zeroizing<String>,
}

fn serialize_assertion<S>(assertion: &Zeroizing<String>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    serializer.serialize_str(assertion.as_str())
}

enum Identity {
    File(PathBuf),
    Env(String),
}

impl Identity {
    fn parse(reference: &str) -> Result<Self, Error> {
        if let Some(path) = reference
            .strip_prefix("oidc/file/")
            .filter(|path| PathBuf::from(path).is_absolute())
        {
            return Ok(Self::File(path.into()));
        }
        if let Some(name) = reference.strip_prefix("oidc/env/").filter(|name| {
            !name.is_empty()
                && name
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
        }) {
            return Ok(Self::Env(name.into()));
        }
        Err(denied(
            "anthropic_identity_token must use an oidc/env/NAME or oidc/file//absolute/path reference",
        ))
    }

    async fn read(&self, roots: &[PathBuf]) -> Result<Zeroizing<String>, Error> {
        match self {
            Self::Env(name) => {
                let token = Zeroizing::new(std::env::var(name).map_err(|_| {
                    denied("Anthropic identity token environment variable is unavailable")
                })?);
                normalize_identity(token.as_bytes())
            }
            Self::File(path) => {
                use std::io::Read;
                let path = path.clone();
                let roots = roots.to_vec();
                tokio::task::spawn_blocking(move || {
                    let path = path.canonicalize().map_err(|_| {
                        denied("Anthropic identity token file could not be read")
                    })?;
                    if !roots
                        .iter()
                        .filter_map(|root| root.canonicalize().ok())
                        .any(|root| path.starts_with(root))
                    {
                        return Err(denied(
                            "Anthropic identity token file is outside OXLLM_OIDC_ALLOWED_CREDENTIAL_DIRS",
                        ));
                    }
                    let file = std::fs::File::open(path).map_err(|_| {
                        denied("Anthropic identity token file could not be read")
                    })?;
                    let mut token = Zeroizing::new(Vec::with_capacity(MAX_CREDENTIAL_BYTES + 1));
                    file.take(MAX_CREDENTIAL_BYTES as u64 + 1)
                        .read_to_end(&mut token)
                        .map_err(|_| denied("Anthropic identity token file could not be read"))?;
                    normalize_identity(&token)
                })
                .await
                .map_err(|_| denied("Anthropic identity token reader failed"))?
            }
        }
    }
}

fn normalize_identity(original: &[u8]) -> Result<Zeroizing<String>, Error> {
    let invalid =
        || denied("Anthropic identity token must be non-empty, header-safe, and at most 64 KiB");
    if original.len() > MAX_CREDENTIAL_BYTES {
        return Err(invalid());
    }
    let token = std::str::from_utf8(original)
        .map_err(|_| denied("Anthropic identity token must be valid UTF-8"))?
        .trim();
    if token.is_empty() || reqwest::header::HeaderValue::from_str(token).is_err() {
        return Err(invalid());
    }
    Ok(Zeroizing::new(token.into()))
}

fn denied(message: impl Into<String>) -> Error {
    Error::Upstream {
        status: 401,
        message: message.into(),
    }
}

#[cfg(test)]
#[path = "anthropic_tests.rs"]
mod validation_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tower::ServiceExt;

    fn deployment() -> Deployment {
        serde_json::from_value(json!({"model_name":"wif", "oxllm_params":{
            "model":"anthropic/claude-haiku-5-5", "anthropic_identity_token_file":"/missing/token"
        }}))
        .unwrap()
    }

    #[tokio::test]
    async fn explicit_identity_refs_without_required_ids_fail_closed_and_static_keys_win() {
        let mut resolver = AnthropicResolver::new();
        resolver.use_env = false;
        for field in ["anthropic_identity_token_file", "anthropic_identity_token"] {
            let mut dep = deployment();
            dep.oxllm_params.extra.clear();
            dep.oxllm_params.extra.insert(
                field.into(),
                json!(if field.ends_with("file") {
                    "/missing/token"
                } else {
                    "oidc/env/MISSING_TOKEN"
                }),
            );
            let error = resolver.resolve(&dep).await.unwrap_err();
            assert!(matches!(error, Error::Upstream { status: 401, .. }));
            assert!(error.to_string().contains("anthropic_federation_rule_id"));
            assert!(error.to_string().contains("anthropic_organization_id"));
            dep.oxllm_params
                .extra
                .insert("anthropic_federation_rule_id".into(), json!("rule"));
            let error = resolver.resolve(&dep).await.unwrap_err();
            assert!(error.to_string().contains("anthropic_organization_id"));
            dep.oxllm_params.api_key = Some("static-key".into());
            assert_eq!(
                resolver.resolve(&dep).await.unwrap().best_effort_bearer(),
                Some("static-key")
            );
        }
    }

    #[tokio::test]
    async fn json_grants_are_coalesced_and_bearer_auth_reaches_chat_and_counting() {
        fn valid_betas(value: &str) -> bool {
            let tags = value.split(',').map(str::trim).collect::<Vec<_>>();
            tags.len() == 3
                && ["client-beta", "extra-beta", "oauth-2025-04-20"]
                    .iter()
                    .all(|flag| tags.iter().filter(|tag| *tag == flag).count() == 1)
        }
        let upstream = httpmock::MockServer::start_async().await;
        let grant = upstream.mock_async(|when, then| {
            when.method("POST").path("/v1/oauth/token").json_body(json!({
                "grant_type":"urn:ietf:params:oauth:grant-type:jwt-bearer", "assertion":"identity.jwt.signature",
                "federation_rule_id":"rule", "organization_id":"org"
            }));
            then.status(200).json_body(json!({"access_token":"sk-ant-oat01-minted", "token_type":"bearer", "expires_in":3600}));
        }).await;
        let inference = upstream.mock_async(|when, then| {
            when.method("POST").path("/v1/messages").header("authorization", "Bearer sk-ant-oat01-minted")
                .header("anthropic-version", "2023-06-01").matches(|request| request.headers.as_ref().is_some_and(|headers| {
                    headers.iter().any(|(name, value)| name.eq_ignore_ascii_case("anthropic-beta")
                        && valid_betas(value))
                }));
            then.status(200).json_body(json!({"id":"federated", "type":"message", "role":"assistant", "model":"claude-haiku-5-5",
                "content":[{"type":"text","text":"ok"}], "stop_reason":"end_turn", "usage":{"input_tokens":10,"output_tokens":2}}));
        }).await;
        let count = upstream
            .mock_async(|when, then| {
                when.method("POST")
                    .path("/v1/messages/count_tokens")
                    .header("authorization", "Bearer sk-ant-oat01-minted")
                    .header("anthropic-version", "2023-06-01")
                    .matches(|request| {
                        request.headers.as_ref().is_some_and(|headers| {
                            headers.iter().any(|(name, value)| {
                                name.eq_ignore_ascii_case("anthropic-beta") && valid_betas(value)
                            })
                        })
                    });
                then.status(200).json_body(json!({"input_tokens":12}));
            })
            .await;
        let directory = tempfile::tempdir().unwrap();
        let mut file = tempfile::NamedTempFile::new_in(directory.path()).unwrap();
        file.write_all(b"identity.jwt.signature").unwrap();
        let mut dep = deployment();
        dep.oxllm_params.api_base = Some(format!("{}/v1", upstream.base_url()));
        dep.oxllm_params
            .extra
            .insert("anthropic_identity_token_file".into(), json!(file.path()));
        dep.oxllm_params
            .extra
            .insert("anthropic_federation_rule_id".into(), json!("rule"));
        dep.oxllm_params
            .extra
            .insert("anthropic_organization_id".into(), json!("org"));
        dep.oxllm_params.extra_headers = Some(HashMap::from([(
            "anthropic-beta".into(),
            ", extra-beta, extra-beta, oauth-2025-04-20,, ".into(),
        )]));
        let mut resolver = AnthropicResolver::new();
        resolver.use_env = false;
        resolver.allow_http = true;
        resolver.trusted_hosts = vec![("127.0.0.1".into(), None)];
        resolver.credential_dirs = vec![directory.path().into()];
        let resolver = Arc::new(resolver);
        let mut jobs = Vec::new();
        for _ in 0..8 {
            let resolver = resolver.clone();
            let dep = dep.clone();
            jobs.push(tokio::spawn(async move {
                resolver.resolve(&dep).await.unwrap()
            }));
        }
        for job in jobs {
            let credentials = job.await.unwrap();
            assert!(!format!("{credentials:?}").contains("sk-ant-oat01-minted"));
        }
        let config = crate::config::parse_config_str(&serde_yml::to_string(&json!({
            "model_list":[dep], "general_settings":{"allow_private_api_base":true}, "router_settings":{"num_retries":0}
        })).unwrap()).unwrap();
        let mut state =
            crate::state::AppState::with_registry(config, oxllm_providers::default_registry())
                .unwrap();
        let mut registry = super::super::CredentialResolverRegistry::new();
        registry.register("anthropic", resolver);
        state.credential_resolver = Arc::new(registry);
        let shared = Arc::new(arc_swap::ArcSwap::from_pointee(state));
        let app = crate::app::build_router(shared);
        for (path, model) in [
            ("/v1/chat/completions", "wif"),
            ("/v1/messages/count_tokens", "wif"),
            ("/anthropic/v1/messages", "wif"),
        ] {
            let request = axum::http::Request::builder()
                .method("POST")
                .uri(path)
                .header("content-type", "application/json")
                .header("anthropic-beta", ", client-beta, oauth-2025-04-20, client-beta,, ")
                .body(axum::body::Body::from(
                    json!({"model":model, "max_tokens":32, "messages":[{"role":"user","content":"hello"}]})
                        .to_string(),
                ))
                .unwrap();
            let response = app.clone().oneshot(request).await.unwrap();
            assert_eq!(
                response.status(),
                200,
                "{}",
                String::from_utf8_lossy(
                    &axum::body::to_bytes(response.into_body(), 65536)
                        .await
                        .unwrap()
                )
            );
        }
        grant.assert_hits_async(1).await;
        inference.assert_hits_async(2).await;
        count.assert_hits_async(1).await;
    }

    #[test]
    fn identity_reference_and_exchange_authority_validation_do_not_echo_secrets() {
        assert!(Identity::parse("oidc/env/TOKEN_NAME").is_ok());
        for invalid in [
            "raw-secret-token",
            "oidc/env_path/SECRET",
            "oidc/file/relative",
            "oidc/env/",
        ] {
            let error = Identity::parse(invalid).err().unwrap();
            assert!(!error.to_string().contains("raw-secret-token"));
            assert!(!error.to_string().contains("SECRET"));
        }
        let resolver = AnthropicResolver::new();
        let mut dep = deployment();
        dep.oxllm_params.api_base = Some("https://api.anthropic.com.evil.example/v1".into());
        assert!(resolver.endpoint(&dep).is_err());
        dep.oxllm_params.api_base =
            Some("https://api.anthropic.com/v1/messages?ignored=yes".into());
        assert_eq!(
            resolver.endpoint(&dep).unwrap(),
            "https://api.anthropic.com/v1/oauth/token"
        );
    }
}
