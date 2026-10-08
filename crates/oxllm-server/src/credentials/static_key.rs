//! The default credential resolver — reads a static `api_key` from config.

use async_trait::async_trait;
use oxllm_core::Deployment;
use oxllm_core::Error;
use oxllm_core::provider::ResolvedCredentials;

use super::CredentialResolver;

/// Resolves credentials from `oxllm_params.api_key`.
///
/// This is the historical (pre-Phase-9) behavior and the default for every
/// "simple key" provider (OpenAI, Anthropic, Gemini, Ollama, …). It performs
/// no network I/O — the key was env-resolved once at config load time.
///
/// For `azure/*` deployments the resolved variant is
/// [`ResolvedCredentials::Azure`] so provider transforms can attach the
/// `api-key` header verbatim. All other slugs yield
/// [`ResolvedCredentials::ApiKey`].
#[derive(Debug, Clone, Default)]
pub struct StaticKeyResolver;

#[async_trait]
impl CredentialResolver for StaticKeyResolver {
    async fn resolve(&self, deployment: &Deployment) -> Result<ResolvedCredentials, Error> {
        deployment.oxllm_params.validate_auth_overrides()?;
        // Azure keyless/Entra ID: an explicitly configured access token
        // (`extra.azure_ad_token`, resolved once at config load like every
        // other `os.environ/` value) takes precedence over the shared key and
        // is sent as `Authorization: Bearer`. Without this branch the
        // `ResolvedCredentials::AzureAd` variant was unreachable, so Entra
        // ID deployments had no config path at all.
        if matches!(deployment.provider_slug(), "azure" | "azure_ai") {
            if let Some(token) = deployment
                .oxllm_params
                .extra
                .get("azure_ad_token")
                .and_then(|value| value.as_str())
                .filter(|token| !token.is_empty())
            {
                validate_static_credential(token, "azure_ad_token")?;
                return Ok(ResolvedCredentials::AzureAd {
                    token: token.to_string(),
                });
            }
            let Some(api_key) = deployment.oxllm_params.api_key.as_deref() else {
                return Err(Error::Config(
                    "azure deployment requires `api_key` or `extra.azure_ad_token`".to_string(),
                ));
            };
            validate_static_credential(api_key, "api_key")?;
            return Ok(ResolvedCredentials::Azure {
                api_key: api_key.to_string(),
            });
        }

        let Some(api_key) = deployment.oxllm_params.api_key.as_deref() else {
            // Ollama's local HTTP API is unauthenticated by design. The provider
            // transform already performs a no-op `apply_auth`; return a benign
            // placeholder so credential resolution does not block that path.
            if deployment.provider_slug() == "ollama" {
                return Ok(ResolvedCredentials::from_key(""));
            }
            return Err(Error::Config("missing api_key for deployment".to_string()));
        };
        if deployment.provider_slug() != "ollama" {
            validate_static_credential(api_key, "api_key")?;
        }
        Ok(ResolvedCredentials::from_key(api_key))
    }
}

fn validate_static_credential(value: &str, field: &str) -> Result<(), Error> {
    if value.trim().is_empty() || reqwest::header::HeaderValue::from_str(value).is_err() {
        return Err(Error::Config(format!(
            "{field} must be non-empty and contain valid HTTP header characters"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxllm_core::deployment::OxllmParams;
    use serde_json::json;

    fn azure_deployment(api_key: Option<&str>, ad_token: Option<&str>) -> Deployment {
        let mut extra = serde_json::Map::new();
        if let Some(token) = ad_token {
            extra.insert("azure_ad_token".to_string(), json!(token));
        }
        Deployment {
            model_name: "my-deployment".to_string(),
            oxllm_params: OxllmParams {
                model: "azure/my-deployment".to_string(),
                api_key: api_key.map(String::from),
                extra,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn invalid_static_credentials_fail_without_disclosing_the_value() {
        for value in [
            "",
            " \t",
            "secret-marker\r\nx-injected: true",
            "secret-marker\0",
        ] {
            for mut deployment in [
                azure_deployment(Some(value), None),
                azure_deployment(None, Some(value)),
            ] {
                let error = StaticKeyResolver.resolve(&deployment).await.unwrap_err();
                assert!(matches!(error, Error::Config(_)));
                assert!(!error.to_string().contains("secret-marker"));
                deployment.oxllm_params.model = "openai/gpt-4o".into();
                deployment.oxllm_params.api_key = Some(value.into());
                assert!(StaticKeyResolver.resolve(&deployment).await.is_err());
            }
        }
    }

    #[tokio::test]
    async fn azure_shared_key_resolves_as_azure_credential() {
        let creds = StaticKeyResolver
            .resolve(&azure_deployment(Some("hex-shared-key"), None))
            .await
            .expect("resolve");
        match creds {
            ResolvedCredentials::Azure { api_key } => assert_eq!(api_key, "hex-shared-key"),
            other => panic!("expected Azure shared key, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn azure_ad_token_extra_resolves_without_an_api_key() {
        let creds = StaticKeyResolver
            .resolve(&azure_deployment(None, Some("entra-token-123")))
            .await
            .expect("keyless Entra deployment must resolve");
        match creds {
            ResolvedCredentials::AzureAd { token } => assert_eq!(token, "entra-token-123"),
            other => panic!("expected AzureAd token, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn azure_ad_token_takes_precedence_over_shared_key() {
        let creds = StaticKeyResolver
            .resolve(&azure_deployment(
                Some("hex-shared-key"),
                Some("entra-token"),
            ))
            .await
            .expect("resolve");
        assert!(
            matches!(creds, ResolvedCredentials::AzureAd { .. }),
            "explicit azure_ad_token must win over the shared key"
        );
    }

    #[tokio::test]
    async fn azure_without_credentials_names_both_options() {
        let err = StaticKeyResolver
            .resolve(&azure_deployment(None, None))
            .await
            .expect_err("must fail without credentials");
        let msg = format!("{err}");
        assert!(
            msg.contains("azure_ad_token"),
            "error should name the keyless option, got: {msg}"
        );
    }
}
