//! Azure OpenAI realtime provider — identity passthrough with Azure-specific
//! connection setup.
//!
//! Azure OpenAI's realtime API speaks the OpenAI realtime wire format, so no
//! event translation is needed. It differs from OpenAI only in connection
//! setup:
//!
//! 1. **WebSocket URL**: Azure's realtime GA endpoint is
//!    `{api_base}/openai/v1/realtime` — the docs require the `/openai/v1`
//!    path segment and explicitly say **not** to use a dated `api-version`
//!    query parameter. `api_base` may be the resource root or the documented
//!    v1 base (`…/openai/v1`); both normalize to the same URL, with any
//!    query string dropped.
//! 2. **Auth header**: Azure authenticates the WS upgrade with an
//!    `api-key: {key}` header, or `Authorization: Bearer` for an Entra ID
//!    access token (`extra.azure_ad_token`, or a JWT-shaped key).
//!
//! All frame transforms are identity (delegated to [`OpenAIRealtimeProvider`]).
//!
//! See [`AzureRealtimeProvider`].

use oxllm_core::RealtimeProvider;
use oxllm_core::deployment::Deployment;
use oxllm_core::error::Result;

use super::openai::{OpenAIRealtimeProvider, bare_model};

/// Azure OpenAI realtime provider — identity transforms, Azure-shaped WS URL
/// and `api-key` auth.
///
/// Construct via [`AzureRealtimeProvider::new`]; selected for the `azure`
/// provider slug by `super::resolve_realtime_provider`.
pub struct AzureRealtimeProvider {
    /// Identity provider handling event transforms (Azure is wire-compatible).
    inner: OpenAIRealtimeProvider,
}

/// Fallback upstream base for Azure realtime if `ws_url` is ever called
/// without an `api_base`. Unreachable in practice:
/// [`super::resolve_realtime_provider`] rejects an `azure` deployment with
/// no `api_base` before the provider is constructed, precisely so the Azure
/// `api-key` header is never sent to an OpenAI endpoint. Kept only so URL
/// construction stays total if the provider is used directly.
const DEFAULT_AZURE_BASE: &str = "wss://azure-api-base-not-configured.invalid";

/// Remove a trailing route suffix the operator already included in
/// `api_base` (`/openai/v1` from the documented v1 base URL, or `/openai`),
/// so the canonical Azure route is not doubled.
fn strip_route_suffix<'a>(path: &'a str, suffix: &str) -> &'a str {
    let trimmed = path.trim_end_matches('/');
    if trimmed.len() > suffix.len()
        && trimmed[trimmed.len() - suffix.len()..].eq_ignore_ascii_case(suffix)
    {
        trimmed[..trimmed.len() - suffix.len()].trim_end_matches('/')
    } else {
        trimmed
    }
}

/// Entra ID access tokens are JWTs (header segment starts with `eyJ`); Azure
/// shared keys are 32-char hex strings.
fn looks_like_jwt(value: &str) -> bool {
    value.starts_with("eyJ") && value.matches('.').count() == 2
}

impl AzureRealtimeProvider {
    /// Construct the Azure realtime provider.
    pub fn new() -> Self {
        Self {
            inner: OpenAIRealtimeProvider::new(),
        }
    }
}

impl Default for AzureRealtimeProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl RealtimeProvider for AzureRealtimeProvider {
    fn transform_client_event(&self, event: &str) -> Result<Vec<String>> {
        self.inner.transform_client_event(event)
    }

    fn transform_upstream_event(&self, event: &str) -> Result<Vec<String>> {
        self.inner.transform_upstream_event(event)
    }

    fn ws_url(&self, deployment: &Deployment) -> String {
        let base = deployment
            .oxllm_params
            .api_base
            .as_deref()
            .map(|b| {
                // Realtime docs: "use the GA endpoint with `/openai/v1` in the
                // URL. Don't use date-based API versions or the api-version
                // query parameter." Drop any carried query and normalize the
                // documented v1 base (`…/openai/v1`) or a `/openai` base so the
                // route is not doubled.
                let path = b.split('?').next().unwrap_or(b);
                let path = strip_route_suffix(strip_route_suffix(path, "/openai/v1"), "/openai");
                super::scheme_rewrite_for_ws(path)
            })
            .unwrap_or_else(|| DEFAULT_AZURE_BASE.to_string());

        let model = bare_model(deployment);
        super::build_realtime_ws_url(&base, "/openai/v1/realtime", model)
    }

    fn ws_auth_headers(&self, deployment: &Deployment) -> Vec<(String, String)> {
        // Entra ID token configured explicitly wins (keyless deployments have
        // no `api_key`), then a JWT-shaped key, then the shared key. Azure
        // rejects an AAD token on `api-key` with a 401.
        if let Some(token) = deployment
            .oxllm_params
            .extra
            .get("azure_ad_token")
            .and_then(|value| value.as_str())
            .filter(|token| !token.is_empty())
        {
            return vec![("Authorization".to_string(), format!("Bearer {token}"))];
        }
        match deployment.oxllm_params.api_key.as_deref() {
            Some(key) if looks_like_jwt(key) => {
                vec![("Authorization".to_string(), format!("Bearer {key}"))]
            }
            // Azure authenticates the WS upgrade with `api-key: {key}` — NOT
            // the OpenAI `Authorization: Bearer {key}` form.
            Some(key) => vec![("api-key".to_string(), key.to_string())],
            None => Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxllm_core::deployment::OxllmParams;
    use serde_json::Map;

    fn dep(api_base: Option<&str>) -> Deployment {
        Deployment {
            model_name: "gpt-4o-realtime".to_string(),
            oxllm_params: OxllmParams {
                model: "azure/gpt-4o-realtime".to_string(),
                api_key: Some("az-test".to_string()),
                api_base: api_base.map(String::from),
                extra: Map::new(),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    // ---- identity transforms ----------------------------------------------

    #[test]
    fn client_event_passes_through_unchanged() {
        let p = AzureRealtimeProvider::new();
        let event = r#"{"type":"session.update","session":{"voice":"alloy"}}"#;
        assert_eq!(
            p.transform_client_event(event)
                .expect("identity never errors"),
            vec![event.to_string()]
        );
    }

    #[test]
    fn upstream_event_passes_through_unchanged() {
        let p = AzureRealtimeProvider::new();
        let event = r#"{"type":"response.audio.delta","delta":"base64"}"#;
        assert_eq!(
            p.transform_upstream_event(event)
                .expect("identity never errors"),
            vec![event.to_string()]
        );
    }

    // ---- ws_url ------------------------------------------------------------

    #[test]
    fn url_uses_v1_realtime_path() {
        let url =
            AzureRealtimeProvider::new().ws_url(&dep(Some("https://my.resource.openai.azure.com")));
        assert!(
            url.starts_with("wss://my.resource.openai.azure.com/openai/v1/realtime"),
            "expected the GA v1 realtime path, got {url}"
        );
        assert!(url.contains("model=gpt-4o-realtime"), "got {url}");
    }

    /// The documented v1 base URL (`…/openai/v1/`) must not be doubled, on
    /// either the Azure OpenAI host or the unified Foundry host.
    #[test]
    fn url_does_not_double_v1_base_suffix() {
        for base in [
            "https://my.resource.openai.azure.com/openai/v1",
            "https://my.resource.openai.azure.com/openai/v1/",
            "https://my.resource.services.ai.azure.com/openai/v1",
        ] {
            let url = AzureRealtimeProvider::new().ws_url(&dep(Some(base)));
            let parsed =
                url::Url::parse(&url).unwrap_or_else(|e| panic!("url parse failed: {e} ({url})"));
            assert_eq!(
                parsed.path(),
                "/openai/v1/realtime",
                "base={base} produced url={url}"
            );
        }
    }

    /// Realtime docs explicitly say not to use the `api-version` query
    /// parameter, so a carried query is dropped and only `model` remains.
    #[test]
    fn url_drops_api_version_query_param() {
        for base in [
            "https://my.resource.openai.azure.com?api-version=2024-10-01-preview",
            "https://my.resource.openai.azure.com/openai?api-version=2024-10-01-preview",
            "https://my.resource.openai.azure.com/openai/v1/?api-version=preview",
        ] {
            let url = AzureRealtimeProvider::new().ws_url(&dep(Some(base)));
            assert!(!url.contains("api-version"), "base={base} url={url}");
            assert!(
                url.contains("model=gpt-4o-realtime"),
                "base={base} url={url}"
            );
            let parsed =
                url::Url::parse(&url).unwrap_or_else(|e| panic!("url parse failed: {e} ({url})"));
            assert_eq!(
                parsed.path(),
                "/openai/v1/realtime",
                "base={base} produced url={url}"
            );
        }
    }

    #[test]
    fn url_strips_trailing_slash_on_api_base() {
        let url = AzureRealtimeProvider::new()
            .ws_url(&dep(Some("https://my.resource.openai.azure.com/")));
        assert!(
            !url.contains("//openai/v1/realtime"),
            "trailing slash must not produce a double slash: {url}"
        );
    }

    // ---- ws_auth_headers ---------------------------------------------------

    #[test]
    fn headers_use_api_key_header_not_bearer() {
        let h = AzureRealtimeProvider::new().ws_auth_headers(&dep(None));
        assert_eq!(h.len(), 1);
        assert_eq!(h[0].0, "api-key");
        assert_eq!(h[0].1, "az-test");
        assert!(
            !h[0].1.contains("Bearer"),
            "azure must NOT use Authorization: Bearer for a shared key"
        );
    }

    #[test]
    fn headers_empty_when_no_api_key() {
        let mut d = dep(None);
        d.oxllm_params.api_key = None;
        assert!(AzureRealtimeProvider::new().ws_auth_headers(&d).is_empty());
    }

    /// Entra ID tokens are JWTs; they must ride `Authorization: Bearer`
    /// because Azure rejects them on `api-key`.
    #[test]
    fn jwt_shaped_api_key_uses_bearer_header() {
        let mut d = dep(None);
        d.oxllm_params.api_key = Some("eyJhbGciOiJub25lIn0.eyJzdWIiOiJ4In0.sig".to_string());
        let h = AzureRealtimeProvider::new().ws_auth_headers(&d);
        assert_eq!(h.len(), 1);
        assert_eq!(h[0].0, "Authorization");
        assert!(h[0].1.starts_with("Bearer eyJ"), "got {:?}", h[0]);
    }

    /// `extra.azure_ad_token` supports keyless deployments (no `api_key`) and
    /// takes precedence over a shared key when both are configured.
    #[test]
    fn azure_ad_token_extra_uses_bearer_and_wins() {
        let mut d = dep(None);
        d.oxllm_params.api_key = Some("shared-key".to_string());
        d.oxllm_params.extra.insert(
            "azure_ad_token".to_string(),
            serde_json::json!("eyJhbGciOiJub25lIn0.eyJzdWIiOiJ4In0.sig"),
        );
        let h = AzureRealtimeProvider::new().ws_auth_headers(&d);
        assert_eq!(h.len(), 1);
        assert_eq!(h[0].0, "Authorization");
        assert!(h[0].1.starts_with("Bearer eyJ"), "got {:?}", h[0]);
    }
}
