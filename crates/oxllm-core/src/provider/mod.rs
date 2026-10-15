//! Provider trait contracts.
//!
//! These traits define how a provider translates between the OpenAI wire
//! format and its own native format. All methods are **synchronous pure
//! functions** — no network, no async.

mod auth;
mod embedding;
pub mod realtime;

pub use auth::apply_auth_overrides;
pub use embedding::{
    EmbeddingRequest, EmbeddingRequestBatch, EmbeddingRequestPlan, is_base64_data_uri,
};

pub use realtime::RealtimeProvider;

use std::collections::HashMap;

use crate::deployment::Deployment;
use crate::endpoint::Endpoint;
use crate::error::Result;
use serde_json::Value;

/// Identifies a provider by its slug (e.g. `"openai"`, `"anthropic"`).
pub type ProviderId = &'static str;

/// The super-trait that advertises which endpoint kinds a provider supports.
///
/// The server uses `supports()` + the typed accessor to dispatch
/// `(Endpoint, Provider) → handler` without a giant match.
pub trait Provider: Send + Sync {
    /// The provider slug (e.g. `"openai"`).
    fn id(&self) -> ProviderId;

    /// Whether this provider supports the given endpoint kind.
    fn supports(&self, endpoint: Endpoint) -> bool;

    /// Typed accessor for chat-completion transforms, if supported.
    fn chat(&self) -> Option<&dyn ChatProvider> {
        None
    }

    /// Typed accessor for embedding transforms, if supported.
    fn embeddings(&self) -> Option<&dyn EmbeddingProvider> {
        None
    }

    /// Typed accessor for rerank transforms, if supported.
    fn rerank(&self) -> Option<&dyn RerankProvider> {
        None
    }
}

/// Per-endpoint transform trait for chat completions.
///
/// All methods are pure synchronous functions. Credential acquisition
/// is done by the host and passed in via [`ResolvedCredentials`].
pub trait ChatProvider: Send + Sync {
    /// Build the upstream URL for this deployment.
    fn endpoint_url(&self, deployment: &Deployment) -> String;

    /// Build the URL for the routed request before provider-specific fields
    /// are removed. Model-specific endpoints resolve wildcard deployments
    /// without requiring an otherwise invalid model field in the wire body.
    fn endpoint_url_for_request(&self, deployment: &Deployment, request: &Value) -> String {
        let _ = request;
        self.endpoint_url(deployment)
    }

    /// Add auth headers (e.g. `Authorization: Bearer …`).
    fn apply_auth(
        &self,
        deployment: &Deployment,
        creds: &ResolvedCredentials,
        headers: &mut HashMap<String, String>,
    );

    /// OpenAI request → provider-native request body.
    fn transform_request(&self, request: &Value, deployment: &Deployment) -> Result<Value>;

    /// Provider response → OpenAI response.
    fn transform_response(&self, body: &Value, model: &str) -> Result<Value>;

    /// One SSE chunk → OpenAI streaming chunk (returns `None` to skip).
    fn transform_stream_chunk(&self, chunk: &str) -> Result<Option<Value>>;

    /// Transform one streaming SSE chunk with host-owned per-stream state.
    ///
    /// The host owns the `state` slot — created once per stream and threaded
    /// through every chunk — so per-stream accumulation survives async task
    /// migration across worker threads, something a provider-side
    /// `thread_local!` cannot do (a `tokio::spawn` task may be rescheduled
    /// onto a different worker thread at every `.await`, orphaning any
    /// thread-local it wrote). `original_request` is the caller's OpenAI-shape
    /// request, made available so stateful providers can derive context that
    /// spans the request/response boundary (e.g. a reverse tool-name map).
    ///
    /// The default delegates to [`ChatProvider::transform_stream_chunk`],
    /// ignoring both `state` and `original_request`, so providers that do not
    /// need per-stream state are unaffected.
    fn transform_stream_chunk_stateful(
        &self,
        chunk: &str,
        state: &mut Option<Box<dyn std::any::Any + Send + Sync>>,
        original_request: &Value,
    ) -> Result<Option<Value>> {
        let _ = (state, original_request);
        self.transform_stream_chunk(chunk)
    }

    /// Transform a provider response with the original request available.
    ///
    /// The default delegates to [`ChatProvider::transform_response`]. Providers
    /// that need request context on the way back — e.g. a reverse tool-name
    /// map derived from the request that cannot survive a `thread_local!`
    /// across async task migration — override this.
    fn transform_response_with_request(
        &self,
        body: &Value,
        model: &str,
        original_request: &Value,
    ) -> Result<Value> {
        let _ = original_request;
        self.transform_response(body, model)
    }

    /// The list of OpenAI request params this provider's `transform_request`
    /// preserves. Used by the host to warn on (or, with `drop_params`, strip)
    /// unsupported params. Returns `None` if the provider does not publish a
    /// list (treat as "passthrough / unknown").
    fn supported_params(&self) -> Option<Vec<&str>> {
        None
    }
}

/// Per-endpoint transform trait for embeddings.
///
/// Mirrors [`ChatProvider`] but for `POST /v1/embeddings`. All methods are
/// pure synchronous functions.
pub trait EmbeddingProvider: Send + Sync {
    /// Prepare native requests, optionally partitioning different input types.
    /// Ownership is transferred without copying the parameter source. The host
    /// applies its outbound policy, authentication and transport afterwards.
    fn prepare_requests(
        &self,
        request: Value,
        deployment: &Deployment,
    ) -> Result<EmbeddingRequestPlan> {
        let url = self.embedding_url_for_request(deployment, &request);
        let body = self.transform_request(&request, deployment)?;
        Ok(EmbeddingRequestPlan::Single(EmbeddingRequest {
            url,
            body,
            params: request,
            allowed_fields: None,
        }))
    }

    /// Build the upstream embeddings URL for this deployment.
    fn embedding_url(&self, deployment: &Deployment) -> String;

    /// Request-aware URL for wildcard deployments or input-specific routes.
    /// Fixed deployments use the ordinary URL unless the provider overrides it.
    fn embedding_url_for_request(&self, deployment: &Deployment, _request: &Value) -> String {
        self.embedding_url(deployment)
    }

    /// Add auth headers for the embeddings call.
    fn apply_auth(
        &self,
        deployment: &Deployment,
        creds: &ResolvedCredentials,
        headers: &mut HashMap<String, String>,
    );

    /// OpenAI embeddings request → provider-native request body.
    fn transform_request(&self, request: &Value, deployment: &Deployment) -> Result<Value>;

    /// Provider response → OpenAI embeddings response.
    fn transform_response(&self, body: &Value, model: &str) -> Result<Value>;
}

/// Per-endpoint transform trait for reranking.
///
/// Mirrors [`ChatProvider`] but for `POST /v1/rerank`. All methods are
/// pure synchronous functions.
pub trait RerankProvider: Send + Sync {
    /// Build the upstream rerank URL for this deployment.
    fn rerank_url(&self, deployment: &Deployment) -> String;

    /// Add auth headers for the rerank call.
    fn apply_auth(
        &self,
        deployment: &Deployment,
        creds: &ResolvedCredentials,
        headers: &mut HashMap<String, String>,
    );

    /// OpenAI rerank request → provider-native request body.
    fn transform_request(&self, request: &Value, deployment: &Deployment) -> Result<Value>;

    /// Provider response → OpenAI rerank response.
    fn transform_response(&self, body: &Value, model: &str) -> Result<Value>;
}

/// Credentials resolved by the host before calling a transform.
///
/// The host fetches short-lived creds (AWS STS, GCP token, etc.) and passes
/// them in here. The transform trait itself never touches the network.
///
/// The host's credential resolver selects the variant; pure provider
/// transforms pattern-match on it in `apply_auth`.
///
/// `Debug` is hand-written so the secret material in every variant
/// (`bearer`, `api_key`, `secret_key`/`session_token`, GCP `token`, and any
/// `Authorization` header inside [`ResolvedCredentials::PreSignedHeaders`])
/// is masked as `<redacted>`. A future `tracing::debug!(?creds)` therefore
/// cannot leak live keys.
#[derive(Clone)]
#[non_exhaustive]
pub enum ResolvedCredentials {
    /// A plain API key or bearer token (OpenAI, Anthropic, Gemini, …).
    ApiKey {
        /// The API key or bearer token sent in the auth header.
        bearer: String,
    },
    /// Azure OpenAI `api-key` header credential.
    Azure {
        /// The Azure OpenAI API key (sent in the `api-key` header).
        api_key: String,
    },
    /// Azure AD / Entra ID bearer token. Sent as
    /// `Authorization: Bearer <token>` (NOT the `api-key` header), matching
    /// the reference Azure-environment AAD fallback. The host is
    /// responsible for obtaining and refreshing the token (client secret,
    /// managed identity, OIDC, …) before resolving this variant; the provider
    /// transform only attaches it.
    AzureAd {
        /// The Azure AD / Entra ID access token.
        token: String,
    },
    /// AWS SigV4 material for Bedrock requests. The host may instead pass
    /// [`ResolvedCredentials::PreSignedHeaders`] when it has already signed
    /// the request.
    AwsSigV4 {
        /// The AWS access key id (`AKIA…`).
        access_key: String,
        /// The AWS secret access key.
        secret_key: String,
        /// Optional short-lived STS session token.
        session_token: Option<String>,
        /// The AWS region (e.g. `us-east-1`).
        region: String,
    },
    /// A pre-exchanged GCP bearer token (Vertex AI).
    GcpBearer {
        /// The OAuth2 access token for the Vertex AI endpoint.
        token: String,
    },
    /// A fully pre-signed header set. The provider only needs to attach these
    /// headers verbatim — useful when signing lives entirely in the host.
    PreSignedHeaders(std::collections::HashMap<String, String>),
}

/// Placeholder emitted for any secret-bearing field under `Debug`.
const REDACTED: &str = "<redacted>";

/// Lowercase header names whose **values** are safe to display verbatim under
/// `Debug` of [`ResolvedCredentials::PreSignedHeaders`]. Matching is
/// ASCII-case-insensitive to match HTTP header-name semantics.
///
/// This is a **default-deny** allowlist: every header value *not* on this list
/// is masked as [`REDACTED`]. The previous denylist approach (`authorization`,
/// `cookie`, …) rotted as new cloud-signing headers appeared (`x-amz-security-
/// token`, `x-goog-credential`, …), risking leaks of STS session tokens and
/// signing material through a future `tracing::debug!(?creds)`. Only headers
/// that are structurally incapable of carrying a secret are listed here.
const SAFE_HEADER_NAMES: &[&str] = &[
    "content-type",
    "content-length",
    "accept",
    "accept-encoding",
    "accept-language",
    "user-agent",
    "host",
    "x-request-id",
];

impl std::fmt::Debug for ResolvedCredentials {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResolvedCredentials::ApiKey { .. } => {
                f.debug_struct("ApiKey").field("bearer", &REDACTED).finish()
            }
            ResolvedCredentials::Azure { .. } => {
                f.debug_struct("Azure").field("api_key", &REDACTED).finish()
            }
            ResolvedCredentials::AzureAd { .. } => {
                f.debug_struct("AzureAd").field("token", &REDACTED).finish()
            }
            ResolvedCredentials::AwsSigV4 { region, .. } => f
                .debug_struct("AwsSigV4")
                .field("access_key", &REDACTED)
                .field("secret_key", &REDACTED)
                .field("session_token", &REDACTED)
                .field("region", region)
                .finish(),
            ResolvedCredentials::GcpBearer { .. } => f
                .debug_struct("GcpBearer")
                .field("token", &REDACTED)
                .finish(),
            ResolvedCredentials::PreSignedHeaders(headers) => {
                f.write_str("PreSignedHeaders(")?;
                {
                    let mut map = f.debug_map();
                    for (name, value) in headers {
                        let lower = name.to_ascii_lowercase();
                        // Default-deny: redact every value unless the header is
                        // on the safe-to-display allowlist. Header names are
                        // always kept so the operator can still see structure.
                        let entry_value = if SAFE_HEADER_NAMES.contains(&lower.as_str()) {
                            value as &dyn std::fmt::Debug
                        } else {
                            &REDACTED as &dyn std::fmt::Debug
                        };
                        map.entry(name, entry_value);
                    }
                    map.finish()?;
                }
                f.write_str(")")
            }
        }
    }
}

impl ResolvedCredentials {
    /// Create a [`ResolvedCredentials::ApiKey`] credential from a simple key/bearer.
    pub fn from_key(key: impl Into<String>) -> Self {
        Self::ApiKey { bearer: key.into() }
    }

    /// Best-effort extraction of a bearer-like string from any variant.
    ///
    /// Providers that only support [`ResolvedCredentials::ApiKey`] can use this to degrade
    /// gracefully when handed a different credential kind. The host should
    /// always resolve the type appropriate for the provider, so reaching
    /// this fallback indicates a misconfiguration rather than a normal path.
    pub fn best_effort_bearer(&self) -> Option<&str> {
        match self {
            ResolvedCredentials::ApiKey { bearer } => Some(bearer.as_str()),
            ResolvedCredentials::Azure { api_key } => Some(api_key.as_str()),
            ResolvedCredentials::AzureAd { token } => Some(token.as_str()),
            ResolvedCredentials::AwsSigV4 { .. } => None,
            ResolvedCredentials::GcpBearer { token } => Some(token.as_str()),
            ResolvedCredentials::PreSignedHeaders(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn ordinary_embedding_plans_move_the_parameter_source_without_a_deep_clone() {
        struct Ordinary;
        impl EmbeddingProvider for Ordinary {
            fn embedding_url(&self, _: &Deployment) -> String {
                "https://example.test/embeddings".into()
            }
            fn apply_auth(
                &self,
                _: &Deployment,
                _: &ResolvedCredentials,
                _: &mut HashMap<String, String>,
            ) {
            }
            fn transform_request(&self, request: &Value, _: &Deployment) -> Result<Value> {
                Ok(request.clone())
            }
            fn transform_response(&self, body: &Value, _: &str) -> Result<Value> {
                Ok(body.clone())
            }
        }
        let request =
            serde_json::json!({"input":["one", "two"], "model":"test-model", "extra":"kept"});
        let pointer = request["input"][0].as_str().unwrap().as_ptr();
        let expected = request.clone();
        let EmbeddingRequestPlan::Single(prepared) = Ordinary
            .prepare_requests(request, &Deployment::default())
            .unwrap()
        else {
            panic!("ordinary providers are not batched")
        };
        assert_eq!(
            prepared.params["input"][0].as_str().unwrap().as_ptr(),
            pointer
        );
        assert_eq!(prepared.body, expected);
        assert_eq!(prepared.params, expected);
        assert!(prepared.allowed_fields.is_none());
    }

    #[test]
    fn best_effort_bearer_returns_some_for_bearer_variants() {
        // Covers the registry.rs omission noted in the review: Azure→Some,
        // SigV4→None, PreSigned→None, plus ApiKey/GcpBearer.
        assert_eq!(
            ResolvedCredentials::ApiKey {
                bearer: "sk-x".to_string(),
            }
            .best_effort_bearer(),
            Some("sk-x")
        );
        assert_eq!(
            ResolvedCredentials::Azure {
                api_key: "az-key".to_string(),
            }
            .best_effort_bearer(),
            Some("az-key")
        );
        assert_eq!(
            ResolvedCredentials::AzureAd {
                token: "aad-tok".to_string(),
            }
            .best_effort_bearer(),
            Some("aad-tok")
        );
        assert_eq!(
            ResolvedCredentials::GcpBearer {
                token: "gcp-tok".to_string(),
            }
            .best_effort_bearer(),
            Some("gcp-tok")
        );
        assert_eq!(
            ResolvedCredentials::AwsSigV4 {
                access_key: "AKIA".to_string(),
                secret_key: "s".to_string(),
                session_token: None,
                region: "us-east-1".to_string(),
            }
            .best_effort_bearer(),
            None
        );
        assert_eq!(
            ResolvedCredentials::PreSignedHeaders(HashMap::new()).best_effort_bearer(),
            None
        );
    }

    #[test]
    fn debug_redacts_api_key_bearer() {
        let c = ResolvedCredentials::ApiKey {
            bearer: "sk-live-bearer".to_string(),
        };
        let s = format!("{c:?}");
        assert!(!s.contains("sk-live-bearer"), "bearer leaked: {s}");
        assert!(s.contains("<redacted>"));
    }

    #[test]
    fn debug_redacts_azure_api_key() {
        let c = ResolvedCredentials::Azure {
            api_key: "az-live-key".to_string(),
        };
        let s = format!("{c:?}");
        assert!(!s.contains("az-live-key"), "azure key leaked: {s}");
        assert!(s.contains("<redacted>"));
    }

    #[test]
    fn debug_redacts_azure_ad_token() {
        let c = ResolvedCredentials::AzureAd {
            token: "aad-live-token".to_string(),
        };
        let s = format!("{c:?}");
        assert!(!s.contains("aad-live-token"), "azure AD token leaked: {s}");
        assert!(s.contains("<redacted>"));
    }

    #[test]
    fn debug_redacts_aws_sigv4_secrets_but_keeps_region() {
        let c = ResolvedCredentials::AwsSigV4 {
            access_key: "AKIA-LIVE".to_string(),
            secret_key: "super-secret".to_string(),
            session_token: Some("tok-live".to_string()),
            region: "us-east-1".to_string(),
        };
        let s = format!("{c:?}");
        assert!(!s.contains("AKIA-LIVE"), "access_key leaked: {s}");
        assert!(!s.contains("super-secret"), "secret_key leaked: {s}");
        assert!(!s.contains("tok-live"), "session_token leaked: {s}");
        assert!(s.contains("us-east-1"), "region should remain visible: {s}");
    }

    #[test]
    fn debug_redacts_gcp_bearer_token() {
        let c = ResolvedCredentials::GcpBearer {
            token: "gcp-live-tok".to_string(),
        };
        let s = format!("{c:?}");
        assert!(!s.contains("gcp-live-tok"), "gcp token leaked: {s}");
        assert!(s.contains("<redacted>"));
    }

    #[test]
    fn debug_redacts_sensitive_presigned_headers_but_keeps_safe() {
        // Default-deny: only headers on SAFE_HEADER_NAMES keep their value;
        // everything else (including cloud-signing headers) is masked.
        let mut h = HashMap::new();
        h.insert(
            "Authorization".to_string(),
            "Bearer sk-live-presh".to_string(),
        );
        h.insert("x-request-id".to_string(), "abc-123".to_string());
        h.insert("X-API-KEY".to_string(), "topsecret".to_string());
        h.insert("content-type".to_string(), "application/json".to_string());
        let c = ResolvedCredentials::PreSignedHeaders(h);
        let s = format!("{c:?}");
        assert!(
            !s.contains("sk-live-presh"),
            "Authorization value leaked: {s}"
        );
        assert!(!s.contains("topsecret"), "X-API-KEY value leaked: {s}");
        // Safe-to-display values remain visible (proves the redactor is
        // targeted, not blanket).
        assert!(
            s.contains("abc-123"),
            "safe x-request-id value should remain visible: {s}"
        );
        assert!(
            s.contains("application/json"),
            "safe content-type value should remain visible: {s}"
        );
        // Header names are kept so the operator can see the structure.
        assert!(s.contains("Authorization"));
        assert!(s.contains("x-request-id"));
    }

    #[test]
    fn debug_redacts_aws_and_gcp_signing_headers_in_presigned() {
        // Cloud-signing headers a host could legitimately stuff into
        // PreSignedHeaders must be masked by the default-deny redactor.
        let mut h = HashMap::new();
        h.insert(
            "x-amz-security-token".to_string(),
            "IQoJb3JpZ2luX2VjEAoaCg==".to_string(),
        );
        h.insert("x-amz-date".to_string(), "20260806T120000Z".to_string());
        h.insert(
            "x-goog-credential".to_string(),
            "svcacct@proj.iam.gserviceaccount.com/20260806/".to_string(),
        );
        h.insert("x-goog-date".to_string(), "20260806T120000Z".to_string());
        h.insert(
            "x-hubspot-access-token".to_string(),
            "hub-secret".to_string(),
        );
        let c = ResolvedCredentials::PreSignedHeaders(h);
        let s = format!("{c:?}");
        assert!(
            !s.contains("IQoJb3JpZ2luX2VjEAoaCg=="),
            "AWS STS session token leaked: {s}"
        );
        assert!(
            !s.contains("svcacct@proj"),
            "GCP signing credential leaked: {s}"
        );
        assert!(
            !s.contains("hub-secret"),
            "hubspot access token leaked: {s}"
        );
        // Every value masked.
        assert_eq!(
            s.matches("<redacted>").count(),
            5,
            "all five signing headers must be redacted: {s}"
        );
    }

    #[test]
    fn debug_redacts_unknown_header_by_default() {
        // Default-deny: an unrecognized header is masked even though it is
        // not on any cloud-provider signing list.
        let mut h = HashMap::new();
        h.insert(
            "x-custom-trace".to_string(),
            "secret-trace-blob".to_string(),
        );
        let c = ResolvedCredentials::PreSignedHeaders(h);
        let s = format!("{c:?}");
        assert!(
            !s.contains("secret-trace-blob"),
            "unknown header value leaked: {s}"
        );
        assert!(
            s.contains("x-custom-trace"),
            "header name should be kept: {s}"
        );
        assert!(s.contains("<redacted>"));
    }
}
