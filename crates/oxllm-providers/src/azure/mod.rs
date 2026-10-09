//! Azure OpenAI provider — OpenAI-compatible with deployment-name routing.
//!
//! Azure exposes two inference surfaces (both live today):
//!
//! - **v1 (GA, recommended)** — `{api_base}/openai/v1/{endpoint}`, deployment
//!   name in the body `model` field, no dated `api-version` required. Selected
//!   with `api_version: v1`; `preview` is also a valid v1 value (the
//!   v1-preview image/audio routes need `?api-version=preview`) and `latest`
//!   is accepted as a legacy-compatible alias but emitted as the documented
//!   `v1` value. `api_base` may be the resource root
//!   (`https://{resource}.openai.azure.com`), the documented v1 base
//!   (`…/openai/v1/`), or the unified Foundry host
//!   (`https://{resource}.services.ai.azure.com/openai/v1/`); a route suffix
//!   for a supported endpoint already present in `api_base` is normalized
//!   away so the path is never doubled.
//! - **classic (deployment-scoped)** — `{api_base}/openai/deployments/
//!   {deployment}/{endpoint}?api-version={dated}`, used whenever `api_version`
//!   is a dated value for these chat, embedding, image, and audio routes.
//!   The URL's deployment name selects the upstream; a classic chat body's
//!   `model` may be a public routing alias.
//!
//! Auth is `api-key: {key}` for shared keys or `Authorization: Bearer` for an
//! Entra ID access token (`extra.azure_ad_token`, or a JWT-shaped value in the
//! key field). Azure has no `/moderations` endpoint — use Azure AI Content
//! Safety.
//!
//! `api_version` defaults to `2025-02-01-preview` (classic; reference parity).
//! Prefer `api_version: v1` for new deployments: Azure's docs recommend the
//! v1 surface and it needs no periodic version bumps. See
//! `examples/config.example.yaml` for a configuration template.

use std::collections::HashMap;

use oxllm_core::Result;
use oxllm_core::deployment::Deployment;
use oxllm_core::provider::{
    ChatProvider, EmbeddingProvider, Provider, ProviderId, ResolvedCredentials,
};
use percent_encoding::{AsciiSet, CONTROLS, utf8_percent_encode};
use serde_json::Value;

/// The default Azure OpenAI API version.
///
/// `2025-02-01-preview` is the reference `AZURE_DEFAULT_API_VERSION` and adds
/// support for o-series reasoning models, `max_completion_tokens`,
/// `stream_options`, and the newer reasoning-effort / service-tier fields used
/// by GPT-5 and later. Older defaults (`2024-10-21`, `2024-02-15-preview`)
/// lacked these and caused silent 400s on the newer parameters.
pub const DEFAULT_AZURE_API_VERSION: &str = "2025-02-01-preview";

/// Sentinel base URL returned when an Azure deployment has no `api_base`.
///
/// The [`ChatProvider::endpoint_url`] / [`EmbeddingProvider::embedding_url`]
/// trait methods return `String` (not `Result`), so a misconfigured
/// deployment cannot be reported through the trait. Panicking is worse: it
/// kills the per-request task with an opaque abort. Instead we return a URL
/// on the reserved `.invalid` TLD (RFC 2606) — the upstream HTTP call fails
/// fast with a clear DNS-resolution error, which surfaces to the caller as a
/// normal provider/network failure.
pub const AZURE_API_BASE_NOT_CONFIGURED: &str = "https://azure-api-base-not-configured.invalid";

/// Sentinel URL returned when no Azure deployment name can be resolved
/// (bare `azure/` model prefix and no `extra.azure_deployment`), so the
/// request fails fast with a clear DNS error instead of a guaranteed 404.
pub const AZURE_DEPLOYMENT_NAME_NOT_CONFIGURED: &str =
    "https://azure-deployment-name-not-configured.invalid";

/// Characters that must be percent-encoded inside a single URL path segment.
///
/// `/` is included so a deployment name like `"foo/bar"` (which Azure does
/// not allow but a misconfigured `model`/`extra.azure_deployment` can
/// produce) is encoded as `foo%2Fbar` instead of being silently split into
/// two path segments that route the request to the wrong deployment.
const DEPLOYMENT_PATH_SEGMENT: &AsciiSet = &CONTROLS
    .add(b' ')
    .add(b'"')
    .add(b'<')
    .add(b'>')
    .add(b'`')
    .add(b'#')
    .add(b'?')
    .add(b'{')
    .add(b'}')
    .add(b'/')
    .add(b'%');

/// Characters that must be percent-encoded inside a query-string value, so
/// an `api_version` containing `&` or `#` cannot inject extra query
/// parameters or a fragment.
const QUERY_VALUE: &AsciiSet = &CONTROLS
    .add(b' ')
    .add(b'"')
    .add(b'#')
    .add(b'<')
    .add(b'>')
    .add(b'&')
    .add(b'=')
    .add(b'+')
    .add(b'%');

/// The Azure OpenAI provider.
#[derive(Debug, Clone, Default)]
pub struct AzureProvider;

impl AzureProvider {
    fn requires_completion_token_limit(model: &str) -> bool {
        let model = model.to_ascii_lowercase();
        model.contains("gpt-5")
            || model.contains("gpt-6")
            || model.contains("gpt5_series")
            || crate::openai::has_known_reasoning_family(&model)
    }
    fn api_version(deployment: &Deployment) -> String {
        deployment
            .oxllm_params
            .api_version
            .clone()
            .unwrap_or_else(|| DEFAULT_AZURE_API_VERSION.to_string())
    }

    /// Resolve the Azure deployment name for the URL path. Returns `None`
    /// when the model id is the bare `azure/` prefix and no
    /// `extra.azure_deployment` fallback exists — in that case there is no
    /// deployment to route to and the caller must fail loudly instead of
    /// percent-encoding the literal prefix into a guaranteed 404.
    fn deployment_name(deployment: &Deployment) -> Option<String> {
        let model = &deployment.oxllm_params.model;
        if let Some(("azure", rest)) = model.split_once('/') {
            if !rest.is_empty() && !rest.contains('*') {
                return Some(
                    rest.strip_prefix("model_router/")
                        .or_else(|| rest.strip_prefix("model-router/"))
                        .unwrap_or(rest)
                        .to_string(),
                );
            }
        }
        if let Some(Value::String(name)) = deployment.oxllm_params.extra.get("azure_deployment") {
            return Some(name.clone());
        }
        if model == "azure/" || model.is_empty() {
            return None;
        }
        Some(model.clone())
    }

    fn api_base(deployment: &Deployment) -> String {
        match deployment.oxllm_params.api_base.as_deref() {
            Some(base) => {
                let trimmed = base.trim_end_matches('/');
                if trimmed.starts_with("http://") {
                    tracing::warn!(
                        model = %deployment.model_name,
                        api_base = %trimmed,
                        "azure deployment api_base uses plain `http://`; the api-key header \
                         will travel in cleartext to the upstream",
                    );
                }
                trimmed.to_string()
            }
            None => {
                tracing::error!(
                model = %deployment.model_name,
                "azure deployment is missing `api_base` (required for Azure OpenAI); \
                using sentinel URL so the request fails with a DNS error instead of panicking"
                );
                AZURE_API_BASE_NOT_CONFIGURED.to_string()
            }
        }
    }

    /// On v1 the deployment is selected by `model` in the body. Use the same
    /// configured deployment-name precedence as the classic URL route.
    fn rewrite_v1_model(request: &mut Value, deployment: &Deployment) -> Result<()> {
        if !is_azure_v1_api_version(&Self::api_version(deployment)) {
            return Ok(());
        }
        let model = Self::resolve_deployment_name(
            deployment,
            request.get("model").and_then(Value::as_str),
        )?;
        let body = request.as_object_mut().ok_or_else(|| {
            oxllm_core::Error::BadRequest("azure request body must be an object".into())
        })?;
        body.insert("model".into(), Value::String(model));
        Ok(())
    }

    /// Pure URL construction for native image/audio routes and chat/embeddings.
    /// Classic wildcard deployments require a concrete requested selector;
    /// a literal public routing alias cannot identify an upstream deployment.
    /// Explicit azure_deployment remains authoritative for wildcard configs.
    /// Resource-root, /openai, and /openai/v1 bases share version/query handling.
    /// On v1 this selects only the URL; the host still owns the native body model.
    pub fn native_endpoint_url_for_model(
        deployment: &Deployment,
        requested_model: Option<&str>,
        endpoint: &str,
    ) -> oxllm_core::Result<String> {
        let base = deployment.oxllm_params.api_base.as_deref().ok_or_else(|| {
            oxllm_core::Error::Config("azure deployment requires api_base".into())
        })?;
        Self::build_url_with_base(deployment, requested_model, endpoint, base)
    }

    fn resolve_deployment_name(
        deployment: &Deployment,
        requested_model: Option<&str>,
    ) -> Result<String> {
        let prefixed = deployment.oxllm_params.model.strip_prefix("azure/");
        if prefixed.is_none_or(|name| name.is_empty() || name.contains('*'))
            && let Some(value) = deployment
                .oxllm_params
                .extra
                .get("azure_deployment")
                .filter(|value| !value.is_null())
            && value
                .as_str()
                .is_none_or(|name| name.trim().is_empty() || name.contains('*'))
        {
            return Err(oxllm_core::Error::Config(
                "azure_deployment must be a concrete non-empty string".into(),
            ));
        }
        let configured = Self::deployment_name(deployment)
            .ok_or_else(|| oxllm_core::Error::Config("azure requires a deployment name".into()))?;
        let wildcard = configured.contains('*');
        let model = if wildcard {
            let requested = requested_model.ok_or_else(|| {
                oxllm_core::Error::BadRequest(
                    "wildcard azure deployment requires a concrete model deployment name".into(),
                )
            })?;
            let model = strip_azure_prefix(requested);
            if (requested == deployment.model_name.as_str()
                || (!requested.starts_with("azure/")
                    && requested == strip_azure_prefix(&deployment.model_name)))
                && !deployment.model_name.contains('*')
            {
                return Err(oxllm_core::Error::BadRequest(
                    "wildcard azure routing alias is ambiguous; supply a concrete native model selector or configure azure_deployment".into(),
                ));
            }
            model.to_string()
        } else {
            configured
        };
        if model.trim().is_empty()
            || model.contains('*')
            || model.chars().any(char::is_control)
            || model.contains('\\')
            || model
                .split('/')
                .any(|segment| segment == "." || segment == "..")
        {
            let message =
                "azure requires a concrete deployment name without traversal segments".into();
            return Err(if wildcard {
                oxllm_core::Error::BadRequest(message)
            } else {
                oxllm_core::Error::Config(message)
            });
        }
        Ok(model)
    }

    /// A fixed Azure deployment label does not identify its model family.
    /// Capability hints are explicit; wildcard selectors use the resolved model.
    fn capability_model(request: &Value, deployment: &Deployment) -> Option<String> {
        if Self::deployment_name(deployment).is_some_and(|name| name.contains('*')) {
            return request
                .get("model")
                .and_then(Value::as_str)
                .map(|model| strip_azure_prefix(model).to_string());
        }
        deployment
            .model_info
            .as_ref()
            .and_then(|info| info.base_model.as_deref())
            .or_else(|| {
                deployment
                    .oxllm_params
                    .extra
                    .get("base_model")
                    .and_then(Value::as_str)
            })
            .filter(|model| !model.is_empty() && !model.contains('*'))
            .map(str::to_string)
            .or_else(|| {
                // Separate unprefixed model + azure_deployment config keeps the
                // actual model available even though the URL uses an alias.
                (!deployment.oxllm_params.model.starts_with("azure/")
                    && deployment
                        .oxllm_params
                        .extra
                        .get("azure_deployment")
                        .and_then(Value::as_str)
                        .is_some())
                .then(|| deployment.oxllm_params.model.clone())
                .filter(|model| !model.is_empty() && !model.contains('*'))
            })
    }

    fn build_request_url(deployment: &Deployment, endpoint: &str) -> String {
        Self::build_request_url_for_model(deployment, None, endpoint)
    }

    /// Native JSON and multipart bodies identify the Azure deployment using
    /// the same selector as the operation URL, including explicit overrides.
    pub fn native_request_model(
        deployment: &Deployment,
        requested_model: Option<&str>,
    ) -> Result<String> {
        Self::resolve_deployment_name(deployment, requested_model)
    }

    fn build_request_url_for_model(
        deployment: &Deployment,
        requested_model: Option<&str>,
        endpoint: &str,
    ) -> String {
        let base = Self::api_base(deployment);
        Self::build_url_with_base(deployment, requested_model, endpoint, &base).unwrap_or_else(|error| {
            tracing::error!(model = %deployment.model_name, %error, "cannot construct Azure endpoint URL");
            AZURE_DEPLOYMENT_NAME_NOT_CONFIGURED.to_string()
        })
    }

    fn build_url_with_base(
        deployment: &Deployment,
        requested_model: Option<&str>,
        endpoint: &str,
        base: &str,
    ) -> Result<String> {
        if endpoint.is_empty()
            || endpoint.split('/').any(|segment| {
                segment.is_empty()
                    || !segment
                        .chars()
                        .all(|ch| ch.is_ascii_alphanumeric() || ch == '_' || ch == '-')
            })
        {
            return Err(oxllm_core::Error::BadRequest(
                "azure endpoint must be a relative operation path".into(),
            ));
        }
        let (base, fragment) = base
            .split_once('#')
            .map_or((base, None), |(base, fragment)| (base, Some(fragment)));
        let (path_base, extra_query) = split_base_and_query(base);
        let authority = path_base
            .strip_prefix("https://")
            .or_else(|| path_base.strip_prefix("http://"))
            .and_then(|rest| rest.split('/').next());
        if authority.is_none_or(str::is_empty)
            || path_base.contains('\\')
            || path_base
                .chars()
                .any(|ch| ch.is_control() || ch.is_whitespace())
        {
            return Err(oxllm_core::Error::Config(
                "azure api_base must be an absolute HTTP(S) URL".into(),
            ));
        }
        let api_version = Self::api_version(deployment);
        if api_version.trim().is_empty() {
            return Err(oxllm_core::Error::Config(
                "azure api_version must not be empty".into(),
            ));
        }
        // latest is a configuration alias, not a documented wire version.
        let api_version_param = if api_version == "latest" {
            "v1"
        } else {
            api_version.as_str()
        };
        let api_version_enc = utf8_percent_encode(api_version_param, QUERY_VALUE);
        let mut path_base = path_base;
        for suffix in [
            "/chat/completions",
            "/embeddings",
            "/images/generations",
            "/images/edits",
            "/images/variations",
            "/audio/speech",
            "/audio/transcriptions",
            "/audio/translations",
        ] {
            path_base = strip_route_suffix(path_base, suffix);
        }
        let path_base = strip_route_suffix(strip_route_suffix(path_base, "/openai/v1"), "/openai");

        let path = if is_azure_v1_api_version(&api_version) {
            format!("{path_base}/openai/v1/{endpoint}")
        } else {
            let deployment_name = Self::resolve_deployment_name(deployment, requested_model)?;
            let deployment_name = utf8_percent_encode(&deployment_name, DEPLOYMENT_PATH_SEGMENT);
            format!("{path_base}/openai/deployments/{deployment_name}/{endpoint}")
        };

        // Replace every existing version parameter while retaining other
        // query values verbatim, including their percent-encoding.
        let mut query: Vec<&str> = extra_query
            .into_iter()
            .flat_map(|query| query.split('&'))
            .filter(|part| {
                if part.is_empty() {
                    return false;
                }
                let key = part.split_once('=').map_or(*part, |(key, _)| key);
                !percent_encoding::percent_decode_str(key)
                    .decode_utf8()
                    .is_ok_and(|key| key.eq_ignore_ascii_case("api-version"))
            })
            .collect();
        let version_param = format!("api-version={api_version_enc}");
        query.push(&version_param);
        let mut url = format!("{path}?{}", query.join("&"));
        if let Some(fragment) = fragment {
            url.push('#');
            url.push_str(fragment);
        }
        Ok(url)
    }
}

fn strip_azure_prefix(model: &str) -> &str {
    model.strip_prefix("azure/").unwrap_or(model)
}

/// Remove a trailing route suffix (case-insensitively) that the operator
/// already included in `api_base`, so the provider can append the canonical
/// Azure route without doubling it.
fn strip_route_suffix<'a>(path: &'a str, suffix: &str) -> &'a str {
    let trimmed = path.trim_end_matches('/');
    if let Some(end) = trimmed.len().checked_sub(suffix.len())
        && trimmed
            .get(end..)
            .is_some_and(|tail| tail.eq_ignore_ascii_case(suffix))
    {
        trimmed[..end].trim_end_matches('/')
    } else {
        trimmed
    }
}

/// Whether the `api_version` selects the Azure v1 API surface, which routes
/// via `/openai/v1/...` (model in body) instead of the deployment-path URL.
/// Mirrors the reference Azure v1 API-version detection.
fn is_azure_v1_api_version(api_version: &str) -> bool {
    matches!(api_version, "preview" | "latest" | "v1")
}

/// Split an `api_base` into its path and (optional) query-string parts, so a
/// base that already carries query params (e.g. `…?api-version=old`) does not
/// produce a malformed URL when the deployment path is appended. The path
/// part has any trailing `/` trimmed (the route format supplies its own
/// leading slash); [`Self::api_base`] already trims the no-query case.
fn split_base_and_query(base: &str) -> (&str, Option<&str>) {
    match base.split_once('?') {
        Some((path, query)) => (path.trim_end_matches('/'), Some(query)),
        None => (base, None),
    }
}

impl ChatProvider for AzureProvider {
    fn endpoint_url(&self, deployment: &Deployment) -> String {
        Self::build_request_url(deployment, "chat/completions")
    }

    fn endpoint_url_for_request(&self, deployment: &Deployment, request: &Value) -> String {
        Self::build_request_url_for_model(
            deployment,
            request.get("model").and_then(Value::as_str),
            "chat/completions",
        )
    }

    fn apply_auth(
        &self,
        deployment: &Deployment,
        creds: &ResolvedCredentials,
        headers: &mut HashMap<String, String>,
    ) {
        match creds {
            ResolvedCredentials::Azure { api_key } => {
                // Keyless workloads commonly paste an Entra ID access token
                // into the key field. Azure rejects a JWT on `api-key` with a
                // 401, so route token-shaped credentials to the bearer header
                // (the documented `Authorization: Bearer` path) and warn that
                // a static token expires (~1h).
                if looks_like_jwt(api_key) {
                    tracing::warn!(
                        deployment = %deployment.model_name,
                        "azure api_key looks like an Entra ID access token; sending it as \
                         `Authorization: Bearer`. Prefer `extra.azure_ad_token` (resolved per \
                         config load) or refresh the token out of band — static tokens expire."
                    );
                    headers.insert("authorization".to_string(), format!("Bearer {api_key}"));
                } else {
                    headers.insert("api-key".to_string(), api_key.clone());
                }
            }
            // A generic `ApiKey` is a plausible Azure shared key when the
            // resolver did not tag it as `Azure`; forward it as `api-key`
            // (or as a bearer token when token-shaped, as above).
            ResolvedCredentials::ApiKey { bearer } => {
                if looks_like_jwt(bearer) {
                    headers.insert("authorization".to_string(), format!("Bearer {bearer}"));
                } else {
                    headers.insert("api-key".to_string(), bearer.clone());
                }
            }
            // An Azure AD / Entra ID bearer token is NOT a shared key — Azure
            // rejects it from the `api-key` header with a 401 — so it must go
            // on `Authorization: Bearer`, matching the reference
            // `_base_validate_azure_environment` AAD fallback.
            ResolvedCredentials::AzureAd { token } => {
                headers.insert("authorization".to_string(), format!("Bearer {token}"));
            }
            // Anything else (GcpBearer, AwsSigV4, PreSignedHeaders, …) cannot
            // be expressed as an Azure credential. Refuse to misroute: leave
            // the request unauthenticated (the upstream 401 is clearer than a
            // silently-wrong header).
            other => {
                let kind = match other {
                    ResolvedCredentials::AwsSigV4 { .. } => "AwsSigV4",
                    ResolvedCredentials::GcpBearer { .. } => "GcpBearer",
                    ResolvedCredentials::PreSignedHeaders(_) => "PreSignedHeaders",
                    _ => "other",
                };
                tracing::warn!(
                    deployment = %deployment.model_name,
                    cred_kind = kind,
                    "azure apply_auth got a credential that is not an Azure shared key or AAD token; \
                     request will be sent unauthenticated",
                );
            }
        }
    }

    fn transform_request(&self, request: &Value, deployment: &Deployment) -> Result<Value> {
        if Self::deployment_name(deployment).is_some_and(|name| name.contains('*')) {
            Self::resolve_deployment_name(
                deployment,
                request.get("model").and_then(Value::as_str),
            )?;
        }
        let mut out = normalize_azure_request(request)?;
        drop_tool_choice_without_tools(&mut out);
        Self::rewrite_v1_model(&mut out, deployment)?;
        if let Some(model) = Self::capability_model(request, deployment) {
            crate::openai::apply_openai_reasoning_limits(&mut out, &model);
            if Self::requires_completion_token_limit(&model) {
                crate::openai::apply_openai_token_limit_alias(&mut out);
            }
        } else if let Some(name) = Self::deployment_name(deployment)
            && Self::requires_completion_token_limit(&name)
        {
            crate::openai::apply_openai_token_limit_alias(&mut out);
        }
        crate::prompt_cache::apply(&mut out, request, deployment);
        Ok(out)
    }

    fn transform_response(&self, body: &Value, _model: &str) -> Result<Value> {
        Ok(body.clone())
    }

    fn transform_stream_chunk(&self, chunk: &str) -> Result<Option<Value>> {
        crate::openai_compat::sse::parse_openai_stream_chunk(chunk)
    }

    fn supported_params(&self) -> Option<Vec<&str>> {
        None
    }
}

impl EmbeddingProvider for AzureProvider {
    fn embedding_url(&self, deployment: &Deployment) -> String {
        Self::build_request_url(deployment, "embeddings")
    }

    fn apply_auth(
        &self,
        deployment: &Deployment,
        creds: &ResolvedCredentials,
        headers: &mut HashMap<String, String>,
    ) {
        ChatProvider::apply_auth(self, deployment, creds, headers);
    }

    fn transform_request(&self, request: &Value, deployment: &Deployment) -> Result<Value> {
        let mut out = request.clone();
        Self::rewrite_v1_model(&mut out, deployment)?;
        Ok(out)
    }

    fn transform_response(&self, body: &Value, _model: &str) -> Result<Value> {
        Ok(body.clone())
    }
}

/// Whether a value is a JWT-shaped credential (Entra ID access tokens are
/// JWTs: three dot-separated segments, the header starting with `eyJ`).
/// Azure shared keys are 32-char hex strings and never match.
fn looks_like_jwt(value: &str) -> bool {
    value.starts_with("eyJ") && value.matches('.').count() == 2
}

/// Azure-specific request normalization, mirroring the reference
/// `convert_to_azure_openai_messages` (`core_utils/prompt_templates/
/// factory.py`). Azure is stricter than OpenAI on two message shapes that
/// OpenAI accepts but Azure rejects with a 400:
///
/// 1. A user message whose `content` array carries an `image_url` part whose
///    `image_url` field is a bare string — Azure requires the object form
///    `{"url": <string>}`.
/// 2. An assistant message with a legacy `function_call` whose `arguments`
///    field is `null` — Azure requires `arguments` to be a (possibly empty)
///    string.
///
/// Modern clients already send these in the Azure-compatible shape; this is a
/// defensive, in-place fixup so a stray OpenAI-shaped request does not get a
/// confusing 400 from the upstream.
/// Azure rejects `tool_choice` when the request has no `tools` or `functions`.
fn drop_tool_choice_without_tools(body: &mut Value) {
    let Some(obj) = body.as_object_mut() else {
        return;
    };
    let has_tools = obj
        .get("tools")
        .is_some_and(|v| v.as_array().is_some_and(|a| !a.is_empty()));
    let has_functions = obj
        .get("functions")
        .is_some_and(|v| v.as_array().is_some_and(|a| !a.is_empty()));
    if !has_tools && !has_functions {
        obj.remove("tool_choice");
    }
}

fn normalize_azure_request(request: &Value) -> Result<Value> {
    let mut out = request.clone();
    let Some(messages) = out.get_mut("messages").and_then(|m| m.as_array_mut()) else {
        return Ok(out);
    };
    for msg in messages.iter_mut() {
        match msg.get("role").and_then(|r| r.as_str()) {
            Some("assistant") => {
                if let Some(fc) = msg
                    .get_mut("function_call")
                    .filter(|value| !value.is_null())
                {
                    // The reference Azure tool-call helper coerces when
                    // `arguments` is missing OR null (`get(...) is None`),
                    // since Azure rejects a `function_call` with no string
                    // `arguments` field with a 400.
                    let fc = fc.as_object_mut().ok_or_else(|| {
                        oxllm_core::Error::BadRequest(
                            "azure assistant function_call must be an object".into(),
                        )
                    })?;
                    match fc.get("arguments") {
                        None | Some(Value::Null) => {
                            fc.insert("arguments".into(), Value::String(String::new()));
                        }
                        Some(Value::String(_)) => {}
                        Some(_) => {
                            return Err(oxllm_core::Error::BadRequest(
                                "azure function_call arguments must be a string".into(),
                            ));
                        }
                    }
                }
            }
            Some("user") => {
                if let Some(parts) = msg.get_mut("content").and_then(|c| c.as_array_mut()) {
                    for part in parts.iter_mut() {
                        let is_image =
                            part.get("type").and_then(|t| t.as_str()) == Some("image_url");
                        if is_image {
                            if let Some(iu) = part.get_mut("image_url") {
                                if iu.is_string() {
                                    let url = iu.take();
                                    *iu = serde_json::json!({"url": url});
                                }
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }
    Ok(out)
}

impl Provider for AzureProvider {
    fn id(&self) -> ProviderId {
        "azure"
    }

    fn supports(&self, endpoint: oxllm_core::Endpoint) -> bool {
        matches!(
            endpoint,
            oxllm_core::Endpoint::ChatCompletions
                | oxllm_core::Endpoint::Embeddings
                | oxllm_core::Endpoint::ImagesGenerations
                | oxllm_core::Endpoint::AudioSpeech
                | oxllm_core::Endpoint::AudioTranscriptions
        )
    }

    fn chat(&self) -> Option<&dyn ChatProvider> {
        Some(self)
    }

    fn embeddings(&self) -> Option<&dyn EmbeddingProvider> {
        Some(self)
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod validation_tests;
