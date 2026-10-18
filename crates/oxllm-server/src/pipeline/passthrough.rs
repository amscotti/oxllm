//! Unified passthrough tiers.
//!
//! Historically oxllm had four inconsistent passthrough behaviors (transformed
//! chat/embeddings, hooked raw via `forward_to_model`, raw `forward_passthrough`,
//! and the catch-all). This module collapses them into a single
//! [`forward_with_mode`] entry point selected by [`PassthroughMode`].
//!
//! - [`PassthroughMode::Transformed`] is documented here for completeness but
//!   never flows through [`forward_with_mode`]: provider-transform paths
//!   (chat, embeddings, messages, responses) own their dedicated handlers.
//! - [`PassthroughMode::HookedRaw`] runs the full [`EndpointPipeline`]
//!   (pre-call hooks, deployment RPM/TPM limits, upstream call, post-success
//!   hooks with cost/spend). This is the default tier for platform endpoints
//!   (batches, files, assistants, fine-tuning) and the LLM-passthrough
//!   surfaces (rerank, images, audio, moderations).
//! - [`PassthroughMode::Raw`] resolves the deployment and forwards without
//!   metering hooks, while retaining model ACLs and configured guardrails. Reserved for
//!   `general_settings.pass_through_endpoints` and explicit opt-outs.
//!
//! Accompanying the entry point is [`upstream_base_url`], which replaces the
//! drift-prone hardcoded defaults in `handlers/passthrough.rs` with a lookup
//! against [`default_base_for_provider`] — a table kept in sync with the
//! `DEFAULT_*_BASE` constants in `oxllm-providers`.

use std::collections::HashMap;

use axum::body::Bytes;
use axum::http::Method;
use axum::response::{IntoResponse, Response};
use oxllm_core::{Deployment, UsageSummary};
use serde_json::{Value, json};

use crate::auth::AuthUser;
use crate::error::ApiError;
use crate::pipeline::{
    EndpointPipeline, apply_cost_header, record_deployment_tpm, stamp_ratelimit_headers,
    stamp_request_id_header,
};
use crate::state::AppState;

pub(crate) const DEFAULT_TIMEOUT_SECS: u64 = 120;

/// Resolve the deployment credentials using the deployment's registered
/// resolver. Passthrough surfaces need the full [`oxllm_core::provider::ResolvedCredentials`]
/// rather than only a bearer string because providers differ in wire auth.
pub(crate) async fn resolve_credentials(
    state: &AppState,
    deployment: &Deployment,
) -> Result<oxllm_core::provider::ResolvedCredentials, ApiError> {
    state
        .credential_resolver
        .resolve_for(deployment)
        .resolve(deployment)
        .await
        .map_err(ApiError::from)
}

/// Build provider-native authentication headers for passthrough surfaces.
///
/// The provider registry already encodes the wire-level authentication for
/// transformed endpoints (`x-api-key` + `anthropic-version`, Azure `api-key`,
/// OpenAI-compatible Bearer). Passthrough must use that same knowledge rather
/// than blindly sending Bearer. If a registered provider has no transform that
/// emits headers, or the provider is not registered, fall back to Bearer to
/// preserve the previous gateway behavior.
///
/// Audio-only providers (ElevenLabs' `xi-api-key`, Deepgram's `Token` scheme)
/// implement only the bare [`oxllm_core::Provider`] trait with no
/// chat/embeddings/rerank surface, so their non-standard auth schemes are
/// dispatched explicitly here by slug — otherwise the Bearer fallback would
/// send a scheme the upstream rejects.
pub(crate) fn provider_auth_headers(
    provider: Option<&std::sync::Arc<dyn oxllm_core::Provider>>,
    deployment: &Deployment,
    creds: &oxllm_core::provider::ResolvedCredentials,
) -> Result<HashMap<String, String>, ApiError> {
    let mut headers = HashMap::new();
    if let Some(provider) = provider {
        if let Some(chat) = provider.chat() {
            chat.apply_auth(deployment, creds, &mut headers);
        } else if let Some(embeddings) = provider.embeddings() {
            embeddings.apply_auth(deployment, creds, &mut headers);
        } else if let Some(rerank) = provider.rerank() {
            rerank.apply_auth(deployment, creds, &mut headers);
        }
    }
    if headers.is_empty() {
        if let Some(provider) = provider {
            match provider.id() {
                "elevenlabs" => {
                    oxllm_providers::elevenlabs::elevenlabs_auth_headers(creds, &mut headers);
                }
                "deepgram" => {
                    oxllm_providers::deepgram::deepgram_auth_headers(creds, &mut headers);
                }
                _ => {}
            }
        }
    }
    if headers.is_empty()
        && !matches!(
            creds,
            oxllm_core::provider::ResolvedCredentials::AwsSigV4 { .. }
        )
    {
        // SigV4 deployments (Bedrock) intentionally emit no headers here:
        // the signature covers the serialized body and is added by the
        // send path, mirroring the chat pipeline.
        let key = creds
            .best_effort_bearer()
            .ok_or_else(|| ApiError::internal("no bearer credential available for passthrough"))?;
        headers.insert("authorization".to_string(), format!("Bearer {key}"));
    }
    oxllm_core::provider::apply_auth_overrides(deployment, creds, &mut headers);
    Ok(headers)
}

/// The three passthrough tiers, unified under [`forward_with_mode`].
pub enum PassthroughMode {
    /// Provider-transform paths (chat, embeddings). Documented here for
    /// completeness; transformed paths use dedicated handlers and never
    /// invoke [`forward_with_mode`].
    Transformed,
    /// Raw HTTP to `{api_base}/{upstream_path}` driven through the full
    /// [`EndpointPipeline`] (pre-call hooks, deployment limits, post-success
    /// hooks with cost/spend). Default for platform endpoints.
    HookedRaw {
        /// Path appended to the deployment's resolved base URL
        /// (e.g. `"batches"`, `"files/{file_id}/content"`).
        upstream_path: String,
    },
    /// Resolves the deployment, enforces ACLs/configured guardrails, and
    /// forwards without running metering hooks.
    /// Reserved for `pass_through_endpoints` and explicit opt-outs.
    Raw {
        /// Path appended to the deployment's resolved base URL.
        upstream_path: String,
    },
}

/// Default upstream base URLs for providers that don't set `api_base` in
/// config. Kept in sync with the `DEFAULT_*_BASE` constants in
/// `oxllm-providers`. Resolved here (rather than via the registry) because
/// passthrough paths bypass the provider transform traits.
pub fn default_base_for_provider(slug: &str) -> Option<&'static str> {
    match slug {
        "openai" => Some("https://api.openai.com/v1"),
        "baseten" => Some(oxllm_providers::baseten::DEFAULT_BASETEN_BASE),
        "anthropic" => Some("https://api.anthropic.com/v1"),
        "cohere" => Some("https://api.cohere.com/v2"),
        "groq" => Some("https://api.groq.com/openai/v1"),
        "xai" => Some("https://api.x.ai/v1"),
        "together_ai" => Some("https://api.together.xyz/v1"),
        "openrouter" => Some("https://openrouter.ai/api/v1"),
        "mistral" => Some("https://api.mistral.ai/v1"),
        "gemini" => Some("https://generativelanguage.googleapis.com/v1beta/openai"),
        "ollama" => Some("http://localhost:11434/v1"),
        "ai21" => Some(oxllm_providers::ai21::DEFAULT_AI21_BASE),
        "bedrock_mantle" => Some(oxllm_providers::bedrock_mantle::DEFAULT_BEDROCK_MANTLE_BASE),
        "cerebras" => Some(oxllm_providers::cerebras::DEFAULT_CEREBRAS_BASE),
        "dashscope" => Some(oxllm_providers::dashscope::DEFAULT_DASHSCOPE_BASE),
        "deepgram" => Some(oxllm_providers::deepgram::DEFAULT_DEEPGRAM_BASE),
        "deepinfra" => Some(oxllm_providers::deepinfra::DEFAULT_DEEPINFRA_BASE),
        "deepseek" => Some(oxllm_providers::deepseek::DEFAULT_DEEPSEEK_BASE),
        "elevenlabs" => Some(oxllm_providers::elevenlabs::DEFAULT_ELEVENLABS_BASE),
        "fireworks_ai" => Some(oxllm_providers::fireworks_ai::DEFAULT_FIREWORKS_AI_BASE),
        "huggingface" => Some(oxllm_providers::huggingface::DEFAULT_HUGGINGFACE_BASE),
        "jina_ai" => Some(oxllm_providers::jina_ai::DEFAULT_JINA_BASE),
        "meta" => Some(oxllm_providers::meta::DEFAULT_META_BASE),
        "nvidia_nim" => Some(oxllm_providers::nvidia_nim::DEFAULT_NVIDIA_NIM_BASE),
        "perplexity" => Some(oxllm_providers::perplexity::DEFAULT_PERPLEXITY_BASE),
        "replicate" => Some(oxllm_providers::replicate::DEFAULT_REPLICATE_BASE),
        "sambanova" => Some(oxllm_providers::sambanova::DEFAULT_SAMBANOVA_BASE),
        "voyage" => Some(oxllm_providers::voyage::DEFAULT_VOYAGE_BASE),
        "watsonx" => Some(oxllm_providers::watsonx::DEFAULT_WATSONX_BASE),
        _ => None,
    }
}

/// Resolve the upstream base URL for a deployment. Uses
/// `deployment.oxllm_params.api_base` when set; otherwise falls back to the
/// provider default table. Returns a typed error when neither is available.
pub fn upstream_base_url(deployment: &Deployment) -> Result<String, ApiError> {
    if let Some(base) = &deployment.oxllm_params.api_base {
        return Ok(base.trim_end_matches('/').to_string());
    }
    default_base_for_provider(deployment.provider_slug())
        .map(|s| s.to_string())
        .ok_or_else(|| {
            ApiError::internal(format!(
                "no api_base configured and no default known for provider '{}'",
                deployment.provider_slug()
            ))
        })
}

/// Resolve the upstream base URL for the catch-all (`provider_passthrough`)
/// path. Differs from [`upstream_base_url`] in the fallback: when no
/// `api_base` is configured, the provider default is reduced to its **host
/// only** (the version path segment is dropped).
///
/// The catch-all forwards the full client-supplied path, which already
/// carries the API version prefix (standard SDKs send
/// `/anthropic/v1/messages`, so `remaining_path = "v1/messages"`). The
/// [`default_base_for_provider`] table includes that same version prefix
/// (e.g. `https://api.anthropic.com/v1`) because the hooked-raw callers
/// append short suffixes (`"batches"`, `"rerank"`) where the in-base `/v1`
/// is correct. Feeding a version-prefixed client path into that same table
/// would double the prefix (`https://api.anthropic.com/v1/v1/messages`) —
/// a guaranteed upstream 404. This matches the reference model: per-provider
/// catch-all routes use a version-less base and let the endpoint path carry
/// `/v1/`. When `api_base` is explicitly configured it is used verbatim
/// (the deployment owner controls the join).
pub fn catch_all_upstream_base_url(deployment: &Deployment) -> Result<String, ApiError> {
    if let Some(base) = &deployment.oxllm_params.api_base {
        return Ok(base.trim_end_matches('/').to_string());
    }
    let default = default_base_for_provider(deployment.provider_slug()).ok_or_else(|| {
        ApiError::internal(format!(
            "no api_base configured and no default known for provider '{}'",
            deployment.provider_slug()
        ))
    })?;
    let mut url = url::Url::parse(default)
        .map_err(|e| ApiError::internal(format!("invalid default base URL '{default}': {e}")))?;
    // Drop the version path so the client-supplied path is the sole source of
    // the version prefix. For special schemes `set_path("")` normalises to
    // "/", so trim the trailing slash to keep the value host-only (matching
    // the convention `upstream_base_url` applies to a configured `api_base`).
    url.set_path("");
    Ok(url.to_string().trim_end_matches('/').to_string())
}

/// Bag of request inputs passed to [`forward_with_mode`]. Grouping these
/// into a struct keeps the entry-point signature under clippy's
/// `too_many_arguments` threshold and gives call sites a clear name for each
/// field.
pub struct PassthroughRequest<'a> {
    pub method: Method,
    pub model: Option<&'a str>,
    pub query: Option<&'a str>,
    pub body: Bytes,
    pub content_type: Option<&'a str>,
    /// Optional normalized sticky-session identity, shared with attribution.
    pub session_id: Option<String>,
    /// Routing tag from `x-oxllm-tag` (optional). Used when
    /// `routing_strategy: tag_based`.
    pub tag: Option<String>,
    /// Every header tag, normalized and bounded by the shared tag parser.
    pub tags: Vec<String>,
    /// Vetted client headers to forward upstream (e.g. `anthropic-beta` for
    /// the native Anthropic passthrough). Built with
    /// [`vetted_client_headers`]; `send_upstream` trusts the slice it
    /// receives, so the filtering helper is the single chokepoint that
    /// strips auth-sensitive and proxy-specific headers.
    pub forward_headers: Vec<(axum::http::HeaderName, axum::http::HeaderValue)>,
    /// Provider slug the route is hardwired to (e.g. `"openai"` for
    /// `/v1/images/*`, `"anthropic"` for the native Anthropic passthrough).
    /// Used by `resolve_model` when neither the `model` parameter nor the
    /// body's `model` field is present (a model-less GET to `/v1/files`, a
    /// multipart file upload with no model). A fallback must identify one
    /// deployment of that provider; missing/ambiguous namespaces are errors.
    pub provider_hint: Option<&'a str>,
}

impl<'a> PassthroughRequest<'a> {
    /// Build a request bag from its in-order fields.
    pub fn new(
        method: Method,
        model: Option<&'a str>,
        query: Option<&'a str>,
        body: Bytes,
        content_type: Option<&'a str>,
    ) -> Self {
        Self {
            method,
            model,
            query,
            body,
            content_type,
            session_id: None,
            tag: None,
            tags: Vec::new(),
            forward_headers: Vec::new(),
            provider_hint: None,
        }
    }

    /// Attach a session id for `session_affinity` routing and attribution.
    /// Blank values are absent; others are trimmed and bounded according to
    /// [`super::normalize_session_id`] and [`super::MAX_SESSION_ID_BYTES`].
    pub fn with_session_id(mut self, session_id: Option<String>) -> Self {
        self.session_id = session_id
            .as_deref()
            .and_then(crate::pipeline::normalize_session_id);
        self
    }

    /// Attach a routing tag for `tag_based` routing.
    pub fn with_tag(mut self, tag: Option<String>) -> Self {
        self.tag = tag;
        self
    }

    /// Attach routing metadata from all header values. Body metadata is merged
    /// by the pipeline after parsing, with body tags first and explicit session
    /// headers taking precedence over Anthropic metadata.
    pub fn with_request_headers(mut self, headers: &axum::http::HeaderMap) -> Self {
        self.session_id = crate::pipeline::extract_session_id(headers);
        self.tags = crate::pipeline::extract_tags(headers, None);
        self.tag = self.tags.first().cloned();
        self
    }

    /// Attach vetted client headers to forward upstream. Use
    /// [`vetted_client_headers`] to build the slice from the incoming
    /// request's `HeaderMap`.
    pub fn with_forward_headers(
        mut self,
        headers: Vec<(axum::http::HeaderName, axum::http::HeaderValue)>,
    ) -> Self {
        self.forward_headers = headers;
        self
    }

    /// Attach a provider slug hint for model-less routing fallback. Endpoint-
    /// specific handlers (e.g. `/v1/images/*` → `"openai"`) should set this
    /// so a model-less request resolves only within that provider's namespace.
    pub fn with_provider_hint(mut self, provider: &'a str) -> Self {
        self.provider_hint = Some(provider);
        self
    }
}

/// Headers a passthrough caller may legitimately forward to the upstream.
/// Kept intentionally small: adding to this set requires a review because
/// each entry is a potential auth-bypass or header-injection surface. Note
/// `authorization`, `host`, `content-length`, `transfer-encoding`,
/// `x-oxllm-*`, and `x-forwarded-*` are deliberately absent — those are
/// either set by [`send_upstream`] itself or are proxy-internal.
const FORWARDABLE_CLIENT_HEADERS: &[&str] = &["anthropic-beta", "accept", "traceparent", "baggage"];

/// Filter a client request's headers down to a vetted allowlist safe to
/// forward upstream. Used by the native Anthropic passthrough (and any other
/// passthrough surface that wants to opt into header forwarding) so that
/// `anthropic-beta` (required for prompt caching, extended output, computer
/// use, PDF input, token-efficient tools, etc.) and `accept` reach the
/// upstream, while auth-sensitive and proxy-specific headers are stripped.
pub fn vetted_client_headers(
    headers: &axum::http::HeaderMap,
) -> Vec<(axum::http::HeaderName, axum::http::HeaderValue)> {
    headers
        .iter()
        .filter(|(name, _)| FORWARDABLE_CLIENT_HEADERS.contains(&name.as_str()))
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect()
}

/// Unified passthrough entry point. Selects behavior via [`PassthroughMode`].
///
/// `model` is used to resolve the deployment (via the router) and to populate
/// the call context for hooked tiers. When `None`, the function tries to
/// extract a `model` field from a JSON body. Model-less platform operations
/// require an unambiguous provider/deployment; inference requires a model.
pub async fn forward_with_mode(
    state: &AppState,
    auth: &AuthUser,
    mode: PassthroughMode,
    req: PassthroughRequest<'_>,
) -> Result<Response, ApiError> {
    let inference = match &mode {
        PassthroughMode::HookedRaw { upstream_path } | PassthroughMode::Raw { upstream_path } => {
            inference_path(upstream_path)
        }
        PassthroughMode::Transformed => false,
    };
    let result = match mode {
        PassthroughMode::Transformed => Err(ApiError::internal(
            "Transformed passthrough is served by dedicated chat/embeddings handlers",
        )),
        PassthroughMode::HookedRaw { upstream_path } => {
            hooked_raw(state, auth, &upstream_path, req).await
        }
        PassthroughMode::Raw { upstream_path } => {
            raw_forward(state, auth, &upstream_path, req).await
        }
    };
    result.map_err(|error| {
        if inference {
            error.for_inference()
        } else {
            error
        }
    })
}

async fn hooked_raw(
    state: &AppState,
    auth: &AuthUser,
    upstream_path: &str,
    req: PassthroughRequest<'_>,
) -> Result<Response, ApiError> {
    // Multipart inference bodies carry their model in a form field. Extract
    // it so the request can be routed and the field rewritten to the
    // deployment's upstream model; without a model the request is rejected
    // rather than silently routed to an arbitrary deployment.
    let multipart_model = if is_json_content_type(req.content_type) || req.body.is_empty() {
        None
    } else {
        crate::pipeline::multipart::extract_field(&req.body, req.content_type, "model")
            .filter(|model| !model.trim().is_empty())
    };
    let effective_model = req.model.or(multipart_model.as_deref());
    validate_raw_model_carrier(upstream_path, req.content_type, effective_model)?;
    let mut request_body: Value = if is_json_content_type(req.content_type) && !req.body.is_empty()
    {
        serde_json::from_slice(&req.body)
            .map_err(|e| ApiError::bad_request(format!("invalid JSON body: {e}")))?
    } else {
        Value::Null
    };
    if !request_body.is_null() {
        crate::security::check_json_depth(&request_body, crate::security::MAX_JSON_DEPTH)?;
        crate::orchestration::request_controls::validate(&request_body)?;
    }

    require_inference_model(upstream_path, effective_model, &request_body)?;
    let resolved_model = resolve_model(state, effective_model, &request_body, req.provider_hint)?;
    let tags = request_tags(&req, &request_body);
    let session_id = request_session_id(&req, &request_body);
    let mut pipeline = EndpointPipeline::for_request_with_metadata(
        state,
        auth,
        &resolved_model,
        request_body.clone(),
        &tags,
        session_id.clone(),
    )
    .await?;
    if let Err(error) = validate_nested_model_acl(auth, &request_body) {
        crate::pipeline::record_request_rejection(state, &pipeline.call_ctx, &error).await;
        return Err(error);
    }
    pipeline.call_ctx.cache_namespace = Some(format!("raw:{}:{upstream_path}", req.method));
    pipeline.apply_request_tags(&tags).await?;
    let guarded = crate::pipeline::guardrails::apply_raw_request_guardrails(
        state,
        &req.method,
        upstream_path,
        &mut request_body,
    )
    .await?;
    if guarded {
        pipeline.call_ctx.request_body = std::sync::Arc::new(request_body.clone());
    }

    // File IDs belong to an upstream account. Never guess a provider/account
    // from SDK auth header spelling or randomly pick among namespaces.
    validate_file_namespace(state, upstream_path, &resolved_model, req.provider_hint)?;

    if let Some(cached) = pipeline.pre_call().await? {
        pipeline.call_ctx.cache_hit = true;
        // Extract usage from the cached body (same as the chat orchestrator's
        // cache-hit path) so post_success debits budgets / writes a spend log
        // for the cached response instead of recording zero cost. No
        // deployment is resolved here, so derive the provider from the cached
        // response's `model` prefix when present.
        let usage = crate::orchestration::usage::extract_usage(&cached);
        let cached_model = cached
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or(&resolved_model)
            .to_string();
        let cached_provider = cached_provider_prefix(&cached_model);
        let cost = pipeline
            .post_success(
                &cached_model,
                usage.as_ref(),
                Some(&cached),
                cached_provider,
            )
            .await;
        let mut resp = axum::Json(cached).into_response();
        apply_cost_header(&mut resp, cost);
        stamp_request_id_header(&mut resp, pipeline.request_id());
        stamp_ratelimit_headers(&mut resp, state, &pipeline.call_ctx).await;
        return Ok(resp);
    }

    let deployment = match resolve_deployment_with_tags(state, &resolved_model, session_id, &tags) {
        Ok(d) => d,
        Err(err) => {
            // Release the budget reservation pinned in pre_call — otherwise
            // it leaks for the entire budget window (router returned no
            // deployment, e.g. all are cooling down).
            pipeline
                .post_failure(&crate::pipeline::core_error_from_api(&err))
                .await;
            return Err(err);
        }
    };
    if let Err(err) = validate_raw_endpoint(state, &deployment, upstream_path, req.provider_hint) {
        pipeline
            .post_failure(&crate::pipeline::core_error_from_api(&err))
            .await;
        return Err(err);
    }
    if let Err(err) =
        validate_selected_file_deployment(state, upstream_path, &resolved_model, &deployment)
    {
        pipeline
            .post_failure(&crate::pipeline::core_error_from_api(&err))
            .await;
        return Err(err);
    }
    let dep_key = oxllm_core::router::deployment_key(&deployment);
    if let Err(error) = crate::pipeline::prepare_deployment_pricing(
        state,
        &mut pipeline.call_ctx,
        &deployment,
        &request_body,
    )
    .await
    {
        pipeline.post_failure(&error).await;
        return Err(ApiError::from(error));
    }
    if let Err(err) = pipeline.check_limits(&deployment, &dep_key).await {
        // Release the budget reservation pinned in pre_call — otherwise it
        // leaks for the entire budget window.
        pipeline
            .post_failure(&crate::pipeline::core_error_from_api(&err))
            .await;
        return Err(err);
    }
    let native_selector =
        effective_model.or_else(|| request_body.get("model").and_then(Value::as_str));
    let url =
        match build_native_upstream_url(&deployment, upstream_path, req.query, native_selector) {
            Ok(u) => u,
            Err(err) => {
                // Release the budget reservation pinned in pre_call (and the
                // deployment RPM slot from check_limits). build_upstream_url
                // fails when no api_base is configured and the provider slug is
                // absent from the default-base table — without this release the
                // key budget estimate leaks for the whole budget window.
                pipeline
                    .post_failure(&crate::pipeline::core_error_from_api(&err))
                    .await;
                return Err(err);
            }
        };

    // Rewrite the model field in JSON bodies to the deployment's actual
    // upstream model name. Clients send oxllm aliases (e.g. "claude") that
    // the upstream provider doesn't recognize — the deployment knows the
    // real model (e.g. "anthropic/claude-sonnet-4-6" → "claude-sonnet-4-6").
    let body_to_send = if is_json_content_type(req.content_type) {
        let body = if guarded {
            Bytes::from(request_body.to_string())
        } else {
            req.body
        };
        match rewrite_model_in_body(body, &deployment, inference_path(upstream_path)) {
            Ok(body) => body,
            Err(err) => {
                pipeline
                    .post_failure(&crate::pipeline::core_error_from_api(&err))
                    .await;
                return Err(err);
            }
        }
    } else if multipart_model.is_some() {
        // Replace the alias in the form field with the deployment's upstream
        // model id (e.g. "my-whisper" → "whisper-1"), preserving every other
        // byte of the multipart body.
        let upstream_model = match resolve_raw_model(&deployment, multipart_model.as_deref()) {
            Ok(model) => model,
            Err(error) => {
                pipeline
                    .post_failure(&crate::pipeline::core_error_from_api(&error))
                    .await;
                return Err(error);
            }
        };
        match crate::pipeline::multipart::rewrite_field(
            &req.body,
            req.content_type,
            "model",
            &upstream_model,
        ) {
            Some(body) => body,
            None => {
                let err = ApiError::bad_request(
                    "multipart model field could not be located for rewrite; \
                     resend with a model field or use the JSON transport",
                );
                pipeline
                    .post_failure(&crate::pipeline::core_error_from_api(&err))
                    .await;
                return Err(err);
            }
        }
    } else {
        req.body
    };

    let body_to_send = if endpoint_path(upstream_path) == "messages/count_tokens"
        && deployment.provider_slug() == "bedrock"
    {
        // Bedrock counting is its own wire API (CountTokens), not the
        // Anthropic messages shape: build the Converse/InvokeModel count
        // body from the (model-rewritten) chat-shaped input.
        let input: Value = serde_json::from_slice(&body_to_send)
            .map_err(|_| ApiError::bad_request("invalid token-counting body"))?;
        let count_body =
            oxllm_providers::bedrock::count_tokens::build_count_tokens_body(&input, &deployment)
                .map_err(ApiError::from)?;
        Bytes::from(count_body.to_string())
    } else if endpoint_path(upstream_path) == "messages/count_tokens" {
        let input: Value = serde_json::from_slice(&body_to_send)
            .map_err(|_| ApiError::bad_request("invalid token-counting body"))?;
        let model = resolve_raw_model(&deployment, input.get("model").and_then(Value::as_str))?;
        let model = if deployment.provider_slug() == "vertex_ai" {
            model
                .rsplit('/')
                .next()
                .unwrap_or(&model)
                .split('@')
                .next()
                .unwrap_or(&model)
        } else {
            &model
        };
        let normalized = oxllm_providers::anthropic::count_tokens::normalize_request(&input, model)
            .map_err(ApiError::from)?;
        Bytes::from(normalized.to_string())
    } else {
        body_to_send
    };
    let reservation = match crate::orchestration::budget::reserve_router_budgets(
        state,
        &deployment,
        &dep_key,
        &pipeline.call_ctx,
    )
    .await
    {
        Ok(reservation) => reservation,
        Err(error) => {
            pipeline
                .post_failure(&crate::pipeline::core_error_from_api(&error.api_error))
                .await;
            return Err(error.api_error);
        }
    };
    let send_result = send_upstream_tracked(
        state,
        &deployment,
        req.method,
        &url,
        body_to_send,
        req.content_type,
        &req.forward_headers,
        Some(&mut pipeline.call_ctx),
    )
    .await;

    let (status, mut resp, usage, upstream_cost) = match send_result {
        Ok(tuple) => {
            pipeline.call_ctx.reached_upstream = true;
            tuple
        }
        Err(api_err) => {
            let core_err = if api_err.status == axum::http::StatusCode::GATEWAY_TIMEOUT {
                oxllm_core::Error::Network("upstream timeout".to_string())
            } else {
                oxllm_core::Error::Network(api_err.message.clone())
            };
            reservation.release(&state.cache).await;
            pipeline.post_failure(&core_err).await;
            return Err(api_err);
        }
    };

    if status.is_success() {
        // U3: when the upstream reported its own computed cost via the
        // recognized gateway response-cost header, that value is authoritative
        // — stash it so `post_success` records it instead
        // of the systematically-undercounted prompt-only estimate the local
        // price table would derive from header-only token attribution.
        pipeline.call_ctx.cost_override = upstream_cost;
        if response_is_sse(&resp) {
            stamp_request_id_header(&mut resp, pipeline.request_id());
            stamp_ratelimit_headers(&mut resp, state, &pipeline.call_ctx).await;
            return Ok(meter_native_stream(
                state,
                deployment,
                pipeline.call_ctx,
                resp,
                reservation,
            ));
        }
        // Native inference APIs report token/cache/reasoning usage in JSON
        // bodies, not just headers. Recover those counters without changing
        // the wire response. Counting replies carry no `usage` object, so
        // recovery is a no-op for them. SSE uses the native relay above.
        let response_body = if inference_path(upstream_path) || upstream_path.starts_with("threads")
        {
            recover_json_response(&mut resp).await
        } else {
            None
        };
        let usage = usage.or_else(|| {
            response_body
                .as_ref()
                .and_then(crate::orchestration::usage::extract_usage)
        });
        // OpenAI Assistant Threads runs (`POST /threads/{id}/runs`,
        // `POST /threads/runs`) are real LLM inference calls whose usage
        // is carried in the JSON response body (`usage.prompt_tokens` /
        // `completion_tokens` / `total_tokens`), not in headers. Without
        // recovery the runs route records zero spend and silently bypasses
        // key/team/model budgets and the deployment TPM cap. Recover usage
        // from the buffered JSON body for OpenAI threads paths only.
        // Streaming relays parse terminal native usage incrementally above.
        let cost = pipeline
            .post_success(
                response_body
                    .as_ref()
                    .and_then(|body| body.get("model"))
                    .and_then(Value::as_str)
                    .filter(|model| !model.trim().is_empty())
                    .unwrap_or(&resolved_model),
                usage.as_ref(),
                response_body.as_ref(),
                deployment.provider_slug(),
            )
            .await;
        crate::orchestration::budget::settle_router_spend(
            state,
            &deployment,
            &dep_key,
            cost,
            &reservation,
        )
        .await;
        apply_cost_header(&mut resp, cost);
        // U57: charge the deployment TPM window when the upstream
        // reported usage via headers. Header-reported usage is an
        // aggregate (total tokens only — no prompt/completion split),
        // so charge `total_tokens` rather than one of the zeroed
        // halves. Without this the per-deployment `tpm` cap was never
        // enforced for hooked passthrough routes that meter usage.
        if let Some(u) = usage.as_ref() {
            record_deployment_tpm(state, &dep_key, u.total_tokens).await;
        }
    } else {
        // Non-2xx upstream response: surface it as a failure so the
        // budget/rpm reservation made in pre_call/check_limits is
        // released instead of leaking until the window expires.
        reservation.release(&state.cache).await;
        pipeline
            .post_failure(&oxllm_core::Error::Upstream {
                status: status.as_u16(),
                message: "passthrough upstream error".to_string(),
            })
            .await;
    }
    // Stamp the standard observability headers on every non-cache-hit
    // passthrough response so clients can correlate the response with the
    // request id (H8) and read the live RPM/TPM counters for client-side
    // throttling, matching the chat / embeddings / catch-all paths that
    // already stamp them. The cache-hit branch stamps the request id itself
    // before returning above.
    stamp_request_id_header(&mut resp, pipeline.request_id());
    stamp_ratelimit_headers(&mut resp, state, &pipeline.call_ctx).await;
    Ok(resp)
}

async fn raw_forward(
    state: &AppState,
    auth: &AuthUser,
    upstream_path: &str,
    req: PassthroughRequest<'_>,
) -> Result<Response, ApiError> {
    validate_raw_model_carrier(upstream_path, req.content_type, req.model)?;
    let mut request_body: Value = if is_json_content_type(req.content_type) && !req.body.is_empty()
    {
        serde_json::from_slice(&req.body)
            .map_err(|e| ApiError::bad_request(format!("invalid JSON body: {e}")))?
    } else {
        Value::Null
    };
    if !request_body.is_null() {
        crate::security::check_json_depth(&request_body, crate::security::MAX_JSON_DEPTH)?;
        crate::orchestration::request_controls::validate(&request_body)?;
    }
    require_inference_model(upstream_path, req.model, &request_body)?;
    let resolved_model = resolve_model(state, req.model, &request_body, req.provider_hint)?;
    // The Raw tier does not run the EndpointPipeline, which means it skips
    // the model ACL check `for_request` performs. The reference enforces model
    // ACLs at the auth layer for every route; replicate that here so a key
    // restricted to `models: ["gpt-4o"]` cannot bypass the ACL by routing
    // through a Raw-mode endpoint.
    crate::pipeline::check_model_acl(
        auth,
        &resolved_model,
        auth.key_models.as_deref(),
        auth.team_models.as_deref(),
        auth.user_models.as_deref(),
    )?;
    validate_nested_model_acl(auth, &request_body)?;
    let guarded = crate::pipeline::guardrails::apply_raw_request_guardrails(
        state,
        &req.method,
        upstream_path,
        &mut request_body,
    )
    .await?;
    let tags = request_tags(&req, &request_body);
    validate_file_namespace(state, upstream_path, &resolved_model, req.provider_hint)?;
    let deployment = resolve_deployment_with_tags(
        state,
        &resolved_model,
        request_session_id(&req, &request_body),
        &tags,
    )?;
    validate_raw_endpoint(state, &deployment, upstream_path, req.provider_hint)?;
    validate_selected_file_deployment(state, upstream_path, &resolved_model, &deployment)?;
    let native_selector = req
        .model
        .or_else(|| request_body.get("model").and_then(Value::as_str));
    let url = build_native_upstream_url(&deployment, upstream_path, req.query, native_selector)?;

    let body = if guarded {
        Bytes::from(request_body.to_string())
    } else {
        req.body
    };
    let body = if is_json_content_type(req.content_type) {
        rewrite_model_in_body(body, &deployment, inference_path(upstream_path))?
    } else {
        body
    };

    // Raw mode bypasses pipeline hooks, so upstream usage headers are
    // intentionally discarded — no post_success call to record spend.
    let (_status, mut resp, _usage, _cost) = send_upstream(
        state,
        &deployment,
        req.method,
        &url,
        body,
        req.content_type,
        &req.forward_headers,
    )
    .await?;
    // H8: stamp the request id on the response so clients can correlate
    // raw-mode responses with logs/traces. Raw mode has no pipeline-owned
    // request_id, so derive one per call (matching the per-call id the
    // pipeline would otherwise stamp on `for_request`).
    let request_id = crate::pipeline::request_id();
    stamp_request_id_header(&mut resp, &request_id);
    Ok(resp)
}

fn request_tags(req: &PassthroughRequest<'_>, body: &Value) -> Vec<String> {
    crate::pipeline::merge_request_tags(
        Some(body),
        req.tags
            .iter()
            .map(String::as_str)
            .chain(req.tag.iter().flat_map(|tag| tag.split(','))),
    )
}

fn request_session_id(req: &PassthroughRequest<'_>, body: &Value) -> Option<String> {
    req.session_id
        .as_deref()
        .and_then(crate::pipeline::normalize_session_id)
        .or_else(|| {
            crate::pipeline::extract_session_id_with_body(&axum::http::HeaderMap::new(), Some(body))
        })
}

pub(crate) fn endpoint_path(path: &str) -> &str {
    let path = path.trim_matches('/');
    path.strip_prefix("v1/").unwrap_or(path)
}

/// Multipart inference cannot be routed without a model selector. Reject it
/// explicitly instead of silently choosing another deployment/credential.
/// The dedicated handlers extract the `model` form field first, so a
/// well-formed SDK request (which carries the field) passes; file uploads
/// have no inference model field and keep their byte-preserving path, with
/// explicit namespace validation below.
pub(crate) fn validate_raw_model_carrier(
    path: &str,
    content_type: Option<&str>,
    model: Option<&str>,
) -> Result<(), ApiError> {
    if matches!(
        endpoint_path(path),
        "images/edits" | "images/variations" | "audio/transcriptions" | "audio/translations"
    ) && !is_json_content_type(content_type)
        && model.is_none()
    {
        return Err(ApiError::bad_request(
            "multipart inference requires a 'model' field in the form body; \
             the gateway will not guess a deployment",
        ));
    }
    Ok(())
}

pub(crate) fn require_inference_model(
    path: &str,
    model: Option<&str>,
    body: &Value,
) -> Result<(), ApiError> {
    if !inference_path(path) {
        return Ok(());
    }
    // Multipart inference carries its model in a form field, not in a parsed
    // JSON object: a caller-supplied model (extracted by the handler) is
    // sufficient evidence of an inference request shape.
    if !body.is_object() && model.is_none() {
        return Err(ApiError::bad_request(
            "inference request body must be a JSON object",
        ));
    }
    if model
        .or_else(|| body.get("model").and_then(Value::as_str))
        .is_none_or(|model| model.trim().is_empty())
    {
        return Err(ApiError::bad_request("missing required field: 'model'"));
    }
    Ok(())
}

pub(crate) fn inference_path(path: &str) -> bool {
    matches!(
        endpoint_path(path),
        "chat/completions"
            | "completions"
            | "complete"
            | "messages"
            | "messages/count_tokens"
            | "embeddings"
            | "images/embeddings"
            | "responses"
            | "rerank"
            | "images/generations"
            | "images/edits"
            | "images/variations"
            | "audio/speech"
            | "audio/transcriptions"
            | "audio/translations"
            | "moderations"
    )
}

pub(crate) fn validate_raw_endpoint(
    state: &AppState,
    deployment: &Deployment,
    path: &str,
    provider_hint: Option<&str>,
) -> Result<(), ApiError> {
    let path = endpoint_path(path);
    let slug = deployment.provider_slug();
    let partner_count = path == "messages/count_tokens"
        && (slug == "vertex_ai"
            || slug == "bedrock"
            || (slug == "azure_ai"
                && oxllm_providers::azure_ai::AzureAiProvider::is_anthropic(deployment)));
    if (path == "messages" || path.starts_with("messages/"))
        && slug != "anthropic"
        && !partner_count
    {
        return Err(ApiError::bad_request(
            "native Messages passthrough requires an Anthropic deployment; use /v1/messages for translation",
        ));
    }
    if (path == "files" || path.starts_with("files/"))
        && provider_hint.is_some_and(|expected| slug != expected)
    {
        return Err(ApiError::bad_request(
            "model does not belong to this Files API provider namespace",
        ));
    }
    if path.starts_with("audio/") && matches!(slug, "deepgram" | "elevenlabs") {
        return Err(ApiError::bad_request(format!(
            "provider '{slug}' uses a native audio format not supported by /v1/audio; use its provider-prefixed native endpoint",
        )));
    }
    let endpoint = match path {
        "images/generations" | "images/edits" | "images/variations" => {
            oxllm_core::Endpoint::ImagesGenerations
        }
        "audio/speech" => oxllm_core::Endpoint::AudioSpeech,
        "audio/transcriptions" | "audio/translations" => oxllm_core::Endpoint::AudioTranscriptions,
        "moderations" => oxllm_core::Endpoint::Moderations,
        _ => return Ok(()),
    };
    if state
        .registry
        .resolve_for_deployment(deployment)
        .is_none_or(|p| !p.supports(endpoint))
    {
        return Err(ApiError::bad_request(format!(
            "provider '{slug}' does not support this endpoint"
        )));
    }
    Ok(())
}

pub(crate) fn validate_file_namespace(
    state: &AppState,
    path: &str,
    model: &str,
    provider: Option<&str>,
) -> Result<(), ApiError> {
    let path = endpoint_path(path);
    if path != "files" && !path.starts_with("files/") {
        return Ok(());
    }
    let candidates = state
        .router
        .deployments_for_model(&state.deployments, model);
    if candidates.is_empty() {
        return Err(ApiError::model_not_found(model));
    }
    if candidates
        .iter()
        .any(|d| provider.is_some_and(|slug| d.provider_slug() != slug))
    {
        return Err(ApiError::bad_request(
            "model does not belong to this Files API provider namespace",
        ));
    }
    if candidates.len() != 1 {
        return Err(ApiError::bad_request(
            "ambiguous Files API namespace; specify a model alias resolving to a single deployment",
        ));
    }
    Ok(())
}

/// A strategy with cross-model expansion must not override a file's validated
/// account namespace after the model selector has identified one deployment.
pub(crate) fn validate_selected_file_deployment(
    state: &AppState,
    path: &str,
    model: &str,
    selected: &Deployment,
) -> Result<(), ApiError> {
    let path = endpoint_path(path);
    if path != "files" && !path.starts_with("files/") {
        return Ok(());
    }
    let candidates = state
        .router
        .deployments_for_model(&state.deployments, model);
    if candidates.len() != 1
        || state.router.cooldown_key(candidates[0]) != state.router.cooldown_key(selected)
    {
        return Err(ApiError::bad_request(
            "routing cannot change the selected Files API account namespace",
        ));
    }
    Ok(())
}

/// Resolve a provider-prefixed request without ever falling back to an
/// arbitrary credential. Native model IDs are accepted only when they identify
/// one configured model group for this provider; router failures stay failures.
pub(crate) fn resolve_provider_model(
    state: &AppState,
    provider: &str,
    model: Option<&str>,
) -> Result<String, ApiError> {
    if let Some(model) = model {
        if model.trim().is_empty() {
            return Err(ApiError::bad_request(
                "field 'model' must be a non-empty string",
            ));
        }
        let routed = state
            .router
            .deployments_for_model(&state.deployments, model);
        if !routed.is_empty() {
            if routed.iter().any(|d| d.provider_slug() != provider) {
                return Err(ApiError::bad_request(
                    "model does not belong exclusively to the requested provider",
                ));
            }
            return Ok(model.to_string());
        }
        let prefix = format!("{provider}/");
        let mut native = state.deployments.iter().filter(|d| {
            d.provider_slug() == provider
                && (d.oxllm_params.model == model
                    || d.oxllm_params.model.strip_prefix(&prefix) == Some(model))
        });
        let first = native
            .next()
            .ok_or_else(|| ApiError::model_not_found(model))?;
        if native.any(|d| d.model_name != first.model_name) {
            return Err(ApiError::bad_request(
                "ambiguous native model id; specify a configured model alias",
            ));
        }
        return Ok(first.model_name.clone());
    }
    resolve_model(state, None, &Value::Null, Some(provider))
}

pub(crate) async fn send_upstream(
    state: &AppState,
    deployment: &Deployment,
    method: Method,
    url: &str,
    body: Bytes,
    content_type: Option<&str>,
    extra_headers: &[(axum::http::HeaderName, axum::http::HeaderValue)],
) -> Result<
    (
        reqwest::StatusCode,
        Response,
        Option<UsageSummary>,
        Option<f64>,
    ),
    ApiError,
> {
    send_upstream_tracked(
        state,
        deployment,
        method,
        url,
        body,
        content_type,
        extra_headers,
        None,
    )
    .await
}

#[expect(clippy::too_many_arguments)]
pub(crate) async fn send_upstream_tracked(
    state: &AppState,
    deployment: &Deployment,
    method: Method,
    url: &str,
    body: Bytes,
    content_type: Option<&str>,
    extra_headers: &[(axum::http::HeaderName, axum::http::HeaderValue)],
    mut ctx: Option<&mut oxllm_core::CallContext>,
) -> Result<
    (
        reqwest::StatusCode,
        Response,
        Option<UsageSummary>,
        Option<f64>,
    ),
    ApiError,
> {
    let params = &deployment.oxllm_params;
    let control_request = ctx
        .as_ref()
        .map(|ctx| ctx.request_body.clone())
        .unwrap_or_else(|| {
            std::sync::Arc::new(if is_json_content_type(content_type) {
                serde_json::from_slice::<Value>(&body).unwrap_or(Value::Null)
            } else {
                Value::Null
            })
        });
    crate::orchestration::request_controls::validate(&control_request)?;
    let body = if !body.is_empty()
        && is_json_content_type(content_type)
        && (crate::orchestration::request_controls::contains(&control_request)
            || params
                .extra_body
                .as_ref()
                .is_some_and(|extra| !extra.is_empty())
            || params
                .additional_drop_params
                .as_ref()
                .is_some_and(|drops| !drops.is_empty()))
    {
        let mut value: Value = serde_json::from_slice(&body).map_err(|_| {
            ApiError::bad_request("invalid JSON body for configured request shaping")
        })?;
        // Generation requests on native relays need the same policy guard as
        // translated chat. Do not apply LLM-format rules to media/embeddings.
        let llm_body = value.get("messages").is_some()
            || ["tools", "functions", "toolConfig"]
                .iter()
                .any(|key| value.get(*key).is_some_and(|value| !value.is_null()))
            || [
                "/response_format/type",
                "/output_config/format/type",
                "/output_format/type",
                "/outputConfig/textFormat/type",
            ]
            .iter()
            .any(|path| {
                matches!(
                    value.pointer(path).and_then(Value::as_str),
                    Some("json_schema" | "json_object")
                )
            });
        let generation = method == Method::POST
            && url::Url::parse(url).is_ok_and(|url| {
                let operation = url
                    .path()
                    .trim_end_matches('/')
                    .rsplit('/')
                    .next()
                    .unwrap_or_default();
                matches!(
                    operation,
                    "messages" | "responses" | "completions" | "converse" | "converse-stream"
                ) || llm_body
                    && (matches!(operation, "invoke" | "invoke-with-response-stream")
                        || operation.ends_with(":rawPredict")
                        || operation.ends_with(":streamRawPredict"))
            });
        let output = if generation {
            crate::native_output_policy::validate_formats(&value)?;
            crate::native_output_policy::contract(&value)
        } else {
            None
        };
        crate::orchestration::upstream::shape_outbound_body(
            &mut value,
            params.additional_drop_params.as_deref(),
            params.extra_body.as_ref(),
        );
        crate::native_output_policy::ensure_preserved(&output, &value)?;
        Bytes::from(
            serde_json::to_vec(&value)
                .map_err(|_| ApiError::internal("failed to serialize configured request body"))?,
        )
    } else {
        body
    };
    let creds = resolve_credentials(state, deployment).await?;
    let url = crate::orchestration::upstream::upstream_url_with_resolved_endpoint(
        deployment, &creds, url,
    )?;
    let provider = state.registry.resolve_for_deployment(deployment);
    let mut auth_headers = provider_auth_headers(provider, deployment, &creds)?;
    crate::orchestration::upstream::merge_extra_headers(&mut auth_headers, deployment);
    for (name, value) in extra_headers {
        if name == "anthropic-beta" {
            let value = value
                .to_str()
                .map_err(|_| ApiError::bad_request("invalid anthropic-beta header"))?;
            crate::orchestration::upstream::merge_anthropic_beta_headers(
                &mut auth_headers,
                vec![value.into()],
            );
        }
    }
    if deployment.provider_slug() == "anthropic"
        && is_json_content_type(content_type)
        && let Ok(request) = serde_json::from_slice::<Value>(&body)
        && oxllm_providers::anthropic::uses_message_output_config(&request)
    {
        crate::orchestration::upstream::merge_anthropic_beta_headers(
            &mut auth_headers,
            vec![oxllm_providers::anthropic::BETA_MID_CONVERSATION_OUTPUT_CONFIG.into()],
        );
    }
    let mut headers = reqwest::header::HeaderMap::new();
    for (key, value) in &auth_headers {
        let name = reqwest::header::HeaderName::from_bytes(key.as_bytes())
            .map_err(|_| ApiError::internal("invalid auth header name"))?;
        let value = reqwest::header::HeaderValue::from_str(value)
            .map_err(|_| ApiError::internal("invalid API key characters"))?;
        headers.insert(name, value);
    }

    // Forward vetted client headers (e.g. `anthropic-beta` for the native
    // Anthropic passthrough). The caller is responsible for filtering —
    // `send_upstream` trusts the slice it receives. Auth-sensitive headers
    // (`authorization`, `host`, `content-length`, `transfer-encoding`) and
    // proxy-specific headers (`x-oxllm-*`, `x-forwarded-*`) must never be
    // added by the filtering helper.
    for (name, value) in extra_headers {
        if name == "anthropic-beta" {
            continue;
        }
        headers.insert(name.clone(), value.clone());
    }

    if let Some(ct) = content_type {
        if let Ok(ct_value) = reqwest::header::HeaderValue::from_str(ct) {
            headers.insert(reqwest::header::CONTENT_TYPE, ct_value);
        }
    }

    let timeout_secs = crate::orchestration::request_controls::seconds(&control_request, "timeout")
        .or(deployment.oxllm_params.timeout)
        .unwrap_or(DEFAULT_TIMEOUT_SECS as f64);

    let streaming = ctx.is_some()
        && (serde_json::from_slice::<Value>(&body)
            .ok()
            .and_then(|body| body.get("stream").and_then(Value::as_bool))
            .unwrap_or(false)
            || extra_headers.iter().any(|(name, value)| {
                name == axum::http::header::ACCEPT
                    && value
                        .to_str()
                        .is_ok_and(|value| value.contains("text/event-stream"))
            }));
    if let Some(ctx) = ctx.as_deref_mut() {
        ctx.upstream_start_time = Some(std::time::Instant::now());
    }
    // AWS SigV4 must sign the fully-materialized body (Bedrock's
    // apply_auth intentionally stays empty); mirror the chat pipeline's
    // signing pass so passthrough routes can reach SigV4 deployments at
    // all (the Bedrock CountTokens utility counter is the first user).
    let headers = if let oxllm_core::provider::ResolvedCredentials::AwsSigV4 {
        access_key,
        secret_key,
        session_token,
        region,
    } = &creds
    {
        let mut header_pairs: Vec<(String, String)> = headers
            .iter()
            .filter_map(|(name, value)| {
                value
                    .to_str()
                    .ok()
                    .map(|value| (name.as_str().to_ascii_lowercase(), value.to_string()))
            })
            .collect();
        crate::bedrock_sigv4::sign_request(
            method.as_str(),
            &url,
            &mut header_pairs,
            &body,
            access_key,
            secret_key,
            region,
            "bedrock",
            session_token.as_deref(),
        )
        .map_err(|e| {
            tracing::error!(
                error.kind = "config",
                detail = %e,
                "SigV4 request signing failed"
            );
            ApiError::internal("upstream request signing failed")
        })?;
        let mut signed = reqwest::header::HeaderMap::new();
        for (key, value) in header_pairs {
            let name = reqwest::header::HeaderName::from_bytes(key.as_bytes())
                .map_err(|_| ApiError::internal("invalid auth header name"))?;
            let value = reqwest::header::HeaderValue::from_str(&value)
                .map_err(|_| ApiError::internal("invalid API key characters"))?;
            signed.insert(name, value);
        }
        signed
    } else {
        headers
    };
    let send = crate::pipeline::execute_upstream_json(
        state,
        deployment,
        method,
        &url,
        body,
        headers,
        (!streaming).then_some(timeout_secs),
    );
    let response = if streaming {
        tokio::time::timeout(
            crate::orchestration::upstream::resolve_stream_idle_timeout_for_request(
                deployment,
                &state.config,
                &control_request,
            ),
            send,
        )
        .await
        .map_err(|_| ApiError::gateway_timeout())??
    } else {
        send.await?
    };
    if let Some(ctx) = ctx {
        ctx.reached_upstream = true;
    }

    let status = response.status();
    // U3: extract cost/usage headers BEFORE sanitizing (which consumes the
    // reqwest::Response) so passthrough routes can track spend when the
    // upstream reports total tokens, and honour the gateway response-cost
    // contract when the upstream reports its own computed cost.
    let usage = crate::pipeline::extract_upstream_usage(response.headers());
    let upstream_cost =
        crate::pipeline::extract_upstream_response_cost(response.headers()).into_override();
    let sanitized = crate::security::sanitize_upstream_response(response).await?;
    Ok((status, sanitized, usage, upstream_cost))
}

pub(crate) fn response_is_sse(response: &Response) -> bool {
    response
        .headers()
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| {
            value
                .split(';')
                .next()
                .is_some_and(|kind| kind.trim().eq_ignore_ascii_case("text/event-stream"))
        })
}

pub(crate) fn meter_native_stream(
    state: &AppState,
    deployment: Deployment,
    mut ctx: oxllm_core::CallContext,
    response: Response,
    reservation: crate::orchestration::budget::RouterBudgetReservation,
) -> Response {
    use crate::orchestration::stream::{StreamExit, finish_passthrough_stream};
    use futures_util::StreamExt;
    let state = state.clone();
    let idle = crate::orchestration::upstream::resolve_stream_idle_timeout_for_request(
        &deployment,
        &state.config,
        &ctx.request_body,
    );
    let (parts, body) = response.into_parts();
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Bytes, std::io::Error>>(16);
    state
        .router
        .record_in_flight(&state.router.cooldown_key(&deployment), 1);
    tokio::spawn(async move {
        let mut input = body.into_data_stream();
        let mut framer = crate::sse::SseFramer::new();
        let mut buffer = String::new();
        let mut usage = None;
        let mut output_usage_reported = false;
        let mut reported_cost = None;
        let mut model = ctx.model.clone();
        let mut bytes_sent = false;
        let mut completion = String::new();
        let original_request = ctx.request_body.clone();
        let mut provider_stream_state = None;
        let chat = state
            .registry
            .resolve_for_deployment(&deployment)
            .and_then(|provider| provider.chat());
        let exit = loop {
            let next = tokio::select! {
                _ = tx.closed() => break StreamExit::ClientDisconnected,
                next = tokio::time::timeout(idle, input.next()) => next,
            };
            match next {
                Ok(Some(Ok(bytes))) => {
                    buffer.push_str(&framer.push(&bytes));
                    let mut done = false;
                    let mut stream_error = None;
                    while let Some(event) = crate::sse::pop_sse_event(&mut buffer) {
                        let Some(data) = crate::sse::extract_sse_data(&event) else {
                            continue;
                        };
                        if data == "[DONE]" {
                            done = true;
                            continue;
                        }
                        let Ok(value) = serde_json::from_str::<Value>(data) else {
                            continue;
                        };
                        let payload = value
                            .get("message")
                            .or_else(|| value.get("response"))
                            .unwrap_or(&value);
                        if value.get("type").and_then(Value::as_str) != Some("message_start") {
                            output_usage_reported |=
                                ["completion_tokens", "output_tokens"].iter().any(|field| {
                                    payload
                                        .get("usage")
                                        .and_then(|usage| usage.get(field))
                                        .and_then(Value::as_u64)
                                        .is_some()
                                });
                        }
                        let normalized = chat.and_then(|provider| {
                            match provider.transform_stream_chunk_stateful(
                                &event,
                                &mut provider_stream_state,
                                &original_request,
                            ) {
                                Ok(chunk) => chunk,
                                Err(
                                    error @ (oxllm_core::Error::Upstream { .. }
                                    | oxllm_core::Error::Network(_)),
                                ) => {
                                    stream_error = Some(error);
                                    None
                                }
                                // Other native APIs have event shapes the chat
                                // adapter does not understand. Read their raw
                                // usage without treating that as a transport fault.
                                Err(_) => None,
                            }
                        });
                        if let Some(error) = value
                            .get("error")
                            .or_else(|| payload.get("error"))
                            .filter(|error| !error.is_null())
                        {
                            let status = if matches!(
                                error.get("code").and_then(Value::as_str),
                                Some("cyber_policy" | "invalid_request_error")
                            ) {
                                400
                            } else {
                                match error.get("type").and_then(Value::as_str) {
                                    Some("rate_limit_error") => 429,
                                    Some("overloaded_error") => 529,
                                    Some("authentication_error") => 401,
                                    Some("billing_error") => 402,
                                    Some("permission_error") => 403,
                                    Some("not_found_error") => 404,
                                    Some("conflict_error") => 409,
                                    Some("request_too_large") => 413,
                                    Some("timeout_error") => 504,
                                    Some("invalid_request_error" | "cyber_policy") => 400,
                                    _ => 500,
                                }
                            };
                            stream_error.get_or_insert(oxllm_core::Error::Upstream {
                                status,
                                message: "native upstream stream reported an error".into(),
                            });
                        }
                        let incoming =
                            crate::orchestration::usage::extract_usage(payload).or_else(|| {
                                normalized
                                    .as_ref()
                                    .and_then(crate::orchestration::usage::extract_usage)
                            });
                        if let Some(incoming) = incoming {
                            crate::orchestration::usage::merge_streaming_usage(
                                &mut usage, incoming,
                            );
                        }
                        if let Some(cost) = crate::pipeline::extract_body_usage_cost(payload) {
                            reported_cost = Some(cost);
                        }
                        if let Some(tier) = oxllm_core::response_service_tier(&value) {
                            ctx.served_service_tier = Some(tier.into());
                        }
                        if let Some(served) = payload.get("model").and_then(Value::as_str) {
                            model = served.into();
                        }
                        for pointer in [
                            "/delta",
                            "/delta/text",
                            "/delta/thinking",
                            "/delta/partial_json",
                            "/choices/0/delta/content",
                            "/choices/0/delta/reasoning_content",
                        ] {
                            if let Some(text) = value.pointer(pointer).and_then(Value::as_str) {
                                append_native_completion(&mut completion, text);
                            }
                        }
                        if let Some(choices) = value.get("choices").and_then(Value::as_array) {
                            for choice in choices {
                                if let Some(calls) = choice
                                    .pointer("/delta/tool_calls")
                                    .and_then(Value::as_array)
                                {
                                    for call in calls {
                                        if let Some(arguments) = call
                                            .pointer("/function/arguments")
                                            .and_then(Value::as_str)
                                        {
                                            append_native_completion(&mut completion, arguments);
                                        }
                                    }
                                }
                            }
                        }
                        done |= matches!(
                            value.get("type").and_then(Value::as_str),
                            Some("message_stop" | "response.completed")
                        );
                    }
                    if buffer.len() > crate::orchestration::MAX_STREAM_BUFFER_BYTES {
                        buffer.clear();
                    }
                    match tokio::time::timeout(
                        std::time::Duration::from_secs(60),
                        tx.send(Ok(bytes)),
                    )
                    .await
                    {
                        Ok(Ok(())) => bytes_sent = true,
                        _ => break StreamExit::ClientDisconnected,
                    }
                    if let Some(error) = stream_error {
                        break StreamExit::Error(error);
                    }
                    if done {
                        break StreamExit::Done;
                    }
                }
                Ok(None) => break StreamExit::Done,
                Ok(Some(Err(_))) => {
                    break StreamExit::Error(oxllm_core::Error::Network(
                        "native upstream stream failed".into(),
                    ));
                }
                Err(_) => {
                    break StreamExit::Error(oxllm_core::Error::Network(
                        "native upstream stream idle timeout".into(),
                    ));
                }
            }
        };
        if matches!(exit, StreamExit::Error(oxllm_core::Error::Network(_))) {
            let _ = tx.try_send(Err(std::io::Error::other("upstream stream failed")));
        }
        // message_start's zero output count is a prompt snapshot, not a final
        // output report. Preserve prompt/cache usage but estimate text already
        // generated when a native stream ends before its final usage frame.
        if !output_usage_reported && !completion.is_empty() {
            if let Some(usage) = usage.as_mut() {
                if usage.completion_tokens == 0 {
                    usage.completion_tokens = oxllm_core::estimate_tokens(&completion, &model);
                    usage.total_tokens =
                        usage.prompt_tokens.saturating_add(usage.completion_tokens);
                }
            }
        }
        finish_passthrough_stream(
            &state,
            &deployment,
            &ctx,
            &reservation,
            exit,
            &mut usage,
            &model,
            bytes_sent,
            &completion,
            reported_cost,
        )
        .await;
    });
    Response::from_parts(
        parts,
        axum::body::Body::from_stream(tokio_stream::wrappers::ReceiverStream::new(rx)),
    )
}

fn append_native_completion(completion: &mut String, text: &str) {
    let mut end = text
        .len()
        .min((4 * 1024 * 1024usize).saturating_sub(completion.len()));
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    completion.push_str(&text[..end]);
}

pub(crate) async fn recover_json_response(resp: &mut Response) -> Option<Value> {
    let content_type = resp
        .headers()
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    let json_body = content_type.is_empty()
        || content_type.contains("application/json")
        || content_type.contains("+json");
    if !json_body {
        return None;
    }
    let bytes = axum::body::to_bytes(
        std::mem::replace(resp.body_mut(), axum::body::Body::empty()),
        crate::security::MAX_RESPONSE_BODY_BYTES,
    )
    .await
    .ok()?;
    *resp.body_mut() = axum::body::Body::from(bytes.clone());
    serde_json::from_slice(&bytes).ok()
}

/// Rewrite the `model` field in a JSON body to the deployment's actual
/// upstream model name (stripped only of its actual provider prefix). Inference
/// inserts the model when a separate selector was used; platform bodies do not.
/// Gateway-only cache/tag fields are removed from JSON before forwarding.
pub(crate) fn rewrite_model_in_body(
    body: Bytes,
    deployment: &Deployment,
    require_model: bool,
) -> Result<Bytes, ApiError> {
    let mut value: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(_) => return Ok(body),
    };
    if value.get("model").is_some() || require_model {
        let model = value
            .get("model")
            .map(|model| {
                model
                    .as_str()
                    .filter(|model| !model.trim().is_empty())
                    .ok_or_else(|| {
                        ApiError::bad_request("field 'model' must be a non-empty string")
                    })
            })
            .transpose()?;
        let model = resolve_raw_model(deployment, model)?;
        value
            .as_object_mut()
            .ok_or_else(|| ApiError::bad_request("inference request body must be a JSON object"))?
            .insert("model".to_string(), Value::String(model));
    }
    // Native batches nest the public alias inside each request's parameters.
    // Other concrete native model IDs keep their identity within the account.
    if let Some(requests) = value.get_mut("requests").and_then(Value::as_array_mut) {
        for request in requests {
            if let Some(params) = request.get_mut("params").and_then(Value::as_object_mut) {
                if params.get("model").and_then(Value::as_str)
                    == Some(deployment.model_name.as_str())
                {
                    params.insert(
                        "model".into(),
                        json!(resolve_raw_model(deployment, Some(&deployment.model_name))?),
                    );
                }
            }
        }
    }
    // Gateway-only metadata must not make otherwise valid raw requests fail
    // provider validation (metadata typically accepts string values only).
    if let Some(object) = value.as_object_mut() {
        object.remove("cache");
        if let Some(metadata) = object.get_mut("metadata").and_then(Value::as_object_mut) {
            metadata.remove("tags");
        }
    }
    match serde_json::to_vec(&value) {
        Ok(bytes) => Ok(Bytes::from(bytes)),
        Err(_) => Err(ApiError::internal("failed to serialize passthrough body")),
    }
}

pub(crate) fn resolve_raw_model(
    deployment: &Deployment,
    requested: Option<&str>,
) -> Result<String, ApiError> {
    if deployment.provider_slug() == "azure" {
        return oxllm_providers::azure::AzureProvider::native_request_model(deployment, requested)
            .map_err(ApiError::from);
    }
    let prefix = format!("{}/", deployment.provider_slug());
    let upstream = deployment
        .oxllm_params
        .model
        .strip_prefix(&prefix)
        .unwrap_or(&deployment.oxllm_params.model);
    if upstream == "*" {
        let model = requested
            .filter(|model| {
                !model.trim().is_empty() && *model != deployment.model_name && !model.contains('*')
            })
            .ok_or_else(|| {
                ApiError::bad_request("wildcard deployment requires a concrete upstream model id")
            })?;
        Ok(model.strip_prefix(&prefix).unwrap_or(model).to_string())
    } else if upstream.contains('*') {
        Err(ApiError::bad_request(
            "raw passthrough does not support this deployment's model template",
        ))
    } else {
        Ok(upstream.to_string())
    }
}

pub(crate) fn validate_nested_model_acl(auth: &AuthUser, body: &Value) -> Result<(), ApiError> {
    if let Some(requests) = body.get("requests").and_then(Value::as_array) {
        for request in requests {
            if let Some(model) = request.pointer("/params/model").and_then(Value::as_str) {
                crate::pipeline::check_model_acl(
                    auth,
                    model,
                    auth.key_models.as_deref(),
                    auth.team_models.as_deref(),
                    auth.user_models.as_deref(),
                )?;
            }
        }
    }
    Ok(())
}

fn resolve_model(
    state: &AppState,
    model: Option<&str>,
    request_body: &Value,
    provider_hint: Option<&str>,
) -> Result<String, ApiError> {
    if let Some(m) = model {
        if m.trim().is_empty() {
            return Err(ApiError::bad_request(
                "field 'model' must be a non-empty string",
            ));
        }
        if request_body
            .get("model")
            .is_some_and(|value| value.as_str() != Some(m))
        {
            return Err(ApiError::bad_request(
                "conflicting body and explicit model selectors",
            ));
        }
        return Ok(m.to_string());
    }
    if let Some(value) = request_body.get("model") {
        let m = value
            .as_str()
            .filter(|m| !m.trim().is_empty())
            .ok_or_else(|| ApiError::bad_request("field 'model' must be a non-empty string"))?;
        return Ok(m.to_string());
    }
    // Fallback for bodies that nest the model under a request array (notably
    // Anthropic Message Batches: `requests[0].params.model`). The reference router
    // uses the same nested extraction. This avoids a double JSON parse in the
    // handler — the body is parsed once here in `hooked_raw` and the model is
    // resolved from the same `Value`.
    if let Some(m) = request_body
        .get("requests")
        .and_then(|r| r.get(0))
        .and_then(|r| r.get("params"))
        .and_then(|p| p.get("model"))
        .and_then(|m| m.as_str())
    {
        return Ok(m.to_string());
    }
    // Model-less platform requests must identify one namespace. Never fall
    // back to a different provider or randomly choose an account by config order.
    let mut candidates = state
        .deployments
        .iter()
        .filter(|d| provider_hint.is_none_or(|provider| d.provider_slug() == provider));
    let deployment = candidates.next().ok_or_else(|| {
        ApiError::bad_request("no deployment configured for this endpoint's provider")
    })?;
    if candidates.next().is_some() {
        return Err(ApiError::bad_request(
            "ambiguous deployment; specify a model alias",
        ));
    }
    Ok(deployment.model_name.clone())
}

pub(crate) fn resolve_deployment_with_tags(
    state: &AppState,
    model: &str,
    session_id: Option<String>,
    tags: &[String],
) -> Result<Deployment, ApiError> {
    let mut ctx = oxllm_core::RequestContext::new(model);
    ctx.session_id = session_id;
    ctx.tag = tags.first().cloned();
    ctx.tags = tags.to_vec();
    crate::pipeline::routing::apply_default_tag(state, &mut ctx);
    let idx = crate::pipeline::routing::try_pick_deployment_index(state, &ctx)?
        .ok_or_else(|| crate::pipeline::routing::no_available_deployment(state, model))?;
    Ok(state.deployments[idx].clone())
}

fn build_upstream_url(
    deployment: &Deployment,
    upstream_path: &str,
    query: Option<&str>,
) -> Result<String, ApiError> {
    let base = upstream_base_url(deployment)?;
    let merged = merge_base_path_query(&base, upstream_path, query)?;
    confine_to_base_path(&base, &merged)?;
    Ok(merged)
}

/// Azure inference routes require provider-owned deployment/version URLs.
/// Other raw APIs retain the configured-base confinement contract.
pub(crate) fn build_native_upstream_url(
    deployment: &Deployment,
    upstream_path: &str,
    query: Option<&str>,
    requested_model: Option<&str>,
) -> Result<String, ApiError> {
    let endpoint = endpoint_path(upstream_path);
    if endpoint == "messages/count_tokens" {
        let url = match deployment.provider_slug() {
            "anthropic" => {
                let mut url = url::Url::parse(
                    deployment
                        .oxllm_params
                        .api_base
                        .as_deref()
                        .unwrap_or("https://api.anthropic.com/v1"),
                )
                .map_err(|_| ApiError::internal("invalid Anthropic API base"))?;
                let path = url.path().trim_end_matches('/');
                let path = if path.ends_with("/messages/count_tokens") {
                    path.to_string()
                } else if path.ends_with("/messages") {
                    format!("{path}/count_tokens")
                } else {
                    format!("{path}/messages/count_tokens")
                };
                url.set_path(&path);
                Some(url.to_string())
            }
            "vertex_ai" => Some(
                oxllm_providers::vertex_ai::VertexProvider::count_tokens_url(
                    deployment,
                    requested_model,
                )
                .map_err(ApiError::from)?,
            ),
            "azure_ai" => Some(
                oxllm_providers::azure_ai::AzureAiProvider::count_tokens_url(deployment)
                    .map_err(ApiError::from)?,
            ),
            "bedrock" => Some(
                oxllm_providers::bedrock::count_tokens::count_tokens_url(
                    deployment,
                    requested_model,
                )
                .map_err(ApiError::from)?,
            ),
            _ => None,
        };
        if let Some(url) = url {
            return merge_base_path_query(&url, "", query);
        }
    }
    if deployment.provider_slug() == "azure"
        && matches!(
            endpoint,
            "images/generations"
                | "images/edits"
                | "images/variations"
                | "audio/speech"
                | "audio/transcriptions"
                | "audio/translations"
                | "moderations"
        )
    {
        let endpoint = oxllm_providers::azure::AzureProvider::native_endpoint_url_for_model(
            deployment,
            requested_model,
            endpoint,
        )
        .map_err(ApiError::from)?;
        // The endpoint is constructed solely from configured values and a
        // validated operation; inbound query pairs cannot replace its version.
        return merge_base_path_query(&endpoint, "", query);
    }
    build_upstream_url(deployment, upstream_path, query)
}

/// Defense-in-depth chokepoint: after `merge_base_path_query` normalises the
/// joined URL (collapsing any `.`/`..` segments per RFC 3986), reject a final
/// path that escapes the deployment's configured base path. This catches a
/// traversal payload that slipped past a handler's per-segment allowlist (or
/// arrived via a route that has none) so the pooled credential cannot reach
/// arbitrary endpoints on the upstream host. Mirrors the confinement the
/// catch-all `provider_passthrough` applies via `confine_upstream_url`.
fn confine_to_base_path(base: &str, merged: &str) -> Result<(), ApiError> {
    let base_path = url::Url::parse(base)
        .map_err(|_| ApiError::internal("invalid upstream base URL"))?
        .path()
        .to_string();
    let final_path = url::Url::parse(merged)
        .map_err(|e| ApiError::bad_request(format!("invalid merged upstream URL: {e}")))?
        .path()
        .to_string();
    if path_within_prefix(&final_path, &base_path) {
        Ok(())
    } else {
        Err(ApiError::forbidden(
            "upstream path escapes the configured base prefix",
        ))
    }
}

/// Boundary-aware prefix check: `final_path` must equal `target_path` or begin
/// with `target_path` followed by `/`. An empty or `"/"` target path accepts
/// anything (no prefix to escape). Shared by the hooked-raw path confinement,
/// the catch-all handler, and the `pass_through_endpoints` forwarder.
pub(crate) fn path_within_prefix(final_path: &str, target_path: &str) -> bool {
    if target_path.is_empty() || target_path == "/" {
        return true;
    }
    if final_path == target_path {
        return true;
    }
    let trimmed = target_path.trim_end_matches('/');
    final_path == trimmed || final_path.starts_with(&format!("{trimmed}/"))
}

/// URL-aware join of a base URL, path, and inbound query string. A configured
/// `api_base` may already carry query parameters (notably Azure's
/// `?api-version=...`); string-concatenating a path after the `?` or blindly
/// appending another `?` produces malformed URLs. Client query pairs are
/// appended after configured pairs and are percent-decoded/re-encoded by the
/// URL parser.
///
/// Bug #43: when an inbound query key is already present in the base URL's
/// query (e.g. base `?api-version=...` and inbound `?api-version=...`), the
/// base URL's value takes precedence and the inbound duplicate is skipped —
/// otherwise the upstream would see two values for the same key and behave
/// unpredictably.
///
/// The `model` query parameter is an oxllm routing input (used by the
/// router to select a deployment), not part of any provider's API contract.
/// It is stripped here so the proxy's internal model alias is never leaked
/// to the upstream, and a future upstream that attaches semantics to
/// `?model=` is not silently affected.
pub(crate) fn merge_base_path_query(
    base: &str,
    path: &str,
    query: Option<&str>,
) -> Result<String, ApiError> {
    let mut url =
        url::Url::parse(base).map_err(|_| ApiError::internal("invalid upstream base URL"))?;
    if !path.trim_matches('/').is_empty() {
        let base_path = url.path().trim_end_matches('/');
        url.set_path(&format!("{}/{}", base_path, path.trim_matches('/')));
    }
    if let Some(q) = query.filter(|q| !q.is_empty()) {
        let inbound: Vec<(String, String)> = url::form_urlencoded::parse(q.as_bytes())
            .into_owned()
            .collect();
        // Snapshot the existing base query keys so we can skip inbound
        // duplicates (base URL wins). Collected as owned `String`s so the
        // borrow on `url` ends before we mutably borrow it via
        // `query_pairs_mut()` (Bug #43).
        let existing_keys: std::collections::HashSet<String> =
            url.query_pairs().map(|(k, _)| k.into_owned()).collect();
        let mut pairs = url.query_pairs_mut();
        for (key, value) in inbound {
            if existing_keys.contains(&key) {
                continue;
            }
            // The `model` key is an oxllm routing input, not a provider API
            // parameter — never forward it upstream.
            if key == "model" {
                continue;
            }
            pairs.append_pair(&key, &value);
        }
    }
    Ok(url.to_string())
}

/// Derive a provider slug from a cached response's `model` string when it
/// carries a `provider/` prefix, falling back to the `"cache"` sentinel when
/// no prefix is present. Used on the cache short-circuit in `hooked_raw`
/// (before any deployment is picked) so the post-success chain attributes
/// spend to the serving provider when known — matching the chat
/// orchestrator's `provider_prefix` — while still recording a self-documenting
/// non-empty label (LOW-2) for bare model names.
fn cached_provider_prefix(model: &str) -> &str {
    match model.split_once('/') {
        Some((provider, _)) if !provider.is_empty() => provider,
        _ => "cache",
    }
}

/// Whether the request content-type should be treated as a single JSON
/// document for body parsing.
///
/// `application/x-ndjson` is deliberately EXCLUDED: JSONL is a stream of
/// newline-delimited JSON documents, which `serde_json::from_slice` cannot
/// parse — treating it as JSON made `hooked_raw` / `raw_forward` /
/// `provider_passthrough` reject a valid JSONL body with a spurious
/// `400 invalid JSON body` instead of forwarding it verbatim. NDJSON bodies
/// fall through to the non-JSON path: forwarded as-is, with model resolution
/// from the explicit model hint / provider default.
pub(crate) fn is_json_content_type(content_type: Option<&str>) -> bool {
    content_type
        .map(|ct| {
            let media_type = ct
                .split(';')
                .next()
                .unwrap_or(ct)
                .trim()
                .to_ascii_lowercase();
            media_type == "application/json" || media_type.ends_with("+json")
        })
        .unwrap_or(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::types::{GeneralSettings, OxllmSettings, ProxyConfig};
    use serde_json::json;

    #[test]
    fn nested_batch_aliases_are_rewritten_and_every_model_is_acl_checked() {
        let deployment = Deployment {
            model_name: "allowed".into(),
            oxllm_params: oxllm_core::deployment::OxllmParams {
                model: "anthropic/claude-sonnet-4-6".into(),
                ..Default::default()
            },
            ..Default::default()
        };
        let body = json!({"requests":[
            {"custom_id":"first","params":{"model":"allowed","messages":[]}},
            {"custom_id":"second","params":{"model":"blocked","messages":[]}}
        ]});
        let rewritten =
            rewrite_model_in_body(Bytes::from(body.to_string()), &deployment, false).unwrap();
        let value: Value = serde_json::from_slice(&rewritten).unwrap();
        assert_eq!(value["requests"][0]["params"]["model"], "claude-sonnet-4-6");
        assert_eq!(value["requests"][1]["params"]["model"], "blocked");
        let mut auth = AuthUser::open();
        auth.key_models = Some(vec!["allowed".into()]);
        assert!(validate_nested_model_acl(&auth, &body).is_err());
        auth.key_models = Some(vec!["*".into()]);
        assert!(validate_nested_model_acl(&auth, &body).is_ok());
    }

    #[test]
    fn wildcard_multipart_selectors_resolve_to_concrete_native_ids() {
        let deployment = Deployment {
            model_name: "*".into(),
            oxllm_params: oxllm_core::deployment::OxllmParams {
                model: "openai/*".into(),
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(
            resolve_raw_model(&deployment, Some("openai/whisper-1")).unwrap(),
            "whisper-1"
        );
        assert!(resolve_raw_model(&deployment, Some("*")).is_err());
        assert!(resolve_raw_model(&deployment, None).is_err());
    }

    #[tokio::test]
    async fn api_handlers_merge_body_and_repeated_header_tags() {
        use async_trait::async_trait;
        use oxllm_core::{CallContext, Hook, HookAction};
        use std::sync::{Arc, Mutex};
        use tower::ServiceExt;

        struct TagCapture(Arc<Mutex<Vec<Vec<String>>>>);
        #[async_trait]
        impl Hook for TagCapture {
            async fn pre_call(&self, ctx: &mut CallContext) -> oxllm_core::Result<HookAction> {
                self.0.lock().unwrap().push(ctx.request_tags.clone());
                Ok(HookAction::Continue)
            }
        }

        let mock = httpmock::MockServer::start();
        let upstream = mock.mock(|when, then| {
            when.method("POST");
            then.status(200).header("content-type", "application/json").json_body(json!({
                "id": "response-1", "object": "chat.completion", "model": "gpt-4o",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "data": [{"index": 0, "object": "embedding", "embedding": [0.1]}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            }));
        });
        let mut config = base_config();
        config.general_settings.allow_private_api_base = Some(true);
        config.general_settings.master_key = Some("sk-tag-test-master-key-00000000000".to_string());
        config.model_list = vec![crate::config::types::ConfigModel {
            model_name: "public-model".to_string(),
            oxllm_params: oxllm_core::deployment::OxllmParams {
                model: "openai/gpt-4o".to_string(),
                api_key: Some("upstream-key".to_string()),
                api_base: Some(mock.base_url()),
                ..Default::default()
            },
            ..Default::default()
        }];
        let mut state =
            AppState::with_registry(config, oxllm_providers::default_registry()).unwrap();
        let captured = Arc::new(Mutex::new(Vec::new()));
        state.hooks = vec![Arc::new(TagCapture(captured.clone()))];
        let shared = Arc::new(arc_swap::ArcSwap::from_pointee(state));
        let app = crate::app::build_router(shared);
        let paths = [
            "/v1/chat/completions",
            "/v1/completions",
            "/v1/embeddings",
            "/v1/messages",
            "/v1/responses",
            "/v1/rerank",
            "/v1/images/generations",
            "/v1/audio/speech",
            "/v1/moderations",
            "/openai/v1/chat/completions",
        ];
        for path in paths {
            let request = axum::http::Request::builder()
                .method("POST")
                .uri(path)
                .header("authorization", "Bearer sk-tag-test-master-key-00000000000")
                .header("content-type", "application/json")
                .header(crate::pipeline::TAG_HEADER, "header-a, shared")
                .header(crate::pipeline::TAG_HEADER, "header-b")
                .body(axum::body::Body::from(
                    json!({
                        "model": "public-model", "messages": [{"role": "user", "content": "hi"}],
                        "prompt": "hi", "input": "hi", "query": "hi", "documents": ["hi"],
                        "max_tokens": 10, "metadata": {"tags": ["body", "shared"]}
                    })
                    .to_string(),
                ))
                .unwrap();
            let response = app.clone().oneshot(request).await.unwrap();
            assert_eq!(response.status(), axum::http::StatusCode::OK, "{path}");
            let _ = axum::body::to_bytes(response.into_body(), 1024 * 1024)
                .await
                .unwrap();
        }
        let captured = captured.lock().unwrap();
        assert_eq!(captured.len(), paths.len());
        for (path, tags) in paths.iter().zip(captured.iter()) {
            assert_eq!(tags, &["body", "shared", "header-a", "header-b"], "{path}");
        }
        upstream.assert_hits(paths.len());
    }

    #[test]
    fn relay_model_rewrites_only_the_actual_provider_prefix() {
        let mut deployment = Deployment {
            model_name: "alias".to_string(),
            oxllm_params: oxllm_core::deployment::OxllmParams {
                model: "openrouter/vendor/model".to_string(),
                ..Default::default()
            },
            ..Default::default()
        };
        let original = Bytes::from_static(
            br#"{"model":"alias","metadata":{"tags":["app"],"user_id":"keep"}}"#,
        );
        let rewritten: Value =
            serde_json::from_slice(&rewrite_model_in_body(original, &deployment, true).unwrap())
                .unwrap();
        assert_eq!(rewritten["model"], "vendor/model");
        assert_eq!(rewritten["metadata"]["user_id"], "keep");
        assert!(rewritten["metadata"].get("tags").is_none());
        let inferred: Value = serde_json::from_slice(
            &rewrite_model_in_body(Bytes::from_static(b"{}"), &deployment, true).unwrap(),
        )
        .unwrap();
        assert_eq!(inferred["model"], "vendor/model");
        let platform: Value = serde_json::from_slice(
            &rewrite_model_in_body(Bytes::from_static(b"{}"), &deployment, false).unwrap(),
        )
        .unwrap();
        assert!(platform.get("model").is_none());

        deployment.oxllm_params.model = "openrouter/*".to_string();
        let body = Bytes::from_static(br#"{"model":"openrouter/vendor/model"}"#);
        let rewritten: Value =
            serde_json::from_slice(&rewrite_model_in_body(body, &deployment, true).unwrap())
                .unwrap();
        assert_eq!(rewritten["model"], "vendor/model");
        assert!(
            rewrite_model_in_body(
                Bytes::from_static(br#"{"model":"alias"}"#),
                &deployment,
                true
            )
            .is_err()
        );
    }

    #[test]
    fn raw_inference_rejects_missing_model_and_unparsed_multipart() {
        for path in [
            "images/generations",
            "audio/speech",
            "audio/transcriptions",
            "images/edits",
        ] {
            assert!(
                require_inference_model(path, None, &json!({})).is_err(),
                "{path}"
            );
        }
        for path in [
            "images/edits",
            "images/variations",
            "audio/transcriptions",
            "audio/translations",
        ] {
            let error =
                validate_raw_model_carrier(path, Some("multipart/form-data; boundary=x"), None)
                    .unwrap_err();
            assert_eq!(error.status, axum::http::StatusCode::BAD_REQUEST);
            assert!(error.message.contains("multipart"));
            // A body-carried model lets the request route (and be rewritten).
            assert!(
                validate_raw_model_carrier(
                    path,
                    Some("multipart/form-data; boundary=x"),
                    Some("my-alias"),
                )
                .is_ok(),
                "{path}"
            );
        }
        assert!(
            validate_raw_model_carrier("files", Some("multipart/form-data; boundary=x"), None)
                .is_ok()
        );
    }

    /// The raw-endpoint capability gate consults the resolved provider's
    /// declared endpoint support: OpenAI (which declares images/audio)
    /// passes, while a chat-only provider on a raw audio path is rejected
    /// with a message naming the provider.
    #[test]
    fn raw_endpoint_gate_allows_declared_capabilities_and_rejects_chat_only_providers() {
        let state = AppState::with_registry(base_config(), oxllm_providers::default_registry())
            .expect("state built");
        let openai = Deployment {
            model_name: "gpt-4o".to_string(),
            oxllm_params: oxllm_core::deployment::OxllmParams {
                model: "openai/gpt-4o".to_string(),
                api_key: Some("sk-test".to_string()),
                ..Default::default()
            },
            ..Default::default()
        };
        for path in [
            "images/generations",
            "images/edits",
            "audio/speech",
            "audio/transcriptions",
            "moderations",
        ] {
            assert!(
                validate_raw_endpoint(&state, &openai, path, None).is_ok(),
                "openai declares {path}"
            );
        }

        let anthropic = Deployment {
            model_name: "claude".to_string(),
            oxllm_params: oxllm_core::deployment::OxllmParams {
                model: "anthropic/claude-3-5-sonnet".to_string(),
                api_key: Some("sk-test".to_string()),
                ..Default::default()
            },
            ..Default::default()
        };
        let error = validate_raw_endpoint(&state, &anthropic, "audio/speech", None)
            .expect_err("a chat-only provider must not serve a raw audio endpoint");
        assert_eq!(error.status, axum::http::StatusCode::BAD_REQUEST);
        assert!(
            error.message.contains("anthropic") && error.message.contains("does not support"),
            "the rejection must name the provider and the unsupported endpoint, got: {}",
            error.message
        );
    }

    fn base_config() -> ProxyConfig {
        ProxyConfig {
            model_list: vec![],
            oxllm_settings: OxllmSettings::default(),
            general_settings: GeneralSettings::default(),
            router_settings: Default::default(),
            extra: serde_json::Map::new(),
        }
    }

    #[test]
    fn default_base_for_known_providers() {
        assert_eq!(
            default_base_for_provider("openai"),
            Some("https://api.openai.com/v1")
        );
        assert_eq!(
            default_base_for_provider("anthropic"),
            Some("https://api.anthropic.com/v1")
        );
        assert_eq!(
            default_base_for_provider("cohere"),
            Some("https://api.cohere.com/v2")
        );
        assert_eq!(
            default_base_for_provider("groq"),
            Some("https://api.groq.com/openai/v1")
        );
    }

    #[test]
    fn default_base_for_unknown_provider_is_none() {
        assert!(default_base_for_provider("made-up").is_none());
    }

    #[test]
    fn upstream_base_url_uses_configured_api_base() {
        use oxllm_core::deployment::OxllmParams;
        let deployment = Deployment {
            model_name: "gpt-4o".to_string(),
            oxllm_params: OxllmParams {
                model: "openai/gpt-4o".to_string(),
                api_key: Some("k".to_string()),
                api_base: Some("https://example.test/v1/".to_string()),
                ..Default::default()
            },
            ..Default::default()
        };
        let url = upstream_base_url(&deployment).expect("configured base");
        assert_eq!(url, "https://example.test/v1");
    }

    #[test]
    fn upstream_base_url_falls_back_to_provider_default() {
        use oxllm_core::deployment::OxllmParams;
        let deployment = Deployment {
            model_name: "gpt-4o".to_string(),
            oxllm_params: OxllmParams {
                model: "openai/gpt-4o".to_string(),
                api_key: Some("k".to_string()),
                ..Default::default()
            },
            ..Default::default()
        };
        let url = upstream_base_url(&deployment).expect("provider default");
        assert_eq!(url, "https://api.openai.com/v1");
    }

    #[test]
    fn upstream_base_url_errors_for_unknown_provider_without_base() {
        use oxllm_core::deployment::OxllmParams;
        let deployment = Deployment {
            model_name: "weird".to_string(),
            oxllm_params: OxllmParams {
                model: "madeup/weird-1".to_string(),
                api_key: Some("k".to_string()),
                ..Default::default()
            },
            ..Default::default()
        };
        let err = upstream_base_url(&deployment).expect_err("should error");
        assert_eq!(err.status, axum::http::StatusCode::INTERNAL_SERVER_ERROR);
        assert!(err.message.contains("madeup"));
    }

    #[test]
    fn build_upstream_url_with_query() {
        use oxllm_core::deployment::OxllmParams;
        let deployment = Deployment {
            model_name: "gpt-4o".to_string(),
            oxllm_params: OxllmParams {
                model: "openai/gpt-4o".to_string(),
                api_key: Some("k".to_string()),
                api_base: Some("https://example.test/v1".to_string()),
                ..Default::default()
            },
            ..Default::default()
        };
        let url = build_upstream_url(&deployment, "batches", Some("limit=10")).unwrap();
        assert_eq!(url, "https://example.test/v1/batches?limit=10");
    }

    #[test]
    fn merge_base_path_query_preserves_base_query() {
        let url = merge_base_path_query(
            "https://example.test/openai?api-version=2024-02-01",
            "batches",
            Some("limit=5&cursor=a%20b"),
        )
        .expect("merged url");
        assert_eq!(
            url,
            "https://example.test/openai/batches?api-version=2024-02-01&limit=5&cursor=a+b"
        );
    }

    #[test]
    fn merge_base_path_query_without_inbound_query() {
        let url = merge_base_path_query(
            "https://example.test/openai?api-version=2024-02-01",
            "batches",
            None,
        )
        .expect("merged url");
        assert_eq!(
            url,
            "https://example.test/openai/batches?api-version=2024-02-01"
        );
    }

    // Bug #43 — when the inbound query contains a key already present in the
    // base URL's query (notably Azure `api-version`), the base value must
    // take precedence and the duplicate must NOT be appended. Otherwise the
    // upstream sees two values for the same key and behaves unpredictably.
    #[test]
    fn merge_base_path_query_skips_inbound_duplicate_keys() {
        let url = merge_base_path_query(
            "https://example.test/openai?api-version=2024-02-01",
            "batches",
            // Inbound repeats `api-version` (should be skipped) plus a new
            // `limit` (should be appended).
            Some("api-version=2099-01-01&limit=5"),
        )
        .expect("merged url");
        assert_eq!(
            url, "https://example.test/openai/batches?api-version=2024-02-01&limit=5",
            "base api-version must win; inbound duplicate must be dropped"
        );
    }

    #[test]
    fn merge_base_path_query_keeps_distinct_inbound_keys() {
        let url = merge_base_path_query(
            "https://example.test/openai?api-version=2024-02-01",
            "batches",
            Some("limit=5&cursor=abc"),
        )
        .expect("merged url");
        // Sanity check that the dedup logic doesn't accidentally drop
        // legitimate distinct keys.
        assert!(url.contains("api-version=2024-02-01"));
        assert!(url.contains("limit=5"));
        assert!(url.contains("cursor=abc"));
    }

    // The `model` query parameter is an oxllm routing input, not a provider
    // API parameter — it must be stripped before forwarding upstream so the
    // internal alias is not leaked.
    #[test]
    fn merge_base_path_query_strips_model_query_param() {
        let url = merge_base_path_query(
            "https://example.test/v1",
            "batches",
            Some("model=internal-alias&limit=5"),
        )
        .expect("merged url");
        assert!(
            url.contains("limit=5"),
            "non-model params must be forwarded: {url}"
        );
        assert!(
            !url.contains("model="),
            "model param must be stripped from upstream URL: {url}"
        );
    }

    #[test]
    fn merge_base_path_query_strips_only_model_not_other_params() {
        let url = merge_base_path_query(
            "https://example.test/v1",
            "files",
            Some("model=foo&purpose=fine-tune&limit=10"),
        )
        .expect("merged url");
        assert!(url.contains("purpose=fine-tune"));
        assert!(url.contains("limit=10"));
        assert!(!url.contains("model="));
    }

    #[test]
    fn provider_auth_headers_use_provider_native_schemes() {
        use oxllm_core::deployment::OxllmParams;
        use oxllm_core::provider::ResolvedCredentials;

        fn dep(model: &str) -> Deployment {
            Deployment {
                model_name: "m".to_string(),
                oxllm_params: OxllmParams {
                    model: model.to_string(),
                    api_key: Some("secret".to_string()),
                    ..Default::default()
                },
                ..Default::default()
            }
        }

        let anthropic = dep("anthropic/claude");
        let anthropic_provider: std::sync::Arc<dyn oxllm_core::Provider> =
            std::sync::Arc::new(oxllm_providers::anthropic::AnthropicProvider);
        let headers = provider_auth_headers(
            Some(&anthropic_provider),
            &anthropic,
            &ResolvedCredentials::from_key("secret"),
        )
        .expect("anthropic auth");
        assert_eq!(headers.get("x-api-key").map(String::as_str), Some("secret"));
        assert_eq!(
            headers.get("anthropic-version").map(String::as_str),
            Some(oxllm_providers::anthropic::ANTHROPIC_VERSION)
        );

        let azure = dep("azure/my-deployment");
        let azure_provider: std::sync::Arc<dyn oxllm_core::Provider> =
            std::sync::Arc::new(oxllm_providers::azure::AzureProvider);
        let headers = provider_auth_headers(
            Some(&azure_provider),
            &azure,
            &ResolvedCredentials::Azure {
                api_key: "secret".to_string(),
            },
        )
        .expect("azure auth");
        assert_eq!(headers.get("api-key").map(String::as_str), Some("secret"));

        let openai = dep("openai/gpt-4o");
        let openai_provider: std::sync::Arc<dyn oxllm_core::Provider> =
            std::sync::Arc::new(oxllm_providers::openai::OpenAiProvider);
        let headers = provider_auth_headers(
            Some(&openai_provider),
            &openai,
            &ResolvedCredentials::from_key("secret"),
        )
        .expect("openai auth");
        assert_eq!(
            headers.get("authorization").map(String::as_str),
            Some("Bearer secret")
        );

        let headers =
            provider_auth_headers(None, &openai, &ResolvedCredentials::from_key("secret"))
                .expect("fallback auth");
        assert_eq!(
            headers.get("authorization").map(String::as_str),
            Some("Bearer secret")
        );
    }

    #[test]
    fn provider_auth_headers_deepgram_uses_token_scheme() {
        // Deepgram requires `Authorization: Token <key>`, not Bearer. The
        // audio-only provider exposes no chat/embeddings/rerank surface, so
        // the auth helper must be reached via the slug dispatch — otherwise
        // the Bearer fallback would send a scheme Deepgram rejects.
        use oxllm_core::deployment::OxllmParams;
        use oxllm_core::provider::ResolvedCredentials;

        let deepgram = Deployment {
            model_name: "nova-2".to_string(),
            oxllm_params: OxllmParams {
                model: "deepgram/nova-2".to_string(),
                api_key: Some("dg-secret".to_string()),
                ..Default::default()
            },
            ..Default::default()
        };
        let provider: std::sync::Arc<dyn oxllm_core::Provider> =
            std::sync::Arc::new(oxllm_providers::deepgram::DeepgramProvider);
        let headers = provider_auth_headers(
            Some(&provider),
            &deepgram,
            &ResolvedCredentials::from_key("dg-secret"),
        )
        .expect("deepgram auth");
        assert_eq!(
            headers.get("authorization").map(String::as_str),
            Some("Token dg-secret")
        );
    }

    #[test]
    fn provider_auth_headers_elevenlabs_uses_xi_api_key() {
        // ElevenLabs authenticates with the custom `xi-api-key` header rather
        // than the `Authorization` scheme. Reached via slug dispatch.
        use oxllm_core::deployment::OxllmParams;
        use oxllm_core::provider::ResolvedCredentials;

        let elevenlabs = Deployment {
            model_name: "multilingual".to_string(),
            oxllm_params: OxllmParams {
                model: "elevenlabs/multilingual".to_string(),
                api_key: Some("xi-secret".to_string()),
                ..Default::default()
            },
            ..Default::default()
        };
        let provider: std::sync::Arc<dyn oxllm_core::Provider> =
            std::sync::Arc::new(oxllm_providers::elevenlabs::ElevenLabsProvider);
        let headers = provider_auth_headers(
            Some(&provider),
            &elevenlabs,
            &ResolvedCredentials::from_key("xi-secret"),
        )
        .expect("elevenlabs auth");
        assert_eq!(
            headers.get("xi-api-key").map(String::as_str),
            Some("xi-secret")
        );
        assert!(!headers.contains_key("authorization"));
    }

    #[test]
    fn build_upstream_url_strips_leading_slash() {
        use oxllm_core::deployment::OxllmParams;
        let deployment = Deployment {
            model_name: "gpt-4o".to_string(),
            oxllm_params: OxllmParams {
                model: "openai/gpt-4o".to_string(),
                api_key: Some("k".to_string()),
                api_base: Some("https://example.test/v1".to_string()),
                ..Default::default()
            },
            ..Default::default()
        };
        let url = build_upstream_url(&deployment, "/files", None).unwrap();
        assert_eq!(url, "https://example.test/v1/files");
    }

    // CRITICAL: a dot-segment payload that would normalise out from under the
    // base path prefix must be rejected at the build_upstream_url chokepoint,
    // even when a handler failed to validate its `{id}` segment.
    #[test]
    fn build_upstream_url_rejects_dot_segment_escape() {
        use oxllm_core::deployment::OxllmParams;
        let deployment = Deployment {
            model_name: "gpt-4o".to_string(),
            oxllm_params: OxllmParams {
                model: "openai/gpt-4o".to_string(),
                api_key: Some("k".to_string()),
                api_base: Some("https://api.openai.com/v1".to_string()),
                ..Default::default()
            },
            ..Default::default()
        };
        let err = build_upstream_url(&deployment, "batches/../../users", None)
            .expect_err("escape must be rejected");
        assert_eq!(err.status, axum::http::StatusCode::FORBIDDEN);
    }

    #[test]
    fn build_upstream_url_accepts_path_within_prefix() {
        use oxllm_core::deployment::OxllmParams;
        let deployment = Deployment {
            model_name: "gpt-4o".to_string(),
            oxllm_params: OxllmParams {
                model: "openai/gpt-4o".to_string(),
                api_key: Some("k".to_string()),
                api_base: Some("https://api.openai.com/v1".to_string()),
                ..Default::default()
            },
            ..Default::default()
        };
        let url = build_upstream_url(&deployment, "batches/batch_1/cancel", None).unwrap();
        assert_eq!(url, "https://api.openai.com/v1/batches/batch_1/cancel");
    }

    #[test]
    fn path_within_prefix_boundary_cases() {
        assert!(path_within_prefix("/v1/batches", "/v1"));
        assert!(path_within_prefix("/v1", "/v1"));
        assert!(path_within_prefix("/anything", "/"));
        assert!(path_within_prefix("/anything", ""));
        assert!(!path_within_prefix("/users", "/v1"));
        assert!(!path_within_prefix("/v1other", "/v1"));
    }

    #[test]
    fn resolve_model_uses_explicit_model() {
        let state = AppState::with_registry(base_config(), oxllm_providers::default_registry())
            .expect("state built");
        let model = resolve_model(&state, Some("gpt-4o"), &Value::Null, None).expect("model");
        assert_eq!(model, "gpt-4o");
    }

    #[test]
    fn resolve_model_extracts_from_body() {
        let state = AppState::with_registry(base_config(), oxllm_providers::default_registry())
            .expect("state built");
        let body = serde_json::json!({"model": "claude-3"});
        let model = resolve_model(&state, None, &body, None).expect("model");
        assert_eq!(model, "claude-3");
    }

    #[test]
    fn resolve_model_extracts_nested_from_requests_params() {
        // Anthropic Message Batches nests the model under
        // requests[0].params.model — resolve_model must extract it without
        // a separate handler-level parse.
        let state = AppState::with_registry(base_config(), oxllm_providers::default_registry())
            .expect("state built");
        let body = serde_json::json!({
            "requests": [
                {"custom_id": "a", "params": {"model": "claude-3-opus", "messages": []}}
            ]
        });
        let model = resolve_model(&state, None, &body, Some("anthropic")).expect("model");
        assert_eq!(model, "claude-3-opus");
    }

    #[test]
    fn resolve_model_prefers_top_level_over_nested() {
        let state = AppState::with_registry(base_config(), oxllm_providers::default_registry())
            .expect("state built");
        let body = serde_json::json!({
            "model": "claude-3-5-sonnet",
            "requests": [{"params": {"model": "claude-3-opus"}}]
        });
        let model = resolve_model(&state, None, &body, None).expect("model");
        assert_eq!(model, "claude-3-5-sonnet");
    }

    #[test]
    fn is_json_content_type_defaults_to_true_when_unset() {
        assert!(is_json_content_type(None));
    }

    #[test]
    fn is_json_content_type_handles_multipart() {
        assert!(!is_json_content_type(Some(
            "multipart/form-data; boundary=xyz"
        )));
    }

    // NDJSON is NOT a single JSON document: `serde_json::from_slice` cannot
    // parse JSONL, so treating it as JSON made `hooked_raw` / `raw_forward`
    // reject a valid JSONL body with a spurious `400 invalid JSON body`
    // instead of forwarding it verbatim.
    #[test]
    fn is_json_content_type_excludes_ndjson() {
        assert!(!is_json_content_type(Some("application/x-ndjson")));
        assert!(!is_json_content_type(Some(
            "application/x-ndjson; charset=utf-8"
        )));
        // The real JSON family still parses.
        assert!(is_json_content_type(Some("application/json")));
        assert!(is_json_content_type(Some(
            "application/json; charset=utf-8"
        )));
    }

    // ---- U57: hooked passthrough charges aggregate TPM --------------------

    fn state_with_mock_deployment(api_base: String) -> AppState {
        use crate::config::types::{ConfigModel, GeneralSettings, ProxyConfig};
        use oxllm_core::deployment::OxllmParams;
        let model_list = vec![ConfigModel {
            model_name: "gpt-4o".to_string(),
            oxllm_params: OxllmParams {
                model: "openai/gpt-4o".to_string(),
                api_key: Some("sk-test".to_string()),
                api_base: Some(api_base),
                ..Default::default()
            },
            ..Default::default()
        }];
        let config = ProxyConfig {
            model_list,
            oxllm_settings: OxllmSettings::default(),
            general_settings: GeneralSettings {
                allow_private_api_base: Some(true),
                ..Default::default()
            },
            router_settings: Default::default(),
            extra: serde_json::Map::new(),
        };
        AppState::with_registry(config, oxllm_providers::default_registry()).expect("state built")
    }

    #[tokio::test]
    async fn hooked_passthrough_charges_total_tokens_to_tpm_window() {
        use crate::pipeline::UPSTREAM_TOTAL_TOKENS_HEADER;
        use httpmock::MockServer;

        let server = MockServer::start();
        server.mock(|when, then| {
            when.method("POST").path("/chat/completions");
            then.status(200)
                .header("content-type", "application/json")
                // Upstream reports an aggregate total only — no
                // prompt/completion split (the U3 header).
                .header(UPSTREAM_TOTAL_TOKENS_HEADER, "1234")
                .body("{}");
        });

        let state = state_with_mock_deployment(server.base_url());
        let auth = AuthUser::open();
        let req = PassthroughRequest::new(
            Method::POST,
            Some("gpt-4o"),
            None,
            Bytes::from_static(b"{}"),
            Some("application/json"),
        );

        let resp = forward_with_mode(
            &state,
            &auth,
            PassthroughMode::HookedRaw {
                upstream_path: "chat/completions".to_string(),
            },
            req,
        )
        .await
        .expect("passthrough ok");
        assert!(resp.status().is_success());

        // The TPM counter must reflect the aggregate total_tokens, not the
        // zeroed completion half — without U57 this counter stayed at 0.
        let tpm = state
            .cache
            .get("dep_tpm:gpt-4o:openai/gpt-4o")
            .await
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(0.0);
        assert!(
            (tpm - 1234.0).abs() < 1e-9,
            "aggregate total_tokens must be charged to the TPM window, got {tpm}"
        );
    }

    #[tokio::test]
    async fn hooked_passthrough_skips_tpm_when_upstream_reports_no_usage() {
        // No usage header → no TPM charge (nothing to charge).
        use httpmock::MockServer;

        let server = MockServer::start();
        server.mock(|when, then| {
            when.method("POST").path("/chat/completions");
            then.status(200)
                .header("content-type", "application/json")
                .body("{}");
        });

        let state = state_with_mock_deployment(server.base_url());
        let auth = AuthUser::open();
        let req = PassthroughRequest::new(
            Method::POST,
            Some("gpt-4o"),
            None,
            Bytes::from_static(b"{}"),
            Some("application/json"),
        );

        let resp = forward_with_mode(
            &state,
            &auth,
            PassthroughMode::HookedRaw {
                upstream_path: "chat/completions".to_string(),
            },
            req,
        )
        .await
        .expect("passthrough ok");
        assert!(resp.status().is_success());

        let present = state.cache.get("dep_tpm:gpt-4o:openai/gpt-4o").await;
        assert!(
            present.is_none(),
            "no usage header → TPM window must not be touched"
        );
    }

    // MEDIUM: OpenAI Threads runs carry usage in the JSON response body, not
    // in headers. Without body-level recovery the runs route records zero
    // spend and the deployment TPM cap is never charged.
    #[tokio::test]
    async fn hooked_passthrough_recovers_threads_runs_usage_from_body() {
        use httpmock::MockServer;

        let server = MockServer::start();
        server.mock(|when, then| {
            when.method("POST").path("/threads/thread_1/runs");
            then.status(200)
                .header("content-type", "application/json")
                .json_body(json!({
                    "id": "run_1",
                    "object": "thread.run",
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 50,
                        "total_tokens": 150
                    }
                }));
        });

        let state = state_with_mock_deployment(server.base_url());
        let auth = AuthUser::open();
        let body = serde_json::json!({
            "assistant_id": "asst_1",
            "model": "gpt-4o"
        });
        let req = PassthroughRequest::new(
            Method::POST,
            None,
            None,
            Bytes::from(serde_json::to_vec(&body).unwrap()),
            Some("application/json"),
        );

        let resp = forward_with_mode(
            &state,
            &auth,
            PassthroughMode::HookedRaw {
                upstream_path: "threads/thread_1/runs".to_string(),
            },
            req,
        )
        .await
        .expect("passthrough ok");
        assert!(resp.status().is_success());

        // The TPM counter must reflect the total_tokens from the body usage.
        let tpm = state
            .cache
            .get("dep_tpm:gpt-4o:openai/gpt-4o")
            .await
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(0.0);
        assert!(
            (tpm - 150.0).abs() < 1e-9,
            "threads runs body usage must be charged to the TPM window, got {tpm}"
        );

        // The response body must be relayed verbatim (not consumed by the
        // usage recovery).
        let body_bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .expect("body readable");
        let v: serde_json::Value = serde_json::from_slice(&body_bytes).expect("body is json");
        assert_eq!(v["id"], "run_1");
    }

    #[tokio::test]
    async fn hooked_passthrough_skips_threads_usage_recovery_for_non_threads_path() {
        // Non-threads OpenAI paths must not trigger the threads usage
        // recovery (e.g. a chat/completions response that happens to carry
        // a usage object — that's handled by header-based recovery).
        use httpmock::MockServer;

        let server = MockServer::start();
        server.mock(|when, then| {
            when.method("POST").path("/threads");
            then.status(200)
                .header("content-type", "application/json")
                .json_body(json!({"id": "thread_1", "object": "thread"}));
        });

        let state = state_with_mock_deployment(server.base_url());
        let auth = AuthUser::open();
        let req = PassthroughRequest::new(
            Method::POST,
            Some("gpt-4o"),
            None,
            Bytes::from_static(b"{}"),
            Some("application/json"),
        );

        let resp = forward_with_mode(
            &state,
            &auth,
            PassthroughMode::HookedRaw {
                upstream_path: "threads".to_string(),
            },
            req,
        )
        .await
        .expect("passthrough ok");
        assert!(resp.status().is_success());

        // Thread creation does not carry usage — TPM must be untouched.
        let present = state.cache.get("dep_tpm:gpt-4o:openai/gpt-4o").await;
        assert!(
            present.is_none(),
            "thread creation has no usage → TPM must not be touched"
        );
    }

    // ---- Budget reservation release on pre-call-then-fail paths ------------

    /// Build a state whose only hook is a [`BudgetHook`] with a small ceiling,
    /// so `pre_call` reserves a spend estimate against `spend:{key_hash}`.
    fn state_with_budget_hook(
        model_name: &str,
        litellm_model: &str,
        api_base: Option<String>,
    ) -> AppState {
        use crate::config::types::ConfigModel;
        use crate::hooks::BudgetHook;
        use oxllm_core::deployment::OxllmParams;
        use std::time::Duration;
        let model_list = vec![ConfigModel {
            model_name: model_name.to_string(),
            oxllm_params: OxllmParams {
                model: litellm_model.to_string(),
                api_key: Some("sk-test".to_string()),
                api_base,
                ..Default::default()
            },
            ..Default::default()
        }];
        let config = ProxyConfig {
            model_list,
            oxllm_settings: OxllmSettings::default(),
            general_settings: GeneralSettings {
                allow_private_api_base: Some(true),
                ..Default::default()
            },
            router_settings: Default::default(),
            extra: serde_json::Map::new(),
        };
        let mut state = AppState::with_registry(config, oxllm_providers::default_registry())
            .expect("state built");
        let budget =
            BudgetHook::with_duration(state.cache.clone(), Some(1.0), Duration::from_secs(3600));
        state.hooks = vec![std::sync::Arc::new(budget)];
        state
    }

    fn auth_with_budget() -> AuthUser {
        AuthUser {
            max_budget: Some(1.0),
            ..AuthUser::open()
        }
    }

    // MEDIUM-1: when the router returns no deployment (all cooling down) after
    // pre_call already reserved the budget estimate, the reservation must be
    // released via post_failure rather than leaking for the budget window.
    #[tokio::test]
    async fn hooked_passthrough_releases_budget_when_deployment_unresolvable() {
        use crate::hooks::spend_keys::spend_key;

        let state = state_with_budget_hook(
            "gpt-4o",
            "openai/gpt-4o",
            Some("https://example.test/v1".to_string()),
        );
        let auth = auth_with_budget();
        let spend_key = spend_key(auth.key_hash_hex());

        // Cool down the only deployment so pick_deployment_index returns None.
        let cooldown_key = state.router.cooldown_key(&state.deployments[0]);
        state.router.mark_cooldown(&cooldown_key);

        let body = serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let req = PassthroughRequest::new(
            Method::POST,
            Some("gpt-4o"),
            None,
            Bytes::from(serde_json::to_vec(&body).unwrap()),
            Some("application/json"),
        );

        let result = forward_with_mode(
            &state,
            &auth,
            PassthroughMode::HookedRaw {
                upstream_path: "chat/completions".to_string(),
            },
            req,
        )
        .await;
        assert!(
            result.is_err(),
            "router returning None must surface an error"
        );

        // The budget estimate reserved in pre_call must be released — without
        // the fix the spend counter holds the estimate for the whole window.
        let spend = state.cache.counter_value(&spend_key).await.unwrap_or(0.0);
        assert!(
            spend.abs() < 1e-9,
            "spend reservation must be released after unresolvable deployment, got {spend}"
        );
    }

    // MEDIUM-2: when build_upstream_url fails (no api_base + unknown provider
    // slug) after pre_call reserved the budget estimate, the reservation must
    // be released via post_failure.
    #[tokio::test]
    async fn hooked_passthrough_releases_budget_when_upstream_url_unresolvable() {
        use crate::hooks::spend_keys::spend_key;

        // This custom provider is absent from default_base_for_provider and no api_base
        // is configured, so build_upstream_url errors after pre_call reserved.
        let state = state_with_budget_hook("pplx", "missing-default/pplx-1", None);
        let auth = auth_with_budget();
        let spend_key = spend_key(auth.key_hash_hex());

        let body = serde_json::json!({
            "model": "pplx",
            "messages": [{"role": "user", "content": "hi"}]
        });
        let req = PassthroughRequest::new(
            Method::POST,
            Some("pplx"),
            None,
            Bytes::from(serde_json::to_vec(&body).unwrap()),
            Some("application/json"),
        );

        let result = forward_with_mode(
            &state,
            &auth,
            PassthroughMode::HookedRaw {
                upstream_path: "batches".to_string(),
            },
            req,
        )
        .await;
        assert!(
            result.is_err(),
            "missing api_base for unknown provider must error"
        );

        let spend = state.cache.counter_value(&spend_key).await.unwrap_or(0.0);
        assert!(
            spend.abs() < 1e-9,
            "spend reservation must be released after unresolvable url, got {spend}"
        );
    }

    // LOW-1: raw passthrough must apply the same JSON-depth DoS guard as the
    // hooked tier. serde_json's own recursion limit (127) is stricter than
    // MAX_JSON_DEPTH (128), so the guard is defense-in-depth: it is wired
    // identically to hooked_raw and would bound traversal if serde's limit were
    // ever raised. This test exercises the new code path with the deepest body
    // serde can still parse, asserting the guard accepts it (does not over-reject
    // with a depth-related 400).
    #[tokio::test]
    async fn raw_passthrough_applies_json_depth_guard() {
        use httpmock::MockServer;

        let server = MockServer::start();
        server.mock(|when, then| {
            when.method("POST").path("/chat/completions");
            then.status(200)
                .header("content-type", "application/json")
                .body("{}");
        });

        let state = state_with_mock_deployment(server.base_url());
        let auth = AuthUser::open();

        // serde_json parses up to depth 127; MAX_JSON_DEPTH is 128, so a
        // 127-deep body is accepted by both serde and check_json_depth.
        let mut body = serde_json::Value::Null;
        for _ in 0..127 {
            body = serde_json::json!({ "a": body });
        }
        let req = PassthroughRequest::new(
            Method::POST,
            Some("gpt-4o"),
            None,
            Bytes::from(serde_json::to_vec(&body).unwrap()),
            Some("application/json"),
        );

        let resp = forward_with_mode(
            &state,
            &auth,
            PassthroughMode::Raw {
                upstream_path: "chat/completions".to_string(),
            },
            req,
        )
        .await
        .expect("127-deep body is within depth limit");
        assert!(
            resp.status().is_success(),
            "depth-valid body must not be rejected by the guard"
        );
    }

    // LOW-2: the cache-hit path must record a self-documenting provider label
    // rather than an empty string in ResponseSummary.
    #[tokio::test]
    async fn hooked_passthrough_cache_hit_records_nonempty_provider() {
        use async_trait::async_trait;
        use oxllm_core::{CallContext, Hook, HookAction, ResponseSummary};

        /// Short-circuits pre_call with a cached body and records the provider
        /// seen by post_success.
        struct CacheHitAndRecord {
            provider: std::sync::Arc<std::sync::Mutex<Option<String>>>,
        }

        #[async_trait]
        impl Hook for CacheHitAndRecord {
            async fn pre_call(
                &self,
                _ctx: &mut CallContext,
            ) -> oxllm_core::error::Result<HookAction> {
                Ok(HookAction::ShortCircuit(
                    serde_json::json!({"cached": true}),
                ))
            }

            async fn post_success(&self, _ctx: &CallContext, resp: &ResponseSummary) {
                *self.provider.lock().unwrap() = Some(resp.provider.clone());
            }
        }

        let provider = std::sync::Arc::new(std::sync::Mutex::new(None));
        let mut state = state_with_mock_deployment("https://example.test/v1".to_string());
        state.hooks = vec![std::sync::Arc::new(CacheHitAndRecord {
            provider: provider.clone(),
        })];
        let auth = AuthUser::open();
        let req = PassthroughRequest::new(
            Method::POST,
            Some("gpt-4o"),
            None,
            Bytes::from_static(b"{}"),
            Some("application/json"),
        );

        let resp = forward_with_mode(
            &state,
            &auth,
            PassthroughMode::HookedRaw {
                upstream_path: "chat/completions".to_string(),
            },
            req,
        )
        .await
        .expect("cache hit short-circuits");
        assert!(resp.status().is_success());

        let recorded = provider.lock().unwrap().clone();
        assert_eq!(
            recorded.as_deref(),
            Some("cache"),
            "cache-hit path must record a non-empty provider sentinel"
        );
    }

    // ---- HIGH: upstream-reported response cost overrides local compute ----

    /// Build a state with the CostTrackingHook so we can observe the cost
    /// the hook records in the spend log.
    fn state_with_cost_recording_hook(
        api_base: String,
        recorded_cost: std::sync::Arc<std::sync::Mutex<Option<f64>>>,
    ) -> AppState {
        use crate::config::types::{ConfigModel, GeneralSettings, ProxyConfig};
        use async_trait::async_trait;
        use oxllm_core::deployment::OxllmParams;
        use oxllm_core::{CallContext, Hook, HookAction, ResponseSummary};

        struct CostCaptureHook {
            captured: std::sync::Arc<std::sync::Mutex<Option<f64>>>,
        }
        #[async_trait]
        impl Hook for CostCaptureHook {
            async fn pre_call(
                &self,
                _ctx: &mut CallContext,
            ) -> oxllm_core::error::Result<HookAction> {
                Ok(HookAction::Continue)
            }
            async fn post_success(&self, _ctx: &CallContext, resp: &ResponseSummary) {
                *self.captured.lock().unwrap() = resp.cost;
            }
        }

        let model_list = vec![ConfigModel {
            model_name: "gpt-4o".to_string(),
            oxllm_params: OxllmParams {
                model: "openai/gpt-4o".to_string(),
                api_key: Some("sk-test".to_string()),
                api_base: Some(api_base),
                ..Default::default()
            },
            ..Default::default()
        }];
        let config = ProxyConfig {
            model_list,
            oxllm_settings: OxllmSettings::default(),
            general_settings: GeneralSettings {
                allow_private_api_base: Some(true),
                ..Default::default()
            },
            router_settings: Default::default(),
            extra: serde_json::Map::new(),
        };
        let mut state = AppState::with_registry(config, oxllm_providers::default_registry())
            .expect("state built");
        state.hooks = vec![std::sync::Arc::new(CostCaptureHook {
            captured: recorded_cost,
        })];
        state
    }

    // HIGH: when an upstream reports the legacy response-cost header, that value is
    // authoritative — it must appear in both the spend log (via the cost
    // hook's ResponseSummary) and the x-oxllm-response-cost response header,
    // not the systematically-undercounted prompt-only estimate the local
    // price table would otherwise derive from the legacy total-token header.
    #[tokio::test]
    async fn hooked_passthrough_honours_upstream_response_cost() {
        use crate::pipeline::{
            COST_HEADER, UPSTREAM_RESPONSE_COST_HEADER_LITELLM,
            UPSTREAM_TOTAL_TOKENS_HEADER_LITELLM,
        };
        use httpmock::MockServer;

        let server = MockServer::start();
        server.mock(|when, then| {
            when.method("POST").path("/chat/completions");
            then.status(200)
                .header("content-type", "application/json")
                .header(UPSTREAM_TOTAL_TOKENS_HEADER_LITELLM, "1000")
                // Upstream reports its own authoritative computed cost.
                .header(UPSTREAM_RESPONSE_COST_HEADER_LITELLM, "0.42")
                .body("{}");
        });

        let recorded = std::sync::Arc::new(std::sync::Mutex::new(None));
        let state = state_with_cost_recording_hook(server.base_url(), recorded.clone());
        let auth = AuthUser::open();
        let req = PassthroughRequest::new(
            Method::POST,
            Some("gpt-4o"),
            None,
            Bytes::from_static(b"{}"),
            Some("application/json"),
        );

        let resp = forward_with_mode(
            &state,
            &auth,
            PassthroughMode::HookedRaw {
                upstream_path: "chat/completions".to_string(),
            },
            req,
        )
        .await
        .expect("passthrough ok");
        assert!(resp.status().is_success());

        // The spend-log row carries the upstream-reported cost verbatim.
        let captured = *recorded.lock().unwrap();
        assert_eq!(
            captured,
            Some(0.42),
            "ResponseSummary.cost must reflect the upstream-reported value, got {captured:?}"
        );

        // The x-oxllm-response-cost response header carries the same value.
        let header = resp
            .headers()
            .get(COST_HEADER)
            .and_then(|v| v.to_str().ok())
            .expect("cost header must be stamped");
        let parsed: f64 = header.parse().expect("header must be a number");
        assert!(
            (parsed - 0.42).abs() < 1e-9,
            "x-oxllm-response-cost must be 0.42, got {parsed}"
        );
    }

    // HIGH: the oxllm-native alias (x-oxllm-response-cost) takes precedence
    // over the legacy alias when both are present.
    #[tokio::test]
    async fn hooked_passthrough_oxllm_cost_header_takes_priority() {
        use crate::pipeline::{
            COST_HEADER, UPSTREAM_RESPONSE_COST_HEADER, UPSTREAM_RESPONSE_COST_HEADER_LITELLM,
        };
        use httpmock::MockServer;

        let server = MockServer::start();
        server.mock(|when, then| {
            when.method("POST").path("/chat/completions");
            then.status(200)
                .header("content-type", "application/json")
                .header(UPSTREAM_RESPONSE_COST_HEADER, "0.7")
                .header(UPSTREAM_RESPONSE_COST_HEADER_LITELLM, "0.1")
                .body("{}");
        });

        let recorded = std::sync::Arc::new(std::sync::Mutex::new(None));
        let state = state_with_cost_recording_hook(server.base_url(), recorded.clone());
        let auth = AuthUser::open();
        let req = PassthroughRequest::new(
            Method::POST,
            Some("gpt-4o"),
            None,
            Bytes::from_static(b"{}"),
            Some("application/json"),
        );

        let resp = forward_with_mode(
            &state,
            &auth,
            PassthroughMode::HookedRaw {
                upstream_path: "chat/completions".to_string(),
            },
            req,
        )
        .await
        .expect("passthrough ok");

        let captured = *recorded.lock().unwrap();
        assert_eq!(captured, Some(0.7), "oxllm header must win over litellm");

        let header = resp
            .headers()
            .get(COST_HEADER)
            .and_then(|v| v.to_str().ok())
            .expect("cost header must be stamped");
        let parsed: f64 = header.parse().expect("header must be a number");
        assert!((parsed - 0.7).abs() < 1e-9);
    }

    // ---- MEDIUM: request-id stamping on every passthrough response -------

    #[tokio::test]
    async fn hooked_passthrough_stamps_request_id_on_success() {
        use crate::pipeline::{REQUEST_ID_HEADER, STANDARD_REQUEST_ID_HEADER};
        use httpmock::MockServer;

        let server = MockServer::start();
        server.mock(|when, then| {
            when.method("POST").path("/chat/completions");
            then.status(200)
                .header("content-type", "application/json")
                .body("{}");
        });

        let state = state_with_mock_deployment(server.base_url());
        let auth = AuthUser::open();
        let req = PassthroughRequest::new(
            Method::POST,
            Some("gpt-4o"),
            None,
            Bytes::from_static(b"{}"),
            Some("application/json"),
        );

        let resp = forward_with_mode(
            &state,
            &auth,
            PassthroughMode::HookedRaw {
                upstream_path: "chat/completions".to_string(),
            },
            req,
        )
        .await
        .expect("passthrough ok");

        // Both headers must be present (H8 contract).
        let oxllm = resp
            .headers()
            .get(REQUEST_ID_HEADER)
            .and_then(|v| v.to_str().ok())
            .expect("x-oxllm-request-id must be stamped");
        let standard = resp
            .headers()
            .get(STANDARD_REQUEST_ID_HEADER)
            .and_then(|v| v.to_str().ok())
            .expect("x-request-id must be stamped");
        assert_eq!(
            oxllm, standard,
            "both request-id headers must carry the same value"
        );
        assert!(!oxllm.is_empty(), "request id must not be empty");
    }

    #[tokio::test]
    async fn hooked_passthrough_stamps_request_id_on_upstream_error() {
        use crate::pipeline::{REQUEST_ID_HEADER, STANDARD_REQUEST_ID_HEADER};
        use httpmock::MockServer;

        let server = MockServer::start();
        server.mock(|when, then| {
            when.method("POST").path("/chat/completions");
            then.status(500).body("oops");
        });

        let state = state_with_mock_deployment(server.base_url());
        let auth = AuthUser::open();
        let req = PassthroughRequest::new(
            Method::POST,
            Some("gpt-4o"),
            None,
            Bytes::from_static(b"{}"),
            Some("application/json"),
        );

        // A non-2xx upstream is relayed verbatim (not converted to Err), so
        // the request id must still be stamped on the returned response.
        let resp = forward_with_mode(
            &state,
            &auth,
            PassthroughMode::HookedRaw {
                upstream_path: "chat/completions".to_string(),
            },
            req,
        )
        .await
        .expect("non-2xx upstream is relayed");
        assert_eq!(resp.status(), 500);

        assert!(
            resp.headers().get(REQUEST_ID_HEADER).is_some(),
            "x-oxllm-request-id must be stamped on the upstream-error path"
        );
        assert!(
            resp.headers().get(STANDARD_REQUEST_ID_HEADER).is_some(),
            "x-request-id must be stamped on the upstream-error path"
        );
    }

    // ---- MEDIUM: cost header stamping -------------------------------------

    #[tokio::test]
    async fn hooked_passthrough_stamps_cost_header_when_upstream_reports_usage() {
        use crate::pipeline::{COST_HEADER, UPSTREAM_TOTAL_TOKENS_HEADER};
        use httpmock::MockServer;

        let server = MockServer::start();
        server.mock(|when, then| {
            when.method("POST").path("/chat/completions");
            then.status(200)
                .header("content-type", "application/json")
                .header(UPSTREAM_TOTAL_TOKENS_HEADER, "100")
                .body("{}");
        });

        let state = state_with_mock_deployment(server.base_url());
        let auth = AuthUser::open();
        let req = PassthroughRequest::new(
            Method::POST,
            Some("gpt-4o"),
            None,
            Bytes::from_static(b"{}"),
            Some("application/json"),
        );

        let resp = forward_with_mode(
            &state,
            &auth,
            PassthroughMode::HookedRaw {
                upstream_path: "chat/completions".to_string(),
            },
            req,
        )
        .await
        .expect("passthrough ok");

        let header = resp
            .headers()
            .get(COST_HEADER)
            .and_then(|v| v.to_str().ok())
            .expect("x-oxllm-response-cost must be stamped when usage is reported");
        let parsed: f64 = header.parse().expect("cost must be numeric");
        assert!(
            parsed > 0.0,
            "cost from header-reported usage must be positive, got {parsed}"
        );
    }

    // ---- MEDIUM: rate-limit header stamping -------------------------------

    #[tokio::test]
    async fn hooked_passthrough_stamps_ratelimit_headers() {
        use crate::config::types::{ConfigModel, GeneralSettings, ProxyConfig};
        use crate::pipeline::{
            RATELIMIT_LIMIT_REQUESTS, RATELIMIT_REMAINING_REQUESTS, RATELIMIT_RESET_REQUESTS,
        };
        use httpmock::MockServer;
        use oxllm_core::deployment::OxllmParams;

        let server = MockServer::start();
        server.mock(|when, then| {
            when.method("POST").path("/chat/completions");
            then.status(200)
                .header("content-type", "application/json")
                .body("{}");
        });

        // A deployment with an RPM cap so the limit/remaining/reset headers
        // carry values.
        let model_list = vec![ConfigModel {
            model_name: "gpt-4o".to_string(),
            oxllm_params: OxllmParams {
                model: "openai/gpt-4o".to_string(),
                api_key: Some("sk-test".to_string()),
                api_base: Some(server.base_url()),
                rpm: Some(10),
                ..Default::default()
            },
            ..Default::default()
        }];
        let config = ProxyConfig {
            model_list,
            oxllm_settings: OxllmSettings::default(),
            general_settings: GeneralSettings {
                allow_private_api_base: Some(true),
                ..Default::default()
            },
            router_settings: Default::default(),
            extra: serde_json::Map::new(),
        };
        // Auth with a per-key RPM so the rate-limit hook populates counters
        // under the key hash that `stamp_ratelimit_headers` reads.
        let auth = AuthUser {
            rpm: Some(10),
            ..AuthUser::open()
        };
        let state = AppState::with_registry(config, oxllm_providers::default_registry())
            .expect("state built");

        let req = PassthroughRequest::new(
            Method::POST,
            Some("gpt-4o"),
            None,
            Bytes::from_static(b"{}"),
            Some("application/json"),
        );

        let resp = forward_with_mode(
            &state,
            &auth,
            PassthroughMode::HookedRaw {
                upstream_path: "chat/completions".to_string(),
            },
            req,
        )
        .await
        .expect("passthrough ok");

        // limit-requests is stamped from the per-key RPM the auth carries.
        assert!(
            resp.headers()
                .get(RATELIMIT_LIMIT_REQUESTS)
                .and_then(|v| v.to_str().ok())
                .is_some(),
            "x-ratelimit-limit-requests must be stamped on passthrough success"
        );
        let remaining = resp
            .headers()
            .get(RATELIMIT_REMAINING_REQUESTS)
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u32>().ok())
            .expect("x-ratelimit-remaining-requests must be a number");
        // The first request consumed one slot out of 10, so 9 remain.
        assert_eq!(
            remaining, 9,
            "passthrough must stamp the live remaining-requests counter"
        );
        assert_eq!(
            resp.headers()
                .get(RATELIMIT_RESET_REQUESTS)
                .and_then(|v| v.to_str().ok()),
            Some("60s")
        );
    }

    // ---- LOW: malformed JSON yields 400, not silent Null -----------------

    #[tokio::test]
    async fn hooked_passthrough_rejects_invalid_json_body() {
        let state = state_with_mock_deployment("https://example.test/v1".to_string());
        let auth = AuthUser::open();
        let req = PassthroughRequest::new(
            Method::POST,
            Some("gpt-4o"),
            None,
            Bytes::from_static(b"{not json"),
            Some("application/json"),
        );

        let err = forward_with_mode(
            &state,
            &auth,
            PassthroughMode::HookedRaw {
                upstream_path: "chat/completions".to_string(),
            },
            req,
        )
        .await
        .expect_err("malformed JSON must yield Err");
        assert_eq!(err.status, axum::http::StatusCode::BAD_REQUEST);
        assert!(
            err.message.contains("invalid JSON body"),
            "error message must explain the parse failure, got: {}",
            err.message
        );
    }

    #[tokio::test]
    async fn raw_passthrough_rejects_invalid_json_body() {
        let state = state_with_mock_deployment("https://example.test/v1".to_string());
        let auth = AuthUser::open();
        let req = PassthroughRequest::new(
            Method::POST,
            Some("gpt-4o"),
            None,
            Bytes::from_static(b"{not json"),
            Some("application/json"),
        );

        let err = forward_with_mode(
            &state,
            &auth,
            PassthroughMode::Raw {
                upstream_path: "chat/completions".to_string(),
            },
            req,
        )
        .await
        .expect_err("malformed JSON must yield Err");
        assert_eq!(err.status, axum::http::StatusCode::BAD_REQUEST);
        assert!(err.message.contains("invalid JSON body"));
    }

    // ---- LOW: Raw mode enforces model ACLs (no ACL bypass) ---------------

    #[tokio::test]
    async fn raw_passthrough_enforces_model_acl() {
        let state = state_with_mock_deployment("https://example.test/v1".to_string());
        // Key is restricted to claude-* but the request targets gpt-4o.
        let auth = AuthUser {
            key_models: Some(vec!["claude-*".to_string()]),
            ..AuthUser::open()
        };
        let req = PassthroughRequest::new(
            Method::POST,
            Some("gpt-4o"),
            None,
            Bytes::from_static(b"{}"),
            Some("application/json"),
        );

        let err = forward_with_mode(
            &state,
            &auth,
            PassthroughMode::Raw {
                upstream_path: "chat/completions".to_string(),
            },
            req,
        )
        .await
        .expect_err("ACL must reject");
        assert_eq!(err.status, axum::http::StatusCode::FORBIDDEN);
    }

    // ---- LOW: resolve_model uses the provider hint -----------------------

    #[test]
    fn resolve_model_uses_provider_hint_when_no_model_in_request() {
        use crate::config::types::{ConfigModel, GeneralSettings, ProxyConfig};
        use oxllm_core::deployment::OxllmParams;

        // Config deliberately lists Anthropic first so a naive
        // `state.deployments.first()` fallback would route OpenAI traffic
        // to Anthropic's api_base with Anthropic credentials.
        let model_list = vec![
            ConfigModel {
                model_name: "claude".to_string(),
                oxllm_params: OxllmParams {
                    model: "anthropic/claude-sonnet-4-6".to_string(),
                    api_key: Some("k1".to_string()),
                    ..Default::default()
                },
                ..Default::default()
            },
            ConfigModel {
                model_name: "gpt-4o".to_string(),
                oxllm_params: OxllmParams {
                    model: "openai/gpt-4o".to_string(),
                    api_key: Some("k2".to_string()),
                    ..Default::default()
                },
                ..Default::default()
            },
        ];
        let config = ProxyConfig {
            model_list,
            oxllm_settings: OxllmSettings::default(),
            general_settings: GeneralSettings::default(),
            router_settings: Default::default(),
            extra: serde_json::Map::new(),
        };
        let state = AppState::with_registry(config, oxllm_providers::default_registry())
            .expect("state built");

        // Model-less request, hint = openai → must resolve to gpt-4o.
        let model = resolve_model(&state, None, &Value::Null, Some("openai")).expect("model");
        assert_eq!(
            model, "gpt-4o",
            "provider hint must pick the OpenAI deployment, not Anthropic"
        );

        let error =
            resolve_model(&state, None, &Value::Null, None).expect_err("ambiguous without a hint");
        assert_eq!(error.status, axum::http::StatusCode::BAD_REQUEST);
        assert!(error.message.contains("ambiguous"));
        assert!(
            resolve_model(&state, None, &Value::Null, Some("groq")).is_err(),
            "missing provider must not fall back to a different provider's key"
        );
    }
}
