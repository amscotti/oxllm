//! Google Vertex AI provider — serves both Gemini and Anthropic models on Vertex.
//!
//! Vertex AI hosts two distinct upstream shapes behind a single provider slug:
//!
//! - **Gemini (and other Google models)** — Vertex exposes an OpenAI-compatible
//!   endpoint at
//!   `{base}/v1beta1/projects/{project}/locations/{location}/endpoints/openapi/chat/completions`.
//!   The request and response are the OpenAI Chat Completions format verbatim,
//!   with publisher-qualified `model` selectors such as `google/gemini-3-flash`.
//!
//! - **Anthropic Claude** — Vertex exposes a `:rawPredict` endpoint that takes
//!   the *native* Anthropic Messages body shape. The endpoint is
//!   `{base}/v1/projects/{project}/locations/{location}/publishers/anthropic/models/{model}:rawPredict`.
//!   The body must include `anthropic_version: "vertex-2023-10-16"`. Aside from
//!   that field, the request/response/streaming chunks are identical to the
//!   direct Anthropic API, so we delegate to [`AnthropicProvider`] for the
//!   transforms and only inject the version field.
//!
//! ## Configuration
//!
//! Per-deployment configuration is read from `oxllm_params.extra` so that no
//! core schema changes are required:
//!
//! - `vertex_project` (string) — the GCP project id (required).
//! - `vertex_location` (string) — the GCP location (default `us-central1`).
//! - `api_base` — the Vertex base URL override; defaults to
//!   `https://{location}-aiplatform.googleapis.com`.
//!
//! The `os.environ/` substitution performed by the config loader means YAML
//! like `vertex_project: os.environ/VERTEX_AI_PROJECT` works out of the box.
//!
//! ## Auth
//!
//! Auth is a bearer token resolved by the host into
//! [`ResolvedCredentials::GcpBearer`]. See `auth`
//! for the env-var contract.

mod auth;

use std::collections::HashMap;

use oxllm_core::Result;
use oxllm_core::deployment::Deployment;
use oxllm_core::provider::{
    ChatProvider, EmbeddingProvider, Provider, ProviderId, ResolvedCredentials,
};
use serde_json::{Value, json};

pub use auth::{ENV_GCP_ACCESS_TOKEN, ENV_VERTEX_ACCESS_TOKEN};

use crate::anthropic::AnthropicProvider;
use crate::openai_compat::{
    SUPPORTED_CHAT_PARAMS, build_embedding_request, parse_openai_stream_chunk,
};

/// The default Vertex AI base URL (`us-central1`).
pub const DEFAULT_VERTEX_BASE: &str = "https://us-central1-aiplatform.googleapis.com";

/// The default GCP location used when neither `vertex_location` nor a
/// location-embedded `api_base` is configured.
pub const DEFAULT_VERTEX_LOCATION: &str = "us-central1";

/// The `anthropic_version` value Vertex requires on Claude `:rawPredict` calls.
pub const VERTEX_ANTHROPIC_VERSION: &str = "vertex-2023-10-16";

/// The model-name prefix that selects this provider in `oxllm_params.model`.
pub const STRIP: &str = "vertex_ai";

/// The Vertex AI provider — serves Gemini via the OpenAI-compat surface and
/// Anthropic Claude via `:rawPredict`.
#[derive(Debug, Clone, Default)]
pub struct VertexProvider;

/// Strip a leading `vertex_ai/` segment from a model id. Kept as a free
/// function so tests can assert against it directly.
pub fn strip_vertex_prefix(model: &str) -> &str {
    match model.split_once('/') {
        Some((prefix, rest)) if prefix == STRIP => rest,
        _ => model,
    }
}

/// Whether a (prefix-stripped) model id refers to a Claude / Anthropic model
/// on Vertex. The check is intentionally broad — Vertex hosts Claude and a
/// handful of "anthropic-*" aliases — so that future Claude snapshot ids
/// (`claude-3-7-sonnet@20250224`, etc.) route to `:rawPredict` without code
/// changes.
fn is_anthropic_model(model: &str) -> bool {
    model.starts_with("claude")
        || model.starts_with("anthropic")
        || model.starts_with("publishers/anthropic/models/")
}

fn native_model_id(model: &str) -> &str {
    model
        .strip_prefix("publishers/anthropic/models/")
        .or_else(|| model.strip_prefix("anthropic/"))
        .unwrap_or(model)
}

fn openapi_model(model: &str) -> String {
    if model.starts_with("google/") {
        model.to_string()
    } else {
        format!("google/{model}")
    }
}

impl VertexProvider {
    /// Resolve the GCP project id for a deployment. Fails fast with
    /// [`Error::Config`] when `vertex_project` is missing or a non-string
    /// JSON type, matching the reference behaviour of refusing to build a URL
    /// against an unresolved project.
    fn project(deployment: &Deployment) -> Result<String> {
        match deployment.oxllm_params.extra.get("vertex_project") {
            Some(Value::String(p)) => Ok(p.clone()),
            Some(other) => Err(oxllm_core::Error::Config(format!(
                "vertex_ai deployment `vertex_project` must be a string, got {}",
                other_kind(other)
            ))),
            None => Err(oxllm_core::Error::Config(format!(
                "vertex_ai deployment `{}` is missing required `vertex_project`; \
                 set it in `oxllm_params.extra` (or via `os.environ/...` in YAML)",
                deployment.oxllm_params.model,
            ))),
        }
    }

    /// Resolve the GCP location for a deployment. Absent values fall back to
    /// [`DEFAULT_VERTEX_LOCATION`]; a present-but-non-string value is a
    /// configuration error (a silent type-mismatch is a debugging trap).
    fn location(deployment: &Deployment) -> Result<String> {
        match deployment.oxllm_params.extra.get("vertex_location") {
            Some(Value::String(l))
                if !l.is_empty()
                    && l.bytes().all(|byte| {
                        byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-'
                    }) =>
            {
                Ok(l.clone())
            }
            Some(Value::String(_)) => Err(oxllm_core::Error::Config(
                "vertex_location must be a non-empty lowercase region, multi-region, or global name".into())),
            Some(other) => Err(oxllm_core::Error::Config(format!(
                "vertex_ai deployment `vertex_location` must be a string, got {}",
                other_kind(other)
            ))),
            None => Ok(DEFAULT_VERTEX_LOCATION.to_string()),
        }
    }

    /// Validate that the deployment carries a usable project and location.
    /// Called at the entry of every `transform_request` so a misconfigured
    /// deployment surfaces a clear [`Error::Config`] at request time instead
    /// of baking a `unknown-project` placeholder into the upstream URL.
    fn validate_config(deployment: &Deployment) -> Result<()> {
        let _ = Self::project(deployment)?;
        let _ = Self::location(deployment)?;
        Ok(())
    }

    fn base_url(deployment: &Deployment) -> String {
        if let Some(base) = deployment.oxllm_params.api_base.as_deref() {
            return base.trim_end_matches('/').to_string();
        }
        let location =
            Self::location(deployment).unwrap_or_else(|_| DEFAULT_VERTEX_LOCATION.into());
        match location.as_str() {
            "global" => "https://aiplatform.googleapis.com".into(),
            "us" | "eu" => format!("https://aiplatform.{location}.rep.googleapis.com"),
            _ => format!("https://{location}-aiplatform.googleapis.com"),
        }
    }

    /// Publisher-level Count Tokens URL for Claude, including its optional
    /// dedicated counting region rather than the generation model's version.
    pub fn count_tokens_url(
        deployment: &Deployment,
        requested_model: Option<&str>,
    ) -> Result<String> {
        let model = Self::resolved_model(&json!({"model":requested_model}), deployment)?;
        if !is_anthropic_model(&model) {
            return Err(oxllm_core::Error::BadRequest(
                "Vertex partner token counting requires a Claude model".into(),
            ));
        }
        let mut counting = deployment.clone();
        if let Some(location) = deployment
            .oxllm_params
            .extra
            .get("vertex_count_tokens_location")
            .filter(|value| !value.is_null())
        {
            counting
                .oxllm_params
                .extra
                .insert("vertex_location".into(), location.clone());
        } else if !counting.oxllm_params.extra.contains_key("vertex_location") {
            counting
                .oxllm_params
                .extra
                .insert("vertex_location".into(), json!("us-east5"));
        }
        let project = Self::project(&counting)?;
        let location = Self::location(&counting)?;
        let base = Self::base_url(&counting);
        Ok(format!(
            "{base}/v1/projects/{project}/locations/{location}/publishers/anthropic/models/count-tokens:rawPredict"
        ))
    }

    /// Resolve the prefix-stripped model id for the *configured* deployment,
    /// expanding wildcards against the request body when the deployment is a
    /// wildcard match. A wildcard is ANY configured model containing a `*` —
    /// the bare `vertex_ai/*`, the path-prefix `vertex_ai/gemini-*`, and the
    /// Claude glob `vertex_ai/claude-*` (the exact configuration this
    /// provider's own bare-wildcard error message recommends) all match the
    /// host router's glob semantics (`wildcard_model_matches`), so all must
    /// resolve the concrete id from the request body. The resolved id is
    /// validated against a path-segment allowlist so a client-controlled
    /// wildcard model cannot inject `/` or `..` into the upstream URL path
    /// (SSRF / path-confusion defence; see `validate_model_segment`).
    fn resolved_model(request: &Value, deployment: &Deployment) -> Result<String> {
        let configured = strip_vertex_prefix(&deployment.oxllm_params.model);
        let is_wildcard = configured.contains('*');
        let resolved = if is_wildcard {
            match request.get("model").and_then(Value::as_str) {
                Some(m) => strip_vertex_prefix(m).to_string(),
                None => {
                    return Err(oxllm_core::Error::Transform(
                        "wildcard vertex_ai deployment requires a string `model` in the request body"
                            .into(),
                    ));
                }
            }
        } else {
            configured.to_string()
        };
        for (prefix, publisher) in [
            ("publishers/google/models/", "google"),
            ("google/", "google"),
            ("publishers/anthropic/models/", "anthropic"),
            ("anthropic/", "anthropic"),
        ] {
            if let Some(id) = resolved.strip_prefix(prefix) {
                validate_model_segment(id)?;
                return Ok(format!("{publisher}/{id}"));
            }
        }
        validate_model_segment(&resolved)?;
        Ok(resolved)
    }

    /// Whether the resolved request uses native Anthropic `rawPredict`, using
    /// the same model resolver and branch predicate as request transformation.
    /// This identifies the API envelope only; body fields and model feature or
    /// pricing availability do not choose the route.
    pub fn uses_anthropic_messages_api(request: &Value, deployment: &Deployment) -> Result<bool> {
        Self::resolved_model(request, deployment).map(|model| is_anthropic_model(&model))
    }

    /// Embeddings URL. Vertex exposes an OpenAI-compatible embeddings
    /// endpoint that lives under the same `/endpoints/openapi` path as the
    /// chat endpoint — the only difference is the trailing `/embeddings`
    /// instead of `/chat/completions`. Used for all Gemini-family embedding
    /// models (`text-embedding-005`, `text-multilingual-embedding-002`, …).
    fn build_embedding_url(&self, deployment: &Deployment) -> String {
        let base = Self::base_url(deployment);
        let project = Self::project(deployment).unwrap_or_else(|e| {
            tracing::error!(error = %e, "using 'unknown-project' placeholder in embedding URL");
            "unknown-project".to_string()
        });
        let location = Self::location(deployment).unwrap_or_else(|e| {
            tracing::error!(error = %e, "using default location in embedding URL");
            DEFAULT_VERTEX_LOCATION.to_string()
        });
        format!(
            "{base}/v1beta1/projects/{project}/locations/{location}/endpoints/openapi/embeddings"
        )
    }
}

impl ChatProvider for VertexProvider {
    fn endpoint_url(&self, deployment: &Deployment) -> String {
        let base = Self::base_url(deployment);
        // `endpoint_url` returns `String` by trait contract, so a missing/
        // invalid project or location cannot propagate as an `Err` here. The
        // fail-fast path lives in `transform_request` (`validate_config`); this
        // fallback only ever fires for direct trait callers (tests / probes)
        // that skip the transform step. We log at `error!` so the misconfigured
        // deployment is still visible in server logs.
        let project = Self::project(deployment).unwrap_or_else(|e| {
            tracing::error!(error = %e, "using 'unknown-project' placeholder in URL");
            "unknown-project".to_string()
        });
        let location = Self::location(deployment).unwrap_or_else(|e| {
            tracing::error!(error = %e, "using default location in URL");
            DEFAULT_VERTEX_LOCATION.to_string()
        });
        let model = strip_vertex_prefix(&deployment.oxllm_params.model);

        if is_anthropic_model(model) {
            let model = native_model_id(model);
            format!(
                "{base}/v1/projects/{project}/locations/{location}/publishers/anthropic/models/{model}:rawPredict"
            )
        } else {
            format!(
                "{base}/v1beta1/projects/{project}/locations/{location}/endpoints/openapi/chat/completions"
            )
        }
    }

    fn endpoint_url_for_request(&self, deployment: &Deployment, request: &Value) -> String {
        let mut resolved = deployment.clone();
        if let Ok(model) = Self::resolved_model(request, deployment) {
            resolved.oxllm_params.model = format!("vertex_ai/{model}");
        }
        self.endpoint_url(&resolved)
    }

    fn apply_auth(
        &self,
        deployment: &Deployment,
        creds: &ResolvedCredentials,
        headers: &mut HashMap<String, String>,
    ) {
        let token = match creds {
            ResolvedCredentials::GcpBearer { token } => token.as_str(),
            other => match other.best_effort_bearer() {
                Some(b) => b,
                None => {
                    tracing::warn!(
                        model = %deployment.oxllm_params.model,
                        "vertex_ai deployment received a credential variant with \
                         no bearer token; the upstream request will have no \
                         Authorization header and will likely 401",
                    );
                    return;
                }
            },
        };
        headers.insert("authorization".to_string(), format!("Bearer {token}"));
    }

    fn transform_request(&self, request: &Value, deployment: &Deployment) -> Result<Value> {
        Self::validate_config(deployment)?;
        let model = Self::resolved_model(request, deployment)?;
        if is_anthropic_model(&model) {
            // Anthropic's transform strips the `anthropic/` prefix but not
            // `vertex_ai/`, so hand it a deployment whose model is already
            // stripped to the bare Claude id. Vertex's `:rawPredict` endpoint
            // encodes the model in the URL path and rejects a top-level
            // `model` field in the body with a 400 ("Extra inputs are not
            // permitted") — the reference explicitly pops it
            // (`transformation.py:102`, "vertex anthropic doesn't accept
            // 'model' parameter") — so we strip it after delegating.
            // Request-aware endpoint selection resolves wildcard models before
            // constructing the URL; the native body never needs a model field.
            let mut stripped_deployment = deployment.clone();
            stripped_deployment.oxllm_params.model = native_model_id(&model).to_string();
            let mut body = AnthropicProvider.transform_request(request, &stripped_deployment)?;
            if let Some(obj) = body.as_object_mut() {
                obj.remove("model");
                obj.insert(
                    "anthropic_version".to_string(),
                    json!(VERTEX_ANTHROPIC_VERSION),
                );
                // the reference Vertex Anthropic request transform
                // (`transformation.py:108-141`) mirrors the computed beta set
                // into the body as `anthropic_beta` (Vertex's `:rawPredict`
                // reads the body, not the `anthropic-beta` header, for
                // feature-gated knobs). The host only computes beta headers
                // for the direct `anthropic` provider, so we compute them
                // here against the caller's OpenAI-shape request.
                let betas = crate::anthropic::collect_beta_headers(request, &model);
                if !betas.is_empty() {
                    obj.insert("anthropic_beta".to_string(), json!(betas));
                }
            }
            Ok(body)
        } else {
            let params = SUPPORTED_CHAT_PARAMS
                .iter()
                .copied()
                .filter(|param| !matches!(*param, "metadata" | "store" | "logit_bias"))
                .collect::<Vec<_>>();
            let mut body = crate::openai_compat::build_chat_request_with_params(
                request,
                &openapi_model(&model),
                &params,
            );
            crate::openai_compat::relax_gemini_unsupported_minimal_effort(&mut body, &model);
            crate::openai_compat::drop_foreign_thinking_blocks(&mut body);
            Ok(body)
        }
    }

    fn transform_response(&self, body: &Value, model: &str) -> Result<Value> {
        let stripped = strip_vertex_prefix(model);
        if is_anthropic_model(stripped)
            || body.get("type").and_then(Value::as_str) == Some("message")
        {
            AnthropicProvider.transform_response(body, model)
        } else {
            let mut out = body.clone();
            crate::gemini_usage::normalize_gemini_usage(&mut out);
            Ok(out)
        }
    }

    /// Stateful response override for the Claude branch: rebuilds the reverse
    /// tool-name map from the original request on the calling thread and seeds
    /// Anthropic's per-thread cell, so `transform_response`'s read of
    /// `TOOL_NAME_MAP` succeeds even after the `.await` on the upstream HTTP
    /// call rescheduled this task onto a different worker thread. Mirrors the
    /// Bedrock and direct-Anthropic overrides. The Gemini branch keeps the
    /// usage normalization without changing the message shape.
    fn transform_response_with_request(
        &self,
        body: &Value,
        model: &str,
        original_request: &Value,
    ) -> Result<Value> {
        let stripped = strip_vertex_prefix(model);
        if is_anthropic_model(stripped)
            || body.get("type").and_then(Value::as_str) == Some("message")
        {
            AnthropicProvider.transform_response_with_request(body, model, original_request)
        } else {
            ChatProvider::transform_response(self, body, model)
        }
    }

    fn transform_stream_chunk(&self, chunk: &str) -> Result<Option<Value>> {
        let trimmed = chunk.trim();
        if trimmed.is_empty() {
            return Ok(None);
        }
        // Cheap shape-based detection: Anthropic streaming events always carry
        // a `type` field whose value begins with `message_` or
        // `content_block_` (or `error`/`ping`). OpenAI-compat chunks carry
        // `choices`. We peek at the first parsed JSON object to dispatch
        // without re-running the SSE extractor twice.
        let data_line = match extract_data_line(trimmed) {
            Some(dl) => dl,
            None => {
                // No `data:` line: a malformed Anthropic frame (e.g. an
                // `event:` line with no payload) would previously fall through
                // to `serde_json::from_str` on the raw chunk (including SSE
                // framing) and silently misroute to the OpenAI parser. Drop it
                // explicitly instead — a well-formed stream always carries a
                // `data:` line on every event.
                return Ok(None);
            }
        };
        if data_line.trim() == "[DONE]" {
            // Both providers may emit `[DONE]` frames; defer to the OpenAI
            // parser which returns `None` for it.
            return parse_openai_stream_chunk(chunk);
        }
        if let Ok(value) = serde_json::from_str::<Value>(&data_line) {
            let event_type = value.get("type").and_then(|v| v.as_str()).unwrap_or("");
            if is_anthropic_event(event_type) {
                return AnthropicProvider.transform_stream_chunk(chunk);
            }
        }
        // Default: OpenAI-compat (Gemini on Vertex), with the same billable
        // usage invariant as non-streaming responses.
        let mut parsed = parse_openai_stream_chunk(chunk)?;
        if let Some(value) = &mut parsed {
            crate::gemini_usage::normalize_gemini_usage(value);
        }
        Ok(parsed)
    }

    /// Stateful streaming override. The host's stream pump calls this (not the
    /// stateless [`ChatProvider::transform_stream_chunk`]) for every chunk, so
    /// for the Claude branch we must delegate to
    /// [`AnthropicProvider::transform_stream_chunk_stateful`] — otherwise the
    /// default impl runs and falls through to Anthropic's `thread_local!`
    /// cells, which are orphaned whenever tokio reschedules the stream task
    /// onto a different worker thread. That would drop the reverse tool-name
    /// map, the stream meta (`id`/`model`/`usage`), and the tool-block
    /// assembly state — breaking tool names, usage accumulation (and thus
    /// billing), and content-block correlation. The Gemini branch keeps the
    /// stateless OpenAI-compat parser.
    fn transform_stream_chunk_stateful(
        &self,
        chunk: &str,
        state: &mut Option<Box<dyn std::any::Any + Send + Sync>>,
        original_request: &Value,
    ) -> Result<Option<Value>> {
        let trimmed = chunk.trim();
        if trimmed.is_empty() {
            return Ok(None);
        }
        let data_line = extract_data_line(trimmed);
        if let Some(dl) = data_line {
            if dl.trim() != "[DONE]" {
                if let Ok(value) = serde_json::from_str::<Value>(&dl) {
                    let event_type = value.get("type").and_then(|v| v.as_str()).unwrap_or("");
                    if is_anthropic_event(event_type) {
                        return AnthropicProvider.transform_stream_chunk_stateful(
                            chunk,
                            state,
                            original_request,
                        );
                    }
                }
            }
        }
        // Gemini (OpenAI-compat) path: no per-stream state needed.
        self.transform_stream_chunk(chunk)
    }

    fn supported_params(&self) -> Option<Vec<&str>> {
        // The Gemini branch publishes the shared OpenAI-compat list. The
        // Anthropic branch accepts fewer params but still tolerates the
        // extras (it ignores unknown keys). Publishing the broader list keeps
        // `drop_params=false` from noisily warning on every Claude request.
        let mut params = SUPPORTED_CHAT_PARAMS.to_vec();
        if let Some(anthropic) = AnthropicProvider.supported_params() {
            for param in anthropic {
                if !params.contains(&param) {
                    params.push(param);
                }
            }
        }
        Some(params)
    }
}

impl Provider for VertexProvider {
    fn id(&self) -> ProviderId {
        "vertex_ai"
    }

    fn supports(&self, endpoint: oxllm_core::Endpoint) -> bool {
        matches!(
            endpoint,
            oxllm_core::Endpoint::ChatCompletions | oxllm_core::Endpoint::Embeddings
        )
    }

    fn chat(&self) -> Option<&dyn ChatProvider> {
        Some(self)
    }

    fn embeddings(&self) -> Option<&dyn EmbeddingProvider> {
        Some(self)
    }
}

impl EmbeddingProvider for VertexProvider {
    fn embedding_url(&self, deployment: &Deployment) -> String {
        Self::build_embedding_url(self, deployment)
    }

    fn apply_auth(
        &self,
        deployment: &Deployment,
        creds: &ResolvedCredentials,
        headers: &mut HashMap<String, String>,
    ) {
        // Same GCP bearer auth as the chat endpoint.
        ChatProvider::apply_auth(self, deployment, creds, headers);
    }

    fn transform_request(&self, request: &Value, deployment: &Deployment) -> Result<Value> {
        // Vertex's OpenAI-compat embeddings endpoint accepts the standard
        // OpenAI embedding request body; we only need to resolve the model id
        // (honouring wildcards) and rewrite the `model` field.
        Self::validate_config(deployment)?;
        let model = Self::resolved_model(request, deployment)?;
        Ok(build_embedding_request(request, &openapi_model(&model)))
    }

    fn transform_response(&self, body: &Value, _model: &str) -> Result<Value> {
        Ok(body.clone())
    }
}

fn extract_data_line(raw: &str) -> Option<String> {
    crate::util::extract_sse_data(raw)
}

/// Human-readable kind name for a JSON value, used in `vertex_project` /
/// `vertex_location` type-mismatch error messages.
fn other_kind(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

/// Reject model ids that would escape their URL path segment. Vertex
/// interpolates the model verbatim into `.../models/{model}:rawPredict`; a
/// client-controlled wildcard model containing `/`, `..`, or other path
/// metacharacters could redirect the call to a different Vertex API path on
/// the same host. The allowlist covers every legitimate Vertex model id shape
/// (alphanumerics, `.`, `_`, `:`, `@`, `-`).
fn validate_model_segment(model: &str) -> Result<()> {
    if model.is_empty() {
        return Err(oxllm_core::Error::Transform(
            "vertex_ai model id is empty".into(),
        ));
    }
    if model.contains("..") {
        return Err(oxllm_core::Error::Transform(format!(
            "vertex_ai model id `{model}` contains a forbidden `..` path segment"
        )));
    }
    let ok = model
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | ':' | '@' | '-'));
    if ok {
        Ok(())
    } else {
        Err(oxllm_core::Error::Transform(format!(
            "vertex_ai model id `{model}` contains characters outside the \
             `[A-Za-z0-9._:@-]` path-segment allowlist"
        )))
    }
}

fn is_anthropic_event(event_type: &str) -> bool {
    matches!(
        event_type,
        "message_start"
            | "message_delta"
            | "message_stop"
            | "content_block_start"
            | "content_block_delta"
            | "content_block_stop"
            | "ping"
            | "error"
    )
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod validation_tests;
