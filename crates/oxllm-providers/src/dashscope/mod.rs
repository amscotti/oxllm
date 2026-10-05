//! DashScope (Alibaba Cloud Model Studio) provider — OpenAI-compatible.
//!
//! Wire targets (China default; override via `api_base` for intl):
//! - chat: `https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions`
//! - embeddings: `…/compatible-mode/v1/embeddings`
//! - rerank: `…/compatible-api/v1/reranks` (plural — DashScope naming; note
//!   the distinct `compatible-api` path segment — DashScope serves rerank
//!   under a different route than chat/embeddings).
//!
//! Image generation is not a typed provider transform; use the images
//! passthrough with `api_base` set to the compatible-mode base (or intl).
//! Auth: `Authorization: Bearer <DASHSCOPE_API_KEY>`.
//!
//! See `docs/adding-a-provider.md`.

use std::collections::HashMap;

use oxllm_core::Result;
use oxllm_core::deployment::Deployment;
use oxllm_core::endpoint::Endpoint;
use oxllm_core::provider::{
    ChatProvider, EmbeddingProvider, Provider, ProviderId, RerankProvider, ResolvedCredentials,
};
use serde_json::{Map, Value};

use crate::openai_compat::OpenAiCompatProvider;
use crate::util::apply_bearer_auth;

/// Default China OpenAI-compatible base URL for chat + embeddings.
///
/// International deployments should set `api_base` to
/// `https://dashscope-intl.aliyuncs.com/compatible-mode/v1`.
pub const DEFAULT_DASHSCOPE_BASE: &str = "https://dashscope.aliyuncs.com/compatible-mode/v1";

/// Default China base URL for rerank. DashScope serves rerank under the
/// distinct `compatible-api` route (not `compatible-mode` like chat/embed),
/// matching LiteLLM's `DashScopeRerankConfig.DEFAULT_RERANK_URL`.
pub const DEFAULT_DASHSCOPE_RERANK_BASE: &str = "https://dashscope.aliyuncs.com/compatible-api/v1";

const STRIP: &str = "dashscope";

/// Shared OpenAI-compat handle used for chat + embeddings transforms.
const COMPAT: OpenAiCompatProvider =
    OpenAiCompatProvider::new("dashscope", DEFAULT_DASHSCOPE_BASE, STRIP, true, true, true);

/// The configured DashScope provider (chat, embeddings, rerank).
pub const PROVIDER: DashScopeProvider = DashScopeProvider;

/// Backward-compatible unit-struct handle.
#[derive(Debug, Clone, Copy, Default)]
pub struct DashScopeProvider;

/// Strip the `dashscope/` provider prefix from a model id.
pub fn strip_dashscope_prefix(model: &str) -> &str {
    OpenAiCompatProvider::strip_one_segment(model, STRIP)
}

impl ChatProvider for DashScopeProvider {
    fn endpoint_url(&self, deployment: &Deployment) -> String {
        ChatProvider::endpoint_url(&COMPAT, deployment)
    }

    fn apply_auth(
        &self,
        deployment: &Deployment,
        creds: &ResolvedCredentials,
        headers: &mut HashMap<String, String>,
    ) {
        ChatProvider::apply_auth(&COMPAT, deployment, creds, headers);
    }

    fn transform_request(&self, request: &Value, deployment: &Deployment) -> Result<Value> {
        ChatProvider::transform_request(&COMPAT, request, deployment)
    }

    fn transform_response(&self, body: &Value, model: &str) -> Result<Value> {
        ChatProvider::transform_response(&COMPAT, body, model)
    }

    fn transform_stream_chunk(&self, chunk: &str) -> Result<Option<Value>> {
        ChatProvider::transform_stream_chunk(&COMPAT, chunk)
    }

    fn supported_params(&self) -> Option<Vec<&str>> {
        ChatProvider::supported_params(&COMPAT)
    }
}

impl EmbeddingProvider for DashScopeProvider {
    fn embedding_url(&self, deployment: &Deployment) -> String {
        EmbeddingProvider::embedding_url(&COMPAT, deployment)
    }

    fn apply_auth(
        &self,
        deployment: &Deployment,
        creds: &ResolvedCredentials,
        headers: &mut HashMap<String, String>,
    ) {
        EmbeddingProvider::apply_auth(&COMPAT, deployment, creds, headers);
    }

    fn transform_request(&self, request: &Value, deployment: &Deployment) -> Result<Value> {
        EmbeddingProvider::transform_request(&COMPAT, request, deployment)
    }

    fn transform_response(&self, body: &Value, model: &str) -> Result<Value> {
        EmbeddingProvider::transform_response(&COMPAT, body, model)
    }
}

impl RerankProvider for DashScopeProvider {
    fn rerank_url(&self, deployment: &Deployment) -> String {
        // DashScope's rerank route lives under `compatible-api/v1` (not
        // `compatible-mode/v1` like chat/embed). When the caller leaves
        // `api_base` unset we use the rerank-specific default. A configured
        // chat base needs the same route change, including regional or proxy
        // bases. Explicit rerank endpoints and unrelated custom bases stay
        // operator-controlled.
        //
        // Parity gap: LiteLLM additionally honors a rerank-specific
        // `DASHSCOPE_API_BASE_RERANK` env override (distinct from the chat
        // base). oxllm resolves a single `api_base` per deployment at config
        // load and has no rerank-specific override key — operators needing a
        // separate rerank host should set `api_base` to the full rerank URL.
        let base = match deployment.oxllm_params.api_base.as_deref() {
            Some(custom) => custom.trim_end_matches('/'),
            None => DEFAULT_DASHSCOPE_RERANK_BASE,
        };
        if base.ends_with("/reranks") || base.ends_with("/rerank") {
            base.to_string()
        } else if let Some(root) = base.strip_suffix("/compatible-mode/v1") {
            format!("{root}/compatible-api/v1/reranks")
        } else {
            format!("{base}/reranks")
        }
    }

    fn apply_auth(
        &self,
        _deployment: &Deployment,
        creds: &ResolvedCredentials,
        headers: &mut HashMap<String, String>,
    ) {
        apply_bearer_auth(creds, headers);
    }

    fn transform_request(&self, request: &Value, deployment: &Deployment) -> Result<Value> {
        let mut out = Map::new();
        let model = resolve_rerank_model(request, deployment);
        out.insert("model".to_string(), Value::String(model.to_string()));

        for key in [
            "query",
            "documents",
            "top_n",
            "return_documents",
            "instruct",
        ] {
            if let Some(v) = request.get(key) {
                if !v.is_null() {
                    out.insert(key.to_string(), v.clone());
                }
            }
        }
        Ok(Value::Object(out))
    }

    fn transform_response(&self, body: &Value, _model: &str) -> Result<Value> {
        // Compatible-mode already returns `{ results: [{ index, relevance_score, … }] }`.
        // Native DashScope wraps under `output.results` — unwrap if present.
        if body.get("results").is_some() {
            return Ok(body.clone());
        }
        if let Some(results) = body.get("output").and_then(|o| o.get("results")) {
            let mut out = Map::new();
            out.insert("results".to_string(), results.clone());
            if let Some(id) = body.get("request_id").or_else(|| body.get("id")).cloned() {
                out.insert("id".to_string(), id);
            }
            if let Some(usage) = body.get("usage").cloned() {
                out.insert("usage".to_string(), usage);
            }
            if let Some(model) = body.get("model").cloned() {
                out.insert("model".to_string(), model);
            }
            return Ok(Value::Object(out));
        }
        // DashScope error envelope (e.g. a 200-status body carrying
        // `{"code":"InvalidApiKey","message":"...","request_id":"..."}`) or any
        // other missing-results shape: surface as an upstream error so the
        // host's normal error/retry path runs. Mirrors LiteLLM's
        // `DashScopeRerankConfig.transform_rerank_response`.
        if let Some(code) = body.get("code") {
            let message = body
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("dashscope rerank error");
            let code = code_to_string(code);
            let status = dashscope_error_status(&code);
            return Err(oxllm_core::Error::Upstream {
                status,
                message: format!("{message} (code: {code})"),
            });
        }
        Err(oxllm_core::Error::Transform(format!(
            "no results in dashscope rerank response: {body}"
        )))
    }
}

/// Render a DashScope error `code` JSON value as a string for use in an
/// error message, handling both string and numeric code shapes.
fn code_to_string(code: &Value) -> String {
    match code {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// Map a DashScope error `code` string to an HTTP status that the host's
/// retry/cooldown logic can act on. DashScope wraps errors inside HTTP 200
/// responses (`{"code":"InvalidApiKey","message":"..."}`); surfacing them with
/// `status: 200` would bypass the host's retry (429/5xx) and cooldown paths,
/// leaving invalid-key and throttling errors silently unretriable. Known
/// codes are mapped by their documented prefixes; unknown codes default to
/// 500 (generic, retriable) so they are not silently swallowed.
fn dashscope_error_status(code: &str) -> u16 {
    if code.starts_with("Throttling") {
        429
    } else if code.starts_with("InvalidApiKey") || code == "AuthorizationRequired" {
        401
    } else if code.starts_with("AccessForbidden") {
        403
    } else if code.starts_with("ModelNotFound")
        || code.starts_with("Model.Unsupported")
        || code.starts_with("Model.Unexpected")
    {
        404
    } else if code.starts_with("BadRequest")
        || code.starts_with("MissingParameter")
        || code.starts_with("InvalidParameter")
        || code.starts_with("ParameterTypeUnmatched")
        || code.starts_with("DataInspection")
    {
        400
    } else {
        // Unknown codes and InternalError.* both default to 500 (generic,
        // retriable) so they are not silently swallowed as status 200.
        500
    }
}

/// Resolve the model id for a rerank request from a configured deployment.
///
/// For non-wildcard deployments the configured (prefix-stripped) model id
/// always wins. For wildcard deployments (`*` or `dashscope/*`), the actual
/// model id is taken from the request body's `model` field (after stripping
/// the provider prefix); if the request carries no `model`, the wildcard
/// literal is forwarded as-is. Mirrors `OpenAiCompatProvider::resolve_model`.
fn resolve_rerank_model<'a>(request: &'a Value, deployment: &'a Deployment) -> &'a str {
    let configured = strip_dashscope_prefix(&deployment.oxllm_params.model);
    let is_wildcard = configured == "*" || configured.ends_with("/*");
    if is_wildcard {
        request
            .get("model")
            .and_then(Value::as_str)
            .map(strip_dashscope_prefix)
            .unwrap_or(configured)
    } else {
        configured
    }
}

impl Provider for DashScopeProvider {
    fn id(&self) -> ProviderId {
        "dashscope"
    }

    fn supports(&self, endpoint: Endpoint) -> bool {
        matches!(
            endpoint,
            Endpoint::ChatCompletions | Endpoint::Embeddings | Endpoint::Rerank
        )
    }

    fn chat(&self) -> Option<&dyn ChatProvider> {
        Some(self)
    }

    fn embeddings(&self) -> Option<&dyn EmbeddingProvider> {
        Some(self)
    }

    fn rerank(&self) -> Option<&dyn RerankProvider> {
        Some(self)
    }
}

#[cfg(test)]
mod tests;
