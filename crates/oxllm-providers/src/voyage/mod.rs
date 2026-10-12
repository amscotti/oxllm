//! Voyage AI provider — embeddings + rerank.
//!
//! Wire targets:
//! - embeddings: `https://api.voyageai.com/v1/embeddings`
//! - rerank: `https://api.voyageai.com/v1/rerank`
//!
//! Voyage is NOT OpenAI-compatible: it has its own request/response shapes.
//! For embeddings the input/model are forwarded (with `dimensions` renamed to
//! `output_dimension`) and the response is passed through with usage mapped.
//! For rerank Voyage accepts `top_k` (not `top_n`) and returns results under a
//! `data` key (not `results`); document strings are wrapped in `{ "text": … }`.
//! See `docs/adding-a-provider.md`.

use std::collections::HashMap;

use oxllm_core::Result;
use oxllm_core::deployment::Deployment;
use oxllm_core::provider::{
    EmbeddingProvider, Provider, ProviderId, RerankProvider, ResolvedCredentials,
};
use serde_json::{Map, Value, json};

use crate::util::apply_bearer_auth;

/// The default Voyage AI base URL.
pub const DEFAULT_VOYAGE_BASE: &str = "https://api.voyageai.com/v1";

/// Hosted endpoint for Voyage credentials issued by MongoDB.
pub const MONGODB_VOYAGE_BASE: &str = "https://ai.mongodb.com/v1";

/// The Voyage AI provider.
#[derive(Debug, Clone, Default)]
pub struct VoyageProvider;

/// Strip the `voyage/` provider prefix from a model id.
pub fn strip_voyage_prefix(model: &str) -> &str {
    match model.split_once('/') {
        Some(("voyage", rest)) => rest,
        _ => model,
    }
}

impl EmbeddingProvider for VoyageProvider {
    fn embedding_url(&self, deployment: &Deployment) -> String {
        // Mirrors `rerank_url`'s base normalisation so a bare custom base
        // (`https://proxy.example.com`) is treated the same way for both
        // endpoints: `/v1` is inserted, full endpoints pass through.
        //
        // Parity note: the reference Voyage embedding URL builder
        // appends `/embeddings` directly to any base not already ending in
        // `/embeddings` (so a bare proxy base yields `…/embeddings`, not
        // `…/v1/embeddings`). We deliberately diverge to keep symmetry with
        // `rerank_url` and the OpenAI-compat convention; callers porting a
        // existing config that sets a bare proxy base should append `/v1`
        // themselves or expect the `/v1` segment to be inserted.
        match deployment.oxllm_params.api_base.as_deref() {
            None => format!("{DEFAULT_VOYAGE_BASE}/embeddings"),
            Some(base) => {
                let trimmed = base.trim_end_matches('/');
                if trimmed.ends_with("/embeddings") {
                    trimmed.to_string()
                } else if trimmed.ends_with("/v1") {
                    format!("{trimmed}/embeddings")
                } else {
                    format!("{trimmed}/v1/embeddings")
                }
            }
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
        // Wildcard deployments (`voyage/*`) resolve the model from the
        // request body — same rule as the rerank path below — so the literal
        // `*` glob is never forwarded upstream as the model id.
        let model = resolve_model(request, deployment);
        out.insert("model".to_string(), Value::String(model.to_string()));
        // `input` is required by Voyage but is forwarded only when present and
        // non-null, matching the rest of this transform — a missing/null input
        // is left to the upstream to reject rather than emitting `input: null`.
        if let Some(input) = request.get("input") {
            if !input.is_null() {
                out.insert("input".to_string(), input.clone());
            }
        }
        // Voyage renames OpenAI `dimensions` to `output_dimension`.
        if let Some(dimensions) = request.get("dimensions") {
            if !dimensions.is_null() {
                out.insert("output_dimension".to_string(), dimensions.clone());
            }
        }
        if let Some(encoding_format) = request.get("encoding_format") {
            if !encoding_format.is_null() {
                out.insert("encoding_format".to_string(), encoding_format.clone());
            }
        }
        // Voyage-specific embeddings params. `input_type` (`"document"` |
        // `"query"`) drives asymmetric retrieval embeddings; `truncation`
        // controls whether over-long inputs are clipped.
        for key in ["input_type", "truncation"] {
            if let Some(v) = request.get(key) {
                if !v.is_null() {
                    out.insert(key.to_string(), v.clone());
                }
            }
        }
        Ok(Value::Object(out))
    }

    fn transform_response(&self, body: &Value, model: &str) -> Result<Value> {
        let total_tokens = body
            .get("usage")
            .and_then(|u| u.get("total_tokens"))
            .and_then(as_token_count)
            .unwrap_or(0);

        let mut out = Map::new();
        out.insert(
            "object".to_string(),
            body.get("object").cloned().unwrap_or_else(|| json!("list")),
        );
        out.insert(
            "data".to_string(),
            body.get("data").cloned().unwrap_or(Value::Null),
        );
        out.insert("model".to_string(), Value::String(model.to_string()));
        out.insert(
            "usage".to_string(),
            json!({
                "prompt_tokens": total_tokens,
                "total_tokens": total_tokens,
            }),
        );
        Ok(Value::Object(out))
    }
}

impl RerankProvider for VoyageProvider {
    fn rerank_url(&self, deployment: &Deployment) -> String {
        match deployment.oxllm_params.api_base.as_deref() {
            None => format!("{DEFAULT_VOYAGE_BASE}/rerank"),
            Some(base) => {
                let trimmed = base.trim_end_matches('/');
                if trimmed.ends_with("/v1/rerank") {
                    trimmed.to_string()
                } else if trimmed.ends_with("/v1") {
                    format!("{trimmed}/rerank")
                } else {
                    format!("{trimmed}/v1/rerank")
                }
            }
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
        let model = resolve_model(request, deployment);
        out.insert("model".to_string(), Value::String(model.to_string()));

        if let Some(query) = request.get("query") {
            if !query.is_null() {
                out.insert("query".to_string(), query.clone());
            }
        }
        if let Some(documents) = request.get("documents") {
            if !documents.is_null() {
                out.insert("documents".to_string(), documents.clone());
            }
        }
        // Voyage renames `top_n` to `top_k`.
        if let Some(top_n) = request.get("top_n") {
            if !top_n.is_null() {
                out.insert("top_k".to_string(), top_n.clone());
            }
        } else if let Some(top_n) = request.get("top_n_override") {
            if !top_n.is_null() {
                out.insert("top_k".to_string(), top_n.clone());
            }
        }
        if let Some(return_documents) = request.get("return_documents") {
            if !return_documents.is_null() {
                out.insert("return_documents".to_string(), return_documents.clone());
            }
        }
        Ok(Value::Object(out))
    }

    fn transform_response(&self, body: &Value, model: &str) -> Result<Value> {
        // Voyage returns results under `data` (not `results`). A missing `data`
        // array signals a malformed/empty/error response — surface it rather
        // than synthesizing a fake empty success that would mask the failure
        // and skip the host's retry/error-hook path. Mirrors the reference
        // `VoyageRerankConfig.transform_rerank_response` (`ValueError`).
        let raw_results = body.get("data").and_then(|v| v.as_array());
        let results: Vec<Value> = match raw_results {
            Some(arr) => arr.iter().map(map_rerank_result).collect(),
            None => {
                return Err(oxllm_core::Error::Transform(format!(
                    "no results in voyage rerank response: {body}"
                )));
            }
        };

        let total_tokens = body
            .get("usage")
            .and_then(|u| u.get("total_tokens"))
            .and_then(as_token_count)
            .unwrap_or(0);

        let id = body
            .get("id")
            .and_then(|v| v.as_str())
            .map(String::from)
            .unwrap_or_else(|| format!("voyage-rerank-{model}"));

        let mut out = Map::new();
        out.insert("id".to_string(), Value::String(id));
        out.insert("results".to_string(), Value::Array(results));
        out.insert("usage".to_string(), json!({"total_tokens": total_tokens}));
        Ok(Value::Object(out))
    }
}

impl Provider for VoyageProvider {
    fn id(&self) -> ProviderId {
        "voyage"
    }

    fn supports(&self, endpoint: oxllm_core::Endpoint) -> bool {
        matches!(
            endpoint,
            oxllm_core::Endpoint::Embeddings | oxllm_core::Endpoint::Rerank
        )
    }

    fn embeddings(&self) -> Option<&dyn EmbeddingProvider> {
        Some(self)
    }

    fn rerank(&self) -> Option<&dyn RerankProvider> {
        Some(self)
    }
}

/// Coerce a JSON token count to `u64`, accepting both integer and float
/// encodings. Some JSON encoders emit whole-number token counts as floats
/// (e.g. `120.0`); `Value::as_u64` returns `None` for those, which would
/// silently zero usage and bill the request at $0. The reference Python
/// `usage.get("total_tokens", 0)` accepts both.
fn as_token_count(v: &Value) -> Option<u64> {
    v.as_u64().or_else(|| v.as_f64().map(|f| f as u64))
}

/// Resolve the model id for an embeddings or rerank request from a
/// configured deployment.
///
/// For non-wildcard deployments the configured (prefix-stripped) model id
/// always wins. For wildcard deployments (`*` or `voyage/*`), the actual
/// model id is taken from the request body's `model` field (after stripping
/// the provider prefix) so a `voyage/*` deployment routes whatever the client
/// sent; if the request carries no `model`, the wildcard literal is forwarded
/// as-is. Mirrors `OpenAiCompatProvider::resolve_model`.
fn resolve_model<'a>(request: &'a Value, deployment: &'a Deployment) -> &'a str {
    let configured = strip_voyage_prefix(&deployment.oxllm_params.model);
    let is_wildcard = configured == "*" || configured.ends_with("/*");
    if is_wildcard {
        request
            .get("model")
            .and_then(Value::as_str)
            .map(strip_voyage_prefix)
            .unwrap_or(configured)
    } else {
        configured
    }
}

/// Wrap a string `document` in `{ "text": … }`; pass objects through unchanged.
fn map_rerank_result(result: &Value) -> Value {
    let mut out = Map::new();
    if let Some(index) = result.get("index") {
        out.insert("index".to_string(), index.clone());
    }
    if let Some(score) = result.get("relevance_score") {
        out.insert("relevance_score".to_string(), score.clone());
    }
    if let Some(document) = result.get("document") {
        let mapped = match document {
            Value::String(s) => json!({ "text": s }),
            other => other.clone(),
        };
        out.insert("document".to_string(), mapped);
    }
    Value::Object(out)
}

#[cfg(test)]
mod tests;
