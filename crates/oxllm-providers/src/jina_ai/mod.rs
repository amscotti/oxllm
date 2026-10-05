//! Jina AI provider — embeddings + rerank.
//!
//! Wire targets:
//! - embeddings: `https://api.jina.ai/v1/embeddings`
//! - rerank: `https://api.jina.ai/v1/rerank`
//!
//! Jina is NOT OpenAI-compatible but its shapes are close. For embeddings a
//! scalar input is wrapped in an array and `dimensions`/`task` are forwarded.
//! The response is OpenAI-shaped and passes through. For rerank Jina returns
//! results under a `results` key with `document` as a string (wrapped in
//! `{ "text": … }`). See `docs/adding-a-provider.md`.

use std::collections::HashMap;

use oxllm_core::Result;
use oxllm_core::deployment::Deployment;
use oxllm_core::provider::{
    EmbeddingProvider, Provider, ProviderId, RerankProvider, ResolvedCredentials,
};
use serde_json::{Map, Value, json};

use crate::util::apply_bearer_auth;

/// The default Jina AI base URL.
pub const DEFAULT_JINA_BASE: &str = "https://api.jina.ai/v1";

/// The Jina AI provider.
#[derive(Debug, Clone, Default)]
pub struct JinaAIProvider;

/// Strip the `jina_ai/` provider prefix from a model id.
pub fn strip_jina_prefix(model: &str) -> &str {
    match model.split_once('/') {
        Some(("jina_ai", rest)) => rest,
        _ => model,
    }
}

/// Normalize an OpenAI-shaped `input` to Jina's expected array form. A bare
/// string becomes `[string]`; arrays pass through; everything else is wrapped
/// in a single-element array.
fn normalize_input(input: &Value) -> Value {
    match input {
        Value::Array(_) => input.clone(),
        Value::String(s) => json!([s]),
        Value::Null => json!([]),
        other => json!([other]),
    }
}

impl EmbeddingProvider for JinaAIProvider {
    fn embedding_url(&self, deployment: &Deployment) -> String {
        let base = deployment
            .oxllm_params
            .api_base
            .as_deref()
            .unwrap_or(DEFAULT_JINA_BASE)
            .trim_end_matches('/');
        if base.ends_with("/embeddings") {
            base.to_string()
        } else {
            format!("{base}/embeddings")
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
        // Wildcard deployments (`jina_ai/*`) resolve the model from the
        // request body — same rule as the rerank path below — so the literal
        // `*` glob is never forwarded upstream as the model id.
        let model = resolve_model(request, deployment);
        out.insert("model".to_string(), Value::String(model.to_string()));

        // `input` is forwarded only when present and non-null, matching the
        // Voyage transform — a missing/null input is left to the upstream to
        // reject rather than emitting `input: []`, which Jina rejects with a
        // less helpful error than omitting the field entirely.
        if let Some(input) = request.get("input") {
            if !input.is_null() {
                out.insert("input".to_string(), normalize_input(input));
            }
        }

        // Jina forwards `dimensions` and the Jina-specific `task` param.
        for key in ["dimensions", "task", "encoding_format", "late_chunking"] {
            if let Some(v) = request.get(key) {
                if !v.is_null() {
                    out.insert(key.to_string(), v.clone());
                }
            }
        }
        Ok(Value::Object(out))
    }

    fn transform_response(&self, body: &Value, model: &str) -> Result<Value> {
        // Jina returns an OpenAI-shaped response; pass it through, normalizing
        // the model field to the resolved model id.
        let mut out = body.clone();
        if let Value::Object(map) = &mut out {
            map.insert("model".to_string(), Value::String(model.to_string()));
        }
        Ok(out)
    }
}

impl RerankProvider for JinaAIProvider {
    fn rerank_url(&self, deployment: &Deployment) -> String {
        // Append semantics: a custom `api_base` has `/v1/rerank` appended
        // unless it already ends in `/v1/rerank` or `/v1`. Parity note:
        // LiteLLM's `JinaAIRerankConfig.get_complete_url` uses
        // `httpx.URL.copy_with(path="/v1/rerank")` which *replaces* the
        // entire path of a custom base (so `https://host/some/path` →
        // `https://host/v1/rerank`). We diverge and append, keeping
        // consistency with the Voyage/DashScope rerank URL helpers; callers
        // fronting Jina behind a non-`/v1` proxy path should set `api_base`
        // to the full rerank endpoint.
        let base = deployment
            .oxllm_params
            .api_base
            .as_deref()
            .unwrap_or(DEFAULT_JINA_BASE)
            .trim_end_matches('/');
        if base.ends_with("/v1/rerank") {
            base.to_string()
        } else if base.ends_with("/v1") {
            format!("{base}/rerank")
        } else {
            format!("{base}/v1/rerank")
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

        for key in ["query", "documents", "top_n", "return_documents"] {
            if let Some(v) = request.get(key) {
                if !v.is_null() {
                    out.insert(key.to_string(), v.clone());
                }
            }
        }
        Ok(Value::Object(out))
    }

    fn transform_response(&self, body: &Value, _model: &str) -> Result<Value> {
        // Jina returns results with `document` as a bare string; wrap it in
        // `{ "text": … }` to match the normalized rerank shape. A missing
        // `results` array signals a malformed/empty/error response — surface
        // it rather than synthesizing a fake empty success that would mask
        // the failure and skip the host's retry/error-hook path. Mirrors
        // LiteLLM's `JinaAIRerankConfig.transform_rerank_response`.
        let raw_results = body.get("results").and_then(|v| v.as_array());
        let results: Vec<Value> = match raw_results {
            Some(arr) => arr.iter().map(map_rerank_result).collect(),
            None => {
                return Err(oxllm_core::Error::Transform(format!(
                    "no results in jina rerank response: {body}"
                )));
            }
        };

        let id = body
            .get("id")
            .and_then(|v| v.as_str())
            .map(String::from)
            .unwrap_or_else(|| format!("jina-rerank-{}", uuid::Uuid::new_v4()));

        let mut out = Map::new();
        out.insert("id".to_string(), Value::String(id));
        out.insert("results".to_string(), Value::Array(results));
        if let Some(usage) = body.get("usage") {
            if !usage.is_null() {
                out.insert("usage".to_string(), usage.clone());
            }
        }
        Ok(Value::Object(out))
    }
}

impl Provider for JinaAIProvider {
    fn id(&self) -> ProviderId {
        "jina_ai"
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

/// Resolve the model id for an embeddings or rerank request from a
/// configured deployment.
///
/// For non-wildcard deployments the configured (prefix-stripped) model id
/// always wins. For wildcard deployments (`*` or `jina_ai/*`), the actual
/// model id is taken from the request body's `model` field (after stripping
/// the provider prefix); if the request carries no `model`, the wildcard
/// literal is forwarded as-is. Mirrors `OpenAiCompatProvider::resolve_model`.
fn resolve_model<'a>(request: &'a Value, deployment: &'a Deployment) -> &'a str {
    let configured = strip_jina_prefix(&deployment.oxllm_params.model);
    let is_wildcard = configured == "*" || configured.ends_with("/*");
    if is_wildcard {
        request
            .get("model")
            .and_then(Value::as_str)
            .map(strip_jina_prefix)
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
