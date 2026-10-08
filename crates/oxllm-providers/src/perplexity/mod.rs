//! Perplexity provider — uses Perplexity's OpenAI-compatible Chat Completions
//! endpoint.
//!
//! Wire target: `https://api.perplexity.ai/chat/completions`.
//!
//! Perplexity exposes an OpenAI-compatible surface, so URL building, auth,
//! stream parsing and the model-prefix strip are delegated to the shared
//! [`OpenAiCompatProvider`] engine. Perplexity diverges on a few points that
//! the wrapper applies (parity with the reference `PerplexityChatConfig`):
//!
//! - **Citation / search-query usage surfaced**: Perplexity's Sonar/online
//!   models bill citation tokens and per-query search fees, reported via the
//!   top-level `citations` array and `usage.num_search_queries`. The response
//!   transform promotes them into the usage object so the cost path can price
//!   them — without this, online models are systematically under-billed.
//! - **Narrow supported-params list**: Perplexity's API rejects `tools`,
//!   `tool_choice`, `seed`, `logprobs`, etc. with a 400, so the published
//!   `supported_params` is narrowed (letting `drop_params` strip them).
//! - **`web_search_options` accepted**: Sonar models support web-search
//!   configuration via the standard OpenAI `web_search_options` param.
//!
//! See `docs/adding-a-provider.md`.

use std::collections::HashMap;

use crate::openai_compat::OpenAiCompatProvider;
use oxllm_core::Result;
use oxllm_core::deployment::Deployment;
use oxllm_core::endpoint::Endpoint;
use oxllm_core::provider::{ChatProvider, Provider, ProviderId, ResolvedCredentials};
use serde_json::{Map, Value};

/// The default Perplexity OpenAI-compatible base URL.
pub const DEFAULT_PERPLEXITY_BASE: &str = "https://api.perplexity.ai";

const STRIP: &str = "perplexity";

/// The subset of the shared OpenAI-compat param list
/// ([`crate::openai_compat::SUPPORTED_CHAT_PARAMS`]) that Perplexity's API
/// accepts. Perplexity rejects `seed`, `logprobs`,
/// `logit_bias`, `store`, `metadata`, `modalities`, `audio`, and `service_tier`
/// with a 400 (parity with the reference Perplexity supported-param list).
/// `tools` / `tool_choice` ARE accepted (verified live 2026-08: sonar-pro
/// performs function calling with them; plain `sonar` answers a clear
/// upstream 400 "Tool calling is not supported for this model" — pass-through
/// beats silently stripping the pro models' capability).
/// `web_search_options` is included so Sonar models accept web-search
/// configuration; Perplexity rejects it on non-search models (acceptable).
/// Narrowing the published list means `drop_params = true` strips the
/// unsupported fields before they reach the upstream.
const PERPLEXITY_SUPPORTED_CHAT_PARAMS: &[&str] = &[
    "model",
    "messages",
    "temperature",
    "top_p",
    "max_tokens",
    "max_completion_tokens",
    "stream",
    "stream_options",
    "frequency_penalty",
    "presence_penalty",
    "response_format",
    "user",
    "reasoning_effort",
    "web_search_options",
    "tools",
    "tool_choice",
];

/// The shared OpenAI-compatible engine backing URL building, bearer auth,
/// stream parsing and the narrowed supported-params list. Only the response
/// transform is overridden on the wrapper to surface Perplexity's
/// citation/search-query usage.
const COMPAT: OpenAiCompatProvider = OpenAiCompatProvider::new(
    "perplexity",
    DEFAULT_PERPLEXITY_BASE,
    STRIP,
    false,
    true,
    true,
)
.with_supported_params(PERPLEXITY_SUPPORTED_CHAT_PARAMS);

/// The Perplexity provider. Supports chat completions only.
#[derive(Debug, Clone, Copy)]
pub struct PerplexityProvider;

/// Configured Perplexity provider handle.
pub const PROVIDER: PerplexityProvider = PerplexityProvider;

/// Strip the `perplexity/` provider prefix from a model id. Kept as a free
/// function for backward compatibility with existing tests.
pub fn strip_perplexity_prefix(model: &str) -> &str {
    OpenAiCompatProvider::strip_one_segment(model, STRIP)
}

/// Apply Perplexity-specific response transforms to an already-built OpenAI
/// chat response body (the output of the shared compat `transform_response`).
/// Promotes citation tokens and the search-query count into the usage object
/// so the cost path (`extract_usage` + `cost_from_rates`) can price them
/// (parity with the reference Perplexity usage enrichment).
#[cfg(test)]
pub(crate) fn apply_perplexity_response_transforms_for_test(out: &mut Value) {
    apply_perplexity_response_transforms(out)
}

fn apply_perplexity_response_transforms(out: &mut Value) {
    // Citation tokens: Perplexity does not report a pre-counted value, so
    // derive an estimate from the top-level `citations` array length (one
    // token per citation, floor 1 when citations exist) — matching the reference
    // heuristic. A precise count, if ever surfaced under
    // `usage.citation_tokens`, wins.
    let citation_tokens = out
        .get("usage")
        .and_then(|u| u.get("citation_tokens"))
        .and_then(|v| v.as_u64())
        .or_else(|| {
            out.get("citations")
                .and_then(|c| c.as_array())
                .map(|c| c.len() as u64)
                .filter(|n| *n > 0)
        });

    // Search-query count: Perplexity reports `usage.num_search_queries`
    // (sometimes top-level `search_queries`). Surface it under both the
    // OpenAI-standard `prompt_tokens_details.web_search_requests` (read by
    // `extract_usage`) and the raw `num_search_queries` key.
    let num_search_queries = out
        .get("usage")
        .and_then(|u| u.get("num_search_queries"))
        .and_then(|v| v.as_u64())
        .or_else(|| out.get("num_search_queries").and_then(|v| v.as_u64()))
        .or_else(|| {
            out.get("usage")
                .and_then(|u| u.get("search_queries"))
                .and_then(|v| v.as_u64())
        });

    if citation_tokens.is_none() && num_search_queries.is_none() {
        return;
    }
    let Some(usage) = out.get_mut("usage").and_then(|u| u.as_object_mut()) else {
        // Ensure a usage object exists so the fields can land on it. `out`
        // itself may be a non-object JSON body (e.g. a bare string/array
        // error payload); there is nowhere to promote the fields then, so
        // leave the body untouched rather than panicking via IndexMut.
        let Some(out_map) = out.as_object_mut() else {
            return;
        };
        let mut map = Map::new();
        if let Some(ct) = citation_tokens {
            map.insert("citation_tokens".to_string(), Value::from(ct));
        }
        if let Some(nsq) = num_search_queries {
            map.insert("num_search_queries".to_string(), Value::from(nsq));
        }
        out_map.insert("usage".to_string(), Value::Object(map));
        return;
    };
    if let Some(ct) = citation_tokens {
        usage.insert("citation_tokens".to_string(), Value::from(ct));
    }
    if let Some(nsq) = num_search_queries {
        usage.insert("num_search_queries".to_string(), Value::from(nsq));
        let details = usage
            .entry("prompt_tokens_details".to_string())
            .or_insert_with(|| Value::Object(Map::new()));
        if let Some(details_map) = details.as_object_mut() {
            details_map.insert("web_search_requests".to_string(), Value::from(nsq));
        }
    }
}

impl ChatProvider for PerplexityProvider {
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
        let mut out = ChatProvider::transform_response(&COMPAT, body, model)?;
        apply_perplexity_response_transforms(&mut out);
        Ok(out)
    }

    fn transform_stream_chunk(&self, chunk: &str) -> Result<Option<Value>> {
        ChatProvider::transform_stream_chunk(&COMPAT, chunk)
    }

    fn supported_params(&self) -> Option<Vec<&str>> {
        ChatProvider::supported_params(&COMPAT)
    }
}

impl Provider for PerplexityProvider {
    fn id(&self) -> ProviderId {
        "perplexity"
    }

    fn supports(&self, endpoint: Endpoint) -> bool {
        matches!(endpoint, Endpoint::ChatCompletions)
    }

    fn chat(&self) -> Option<&dyn ChatProvider> {
        Some(self)
    }
}

#[cfg(test)]
mod tests;
