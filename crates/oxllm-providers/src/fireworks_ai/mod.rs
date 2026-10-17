//! Fireworks AI provider — uses Fireworks' OpenAI-compatible endpoints.
//!
//! Wire targets:
//! - chat: `https://api.fireworks.ai/inference/v1/chat/completions`
//! - embeddings: `https://api.fireworks.ai/inference/v1/embeddings`
//!
//! Fireworks AI exposes an OpenAI-compatible surface, so URL building, auth,
//! stream parsing and the supported-params list are delegated to the shared
//! [`OpenAiCompatProvider`] engine. Fireworks diverges on a few request-shape
//! points that the wrapper applies on top of the shared body builder (parity
//! with the reference `FireworksAIConfig`):
//!
//! - **Model name auto-prefixed**: the Fireworks API expects model names in
//!   the form `accounts/fireworks/models/<model>` (or
//!   `accounts/fireworks/routers/<model>` for `-fast` suffixed models). A
//!   short model name is prefixed; an already-prefixed name is left alone.
//! - **`tool_choice: "required"` → `"any"`**: Fireworks uses `"any"` (not the
//!   OpenAI `"required"`) to force tool calling.
//! - **`stream_options.include_usage` auto-injected**: when `stream` is set,
//!   usage reporting is forced on so Fireworks returns a final usage chunk
//!   (otherwise streaming responses carry no usage data → zero-cost streams).
//! - **`strict` stripped from tool definitions**: Fireworks rejects the OpenAI
//!   structured-outputs `strict` flag on tool function definitions.
//!
//! See `docs/adding-a-provider.md`.

use std::collections::HashMap;

use crate::openai_compat::OpenAiCompatProvider;
use oxllm_core::Result;
use oxllm_core::deployment::Deployment;
use oxllm_core::endpoint::Endpoint;
use oxllm_core::provider::{
    ChatProvider, EmbeddingProvider, Provider, ProviderId, ResolvedCredentials,
};
use serde_json::{Map, Value};

/// The default Fireworks AI OpenAI-compatible base URL.
pub const DEFAULT_FIREWORKS_AI_BASE: &str = "https://api.fireworks.ai/inference/v1";

/// The provider prefix stripped from model ids (`fireworks_ai/<model>`).
const STRIP: &str = "fireworks_ai";

/// The shared OpenAI-compatible engine backing URL building, bearer auth,
/// stream parsing and the supported-params list. Only the chat and embedding
/// request transforms are overridden on the wrapper to apply Fireworks'
/// divergences on top of the shared body builder.
const COMPAT: OpenAiCompatProvider = OpenAiCompatProvider::new(
    "fireworks_ai",
    DEFAULT_FIREWORKS_AI_BASE,
    STRIP,
    true,
    true,
    true,
);

/// The Fireworks AI provider. Supports both chat completions and embeddings.
#[derive(Debug, Clone, Copy)]
pub struct FireworksAiProvider;

/// Configured Fireworks AI provider handle.
pub const PROVIDER: FireworksAiProvider = FireworksAiProvider;

/// Strip the `fireworks_ai/` provider prefix from a model id. Kept as a free
/// function for backward compatibility with existing tests.
pub fn strip_fireworks_ai_prefix(model: &str) -> &str {
    OpenAiCompatProvider::strip_one_segment(model, STRIP)
}

/// Resolve a (prefix-stripped) Fireworks model id to the form the upstream API
/// expects. Short names are prefixed with `accounts/fireworks/models/` (or
/// `accounts/fireworks/routers/` for `-fast` suffixed models); names already
/// carrying the `accounts/` path, a `#` deployment separator, or a `/` (e.g.
/// a cross-provider wildcard like `openai/gpt-4o`) are forwarded unchanged.
/// Parity with the reference Fireworks request transform, with an added
/// `/` guard so cross-provider wildcard resolution is not mangled.
fn resolve_fireworks_model(model: &str) -> String {
    if model.starts_with("accounts/") || model.contains('#') || model.contains('/') {
        return model.to_string();
    }
    if model.ends_with("-fast") {
        format!("accounts/fireworks/routers/{model}")
    } else {
        format!("accounts/fireworks/models/{model}")
    }
}

/// Apply Fireworks-specific request transforms to an already-built OpenAI chat
/// request body (the output of the shared compat `transform_request`).
fn apply_fireworks_request_transforms(out: &mut Value) {
    if let Some(model) = out.get("model").and_then(|m| m.as_str()) {
        let resolved = resolve_fireworks_model(model);
        if resolved != model {
            out["model"] = Value::String(resolved);
        }
    }
    map_tool_choice_required_to_any(out);
    ensure_stream_usage(out);
    // Fireworks' message schema is strict: response-only fields from a
    // previous reply cannot be replayed (reference
    // `strip_litellm_response_only_fields`).
    crate::openai_compat::strip_response_only_fields(out);
    strip_strict_from_tools(out);
}

/// Rewrite `tool_choice: "required"` → `"any"` (parity with the reference
/// `FireworksAIConfig.map_openai_params`). Fireworks uses `"any"` to force
/// tool calling and rejects the OpenAI `"required"` value with a 400.
fn map_tool_choice_required_to_any(out: &mut Value) {
    let is_required = out
        .get("tool_choice")
        .and_then(|t| t.as_str())
        .is_some_and(|t| t == "required");
    if is_required {
        out["tool_choice"] = Value::String("any".to_string());
    }
}

/// Inject `stream_options.include_usage = true` when `stream` is set and the
/// caller has not explicitly disabled usage reporting (parity with the reference
/// `FireworksAIConfig.transform_request`). Without this, Fireworks streaming
/// responses carry no usage data, producing zero-cost streams.
fn ensure_stream_usage(out: &mut Value) {
    let streaming = out.get("stream").and_then(|s| s.as_bool()).unwrap_or(false);
    if !streaming {
        return;
    }
    let include_usage_explicitly_false = out
        .get("stream_options")
        .and_then(|s| s.get("include_usage"))
        .and_then(|v| v.as_bool())
        .is_some_and(|v| !v);
    if include_usage_explicitly_false {
        return;
    }
    let Some(obj) = out.as_object_mut() else {
        return;
    };
    let opts = obj
        .entry("stream_options".to_string())
        .or_insert_with(|| Value::Object(Map::new()));
    if let Some(opts_map) = opts.as_object_mut() {
        opts_map.insert("include_usage".to_string(), Value::Bool(true));
    }
}

/// Remove the `strict` flag from each tool's `function` object (parity with
/// the reference Fireworks tool transform). Fireworks rejects the
/// OpenAI structured-outputs `strict` field on tool function definitions.
fn strip_strict_from_tools(out: &mut Value) {
    let Some(tools) = out.get_mut("tools").and_then(|v| v.as_array_mut()) else {
        return;
    };
    for tool in tools.iter_mut() {
        if let Some(func) = tool.get_mut("function").and_then(|f| f.as_object_mut()) {
            func.remove("strict");
        }
    }
}

impl ChatProvider for FireworksAiProvider {
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
        let mut out = ChatProvider::transform_request(&COMPAT, request, deployment)?;
        apply_fireworks_request_transforms(&mut out);
        Ok(out)
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

impl EmbeddingProvider for FireworksAiProvider {
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
        let mut out = EmbeddingProvider::transform_request(&COMPAT, request, deployment)?;
        if let Some(model) = out.get("model").and_then(|m| m.as_str()) {
            let resolved = resolve_fireworks_model(model);
            if resolved != model {
                out["model"] = Value::String(resolved);
            }
        }
        Ok(out)
    }

    fn transform_response(&self, body: &Value, model: &str) -> Result<Value> {
        EmbeddingProvider::transform_response(&COMPAT, body, model)
    }
}

impl Provider for FireworksAiProvider {
    fn id(&self) -> ProviderId {
        "fireworks_ai"
    }

    fn supports(&self, endpoint: Endpoint) -> bool {
        matches!(endpoint, Endpoint::ChatCompletions | Endpoint::Embeddings)
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
