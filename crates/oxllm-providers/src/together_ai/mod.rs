//! Together AI provider — uses Together's OpenAI-compatible endpoints.
//!
//! Wire targets:
//! - chat: `https://api.together.xyz/v1/chat/completions`
//! - embeddings: `https://api.together.xyz/v1/embeddings`
//!
//! Together AI exposes an OpenAI-compatible surface, so URL building, auth,
//! stream parsing and the supported-params list are delegated to the shared
//! [`OpenAiCompatProvider`] engine. Together diverges on one request-shape
//! point (parity with the reference Together parameter mapping):
//!
//! - **`response_format: {"type": "text"}` dropped**: Together treats a
//!   no-op `{"type": "text"}` `response_format` as an error on some models;
//!   The reference strips it before forwarding.
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
use serde_json::Value;

/// The default Together AI OpenAI-compatible base URL.
pub const DEFAULT_TOGETHER_BASE: &str = "https://api.together.xyz/v1";

/// The provider prefix stripped from model ids (`together_ai/<model>`).
const STRIP: &str = "together_ai";

/// The shared OpenAI-compatible engine backing URL building, bearer auth,
/// stream parsing and the supported-params list. Only the chat request
/// transform is overridden on the wrapper to apply Together's divergence.
const COMPAT: OpenAiCompatProvider = OpenAiCompatProvider::new(
    "together_ai",
    DEFAULT_TOGETHER_BASE,
    STRIP,
    true,
    true,
    true,
);

/// The Together AI provider. Supports both chat completions and embeddings.
#[derive(Debug, Clone, Copy)]
pub struct TogetherAiProvider;

/// Configured Together AI provider handle.
pub const PROVIDER: TogetherAiProvider = TogetherAiProvider;

/// Strip the `together_ai/` provider prefix from a model id. Kept as a free
/// function for backward compatibility with existing tests.
pub fn strip_together_prefix(model: &str) -> &str {
    OpenAiCompatProvider::strip_one_segment(model, STRIP)
}

/// Apply Together-specific request transforms to an already-built OpenAI chat
/// request body (the output of the shared compat `transform_request`).
fn apply_together_request_transforms(out: &mut Value) {
    drop_text_response_format(out);
    // Together's message schema is strict: response-only fields from a
    // previous reply cannot be replayed (reference
    // `strip_litellm_response_only_fields`).
    crate::openai_compat::strip_response_only_fields(out);
}

/// Remove `response_format` when it equals `{"type": "text"}` (parity with
/// the reference Together parameter mapping). Together rejects the
/// no-op `{"type": "text"}` value on some models with a 400.
fn drop_text_response_format(out: &mut Value) {
    let is_text = out
        .get("response_format")
        .and_then(|rf| rf.get("type"))
        .and_then(|t| t.as_str())
        .is_some_and(|t| t == "text");
    if is_text {
        if let Some(obj) = out.as_object_mut() {
            obj.remove("response_format");
        }
    }
}

impl ChatProvider for TogetherAiProvider {
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
        apply_together_request_transforms(&mut out);
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

impl EmbeddingProvider for TogetherAiProvider {
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

impl Provider for TogetherAiProvider {
    fn id(&self) -> ProviderId {
        "together_ai"
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
