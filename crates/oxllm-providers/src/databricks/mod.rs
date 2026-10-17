//! Databricks provider — uses Databricks' OpenAI-compatible serving endpoints.
//!
//! Wire targets:
//! - chat: `<api_base>/chat/completions`
//! - embeddings: `<api_base>/embeddings`
//!
//! Because Databricks exposes an OpenAI-compatible surface, URL building,
//! auth, stream parsing and the supported-params list are delegated to the
//! shared [`OpenAiCompatProvider`] engine. Databricks diverges on one
//! request-shape point (parity with the reference Databricks transform):
//!
//! - **Response-only message fields stripped**: Databricks rejects replayed
//!   gateway extension fields (`thinking_blocks`, `reasoning_content`,
//!   `reasoning_items`, `provider_specific_fields`) on messages; they are
//!   removed from every message before forwarding (reference
//!   `strip_litellm_internal_message_fields`).
//!
//! Databricks requires the per-workspace URL, so `api_base` is typically set
//! per-deployment rather than relying on the generic default below. See
//! `docs/adding-a-provider.md`.

use std::collections::HashMap;

use crate::openai_compat::OpenAiCompatProvider;
use oxllm_core::Result;
use oxllm_core::deployment::Deployment;
use oxllm_core::endpoint::Endpoint;
use oxllm_core::provider::{
    ChatProvider, EmbeddingProvider, Provider, ProviderId, ResolvedCredentials,
};
use serde_json::Value;

/// The generic Databricks serving-endpoints base URL. In practice the
/// workspace-specific URL (e.g. `https://<workspace>.cloud.databricks.com/serving-endpoints`)
/// is supplied via `api_base` on the deployment.
pub const DEFAULT_DATABRICKS_BASE: &str = "https://api.databricks.com/serving-endpoints";

/// The provider prefix stripped from model ids (`databricks/<model>`).
const STRIP: &str = "databricks";

/// The shared OpenAI-compatible engine backing URL building, bearer auth,
/// stream parsing and the supported-params list. Only the chat request
/// transform is overridden on the wrapper to apply Databricks' divergence.
const COMPAT: OpenAiCompatProvider = OpenAiCompatProvider::new(
    "databricks",
    DEFAULT_DATABRICKS_BASE,
    STRIP,
    true,
    true,
    true,
);

/// The Databricks provider. Supports both chat completions and embeddings.
#[derive(Debug, Clone, Copy)]
pub struct DatabricksProvider;

/// Configured Databricks provider handle.
pub const PROVIDER: DatabricksProvider = DatabricksProvider;

/// Strip the `databricks/` provider prefix from a model id. Kept as a free
/// function for backward compatibility with existing tests.
pub fn strip_databricks_prefix(model: &str) -> &str {
    OpenAiCompatProvider::strip_one_segment(model, STRIP)
}

/// Apply Databricks-specific request transforms to an already-built OpenAI
/// chat request body (the output of the shared compat `transform_request`).
fn apply_databricks_request_transforms(out: &mut Value) {
    // Databricks rejects replayed gateway extension fields on messages —
    // including reasoning text — for every role (reference
    // `strip_litellm_internal_message_fields`).
    crate::openai_compat::strip_internal_message_fields(out);
}

impl ChatProvider for DatabricksProvider {
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
        apply_databricks_request_transforms(&mut out);
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

impl EmbeddingProvider for DatabricksProvider {
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

impl Provider for DatabricksProvider {
    fn id(&self) -> ProviderId {
        "databricks"
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
