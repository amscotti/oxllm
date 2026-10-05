//! Cerebras provider — uses Cerebras' OpenAI-compatible Chat Completions endpoint.
//!
//! Wire target: `https://api.cerebras.ai/v1/chat/completions`.
//!
//! Because Cerebras exposes an OpenAI-compatible surface, the transform is a
//! near-identity. Parsed reasoning is exposed as both Cerebras' native
//! `reasoning` and the gateway's `reasoning_content` extension. Raw reasoning
//! embedded in content is left untouched. See `docs/adding-a-provider.md`.

use std::collections::HashMap;

use oxllm_core::provider::{ChatProvider, Provider, ProviderId, ResolvedCredentials};
use oxllm_core::{Deployment, Endpoint, Result};
use serde_json::Value;

use crate::openai_compat::OpenAiCompatProvider;

/// The default Cerebras OpenAI-compatible base URL.
pub const DEFAULT_CEREBRAS_BASE: &str = "https://api.cerebras.ai/v1";

const STRIP: &str = "cerebras";

/// The configured Cerebras provider.
pub const PROVIDER: CerebrasProvider = CerebrasProvider;

const COMPAT: OpenAiCompatProvider =
    OpenAiCompatProvider::new("cerebras", DEFAULT_CEREBRAS_BASE, STRIP, false, true, true);

/// Cerebras chat adapter with parsed-reasoning normalization.
#[derive(Debug, Clone, Copy, Default)]
pub struct CerebrasProvider;

impl ChatProvider for CerebrasProvider {
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
        if let Some(format) = request.get("reasoning_format").filter(|v| !v.is_null()) {
            out["reasoning_format"] = format.clone();
        }
        // Clients may replay our normalized assistant messages. Cerebras
        // accepts historical assistant reasoning under its native field.
        if let Some(messages) = out.get_mut("messages").and_then(Value::as_array_mut) {
            for message in messages {
                if message.get("role").and_then(Value::as_str) == Some("assistant") {
                    if let Some(map) = message.as_object_mut() {
                        if let Some(reasoning) =
                            map.remove("reasoning_content").filter(Value::is_string)
                        {
                            if map.get("reasoning").is_none_or(Value::is_null) {
                                map.insert("reasoning".into(), reasoning);
                            }
                        }
                    }
                }
            }
        }
        Ok(out)
    }

    fn transform_response(&self, body: &Value, model: &str) -> Result<Value> {
        let mut out = ChatProvider::transform_response(&COMPAT, body, model)?;
        normalize_reasoning(&mut out, "message");
        Ok(out)
    }

    fn transform_stream_chunk(&self, chunk: &str) -> Result<Option<Value>> {
        let mut out = ChatProvider::transform_stream_chunk(&COMPAT, chunk)?;
        if let Some(value) = out.as_mut() {
            normalize_reasoning(value, "delta");
        }
        Ok(out)
    }

    fn supported_params(&self) -> Option<Vec<&str>> {
        let mut params = ChatProvider::supported_params(&COMPAT).unwrap_or_default();
        params.push("reasoning_format");
        Some(params)
    }
}

/// Parsed reasoning is a string at message.reasoning / delta.reasoning:
/// https://inference-docs.cerebras.ai/capabilities/reasoning
fn normalize_reasoning(value: &mut Value, field: &str) {
    if let Some(choices) = value.get_mut("choices").and_then(Value::as_array_mut) {
        for choice in choices {
            if let Some(message) = choice.get_mut(field).and_then(Value::as_object_mut) {
                if message.get("reasoning_content").is_none_or(Value::is_null) {
                    if let Some(reasoning) =
                        message.get("reasoning").filter(|v| v.is_string()).cloned()
                    {
                        message.insert("reasoning_content".into(), reasoning);
                    }
                }
            }
        }
    }
}

impl Provider for CerebrasProvider {
    fn id(&self) -> ProviderId {
        "cerebras"
    }
    fn supports(&self, endpoint: Endpoint) -> bool {
        COMPAT.supports(endpoint)
    }
    fn chat(&self) -> Option<&dyn ChatProvider> {
        Some(self)
    }
}

/// Strip the `cerebras/` provider prefix from a model id. Kept as a free
/// function for backward compatibility with existing tests.
pub fn strip_cerebras_prefix(model: &str) -> &str {
    OpenAiCompatProvider::strip_one_segment(model, STRIP)
}

#[cfg(test)]
mod tests;
