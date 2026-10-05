//! Baseten Model APIs, using OpenAI-compatible chat completions and SSE.

use std::collections::HashMap;

use oxllm_core::provider::{ChatProvider, Provider, ProviderId, ResolvedCredentials};
use oxllm_core::{Deployment, Endpoint, Result};
use serde_json::Value;

use crate::openai_compat::OpenAiCompatProvider;

/// Default base URL for the hosted Model APIs.
pub const DEFAULT_BASETEN_BASE: &str = "https://inference.baseten.co/v1";

const PARAMS: &[&str] = &[
    "model",
    "messages",
    "max_tokens",
    "max_completion_tokens",
    "response_format",
    "seed",
    "stop",
    "stream",
    "temperature",
    "top_p",
    "tool_choice",
    "tools",
    "user",
    "presence_penalty",
    "frequency_penalty",
    "stream_options",
];

const COMPAT: OpenAiCompatProvider = OpenAiCompatProvider::new(
    "baseten",
    DEFAULT_BASETEN_BASE,
    "baseten",
    false,
    true,
    true,
)
.with_supported_params(PARAMS);

/// Baseten chat transform. Token limits use the API's `max_tokens` field.
#[derive(Debug, Clone, Copy, Default)]
pub struct BasetenProvider;

/// Configured Baseten provider handle.
pub const PROVIDER: BasetenProvider = BasetenProvider;

impl ChatProvider for BasetenProvider {
    fn endpoint_url(&self, deployment: &Deployment) -> String {
        COMPAT.endpoint_url(deployment)
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
        let mut body = ChatProvider::transform_request(&COMPAT, request, deployment)?;
        if let Some(object) = body.as_object_mut() {
            if let Some(limit) = object.remove("max_completion_tokens") {
                object.insert("max_tokens".to_string(), limit);
            }
        }
        Ok(body)
    }

    fn transform_response(&self, body: &Value, model: &str) -> Result<Value> {
        ChatProvider::transform_response(&COMPAT, body, model)
    }

    fn transform_stream_chunk(&self, chunk: &str) -> Result<Option<Value>> {
        COMPAT.transform_stream_chunk(chunk)
    }

    fn supported_params(&self) -> Option<Vec<&str>> {
        COMPAT.supported_params()
    }
}

impl Provider for BasetenProvider {
    fn id(&self) -> ProviderId {
        "baseten"
    }

    fn supports(&self, endpoint: Endpoint) -> bool {
        endpoint == Endpoint::ChatCompletions
    }

    fn chat(&self) -> Option<&dyn ChatProvider> {
        Some(self)
    }
}

#[cfg(test)]
mod tests;
