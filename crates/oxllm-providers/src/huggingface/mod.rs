//! Hugging Face provider.
//!
//! Wire target: `https://router.huggingface.co`. Hugging Face exposes an
//! OpenAI-compatible Chat Completions surface — in LiteLLM its
//! `HuggingFaceChatConfig` extends `OpenAIGPTConfig` and the request/response
//! body is an OpenAI `ChatCompletionRequest` verbatim. The body transform is
//! therefore a near-identity: we strip the `huggingface/` provider prefix and
//! forward the rest.
//!
//! The one place Hugging Face diverges from a plain OpenAI-compat host is
//! **URL construction**. A request can be routed to a per-provider sub-path
//! selected by the leading segment of the model id, e.g.
//! `together/meta-llama/Llama-3-8B-Instruct` →
//! `https://router.huggingface.co/together/v1/chat/completions`. We reuse
//! [`OpenAiCompatProvider`] for the body/stream/auth/param logic and override
//! only [`ChatProvider::endpoint_url`].
//!
//! # Parity gap: provider-routed model ids are not remapped to `providerId`
//!
//! LiteLLM's `transform_request` performs a provider-mapping lookup
//! (`_fetch_inference_provider_mapping(model_id)[provider]["providerId"]`) so
//! the upstream body model becomes the provider-specific id the HF Router
//! expects (e.g. `meta-llama/Llama-3-8B-Instruct` rather than the raw
//! `together/meta-llama/...` form). This pure-transform layer cannot do that
//! lookup — it would need the bundled `huggingface_llms_metadata` mapping
//! table that LiteLLM ships — so the model id is forwarded with only the
//! `huggingface/` prefix stripped. **Provider-routed HF deployments**
//! (`together/...`, `novita/...`, `fireworks-ai/...`, `hf-inference/...`)
//! **may therefore 404 or route incorrectly** when the model id does not
//! match the expected `providerId`. Operators who need provider routing
//! should set `api_base` to a specific inference endpoint URL (bypassing
//! router routing), or use the default `hf-inference` provider whose model
//! id passes through unchanged.

use std::collections::HashMap;

use crate::openai_compat::OpenAiCompatProvider;
use oxllm_core::Result;
use oxllm_core::deployment::Deployment;
use oxllm_core::endpoint::Endpoint;
use oxllm_core::provider::{ChatProvider, Provider, ProviderId, ResolvedCredentials};
use serde_json::Value;

/// The default Hugging Face router base URL (no trailing slash).
///
/// This is the base LiteLLM's `HuggingFaceChatConfig` uses for default routing
/// (`router.huggingface.co`, not the legacy `api-inference.huggingface.co`).
pub const DEFAULT_HUGGINGFACE_BASE: &str = "https://router.huggingface.co";

/// The provider prefix stripped from model ids (`huggingface/<model>`).
const STRIP: &str = "huggingface";

/// The shared OpenAI-compatible engine backing the body transform, stream
/// parsing, auth and supported-params. Only [`ChatProvider::endpoint_url`] is
/// overridden by [`HuggingFaceProvider`].
const COMPAT: OpenAiCompatProvider = OpenAiCompatProvider::new(
    "huggingface",
    DEFAULT_HUGGINGFACE_BASE,
    STRIP,
    false,
    true,
    true,
);

/// The Hugging Face provider.
#[derive(Debug, Clone, Copy)]
pub struct HuggingFaceProvider;

/// Configured Hugging Face provider handle.
pub const PROVIDER: HuggingFaceProvider = HuggingFaceProvider;

/// Strip the `huggingface/` provider prefix from a model id. Kept as a free
/// function for backward compatibility with tests.
pub fn strip_huggingface_prefix(model: &str) -> &str {
    OpenAiCompatProvider::strip_one_segment(model, STRIP)
}

/// Append `/v1/chat/completions` to a base URL, matching Hugging Face's
/// `_build_chat_completion_url` helper from LiteLLM.
///
/// - A URL already ending in `/chat/completions` is returned unchanged.
/// - A URL ending in `/v1` gets `/chat/completions` appended.
/// - Anything else gets `/v1/chat/completions` appended.
pub fn build_chat_completion_url(url: &str) -> String {
    let mut s = url.trim_end_matches('/').to_string();
    if s.ends_with("/v1") {
        s.push_str("/chat/completions");
    }
    if !s.ends_with("/chat/completions") {
        s.push_str("/v1/chat/completions");
    }
    s
}

/// Build the router URL for a model id whose `huggingface/` prefix has already
/// been stripped and which is not a full URL.
///
/// A leading segment followed by another `/` is a provider route (e.g.
/// `together/meta-llama/Llama-3-8B-Instruct`); anything else uses the default
/// `/v1/chat/completions` path.
fn route_url(model: &str) -> String {
    match model.split_once('/') {
        Some((provider, remaining)) if remaining.contains('/') => {
            let route = match provider {
                "hf-inference" => format!("{provider}/models/{remaining}/v1/chat/completions"),
                "novita" => format!("{provider}/v3/openai/chat/completions"),
                "fireworks-ai" => format!("{provider}/inference/v1/chat/completions"),
                _ => format!("{provider}/v1/chat/completions"),
            };
            format!("{DEFAULT_HUGGINGFACE_BASE}/{route}")
        }
        _ => format!("{DEFAULT_HUGGINGFACE_BASE}/v1/chat/completions"),
    }
}

impl ChatProvider for HuggingFaceProvider {
    fn endpoint_url(&self, deployment: &Deployment) -> String {
        if let Some(api_base) = deployment.oxllm_params.api_base.as_deref() {
            return build_chat_completion_url(api_base);
        }
        let model = strip_huggingface_prefix(&deployment.oxllm_params.model);
        if model.starts_with("http://") || model.starts_with("https://") {
            return build_chat_completion_url(model);
        }
        route_url(model)
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

impl Provider for HuggingFaceProvider {
    fn id(&self) -> ProviderId {
        "huggingface"
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
