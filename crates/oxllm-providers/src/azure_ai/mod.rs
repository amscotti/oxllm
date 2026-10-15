//! Azure AI Foundry chat and embeddings. Claude uses native Anthropic messages;
//! Cohere image embeddings use a separate route from OpenAI-shaped text inputs.
//! Explicit resource bases, endpoint queries and deployment names are retained.

use std::collections::HashMap;

use oxllm_core::provider::{
    ChatProvider, EmbeddingProvider, EmbeddingRequestPlan, Provider, ProviderId,
    ResolvedCredentials,
};
use oxllm_core::{Deployment, Endpoint, Error, Result};
use serde_json::{Value, json};

use crate::anthropic::AnthropicProvider;
use crate::openai_compat::OpenAiCompatProvider;

mod embeddings;

const COMPAT: OpenAiCompatProvider = OpenAiCompatProvider::new(
    "azure_ai",
    "https://azure-ai-api-base-not-configured.invalid",
    "azure_ai",
    true,
    true,
    true,
);

/// Azure AI Foundry adapter for vendor chat and embeddings.
#[derive(Debug, Clone, Default)]
pub struct AzureAiProvider;

impl AzureAiProvider {
    /// Foundry's native Anthropic Count Tokens endpoint.
    pub fn count_tokens_url(deployment: &Deployment) -> Result<String> {
        if !Self::is_anthropic(deployment) {
            return Err(Error::BadRequest(
                "Foundry token counting requires a Claude deployment".into(),
            ));
        }
        Ok(crate::util::foundry_endpoint_url(
            deployment,
            "messages/count_tokens",
            true,
        ))
    }
    fn resolved(request: &Value, deployment: &Deployment) -> Result<Deployment> {
        if deployment
            .oxllm_params
            .api_base
            .as_deref()
            .is_none_or(|base| base.trim().is_empty())
        {
            return Err(Error::Config(
                "azure_ai deployment requires api_base".into(),
            ));
        }
        let configured = deployment
            .oxllm_params
            .model
            .strip_prefix("azure_ai/")
            .unwrap_or(&deployment.oxllm_params.model);
        let model = if configured.contains('*') {
            request
                .get("model")
                .and_then(Value::as_str)
                .map(|model| model.strip_prefix("azure_ai/").unwrap_or(model))
                .ok_or_else(|| {
                    Error::BadRequest("wildcard azure_ai deployment requires a model".into())
                })?
        } else {
            configured
        };
        let model = model
            .strip_prefix("model_router/")
            .or_else(|| model.strip_prefix("model-router/"))
            .unwrap_or(model);
        if model.is_empty() || model.contains('*') {
            return Err(Error::BadRequest(
                "azure_ai requires a concrete deployment/model name".into(),
            ));
        }
        let mut resolved = deployment.clone();
        resolved.oxllm_params.model = format!("azure_ai/{model}");
        Ok(resolved)
    }

    /// Whether this deployment uses Foundry's native Anthropic Messages route.
    pub fn is_anthropic(deployment: &Deployment) -> bool {
        let model = deployment
            .oxllm_params
            .model
            .strip_prefix("azure_ai/")
            .unwrap_or(&deployment.oxllm_params.model);
        let base_model = deployment
            .model_info
            .as_ref()
            .and_then(|info| info.base_model.as_deref())
            .or_else(|| {
                deployment
                    .oxllm_params
                    .extra
                    .get("base_model")
                    .and_then(Value::as_str)
            });
        let path = deployment
            .oxllm_params
            .api_base
            .as_deref()
            .unwrap_or("")
            .split(['?', '#'])
            .next()
            .unwrap_or("");
        let path = path
            .split_once("://")
            .and_then(|(_, rest)| rest.find('/').map(|index| &rest[index..]))
            .unwrap_or("");
        model.starts_with("claude-")
            || model.starts_with("anthropic/")
            || base_model.is_some_and(|model| model.contains("claude-"))
            || path.split('/').any(|segment| segment == "anthropic")
    }

    fn anthropic_deployment(deployment: &Deployment) -> Deployment {
        let mut native = deployment.clone();
        let model = deployment
            .oxllm_params
            .model
            .strip_prefix("azure_ai/")
            .unwrap_or(&deployment.oxllm_params.model);
        native.oxllm_params.model = if model.starts_with("anthropic/") {
            model.to_string()
        } else {
            format!("anthropic/{model}")
        };
        native
    }

    fn auth(
        deployment: &Deployment,
        creds: &ResolvedCredentials,
        headers: &mut HashMap<String, String>,
    ) {
        if Self::is_anthropic(deployment) {
            headers.insert(
                "anthropic-version".into(),
                crate::anthropic::ANTHROPIC_VERSION.into(),
            );
        }
        if let Some(token) = creds.best_effort_bearer() {
            if matches!(creds, ResolvedCredentials::AzureAd { .. }) {
                headers.insert("authorization".into(), format!("Bearer {token}"));
            } else if Self::is_anthropic(deployment) {
                headers.insert("x-api-key".into(), token.to_string());
            } else if deployment
                .oxllm_params
                .api_base
                .as_deref()
                .is_some_and(crate::util::is_foundry_base)
            {
                headers.insert("api-key".into(), token.to_string());
            } else {
                headers.insert("authorization".into(), format!("Bearer {token}"));
            }
        }
        oxllm_core::provider::apply_auth_overrides(deployment, creds, headers);
    }

    fn anthropic_chunk(chunk: &str) -> bool {
        crate::util::extract_sse_data(chunk)
            .and_then(|data| serde_json::from_str::<Value>(&data).ok())
            .and_then(|value| value.get("type").and_then(Value::as_str).map(String::from))
            .is_some_and(|kind| {
                kind.starts_with("message_")
                    || kind.starts_with("content_block_")
                    || matches!(kind.as_str(), "ping" | "error")
            })
    }
}

impl ChatProvider for AzureAiProvider {
    fn endpoint_url(&self, deployment: &Deployment) -> String {
        let native = Self::is_anthropic(deployment);
        crate::util::foundry_endpoint_url(
            deployment,
            if native {
                "messages"
            } else {
                "chat/completions"
            },
            native,
        )
    }

    fn endpoint_url_for_request(&self, deployment: &Deployment, request: &Value) -> String {
        self.endpoint_url(
            &Self::resolved(request, deployment).unwrap_or_else(|_| deployment.clone()),
        )
    }

    fn apply_auth(
        &self,
        deployment: &Deployment,
        creds: &ResolvedCredentials,
        headers: &mut HashMap<String, String>,
    ) {
        Self::auth(deployment, creds, headers);
    }

    fn transform_request(&self, request: &Value, deployment: &Deployment) -> Result<Value> {
        let resolved = Self::resolved(request, deployment)?;
        if Self::is_anthropic(&resolved) {
            AnthropicProvider.transform_request(request, &Self::anthropic_deployment(&resolved))
        } else {
            // Sanitize before the compat transform places native cache marks,
            // so incompatible controls cannot consume its breakpoint budget.
            let mut sanitized = request.clone();
            if let Some(messages) = sanitized.get_mut("messages").and_then(Value::as_array_mut) {
                for message in messages {
                    if let Some(object) = message.as_object_mut() {
                        for field in [
                            "thinking_blocks",
                            "reasoning_content",
                            "provider_specific_fields",
                            "cache_control",
                        ] {
                            object.remove(field);
                        }
                        if let Some(content) =
                            object.get_mut("content").and_then(Value::as_array_mut)
                        {
                            for block in content {
                                if let Some(block) = block.as_object_mut() {
                                    block.remove("cache_control");
                                }
                            }
                        }
                        if let Some(content) = object.get("content").and_then(Value::as_array)
                            && !content.is_empty()
                            && content.iter().all(|block| {
                                block.get("type").and_then(Value::as_str) == Some("text")
                                    && block
                                        .get("prompt_cache_breakpoint")
                                        .is_none_or(Value::is_null)
                            })
                        {
                            let text = content
                                .iter()
                                .filter_map(|block| block.get("text").and_then(Value::as_str))
                                .collect::<Vec<_>>()
                                .join("\n");
                            object.insert("content".into(), json!(text));
                        }
                    }
                }
            }
            if let Some(tools) = sanitized.get_mut("tools").and_then(Value::as_array_mut) {
                for tool in tools {
                    if let Some(tool) = tool.as_object_mut() {
                        tool.remove("cache_control");
                        if let Some(function) =
                            tool.get_mut("function").and_then(Value::as_object_mut)
                        {
                            function.remove("cache_control");
                        }
                    }
                }
            }
            ChatProvider::transform_request(&COMPAT, &sanitized, &resolved)
        }
    }

    fn transform_response(&self, body: &Value, model: &str) -> Result<Value> {
        if body.get("type").and_then(Value::as_str) == Some("message") {
            AnthropicProvider.transform_response(body, model)
        } else {
            ChatProvider::transform_response(&COMPAT, body, model)
        }
    }

    fn transform_response_with_request(
        &self,
        body: &Value,
        model: &str,
        request: &Value,
    ) -> Result<Value> {
        if body.get("type").and_then(Value::as_str) == Some("message") {
            AnthropicProvider.transform_response_with_request(body, model, request)
        } else {
            ChatProvider::transform_response(self, body, model)
        }
    }

    fn transform_stream_chunk(&self, chunk: &str) -> Result<Option<Value>> {
        if Self::anthropic_chunk(chunk) {
            AnthropicProvider.transform_stream_chunk(chunk)
        } else {
            COMPAT.transform_stream_chunk(chunk)
        }
    }

    fn transform_stream_chunk_stateful(
        &self,
        chunk: &str,
        state: &mut Option<Box<dyn std::any::Any + Send + Sync>>,
        request: &Value,
    ) -> Result<Option<Value>> {
        if Self::anthropic_chunk(chunk) {
            AnthropicProvider.transform_stream_chunk_stateful(chunk, state, request)
        } else {
            COMPAT.transform_stream_chunk_stateful(chunk, state, request)
        }
    }

    fn supported_params(&self) -> Option<Vec<&str>> {
        let mut params = COMPAT.supported_params().unwrap_or_default();
        for param in AnthropicProvider.supported_params().unwrap_or_default() {
            if !params.contains(&param) {
                params.push(param);
            }
        }
        Some(params)
    }
}

impl EmbeddingProvider for AzureAiProvider {
    fn prepare_requests(
        &self,
        request: Value,
        deployment: &Deployment,
    ) -> Result<EmbeddingRequestPlan> {
        let resolved = Self::resolved(&request, deployment)?;
        embeddings::prepare_requests(request, &resolved)
    }

    fn embedding_url(&self, deployment: &Deployment) -> String {
        crate::util::foundry_endpoint_url(deployment, "embeddings", false)
    }

    fn embedding_url_for_request(&self, deployment: &Deployment, request: &Value) -> String {
        let Ok(resolved) = Self::resolved(request, deployment) else {
            return self.embedding_url(deployment);
        };
        if embeddings::is_image_request(request, &resolved) {
            embeddings::image_url(&resolved)
        } else {
            self.embedding_url(&resolved)
        }
    }

    fn apply_auth(
        &self,
        deployment: &Deployment,
        creds: &ResolvedCredentials,
        headers: &mut HashMap<String, String>,
    ) {
        Self::auth(deployment, creds, headers);
    }

    fn transform_request(&self, request: &Value, deployment: &Deployment) -> Result<Value> {
        let resolved = Self::resolved(request, deployment)?;
        let request = embeddings::normalize_input(request);
        if embeddings::is_image_request(&request, &resolved) {
            return Ok(embeddings::image_body(&request, &resolved));
        }
        if embeddings::has_image_input(&request, &resolved) {
            return Err(Error::BadRequest(
                "mixed image/text embeddings require separate request batches".into(),
            ));
        }
        EmbeddingProvider::transform_request(&COMPAT, &request, &resolved)
    }

    fn transform_response(&self, body: &Value, model: &str) -> Result<Value> {
        EmbeddingProvider::transform_response(&COMPAT, body, model)
    }
}

impl Provider for AzureAiProvider {
    fn id(&self) -> ProviderId {
        "azure_ai"
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
