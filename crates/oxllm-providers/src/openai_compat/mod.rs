//! Shared implementation for OpenAI-compatible providers.
//!
//! A handful of upstreams (Groq, xAI, Together, Gemini's OpenAI shim, Ollama,
//! Mistral) speak the OpenAI Chat Completions wire format almost verbatim.
//! Each only differs in a few constants: the provider slug, the default base
//! URL, the model-prefix segment to strip, whether embeddings are offered,
//! whether a `supported_params` list is published, and whether auth is
//! required.
//!
//! [`OpenAiCompatProvider`] captures those constants and implements
//! [`Provider`], [`ChatProvider`], and (optionally) [`EmbeddingProvider`]
//! generically. A concrete provider is then just a `const`:
//!
//! ```ignore
//! pub const PROVIDER: OpenAiCompatProvider = OpenAiCompatProvider::new(
//! "groq", DEFAULT_GROQ_BASE, "groq", false, true, true,
//! );
//! ```

pub(crate) mod sse;

use std::collections::HashMap;

use oxllm_core::Result;
use oxllm_core::deployment::Deployment;
use oxllm_core::endpoint::Endpoint;
use oxllm_core::provider::{
    ChatProvider, EmbeddingProvider, Provider, ProviderId, ResolvedCredentials,
};
use serde_json::{Map, Value, json};

use crate::util::apply_bearer_auth;

pub(crate) use sse::parse_openai_stream_chunk;

/// The request parameters every compat provider preserves in
/// `transform_request`. Shared so the list cannot drift between providers.
pub const SUPPORTED_CHAT_PARAMS: &[&str] = &[
    // Core fields — must be listed so drop_params does not strip them before
    // transform_request re-reads the body (messages would otherwise vanish).
    "model",
    "messages",
    "temperature",
    "top_p",
    "top_k",
    "max_tokens",
    "max_completion_tokens",
    "stream",
    "stream_options",
    "stop",
    "n",
    "presence_penalty",
    "frequency_penalty",
    "user",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "response_format",
    "reasoning_effort",
    "seed",
    "logprobs",
    "top_logprobs",
    // Newer OpenAI params — kept here so drop_params does not silently strip
    // them when a client sends them to a compat upstream that accepts them.
    "logit_bias",
    "store",
    "metadata",
    "modalities",
    "audio",
    "service_tier",
];

/// The request parameters preserved by the embeddings transform.
pub const SUPPORTED_EMBEDDING_PARAMS: &[&str] =
    &["model", "input", "encoding_format", "dimensions", "user"];

/// A config-driven OpenAI-compatible provider.
///
/// All behavior (URL building, bearer auth, model-prefix strip, identity
/// request/response transforms, SSE parsing) is shared; only the `const`
/// fields differ between concrete providers.
#[derive(Debug, Clone, Copy)]
pub struct OpenAiCompatProvider {
    slug: &'static str,
    default_base: &'static str,
    strip_prefix: &'static str,
    supports_embeddings: bool,
    publish_supported_params: bool,
    requires_auth: bool,
    /// Provider-specific replacement for [`SUPPORTED_CHAT_PARAMS`]. When set
    /// (via [`with_supported_params`]) it is returned verbatim by
    /// [`ChatProvider::supported_params`], letting a provider narrow the
    /// shared list to the subset its upstream actually accepts. `None` falls
    /// back to the shared list gated by `publish_supported_params`.
    supported_params_override: Option<&'static [&'static str]>,
}

impl OpenAiCompatProvider {
    /// Construct a provider from its constant configuration.
    ///
    /// - `slug` — the registry/provider slug (e.g. `"groq"`).
    /// - `default_base` — the upstream base URL without a trailing slash.
    /// - `strip_prefix` — the leading model segment to drop (e.g. `"groq"` in
    ///   `groq/llama-3.1-70b`).
    /// - `supports_embeddings` — whether the provider also implements
    ///   [`EmbeddingProvider`].
    /// - `publish_supported_params` — whether `supported_params` returns the
    ///   shared list (true) or `None` (false, treat as passthrough).
    /// - `requires_auth` — whether `apply_auth` attaches a bearer token. Ollama
    ///   is the lone provider that passes `false` here.
    pub const fn new(
        slug: &'static str,
        default_base: &'static str,
        strip_prefix: &'static str,
        supports_embeddings: bool,
        publish_supported_params: bool,
        requires_auth: bool,
    ) -> Self {
        Self {
            slug,
            default_base,
            strip_prefix,
            supports_embeddings,
            publish_supported_params,
            requires_auth,
            supported_params_override: None,
        }
    }

    /// Override the shared [`SUPPORTED_CHAT_PARAMS`] list with a
    /// provider-specific subset. Use this when an upstream's OpenAI-compat
    /// surface rejects fields the shared list forwards (e.g. Gemini does not
    /// accept `logit_bias`, `store`, or `metadata`). Const-evaluable so it can
    /// chain off a `const` provider.
    pub const fn with_supported_params(mut self, params: &'static [&'static str]) -> Self {
        self.supported_params_override = Some(params);
        self
    }

    /// Strip the configured provider prefix from a model id. Only the first
    /// segment is dropped: `openrouter/openai/gpt-4o` is handled by OpenRouter
    /// (a full module), not here.
    fn strip_model<'a>(&self, model: &'a str) -> &'a str {
        match model.split_once('/') {
            Some((prefix, rest)) if prefix == self.strip_prefix => rest,
            _ => model,
        }
    }

    /// Strip a single named segment from a model id. Exposed so each provider
    /// module can keep its `strip_<slug>_prefix` helper for backward
    /// compatibility with existing tests.
    pub fn strip_one_segment<'a>(model: &'a str, segment: &str) -> &'a str {
        match model.split_once('/') {
            Some((prefix, rest)) if prefix == segment => rest,
            _ => model,
        }
    }

    /// Resolve the model id to send upstream from a configured deployment.
    ///
    /// For non-wildcard deployments the configured (prefix-stripped) model id
    /// always wins — it overrides any model the client may have sent. For
    /// wildcard deployments (`*` or `<provider>/*`), the actual model id is
    /// taken from the request body's `model` field (after stripping the
    /// provider prefix) so a `together_ai/*` deployment routes whatever the
    /// client sent; if the request carries no `model`, the wildcard literal is
    /// forwarded as-is.
    fn resolve_model<'a>(&self, request: &'a Value, deployment: &'a Deployment) -> &'a str {
        let configured = self.strip_model(&deployment.oxllm_params.model);
        let is_wildcard = configured.contains('*');
        if is_wildcard {
            request
                .get("model")
                .and_then(Value::as_str)
                .map(|m| self.strip_model(m))
                .unwrap_or(configured)
        } else {
            configured
        }
    }

    fn chat_base<'a>(&self, deployment: &'a Deployment) -> &'a str {
        deployment
            .oxllm_params
            .api_base
            .as_deref()
            .unwrap_or(self.default_base)
            .trim_end_matches('/')
    }

    fn chat_endpoint(&self, deployment: &Deployment, model: &str) -> String {
        let base = self.chat_base(deployment);
        if self.slug == "bedrock_mantle"
            && deployment.oxllm_params.api_base.is_none()
            && (model.starts_with("openai.gpt-5")
                || model.starts_with("openai.gpt-6")
                || model.starts_with("google.gemma-4"))
        {
            return format!(
                "{}/openai/v1/chat/completions",
                base.trim_end_matches("/v1")
            );
        }
        crate::util::append_operation_path(base, "chat/completions")
    }
}

/// Gemini 3.7 Flash and 3.8 Flash reject a thinking level of `minimal`.
/// Callers still send `reasoning_effort: "minimal"`; rewrite it to `low`
/// before the OpenAI-compatible Gemini surface maps it.
pub(crate) fn relax_gemini_unsupported_minimal_effort(body: &mut Value, model: &str) {
    if !gemini_flash_rejects_minimal(model) {
        return;
    }
    let Some(obj) = body.as_object_mut() else {
        return;
    };
    let shape = match obj.get("reasoning_effort") {
        Some(Value::String(level)) if level.eq_ignore_ascii_case("minimal") => 1,
        Some(Value::Object(dict))
            if dict
                .get("effort")
                .and_then(Value::as_str)
                .is_some_and(|level| level.eq_ignore_ascii_case("minimal")) =>
        {
            2
        }
        _ => 0,
    };
    match shape {
        1 => {
            obj.insert(
                "reasoning_effort".to_string(),
                Value::String("low".to_string()),
            );
        }
        2 => {
            if let Some(Value::Object(dict)) = obj.get_mut("reasoning_effort") {
                dict.insert("effort".to_string(), Value::String("low".to_string()));
            }
        }
        _ => {}
    }
}

/// Gemini signatures belong to its own provider fields and tool IDs, never to
/// Anthropic thinking blocks carried across a mixed-provider conversation.
pub(crate) fn drop_foreign_thinking_blocks(body: &mut Value) {
    if let Some(messages) = body.get_mut("messages").and_then(Value::as_array_mut) {
        for message in messages {
            if let Some(message) = message.as_object_mut() {
                message.remove("thinking_blocks");
            }
        }
    }
}

fn gemini_flash_rejects_minimal(model: &str) -> bool {
    let lower = model.to_ascii_lowercase();
    let leaf = lower.rsplit('/').next().unwrap_or(lower.as_str());
    leaf.contains("gemini-3.7-flash") || leaf.contains("gemini-3.8-flash")
}

/// Build a chat request body using an explicit allow-list of param names.
/// Providers with a narrower surface (e.g. Gemini via
/// [`OpenAiCompatProvider::with_supported_params`]) pass their override so
/// OpenAI-only fields are never forwarded upstream.
pub(crate) fn build_chat_request_with_params(
    request: &Value,
    model: &str,
    params: &[&str],
) -> Value {
    let mut out = Map::new();
    if let Some(messages) = request.get("messages").filter(|v| !v.is_null()) {
        out.insert("messages".to_string(), messages.clone());
    }
    for key in params {
        // `model` is always taken from the deployment (after prefix strip),
        // never from the client body — the client may send an alias like
        // `grok` while upstream needs `grok-3`.
        if *key == "model" || *key == "messages" {
            continue;
        }
        if let Some(v) = request.get(key) {
            if !v.is_null() {
                out.insert(key.to_string(), v.clone());
            }
        }
    }
    out.insert("model".to_string(), Value::String(model.to_string()));
    Value::Object(out)
}

pub(crate) fn build_embedding_request(request: &Value, model: &str) -> Value {
    let mut out = Map::new();
    if let Some(v) = request.get("input").filter(|v| !v.is_null()) {
        out.insert("input".to_string(), v.clone());
    }
    for key in SUPPORTED_EMBEDDING_PARAMS {
        if *key == "model" || *key == "input" {
            continue;
        }
        if let Some(v) = request.get(key) {
            if !v.is_null() {
                out.insert(key.to_string(), v.clone());
            }
        }
    }
    out.insert("model".to_string(), Value::String(model.to_string()));
    Value::Object(out)
}

/// OpenRouter and other OpenAI-compatible providers emit `finish_reason:
/// "error"` when a backing provider fails mid-stream. That is not a valid
/// OpenAI finish reason; map it to `stop`.
pub(crate) fn map_generic_error_finish_reason(body: &mut Value) {
    let Some(choices) = body.get_mut("choices").and_then(Value::as_array_mut) else {
        return;
    };
    for choice in choices {
        if choice.get("finish_reason").and_then(Value::as_str) == Some("error") {
            choice["finish_reason"] = json!("stop");
        }
    }
}

impl ChatProvider for OpenAiCompatProvider {
    fn endpoint_url(&self, deployment: &Deployment) -> String {
        self.chat_endpoint(deployment, self.strip_model(&deployment.oxllm_params.model))
    }

    fn endpoint_url_for_request(&self, deployment: &Deployment, request: &Value) -> String {
        self.chat_endpoint(deployment, self.resolve_model(request, deployment))
    }

    fn apply_auth(
        &self,
        deployment: &Deployment,
        creds: &ResolvedCredentials,
        headers: &mut HashMap<String, String>,
    ) {
        if self.requires_auth {
            apply_bearer_auth(creds, headers);
            oxllm_core::provider::apply_auth_overrides(deployment, creds, headers);
        }
    }

    fn transform_request(&self, request: &Value, deployment: &Deployment) -> Result<Value> {
        let model = self.resolve_model(request, deployment);
        let params = self
            .supported_params_override
            .unwrap_or(SUPPORTED_CHAT_PARAMS);
        let mut out = build_chat_request_with_params(request, model, params);
        // Mantle and Databricks-hosted OpenAI models reject the same sampling
        // params, and the gpt-5.6 family rejects function tools unless
        // reasoning is explicitly off. Other compat hosts keep the caller's
        // params.
        if self.slug == "bedrock_mantle" || self.slug == "databricks" {
            crate::openai::apply_openai_reasoning_limits(&mut out, model);
        }
        if self.slug == "gemini" {
            relax_gemini_unsupported_minimal_effort(&mut out, model);
            drop_foreign_thinking_blocks(&mut out);
        }
        crate::prompt_cache::apply(&mut out, request, deployment);
        Ok(out)
    }

    fn transform_response(&self, body: &Value, _model: &str) -> Result<Value> {
        let mut out = body.clone();
        crate::openai::sanitize_null_tokens(&mut out);
        map_generic_error_finish_reason(&mut out);
        Ok(out)
    }

    fn transform_stream_chunk(&self, chunk: &str) -> Result<Option<Value>> {
        let mut value = match parse_openai_stream_chunk(chunk)? {
            Some(v) => v,
            None => return Ok(None),
        };
        crate::openai::sanitize_null_tokens(&mut value);
        map_generic_error_finish_reason(&mut value);
        Ok(Some(value))
    }

    fn supported_params(&self) -> Option<Vec<&str>> {
        if let Some(override_list) = self.supported_params_override {
            return Some(override_list.to_vec());
        }
        if self.publish_supported_params {
            let mut params = SUPPORTED_CHAT_PARAMS.to_vec();
            if matches!(self.slug, "bedrock_mantle" | "azure_ai") {
                params.extend(["cache_control_injection_points", "prompt_cache_options"]);
            }
            Some(params)
        } else {
            None
        }
    }
}

impl EmbeddingProvider for OpenAiCompatProvider {
    fn embedding_url(&self, deployment: &Deployment) -> String {
        // A caller who configures `api_base` as the full embeddings endpoint
        // must not get `/embeddings` appended twice. Query and fragment stay.
        crate::util::append_operation_path(self.chat_base(deployment), "embeddings")
    }

    fn apply_auth(
        &self,
        deployment: &Deployment,
        creds: &ResolvedCredentials,
        headers: &mut HashMap<String, String>,
    ) {
        ChatProvider::apply_auth(self, deployment, creds, headers);
    }

    fn transform_request(&self, request: &Value, deployment: &Deployment) -> Result<Value> {
        let model = self.resolve_model(request, deployment);
        Ok(build_embedding_request(request, model))
    }

    fn transform_response(&self, body: &Value, _model: &str) -> Result<Value> {
        let mut out = body.clone();
        crate::openai::sanitize_null_tokens(&mut out);
        Ok(out)
    }
}

impl Provider for OpenAiCompatProvider {
    fn id(&self) -> ProviderId {
        self.slug
    }

    fn supports(&self, endpoint: Endpoint) -> bool {
        matches!(endpoint, Endpoint::ChatCompletions)
            || (self.supports_embeddings && matches!(endpoint, Endpoint::Embeddings))
    }

    fn chat(&self) -> Option<&dyn ChatProvider> {
        Some(self)
    }

    fn embeddings(&self) -> Option<&dyn EmbeddingProvider> {
        if self.supports_embeddings {
            Some(self)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests;
