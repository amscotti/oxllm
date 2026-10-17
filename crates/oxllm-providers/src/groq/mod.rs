//! Groq provider — uses Groq's OpenAI-compatible Chat Completions endpoint.
//!
//! Wire target: `https://api.groq.com/openai/v1/chat/completions`.
//!
//! Groq exposes an OpenAI-compatible surface, so URL building, auth, stream
//! parsing and the supported-params list are delegated to the shared
//! [`OpenAiCompatProvider`] engine. Groq diverges on two points (parity with
//! the reference `GroqChatConfig`):
//!
//! - **`delta.reasoning` → `delta.reasoning_content`**: Groq emits reasoning
//!   content under the non-standard `reasoning` key on streaming deltas; it is
//!   normalized to `reasoning_content` so downstream
//!   consumers (and oxllm's own response handling) find it.
//! - **`null` fields stripped from assistant messages**: Groq rejects
//!   `null`-valued fields on assistant messages (e.g. a round-tripped
//!   `function_call: null`, reference issue #5839); null-valued keys are removed
//!   from assistant messages before forwarding.
//!
//! See `docs/adding-a-provider.md`.

use std::collections::HashMap;

use crate::openai_compat::OpenAiCompatProvider;
use oxllm_core::Result;
use oxllm_core::deployment::Deployment;
use oxllm_core::endpoint::Endpoint;
use oxllm_core::provider::{ChatProvider, Provider, ProviderId, ResolvedCredentials};
use serde_json::{Value, json};

/// The default Groq OpenAI-compatible base URL.
pub const DEFAULT_GROQ_BASE: &str = "https://api.groq.com/openai/v1";

/// The provider prefix stripped from model ids (`groq/<model>`).
const STRIP: &str = "groq";

/// The shared OpenAI-compatible engine backing URL building, bearer auth,
/// stream parsing and the supported-params list. Only the request and stream
/// transforms are overridden on the wrapper to apply Groq's divergences.
const COMPAT: OpenAiCompatProvider =
    OpenAiCompatProvider::new("groq", DEFAULT_GROQ_BASE, STRIP, false, true, true);

/// The Groq provider. Supports chat completions only.
#[derive(Debug, Clone, Copy)]
pub struct GroqProvider;

/// Configured Groq provider handle.
pub const PROVIDER: GroqProvider = GroqProvider;

/// Strip the `groq/` provider prefix from a model id. Kept as a free function
/// for backward compatibility with existing tests.
pub fn strip_groq_prefix(model: &str) -> &str {
    OpenAiCompatProvider::strip_one_segment(model, STRIP)
}

/// Apply Groq-specific request transforms to an already-built OpenAI chat
/// request body (the output of the shared compat `transform_request`).
fn apply_groq_request_transforms(out: &mut Value) {
    strip_null_fields_from_assistant_messages(out);
    // Groq's message schema is strict: response-only fields the gateway
    // attached to a previous reply (thinking blocks, provider extension
    // fields) cannot be replayed (reference
    // `strip_litellm_response_only_fields`).
    crate::openai_compat::strip_response_only_fields(out);
}

/// Remove keys with `null` values from assistant messages (parity with
/// The reference Groq message transform, issue #5839: Groq rejects
/// `function_call: null`). The top-level `build_chat_request` already drops
/// `null` top-level params, but nested message fields are preserved — this
/// closes that gap for assistant messages.
fn strip_null_fields_from_assistant_messages(out: &mut Value) {
    let Some(messages) = out.get_mut("messages").and_then(|v| v.as_array_mut()) else {
        return;
    };
    for message in messages.iter_mut() {
        let Some(map) = message.as_object_mut() else {
            continue;
        };
        let is_assistant = map
            .get("role")
            .and_then(|r| r.as_str())
            .is_some_and(|r| r == "assistant");
        if !is_assistant {
            continue;
        }
        let null_keys: Vec<String> = map
            .iter()
            .filter(|(_, v)| v.is_null())
            .map(|(k, _)| k.clone())
            .collect();
        for key in null_keys {
            map.remove(&key);
        }
    }
}

/// Apply Groq-specific stream-chunk transforms to an already-parsed OpenAI
/// stream chunk (the output of the shared compat `transform_stream_chunk`).
fn apply_groq_stream_transforms(value: &mut Value) {
    let Some(choices) = value.get_mut("choices").and_then(|c| c.as_array_mut()) else {
        return;
    };
    for choice in choices.iter_mut() {
        let Some(delta) = choice.get_mut("delta").and_then(|d| d.as_object_mut()) else {
            continue;
        };
        // Prefer an existing normalized trace. Only a non-empty alternate
        // text value can fill an absent, null, or empty reasoning_content.
        if let Some(reasoning) = delta.remove("reasoning") {
            let has_reasoning = delta
                .get("reasoning_content")
                .is_some_and(|value| !value.is_null() && value.as_str() != Some(""));
            if !has_reasoning && reasoning.as_str().is_some_and(|text| !text.is_empty()) {
                delta.insert("reasoning_content".to_string(), reasoning);
            }
        }
    }
}

/// Derive hosted server-tool usage from Groq's `executed_tools` actions.
///
/// Groq's OpenAI-compatible usage object carries no web-search counter;
/// instead the assistant message lists the executed compound-tool actions
/// (parity with the reference `GroqChatConfig._add_web_search_usage`).
/// `browser.search`/`browser_search` actions become
/// `usage.server_tool_use.web_search_requests` — the same bucket the
/// per-query search rate bills — and `browser.open` actions are mirrored as
/// `browser_open_requests` for shape parity. Anything else is ignored.
fn add_web_search_usage(out: &mut Value) {
    let Some(choices) = out.get("choices").and_then(|c| c.as_array()) else {
        return;
    };
    let mut searches = 0u64;
    let mut opens = 0u64;
    for choice in choices {
        let Some(executed) = choice
            .pointer("/message/executed_tools")
            .and_then(Value::as_array)
        else {
            continue;
        };
        for tool in executed {
            let action = tool
                .get("name")
                .and_then(Value::as_str)
                .or_else(|| tool.get("type").and_then(Value::as_str));
            match action {
                Some("browser.search") | Some("browser_search") => searches += 1,
                Some("browser.open") => opens += 1,
                _ => {}
            }
        }
    }
    if searches == 0 && opens == 0 {
        return;
    }
    if let Some(usage) = out.get_mut("usage").and_then(Value::as_object_mut) {
        usage.insert(
            "server_tool_use".to_string(),
            json!({
                "web_search_requests": searches,
                "browser_open_requests": opens,
            }),
        );
    }
}

impl ChatProvider for GroqProvider {
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
        apply_groq_request_transforms(&mut out);
        Ok(out)
    }

    fn transform_response(&self, body: &Value, model: &str) -> Result<Value> {
        let mut out = ChatProvider::transform_response(&COMPAT, body, model)?;
        add_web_search_usage(&mut out);
        Ok(out)
    }

    fn transform_stream_chunk(&self, chunk: &str) -> Result<Option<Value>> {
        let mut value = match ChatProvider::transform_stream_chunk(&COMPAT, chunk)? {
            Some(v) => v,
            None => return Ok(None),
        };
        apply_groq_stream_transforms(&mut value);
        Ok(Some(value))
    }

    fn supported_params(&self) -> Option<Vec<&str>> {
        ChatProvider::supported_params(&COMPAT)
    }
}

impl Provider for GroqProvider {
    fn id(&self) -> ProviderId {
        "groq"
    }

    fn supports(&self, endpoint: Endpoint) -> bool {
        matches!(
            endpoint,
            Endpoint::ChatCompletions | Endpoint::AudioSpeech | Endpoint::AudioTranscriptions
        )
    }

    fn chat(&self) -> Option<&dyn ChatProvider> {
        Some(self)
    }
}

#[cfg(test)]
mod tests;
