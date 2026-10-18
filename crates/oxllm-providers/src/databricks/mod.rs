//! Databricks provider — uses Databricks' OpenAI-compatible serving endpoints.
//!
//! Wire targets:
//! - chat: `<api_base>/chat/completions`
//! - embeddings: `<api_base>/embeddings`
//!
//! Because Databricks exposes an OpenAI-compatible surface, URL building,
//! auth, stream parsing and the supported-params list are delegated to the
//! shared [`OpenAiCompatProvider`] engine. Databricks diverges on these
//! request-shape points (parity with the reference Databricks transform):
//!
//! - **Response-only message fields stripped**: Databricks rejects replayed
//!   gateway extension fields (`thinking_blocks`, `reasoning_content`,
//!   `reasoning_items`, `provider_specific_fields`) on messages; they are
//!   removed from every message before forwarding (reference
//!   `strip_litellm_internal_message_fields`).
//! - **Bare assistant turns dropped**: after stripping, assistant messages
//!   without content or tool/function calls are omitted. Whitespace-only
//!   text counts as no content (Databricks rejects empty text blocks);
//!   a `refusal` or `audio` payload keeps the turn — silently deleting a
//!   replayed refusal corrupts history worse than an upstream 400 would.
//!   Other roles and surviving message payloads are left intact.
//! - **Empty text blocks filtered**: whitespace-only text blocks are
//!   removed from array content (reference `_sanitize_empty_content`).
//! - **Message-level `cache_control` relocated**: with string content it
//!   moves into a wrapping text block (reference
//!   `_move_cache_control_into_string_content_block`).
//! - **Non-Claude models**: a lone `{"type":"text",...}` block collapses
//!   to a plain string, and parallel tool calls are split so every
//!   `tool` result is immediately preceded by an assistant turn carrying
//!   its call (reference `_collapse_single_text_block` /
//!   `_split_parallel_tool_calls`; Databricks 400s otherwise).
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

    let is_claude = out
        .get("model")
        .and_then(Value::as_str)
        .is_some_and(|model| model.contains("claude"));
    let Some(messages) = out.get_mut("messages").and_then(Value::as_array_mut) else {
        return;
    };
    for message in messages.iter_mut() {
        move_cache_control_into_string_content(message);
        filter_empty_text_blocks(message);
    }
    // A thinking-only turn is now bare, which Databricks also rejects. Keep
    // this cleanup local: stripping metadata is not universal compat behavior.
    messages.retain(|message| !is_bare_assistant_message(message));
    if !is_claude {
        for message in messages.iter_mut() {
            collapse_single_text_block(message);
        }
        *messages = split_parallel_tool_calls(std::mem::take(messages));
    }
}

/// Reference `_move_cache_control_into_string_content_block`: Anthropic
/// prompt caching reads `cache_control` on content blocks, so a
/// message-level marker on string content wraps the string in one block.
fn move_cache_control_into_string_content(message: &mut Value) {
    let Some(obj) = message.as_object_mut() else {
        return;
    };
    if !matches!(obj.get("content"), Some(Value::String(_))) {
        return;
    }
    let Some(cache_control) = obj.remove("cache_control") else {
        return;
    };
    let Some(Value::String(text)) = obj.remove("content") else {
        return;
    };
    obj.insert(
        "content".to_string(),
        serde_json::json!([{ "type": "text", "text": text, "cache_control": cache_control }]),
    );
}

/// A text block carrying no visible text (reference empty-block filter).
/// Non-object and non-text blocks are payload, never filtered.
fn is_empty_text_block(block: &Value) -> bool {
    let Some(obj) = block.as_object() else {
        return false;
    };
    if obj.get("type").and_then(Value::as_str) != Some("text") {
        return false;
    }
    match obj.get("text") {
        None | Some(Value::Null) => true,
        Some(Value::String(text)) => text.trim().is_empty(),
        _ => false,
    }
}

/// Databricks rejects empty text blocks; drop them from array content.
/// When nothing survives, the `content` key is removed entirely
/// (reference `_sanitize_empty_content` list arm).
fn filter_empty_text_blocks(message: &mut Value) {
    let Some(obj) = message.as_object_mut() else {
        return;
    };
    let Some(Value::Array(parts)) = obj.get_mut("content") else {
        return;
    };
    if parts.is_empty() {
        return;
    }
    parts.retain(|block| !is_empty_text_block(block));
    if parts.is_empty() {
        obj.remove("content");
    }
}

/// Reference `_collapse_single_text_block`: non-Claude Databricks tooling
/// reads only string content, so a lone bare text block becomes a string.
fn collapse_single_text_block(message: &mut Value) {
    let Some(obj) = message.as_object_mut() else {
        return;
    };
    let text = match obj.get("content") {
        Some(Value::Array(parts)) if parts.len() == 1 => {
            let block = &parts[0];
            let is_bare_text = block.as_object().is_some_and(|block| {
                block.len() == 2
                    && block.get("type").and_then(Value::as_str) == Some("text")
                    && matches!(block.get("text"), Some(Value::String(_)))
            });
            if !is_bare_text {
                return;
            }
            block["text"].clone()
        }
        _ => return,
    };
    obj.insert("content".to_string(), text);
}

/// Reference `_split_parallel_tool_calls`: Databricks rejects a `tool`
/// message unless the message immediately before it carries `tool_calls`,
/// so one assistant turn with parallel calls followed by one result per
/// call 400s from the second result on. Re-emit each result right after
/// an assistant holding only its matching call. No-op unless the calls
/// and results line up exactly, so malformed history is never rewritten.
fn split_parallel_tool_calls(messages: Vec<Value>) -> Vec<Value> {
    fn role(message: &Value) -> Option<&str> {
        message.get("role").and_then(Value::as_str)
    }
    let mut out: Vec<Value> = Vec::with_capacity(messages.len());
    let mut iter = messages.into_iter().peekable();
    while let Some(message) = iter.next() {
        let calls: Vec<Value> = match (role(&message), message.get("tool_calls")) {
            (Some("assistant"), Some(Value::Array(calls))) if calls.len() >= 2 => calls.clone(),
            _ => {
                out.push(message);
                continue;
            }
        };
        // Gather the run of tool results that follows.
        let mut results: Vec<Value> = Vec::new();
        while iter.peek().is_some_and(|next| role(next) == Some("tool")) {
            results.push(iter.next().expect("peeked tool message"));
        }
        let call_ids: Vec<Option<String>> = calls
            .iter()
            .map(|call| call.get("id").and_then(Value::as_str).map(str::to_string))
            .collect();
        let result_ids: Vec<Option<String>> = results
            .iter()
            .map(|result| {
                result
                    .get("tool_call_id")
                    .and_then(Value::as_str)
                    .map(str::to_string)
            })
            .collect();
        let aligned = results.len() == calls.len()
            && call_ids.iter().all(|id| id.is_some())
            && result_ids.iter().all(|id| id.is_some() && call_ids.contains(id))
            && {
                let mut sorted_calls = call_ids.clone();
                let mut sorted_results = result_ids.clone();
                sorted_calls.sort_unstable();
                sorted_results.sort_unstable();
                sorted_calls == sorted_results
            };
        if !aligned {
            out.push(message);
            out.extend(results);
            continue;
        }
        for (position, result) in results.into_iter().enumerate() {
            let matched = result_ids[position]
                .as_ref()
                .and_then(|id| call_ids.iter().position(|call_id| call_id.as_ref() == Some(id)))
                .map(|index| calls[index].clone())
                .expect("aligned ids guarantee a match");
            if position == 0 {
                let mut first = message.clone();
                first["tool_calls"] = Value::Array(vec![matched]);
                out.push(first);
            } else {
                out.push(serde_json::json!({
                    "role": "assistant",
                    "tool_calls": [matched],
                }));
            }
            out.push(result);
        }
    }
    out
}

fn is_bare_assistant_message(message: &Value) -> bool {
    if message.get("role").and_then(Value::as_str) != Some("assistant") {
        return false;
    }

    // A refusal or audio payload is real assistant output: deleting the
    // turn would silently rewrite history. Only schema-valid empty shapes
    // count as absent; malformed non-null values keep the turn so invalid
    // history surfaces upstream instead of being hidden.
    let empty_refusal = match message.get("refusal") {
        None | Some(Value::Null) => true,
        Some(Value::String(text)) => text.trim().is_empty(),
        _ => false,
    };
    let empty_audio = match message.get("audio") {
        None | Some(Value::Null) => true,
        Some(Value::Object(audio)) => audio.is_empty(),
        _ => false,
    };
    if !empty_refusal || !empty_audio {
        return false;
    }

    let empty_content = match message.get("content") {
        None | Some(Value::Null) => true,
        Some(Value::String(text)) => text.trim().is_empty(),
        Some(Value::Array(parts)) => parts.iter().all(is_empty_text_block),
        _ => false,
    };
    let empty_tool_calls = match message.get("tool_calls") {
        None | Some(Value::Null) => true,
        Some(Value::Array(calls)) => calls.is_empty(),
        _ => false,
    };
    let empty_function_call = match message.get("function_call") {
        None | Some(Value::Null) => true,
        Some(Value::Object(call)) => call.is_empty(),
        _ => false,
    };
    empty_content && empty_tool_calls && empty_function_call
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
