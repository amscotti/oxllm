//! xAI provider — uses xAI's OpenAI-compatible Chat Completions endpoint.
//!
//! Wire target: `https://api.x.ai/v1/chat/completions`.
//!
//! xAI exposes an OpenAI-compatible surface, so URL building, auth, stream
//! parsing and the supported-params list are delegated to the shared
//! [`OpenAiCompatProvider`] engine. xAI diverges from the OpenAI contract on a
//! handful of points that the wrapper applies on top of the shared body
//! builder (parity with the reference `XAIChatConfig`):
//!
//! - **`name` stripped from non-user messages**: the xAI API rejects any
//!   non-user message carrying a `name` field with a 400 (reference issue
//!   #9720). Only the `user` role is permitted to keep `name`.
//! - **`strict` stripped from tool definitions**: xAI rejects the OpenAI
//!   structured-outputs `strict` flag on tool function definitions.
//! - **Reasoning tokens folded into completion**: xAI accounts
//!   `reasoning_tokens` additively (`total = prompt + completion + reasoning`),
//!   unlike OpenAI where reasoning is a subset of completion. The response and
//!   stream usage are normalized so `total = prompt + completion` holds,
//!   keeping the cost calculator's subset invariant valid.
//! - **Empty `finish_reason` fixed for tool calls**: xAI returns `""` for
//!   `finish_reason` when the response carries tool calls; the OpenAI contract
//!   is `"tool_calls"`.
//! - **Empty-choices usage chunks normalized**: xAI emits a final stream chunk
//!   with an empty `choices` array but a populated `usage` object; a dummy
//!   choice is injected so the orchestrator's usage accumulator processes it.
//!
//! See `docs/adding-a-provider.md`.

use std::collections::HashMap;

use crate::openai_compat::OpenAiCompatProvider;
use oxllm_core::Result;
use oxllm_core::deployment::Deployment;
use oxllm_core::endpoint::Endpoint;
use oxllm_core::provider::{ChatProvider, Provider, ProviderId, ResolvedCredentials};
use serde_json::{Map, Value};

/// The default xAI OpenAI-compatible base URL.
pub const DEFAULT_XAI_BASE: &str = "https://api.x.ai/v1";

/// The provider prefix stripped from model ids (`xai/<model>`).
const STRIP: &str = "xai";

/// The shared OpenAI-compatible engine backing URL building, bearer auth,
/// stream parsing and the supported-params list. Only the request/response/
/// stream transforms are overridden on the wrapper to apply xAI's
/// divergences on top of the shared body builder.
const COMPAT: OpenAiCompatProvider =
    OpenAiCompatProvider::new("xai", DEFAULT_XAI_BASE, STRIP, false, true, true);

/// The xAI provider. Supports chat completions only.
#[derive(Debug, Clone, Copy)]
pub struct XaiProvider;

/// Configured xAI provider handle.
pub const PROVIDER: XaiProvider = XaiProvider;

/// Strip the `xai/` provider prefix from a model id. Kept as a free function
/// for backward compatibility with existing tests.
pub fn strip_xai_prefix(model: &str) -> &str {
    OpenAiCompatProvider::strip_one_segment(model, STRIP)
}

/// Apply xAI-specific request transforms to an already-built OpenAI chat
/// request body (the output of the shared compat `transform_request`).
fn apply_xai_request_transforms(out: &mut Value) {
    strip_name_from_non_user_messages(out);
    strip_strict_from_tools(out);
}

/// Remove the `name` field from every message whose role is not `user`
/// (parity with the reference message-name filtering).
/// xAI rejects non-user messages carrying `name` with a 400. The `user` role
/// is permitted to keep `name` (matching OpenAI's multi-user convention).
fn strip_name_from_non_user_messages(out: &mut Value) {
    let Some(messages) = out.get_mut("messages").and_then(|v| v.as_array_mut()) else {
        return;
    };
    for message in messages.iter_mut() {
        let Some(map) = message.as_object_mut() else {
            continue;
        };
        let is_user = map
            .get("role")
            .and_then(|r| r.as_str())
            .is_some_and(|r| r == "user");
        if !is_user {
            map.remove("name");
        }
    }
}

/// Remove the `strict` flag from each tool's `function` object (parity with
/// the reference tool-strict filtering). xAI rejects the
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

/// Apply xAI-specific response transforms to an already-built OpenAI chat
/// response body (the output of the shared compat `transform_response`).
fn apply_xai_response_transforms(out: &mut Value) {
    fix_empty_finish_reason_for_tool_calls(out);
    fold_reasoning_tokens_into_completion(out);
}

/// Set `finish_reason = "tool_calls"` on choices where xAI returned an empty
/// string but the message carries tool calls (parity with the reference
/// `_fix_choice_finish_reason_for_tool_calls`). The OpenAI contract is that a
/// tool-call response finishes with `"tool_calls"`.
fn fix_empty_finish_reason_for_tool_calls(out: &mut Value) {
    let Some(choices) = out.get_mut("choices").and_then(|v| v.as_array_mut()) else {
        return;
    };
    for choice in choices.iter_mut() {
        let Some(map) = choice.as_object_mut() else {
            continue;
        };
        let empty_finish = map
            .get("finish_reason")
            .and_then(|f| f.as_str())
            .is_some_and(|f| f.is_empty());
        if !empty_finish {
            continue;
        }
        let has_tool_calls = map
            .get("message")
            .and_then(|m| m.get("tool_calls"))
            .and_then(|t| t.as_array())
            .is_some_and(|t| !t.is_empty());
        if has_tool_calls {
            map.insert(
                "finish_reason".to_string(),
                Value::String("tool_calls".to_string()),
            );
        }
    }
}

/// Fold `reasoning_tokens` into `completion_tokens` so the OpenAI subset
/// invariant `total = prompt + completion` holds (parity with the reference
/// `_fold_reasoning_tokens_into_completion`). xAI accounts reasoning
/// additively; without this fold the cost calculator subtracts reasoning
/// tokens that were never inside `completion_tokens`, mis-pricing the output
/// bucket. Idempotent and guarded against double-counting.
fn fold_reasoning_tokens_into_completion(out: &mut Value) {
    let Some(usage) = out.get_mut("usage").and_then(|u| u.as_object_mut()) else {
        return;
    };
    fold_reasoning_in_usage_object(usage);
}

/// Shared inner: fold reasoning into completion for one usage object map.
fn fold_reasoning_in_usage_object(usage: &mut Map<String, Value>) {
    let reasoning = usage
        .get("completion_tokens_details")
        .and_then(|d| d.get("reasoning_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    if reasoning == 0 {
        return;
    }
    let prompt = usage
        .get("prompt_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let completion = usage
        .get("completion_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let total = usage
        .get("total_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    // Already consistent — nothing to do.
    if total == prompt.saturating_add(completion) {
        return;
    }
    // Guard against double-counting if xAI changes accounting: only fold when
    // the additive identity holds exactly.
    if total != prompt.saturating_add(completion).saturating_add(reasoning) {
        return;
    }
    usage.insert(
        "completion_tokens".to_string(),
        Value::Number(serde_json::Number::from(
            completion.saturating_add(reasoning),
        )),
    );
}

/// Apply xAI-specific stream-chunk transforms to an already-parsed OpenAI
/// stream chunk (the output of the shared compat `transform_stream_chunk`).
fn apply_xai_stream_transforms(value: &mut Value) {
    // Inject a dummy choice when `choices` is empty but `usage` is present, so
    // the orchestrator's usage accumulator processes the final usage chunk
    // (parity with the reference xAI streaming chunk parser).
    let needs_dummy_choice = {
        let choices_empty = value
            .get("choices")
            .and_then(|c| c.as_array())
            .is_some_and(|c| c.is_empty());
        let has_usage = value.get("usage").is_some();
        choices_empty && has_usage
    };
    if needs_dummy_choice {
        let dummy = serde_json::json!({"index": 0, "delta": {}, "finish_reason": null});
        if let Some(choices) = value.get_mut("choices").and_then(|c| c.as_array_mut()) {
            choices.push(dummy);
        }
    }
    // Fold reasoning tokens into completion in the usage object.
    if let Some(usage) = value.get_mut("usage").and_then(|u| u.as_object_mut()) {
        fold_reasoning_in_usage_object(usage);
    }
}

impl ChatProvider for XaiProvider {
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
        apply_xai_request_transforms(&mut out);
        Ok(out)
    }

    fn transform_response(&self, body: &Value, model: &str) -> Result<Value> {
        let mut out = ChatProvider::transform_response(&COMPAT, body, model)?;
        apply_xai_response_transforms(&mut out);
        Ok(out)
    }

    fn transform_stream_chunk(&self, chunk: &str) -> Result<Option<Value>> {
        let mut value = match ChatProvider::transform_stream_chunk(&COMPAT, chunk)? {
            Some(v) => v,
            None => return Ok(None),
        };
        apply_xai_stream_transforms(&mut value);
        Ok(Some(value))
    }

    fn supported_params(&self) -> Option<Vec<&str>> {
        ChatProvider::supported_params(&COMPAT)
    }
}

impl Provider for XaiProvider {
    fn id(&self) -> ProviderId {
        "xai"
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
