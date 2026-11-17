//! Mistral provider — uses Mistral's OpenAI-compatible endpoints.
//!
//! Wire targets:
//! - chat: `https://api.mistral.ai/v1/chat/completions`
//! - embeddings: `https://api.mistral.ai/v1/embeddings`
//!
//! Mistral exposes an OpenAI-compatible surface, so URL building, auth, stream
//! parsing and the supported-params list are delegated to the shared
//! [`OpenAiCompatProvider`] engine. Mistral only diverges on a handful of
//! request field names; those remappings live in
//! `apply_mistral_request_transforms` and run on top of the shared body
//! builder in [`ChatProvider::transform_request`]:
//!
//! - `tool_choice`: any non-`auto`/`none` string (`"required"`, typos, future
//!   OpenAI values) → `"any"` (Mistral has no `"required"` and rejects unknown
//!   string values).
//! - `seed` → `random_seed` (Mistral's top-level name). A client may also send
//!   `random_seed` directly; when both are present, the native `random_seed`
//!   wins to respect the explicit value.
//! - `max_completion_tokens` → `max_tokens`.
//! - `$id` / `$schema` stripped recursively from `tools` (Mistral's grammar
//!   validator rejects JSON-schema reference fields with a 422).
//! - `reasoning_effort` and `thinking`: for reasoning-capable Mistral models
//!   (`magistral`, `devstral`, or `supports_reasoning`), `reasoning_effort`
//!   (`none` / `high` for Magistral; `default` leaves Mistral's default; invalid
//!   levels 400 unless `drop_params`) and `thinking` (`disabled` → `none`,
//!   `enabled`/`adaptive` → `high`) map onto Mistral's native `reasoning_effort`
//!   field. Non-reasoning models drop both fields.
//! - `reasoning_content` / `thinking_blocks` stripped as top-level fields from
//!   `assistant` messages (Mistral 422s on unknown message keys); for models
//!   that accept reasoning, assistant reasoning is replayed inside `content` as
//!   Mistral `{"type":"thinking","thinking":[{"type":"text","text":...}]}`
//!   chunks (while stripping native thinking chunks on non-reasoning models).
//! - `name` stripped from non-`tool` messages (Mistral only accepts it on
//!   tool messages); empty `name` on tool messages is also dropped.
//! - `tool_calls` on any message are rewritten to Mistral's strict
//!   `{id, type:"function", function}` shape, dropping extra keys (e.g.
//!   OpenAI's `index`) that would trip Mistral's `extra_forbidden` validator.
//! - Empty assistant messages (`content: ""` with no other payload) are
//!   dropped (Mistral rejects them with a 422).
//!
//! On the response side, magistral reasoning models return `content` as a list
//! of content blocks; [`ChatProvider::transform_response`] and
//! [`ChatProvider::transform_stream_chunk`] normalize that list into the
//! OpenAI-standard string `content` + `reasoning_content` + `thinking_blocks`.
//! The non-streaming path additionally rewrites an empty-string `content`
//! field to `null` (a Mistral quirk; parity with the reference
//! `_handle_empty_content_response`).

use std::collections::HashMap;

use crate::openai_compat::OpenAiCompatProvider;
use oxllm_core::Result;
use oxllm_core::deployment::Deployment;
use oxllm_core::endpoint::Endpoint;
use oxllm_core::provider::{
    ChatProvider, EmbeddingProvider, Provider, ProviderId, ResolvedCredentials,
};
use serde_json::Value;

/// The default Mistral OpenAI-compatible base URL.
pub const DEFAULT_MISTRAL_BASE: &str = "https://api.mistral.ai/v1";

/// The provider prefix stripped from model ids (`mistral/<model>`).
const STRIP: &str = "mistral";

/// The subset of the shared OpenAI-compat param list
/// ([`SUPPORTED_CHAT_PARAMS`]) that Mistral's API accepts. The shared list
/// forwards several OpenAI-platform-only fields (`store`, `modalities`,
/// `audio`, `service_tier`, `metadata`) that Mistral rejects with a `422`; with
/// `drop_params = true` they would otherwise survive stripping and reach the
/// upstream, and with `drop_params = false` the host would not warn about
/// them. `random_seed` and `safe_prompt` (Mistral's native top-level names)
/// are advertised in addition to the OpenAI-standard `seed` so a client
/// sending them directly gets pass-through. `reasoning_effort` and `thinking`
/// are listed so `drop_params` does not strip them before
/// [`apply_mistral_request_transforms`] can consume them (they are never
/// forwarded verbatim — see the module docs). Kept in lock-step with the
/// shared list by `supported_params_excludes_openai_only_fields` in
/// `tests.rs`.
const MISTRAL_SUPPORTED_CHAT_PARAMS: &[&str] = &[
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
    "thinking",
    "seed",
    "random_seed",
    "safe_prompt",
    "logprobs",
    "top_logprobs",
    "logit_bias",
    "prompt_cache_key",
];

/// The shared OpenAI-compatible engine backing URL building, bearer auth,
/// stream parsing and the supported-params list. Only
/// [`ChatProvider::transform_request`] is overridden, to apply Mistral's
/// field remappings on top of the shared body builder. The published
/// `supported_params` list is narrowed to
/// [`MISTRAL_SUPPORTED_CHAT_PARAMS`] so the host's `drop_params` mechanism
/// strips OpenAI-platform fields Mistral rejects (`store`, `modalities`,
/// `audio`, `service_tier`).
const COMPAT: OpenAiCompatProvider =
    OpenAiCompatProvider::new("mistral", DEFAULT_MISTRAL_BASE, STRIP, true, true, true)
        .with_supported_params(MISTRAL_SUPPORTED_CHAT_PARAMS);

/// The Mistral provider. Supports both chat completions and embeddings.
#[derive(Debug, Clone, Copy)]
pub struct MistralProvider;

/// Configured Mistral provider handle.
pub const PROVIDER: MistralProvider = MistralProvider;

/// Strip the `mistral/` provider prefix from a model id. Kept as a free
/// function for backward compatibility with existing tests.
pub fn strip_mistral_prefix(model: &str) -> &str {
    OpenAiCompatProvider::strip_one_segment(model, STRIP)
}

const MISTRAL_REASONING_EFFORTS: &[&str] =
    &["none", "minimal", "low", "medium", "high", "xhigh", "max"];

fn mistral_accepts_reasoning(model: &str, deployment: &Deployment) -> bool {
    let lower = model.to_ascii_lowercase();
    if lower.contains("magistral")
        || lower.contains("devstral")
        || lower.contains("mistral-small-2603")
        || lower.contains("mistral-small-2503")
    {
        return true;
    }
    deployment
        .model_info
        .as_ref()
        .and_then(|info| info.extra.get("supports_reasoning"))
        .and_then(Value::as_bool)
        == Some(true)
}

fn nearest_magistral_effort(level: &str) -> &'static str {
    if level == "none" { "none" } else { "high" }
}

fn map_mistral_reasoning_params(
    obj: &mut serde_json::Map<String, Value>,
    model: &str,
    deployment: &Deployment,
    drop_params: bool,
) -> Result<()> {
    let raw_effort = obj.remove("reasoning_effort").filter(|v| !v.is_null());
    let raw_thinking = obj.remove("thinking").filter(|v| !v.is_null());
    if !mistral_accepts_reasoning(model, deployment) {
        return Ok(());
    }
    let is_magistral = model.to_ascii_lowercase().contains("magistral");
    let mut leaves_default = false;
    let mut resolved_effort: Option<String> = None;
    if let Some(effort_val) = raw_effort {
        let level_val = match &effort_val {
            Value::Object(map) if !map.contains_key("summary") => map.get("effort"),
            Value::Object(map) => {
                if map.get("summary").is_some_and(|v| !v.is_null()) && !drop_params {
                    return Err(oxllm_core::Error::BadRequest(
                        "mistral does not support reasoning_effort.summary; enable drop_params to ignore".into(),
                    ));
                }
                map.get("effort")
            }
            other => Some(other),
        };
        if level_val.and_then(Value::as_str) == Some("default") {
            leaves_default = true;
        } else if let Some(level) = level_val.and_then(Value::as_str)
            && MISTRAL_REASONING_EFFORTS.contains(&level)
        {
            let mapped = if is_magistral {
                nearest_magistral_effort(level)
            } else {
                level
            };
            resolved_effort = Some(mapped.to_string());
        } else if !drop_params {
            return Err(oxllm_core::Error::BadRequest(format!(
                "mistral does not support reasoning_effort={effort_val}; expected one of {}, or 'default'",
                MISTRAL_REASONING_EFFORTS.join(", ")
            )));
        }
    }
    if resolved_effort.is_none()
        && !leaves_default
        && let Some(thinking_val) = raw_thinking
    {
        let mapped = match &thinking_val {
            Value::Bool(true) => Some("high"),
            Value::Bool(false) => Some("none"),
            Value::Object(map) => match map.get("type").and_then(Value::as_str) {
                Some("disabled") => Some("none"),
                Some("adaptive") => Some(if is_magistral { "high" } else { "medium" }),
                Some("enabled") => {
                    let budget = map.get("budget_tokens").and_then(Value::as_u64);
                    let raw = match budget {
                        Some(0) => "none",
                        Some(b) if b <= 1024 => "low",
                        Some(b) if b <= 4096 => "medium",
                        _ => "high",
                    };
                    Some(if is_magistral {
                        nearest_magistral_effort(raw)
                    } else {
                        raw
                    })
                }
                _ => None,
            },
            _ => None,
        };
        if let Some(level) = mapped {
            resolved_effort = Some(level.to_string());
        }
    }
    if let Some(level) = resolved_effort {
        obj.insert("reasoning_effort".to_string(), Value::String(level));
    }
    Ok(())
}

/// Apply Mistral-specific request field remappings to an already-built OpenAI
/// chat request body (the output of the shared compat `transform_request`).
/// `request` is the original client body; it is consulted only for fields the
/// shared `build_chat_request` does not copy (notably `random_seed`).
fn apply_mistral_request_transforms(
    out: &mut Value,
    request: &Value,
    deployment: &Deployment,
    drop_params: bool,
) -> Result<()> {
    let Some(obj) = out.as_object_mut() else {
        return Ok(());
    };

    if let Some(cache_key) = request.get("prompt_cache_key").filter(|v| !v.is_null()) {
        match cache_key.as_str() {
            Some(key) if key.is_ascii() => {
                obj.insert(
                    "prompt_cache_key".to_string(),
                    Value::String(key.to_string()),
                );
            }
            _ if drop_params => {
                obj.remove("prompt_cache_key");
            }
            _ => {
                return Err(oxllm_core::Error::BadRequest(
                    "mistral prompt_cache_key must be an ASCII string".into(),
                ));
            }
        }
    }

    // Mistral has no `"required"` tool_choice and rejects unknown string
    // values; map any non-`auto`/`none` string to `"any"` (parity with
    // the reference tool-choice catch-all). Object forms pass through.
    let needs_tool_choice_remap = obj
        .get("tool_choice")
        .and_then(|v| v.as_str())
        .is_some_and(|s| s != "auto" && s != "none");
    if needs_tool_choice_remap {
        obj.insert("tool_choice".to_string(), Value::String("any".to_string()));
    }

    // Mistral reads `random_seed`, not `seed`, at the top level. The shared
    // body builder copies the OpenAI-standard `seed` (and drops the native
    // `random_seed`, which is not on the shared list), so recover a
    // client-sent `random_seed` from the original request. When both are
    // present, the explicit native `random_seed` wins; otherwise `seed` is
    // renamed. Null values are dropped either way.
    let direct_random_seed = request.get("random_seed").filter(|v| !v.is_null());
    if let Some(random_seed) = direct_random_seed {
        obj.remove("seed");
        obj.insert("random_seed".to_string(), random_seed.clone());
    } else if let Some(seed) = obj.remove("seed") {
        if !seed.is_null() {
            obj.insert("random_seed".to_string(), seed);
        }
    }

    // Mistral reads `max_tokens`, not `max_completion_tokens`.
    if let Some(max_completion_tokens) = obj.remove("max_completion_tokens") {
        if !max_completion_tokens.is_null() {
            obj.insert("max_tokens".to_string(), max_completion_tokens);
        }
    }

    // `safe_prompt` is a Mistral-native top-level param. It is on the Mistral
    // supported list (so drop_params keeps it) but not on the shared
    // `SUPPORTED_CHAT_PARAMS` list the body builder copies from, so recover it
    // from the original request.
    if let Some(safe_prompt) = request.get("safe_prompt").filter(|v| !v.is_null()) {
        obj.insert("safe_prompt".to_string(), safe_prompt.clone());
    }

    // Mistral's grammar validator rejects JSON-schema reference fields (`$id`,
    // `$schema`) on tool definitions with a 422. Walk the whole `tools` array
    // and strip them recursively, with a depth bound (parity with the reference
    // `_clean_tool_schema_for_mistral` / `_remove_json_schema_refs`).
    if let Some(tools) = obj.get_mut("tools") {
        remove_json_schema_refs(tools, MAX_SCHEMA_RECURSE_DEPTH);
    }

    let model = obj
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    map_mistral_reasoning_params(obj, &model, deployment, drop_params)?;
    let keeps_reasoning = mistral_accepts_reasoning(&model, deployment);

    // Message-level normalization: drop empty assistant messages, replay or
    // strip prior-turn reasoning based on model capability, enforce Mistral's
    // `name` placement rules, and rewrite `tool_calls` to Mistral's
    // `{id,type,function}` shape.
    let Some(messages) = obj.get_mut("messages").and_then(|v| v.as_array_mut()) else {
        return Ok(());
    };
    let last_idx = messages.len().saturating_sub(1);
    for (idx, message) in messages.iter_mut().enumerate() {
        let is_final_turn = idx == last_idx;
        let Some(map) = message.as_object_mut() else {
            continue;
        };
        let role = map
            .get("role")
            .and_then(|r| r.as_str())
            .unwrap_or("")
            .to_string();
        if role == "assistant" {
            replay_or_strip_assistant_reasoning(map, keeps_reasoning, is_final_turn);
        }
        normalize_tool_calls(map);
        // Mistral only accepts `name` on tool messages (and only when
        // non-empty). Keep it iff this is a tool message with a non-empty
        // name; otherwise drop it.
        let keep_name = role == "tool"
            && map
                .get("name")
                .and_then(|n| n.as_str())
                .is_some_and(|n| !n.trim().is_empty());
        if !keep_name {
            map.remove("name");
        }
    }
    // Drop emptied assistant messages AFTER the strip: a turn that carried
    // only gateway extension fields (e.g. a thinking-only replay) must not
    // survive as `content: ""`, which Mistral rejects (reference order:
    // strip, then normalize, then drop).
    messages.retain(|m| !is_empty_assistant_message(m));
    Ok(())
}

/// Rewrite a message's `tool_calls` array into Mistral's strict
/// `{id, type:"function", function}` shape, dropping any extra keys (e.g.
/// OpenAI's `index`, `logprobs`) that would trip Mistral's
/// `extra_forbidden` validator with a 422. Mirrors the reference
/// `MistralToolCallMessage` / `_handle_tool_call_message`. A null or non-array
/// `tool_calls` is left untouched.
fn normalize_tool_calls(map: &mut serde_json::Map<String, Value>) {
    let Some(tool_calls) = map.get_mut("tool_calls").and_then(|v| v.as_array_mut()) else {
        return;
    };
    for tool_call in tool_calls.iter_mut() {
        let Some(tc) = tool_call.as_object_mut() else {
            continue;
        };
        let id = tc.remove("id").unwrap_or(Value::Null);
        let mut function = tc.remove("function").unwrap_or(Value::Null);
        // Provider extension fields ride on the `function` sub-object too;
        // Mistral's extra_forbidden validator rejects them there as well
        // (reference `_tool_call_without_provider_specific_fields`).
        if let Some(function_map) = function.as_object_mut() {
            function_map.remove("provider_specific_fields");
        }
        tc.clear();
        tc.insert("id".to_string(), id);
        tc.insert("type".to_string(), Value::String("function".to_string()));
        tc.insert("function".to_string(), function);
    }
}

/// Maximum recursion depth for [`remove_json_schema_refs`]. Mirrors the reference
/// `DEFAULT_MAX_RECURSE_DEPTH` to bound deeply-nested schemas (DoS defense).
const MAX_SCHEMA_RECURSE_DEPTH: i32 = 100;

/// Recursively remove JSON-schema reference fields (`$id`, `$schema`) from a
/// value. Walks objects and arrays; other types are returned unchanged. The
/// `depth` counter bounds recursion so a pathological (or hostile) schema
/// cannot exhaust the stack.
fn remove_json_schema_refs(value: &mut Value, depth: i32) {
    if depth <= 0 {
        return;
    }
    match value {
        Value::Object(map) => {
            map.remove("$id");
            map.remove("$schema");
            for child in map.values_mut() {
                remove_json_schema_refs(child, depth - 1);
            }
        }
        Value::Array(items) => {
            for item in items.iter_mut() {
                remove_json_schema_refs(item, depth - 1);
            }
        }
        _ => {}
    }
}

/// Mistral rejects an assistant message whose `content` is the empty string
/// when it carries no other payload (no `tool_calls`, no `function_call`,
/// etc.). Such messages are dropped entirely before the request is sent
/// (parity with the reference empty-assistant-message check, which walks every
/// typed field of `ChatCompletionAssistantMessage`).
fn is_empty_assistant_message(message: &Value) -> bool {
    let is_assistant = message
        .get("role")
        .and_then(|r| r.as_str())
        .is_some_and(|r| r == "assistant");
    if !is_assistant {
        return false;
    }
    let Some(map) = message.as_object() else {
        return false;
    };
    for (key, value) in map {
        if key == "role" {
            continue;
        }
        if key == "content" {
            if value.is_null() {
                continue;
            }
            if value.as_str().is_some_and(|s| s.is_empty()) {
                continue;
            }
            return false;
        }
        if !value.is_null() {
            return false;
        }
    }
    true
}

fn is_mistral_think_chunk(part: &Value) -> bool {
    part.get("type").and_then(Value::as_str) == Some("thinking")
        && part.get("thinking").is_some_and(Value::is_array)
}

fn sanitize_think_chunk(chunk: &Value, may_stay_open: bool) -> Option<Value> {
    let items = chunk.get("thinking")?.as_array()?;
    let text_items: Vec<Value> = items
        .iter()
        .filter(|item| item.get("type").and_then(Value::as_str) == Some("text"))
        .filter_map(|item| item.get("text").and_then(Value::as_str))
        .map(|text| serde_json::json!({"type": "text", "text": text}))
        .collect();
    if text_items.is_empty() {
        return None;
    }
    let mut out = serde_json::Map::new();
    out.insert("type".to_string(), Value::String("thinking".to_string()));
    out.insert("thinking".to_string(), Value::Array(text_items));
    if let Some(closed) = chunk.get("closed").and_then(Value::as_bool)
        && (closed || may_stay_open)
    {
        out.insert("closed".to_string(), Value::Bool(closed));
    }
    Some(Value::Object(out))
}

fn replay_or_strip_assistant_reasoning(
    map: &mut serde_json::Map<String, Value>,
    keeps_reasoning: bool,
    is_final_turn: bool,
) {
    let reasoning_content = map.remove("reasoning_content");
    let thinking_blocks = map.remove("thinking_blocks");
    map.remove("reasoning_items");
    map.remove("provider_specific_fields");

    if !keeps_reasoning {
        if let Some(Value::Array(parts)) = map.get_mut("content") {
            parts.retain(|part| !is_mistral_think_chunk(part));
            if parts.is_empty() {
                map.insert("content".to_string(), Value::Null);
            }
        }
        return;
    }

    let is_prefix = map.get("prefix").and_then(Value::as_bool) == Some(true);
    let may_end_open = is_final_turn && is_prefix;

    if let Some(Value::Array(parts)) = map.get("content")
        && parts.iter().any(is_mistral_think_chunk)
    {
        let last_pos = parts.len().saturating_sub(1);
        let sendable: Vec<Value> = parts
            .iter()
            .enumerate()
            .filter_map(|(pos, part)| {
                if is_mistral_think_chunk(part) {
                    sanitize_think_chunk(part, may_end_open && pos == last_pos)
                } else {
                    Some(part.clone())
                }
            })
            .collect();
        if sendable.is_empty() {
            map.insert("content".to_string(), Value::Null);
        } else {
            map.insert("content".to_string(), Value::Array(sendable));
        }
        return;
    }

    let mut reasoning_text = reasoning_content
        .as_ref()
        .and_then(Value::as_str)
        .filter(|s| !s.trim().is_empty())
        .map(str::to_string);
    if reasoning_text.is_none()
        && let Some(Value::Array(blocks)) = &thinking_blocks
    {
        let combined: String = blocks
            .iter()
            .filter(|b| b.get("type").and_then(Value::as_str) == Some("thinking"))
            .filter_map(|b| b.get("thinking").and_then(Value::as_str))
            .collect();
        if !combined.trim().is_empty() {
            reasoning_text = Some(combined);
        }
    }
    let Some(reasoning_text) = reasoning_text else {
        return;
    };

    let content_parts: Option<Vec<Value>> = match map.get("content") {
        None | Some(Value::Null) => Some(Vec::new()),
        Some(Value::String(s)) if s.is_empty() => Some(Vec::new()),
        Some(Value::String(s)) => Some(vec![serde_json::json!({"type": "text", "text": s})]),
        Some(Value::Array(parts)) => Some(parts.clone()),
        Some(_) => None,
    };
    let Some(content_parts) = content_parts else {
        return;
    };
    let has_tool_calls = map
        .get("tool_calls")
        .and_then(Value::as_array)
        .is_some_and(|arr| !arr.is_empty());
    if content_parts.is_empty() && !has_tool_calls {
        return;
    }
    let mut new_parts = Vec::with_capacity(content_parts.len() + 1);
    new_parts.push(serde_json::json!({
        "type": "thinking",
        "thinking": [{"type": "text", "text": reasoning_text}],
        "closed": true,
    }));
    new_parts.extend(content_parts);
    map.insert("content".to_string(), Value::Array(new_parts));
}

/// Normalize magistral content-list responses on the non-streaming path. For
/// each choice, if `message.content` is an array of content blocks it is split
/// into OpenAI-standard string `content` + `reasoning_content` +
/// `thinking_blocks`; string/null content is left untouched (parity with
/// the reference content-list-to-string conversion). Empty-string `content`
/// is first rewritten to `null` (parity with the reference
/// `_handle_empty_content_response`).
fn normalize_magistral_response(body: &mut Value) {
    let Some(choices) = body.get_mut("choices").and_then(|v| v.as_array_mut()) else {
        return;
    };
    for choice in choices {
        let Some(message) = choice.get_mut("message") else {
            continue;
        };
        normalize_empty_content_to_null(message);
        normalize_content_field(message);
    }
}

/// Normalize magistral content-list deltas on the streaming path (parity with
/// the reference `MistralChatResponseIterator.chunk_parser`).
fn normalize_magistral_stream_chunk(chunk: &mut Value) {
    let Some(choices) = chunk.get_mut("choices").and_then(|v| v.as_array_mut()) else {
        return;
    };
    for choice in choices {
        let Some(delta) = choice.get_mut("delta") else {
            continue;
        };
        normalize_content_field(delta);
    }
}

/// Rewrite an empty-string `content` field to `null` on a message/delta.
/// Mistral sometimes returns `content: ""` instead of `null`; downstream
/// consumers that test `content == null` to mean "no text" need the canonical
/// null form. Mirrors the reference empty-content response handling.
fn normalize_empty_content_to_null(target: &mut Value) {
    let Some(map) = target.as_object_mut() else {
        return;
    };
    if map
        .get("content")
        .and_then(|c| c.as_str())
        .is_some_and(|s| s.is_empty())
    {
        map.insert("content".to_string(), Value::Null);
    }
}

/// If `target.content` is an array, split its content blocks into the
/// OpenAI-standard `content` (string, or null if no text block) plus
/// `reasoning_content` and `thinking_blocks` derived from `thinking` blocks.
fn normalize_content_field(target: &mut Value) {
    let Some(map) = target.as_object_mut() else {
        return;
    };
    if !matches!(map.get("content"), Some(Value::Array(_))) {
        return;
    }
    // `remove` cannot fail here (the key was just observed), but a `let-else`
    // keeps the no-panic guarantee even if the check above ever changes.
    let Some(content) = map.remove("content") else {
        return;
    };
    let (text, thinking_blocks, reasoning_content) = normalize_content_blocks(&content);
    match text {
        Some(t) => {
            map.insert("content".to_string(), Value::String(t));
        }
        None => {
            map.insert("content".to_string(), Value::Null);
        }
    }
    if !thinking_blocks.is_empty() {
        map.insert("thinking_blocks".to_string(), Value::Array(thinking_blocks));
        map.insert(
            "reasoning_content".to_string(),
            Value::String(reasoning_content.unwrap_or_default()),
        );
    }
}

/// Split a magistral content-block list into `(text, thinking_blocks,
/// reasoning_content)`, mirroring the reference content-block normalization.
/// `thinking` blocks contribute to `reasoning_content` (joined with `\n`) and
/// `thinking_blocks`; `text` blocks contribute to the concatenated `content`.
fn normalize_content_blocks(content: &Value) -> (Option<String>, Vec<Value>, Option<String>) {
    let mut text_segments: Vec<String> = Vec::new();
    let mut thinking_blocks: Vec<Value> = Vec::new();
    let mut reasoning_segments: Vec<String> = Vec::new();

    for block in content.as_array().into_iter().flatten() {
        let block_type = block.get("type").and_then(|t| t.as_str()).unwrap_or("");
        if block_type == "thinking" {
            let mut parts: Vec<String> = Vec::new();
            for inner in block
                .get("thinking")
                .and_then(|t| t.as_array())
                .into_iter()
                .flatten()
            {
                let is_text = inner
                    .get("type")
                    .and_then(|t| t.as_str())
                    .is_some_and(|t| t == "text");
                if is_text {
                    parts.push(
                        inner
                            .get("text")
                            .and_then(|t| t.as_str())
                            .unwrap_or("")
                            .to_string(),
                    );
                }
            }
            let thinking_text = parts.join("");
            if !thinking_text.is_empty() {
                reasoning_segments.push(thinking_text.clone());
                thinking_blocks.push(serde_json::json!({
                    "type": "thinking",
                    "thinking": thinking_text,
                    "signature": "mistral",
                }));
            }
        } else if block_type == "text" {
            text_segments.push(
                block
                    .get("text")
                    .and_then(|t| t.as_str())
                    .unwrap_or("")
                    .to_string(),
            );
        }
    }

    let normalized_text = if text_segments.is_empty() {
        None
    } else {
        Some(text_segments.join(""))
    };
    let reasoning_content = if reasoning_segments.is_empty() {
        None
    } else {
        Some(reasoning_segments.join("\n"))
    };
    (normalized_text, thinking_blocks, reasoning_content)
}

impl ChatProvider for MistralProvider {
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
        let drop_params = deployment.oxllm_params.drop_params == Some(true)
            || request.get("drop_params").and_then(Value::as_bool) == Some(true);
        apply_mistral_request_transforms(&mut out, request, deployment, drop_params)?;
        Ok(out)
    }

    fn transform_response(&self, body: &Value, model: &str) -> Result<Value> {
        let mut out = ChatProvider::transform_response(&COMPAT, body, model)?;
        normalize_magistral_response(&mut out);
        Ok(out)
    }

    fn transform_stream_chunk(&self, chunk: &str) -> Result<Option<Value>> {
        let Some(mut value) = ChatProvider::transform_stream_chunk(&COMPAT, chunk)? else {
            return Ok(None);
        };
        normalize_magistral_stream_chunk(&mut value);
        Ok(Some(value))
    }

    fn supported_params(&self) -> Option<Vec<&str>> {
        ChatProvider::supported_params(&COMPAT)
    }
}

impl EmbeddingProvider for MistralProvider {
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

impl Provider for MistralProvider {
    fn id(&self) -> ProviderId {
        "mistral"
    }

    fn supports(&self, endpoint: Endpoint) -> bool {
        matches!(
            endpoint,
            Endpoint::ChatCompletions | Endpoint::Embeddings | Endpoint::AudioTranscriptions
        )
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
