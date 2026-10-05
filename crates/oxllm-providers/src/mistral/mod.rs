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
//! - `reasoning_effort` and `thinking` are consumed internally and never
//!   forwarded; for `magistral` models either triggers injection of Mistral's
//!   recommended reasoning system prompt (parity with LiteLLM). Non-magistral
//!   models simply drop both fields.
//! - `reasoning_content` / `thinking_blocks` stripped from `assistant`
//!   messages (Mistral 422s on prior-turn reasoning fields).
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
//! field to `null` (a Mistral quirk; parity with LiteLLM's
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

/// Apply Mistral-specific request field remappings to an already-built OpenAI
/// chat request body (the output of the shared compat `transform_request`).
/// `request` is the original client body; it is consulted only for fields the
/// shared `build_chat_request` does not copy (notably `random_seed`).
fn apply_mistral_request_transforms(out: &mut Value, request: &Value) {
    let Some(obj) = out.as_object_mut() else {
        return;
    };

    // Mistral has no `"required"` tool_choice and rejects unknown string
    // values; map any non-`auto`/`none` string to `"any"` (parity with
    // LiteLLM's `_map_tool_choice` catch-all). Object forms pass through.
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
    // and strip them recursively, with a depth bound (parity with LiteLLM's
    // `_clean_tool_schema_for_mistral` / `_remove_json_schema_refs`).
    if let Some(tools) = obj.get_mut("tools") {
        remove_json_schema_refs(tools, MAX_SCHEMA_RECURSE_DEPTH);
    }

    // Mistral does not accept `reasoning_effort` or `thinking` on the wire.
    // For magistral models either is consumed internally as a signal to
    // inject Mistral's recommended reasoning system prompt (parity with
    // LiteLLM, which treats both `reasoning_effort` and `thinking` as
    // alternative triggers); for all other models they are simply dropped.
    let is_magistral = obj
        .get("model")
        .and_then(|m| m.as_str())
        .is_some_and(|m| m.to_ascii_lowercase().contains("magistral"));
    // Always drop both fields from the upstream body: Mistral does not accept
    // them on the wire. For magistral models a non-null value in either field
    // additionally triggers reasoning-prompt injection below.
    let reasoning_effort = obj.remove("reasoning_effort");
    let thinking = obj.remove("thinking");
    let inject_prompt = is_magistral
        && (reasoning_effort.is_some_and(|v| !v.is_null())
            || thinking.is_some_and(|v| !v.is_null()));

    // Message-level normalization: drop empty assistant messages, strip
    // prior-turn reasoning fields from assistant messages, enforce Mistral's
    // `name` placement rules, and rewrite `tool_calls` to Mistral's
    // `{id,type,function}` shape (parity with LiteLLM's
    // `_handle_tool_call_message`).
    let Some(messages) = obj.get_mut("messages").and_then(|v| v.as_array_mut()) else {
        return;
    };
    if inject_prompt {
        inject_reasoning_into_messages(messages);
    }
    messages.retain(|m| !is_empty_assistant_message(m));
    for message in messages.iter_mut() {
        let Some(map) = message.as_object_mut() else {
            continue;
        };
        let role = map
            .get("role")
            .and_then(|r| r.as_str())
            .unwrap_or("")
            .to_string();
        if role == "assistant" {
            map.remove("reasoning_content");
            map.remove("thinking_blocks");
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
}

/// Rewrite a message's `tool_calls` array into Mistral's strict
/// `{id, type:"function", function}` shape, dropping any extra keys (e.g.
/// OpenAI's `index`, `logprobs`) that would trip Mistral's
/// `extra_forbidden` validator with a 422. Mirrors LiteLLM's
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
        let function = tc.remove("function").unwrap_or(Value::Null);
        tc.clear();
        tc.insert("id".to_string(), id);
        tc.insert("type".to_string(), Value::String("function".to_string()));
        tc.insert("function".to_string(), function);
    }
}

/// Maximum recursion depth for [`remove_json_schema_refs`]. Mirrors LiteLLM's
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
/// (parity with LiteLLM's `_is_empty_assistant_message`, which walks every
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

/// Mistral's recommended reasoning system prompt for magistral models, per
/// <https://huggingface.co/mistralai/Magistral-Small-2506> (mirrors LiteLLM's
/// `_get_mistral_reasoning_system_prompt`).
const MISTRAL_REASONING_SYSTEM_PROMPT: &str = r#"<s>[SYSTEM_PROMPT]system_prompt
A user will ask you to solve a task. You should first draft your thinking process (inner monologue) until you have derived the final answer. Afterwards, write a self-contained summary of your thoughts (i.e. your summary should be succinct but contain all the critical steps you needed to reach the conclusion). You should use Markdown to format your response. Write both your thoughts and summary in the same language as the task posed by the user. NEVER use \boxed{} in your response.

Your thinking process must follow the template below:
<think>
Your thoughts or/and draft, like working through an exercise on scratch paper. Be as casual and as long as you want until you are confident to generate a correct answer.
</think>

Here, provide a concise summary that reflects your reasoning and presents a clear final answer to the user. Don't mention that this is a summary.

Problem:

[/SYSTEM_PROMPT][INST]user_message[/INST]<think>
reasoning_traces
</think>
assistant_response</s>[INST]user_message[/INST]"#;

/// Prepend the reasoning system prompt to the conversation. If a `system`
/// message already exists, the prompt is prepended to its content; otherwise a
/// new `system` message is inserted at the front (parity with LiteLLM's
/// `_add_reasoning_system_prompt_if_needed`).
///
/// The original system message content shape is preserved: a string content is
/// prepended in-place (`"{prompt}\n\n{content}"`); an array (multimodal) content
/// has a `{"type":"text","text":"{prompt}\n\n"}` block prepended to the list.
/// Missing/null content is replaced with the bare prompt string. This matches
/// LiteLLM's three-way branch and avoids silently overwriting array-form
/// system content with a bare string.
fn inject_reasoning_into_messages(messages: &mut Vec<Value>) {
    for message in messages.iter_mut() {
        let is_system = message
            .get("role")
            .and_then(|r| r.as_str())
            .is_some_and(|r| r == "system");
        if !is_system {
            continue;
        }
        if let Some(map) = message.as_object_mut() {
            let existing = map.remove("content");
            let new_content = match existing {
                Some(Value::String(c)) => {
                    Value::String(format!("{MISTRAL_REASONING_SYSTEM_PROMPT}\n\n{c}"))
                }
                Some(Value::Array(blocks)) => {
                    let mut new_blocks = Vec::with_capacity(blocks.len() + 1);
                    new_blocks.push(serde_json::json!({
                        "type": "text",
                        "text": format!("{MISTRAL_REASONING_SYSTEM_PROMPT}\n\n"),
                    }));
                    new_blocks.extend(blocks);
                    Value::Array(new_blocks)
                }
                _ => Value::String(MISTRAL_REASONING_SYSTEM_PROMPT.to_string()),
            };
            map.insert("content".to_string(), new_content);
        }
        return;
    }
    messages.insert(
        0,
        serde_json::json!({"role": "system", "content": MISTRAL_REASONING_SYSTEM_PROMPT}),
    );
}

/// Normalize magistral content-list responses on the non-streaming path. For
/// each choice, if `message.content` is an array of content blocks it is split
/// into OpenAI-standard string `content` + `reasoning_content` +
/// `thinking_blocks`; string/null content is left untouched (parity with
/// LiteLLM's `_handle_content_list_to_str_conversion`). Empty-string `content`
/// is first rewritten to `null` (parity with LiteLLM's
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
/// LiteLLM's `MistralChatResponseIterator.chunk_parser`).
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
/// null form. Mirrors LiteLLM's `_handle_empty_content_response`.
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
/// reasoning_content)`, mirroring LiteLLM's `_normalize_content_blocks`.
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
        apply_mistral_request_transforms(&mut out, request);
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
