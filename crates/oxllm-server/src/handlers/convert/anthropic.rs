//! Anthropic Messages API ↔ OpenAI chat format conversion.
//!
//! Pure data-transformation helpers extracted from `handlers::messages`. This
//! module knows nothing about axum, tokio, or SSE transport — it maps JSON
//! shapes only, so it can be unit-tested in isolation and reused.

use serde_json::{Map, Value, json};

use crate::sse::extract_sse_data;

/// Defence-in-depth cap on per-converter argument accumulators (notably
/// `pending_arguments` for tool calls whose `id`/`name` haven't arrived
/// yet). Matches `orchestration::MAX_STREAM_BUFFER_BYTES` so the converter
/// layer does not hold more memory than the wire-level cap.
const MAX_STREAM_BUFFER_BYTES: usize = 4 * 1024 * 1024;
/// Defence-in-depth cap on the number of distinct tool-call indices tracked
/// per stream. A healthy stream carries a handful; thousands indicates a
/// pathological / attacker-controlled upstream.
const MAX_STREAM_TOOL_CALLS: usize = 1024;

// ---------------------------------------------------------------------------
// Request conversion: Anthropic → OpenAI
// ---------------------------------------------------------------------------

pub(crate) fn anthropic_to_openai_request(req: &Value) -> Value {
    let mut out = Map::new();

    if let Some(model) = req.get("model") {
        out.insert("model".to_string(), model.clone());
    }

    let mut messages: Vec<Value> = Vec::new();

    if let Some(system) = req.get("system") {
        let system_content = extract_system_text(system);
        let has_content = match &system_content {
            Value::String(s) => !s.is_empty(),
            Value::Array(a) => !a.is_empty(),
            _ => false,
        };
        if has_content {
            messages.push(json!({"role": "system", "content": system_content}));
        }
    }

    if let Some(msgs) = req.get("messages").and_then(|v| v.as_array()) {
        convert_anthropic_messages_to_openai(msgs, &mut messages);
    }
    out.insert("messages".to_string(), Value::Array(messages));

    if let Some(max_tokens) = req.get("max_tokens") {
        out.insert("max_tokens".to_string(), max_tokens.clone());
    }

    for key in &["temperature", "top_p"] {
        if let Some(val) = req.get(*key) {
            if !val.is_null() {
                out.insert(key.to_string(), val.clone());
            }
        }
    }

    if let Some(stop) = req.get("stop_sequences") {
        if !stop.is_null() {
            out.insert("stop".to_string(), stop.clone());
        }
    }

    if let Some(stream) = req.get("stream") {
        if !stream.is_null() {
            out.insert("stream".to_string(), stream.clone());
        }
    }
    if req.get("stream").and_then(Value::as_bool) == Some(true) {
        // The converter needs the final chat usage chunk to populate native
        // Messages usage, even though Anthropic clients do not send this
        // OpenAI-only option themselves.
        out.insert("stream_options".to_string(), json!({"include_usage": true}));
    }

    // Pass the cache opt-in (`cache: {"insert": true}`) through so the chat
    // pipeline's CacheHook can see it; dropping it would silently disable
    // caching for Anthropic-shaped requests.
    if let Some(cache) = req.get("cache") {
        if !cache.is_null() {
            out.insert("cache".to_string(), cache.clone());
        }
    }

    if let Some(tools) = req.get("tools").and_then(|v| v.as_array()) {
        let mapped: Vec<Value> = tools.iter().map(map_anthropic_tool_to_openai).collect();
        if !mapped.is_empty() {
            out.insert("tools".to_string(), Value::Array(mapped));
        }
    }

    if let Some(tool_choice) = req.get("tool_choice") {
        if let Some(mapped) = map_anthropic_tool_choice_to_openai(tool_choice) {
            out.insert("tool_choice".to_string(), mapped);
        }
    }

    // `output_config` (Bedrock adaptive-thinking `effort`) must survive the
    // Messages → chat bridge; dropping it makes Converse emit zero
    // `reasoningContent`.
    for key in &[
        "thinking",
        "top_k",
        "metadata",
        "output_config",
        "speed",
        "service_tier",
    ] {
        if let Some(val) = req.get(*key) {
            if !val.is_null() {
                out.insert(key.to_string(), val.clone());
            }
        }
    }

    Value::Object(out)
}

fn extract_system_text(system: &Value) -> Value {
    match system {
        Value::String(s) => Value::String(s.clone()),
        Value::Array(blocks) => {
            let has_cache_control = blocks.iter().any(|b| b.get("cache_control").is_some());
            if has_cache_control {
                let mapped: Vec<Value> = blocks
                    .iter()
                    .map(|b| {
                        let text = b.get("text").and_then(|t| t.as_str()).unwrap_or("");
                        let mut block = json!({"type": "text", "text": text});
                        if let Some(cc) = b.get("cache_control") {
                            block["cache_control"] = cc.clone();
                        }
                        block
                    })
                    .collect();
                Value::Array(mapped)
            } else {
                let text = blocks
                    .iter()
                    .filter_map(|b| b.get("text").and_then(|t| t.as_str()).map(String::from))
                    .collect::<Vec<_>>()
                    .join(" ");
                Value::String(text)
            }
        }
        _ => Value::String(String::new()),
    }
}

fn convert_anthropic_messages_to_openai(messages: &[Value], out: &mut Vec<Value>) {
    for msg in messages {
        let first = out.len();
        convert_single_anthropic_message(msg, out);
        if let Some(config) = msg.get("output_config").filter(|value| !value.is_null()) {
            for message in &mut out[first..] {
                message["output_config"] = config.clone();
            }
        }
    }
}

/// Map an Anthropic image block's `source` to an OpenAI
/// `image_url` content-part.
///
/// Anthropic supports two source shapes:
/// - base64: `{"type":"base64","media_type":"image/jpeg","data":"..."}`
/// - url: `{"type":"url","url":"https://..."}`
///
/// Returns `None` when the source shape is unrecognised so the caller can
/// decide whether to drop the block or fall back.
fn anthropic_source_to_openai_image_url(source: &Value) -> Option<Value> {
    let source_type = source.get("type").and_then(|v| v.as_str())?;
    let url = match source_type {
        "base64" => {
            let media_type = source
                .get("media_type")
                .and_then(|v| v.as_str())
                .unwrap_or("image/jpeg");
            let data = source.get("data").and_then(|v| v.as_str())?;
            if data.is_empty() {
                return None;
            }
            format!("data:{media_type};base64,{data}")
        }
        "url" => {
            let u = source.get("url").and_then(|v| v.as_str())?;
            if u.is_empty() {
                return None;
            }
            u.to_string()
        }
        _ => return None,
    };
    Some(json!({"type": "image_url", "image_url": {"url": url}}))
}

fn convert_single_anthropic_message(msg: &Value, out: &mut Vec<Value>) {
    let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("user");

    match msg.get("content") {
        Some(Value::String(s)) => {
            out.push(json!({"role": role, "content": s}));
        }
        Some(Value::Array(blocks)) => {
            let first_message = out.len();
            let mut text_parts: Vec<String> = Vec::new();
            let mut thinking_blocks: Vec<Value> = Vec::new();
            let mut tool_calls: Vec<Value> = Vec::new();
            let mut tool_results: Vec<Value> = Vec::new();
            let mut content_blocks_raw: Vec<Value> = Vec::new();

            for block in blocks {
                let btype = block.get("type").and_then(|v| v.as_str());
                match btype {
                    Some("text") => {
                        if let Some(t) = block.get("text").and_then(|v| v.as_str()) {
                            let mut cb = Map::new();
                            cb.insert("type".to_string(), json!("text"));
                            cb.insert("text".to_string(), json!(t));
                            if let Some(cc) = block.get("cache_control") {
                                cb.insert("cache_control".to_string(), cc.clone());
                            }
                            content_blocks_raw.push(Value::Object(cb));
                            text_parts.push(t.to_string());
                        }
                    }
                    Some("tool_use") => {
                        let id = block.get("id").and_then(|v| v.as_str()).unwrap_or("");
                        let name = block.get("name").and_then(|v| v.as_str()).unwrap_or("");
                        let input = block.get("input").cloned().unwrap_or_else(|| json!({}));
                        let input = if input.is_object() { input } else { json!({}) };
                        let args = serde_json::to_string(&input).unwrap_or_else(|_| "{}".into());
                        tool_calls.push(json!({
                        "id": id,
                        "type": "function",
                        "function": {"name": name, "arguments": args}
                        }));
                    }
                    Some("tool_result") => {
                        let tool_use_id = block
                            .get("tool_use_id")
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        let content = convert_tool_result_content(block.get("content"));
                        tool_results.push(json!({
                        "role": "tool",
                        "tool_call_id": tool_use_id,
                        "content": content
                        }));
                    }
                    Some("document") => {
                        // A PDF is not an image. Keep the native block intact
                        // for Anthropic; the upstream boundary translates it
                        // to a file part for supported non-Anthropic targets.
                        content_blocks_raw.push(block.clone());
                    }
                    Some("thinking" | "redacted_thinking") => thinking_blocks.push(block.clone()),
                    Some("image") => {
                        let source = match block.get("source") {
                            Some(s) if s.is_object() => s,
                            _ => {
                                tracing::warn!(
                                    block_type = %btype.unwrap_or(""),
                                    "anthropic {} block missing `source`; dropping block",
                                    btype.unwrap_or("?")
                                );
                                continue;
                            }
                        };
                        match anthropic_source_to_openai_image_url(source) {
                            Some(part) => content_blocks_raw.push(part),
                            None => tracing::warn!(
                                block_type = %btype.unwrap_or(""),
                                "anthropic {} block has unrecognised source shape; dropping block",
                                btype.unwrap_or("?")
                            ),
                        }
                    }
                    Some(other) => {
                        tracing::warn!(
                            block_type = %other,
                            "unsupported anthropic content block type `{other}` dropped during conversion"
                        );
                    }
                    None => {
                        tracing::warn!(
                            "anthropic content block without `type` dropped during conversion"
                        );
                    }
                }
            }

            if content_blocks_raw.iter().any(|block| {
                block.get("type").and_then(Value::as_str) != Some("text")
                    || block.get("cache_control").is_some()
            }) {
                text_parts.clear();
            } else {
                content_blocks_raw.clear();
            }
            if !tool_results.is_empty() {
                for tr in tool_results {
                    out.push(tr);
                }
                if !text_parts.is_empty() || !content_blocks_raw.is_empty() {
                    let mut openai_msg = Map::new();
                    openai_msg.insert("role".to_string(), json!(role));
                    if !content_blocks_raw.is_empty() {
                        if text_parts.is_empty() {
                            openai_msg
                                .insert("content".to_string(), Value::Array(content_blocks_raw));
                        } else {
                            let mut combined = content_blocks_raw;
                            combined.push(json!({"type": "text", "text": text_parts.join("")}));
                            openai_msg.insert("content".to_string(), Value::Array(combined));
                        }
                    } else {
                        openai_msg.insert("content".to_string(), json!(text_parts.join("")));
                    }
                    out.push(Value::Object(openai_msg));
                }
                // A well-formed Anthropic conversation never mixes
                // `tool_result` (user-side) and `tool_use` (assistant-side)
                // in one message, but a malformed payload would otherwise
                // silently lose the tool-use blocks. Emit them as a separate
                // assistant message so the tool calls survive.
                if !tool_calls.is_empty() {
                    out.push(json!({
                        "role": "assistant",
                        "content": Value::Null,
                        "tool_calls": Value::Array(tool_calls)
                    }));
                }
            } else {
                let mut openai_msg = Map::new();
                openai_msg.insert("role".to_string(), json!(role));
                if !content_blocks_raw.is_empty() {
                    if !text_parts.is_empty() {
                        let mut combined = content_blocks_raw;
                        combined.push(json!({"type": "text", "text": text_parts.join("")}));
                        openai_msg.insert("content".to_string(), Value::Array(combined));
                    } else {
                        openai_msg.insert("content".to_string(), Value::Array(content_blocks_raw));
                    }
                } else if !text_parts.is_empty() {
                    openai_msg.insert("content".to_string(), json!(text_parts.join("")));
                } else {
                    openai_msg.insert("content".to_string(), Value::Null);
                }
                if !tool_calls.is_empty() {
                    openai_msg.insert("tool_calls".to_string(), Value::Array(tool_calls));
                }
                out.push(Value::Object(openai_msg));
            }
            if !thinking_blocks.is_empty() {
                if let Some(message) = out[first_message..].iter_mut().find(|message| {
                    message.get("role").and_then(Value::as_str) == Some("assistant")
                }) {
                    message["thinking_blocks"] = Value::Array(thinking_blocks);
                }
            }
        }
        _ => {
            let mut m = Map::new();
            m.insert("role".to_string(), json!(role));
            if let Some(content) = msg.get("content") {
                m.insert("content".to_string(), content.clone());
            }
            out.push(Value::Object(m));
        }
    }
}

fn convert_tool_result_content(content: Option<&Value>) -> Value {
    match content {
        Some(Value::String(text)) => json!(text),
        Some(Value::Array(parts))
            if parts.iter().any(|part| {
                matches!(
                    part.get("type").and_then(Value::as_str),
                    Some("image" | "document")
                )
            }) =>
        {
            Value::Array(
                parts
                    .iter()
                    .map(|part| {
                        if part.get("type").and_then(Value::as_str) == Some("image") {
                            part.get("source")
                                .and_then(anthropic_source_to_openai_image_url)
                                .unwrap_or_else(|| part.clone())
                        } else {
                            part.clone()
                        }
                    })
                    .collect(),
            )
        }
        Some(Value::Array(parts)) => json!(
            parts
                .iter()
                .filter_map(|part| part.get("text").and_then(Value::as_str))
                .collect::<String>()
        ),
        Some(other) => json!(other.to_string()),
        None => json!(""),
    }
}

fn map_anthropic_tool_to_openai(tool: &Value) -> Value {
    let name = tool.get("name").cloned().unwrap_or_else(|| json!(""));
    let description = tool.get("description").cloned();
    let input_schema = tool
        .get("input_schema")
        .cloned()
        .unwrap_or_else(|| json!({"type": "object", "properties": {}}));

    let mut function = Map::new();
    function.insert("name".to_string(), name);
    if let Some(desc) = description {
        function.insert("description".to_string(), desc);
    }
    function.insert("parameters".to_string(), input_schema);

    let mut mapped = json!({"type": "function", "function": Value::Object(function)});
    if let Some(control) = tool.get("cache_control") {
        mapped["cache_control"] = control.clone();
    }
    mapped
}

fn map_anthropic_tool_choice_to_openai(tool_choice: &Value) -> Option<Value> {
    if let Some(s) = tool_choice.as_str() {
        return Some(json!(s));
    }
    let obj = tool_choice.as_object()?;
    let tc_type = obj.get("type").and_then(|v| v.as_str()).unwrap_or("auto");
    Some(match tc_type {
        "auto" => json!("auto"),
        "any" => json!("required"),
        "none" => json!("none"),
        "tool" => {
            let name = obj.get("name").and_then(|v| v.as_str()).unwrap_or("");
            json!({"type": "function", "function": {"name": name}})
        }
        other => json!(other),
    })
}

// ---------------------------------------------------------------------------
// Response conversion: OpenAI → Anthropic
// ---------------------------------------------------------------------------

pub(crate) fn openai_to_anthropic_response(resp: &Value) -> Value {
    let id = resp
        .get("id")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let model = resp
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let choice = resp
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|a| a.first());
    let message = choice.and_then(|c| c.get("message"));

    let mut content: Vec<Value> = Vec::new();
    if let Some(msg) = message {
        // Combined thinking + signature: `thinking_blocks` already carries
        // signature; otherwise lift `reasoning_content` into a thinking
        // block.
        if let Some(blocks) = msg.get("thinking_blocks").and_then(|v| v.as_array()) {
            for block in blocks {
                if block.get("type").and_then(Value::as_str) == Some("redacted_thinking") {
                    content.push(block.clone());
                    continue;
                }
                let thinking = block
                    .get("thinking")
                    .or_else(|| block.get("text"))
                    .cloned()
                    .unwrap_or(json!(""));
                let mut out_block = json!({
                    "type": "thinking",
                    "thinking": thinking,
                });
                if let Some(sig) = block.get("signature") {
                    out_block["signature"] = sig.clone();
                }
                content.push(out_block);
            }
        } else if let Some(text) = msg.get("reasoning_content").and_then(|v| v.as_str()) {
            if !text.is_empty() {
                let mut block = json!({"type": "thinking", "thinking": text});
                if let Some(sig) = msg.get("thinking_signature") {
                    block["signature"] = sig.clone();
                }
                content.push(block);
            }
        }
        if let Some(text) = msg.get("content").and_then(|v| v.as_str()) {
            if !text.is_empty() {
                content.push(json!({"type": "text", "text": text}));
            }
        }
        if let Some(tool_calls) = msg.get("tool_calls").and_then(|v| v.as_array()) {
            for tc in tool_calls {
                let id = tc.get("id").and_then(|v| v.as_str()).unwrap_or("");
                let function = tc.get("function").cloned().unwrap_or_else(|| json!({}));
                let name = function.get("name").and_then(|v| v.as_str()).unwrap_or("");
                let args_str = function
                    .get("arguments")
                    .and_then(|v| v.as_str())
                    .unwrap_or("{}");
                let input: Value = serde_json::from_str(args_str).unwrap_or_else(|_| json!({}));
                content.push(json!({
                "type": "tool_use",
                "id": id,
                "name": name,
                "input": input,
                }));
            }
        }
    }

    if content.is_empty() {
        content.push(json!({"type": "text", "text": ""}));
    }

    let stop_reason = choice
        .and_then(|c| c.get("finish_reason"))
        .and_then(|v| v.as_str())
        .map(map_openai_finish_to_anthropic);

    let usage = resp.get("usage");
    let input_tokens = usage
        .and_then(|u| u.get("prompt_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let output_tokens = usage
        .and_then(|u| u.get("completion_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    let mut anthropic_usage = json!({
    "input_tokens": input_tokens,
    "output_tokens": output_tokens,
    });
    if let Some(details) = usage.and_then(|u| u.get("prompt_tokens_details")) {
        if let Some(cached) = details.get("cached_tokens").and_then(|v| v.as_u64()) {
            anthropic_usage["cache_read_input_tokens"] = json!(cached);
        }
        let cache_creation = details
            .get("cache_write_tokens")
            .and_then(|v| v.as_u64())
            .or_else(|| {
                details
                    .get("cache_creation_tokens")
                    .and_then(|v| v.as_u64())
            });
        if let Some(cc) = cache_creation {
            anthropic_usage["cache_creation_input_tokens"] = json!(cc);
        }
        if let Some(ephemeral_1h) = details
            .get("ephemeral_1h_input_tokens")
            .and_then(|v| v.as_u64())
        {
            anthropic_usage["cache_creation"] = json!({"ephemeral_1h_input_tokens": ephemeral_1h});
        }
    }

    json!({
    "id": id,
    "type": "message",
    "role": "assistant",
    "model": model,
    "content": content,
    "stop_reason": stop_reason,
    "stop_sequence": null,
    "usage": anthropic_usage,
    })
}

fn map_openai_finish_to_anthropic(reason: &str) -> String {
    match reason {
        "stop" => "end_turn".to_string(),
        "tool_calls" => "tool_use".to_string(),
        "length" => "max_tokens".to_string(),
        // LiteLLM's `_FINISH_REASON_MAP` maps Anthropic `"refusal"` ↔ OpenAI
        // `"content_filter"` bidirectionally; the reverse mapping restores
        // `content_filter`/`refusal` → `refusal` so an Anthropic SDK client
        // can distinguish a safety intervention from a natural stop.
        "content_filter" | "refusal" => "refusal".to_string(),
        _ => "end_turn".to_string(),
    }
}

// ---------------------------------------------------------------------------
// Stream converter: OpenAI SSE chunks → Anthropic SSE events
// ---------------------------------------------------------------------------

/// Build an Anthropic-shaped `usage` object for a streaming event
/// (`message_start` or `message_delta`), including the cache-token fields
/// when they are non-zero. Anthropic SDK clients rely on
/// `cache_read_input_tokens` / `cache_creation_input_tokens` for cache
/// accounting; LiteLLM preserves them end-to-end.
fn build_anthropic_streaming_usage(
    input_tokens: u64,
    output_tokens: u64,
    cache_read_input_tokens: Option<u64>,
    cache_creation_input_tokens: Option<u64>,
    cache_creation_1h_input_tokens: Option<u64>,
) -> Value {
    let mut usage = json!({
    "input_tokens": input_tokens,
    "output_tokens": output_tokens,
    });
    if let Some(cached) = cache_read_input_tokens {
        usage["cache_read_input_tokens"] = json!(cached);
    }
    if let Some(cc) = cache_creation_input_tokens {
        usage["cache_creation_input_tokens"] = json!(cc);
    }
    if let Some(eph1h) = cache_creation_1h_input_tokens {
        usage["cache_creation"] = json!({"ephemeral_1h_input_tokens": eph1h});
    }
    usage
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OpenBlockKind {
    Text,
    ToolUse,
    Thinking,
}

#[derive(Debug, Default)]
struct ToolCallBlock {
    anthropic_index: i64,
    started: bool,
    id: String,
    name: String,
    pending_arguments: String,
}

pub(crate) struct AnthropicStreamConverter {
    started: bool,
    finished: bool,
    /// Defer the terminal delta until `[DONE]`/EOF: OpenAI sends its usage-only
    /// chunk after `finish_reason`, and native Messages usage must include it.
    pending_stop_reason: Option<String>,
    message_id: String,
    model: String,
    next_block_index: i64,
    open_block: Option<(i64, OpenBlockKind)>,
    tool_calls: std::collections::HashMap<u64, ToolCallBlock>,
    input_tokens: u64,
    output_tokens: u64,
    cache_read_input_tokens: Option<u64>,
    cache_creation_input_tokens: Option<u64>,
    cache_creation_1h_input_tokens: Option<u64>,
}

impl AnthropicStreamConverter {
    pub(crate) fn new() -> Self {
        Self {
            started: false,
            finished: false,
            pending_stop_reason: None,
            message_id: format!("msg_{}", uuid::Uuid::new_v4()),
            model: String::new(),
            next_block_index: 0,
            open_block: None,
            tool_calls: std::collections::HashMap::new(),
            input_tokens: 0,
            output_tokens: 0,
            cache_read_input_tokens: None,
            cache_creation_input_tokens: None,
            cache_creation_1h_input_tokens: None,
        }
    }

    pub(crate) fn convert_chunk(&mut self, chunk: &Value) -> Vec<Value> {
        if self.finished {
            return Vec::new();
        }
        if let Some(error) = chunk.get("error").filter(|e| !e.is_null()) {
            self.finished = true;
            let mut ev = Vec::new();
            // Close any open content block before the terminal error so the
            // event stream stays balanced (every `content_block_start` pairs
            // with a `content_block_stop`). Anthropic SDK clients that track
            // block state would otherwise see an unbalanced start.
            if let Some((idx, _)) = self.open_block.take() {
                ev.push(json!({"type": "content_block_stop", "index": idx}));
            }
            ev.push(json!({"type": "error", "error": error}));
            return ev;
        }

        // Merge usage BEFORE emitting `message_start` so a first chunk that
        // already carries `usage.prompt_tokens` (Anthropic-backed streams via
        // the provider transform) is reflected in `message_start.usage`
        // instead of the 0/0 placeholder, and so cache-token fields are
        // tracked from the first chunk.
        if let Some(usage) = chunk.get("usage").filter(|u| !u.is_null()) {
            // Max-merge rather than overwrite, matching
            // `orchestration::usage::merge_streaming_usage`: Anthropic-backed
            // streams report prompt tokens only on the first chunk and
            // `prompt_tokens: 0` on the finish chunk, so a plain overwrite
            // would clobber the real input token count back to zero.
            if let Some(p) = usage.get("prompt_tokens").and_then(|v| v.as_u64()) {
                self.input_tokens = self.input_tokens.max(p);
            }
            if let Some(c) = usage.get("completion_tokens").and_then(|v| v.as_u64()) {
                self.output_tokens = self.output_tokens.max(c);
            }
            if let Some(details) = usage.get("prompt_tokens_details") {
                if let Some(cached) = details.get("cached_tokens").and_then(|v| v.as_u64()) {
                    self.cache_read_input_tokens =
                        Some(self.cache_read_input_tokens.unwrap_or(0).max(cached));
                }
                let cc = details
                    .get("cache_write_tokens")
                    .and_then(|v| v.as_u64())
                    .or_else(|| {
                        details
                            .get("cache_creation_tokens")
                            .and_then(|v| v.as_u64())
                    });
                if let Some(cc) = cc {
                    self.cache_creation_input_tokens =
                        Some(self.cache_creation_input_tokens.unwrap_or(0).max(cc));
                }
                if let Some(eph1h) = details
                    .get("ephemeral_1h_input_tokens")
                    .and_then(|v| v.as_u64())
                {
                    self.cache_creation_1h_input_tokens =
                        Some(self.cache_creation_1h_input_tokens.unwrap_or(0).max(eph1h));
                }
            }
        }

        let mut events = Vec::new();

        if self.pending_stop_reason.is_some() {
            // Usage above is still merged, but content after a finish and
            // duplicate finish reasons do not reopen stopped blocks.
            return events;
        }

        if !self.started {
            self.started = true;
            if let Some(id) = chunk.get("id").and_then(|v| v.as_str()) {
                if !id.is_empty() {
                    self.message_id = id.to_string();
                }
            }
            if let Some(m) = chunk.get("model").and_then(|v| v.as_str()) {
                self.model = m.to_string();
            }
            events.push(json!({
            "type": "message_start",
            "message": {
            "id": self.message_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": self.model,
            "stop_reason": null,
            "stop_sequence": null,
            "usage": build_anthropic_streaming_usage(
                self.input_tokens,
                self.output_tokens,
                self.cache_read_input_tokens,
                self.cache_creation_input_tokens,
                self.cache_creation_1h_input_tokens,
            )
            }
            }));
        }

        if let Some(choices) = chunk.get("choices").and_then(|c| c.as_array()) {
            if let Some(choice) = choices.first() {
                if let Some(delta) = choice.get("delta") {
                    if let Some(text) = delta.get("reasoning_content").and_then(|v| v.as_str()) {
                        if !text.is_empty() {
                            let need_new =
                                !matches!(self.open_block, Some((_, OpenBlockKind::Thinking)));
                            if need_new {
                                if let Some((idx, _)) = self.open_block.take() {
                                    events
                                        .push(json!({"type": "content_block_stop", "index": idx}));
                                }
                                let new_idx = self.next_block_index;
                                self.next_block_index += 1;
                                self.open_block = Some((new_idx, OpenBlockKind::Thinking));
                                events.push(json!({
                                    "type": "content_block_start",
                                    "index": new_idx,
                                    "content_block": {"type": "thinking", "thinking": ""}
                                }));
                            }
                            if let Some((idx, _)) = self.open_block {
                                events.push(json!({
                                    "type": "content_block_delta",
                                    "index": idx,
                                    "delta": {"type": "thinking_delta", "thinking": text}
                                }));
                            }
                        }
                    }
                    if let Some(blocks) = delta.get("thinking_blocks").and_then(|v| v.as_array()) {
                        for block in blocks {
                            if let Some(sig) = block.get("signature").and_then(|v| v.as_str())
                                && !sig.is_empty()
                                && matches!(self.open_block, Some((_, OpenBlockKind::Thinking)))
                            {
                                if let Some((idx, _)) = self.open_block {
                                    events.push(json!({
                                        "type": "content_block_delta",
                                        "index": idx,
                                        "delta": {"type": "signature_delta", "signature": sig}
                                    }));
                                }
                            }
                        }
                    }
                    if let Some(text) = delta.get("content").and_then(|v| v.as_str()) {
                        if !text.is_empty() {
                            let need_new_text_block = match self.open_block {
                                None => true,
                                Some((_, OpenBlockKind::Text)) => false,
                                Some((_, OpenBlockKind::ToolUse | OpenBlockKind::Thinking)) => true,
                            };
                            if need_new_text_block {
                                if let Some((idx, _)) = self.open_block.take() {
                                    events
                                        .push(json!({"type": "content_block_stop", "index": idx}));
                                }
                                let new_idx = self.next_block_index;
                                self.next_block_index += 1;
                                self.open_block = Some((new_idx, OpenBlockKind::Text));
                                events.push(json!({
                                "type": "content_block_start",
                                "index": new_idx,
                                "content_block": {"type": "text", "text": ""}
                                }));
                            }
                            // Invariant: the ensure-logic above always leaves
                            // a text block open. Debug-assert it and fall back
                            // to skipping the delta rather than panicking in
                            // release builds.
                            if let Some((idx, _)) = self.open_block {
                                events.push(json!({
                                "type": "content_block_delta",
                                "index": idx,
                                "delta": {"type": "text_delta", "text": text}
                                }));
                            } else {
                                debug_assert!(false, "text block must be open");
                            }
                        }
                    }

                    if let Some(tool_calls) = delta.get("tool_calls").and_then(|v| v.as_array()) {
                        for tool_call in tool_calls {
                            let openai_index =
                                tool_call.get("index").and_then(|v| v.as_u64()).unwrap_or(0);
                            // Defence-in-depth: bound the per-stream tool-call
                            // map so a pathological index flood cannot grow it
                            // unboundedly.
                            if self.tool_calls.len() >= MAX_STREAM_TOOL_CALLS
                                && !self.tool_calls.contains_key(&openai_index)
                            {
                                tracing::warn!(
                                    tool_call_index = openai_index,
                                    "dropping tool call beyond MAX_STREAM_TOOL_CALLS cap"
                                );
                                continue;
                            }
                            let entry = self.tool_calls.entry(openai_index).or_default();

                            if let Some(id) = tool_call.get("id").and_then(|v| v.as_str()) {
                                if entry.id.is_empty() {
                                    entry.id = id.to_string();
                                }
                            }
                            if let Some(name) = tool_call
                                .get("function")
                                .and_then(|f| f.get("name"))
                                .and_then(|v| v.as_str())
                            {
                                entry.name = name.to_string();
                            }

                            // Open the tool-use content block once both `id`
                            // and `name` are known. OpenAI's strict shape sends
                            // them together in the first delta, but other
                            // providers split them across chunks; requiring them
                            // in a single delta would silently drop the whole
                            // tool call (block start plus all argument deltas).
                            if !entry.started && !entry.id.is_empty() && !entry.name.is_empty() {
                                if let Some((idx, _)) = self.open_block.take() {
                                    events
                                        .push(json!({"type": "content_block_stop", "index": idx}));
                                }
                                let new_idx = self.next_block_index;
                                self.next_block_index += 1;
                                entry.anthropic_index = new_idx;
                                entry.started = true;
                                let id_val = entry.id.clone();
                                let name_val = entry.name.clone();
                                events.push(json!({
                                "type": "content_block_start",
                                "index": new_idx,
                                "content_block": {
                                "type": "tool_use",
                                "id": id_val,
                                "name": name_val,
                                "input": {}
                                }
                                }));
                                self.open_block = Some((new_idx, OpenBlockKind::ToolUse));
                                if !entry.pending_arguments.is_empty() {
                                    let pending = std::mem::take(&mut entry.pending_arguments);
                                    events.push(json!({
                                    "type": "content_block_delta",
                                    "index": new_idx,
                                    "delta": {"type": "input_json_delta", "partial_json": pending}
                                    }));
                                }
                            }

                            if let Some(args) = tool_call
                                .get("function")
                                .and_then(|f| f.get("arguments"))
                                .and_then(|v| v.as_str())
                            {
                                if !args.is_empty() {
                                    let started = entry.started;
                                    let anthropic_index = entry.anthropic_index;
                                    // Only emit a delta when this block is
                                    // still the active one. The Anthropic
                                    // protocol requires sequential content
                                    // blocks, so once a newer block opened
                                    // (`open_block` moved on) emitting a delta
                                    // for the stopped index would be a
                                    // delta-after-stop violation. Skip the
                                    // fragment rather than corrupt the stream.
                                    let is_active = started
                                        && self.open_block
                                            == Some((anthropic_index, OpenBlockKind::ToolUse));
                                    if is_active {
                                        events.push(json!({
                                        "type": "content_block_delta",
                                        "index": anthropic_index,
                                        "delta": {"type": "input_json_delta", "partial_json": args}
                                        }));
                                    } else if !started {
                                        // Block not open yet (id/name still
                                        // arriving); buffer until open so no
                                        // argument delta is lost.
                                        // Defence-in-depth: cap the buffer so
                                        // a misbehaving upstream streaming
                                        // endless arguments into an unopened
                                        // block cannot hold unbounded memory.
                                        if entry.pending_arguments.len() + args.len()
                                            <= MAX_STREAM_BUFFER_BYTES
                                        {
                                            entry.pending_arguments.push_str(args);
                                        }
                                    }
                                    // else: block was started but is no longer
                                    // the active one — a non-conforming
                                    // upstream interleaved tool calls. Skip.
                                }
                            }
                        }
                    }
                }

                if let Some(finish) = choice.get("finish_reason").and_then(|v| v.as_str()) {
                    self.pending_stop_reason = Some(map_openai_finish_to_anthropic(finish));
                    if let Some((idx, _)) = self.open_block.take() {
                        events.push(json!({"type": "content_block_stop", "index": idx}));
                    }
                }
            }
        }

        events
    }

    pub(crate) fn finish_events(&mut self) -> Vec<Value> {
        if self.finished {
            return Vec::new();
        }
        self.finished = true;
        let mut events = Vec::new();
        if let Some((idx, _)) = self.open_block.take() {
            events.push(json!({"type": "content_block_stop", "index": idx}));
        }
        if let Some(stop_reason) = self.pending_stop_reason.take() {
            events.push(json!({
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": null},
                "usage": build_anthropic_streaming_usage(
                    self.input_tokens, self.output_tokens, self.cache_read_input_tokens,
                    self.cache_creation_input_tokens, self.cache_creation_1h_input_tokens,
                )
            }));
        }
        events.push(json!({"type": "message_stop"}));
        events
    }

    /// Emit a `content_block_stop` for any currently-open content block
    /// without marking the converter finished. Used by the transport-level
    /// stream-read-error path (which synthesises its own terminal `error`
    /// event outside the converter) to keep the block stream balanced —
    /// every emitted `content_block_start` must pair with a
    /// `content_block_stop`, matching the chunk-level error arm in
    /// [`Self::convert_chunk`]. Returns an empty `Vec` when no block is open.
    pub(crate) fn close_open_block(&mut self) -> Vec<Value> {
        let mut events = Vec::new();
        if let Some((idx, _)) = self.open_block.take() {
            events.push(json!({"type": "content_block_stop", "index": idx}));
        }
        events
    }
}

/// Bridge an SSE framing layer and the converter: decode one raw SSE frame
/// into an OpenAI chunk and map it through the converter to Anthropic events.
/// Pure (no axum/tokio), so it lives next to the converter it drives.
pub(crate) fn process_openai_sse_event(
    converter: &mut AnthropicStreamConverter,
    raw: &str,
) -> Vec<Value> {
    let data_str = match extract_sse_data(raw) {
        Some(s) => s,
        None => return Vec::new(),
    };

    if data_str == "[DONE]" {
        return converter.finish_events();
    }

    let chunk: Value = match serde_json::from_str(data_str) {
        Ok(v) => v,
        Err(e) => {
            tracing::debug!(
                error = %e,
                payload_preview = %data_str.chars().take(120).collect::<String>(),
                "dropping unparseable Anthropic SSE data payload"
            );
            return Vec::new();
        }
    };

    converter.convert_chunk(&chunk)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replay_preserves_signed_thinking_and_mixed_content_order() {
        let request = json!({"model":"m","messages":[{"role":"assistant","content":[
            {"type":"thinking","thinking":"reason","signature":"signed"},
            {"type":"redacted_thinking","data":"opaque"},
            {"type":"text","text":"before"},
            {"type":"image","source":{"type":"url","url":"https://example.test/image"}},
            {"type":"text","text":"after"}
        ]}]});
        let chat = anthropic_to_openai_request(&request);
        let message = &chat["messages"][0];
        assert_eq!(message["thinking_blocks"][0]["signature"], "signed");
        assert_eq!(message["thinking_blocks"][1]["data"], "opaque");
        assert_eq!(message["content"][0]["text"], "before");
        assert_eq!(message["content"][1]["type"], "image_url");
        assert_eq!(message["content"][2]["text"], "after");
        let native = openai_to_anthropic_response(
            &json!({"choices":[{"message":{"thinking_blocks":[{"type":"redacted_thinking","data":"opaque"}]}}]}),
        );
        assert_eq!(
            native["content"][0],
            json!({"type":"redacted_thinking","data":"opaque"})
        );
    }
    use serde_json::json;

    #[test]
    fn anthropic_request_system_string_prepended_as_message() {
        let req = json!({
        "model": "claude-3-5-sonnet",
        "system": "Be brief.",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 100,
        });
        let openai = anthropic_to_openai_request(&req);
        let messages = openai["messages"].as_array().expect("messages");
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[0]["content"], "Be brief.");
        assert_eq!(messages[1]["role"], "user");
        assert_eq!(messages[1]["content"], "Hi");
        assert_eq!(openai["max_tokens"], 100);
    }

    #[test]
    fn anthropic_request_system_array_joined() {
        let req = json!({
        "model": "claude",
        "system": [{"type": "text", "text": "Be brief."}, {"type": "text", "text": "Be kind."}],
        "messages": [],
        "max_tokens": 10,
        });
        let openai = anthropic_to_openai_request(&req);
        assert_eq!(openai["messages"][0]["role"], "system");
        assert_eq!(openai["messages"][0]["content"], "Be brief. Be kind.");
    }

    #[test]
    fn anthropic_request_no_system_no_system_message() {
        let req = json!({
        "model": "claude",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 10,
        });
        let openai = anthropic_to_openai_request(&req);
        let messages = openai["messages"].as_array().expect("messages");
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["role"], "user");
    }

    #[test]
    fn anthropic_request_tools_converted_to_openai_format() {
        let req = json!({
        "model": "claude",
        "messages": [],
        "max_tokens": 10,
        "tools": [{
        "name": "get_weather",
        "description": "Get weather",
        "input_schema": {"type": "object", "properties": {"location": {"type": "string"}}}
        }],
        });
        let openai = anthropic_to_openai_request(&req);
        let tools = openai["tools"].as_array().expect("tools");
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[0]["function"]["name"], "get_weather");
        assert_eq!(tools[0]["function"]["description"], "Get weather");
        assert!(tools[0]["function"]["parameters"]["properties"]["location"].is_object());
    }

    #[test]
    fn anthropic_request_tool_choice_auto() {
        let req = json!({
        "model": "claude",
        "messages": [],
        "max_tokens": 10,
        "tool_choice": {"type": "auto"},
        });
        let openai = anthropic_to_openai_request(&req);
        assert_eq!(openai["tool_choice"], "auto");
    }

    #[test]
    fn anthropic_request_tool_choice_any_to_required() {
        let req = json!({
        "model": "claude",
        "messages": [],
        "max_tokens": 10,
        "tool_choice": {"type": "any"},
        });
        let openai = anthropic_to_openai_request(&req);
        assert_eq!(openai["tool_choice"], "required");
    }

    #[test]
    fn anthropic_request_tool_choice_tool() {
        let req = json!({
        "model": "claude",
        "messages": [],
        "max_tokens": 10,
        "tool_choice": {"type": "tool", "name": "get_weather"},
        });
        let openai = anthropic_to_openai_request(&req);
        assert_eq!(openai["tool_choice"]["type"], "function");
        assert_eq!(openai["tool_choice"]["function"]["name"], "get_weather");
    }

    #[test]
    fn anthropic_request_stop_sequences_mapped() {
        let req = json!({
        "model": "claude",
        "messages": [],
        "max_tokens": 10,
        "stop_sequences": ["END", "STOP"],
        });
        let openai = anthropic_to_openai_request(&req);
        assert_eq!(openai["stop"], json!(["END", "STOP"]));
    }

    #[test]
    fn anthropic_request_stream_passthrough() {
        let req = json!({
        "model": "claude",
        "messages": [],
        "max_tokens": 10,
        "stream": true,
        });
        let openai = anthropic_to_openai_request(&req);
        assert_eq!(openai["stream"], true);
    }

    #[test]
    fn anthropic_request_cache_opt_in_passthrough() {
        // Regression: the converter must forward `cache: {"insert": true}` so
        // the chat pipeline's CacheHook can see the opt-in.
        let req = json!({
        "model": "claude",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 10,
        "cache": {"insert": true},
        });
        let openai = anthropic_to_openai_request(&req);
        assert_eq!(openai["cache"], json!({"insert": true}));
        assert!(
            crate::hooks::cache_hook::request_wants_cache(&openai),
            "the converted request must satisfy the cache opt-in check"
        );

        // Absent or null cache fields must not produce one.
        let req = json!({"model": "claude", "messages": [], "max_tokens": 10});
        assert!(anthropic_to_openai_request(&req).get("cache").is_none());
        let req = json!({"model": "claude", "messages": [], "max_tokens": 10, "cache": null});
        assert!(anthropic_to_openai_request(&req).get("cache").is_none());
    }

    #[test]
    fn anthropic_tool_use_message_converted_to_tool_calls() {
        let req = json!({
        "model": "claude",
        "messages": [{
        "role": "assistant",
        "content": [
        {"type": "text", "text": "Let me check."},
        {"type": "tool_use", "id": "tool_1", "name": "get_weather", "input": {"location": "NYC"}}
        ]
        }],
        "max_tokens": 10,
        });
        let openai = anthropic_to_openai_request(&req);
        let msg = &openai["messages"][0];
        assert_eq!(msg["role"], "assistant");
        assert_eq!(msg["content"], "Let me check.");
        assert_eq!(msg["tool_calls"][0]["id"], "tool_1");
        assert_eq!(msg["tool_calls"][0]["type"], "function");
        assert_eq!(msg["tool_calls"][0]["function"]["name"], "get_weather");
        assert!(msg["tool_calls"][0]["function"]["arguments"].is_string());
    }

    #[test]
    fn anthropic_tool_result_converted_to_tool_message() {
        let req = json!({
        "model": "claude",
        "messages": [{
        "role": "user",
        "content": [
        {"type": "tool_result", "tool_use_id": "tool_1", "content": "Sunny, 72F"}
        ]
        }],
        "max_tokens": 10,
        });
        let openai = anthropic_to_openai_request(&req);
        let msg = &openai["messages"][0];
        assert_eq!(msg["role"], "tool");
        assert_eq!(msg["tool_call_id"], "tool_1");
        assert_eq!(msg["content"], "Sunny, 72F");
    }

    // ================================================================
    // C24 MEDIUM: image / document content blocks are converted, not dropped
    // ================================================================

    /// C24 MEDIUM: an Anthropic user message carrying a base64 `image` block
    /// must convert to an OpenAI `image_url` content part so the routed
    /// model can actually see the image. Before the fix the catchall `_ =>`
    /// arm silently dropped the block and the message reached the chat
    /// pipeline as `content: null`.
    #[test]
    fn anthropic_request_image_block_converted_to_image_url() {
        let req = json!({
        "model": "claude",
        "messages": [{
        "role": "user",
        "content": [
        {"type": "text", "text": "what is this?"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0K"}}
        ]
        }],
        "max_tokens": 10,
        });
        let openai = anthropic_to_openai_request(&req);
        let msg = &openai["messages"][0];
        assert_eq!(msg["role"], "user");
        let content = msg["content"]
            .as_array()
            .expect("content must be an array when an image block is present");
        assert!(
            !content.is_empty(),
            "image block must not be silently dropped: {content:?}"
        );
        let image_part = content
            .iter()
            .find(|p| p["type"] == "image_url")
            .expect("converted message must include an image_url part");
        assert_eq!(
            image_part["image_url"]["url"],
            "data:image/png;base64,iVBORw0K"
        );
        // The text part is preserved alongside the image_url part.
        assert!(
            content
                .iter()
                .any(|p| p["type"] == "text" && p["text"] == "what is this?"),
            "text part must be preserved: {content:?}"
        );
    }

    /// C24 MEDIUM variant: the URL source shape maps to a plain `image_url`
    /// pointing at the upstream URL.
    #[test]
    fn anthropic_request_image_url_source_converted() {
        let req = json!({
        "model": "claude",
        "messages": [{
        "role": "user",
        "content": [
        {"type": "image", "source": {"type": "url", "url": "https://example.com/cat.png"}}
        ]
        }],
        "max_tokens": 10,
        });
        let openai = anthropic_to_openai_request(&req);
        let content = &openai["messages"][0]["content"];
        assert_eq!(content[0]["type"], "image_url");
        assert_eq!(
            content[0]["image_url"]["url"],
            "https://example.com/cat.png"
        );
    }

    /// PDFs retain their document shape for native Anthropic targets. The
    /// upstream boundary maps them to file parts for other supported targets.
    #[test]
    fn anthropic_request_document_block_is_not_an_image() {
        let req = json!({
        "model": "claude",
        "messages": [{
        "role": "user",
        "content": [
        {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "JVBERi0="}}
        ]
        }],
        "max_tokens": 10,
        });
        let openai = anthropic_to_openai_request(&req);
        let content = &openai["messages"][0]["content"];
        assert_eq!(content[0], req["messages"][0]["content"][0]);
        assert!(content[0].get("image_url").is_none());
    }

    #[test]
    fn documents_in_tool_results_keep_their_content_and_order() {
        let document = json!({"type":"document","source":{"type":"base64","media_type":"application/pdf","data":"JVBERi0="}});
        let request = json!({"model":"claude","messages":[{"role":"user","content":[{
            "type":"tool_result","tool_use_id":"call-1","content":[{"type":"text","text":"report"},document.clone()]
        }]}]});
        let converted = anthropic_to_openai_request(&request);
        assert_eq!(converted["messages"][0]["role"], "tool");
        assert_eq!(converted["messages"][0]["content"][0]["text"], "report");
        assert_eq!(converted["messages"][0]["content"][1], document);
    }

    /// A message that carries ONLY an image block must still produce a
    /// non-null content array (regression for the prior silent-drop path
    /// which emitted `content: null`).
    #[test]
    fn anthropic_request_only_image_block_yields_non_null_content() {
        let req = json!({
        "model": "claude",
        "messages": [{
        "role": "user",
        "content": [
        {"type": "image", "source": {"type": "url", "url": "https://example.com/i.png"}}
        ]
        }],
        "max_tokens": 10,
        });
        let openai = anthropic_to_openai_request(&req);
        let msg = &openai["messages"][0];
        assert!(
            !msg["content"].is_null(),
            "image-only message must not collapse to content: null: {msg}"
        );
        assert_eq!(msg["content"][0]["type"], "image_url");
    }

    #[test]
    fn openai_response_converted_to_anthropic() {
        let resp = json!({
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": "gpt-4o",
        "choices": [{
        "index": 0,
        "message": {"role": "assistant", "content": "Hello!"},
        "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        });
        let anthropic = openai_to_anthropic_response(&resp);
        assert_eq!(anthropic["type"], "message");
        assert_eq!(anthropic["role"], "assistant");
        assert_eq!(anthropic["model"], "gpt-4o");
        assert_eq!(anthropic["content"][0]["type"], "text");
        assert_eq!(anthropic["content"][0]["text"], "Hello!");
        assert_eq!(anthropic["stop_reason"], "end_turn");
        assert_eq!(anthropic["usage"]["input_tokens"], 10);
        assert_eq!(anthropic["usage"]["output_tokens"], 5);
    }

    #[test]
    fn openai_response_tool_calls_converted_to_tool_use() {
        let resp = json!({
        "id": "chatcmpl-1",
        "model": "gpt-4o",
        "choices": [{
        "index": 0,
        "message": {
        "role": "assistant",
        "content": null,
        "tool_calls": [{
        "id": "call_1",
        "type": "function",
        "function": {"name": "get_weather", "arguments": "{\"location\":\"NYC\"}"}
        }]
        },
        "finish_reason": "tool_calls"
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        });
        let anthropic = openai_to_anthropic_response(&resp);
        assert_eq!(anthropic["content"][0]["type"], "tool_use");
        assert_eq!(anthropic["content"][0]["id"], "call_1");
        assert_eq!(anthropic["content"][0]["name"], "get_weather");
        assert_eq!(anthropic["content"][0]["input"]["location"], "NYC");
        assert_eq!(anthropic["stop_reason"], "tool_use");
    }

    #[test]
    fn openai_response_finish_reason_mapping() {
        assert_eq!(map_openai_finish_to_anthropic("stop"), "end_turn");
        assert_eq!(map_openai_finish_to_anthropic("length"), "max_tokens");
        assert_eq!(map_openai_finish_to_anthropic("tool_calls"), "tool_use");
        assert_eq!(map_openai_finish_to_anthropic("unknown"), "end_turn");
        // C24 LOW: content_filter / refusal must map to Anthropic "refusal",
        // not the catch-all "end_turn", so a safety intervention is
        // distinguishable from a natural stop (LiteLLM `_FINISH_REASON_MAP`).
        assert_eq!(map_openai_finish_to_anthropic("content_filter"), "refusal");
        assert_eq!(map_openai_finish_to_anthropic("refusal"), "refusal");
    }

    #[test]
    fn stream_converter_emits_message_start_then_deltas() {
        let mut converter = AnthropicStreamConverter::new();

        let chunk1 = json!({
        "id": "chatcmpl-1",
        "model": "gpt-4o",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": null}]
        });
        let events1 = converter.convert_chunk(&chunk1);
        assert_eq!(events1.len(), 1);
        assert_eq!(events1[0]["type"], "message_start");

        let chunk2 = json!({
        "choices": [{"index": 0, "delta": {"content": "Hi"}, "finish_reason": null}]
        });
        let events2 = converter.convert_chunk(&chunk2);
        assert_eq!(events2.len(), 2);
        assert_eq!(events2[0]["type"], "content_block_start");
        assert_eq!(events2[1]["type"], "content_block_delta");
        assert_eq!(events2[1]["delta"]["text"], "Hi");

        let chunk3 = json!({
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {"completion_tokens": 2}
        });
        let events3 = converter.convert_chunk(&chunk3);
        assert_eq!(events3[0]["type"], "content_block_stop");
        assert_eq!(events3.len(), 1, "terminal delta waits for trailing usage");
        let terminal = converter.finish_events();
        assert_eq!(terminal[0]["type"], "message_delta");
        assert_eq!(terminal[0]["delta"]["stop_reason"], "end_turn");
    }

    #[test]
    fn stream_converter_finish_emits_message_stop() {
        let mut converter = AnthropicStreamConverter::new();
        let chunk = json!({
        "id": "x",
        "model": "m",
        "choices": [{"index": 0, "delta": {"content": "Hi"}, "finish_reason": null}]
        });
        let _ = converter.convert_chunk(&chunk);
        let finish = converter.finish_events();
        assert!(finish.iter().any(|e| e["type"] == "message_stop"));
    }

    #[test]
    fn stream_converter_finish_idempotent() {
        let mut converter = AnthropicStreamConverter::new();
        let first = converter.finish_events();
        let second = converter.finish_events();
        assert!(!first.is_empty());
        assert!(second.is_empty());
    }

    #[test]
    fn stream_error_does_not_emit_message_stop() {
        let mut converter = AnthropicStreamConverter::new();
        let events = converter.convert_chunk(&json!({
            "error": {"message": "capacity exhausted", "code": 429}
        }));
        assert_eq!(events.len(), 1);
        assert_eq!(events[0]["type"], "error");
        assert_eq!(events[0]["error"]["code"], 429);
        assert!(converter.finish_events().is_empty());
    }

    #[test]
    fn stream_null_error_does_not_produce_error_event() {
        let mut converter = AnthropicStreamConverter::new();
        let chunk = json!({
            "id": "chatcmpl-1",
            "model": "gpt-4o",
            "choices": [{"index": 0, "delta": {"content": "Hi"}, "finish_reason": null}],
            "error": null
        });
        let events = converter.convert_chunk(&chunk);
        assert!(
            events.iter().all(|e| e["type"] != "error"),
            "null error must not produce an error event: {events:?}"
        );
        assert!(!converter.finished, "converter must not be marked finished");
        assert!(
            events.iter().any(|e| e["type"] == "message_start"),
            "stream should still progress normally"
        );
        assert!(
            events
                .iter()
                .any(|e| e["type"] == "content_block_delta" && e["delta"]["text"] == "Hi"),
            "content delta must still be emitted"
        );
    }

    #[test]
    fn extract_sse_data_handles_space_and_no_space() {
        assert_eq!(extract_sse_data("data: {\"a\":1}"), Some("{\"a\":1}"));
        assert_eq!(extract_sse_data("data:{\"a\":1}"), Some("{\"a\":1}"));
        assert_eq!(extract_sse_data("data: [DONE]"), Some("[DONE]"));
        assert_eq!(extract_sse_data("not data"), None);
    }

    #[test]
    fn process_done_returns_message_stop() {
        let mut converter = AnthropicStreamConverter::new();
        let chunk = json!({
        "id": "x",
        "model": "m",
        "choices": [{"index": 0, "delta": {}, "finish_reason": null}]
        });
        let _ = converter.convert_chunk(&chunk);
        let events = process_openai_sse_event(&mut converter, "data: [DONE]");
        assert!(events.iter().any(|e| e["type"] == "message_stop"));
    }

    #[test]
    fn stream_tool_calls_produce_content_block_start_and_delta() {
        let mut converter = AnthropicStreamConverter::new();

        let chunk1 = json!({
        "id": "chatcmpl-tools",
        "model": "gpt-4o",
        "choices": [{
        "index": 0,
        "delta": {
        "role": "assistant",
        "tool_calls": [{
        "index": 0,
        "id": "call_1",
        "type": "function",
        "function": {"name": "get_weather", "arguments": ""}
        }]
        },
        "finish_reason": null
        }]
        });
        let events1 = converter.convert_chunk(&chunk1);
        assert!(
            events1.iter().any(|e| e["type"] == "message_start"),
            "missing message_start: {events1:?}"
        );
        let block_start = events1
            .iter()
            .find(|e| e["type"] == "content_block_start")
            .expect("missing content_block_start for tool_use: see events1");
        assert_eq!(block_start["content_block"]["type"], "tool_use");
        assert_eq!(block_start["content_block"]["id"], "call_1");
        assert_eq!(block_start["content_block"]["name"], "get_weather");

        let chunk2 = json!({
        "choices": [{
        "index": 0,
        "delta": {
        "tool_calls": [{"index": 0, "function": {"arguments": "{\"location\":"}}]
        },
        "finish_reason": null
        }]
        });
        let events2 = converter.convert_chunk(&chunk2);
        let arg_delta = events2
            .iter()
            .find(|e| {
                e["type"] == "content_block_delta" && e["delta"]["type"] == "input_json_delta"
            })
            .expect("missing input_json_delta after arguments fragment");
        assert_eq!(arg_delta["delta"]["partial_json"], "{\"location\":");

        let chunk3 = json!({
        "choices": [{
        "index": 0,
        "delta": {
        "tool_calls": [{"index": 0, "function": {"arguments": "\"NYC\"}"}}]
        },
        "finish_reason": "tool_calls"
        }]
        });
        let events3 = converter.convert_chunk(&chunk3);
        assert!(
            events3.iter().any(|e| e["type"] == "content_block_delta"
                && e["delta"]["type"] == "input_json_delta"
                && e["delta"]["partial_json"] == "\"NYC\"}"),
            "second arguments fragment not emitted: {events3:?}"
        );
        assert!(
            events3.iter().any(|e| e["type"] == "content_block_stop"),
            "missing content_block_stop on finish: {events3:?}"
        );
        let terminal = converter.finish_events();
        let msg_delta = terminal
            .iter()
            .find(|e| e["type"] == "message_delta")
            .expect("missing message_delta on finish");
        assert_eq!(msg_delta["delta"]["stop_reason"], "tool_use");
    }

    #[test]
    fn stream_usage_accumulated_in_message_delta() {
        let mut converter = AnthropicStreamConverter::new();

        let chunk1 = json!({
        "id": "chatcmpl-usage",
        "model": "gpt-4o",
        "choices": [{
        "index": 0,
        "delta": {"role": "assistant", "content": "Hi"},
        "finish_reason": null
        }]
        });
        let _ = converter.convert_chunk(&chunk1);

        let chunk2 = json!({
        "choices": [{
        "index": 0,
        "delta": {"content": "!"},
        "finish_reason": null
        }],
        "usage": {"prompt_tokens": 50, "completion_tokens": 3, "total_tokens": 53}
        });
        let _ = converter.convert_chunk(&chunk2);

        let chunk3 = json!({
        "choices": [{
        "index": 0,
        "delta": {},
        "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 50, "completion_tokens": 7, "total_tokens": 57}
        });
        converter.convert_chunk(&chunk3);
        let events = converter.finish_events();
        let msg_delta = events
            .iter()
            .find(|e| e["type"] == "message_delta")
            .expect("missing message_delta on finish");
        assert_eq!(
            msg_delta["usage"]["output_tokens"], 7,
            "accumulated output_tokens not forwarded: {events:?}"
        );
        assert_eq!(
            msg_delta["usage"]["input_tokens"], 50,
            "accumulated input_tokens not forwarded: {events:?}"
        );
    }

    #[test]
    fn stream_usage_max_merged_not_clobbered_by_finish_chunk() {
        // Anthropic-backed streams split usage across chunks: the first
        // (role) chunk carries the real `prompt_tokens`, and the finish chunk
        // carries `prompt_tokens: 0` plus the real `completion_tokens`. The
        // converter must max-merge (like `merge_streaming_usage`) so the
        // finish chunk cannot clobber the input token count back to zero.
        let mut converter = AnthropicStreamConverter::new();

        let role_chunk = json!({
        "id": "chatcmpl-anthropic",
        "model": "claude-3-5-sonnet",
        "choices": [{
        "index": 0,
        "delta": {"role": "assistant", "content": ""},
        "finish_reason": null
        }],
        "usage": {"prompt_tokens": 42, "completion_tokens": 0, "total_tokens": 42}
        });
        let _ = converter.convert_chunk(&role_chunk);

        let text_chunk = json!({
        "choices": [{
        "index": 0,
        "delta": {"content": "Hi"},
        "finish_reason": null
        }]
        });
        let _ = converter.convert_chunk(&text_chunk);

        let finish_chunk = json!({
        "choices": [{
        "index": 0,
        "delta": {},
        "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 13, "total_tokens": 13}
        });
        converter.convert_chunk(&finish_chunk);
        let events = converter.finish_events();
        let msg_delta = events
            .iter()
            .find(|e| e["type"] == "message_delta")
            .expect("missing message_delta on finish");
        assert_eq!(
            msg_delta["usage"]["input_tokens"], 42,
            "finish chunk's prompt_tokens: 0 must not clobber the real count: {events:?}"
        );
        assert_eq!(
            msg_delta["usage"]["output_tokens"], 13,
            "completion tokens from the finish chunk must be kept: {events:?}"
        );
    }

    // ================================================================
    // Thinking + cache_control pass-through tests
    // ================================================================

    #[test]
    fn anthropic_to_openai_passes_through_thinking() {
        let req = json!({
            "model": "claude-sonnet",
            "max_tokens": 16000,
            "thinking": {"type": "enabled", "budget_tokens": 5000},
            "messages": [{"role": "user", "content": "Hi"}],
        });
        let out = anthropic_to_openai_request(&req);
        assert!(
            out.get("thinking").is_some(),
            "thinking must be passed through"
        );
        assert_eq!(out["thinking"]["type"], "enabled");
        assert_eq!(out["thinking"]["budget_tokens"], 5000);
    }

    #[test]
    fn openai_to_anthropic_emits_thinking_block_with_signature() {
        let resp = json!({
            "id": "chatcmpl-1",
            "model": "claude",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Paris",
                    "thinking_blocks": [{
                        "type": "thinking",
                        "thinking": "Let me think.",
                        "signature": "sig_01"
                    }]
                },
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}
        });
        let out = openai_to_anthropic_response(&resp);
        let content = out["content"].as_array().expect("content");
        assert_eq!(content[0]["type"], "thinking");
        assert_eq!(content[0]["thinking"], "Let me think.");
        assert_eq!(content[0]["signature"], "sig_01");
        assert_eq!(content[1]["type"], "text");
        assert_eq!(content[1]["text"], "Paris");
    }

    #[test]
    fn anthropic_to_openai_passes_through_top_k() {
        let req = json!({
            "model": "claude-sonnet",
            "max_tokens": 100,
            "top_k": 40,
            "messages": [{"role": "user", "content": "Hi"}],
        });
        let out = anthropic_to_openai_request(&req);
        assert_eq!(out["top_k"], 40);
    }

    #[test]
    fn extract_system_text_preserves_cache_control_as_array() {
        let system = json!([
            {"type": "text", "text": "Long context.", "cache_control": {"type": "ephemeral"}}
        ]);
        let result = extract_system_text(&system);
        assert!(result.is_array(), "system with cache_control must be array");
        let block = &result[0];
        assert_eq!(block["type"], "text");
        assert_eq!(block["text"], "Long context.");
        assert_eq!(block["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn extract_system_text_joins_when_no_cache_control() {
        let system = json!([
            {"type": "text", "text": "Be helpful."},
            {"type": "text", "text": "Be concise."}
        ]);
        let result = extract_system_text(&system);
        assert!(
            result.is_string(),
            "system without cache_control should be string"
        );
        let s = result.as_str().unwrap();
        assert!(s.contains("Be helpful."));
        assert!(s.contains("Be concise."));
    }

    #[test]
    fn extract_system_text_string_passthrough() {
        let system = json!("You are helpful.");
        let result = extract_system_text(&system);
        assert_eq!(result, json!("You are helpful."));
    }

    #[test]
    fn convert_preserves_cache_control_on_user_content() {
        let msgs = vec![json!({
            "role": "user",
            "content": [
                {"type": "text", "text": "Long doc.", "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "Question?"}
            ]
        })];
        let mut out = Vec::new();
        convert_anthropic_messages_to_openai(&msgs, &mut out);
        assert_eq!(out.len(), 1);
        let content = &out[0]["content"];
        assert!(
            content.is_array(),
            "content with cache_control must be array"
        );
        assert_eq!(content[0]["cache_control"]["type"], "ephemeral");
        assert_eq!(content[1]["text"], "Question?");
        assert!(content[1].get("cache_control").is_none());
    }

    #[test]
    fn convert_preserves_cache_control_on_mixed_blocks() {
        let msgs = vec![json!({
            "role": "user",
            "content": [
                {"type": "text", "text": "Doc.", "cache_control": {"type": "ephemeral"}}
            ]
        })];
        let mut out = Vec::new();
        convert_anthropic_messages_to_openai(&msgs, &mut out);
        assert_eq!(out.len(), 1);
        let content = &out[0]["content"];
        assert!(
            content.is_array(),
            "single block with cache_control must stay as array"
        );
        assert_eq!(content[0]["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn convert_plain_text_without_cache_control_joins() {
        let msgs = vec![json!({
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello "},
                {"type": "text", "text": "world"}
            ]
        })];
        let mut out = Vec::new();
        convert_anthropic_messages_to_openai(&msgs, &mut out);
        assert_eq!(out.len(), 1);
        let content = &out[0]["content"];
        assert_eq!(
            content, "Hello world",
            "plain text blocks without cache_control should join"
        );
    }

    // ================================================================
    // C18 LOW: converter finishes after `finish_reason`
    // ================================================================

    /// Duplicate finish chunks do not produce duplicate terminal events;
    /// trailing usage is merged before the single terminal delta is emitted.
    #[test]
    fn duplicate_finish_reason_chunks_emit_a_single_message_delta() {
        let mut converter = AnthropicStreamConverter::new();

        let first = json!({
            "id": "x",
            "model": "m",
            "choices": [{"index": 0, "delta": {"content": "Hi"}, "finish_reason": null}]
        });
        let _ = converter.convert_chunk(&first);

        let finish_chunk = json!({
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
        });
        let first_finish = converter.convert_chunk(&finish_chunk);
        let first_deltas = first_finish
            .iter()
            .filter(|e| e["type"] == "message_delta")
            .count();
        assert_eq!(
            first_deltas, 0,
            "finish_reason must wait for the final usage chunk"
        );

        // A duplicate finish_reason chunk (or any trailing chunk) must be a
        // no-op — the converter is finished.
        let second_finish = converter.convert_chunk(&finish_chunk);
        assert!(
            second_finish.is_empty(),
            "post-finish chunks must be dropped, got: {second_finish:?}"
        );

        // `finish_events()` still emits the terminal `message_stop` exactly once.
        let terminal = converter.finish_events();
        assert_eq!(
            terminal
                .iter()
                .filter(|e| e["type"] == "message_delta")
                .count(),
            1
        );
        let stops = terminal
            .iter()
            .filter(|e| e["type"] == "message_stop")
            .count();
        assert_eq!(stops, 1, "exactly one message_stop expected: {terminal:?}");
        assert!(
            converter.finish_events().is_empty(),
            "finish_events must be idempotent"
        );
    }

    // ================================================================
    // C24: robustness fixes
    // ================================================================

    /// C24 MEDIUM: a provider that splits a tool call's `id` and `name`
    /// across two chunks must still produce a `content_block_start` with the
    /// full tool_use block (id + name) and forward argument deltas. Before
    /// the fix, requiring both in a single delta silently dropped the whole
    /// tool call.
    #[test]
    fn stream_tool_call_id_and_name_split_across_chunks() {
        let mut converter = AnthropicStreamConverter::new();

        // Chunk 1: only the tool call id, no function.name yet.
        let chunk1 = json!({
        "id": "chatcmpl-split",
        "model": "gpt-4o",
        "choices": [{
        "index": 0,
        "delta": {
        "role": "assistant",
        "tool_calls": [{"index": 0, "id": "call_1", "type": "function", "function": {}}]
        },
        "finish_reason": null
        }]
        });
        let events1 = converter.convert_chunk(&chunk1);
        assert!(
            events1
                .iter()
                .all(|e| e["type"] != "content_block_start"
                    || e["content_block"]["type"] != "tool_use"),
            "block must not open before name is known: {events1:?}"
        );

        // Chunk 2: function.name arrives (no id). Now the block can open.
        let chunk2 = json!({
        "choices": [{
        "index": 0,
        "delta": {
        "tool_calls": [{"index": 0, "function": {"name": "get_weather"}}]
        },
        "finish_reason": null
        }]
        });
        let events2 = converter.convert_chunk(&chunk2);
        let block_start = events2
            .iter()
            .find(|e| e["type"] == "content_block_start")
            .expect("content_block_start must fire once name is known: {events2:?}");
        assert_eq!(block_start["content_block"]["type"], "tool_use");
        assert_eq!(block_start["content_block"]["id"], "call_1");
        assert_eq!(block_start["content_block"]["name"], "get_weather");

        // Argument deltas arrive afterwards and must be forwarded.
        let chunk3 = json!({
        "choices": [{
        "index": 0,
        "delta": {
        "tool_calls": [{"index": 0, "function": {"arguments": "{\"city\":\"SF\"}"}}]
        },
        "finish_reason": "tool_calls"
        }]
        });
        let events3 = converter.convert_chunk(&chunk3);
        assert!(
            events3.iter().any(|e| e["type"] == "content_block_delta"
                && e["delta"]["type"] == "input_json_delta"
                && e["delta"]["partial_json"] == "{\"city\":\"SF\"}"),
            "argument delta must be forwarded once block is open: {events3:?}"
        );
    }

    /// C24 MEDIUM variant: argument fragments that arrive BEFORE the block
    /// opens (id+name split) must be buffered and flushed on open, not lost.
    #[test]
    fn stream_tool_call_arguments_buffered_until_block_opens() {
        let mut converter = AnthropicStreamConverter::new();

        // Chunk 1: id only.
        let _ = converter.convert_chunk(&json!({
        "id": "chatcmpl-buf",
        "model": "gpt-4o",
        "choices": [{
        "index": 0,
        "delta": {"tool_calls": [{"index": 0, "id": "call_1", "type": "function"}]},
        "finish_reason": null
        }]
        }));
        // Chunk 2: arguments arrive before name — must be buffered.
        let _ = converter.convert_chunk(&json!({
        "choices": [{
        "index": 0,
        "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\"a\":1}"}}]},
        "finish_reason": null
        }]
        }));
        // Chunk 3: name arrives → block opens and buffered arguments flush.
        let events = converter.convert_chunk(&json!({
        "choices": [{
        "index": 0,
        "delta": {"tool_calls": [{"index": 0, "function": {"name": "f"}}]},
        "finish_reason": null
        }]
        }));
        assert!(
            events
                .iter()
                .any(|e| e["type"] == "content_block_start" && e["content_block"]["name"] == "f"),
            "block must open when name arrives: {events:?}"
        );
        assert!(
            events.iter().any(|e| e["type"] == "content_block_delta"
                && e["delta"]["type"] == "input_json_delta"
                && e["delta"]["partial_json"] == "{\"a\":1}"),
            "buffered arguments must flush on open: {events:?}"
        );
    }

    /// C24 LOW: a non-conforming upstream that interleaves tool calls (opens
    /// tool 1, then sends args for the already-stopped tool 0) must NOT
    /// produce a `content_block_delta` for the closed block. The Anthropic
    /// protocol requires sequential blocks; a delta after stop is a protocol
    /// violation. The fragment is silently skipped.
    #[test]
    fn stream_interleaved_tool_args_after_block_closed_are_skipped() {
        let mut converter = AnthropicStreamConverter::new();

        // Chunk 1: open tool call 0.
        let _ = converter.convert_chunk(&json!({
        "id": "chatcmpl-interleave",
        "model": "gpt-4o",
        "choices": [{
        "index": 0,
        "delta": {"tool_calls": [{"index": 0, "id": "call_0", "type": "function", "function": {"name": "f0"}}]},
        "finish_reason": null
        }]
        }));

        // Chunk 2: open tool call 1 — this stops tool 0's block.
        let open1 = converter.convert_chunk(&json!({
        "choices": [{
        "index": 0,
        "delta": {"tool_calls": [{"index": 1, "id": "call_1", "type": "function", "function": {"name": "f1"}}]},
        "finish_reason": null
        }]
        }));
        assert!(
            open1.iter().any(|e| e["type"] == "content_block_stop"),
            "tool 0's block must be stopped when tool 1 opens: {open1:?}"
        );

        // Chunk 3: args arrive for tool 0 (the already-stopped block).
        let late_args = converter.convert_chunk(&json!({
        "choices": [{
        "index": 0,
        "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\"late\":true}"}}]},
        "finish_reason": null
        }]
        }));
        assert!(
            !late_args.iter().any(|e| e["type"] == "content_block_delta"
                && e["delta"]["type"] == "input_json_delta"
                && e["delta"]["partial_json"] == "{\"late\":true}"),
            "args for the already-stopped block must be skipped, not emitted as a delta: {late_args:?}"
        );

        // Sanity: args for the ACTIVE block (tool 1) are still forwarded.
        let active_args = converter.convert_chunk(&json!({
        "choices": [{
        "index": 0,
        "delta": {"tool_calls": [{"index": 1, "function": {"arguments": "{\"ok\":1}"}}]},
        "finish_reason": null
        }]
        }));
        assert!(
            active_args
                .iter()
                .any(|e| e["type"] == "content_block_delta"
                    && e["delta"]["type"] == "input_json_delta"
                    && e["delta"]["partial_json"] == "{\"ok\":1}"),
            "args for the active block must still be emitted: {active_args:?}"
        );
    }

    /// C24 LOW: an Anthropic `tool_use` block with `input: null` must
    /// convert to `arguments: "{}"`, not `"null"`. Non-object inputs (null,
    /// scalars) are normalised to an empty object.
    #[test]
    fn anthropic_tool_use_null_input_becomes_empty_arguments() {
        let req = json!({
        "model": "claude",
        "messages": [{
        "role": "assistant",
        "content": [
        {"type": "tool_use", "id": "t1", "name": "f", "input": null}
        ]
        }],
        "max_tokens": 10,
        });
        let openai = anthropic_to_openai_request(&req);
        assert_eq!(
            openai["messages"][0]["tool_calls"][0]["function"]["arguments"], "{}",
            "null input must serialise to empty object arguments"
        );
    }

    /// C24 LOW: an OpenAI response with no content and no tool calls (e.g.
    /// `content: null, finish_reason: "stop"`) must still yield at least one
    /// Anthropic content block, so strict Anthropic SDK clients don't reject
    /// the empty `content: []` array.
    #[test]
    fn openai_empty_response_falls_back_to_empty_text_block() {
        let resp = json!({
        "id": "chatcmpl-empty",
        "model": "gpt-4o",
        "choices": [{
        "index": 0,
        "message": {"role": "assistant", "content": null},
        "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1}
        });
        let anthropic = openai_to_anthropic_response(&resp);
        let content = anthropic["content"].as_array().expect("content array");
        assert!(
            !content.is_empty(),
            "content must not be empty: {anthropic}"
        );
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[0]["text"], "");
    }

    /// C24 LOW variant: when there is no message/choice at all, the
    /// converter still emits a single empty text block fallback.
    #[test]
    fn openai_no_choice_falls_back_to_empty_text_block() {
        let resp = json!({"id": "x", "model": "m", "choices": []});
        let anthropic = openai_to_anthropic_response(&resp);
        let content = anthropic["content"].as_array().expect("content array");
        assert_eq!(content.len(), 1);
        assert_eq!(content[0]["type"], "text");
    }

    /// C24 LOW: a malformed Anthropic message that mixes `tool_result`
    /// (user-side) and `tool_use` (assistant-side) blocks must not silently
    /// drop the tool-use blocks. The converter emits the tool results, then a
    /// separate assistant message carrying the tool calls.
    #[test]
    fn anthropic_message_with_tool_result_and_tool_use_preserves_tool_calls() {
        let msgs = vec![json!({
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "ok"},
                {"type": "tool_use", "id": "t2", "name": "get_weather", "input": {"city": "SF"}}
            ]
        })];
        let mut out = Vec::new();
        convert_anthropic_messages_to_openai(&msgs, &mut out);
        let tool_msg = out
            .iter()
            .find(|m| m["role"] == "tool")
            .expect("tool result message must survive: {out:?}");
        assert_eq!(tool_msg["tool_call_id"], "t1");
        let assistant_msg = out
            .iter()
            .find(|m| m["role"] == "assistant" && m.get("tool_calls").is_some())
            .expect("tool_use blocks must survive as an assistant message: {out:?}");
        assert_eq!(
            assistant_msg["tool_calls"][0]["function"]["name"],
            "get_weather"
        );
    }

    // ================================================================
    // C18 HIGH: Anthropic cache-token fields survive OpenAI→Anthropic
    // ================================================================

    /// An OpenAI response whose `usage` carries `prompt_tokens_details` with
    /// `cached_tokens` and `cache_write_tokens` must surface those values as
    /// Anthropic-native `cache_read_input_tokens` /
    /// `cache_creation_input_tokens` so Anthropic SDK clients see the same
    /// cache pricing data the internal cost pipeline already uses.
    #[test]
    fn openai_response_with_cache_details_maps_to_anthropic_usage() {
        let resp = json!({
        "id": "chatcmpl-cache",
        "object": "chat.completion",
        "model": "claude-3-5-sonnet",
        "choices": [{
        "index": 0,
        "message": {"role": "assistant", "content": "hi"},
        "finish_reason": "stop"
        }],
        "usage": {
        "prompt_tokens": 100,
        "completion_tokens": 5,
        "total_tokens": 105,
        "prompt_tokens_details": {
        "cached_tokens": 80,
        "cache_write_tokens": 20
        }
        }
        });
        let anthropic = openai_to_anthropic_response(&resp);
        assert_eq!(anthropic["usage"]["input_tokens"], 100);
        assert_eq!(anthropic["usage"]["output_tokens"], 5);
        assert_eq!(
            anthropic["usage"]["cache_read_input_tokens"], 80,
            "cached_tokens must map to cache_read_input_tokens: {anthropic}"
        );
        assert_eq!(
            anthropic["usage"]["cache_creation_input_tokens"], 20,
            "cache_write_tokens must map to cache_creation_input_tokens: {anthropic}"
        );
    }

    /// The `cache_creation_tokens` alternative spelling (used by some
    /// providers) maps to `cache_creation_input_tokens` just like
    /// `cache_write_tokens`.
    #[test]
    fn openai_response_cache_creation_tokens_spelling_maps() {
        let resp = json!({
        "id": "x",
        "object": "chat.completion",
        "model": "m",
        "choices": [{
        "index": 0,
        "message": {"role": "assistant", "content": "hi"},
        "finish_reason": "stop"
        }],
        "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "prompt_tokens_details": {"cache_creation_tokens": 7}
        }
        });
        let anthropic = openai_to_anthropic_response(&resp);
        assert_eq!(anthropic["usage"]["cache_creation_input_tokens"], 7);
        assert!(
            anthropic["usage"].get("cache_read_input_tokens").is_none(),
            "absent cached_tokens must not produce a zero cache_read field"
        );
    }

    /// An OpenAI response without `prompt_tokens_details` must produce a
    /// clean Anthropic usage object (no cache fields), preserving backwards
    /// compatibility with non-Anthropic-backed upstreams.
    #[test]
    fn openai_response_without_cache_details_has_no_cache_fields() {
        let resp = json!({
        "id": "x",
        "object": "chat.completion",
        "model": "m",
        "choices": [{
        "index": 0,
        "message": {"role": "assistant", "content": "hi"},
        "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        });
        let anthropic = openai_to_anthropic_response(&resp);
        assert!(anthropic["usage"].get("cache_read_input_tokens").is_none());
        assert!(
            anthropic["usage"]
                .get("cache_creation_input_tokens")
                .is_none()
        );
    }

    /// Streaming variant: cache-token fields from `prompt_tokens_details`
    /// must survive into the terminal `message_delta.usage`.
    #[test]
    fn stream_cache_details_survive_into_message_delta_usage() {
        let mut converter = AnthropicStreamConverter::new();

        let role_chunk = json!({
        "id": "chatcmpl-stream-cache",
        "model": "claude-3-5-sonnet",
        "choices": [{
        "index": 0,
        "delta": {"role": "assistant", "content": "hi"},
        "finish_reason": null
        }],
        "usage": {
        "prompt_tokens": 100,
        "completion_tokens": 0,
        "prompt_tokens_details": {"cached_tokens": 80, "cache_write_tokens": 20}
        }
        });
        let _ = converter.convert_chunk(&role_chunk);

        let finish_chunk = json!({
        "choices": [{
        "index": 0,
        "delta": {},
        "finish_reason": "stop"
        }],
        "usage": {
        "prompt_tokens": 0,
        "completion_tokens": 5,
        "prompt_tokens_details": {"cached_tokens": 80, "cache_write_tokens": 20}
        }
        });
        converter.convert_chunk(&finish_chunk);
        let events = converter.finish_events();
        let msg_delta = events
            .iter()
            .find(|e| e["type"] == "message_delta")
            .expect("missing message_delta on finish");
        assert_eq!(
            msg_delta["usage"]["input_tokens"], 100,
            "prompt_tokens must survive max-merge: {events:?}"
        );
        assert_eq!(
            msg_delta["usage"]["output_tokens"], 5,
            "completion_tokens must survive: {events:?}"
        );
        assert_eq!(
            msg_delta["usage"]["cache_read_input_tokens"], 80,
            "cache_read must survive into message_delta: {events:?}"
        );
        assert_eq!(
            msg_delta["usage"]["cache_creation_input_tokens"], 20,
            "cache_creation must survive into message_delta: {events:?}"
        );
    }

    // ================================================================
    // C18 MEDIUM: mid-stream error closes the open content block
    // ================================================================

    /// When a content block is open and an error chunk arrives, the converter
    /// must emit `content_block_stop` for the open block before the terminal
    /// `error` event so the Anthropic event stream stays balanced.
    #[test]
    fn stream_error_after_content_block_emits_content_block_stop() {
        let mut converter = AnthropicStreamConverter::new();

        let text_chunk = json!({
        "id": "chatcmpl-err-mid",
        "model": "gpt-4o",
        "choices": [{
        "index": 0,
        "delta": {"content": "partial"},
        "finish_reason": null
        }]
        });
        let start_events = converter.convert_chunk(&text_chunk);
        assert!(
            start_events
                .iter()
                .any(|e| e["type"] == "content_block_start"),
            "a text block must be open before the error: {start_events:?}"
        );

        let error_events = converter.convert_chunk(&json!({
        "error": {"type": "overloaded_error", "message": "capacity"}
        }));
        let types: Vec<&str> = error_events
            .iter()
            .map(|e| e["type"].as_str().unwrap_or(""))
            .collect();
        assert_eq!(
            types,
            vec!["content_block_stop", "error"],
            "open block must be closed before the terminal error: {error_events:?}"
        );
        assert!(
            converter.finish_events().is_empty(),
            "post-error finish_events must be empty"
        );
    }

    // ================================================================
    // C18 LOW: stream:null forwarded consistently
    // ================================================================

    /// A null `stream` field must not produce `"stream": null` in the OpenAI
    /// body, matching the `temperature`/`top_p` guard. The chat pipeline
    /// reads `stream` with `as_bool().unwrap_or(false)` so `null` is treated
    /// as `false`, but the converted body should stay clean.
    #[test]
    fn anthropic_request_stream_null_treated_as_false() {
        let req = json!({
        "model": "claude",
        "messages": [],
        "max_tokens": 10,
        "stream": null
        });
        let openai = anthropic_to_openai_request(&req);
        assert!(
            openai.get("stream").is_none(),
            "null stream must not be forwarded: {openai}"
        );

        let req_true = json!({
        "model": "claude",
        "messages": [],
        "max_tokens": 10,
        "stream": true
        });
        assert_eq!(anthropic_to_openai_request(&req_true)["stream"], true);
    }

    // ================================================================
    // C18 LOW: first chunk's usage stamps into message_start
    // ================================================================

    /// When the first OpenAI chunk carries `usage.prompt_tokens` (Anthropic-
    /// backed streams via the provider transform), the converter must stamp
    /// the real value into `message_start.usage` instead of the 0/0
    /// placeholder.
    #[test]
    fn stream_first_chunk_with_usage_stamps_message_start() {
        let mut converter = AnthropicStreamConverter::new();
        let chunk = json!({
        "id": "chatcmpl-first-usage",
        "model": "claude-3-5-sonnet",
        "choices": [{
        "index": 0,
        "delta": {"role": "assistant", "content": ""},
        "finish_reason": null
        }],
        "usage": {
        "prompt_tokens": 42,
        "completion_tokens": 0,
        "prompt_tokens_details": {"cached_tokens": 10}
        }
        });
        let events = converter.convert_chunk(&chunk);
        let message_start = events
            .iter()
            .find(|e| e["type"] == "message_start")
            .expect("message_start must be emitted");
        assert_eq!(
            message_start["message"]["usage"]["input_tokens"], 42,
            "first-chunk prompt_tokens must stamp message_start: {events:?}"
        );
        assert_eq!(
            message_start["message"]["usage"]["cache_read_input_tokens"], 10,
            "first-chunk cache tokens must stamp message_start: {events:?}"
        );
    }

    // ================================================================
    // C24 LOW: malformed SSE data: payloads are dropped (now logged)
    // ================================================================

    /// A malformed `data:` payload must yield no events (pins the current
    /// drop-on-error behaviour). The accompanying `tracing::debug!` is not
    /// asserted here — observability hooks can capture the line.
    #[test]
    fn anthropic_stream_malformed_data_payload_yields_no_events() {
        let mut converter = AnthropicStreamConverter::new();
        let events = process_openai_sse_event(&mut converter, "data: not-json");
        assert!(
            events.is_empty(),
            "malformed payload must not produce events: {events:?}"
        );
    }
}
