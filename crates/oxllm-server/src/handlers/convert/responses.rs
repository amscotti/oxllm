//! OpenAI Responses API ↔ chat completions format conversion.
//!
//! Pure data-transformation helpers extracted from `handlers::responses`. This
//! module knows nothing about axum, tokio, or SSE transport — it maps JSON
//! shapes only, so it can be unit-tested in isolation and reused.

use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{Map, Value, json};

use crate::sse::extract_sse_data;

/// Defence-in-depth cap on per-converter text/argument accumulators. Matches
/// `orchestration::MAX_STREAM_BUFFER_BYTES` so the converter layer does not
/// hold more memory than the wire-level cap.
const MAX_STREAM_BUFFER_BYTES: usize = 4 * 1024 * 1024;
/// Defence-in-depth cap on the number of distinct tool-call indices tracked
/// per stream. A healthy stream carries a handful; thousands indicates a
/// pathological / attacker-controlled upstream.
const MAX_STREAM_TOOL_CALLS: usize = 1024;

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Request conversion: Responses → Chat
// ---------------------------------------------------------------------------

pub(crate) fn responses_to_chat_request(req: &Value) -> Value {
    let mut out = Map::new();

    if let Some(model) = req.get("model") {
        out.insert("model".to_string(), model.clone());
    }

    let mut messages: Vec<Value> = Vec::new();

    if let Some(instructions) = req.get("instructions") {
        if let Some(s) = instructions.as_str() {
            if !s.is_empty() {
                messages.push(json!({"role": "system", "content": s}));
            }
        }
    }

    match req.get("input") {
        Some(Value::String(s)) => {
            messages.push(json!({"role": "user", "content": s}));
        }
        Some(Value::Array(arr)) => {
            for item in arr {
                match item.get("type").and_then(|v| v.as_str()) {
                    Some("function_call") | Some("custom_tool_call") => {
                        let call_id = item
                            .get("call_id")
                            .or_else(|| item.get("id"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        // Custom tool calls carry their payload in an `input`
                        // field (raw string) rather than `arguments` (JSON
                        // string). Normalise to arguments so the chat pipeline
                        // sees a non-empty payload — mirrors the reference
                        // `_transform_responses_api_function_call_to_chat_completion_message`.
                        let mut arguments = item
                            .get("arguments")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        if arguments.is_empty()
                            && item.get("type").and_then(|v| v.as_str()) == Some("custom_tool_call")
                        {
                            if let Some(input) = item.get("input").and_then(|v| v.as_str()) {
                                if !input.is_empty() {
                                    arguments = serde_json::to_string(&json!({"content": input}))
                                        .unwrap_or_else(|_| String::new());
                                }
                            }
                        }
                        let tool_call = json!({
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": item.get("name").and_then(|v| v.as_str()).unwrap_or(""),
                                "arguments": arguments
                            }
                        });
                        match messages.last_mut() {
                            Some(Value::Object(last))
                                if last.get("role") == Some(&json!("assistant")) =>
                            {
                                if let Some(arr) = last
                                    .entry("tool_calls".to_string())
                                    .or_insert_with(|| Value::Array(Vec::new()))
                                    .as_array_mut()
                                {
                                    arr.push(tool_call);
                                } else {
                                    // A passthrough input message may carry a
                                    // non-array `tool_calls` field; skip rather
                                    // than panic.
                                    tracing::warn!(
                                        "existing tool_calls field is not an array; dropping tool call"
                                    );
                                }
                            }
                            _ => messages.push(json!({
                                "role": "assistant",
                                "content": Value::Null,
                                "tool_calls": [tool_call]
                            })),
                        }
                    }
                    Some("function_call_output")
                    | Some("custom_tool_call_output")
                    | Some("tool_result") => {
                        // The reference `_is_input_item_tool_call_output` treats
                        // `function_call_output`, `custom_tool_call_output`,
                        // and `tool_result` (Anthropic/MCP spelling) as tool
                        // outputs. Without this arm they fall into the
                        // catchall and get wrapped as malformed user messages.
                        // `tool_result` carries its id in `tool_use_id` and
                        // its payload in `content`; the other two use
                        // `call_id` and `output`.
                        let call_id = item
                            .get("call_id")
                            .or_else(|| item.get("tool_use_id"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        if call_id.is_empty() {
                            continue;
                        }
                        let output = item
                            .get("output")
                            .or_else(|| item.get("content"))
                            .unwrap_or(&Value::Null);
                        messages.push(json!({
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": normalize_function_call_output(output)
                        }));
                    }
                    Some("web_search_call") | Some("computer_call_output") => {
                        // Hosted web-search/computer-use items have no
                        // supported chat-completions bridge here.
                        // `computer_call_output` does carry a `call_id`, but
                        // its screenshot/safety payload requires the hosted
                        // computer-use protocol. Drop these items rather than
                        // wrapping them as user messages via the catchall.
                        continue;
                    }
                    Some("reasoning") => {
                        // U29: reasoning input items carry prior-turn reasoning
                        // context (encrypted or summary text) that has no
                        // chat-completions equivalent. Dropping them silently
                        // matches how the reference handles the bridge path — without
                        // this they fall into the catchall and get wrapped as
                        // malformed user messages.
                        continue;
                    }
                    _ => {
                        if let Some(role) = item.get("role").and_then(|v| v.as_str()) {
                            let mut msg = Map::new();
                            msg.insert("role".to_string(), json!(role));
                            if let Some(content) = item.get("content") {
                                msg.insert(
                                    "content".to_string(),
                                    normalize_responses_message_content(content),
                                );
                            }
                            messages.push(Value::Object(msg));
                        } else {
                            messages.push(json!({"role": "user", "content": item}));
                        }
                    }
                }
            }
        }
        _ => {}
    }
    out.insert("messages".to_string(), Value::Array(messages));

    if let Some(temp) = req.get("temperature") {
        if !temp.is_null() {
            out.insert("temperature".to_string(), temp.clone());
        }
    }
    if let Some(top_p) = req.get("top_p") {
        if !top_p.is_null() {
            out.insert("top_p".to_string(), top_p.clone());
        }
    }
    if let Some(max_tokens) = req.get("max_output_tokens") {
        if !max_tokens.is_null() {
            out.insert("max_tokens".to_string(), max_tokens.clone());
        }
    }
    if let Some(stream) = req.get("stream") {
        if !stream.is_null() {
            out.insert("stream".to_string(), stream.clone());
        }
    }
    // Pass the cache opt-in (`cache: {"insert": true}`) through so the chat
    // pipeline's CacheHook can see it; dropping it would silently disable
    // caching for Responses-shaped requests.
    if let Some(cache) = req.get("cache") {
        if !cache.is_null() {
            out.insert("cache".to_string(), cache.clone());
        }
    }
    for key in ["metadata", "user", "service_tier"] {
        if let Some(value) = req.get(key).filter(|value| !value.is_null()) {
            out.insert(key.to_string(), value.clone());
        }
    }
    if let Some(tools) = req.get("tools").and_then(|v| v.as_array()) {
        let mapped: Vec<Value> = tools
            .iter()
            .filter_map(map_responses_tool_to_chat)
            .collect();
        if !mapped.is_empty() {
            out.insert("tools".to_string(), Value::Array(mapped));
        }
    }
    if let Some(tool_choice) = req
        .get("tool_choice")
        .and_then(map_responses_tool_choice_to_chat)
    {
        out.insert("tool_choice".to_string(), tool_choice);
    }
    if let Some(parallel) = req.get("parallel_tool_calls") {
        if !parallel.is_null() {
            out.insert("parallel_tool_calls".to_string(), parallel.clone());
        }
    }
    if let Some(reasoning) = req.get("reasoning") {
        if !reasoning.is_null() {
            // Mirror the reference precedence (transformation.py:216-231):
            // - dict with `summary` → forward the whole dict so the
            //   Responses bridge can pass `summary` to providers that use it
            // - dict with only `effort` → forward just the effort string
            // - dict without `effort`/`summary` → forward as-is
            // - bare string → forward as-is
            let effort = if reasoning.is_object() {
                if reasoning.get("summary").is_some() {
                    reasoning.clone()
                } else if let Some(e) = reasoning
                    .get("effort")
                    .filter(|value| !value.is_null())
                    .cloned()
                {
                    e
                } else {
                    reasoning.clone()
                }
            } else {
                reasoning.clone()
            };
            out.insert("reasoning_effort".to_string(), effort);
        }
    }
    if req.get("stream").and_then(|v| v.as_bool()).unwrap_or(false) {
        // U27: `stream_options` is a chat
        // completions parameter, NOT a Responses API parameter. The
        // converter must NOT forward any client-supplied `stream_options`
        // (and it doesn't — there is no `req.get("stream_options")` clone
        // anywhere in this function). When streaming, we unconditionally
        // inject `stream_options: {include_usage: true}` here so the
        // downstream chat pipeline returns the trailing usage-only chunk
        // the Responses stream converter needs to populate
        // `response.completed.usage`. The host-side
        // `inject_stream_usage` (in `orchestration::upstream`) is a
        // no-op for Anthropic/Cohere/Bedrock/Vertex AI which carry usage
        // natively; for OpenAI-shaped providers it overrides
        // `include_usage` to true, so the value we set here is harmless
        // if the host already injects one.
        out.insert("stream_options".to_string(), json!({"include_usage": true}));
    }

    Value::Object(out)
}

fn normalize_responses_message_content(content: &Value) -> Value {
    let Some(parts) = content.as_array() else {
        return content.clone();
    };
    Value::Array(
        parts
            .iter()
            .map(|part| match part.get("type").and_then(Value::as_str) {
                Some("input_text" | "output_text") => {
                    let mut mapped = part.clone();
                    mapped["type"] = json!("text");
                    mapped
                }
                Some("input_image") => {
                    let mut image = part.get("image_url").cloned().unwrap_or(Value::Null);
                    if image.is_string() {
                        image = json!({"url":image});
                    }
                    if let Some(detail) = part.get("detail").filter(|value| !value.is_null()) {
                        if image.is_object() {
                            image["detail"] = detail.clone();
                        }
                    }
                    json!({"type":"image_url","image_url":image})
                }
                _ => part.clone(),
            })
            .collect(),
    )
}

fn map_responses_tool_to_chat(tool: &Value) -> Option<Value> {
    match tool.get("type").and_then(|v| v.as_str()) {
        Some("function") => {
            let mut parameters = tool.get("parameters").cloned().unwrap_or_else(|| json!({}));
            if parameters.get("type").is_none() {
                parameters["type"] = json!("object");
            }
            let mut function = Map::new();
            function.insert(
                "name".to_string(),
                tool.get("name").cloned().unwrap_or_else(|| json!("")),
            );
            if let Some(description) = tool.get("description") {
                function.insert("description".to_string(), description.clone());
            }
            function.insert("parameters".to_string(), parameters);
            if let Some(strict) = tool.get("strict") {
                function.insert("strict".to_string(), strict.clone());
            }
            Some(json!({"type": "function", "function": function}))
        }
        Some(other) => {
            tracing::warn!(tool_type = %other, "dropping Responses tool with no chat equivalent");
            None
        }
        None => None,
    }
}

fn map_responses_tool_choice_to_chat(tool_choice: &Value) -> Option<Value> {
    if tool_choice.is_null() {
        return None;
    }
    if tool_choice.is_string() {
        return Some(tool_choice.clone());
    }
    let object = tool_choice.as_object()?;
    if object.get("function").and_then(|f| f.get("name")).is_some() {
        return Some(tool_choice.clone());
    }
    match object.get("type").and_then(|v| v.as_str()) {
        Some("auto") => Some(json!("auto")),
        Some("none") => Some(json!("none")),
        Some("required") | Some("tool") | Some("any") => Some(json!("required")),
        Some("function") => match object.get("name").and_then(|v| v.as_str()) {
            Some(name) => Some(json!({"type": "function", "function": {"name": name}})),
            None => Some(json!("required")),
        },
        _ => Some(tool_choice.clone()),
    }
}

fn normalize_function_call_output(output: &Value) -> Value {
    match output {
        Value::Null => json!(""),
        Value::String(_) => output.clone(),
        Value::Array(parts) => {
            // Mirror the reference tool-output content normalization:
            // text parts (`input_text`/`output_text`/`text`) accumulate as a
            // string; image parts (`input_image`/`image_url`) promote the
            // whole output to a list of `{"type":"text"|image_url",...}`
            // blocks so multimodal tool results survive instead of being
            // stringified away.
            let mut blocks: Vec<Value> = Vec::new();
            let mut text_acc: String = String::new();
            for part in parts {
                let kind = part.get("type").and_then(|v| v.as_str());
                match kind {
                    Some("input_text") | Some("output_text") | Some("text") => {
                        if let Some(t) = part.get("text").and_then(|v| v.as_str()) {
                            text_acc.push_str(t);
                            blocks.push(json!({"type": "text", "text": t}));
                        }
                    }
                    Some("input_image") | Some("image_url") => {
                        // `image_url` may be either a bare string (the
                        // canonical Responses `input_image` shape,
                        // `{"type":"input_image","image_url":"data:..."}`)
                        // or an object (`{"url":"..."}`). Accept both so a
                        // bare-string value is not silently dropped. Mirrors
                        // the reference tool-output content normalization.
                        let url = part
                            .get("image_url")
                            .and_then(|v| {
                                v.as_str().or_else(|| v.get("url").and_then(|u| u.as_str()))
                            })
                            .or_else(|| part.get("url").and_then(|v| v.as_str()));
                        if let Some(url) = url {
                            if !url.is_empty() {
                                blocks
                                    .push(json!({"type": "image_url", "image_url": {"url": url}}));
                            }
                        }
                    }
                    _ => {}
                }
            }
            let has_image = blocks.iter().any(|b| b["type"] == "image_url");
            if has_image {
                Value::Array(blocks)
            } else if !text_acc.is_empty() {
                json!(text_acc)
            } else {
                json!(serde_json::to_string(parts).unwrap_or_default())
            }
        }
        other => json!(serde_json::to_string(other).unwrap_or_default()),
    }
}

// ---------------------------------------------------------------------------
// Response conversion: Chat → Responses
// ---------------------------------------------------------------------------

fn build_output_items(message: Option<&Value>) -> Vec<Value> {
    let mut output = Vec::new();
    let Some(message) = message else {
        return output;
    };

    if let Some(reasoning) = message.get("reasoning_content").and_then(|v| v.as_str()) {
        if !reasoning.is_empty() {
            output.push(json!({
                "type": "reasoning",
                "id": format!("rs_{}", uuid::Uuid::new_v4()),
                "status": "completed",
                "summary": [{"type": "summary_text", "text": reasoning}]
            }));
        }
    }

    if let Some(text) = message.get("content").and_then(|v| v.as_str()) {
        if !text.is_empty() {
            output.push(json!({
                "type": "message",
                "id": format!("msg_{}", uuid::Uuid::new_v4()),
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}]
            }));
        }
    }

    if let Some(tool_calls) = message.get("tool_calls").and_then(|v| v.as_array()) {
        for tool_call in tool_calls {
            let id = tool_call.get("id").and_then(|v| v.as_str()).unwrap_or("");
            let function = tool_call
                .get("function")
                .cloned()
                .unwrap_or_else(|| json!({}));
            output.push(json!({
                "type": "function_call",
                "id": id,
                "call_id": id,
                "name": function.get("name").and_then(|v| v.as_str()).unwrap_or(""),
                "arguments": function.get("arguments").and_then(|v| v.as_str()).unwrap_or(""),
                "status": "completed"
            }));
        }
    }

    output
}

/// Merge an incoming streaming usage snapshot into the accumulated one using
/// max-merge semantics, matching `orchestration::usage::merge_streaming_usage`.
///
/// Anthropic-backed streams split usage across chunks: the first chunk carries
/// the real `prompt_tokens` and the finish chunk carries `prompt_tokens: 0`
/// plus the real `completion_tokens`. A plain overwrite would clobber the
/// input token count back to zero, so numeric fields keep their running max.
/// Detail objects (e.g. `prompt_tokens_details`) only appear on chunks that
/// carry them, so the incoming copy wins when present.
fn merge_stream_usage(acc: &mut Option<Value>, incoming: &Value) {
    let Some(existing) = acc else {
        *acc = Some(incoming.clone());
        return;
    };
    for key in ["prompt_tokens", "completion_tokens", "total_tokens"] {
        if let Some(v) = incoming.get(key).and_then(|v| v.as_u64()) {
            let current = existing.get(key).and_then(|v| v.as_u64()).unwrap_or(0);
            existing[key] = json!(current.max(v));
        }
    }
    for key in ["prompt_tokens_details", "completion_tokens_details"] {
        if let Some(v) = incoming.get(key).filter(|v| !v.is_null()) {
            existing[key] = v.clone();
        }
    }
}

fn build_responses_usage(usage: Option<&Value>) -> Value {
    let input = usage
        .and_then(|u| u.get("prompt_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let output = usage
        .and_then(|u| u.get("completion_tokens"))
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let total = input.saturating_add(output);
    let mut out = json!({
        "input_tokens": input,
        "output_tokens": output,
        "total_tokens": total
    });
    if let Some(cached) = usage
        .and_then(|u| u.get("prompt_tokens_details"))
        .and_then(|d| d.get("cached_tokens"))
        .and_then(|v| v.as_u64())
    {
        out["input_tokens_details"] = json!({"cached_tokens": cached});
    }
    // U47: surface cache-creation (cache-write) tokens. OpenAI-compatible
    // providers report them under `prompt_tokens_details.cache_write_tokens`
    // (or the `cache_creation_tokens` alias); Anthropic reports them as a
    // top-level `cache_creation_input_tokens`. Mirror whichever is present
    // into `input_tokens_details.cache_write_tokens` so Responses-API clients
    // (and U48 spend/analytics logs reading off this usage object) see the
    // cache-write count alongside `cached_tokens`.
    let cache_creation = cache_creation_tokens_from_chat_usage(usage);
    if let Some(created) = cache_creation {
        // Preserve any pre-existing `input_tokens_details` (e.g. cached_tokens
        // set above) and add the cache-write count to it.
        let details = out
            .get("input_tokens_details")
            .cloned()
            .filter(|v| v.is_object())
            .unwrap_or_else(|| json!({}));
        let mut details = match details {
            Value::Object(map) => map,
            _ => serde_json::Map::new(),
        };
        details.insert("cache_write_tokens".to_string(), json!(created));
        out["input_tokens_details"] = Value::Object(details);
    }
    if let Some(reasoning) = usage
        .and_then(|u| u.get("completion_tokens_details"))
        .and_then(|d| d.get("reasoning_tokens"))
        .and_then(|v| v.as_u64())
    {
        out["output_tokens_details"] = json!({"reasoning_tokens": reasoning});
    }
    out
}

/// Resolve cache-creation (cache-write) tokens from a chat-completions
/// `usage` object. Order matches `orchestration::usage::extract_usage`:
/// Anthropic top-level `cache_creation_input_tokens` → OpenAI
/// `prompt_tokens_details.cache_write_tokens` → the `cache_creation_tokens`
/// alias. Returns `None` when none are present.
fn cache_creation_tokens_from_chat_usage(usage: Option<&Value>) -> Option<u64> {
    let usage = usage?;
    usage
        .get("cache_creation_input_tokens")
        .and_then(|v| v.as_u64())
        .or_else(|| {
            usage
                .get("prompt_tokens_details")
                .and_then(|d| d.get("cache_write_tokens"))
                .and_then(|v| v.as_u64())
        })
        .or_else(|| {
            usage
                .get("prompt_tokens_details")
                .and_then(|d| d.get("cache_creation_tokens"))
                .and_then(|v| v.as_u64())
        })
}

fn responses_status_for_finish_reason(reason: Option<&str>) -> &'static str {
    match reason {
        Some("length") | Some("content_filter") | Some("refusal") => "incomplete",
        _ => "completed",
    }
}

pub(crate) fn chat_to_responses_response(resp: &Value) -> Value {
    let raw_id = resp.get("id").and_then(|v| v.as_str()).unwrap_or("");
    let response_id = if let Some(rest) = raw_id.strip_prefix("chatcmpl-") {
        format!("resp_{rest}")
    } else if raw_id.is_empty() {
        format!("resp_{}", uuid::Uuid::new_v4())
    } else {
        raw_id.to_string()
    };

    let model = resp
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let choice = resp
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|a| a.first());
    let finish_reason = choice
        .and_then(|c| c.get("finish_reason"))
        .and_then(|v| v.as_str());
    let message = choice.and_then(|c| c.get("message"));
    let output = build_output_items(message);
    let status = responses_status_for_finish_reason(finish_reason);
    let mut response = json!({
    "id": response_id,
    "object": "response",
    "created_at": unix_now(),
    "model": model,
    "output": output,
    "usage": build_responses_usage(resp.get("usage")),
    "status": status,
    });
    if finish_reason == Some("length") {
        response["incomplete_details"] = json!({"reason": "max_output_tokens"});
    }
    if let Some(tier) = oxllm_core::response_service_tier(resp) {
        response["service_tier"] = json!(tier);
    }
    response
}

// ---------------------------------------------------------------------------
// Stream converter: OpenAI SSE chunks → Responses SSE events
// ---------------------------------------------------------------------------

pub(crate) struct ResponsesStreamConverter {
    started: bool,
    finished: bool,
    /// `finish_reason` has arrived but `response.completed` is being held back
    /// until either the trailing usage-only chunk or the terminal `[DONE]` /
    /// EOF marker, so that the completion event reports real usage.
    pending_completion: bool,
    response_id: String,
    model: String,
    text: String,
    reasoning: String,
    reasoning_item_id: Option<String>,
    tool_calls: std::collections::HashMap<u64, StreamToolCall>,
    finish_reason: Option<String>,
    usage: Option<Value>,
    service_tier: Option<String>,
    message_item_id: Option<String>,
    message_output_index: Option<u64>,
    reasoning_output_index: Option<u64>,
    next_output_index: u64,
    message_item_added: bool,
    content_part_added: bool,
    sequence_number: u64,
    /// Whether `finalize_events` has already emitted the item-lifecycle
    /// `done` events. Makes `finalize_events` idempotent so the chunk-level
    /// error arm, `finish_events`, and `close_open_items` cannot double-emit
    /// done events when their paths overlap (e.g. `finish_reason` finalizes,
    /// then a stream-read error triggers `close_open_items`).
    items_finalized: bool,
}

#[derive(Debug, Clone, Default)]
struct StreamToolCall {
    output_index: u64,
    call_id: String,
    name: String,
    arguments: String,
    /// Whether `response.output_item.added` has already been emitted for
    /// this tool call. Tracked separately from `call_id` because the added
    /// event fires once we know enough to identify the item (real id OR a
    /// synthesised id backed by a name), not necessarily when the upstream
    /// id arrives.
    added_emitted: bool,
}

impl ResponsesStreamConverter {
    pub(crate) fn new() -> Self {
        Self {
            started: false,
            finished: false,
            pending_completion: false,
            response_id: String::new(),
            model: String::new(),
            text: String::new(),
            reasoning: String::new(),
            reasoning_item_id: None,
            tool_calls: std::collections::HashMap::new(),
            finish_reason: None,
            usage: None,
            service_tier: None,
            message_item_id: None,
            message_output_index: None,
            reasoning_output_index: None,
            next_output_index: 0,
            message_item_added: false,
            content_part_added: false,
            sequence_number: 0,
            items_finalized: false,
        }
    }

    fn stamp_seq(&mut self, events: &mut [(String, Value)]) {
        for (_, payload) in events.iter_mut() {
            self.sequence_number += 1;
            if let Some(obj) = payload.as_object_mut() {
                obj.insert("sequence_number".to_string(), json!(self.sequence_number));
                if let (Some(response), Some(tier)) = (obj.get_mut("response"), &self.service_tier)
                {
                    response["service_tier"] = json!(tier);
                }
            }
        }
    }

    fn ensure_message_item_started(&mut self, events: &mut Vec<(String, Value)>) {
        if self.message_item_added {
            return;
        }
        self.message_item_added = true;
        self.message_output_index = Some(self.next_output_index);
        self.next_output_index += 1;
        let msg_id = self
            .message_item_id
            .get_or_insert_with(|| format!("msg_{}", uuid::Uuid::new_v4()))
            .clone();
        events.push((
            "response.output_item.added".to_string(),
            json!({
            "type": "response.output_item.added",
            "response_id": self.response_id,
            "output_index": self.message_output_index,
            "item": {
            "id": msg_id,
            "type": "message",
            "status": "in_progress",
            "role": "assistant",
            "content": []
            }
            }),
        ));
        self.content_part_added = true;
        events.push((
            "response.content_part.added".to_string(),
            json!({
            "type": "response.content_part.added",
            "response_id": self.response_id,
            "item_id": msg_id,
            "output_index": self.message_output_index,
            "content_index": 0,
            "part": {"type": "output_text", "text": "", "annotations": []}
            }),
        ));
    }

    pub(crate) fn convert_chunk(&mut self, chunk: &Value) -> Vec<(String, Value)> {
        if self.finished {
            return Vec::new();
        }
        if let Some(tier) = oxllm_core::response_service_tier(chunk) {
            if self.service_tier.as_deref() != Some(tier) {
                self.service_tier = Some(tier.to_string());
            }
        }
        if let Some(error) = chunk.get("error").filter(|e| !e.is_null()) {
            self.finished = true;
            let mut events = Vec::new();
            // Close any started message item / tool calls before the terminal
            // error so the item lifecycle stays balanced (every
            // `response.output_item.added` pairs with a
            // `response.output_item.done`), matching the Anthropic converter's
            // error arm. Skip when `finish_reason` already finalized the items
            // (`pending_completion` is set) to avoid duplicate done events.
            if !self.pending_completion {
                events.extend(self.finalize_events());
            }
            events.push((
                "error".to_string(),
                json!({"type": "error", "error": error}),
            ));
            self.stamp_seq(&mut events);
            return events;
        }

        let mut events = Vec::new();

        if !self.started {
            self.started = true;
            let raw_id = chunk.get("id").and_then(|v| v.as_str()).unwrap_or("");
            self.response_id = if let Some(rest) = raw_id.strip_prefix("chatcmpl-") {
                format!("resp_{rest}")
            } else if raw_id.is_empty() {
                format!("resp_{}", uuid::Uuid::new_v4())
            } else {
                raw_id.to_string()
            };
            if let Some(m) = chunk.get("model").and_then(|v| v.as_str()) {
                self.model = m.to_string();
            }
            events.push((
                "response.created".to_string(),
                json!({
                "type": "response.created",
                "response": {
                "id": self.response_id,
                "object": "response",
                "created_at": unix_now(),
                "model": self.model,
                "status": "in_progress",
                "output": []
                }
                }),
            ));
            events.push((
                "response.in_progress".to_string(),
                json!({
                "type": "response.in_progress",
                "response": {
                "id": self.response_id,
                "object": "response",
                "created_at": unix_now(),
                "model": self.model,
                "status": "in_progress",
                "output": []
                }
                }),
            ));
        }

        if let Some(usage) = chunk.get("usage").filter(|u| !u.is_null()) {
            merge_stream_usage(&mut self.usage, usage);
        }

        if let Some(choices) = chunk.get("choices").and_then(|c| c.as_array()) {
            if let Some(choice) = choices.first() {
                if let Some(delta) = choice.get("delta") {
                    if let Some(text) = delta.get("content").and_then(|v| v.as_str()) {
                        if !text.is_empty() {
                            self.ensure_message_item_started(&mut events);
                            // Defence-in-depth: cap the text accumulator so a
                            // misbehaving upstream cannot hold unbounded memory
                            // via the converter's `self.text` (the
                            // incremental `output_text.delta` event is still
                            // emitted). Mirrors the orchestrator's
                            // `MAX_STREAM_BUFFER_BYTES` cap on the wire.
                            if self.text.len() + text.len() <= MAX_STREAM_BUFFER_BYTES {
                                self.text.push_str(text);
                            }
                            let item_id = self.message_item_id.clone().unwrap_or_default();
                            events.push((
                                "response.output_text.delta".to_string(),
                                json!({
                                "type": "response.output_text.delta",
                                "response_id": self.response_id,
                                "item_id": item_id,
                                "output_index": self.message_output_index,
                                "content_index": 0,
                                "delta": text
                                }),
                            ));
                        }
                    }
                    if let Some(reasoning) = delta.get("reasoning_content").and_then(|v| v.as_str())
                    {
                        if !reasoning.is_empty() {
                            if self.reasoning_item_id.is_none() {
                                self.reasoning_output_index = Some(self.next_output_index);
                                self.next_output_index += 1;
                                self.reasoning_item_id =
                                    Some(format!("rs_{}", uuid::Uuid::new_v4()));
                                events.push(("response.output_item.added".into(), json!({
                                    "type":"response.output_item.added","response_id":self.response_id,
                                    "output_index":self.reasoning_output_index,
                                    "item":{"id":self.reasoning_item_id,"type":"reasoning","status":"in_progress","summary":[]}
                                })));
                                events.push(("response.reasoning_summary_part.added".into(), json!({
                                    "type":"response.reasoning_summary_part.added","response_id":self.response_id,
                                    "item_id":self.reasoning_item_id,"output_index":self.reasoning_output_index,
                                    "summary_index":0,"part":{"type":"summary_text","text":""}
                                })));
                            }
                            let item_id = self
                                .reasoning_item_id
                                .get_or_insert_with(|| format!("rs_{}", uuid::Uuid::new_v4()))
                                .clone();
                            if self.reasoning.len() + reasoning.len() <= MAX_STREAM_BUFFER_BYTES {
                                self.reasoning.push_str(reasoning);
                            }
                            events.push((
                                "response.reasoning_summary_text.delta".to_string(),
                                json!({
                                "type": "response.reasoning_summary_text.delta",
                                "response_id": self.response_id,
                                "item_id": item_id,
                                "output_index": self.reasoning_output_index,
                                "summary_index": 0,
                                "delta": reasoning
                                }),
                            ));
                        }
                    }
                    if let Some(tool_calls) = delta.get("tool_calls").and_then(|v| v.as_array()) {
                        for tool_call in tool_calls {
                            let index =
                                tool_call.get("index").and_then(|v| v.as_u64()).unwrap_or(0);
                            // Defence-in-depth: bound the per-stream tool-call
                            // map so a pathological index flood cannot grow it
                            // unboundedly. Distinct indices are normally small.
                            if self.tool_calls.len() >= MAX_STREAM_TOOL_CALLS
                                && !self.tool_calls.contains_key(&index)
                            {
                                tracing::warn!(
                                    tool_call_index = index,
                                    "dropping tool call beyond MAX_STREAM_TOOL_CALLS cap"
                                );
                                continue;
                            }
                            if !self.tool_calls.contains_key(&index) {
                                let output_index = self.next_output_index;
                                self.next_output_index += 1;
                                self.tool_calls.insert(
                                    index,
                                    StreamToolCall {
                                        output_index,
                                        ..Default::default()
                                    },
                                );
                            }
                            let acc = self.tool_calls.entry(index).or_default();
                            let already_added = acc.added_emitted;
                            let mut arguments_delta = None;
                            if let Some(id) = tool_call.get("id").and_then(|v| v.as_str()) {
                                if acc.call_id.is_empty() {
                                    acc.call_id = id.to_string();
                                }
                            }
                            if let Some(function) = tool_call.get("function") {
                                if let Some(name) = function.get("name").and_then(|v| v.as_str()) {
                                    acc.name = name.to_string();
                                }
                                if let Some(arguments) =
                                    function.get("arguments").and_then(|v| v.as_str())
                                {
                                    if !arguments.is_empty() {
                                        // Cap the per-call arguments accumulator
                                        // for defence-in-depth.
                                        if acc.arguments.len() + arguments.len()
                                            <= MAX_STREAM_BUFFER_BYTES
                                        {
                                            acc.arguments.push_str(arguments);
                                        }
                                        arguments_delta = Some(arguments.to_string());
                                    }
                                }
                            }
                            // Synthesise a fallback id if none arrived from
                            // the upstream but a name is present, so the tool
                            // call is not silently dropped from the stream
                            // (matches the non-streaming `build_output_items`
                            // behaviour and the Anthropic converter's id
                            // generation). Previously a tool call carrying a
                            // name + arguments but no id vanished from the
                            // stream entirely.
                            if acc.call_id.is_empty() && !acc.name.is_empty() {
                                acc.call_id = format!("call_{}", uuid::Uuid::new_v4());
                            }
                            // Emit `response.output_item.added` once we know
                            // enough to identify the item (id OR a synthesised
                            // id backed by a name). Doing this AFTER the
                            // id/name assignment (rather than inside the
                            // `id` branch) means the added event also carries
                            // the function name when both fields arrive in the
                            // same delta — matching the OpenAI Responses
                            // streaming spec.
                            if !acc.added_emitted && !acc.call_id.is_empty() {
                                acc.added_emitted = true;
                                let name_for_added = acc.name.clone();
                                let mut item = Map::new();
                                item.insert("type".to_string(), json!("function_call"));
                                item.insert("id".to_string(), json!(acc.call_id));
                                item.insert("call_id".to_string(), json!(acc.call_id));
                                item.insert("status".to_string(), json!("in_progress"));
                                if !name_for_added.is_empty() {
                                    item.insert("name".to_string(), json!(name_for_added));
                                }
                                events.push((
                                    "response.output_item.added".to_string(),
                                    json!({
                                    "type": "response.output_item.added",
                                    "response_id": self.response_id,
                                    "output_index": acc.output_index,
                                    "item": Value::Object(item)
                                    }),
                                ));
                            }
                            let delta = if !already_added && acc.added_emitted {
                                (!acc.arguments.is_empty()).then(|| acc.arguments.clone())
                            } else {
                                arguments_delta
                            };
                            if let Some(delta) = delta.filter(|_| acc.added_emitted) {
                                events.push(("response.function_call_arguments.delta".into(), json!({
                                    "type":"response.function_call_arguments.delta","response_id":self.response_id,
                                    "item_id":acc.call_id,"output_index":acc.output_index,"delta":delta
                                })));
                            }
                        }
                    }
                }

                if let Some(reason) = choice.get("finish_reason").and_then(|v| v.as_str()) {
                    self.finish_reason = Some(reason.to_string());
                    // Hold back `response.completed` until usage arrives. The
                    // chat upstream sends a trailing usage-only chunk after
                    // the finish_reason chunk when `include_usage` is set, and
                    // emitting now would report zero usage. Finalize tool-call
                    // events immediately so they precede completion.
                    self.pending_completion = true;
                    events.extend(self.finalize_events());
                }
            }
        }

        // Emit the deferred completion once usage is available — either in the
        // same chunk as `finish_reason` or in the trailing usage-only chunk.
        if self.pending_completion && self.usage.is_some() {
            self.pending_completion = false;
            self.finished = true;
            events.push(self.build_completed_event());
        }

        self.stamp_seq(&mut events);
        events
    }

    fn finalize_events(&mut self) -> Vec<(String, Value)> {
        if self.items_finalized {
            return Vec::new();
        }
        self.items_finalized = true;
        let mut events = Vec::new();

        if !self.reasoning.is_empty() {
            if let Some(item_id) = &self.reasoning_item_id {
                events.push((
                    "response.reasoning_summary_text.done".to_string(),
                    json!({
                    "type": "response.reasoning_summary_text.done",
                    "response_id": self.response_id,
                    "item_id": item_id,
                    "output_index": self.reasoning_output_index,
                    "summary_index": 0,
                    "text": self.reasoning
                    }),
                ));
                events.push(("response.reasoning_summary_part.done".into(), json!({
                    "type":"response.reasoning_summary_part.done","response_id":self.response_id,
                    "item_id":item_id,"output_index":self.reasoning_output_index,"summary_index":0,
                    "part":{"type":"summary_text","text":self.reasoning}
                })));
                events.push(("response.output_item.done".into(), json!({
                    "type":"response.output_item.done","response_id":self.response_id,
                    "output_index":self.reasoning_output_index,
                    "item":{"id":item_id,"type":"reasoning","status":"completed","summary":[{"type":"summary_text","text":self.reasoning}]}
                })));
            }
        }

        if self.message_item_added {
            let item_id = self.message_item_id.clone().unwrap_or_default();
            events.push((
                "response.output_text.done".to_string(),
                json!({
                "type": "response.output_text.done",
                "response_id": self.response_id,
                "item_id": item_id,
                "output_index": self.message_output_index,
                "content_index": 0,
                "text": self.text
                }),
            ));
            if self.content_part_added {
                events.push((
                    "response.content_part.done".to_string(),
                    json!({
                    "type": "response.content_part.done",
                    "response_id": self.response_id,
                    "item_id": item_id,
                    "output_index": self.message_output_index,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": self.text, "annotations": []}
                    }),
                ));
            }
            events.push((
                "response.output_item.done".to_string(),
                json!({
                "type": "response.output_item.done",
                "response_id": self.response_id,
                "output_index": self.message_output_index,
                "item": {
                "id": item_id,
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": self.text, "annotations": []}]
                }
                }),
            ));
        }

        let mut indices: Vec<u64> = self.tool_calls.keys().copied().collect();
        indices.sort_unstable();
        for index in indices {
            let Some(tool_call) = self.tool_calls.get(&index) else {
                continue;
            };
            if tool_call.call_id.is_empty() {
                continue;
            }
            events.push((
                "response.function_call_arguments.done".to_string(),
                json!({
                "type": "response.function_call_arguments.done",
                "response_id": self.response_id,
                "item_id": tool_call.call_id,
                "output_index": tool_call.output_index,
                "arguments": tool_call.arguments
                }),
            ));
            events.push((
                "response.output_item.done".to_string(),
                json!({
                "type": "response.output_item.done",
                "response_id": self.response_id,
                "output_index": tool_call.output_index,
                "item": {
                "type": "function_call",
                "id": tool_call.call_id,
                "call_id": tool_call.call_id,
                "name": tool_call.name,
                "arguments": tool_call.arguments,
                "status": "completed"
                }
                }),
            ));
        }
        events
    }

    fn build_completed_output(&self) -> Vec<Value> {
        let tool_calls: Vec<Value> = {
            let mut indices: Vec<u64> = self.tool_calls.keys().copied().collect();
            indices.sort_unstable();
            indices
                .into_iter()
                .filter_map(|index| self.tool_calls.get(&index))
                .filter(|tool_call| !tool_call.call_id.is_empty())
                .map(|tool_call| {
                    json!({
                    "id": tool_call.call_id,
                    "type": "function",
                    "function": {
                    "name": tool_call.name,
                    "arguments": tool_call.arguments
                    }
                    })
                })
                .collect()
        };
        let mut message = Map::new();
        message.insert("content".to_string(), json!(self.text));
        if !tool_calls.is_empty() {
            message.insert("tool_calls".to_string(), Value::Array(tool_calls));
        }
        if !self.reasoning.is_empty() {
            message.insert("reasoning_content".to_string(), json!(self.reasoning));
        }
        let mut output = build_output_items(Some(&Value::Object(message)));
        for item in &mut output {
            match item.get("type").and_then(Value::as_str) {
                Some("message") => item["id"] = json!(self.message_item_id),
                Some("reasoning") => item["id"] = json!(self.reasoning_item_id),
                _ => {}
            }
        }
        output.sort_by_key(|item| match item.get("type").and_then(Value::as_str) {
            Some("message") => self.message_output_index.unwrap_or(0),
            Some("reasoning") => self.reasoning_output_index.unwrap_or(0),
            _ => self
                .tool_calls
                .values()
                .find(|tool| item.get("id").and_then(Value::as_str) == Some(tool.call_id.as_str()))
                .map_or(0, |tool| tool.output_index),
        });
        output
    }

    pub(crate) fn finish_events(&mut self) -> Vec<(String, Value)> {
        if self.finished {
            return Vec::new();
        }
        self.finished = true;
        let mut events = Vec::new();
        // If `finish_reason` already arrived, tool-call finalization happened
        // in `convert_chunk`; only finalize here when the stream ended without
        // an explicit finish_reason.
        if !self.pending_completion {
            events.extend(self.finalize_events());
        }
        self.pending_completion = false;
        events.push(self.build_completed_event());
        self.stamp_seq(&mut events);
        events
    }

    fn build_completed_event(&self) -> (String, Value) {
        (
            "response.completed".to_string(),
            json!({
            "type": "response.completed",
            "response": {
            "id": self.response_id,
            "object": "response",
            "created_at": unix_now(),
            "model": self.model,
            "status": responses_status_for_finish_reason(self.finish_reason.as_deref()),
            "output": self.build_completed_output(),
            "usage": build_responses_usage(self.usage.as_ref())
            }
            }),
        )
    }

    /// Emit `done` events for any started message item and tool calls without
    /// emitting `response.completed`. Used by the transport-level stream-read-
    /// error / timeout paths (which synthesise their own terminal `error`
    /// event outside the converter) to keep the item lifecycle balanced —
    /// every `response.output_item.added` must pair with a
    /// `response.output_item.done`, matching the chunk-level error arm in
    /// [`Self::convert_chunk`]. Returns an empty `Vec` when the items were
    /// already finalized (via `finish_reason`) or the converter is finished.
    pub(crate) fn close_open_items(&mut self) -> Vec<(String, Value)> {
        if self.finished || self.pending_completion {
            return Vec::new();
        }
        let mut events = self.finalize_events();
        self.stamp_seq(&mut events);
        events
    }
}

/// Bridge an SSE framing layer and the converter: decode one raw SSE frame
/// into an OpenAI chunk and map it through the converter to Responses events.
/// Pure (no axum/tokio), so it lives next to the converter it drives.
pub(crate) fn process_openai_sse_event_responses(
    converter: &mut ResponsesStreamConverter,
    raw: &str,
) -> Vec<(String, Value)> {
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
                "dropping unparseable Responses SSE data payload"
            );
            return Vec::new();
        }
    };

    converter.convert_chunk(&chunk)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn message_parts_use_chat_types_and_null_controls_are_omitted() {
        let request = json!({"model":"m","input":[{"role":"user","content":[
            {"type":"input_text","text":"look"},
            {"type":"input_image","image_url":"data:image/png;base64,image","detail":"low"}
        ]}],"metadata":null,"user":null,"service_tier":null});
        let chat = responses_to_chat_request(&request);
        assert_eq!(chat["messages"][0]["content"][0]["type"], "text");
        assert_eq!(
            chat["messages"][0]["content"][1]["image_url"]["url"],
            "data:image/png;base64,image"
        );
        assert_eq!(
            chat["messages"][0]["content"][1]["image_url"]["detail"],
            "low"
        );
        for key in ["metadata", "user", "service_tier"] {
            assert!(chat.get(key).is_none());
        }
    }

    #[test]
    fn streamed_item_ids_indices_and_lifecycles_match_completed_output() {
        let mut converter = ResponsesStreamConverter::new();
        let mut events = converter.convert_chunk(&json!({"id":"chatcmpl-items","model":"m","choices":[{"delta":{"reasoning_content":"reason"}}]}));
        events.extend(converter.convert_chunk(
            &json!({"choices":[{"delta":{"content":"answer","tool_calls":[{
            "index":0,"id":"call-1","function":{"name":"lookup","arguments":"{}"}
        }]},"finish_reason":"stop"}]}),
        ));
        events.extend(converter.finish_events());
        let added = events
            .iter()
            .filter(|(kind, _)| kind == "response.output_item.added")
            .map(|(_, event)| event)
            .collect::<Vec<_>>();
        let completed = events
            .iter()
            .find(|(kind, _)| kind == "response.completed")
            .unwrap();
        let output = completed.1["response"]["output"].as_array().unwrap();
        assert_eq!(added.len(), 3);
        let indices = added
            .iter()
            .map(|event| event["output_index"].as_u64().unwrap())
            .collect::<std::collections::HashSet<_>>();
        assert_eq!(indices.len(), 3);
        for item in added {
            let id = &item["item"]["id"];
            assert!(output.iter().any(|output| &output["id"] == id));
            assert_eq!(
                events
                    .iter()
                    .filter(|(kind, event)| kind == "response.output_item.done"
                        && &event["item"]["id"] == id)
                    .count(),
                1
            );
        }
        let tool_added = events
            .iter()
            .position(|(kind, event)| {
                kind == "response.output_item.added" && event["item"]["id"] == "call-1"
            })
            .unwrap();
        let tool_delta = events
            .iter()
            .position(|(kind, _)| kind == "response.function_call_arguments.delta")
            .unwrap();
        assert!(tool_added < tool_delta);
        assert!(
            events
                .windows(2)
                .all(|pair| pair[0].1["sequence_number"].as_u64().unwrap()
                    < pair[1].1["sequence_number"].as_u64().unwrap())
        );
    }

    #[test]
    fn responses_string_input_wraps_as_user_message() {
        let req = json!({
        "model": "gpt-4o",
        "input": "Hello!",
        });
        let chat = responses_to_chat_request(&req);
        let messages = chat["messages"].as_array().expect("messages");
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["role"], "user");
        assert_eq!(messages[0]["content"], "Hello!");
    }

    #[test]
    fn responses_instructions_prepended_as_system() {
        let req = json!({
        "model": "gpt-4o",
        "input": "Hello!",
        "instructions": "Be helpful.",
        });
        let chat = responses_to_chat_request(&req);
        let messages = chat["messages"].as_array().expect("messages");
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[0]["content"], "Be helpful.");
        assert_eq!(messages[1]["role"], "user");
    }

    #[test]
    fn responses_array_input_with_role_passthrough() {
        let req = json!({
        "model": "gpt-4o",
        "input": [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "Bye"},
        ],
        });
        let chat = responses_to_chat_request(&req);
        let messages = chat["messages"].as_array().expect("messages");
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0]["role"], "user");
        assert_eq!(messages[1]["role"], "assistant");
        assert_eq!(messages[2]["role"], "user");
    }

    #[test]
    fn responses_max_output_tokens_mapped_to_max_tokens() {
        let req = json!({
        "model": "gpt-4o",
        "input": "Hi",
        "max_output_tokens": 100,
        });
        let chat = responses_to_chat_request(&req);
        assert_eq!(chat["max_tokens"], 100);
    }

    #[test]
    fn responses_temperature_and_top_p_passthrough() {
        let req = json!({
        "model": "gpt-4o",
        "input": "Hi",
        "temperature": 0.7,
        "top_p": 0.9,
        });
        let chat = responses_to_chat_request(&req);
        assert_eq!(chat["temperature"], 0.7);
        assert_eq!(chat["top_p"], 0.9);
    }

    #[test]
    fn responses_stream_passthrough() {
        let req = json!({
        "model": "gpt-4o",
        "input": "Hi",
        "stream": true,
        });
        let chat = responses_to_chat_request(&req);
        assert_eq!(chat["stream"], true);
    }

    /// C24 LOW: null `stream` / `max_output_tokens` must not be forwarded as
    /// `"stream": null` / `"max_tokens": null`, matching the null-filtering
    /// applied to every sibling optional field (temperature, top_p, …).
    #[test]
    fn responses_null_stream_and_max_output_tokens_not_forwarded() {
        let req = json!({
        "model": "gpt-4o",
        "input": "Hi",
        "stream": null,
        "max_output_tokens": null
        });
        let chat = responses_to_chat_request(&req);
        assert!(
            chat.get("stream").is_none(),
            "null stream must not be forwarded: {chat}"
        );
        assert!(
            chat.get("max_tokens").is_none(),
            "null max_output_tokens must not be forwarded: {chat}"
        );
    }

    #[test]
    fn responses_cache_opt_in_passthrough() {
        // Regression: the converter must forward `cache: {"insert": true}` so
        // the chat pipeline's CacheHook can see the opt-in.
        let req = json!({
        "model": "gpt-4o",
        "input": "Hi",
        "cache": {"insert": true},
        });
        let chat = responses_to_chat_request(&req);
        assert_eq!(chat["cache"], json!({"insert": true}));
        assert!(
            crate::hooks::cache_hook::request_wants_cache(&chat),
            "the converted request must satisfy the cache opt-in check"
        );

        // Absent or null cache fields must not produce one.
        let req = json!({"model": "gpt-4o", "input": "Hi"});
        assert!(responses_to_chat_request(&req).get("cache").is_none());
        let req = json!({"model": "gpt-4o", "input": "Hi", "cache": null});
        assert!(responses_to_chat_request(&req).get("cache").is_none());
    }

    #[test]
    fn chat_response_converted_to_responses_format() {
        let chat = json!({
        "id": "chatcmpl-abc123",
        "model": "gpt-4o",
        "choices": [{
        "index": 0,
        "message": {"role": "assistant", "content": "Hello!"},
        "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        });
        let resp = chat_to_responses_response(&chat);
        assert_eq!(resp["object"], "response");
        assert!(resp["id"].as_str().expect("id").starts_with("resp_"));
        assert_eq!(resp["model"], "gpt-4o");
        assert_eq!(resp["output"][0]["type"], "message");
        assert_eq!(resp["output"][0]["role"], "assistant");
        assert_eq!(resp["output"][0]["content"][0]["type"], "output_text");
        assert_eq!(resp["output"][0]["content"][0]["text"], "Hello!");
        assert_eq!(resp["usage"]["input_tokens"], 10);
        assert_eq!(resp["usage"]["output_tokens"], 5);
        assert_eq!(resp["usage"]["total_tokens"], 15);
        assert_eq!(resp["status"], "completed");
    }

    #[test]
    fn responses_stream_converter_emits_created_delta_completed() {
        let mut converter = ResponsesStreamConverter::new();

        let chunk1 = json!({
        "id": "chatcmpl-1",
        "model": "gpt-4o",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": null}]
        });
        let events1 = converter.convert_chunk(&chunk1);
        assert_eq!(events1.len(), 2);
        assert_eq!(events1[0].0, "response.created");
        assert_eq!(events1[1].0, "response.in_progress");

        let chunk2 = json!({
        "choices": [{"index": 0, "delta": {"content": "Hi"}, "finish_reason": null}]
        });
        let events2 = converter.convert_chunk(&chunk2);
        let types: Vec<&str> = events2.iter().map(|(ty, _)| ty.as_str()).collect();
        assert_eq!(
            types,
            [
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta"
            ]
        );
        let delta_event = events2
            .iter()
            .find(|(ty, _)| ty == "response.output_text.delta")
            .expect("output_text.delta must be emitted");
        assert_eq!(delta_event.1["delta"], "Hi");

        let chunk3 = json!({
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
        });
        let events3 = converter.convert_chunk(&chunk3);
        assert!(
            events3.iter().all(|(ty, _)| ty != "response.completed"),
            "completion must be deferred until usage arrives or stream ends: {events3:?}"
        );

        // Without a usage chunk, the terminal [DONE]/EOF marker flushes the
        // deferred completion.
        let done_events = converter.finish_events();
        assert_eq!(
            done_events.last().expect("terminal completion").0,
            "response.completed"
        );
        assert!(
            converter.finish_events().is_empty(),
            "the terminal [DONE] marker must not emit a second completion"
        );
    }

    #[test]
    fn stream_usage_arrives_after_finish_reason() {
        let mut converter = ResponsesStreamConverter::new();

        let _ = converter.convert_chunk(&json!({
        "id": "chatcmpl-usage",
        "model": "gpt-4o",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": null}]
        }));
        let delta_events = converter.convert_chunk(&json!({
        "choices": [{"index": 0, "delta": {"content": "Hi"}, "finish_reason": null}]
        }));
        let delta = delta_events
            .iter()
            .find(|(ty, _)| ty == "response.output_text.delta")
            .expect("output_text.delta must be emitted");
        assert_eq!(delta.1["delta"], "Hi");

        // `finish_reason` arrives WITHOUT usage: stash it, mark a pending
        // completion, but do NOT emit `response.completed` yet.
        let finish_events = converter.convert_chunk(&json!({
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
        }));
        assert!(
            finish_events
                .iter()
                .all(|(ty, _)| ty != "response.completed"),
            "completion must not be emitted before the usage chunk: {finish_events:?}"
        );

        // The trailing usage-only chunk arrives: NOW emit `response.completed`
        // carrying the real usage numbers.
        let usage_events = converter.convert_chunk(&json!({
        "choices": [],
        "usage": {
        "prompt_tokens": 42,
        "completion_tokens": 7,
        "total_tokens": 49,
        "completion_tokens_details": {"reasoning_tokens": 3}
        }
        }));
        let completed = usage_events
            .iter()
            .find(|(ty, _)| ty == "response.completed")
            .expect("completion must be emitted once the usage chunk arrives");
        let usage = &completed.1["response"]["usage"];
        assert_eq!(usage["input_tokens"], 42);
        assert_eq!(usage["output_tokens"], 7);
        assert_eq!(usage["total_tokens"], 49);
        assert_eq!(usage["output_tokens_details"]["reasoning_tokens"], 3);

        // The terminal [DONE] marker must not emit a duplicate completion.
        assert!(
            converter.finish_events().is_empty(),
            "completion already emitted; [DONE] must not duplicate it"
        );
    }

    #[test]
    fn responses_tools_and_tool_choice_map_to_chat() {
        let req = json!({
        "model": "gpt-4o",
        "input": "Hi",
        "tools": [{
        "type": "function",
        "name": "get_weather",
        "description": "Get weather",
        "parameters": {"properties": {"city": {"type": "string"}}},
        "strict": true
        }],
        "tool_choice": {"type": "function", "name": "get_weather"},
        "parallel_tool_calls": false
        });
        let chat = responses_to_chat_request(&req);
        assert_eq!(chat["tools"][0]["type"], "function");
        assert_eq!(chat["tools"][0]["function"]["name"], "get_weather");
        assert_eq!(chat["tools"][0]["function"]["parameters"]["type"], "object");
        assert_eq!(chat["tools"][0]["function"]["strict"], true);
        assert_eq!(chat["tool_choice"]["function"]["name"], "get_weather");
        assert_eq!(chat["parallel_tool_calls"], false);
    }

    #[test]
    fn responses_function_call_roundtrip_maps_to_chat_messages() {
        let req = json!({
        "model": "gpt-4o",
        "input": [
        {"role": "user", "content": "weather?"},
        {"type": "function_call", "call_id": "call-1", "name": "get_weather", "arguments": "{\"city\":\"SF\"}"},
        {"type": "function_call_output", "call_id": "call-1", "output": [{"type": "input_text", "text": "sunny"}]}
        ]
        });
        let chat = responses_to_chat_request(&req);
        let messages = chat["messages"].as_array().expect("messages");
        assert_eq!(messages[1]["role"], "assistant");
        assert_eq!(messages[1]["tool_calls"][0]["id"], "call-1");
        assert_eq!(
            messages[1]["tool_calls"][0]["function"]["name"],
            "get_weather"
        );
        assert_eq!(messages[2]["role"], "tool");
        assert_eq!(messages[2]["tool_call_id"], "call-1");
        assert_eq!(messages[2]["content"], "sunny");
    }

    #[test]
    fn responses_reasoning_and_stream_usage_are_forwarded() {
        let req = json!({
        "model": "gpt-4o",
        "input": "think",
        "stream": true,
        "reasoning": {"effort": "high"}
        });
        let chat = responses_to_chat_request(&req);
        assert_eq!(chat["reasoning_effort"], "high");
        assert_eq!(chat["stream_options"]["include_usage"], true);
    }

    /// U27: `stream_options` is a chat
    /// completions parameter, not a Responses API parameter. The converter
    /// must NOT pass through a client-supplied `stream_options` field —
    /// only the internally-injected `{include_usage: true}` (set when
    /// `stream: true`) may appear on the converted chat body. A
    /// client-supplied `stream_options` (e.g. with extra fields) would
    /// otherwise leak into the chat request and could confuse the host's
    /// `inject_stream_usage` or the upstream provider.
    #[test]
    fn responses_converter_does_not_passthrough_client_stream_options() {
        let req = json!({
        "model": "gpt-4o",
        "input": "Hi",
        "stream": true,
        // A malicious / confused caller tries to slip stream_options through.
        "stream_options": {"include_usage": false, "custom_field": "evil"}
        });
        let chat = responses_to_chat_request(&req);
        // The converter ALWAYS injects {include_usage: true} when streaming,
        // overriding whatever the client tried to send.
        assert_eq!(
            chat["stream_options"]["include_usage"], true,
            "stream_options must be the converter-injected value, not the client's"
        );
        // The client's custom field must NOT leak through.
        assert!(
            chat["stream_options"].get("custom_field").is_none(),
            "client-supplied stream_options fields must not be passed through: {}",
            chat["stream_options"]
        );
    }

    /// U27 non-streaming variant: when `stream` is false/absent, no
    /// `stream_options` should appear on the converted chat body at all
    /// (neither an injected one nor a passthrough of a client-supplied one).
    #[test]
    fn responses_converter_no_stream_options_when_not_streaming() {
        let req = json!({
        "model": "gpt-4o",
        "input": "Hi",
        "stream": false,
        "stream_options": {"include_usage": true}
        });
        let chat = responses_to_chat_request(&req);
        assert!(
            chat.get("stream_options").is_none(),
            "no stream_options should be present on a non-streaming conversion: {}",
            chat
        );
    }

    #[test]
    fn chat_response_tool_calls_and_usage_details_are_mapped() {
        let chat = json!({
        "id": "chatcmpl-tools",
        "model": "gpt-4o",
        "choices": [{
        "index": 0,
        "message": {
        "role": "assistant",
        "content": null,
        "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\":\"SF\"}"}}]
        },
        "finish_reason": "tool_calls"
        }],
        "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "prompt_tokens_details": {"cached_tokens": 3},
        "completion_tokens_details": {"reasoning_tokens": 2}
        }
        });
        let resp = chat_to_responses_response(&chat);
        assert_eq!(resp["output"][0]["type"], "function_call");
        assert_eq!(resp["output"][0]["call_id"], "call-1");
        assert_eq!(resp["output"][0]["name"], "get_weather");
        assert_eq!(resp["usage"]["input_tokens_details"]["cached_tokens"], 3);
        assert_eq!(
            resp["usage"]["output_tokens_details"]["reasoning_tokens"],
            2
        );
    }

    /// U47: cache-creation tokens reported as
    /// `prompt_tokens_details.cache_write_tokens` (the OpenAI-compatible
    /// spelling) must be surfaced in the Responses usage object.
    #[test]
    fn responses_usage_surfaces_cache_write_tokens_from_prompt_details() {
        let usage = json!({
        "prompt_tokens": 100,
        "completion_tokens": 5,
        "prompt_tokens_details": {"cache_write_tokens": 250}
        });
        let out = build_responses_usage(Some(&usage));
        assert_eq!(out["input_tokens"], 100);
        assert_eq!(out["output_tokens"], 5);
        assert_eq!(
            out["input_tokens_details"]["cache_write_tokens"], 250,
            "cache_write_tokens must be mirrored into the Responses usage"
        );
        assert!(
            out["input_tokens_details"].get("cached_tokens").is_none(),
            "no cached_tokens were reported, so the field must be absent"
        );
    }

    /// U47: the Anthropic top-level `cache_creation_input_tokens` spelling is
    /// also accepted and surfaced as `input_tokens_details.cache_write_tokens`.
    #[test]
    fn responses_usage_surfaces_anthropic_cache_creation_input_tokens() {
        let usage = json!({
        "prompt_tokens": 100,
        "completion_tokens": 5,
        "cache_creation_input_tokens": 80
        });
        let out = build_responses_usage(Some(&usage));
        assert_eq!(out["input_tokens_details"]["cache_write_tokens"], 80);
    }

    /// U47: when BOTH cache-read (`cached_tokens`) and cache-creation
    /// (`cache_write_tokens`) are present, both must appear under
    /// `input_tokens_details`.
    #[test]
    fn responses_usage_preserves_cached_and_write_tokens_together() {
        let usage = json!({
        "prompt_tokens": 1000,
        "completion_tokens": 5,
        "prompt_tokens_details": {"cached_tokens": 400, "cache_write_tokens": 600}
        });
        let out = build_responses_usage(Some(&usage));
        assert_eq!(out["input_tokens_details"]["cached_tokens"], 400);
        assert_eq!(out["input_tokens_details"]["cache_write_tokens"], 600);
    }

    /// U47: no cache fields → no `input_tokens_details` at all (back-compat).
    #[test]
    fn responses_usage_omits_details_when_no_cache_fields() {
        let usage = json!({"prompt_tokens": 10, "completion_tokens": 5});
        let out = build_responses_usage(Some(&usage));
        assert!(out.get("input_tokens_details").is_none());
    }

    /// U48: end-to-end — a chat completion response carrying cache-creation
    /// tokens must surface them on the converted Responses response so the
    /// spend/cost log path that reads off the Responses usage object receives
    /// them. (The billing hook itself runs on the chat-shaped usage via
    /// `extract_usage`, which already reads `cache_write_tokens` — this test
    /// pins the Responses-surface half.)
    #[test]
    fn responses_end_to_end_surfaces_cache_creation_for_logs() {
        let chat = json!({
        "id": "chatcmpl-cache",
        "model": "claude-sonnet-4-20250514",
        "choices": [{
        "index": 0,
        "message": {"role": "assistant", "content": "hi"},
        "finish_reason": "stop"
        }],
        "usage": {
        "prompt_tokens": 1200,
        "completion_tokens": 8,
        "cache_creation_input_tokens": 900,
        "cache_read_input_tokens": 200
        }
        });
        let resp = chat_to_responses_response(&chat);
        // The Responses usage object must carry the cache-creation count for
        // any downstream spend/analytics log reading it.
        assert_eq!(
            resp["usage"]["input_tokens_details"]["cache_write_tokens"],
            900
        );
    }

    #[test]
    fn stream_tool_calls_are_completed_with_arguments() {
        let mut converter = ResponsesStreamConverter::new();
        let _ = converter.convert_chunk(&json!({
        "id": "chatcmpl-stream",
        "model": "gpt-4o",
        "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "call-1", "function": {"name": "get_weather"}}]}, "finish_reason": null}]
        }));
        let _ = converter.convert_chunk(&json!({
        "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\"city\":"}}]}, "finish_reason": null}]
        }));
        let events = converter.convert_chunk(&json!({
        "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "\"SF\"}"}}]}, "finish_reason": "tool_calls"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "completion_tokens_details": {"reasoning_tokens": 1}}
        }));
        let types: Vec<&str> = events.iter().map(|(ty, _)| ty.as_str()).collect();
        assert!(types.contains(&"response.function_call_arguments.done"));
        assert!(types.contains(&"response.output_item.done"));
        assert_eq!(types.last(), Some(&"response.completed"));
        let completed = &events.last().expect("completed").1;
        assert_eq!(completed["response"]["output"][0]["type"], "function_call");
        assert_eq!(
            completed["response"]["output"][0]["arguments"],
            "{\"city\":\"SF\"}"
        );
        assert_eq!(
            completed["response"]["usage"]["output_tokens_details"]["reasoning_tokens"],
            1
        );
    }

    #[test]
    fn responses_stream_converter_finish_idempotent() {
        let mut converter = ResponsesStreamConverter::new();
        let first = converter.finish_events();
        let second = converter.finish_events();
        assert!(!first.is_empty());
        assert!(second.is_empty());
    }

    #[test]
    fn responses_stream_error_does_not_complete() {
        let mut converter = ResponsesStreamConverter::new();
        let events = converter.convert_chunk(&json!({
            "error": {"message": "capacity exhausted", "code": 429}
        }));
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].0, "error");
        assert_eq!(events[0].1["error"]["code"], 429);
        assert!(converter.finish_events().is_empty());
    }

    /// C24 MEDIUM: when a message item is already started (content delta
    /// emitted) and an error chunk arrives, the converter must close the open
    /// item (`output_text.done`, `content_part.done`, `output_item.done`)
    /// BEFORE the terminal `error` event so strict SDK clients that track
    /// item lifecycle see a balanced stream.
    #[test]
    fn responses_stream_error_after_content_closes_open_items() {
        let mut converter = ResponsesStreamConverter::new();

        let _ = converter.convert_chunk(&json!({
        "id": "chatcmpl-err-mid",
        "model": "gpt-4o",
        "choices": [{"index": 0, "delta": {"content": "partial"}, "finish_reason": null}]
        }));

        let error_events = converter.convert_chunk(&json!({
        "error": {"type": "api_error", "message": "capacity"}
        }));
        let types: Vec<&str> = error_events.iter().map(|(ty, _)| ty.as_str()).collect();
        let error_pos = types
            .iter()
            .position(|&t| t == "error")
            .expect("error event");
        let item_done_pos = types
            .iter()
            .position(|&t| t == "response.output_item.done")
            .expect("output_item.done must be emitted before the error");
        assert!(
            item_done_pos < error_pos,
            "output_item.done must precede the error event: {types:?}"
        );
        assert!(
            types.contains(&"response.content_part.done"),
            "content_part.done must be emitted: {types:?}"
        );
        assert!(
            types.contains(&"response.output_text.done"),
            "output_text.done must be emitted: {types:?}"
        );
        assert!(
            converter.finish_events().is_empty(),
            "post-error finish_events must be empty"
        );
    }

    /// C24 MEDIUM: `close_open_items` (used by the handler's stream-read-error
    /// path) must emit done events for a started message item and be a no-op
    /// once the items are already finalized via `finish_reason`.
    #[test]
    fn responses_stream_close_open_items_balances_lifecycle() {
        let mut converter = ResponsesStreamConverter::new();
        let _ = converter.convert_chunk(&json!({
        "id": "chatcmpl-close",
        "model": "gpt-4o",
        "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": null}]
        }));

        let closed = converter.close_open_items();
        assert!(
            closed
                .iter()
                .any(|(ty, _)| ty == "response.output_item.done"),
            "close_open_items must emit output_item.done: {closed:?}"
        );

        // After close, a second call is a no-op (items already finalized).
        assert!(
            converter.close_open_items().is_empty(),
            "second close_open_items must be a no-op"
        );
    }

    #[test]
    fn responses_stream_null_error_does_not_produce_error_event() {
        let mut converter = ResponsesStreamConverter::new();
        let chunk = json!({
            "id": "chatcmpl-1",
            "model": "gpt-4o",
            "choices": [{"index": 0, "delta": {"content": "Hi"}, "finish_reason": null}],
            "error": null
        });
        let events = converter.convert_chunk(&chunk);
        assert!(
            events.iter().all(|(ty, _)| ty != "error"),
            "null error must not produce an error event: {events:?}"
        );
        assert!(!converter.finished, "converter must not be marked finished");
        assert!(
            events.iter().any(|(ty, _)| ty == "response.created"),
            "stream should still emit response.created"
        );
        assert!(
            events
                .iter()
                .any(|(ty, v)| ty == "response.output_text.delta" && v["delta"] == "Hi"),
            "content delta must still be emitted"
        );
    }

    /// U47 streaming: cache-write tokens arriving in the trailing usage chunk
    /// (under `prompt_tokens_details.cache_write_tokens`) must surface on the
    /// `response.completed` event's usage object, so streaming spend logs see
    /// the cache-creation count (U48).
    #[test]
    fn responses_stream_surfaces_cache_write_tokens_on_completion() {
        let mut converter = ResponsesStreamConverter::new();
        let _ = converter.convert_chunk(&json!({
        "id": "chatcmpl-cache-stream",
        "model": "gpt-4o",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": null}]
        }));
        let _ = converter.convert_chunk(&json!({
        "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": "stop"}]
        }));
        let events = converter.convert_chunk(&json!({
        "choices": [],
        "usage": {
        "prompt_tokens": 500,
        "completion_tokens": 4,
        "total_tokens": 504,
        "prompt_tokens_details": {"cache_write_tokens": 300, "cached_tokens": 100}
        }
        }));
        let completed = events
            .iter()
            .find(|(ty, _)| ty == "response.completed")
            .expect("response.completed must be emitted once usage is known");
        let usage = &completed.1["response"]["usage"];
        assert_eq!(usage["input_tokens"], 500);
        assert_eq!(usage["input_tokens_details"]["cached_tokens"], 100);
        assert_eq!(
            usage["input_tokens_details"]["cache_write_tokens"], 300,
            "stream completion must surface cache_write_tokens (U47/U48)"
        );
    }

    #[test]
    fn responses_stream_usage_max_merges_split_usage_chunks() {
        // Anthropic-backed streams split usage across chunks: the first
        // (role) chunk carries the real `prompt_tokens`, and the finish chunk
        // carries `prompt_tokens: 0` plus the real `completion_tokens`. The
        // converter must max-merge (like `merge_streaming_usage`) so the
        // finish chunk cannot clobber the input token count back to zero in
        // `response.completed`.
        let mut converter = ResponsesStreamConverter::new();
        let _ = converter.convert_chunk(&json!({
        "id": "chatcmpl-split",
        "model": "claude-3-5-sonnet",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": null}],
        "usage": {"prompt_tokens": 42, "completion_tokens": 0, "total_tokens": 42}
        }));
        let _ = converter.convert_chunk(&json!({
        "choices": [{"index": 0, "delta": {"content": "Hi"}, "finish_reason": null}]
        }));
        let events = converter.convert_chunk(&json!({
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 13, "total_tokens": 13}
        }));
        let completed = events
            .iter()
            .find(|(ty, _)| ty == "response.completed")
            .expect("response.completed must be emitted once usage is known");
        let usage = &completed.1["response"]["usage"];
        assert_eq!(
            usage["input_tokens"], 42,
            "finish chunk's prompt_tokens: 0 must not clobber the real count"
        );
        assert_eq!(usage["output_tokens"], 13);
        assert_eq!(usage["total_tokens"], 55);
    }

    /// U29: reasoning input items (`{"type":"reasoning","summary":[...]}`)
    /// carry prior-turn reasoning context that has no chat-completions
    /// equivalent. They must be dropped silently instead of falling into the
    /// catchall arm that wraps them as malformed user messages.
    #[test]
    fn responses_reasoning_input_item_is_dropped() {
        let req = json!({
        "model": "gpt-4o",
        "input": [
        {"role": "user", "content": "What is 2+2?"},
        {"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "thinking about addition"}]},
        {"role": "user", "content": "Explain."},
        ],
        });
        let chat = responses_to_chat_request(&req);
        let messages = chat["messages"].as_array().expect("messages");
        assert_eq!(
            messages.len(),
            2,
            "reasoning item must be dropped, not wrapped as a message: {messages:?}"
        );
        assert_eq!(messages[0]["role"], "user");
        assert_eq!(messages[0]["content"], "What is 2+2?");
        assert_eq!(messages[1]["role"], "user");
        assert_eq!(messages[1]["content"], "Explain.");
        // The reasoning summary must not leak into any message content.
        assert!(
            messages.iter().all(|m| {
                m.get("content")
                    .map(|c| {
                        !c.to_string().contains("summary_text")
                            && !c.to_string().contains("thinking about addition")
                    })
                    .unwrap_or(true)
            }),
            "reasoning summary must not leak as a user message: {messages:?}"
        );
    }

    /// U29: a reasoning item as the only input still yields a valid (empty)
    /// messages array rather than a malformed user message.
    #[test]
    fn responses_reasoning_only_input_yields_no_messages() {
        let req = json!({
        "model": "gpt-4o",
        "input": [
        {"type": "reasoning", "summary": [{"type": "summary_text", "text": "prior reasoning"}]},
        ],
        });
        let chat = responses_to_chat_request(&req);
        let messages = chat["messages"].as_array().expect("messages");
        assert!(
            messages.is_empty(),
            "sole reasoning item must produce no messages: {messages:?}"
        );
    }

    /// A `reasoning: null` field must NOT be forwarded as
    /// `reasoning_effort: null` — matching how every other optional field
    /// (temperature, top_p, cache) is filtered. A null reasoning_effort can
    /// confuse downstream providers that expect a string.
    #[test]
    fn responses_null_reasoning_is_not_forwarded_as_reasoning_effort() {
        let req = json!({"model": "gpt-4o", "input": "Hi", "reasoning": null});
        let chat = responses_to_chat_request(&req);
        assert!(
            chat.get("reasoning_effort").is_none(),
            "reasoning: null must not produce reasoning_effort: {chat}"
        );

        // Sanity: a real effort still flows through.
        let req = json!({"model": "gpt-4o", "input": "Hi", "reasoning": {"effort": "high"}});
        assert_eq!(responses_to_chat_request(&req)["reasoning_effort"], "high");
    }

    /// C24 MEDIUM: once `response.completed` has fired, trailing chunks (a
    /// duplicate `finish_reason` chunk or a stray content delta from a
    /// non-conforming upstream / keepalive proxy) must NOT emit a second
    /// completion, post-completion output_text deltas, or duplicate
    /// output_item.done events. Mirrors the Anthropic converter's
    /// `duplicate_finish_reason_chunks_emit_a_single_message_delta`.
    #[test]
    fn responses_stream_finished_guard_drops_post_completion_chunks() {
        let mut converter = ResponsesStreamConverter::new();
        let _ = converter.convert_chunk(&json!({
        "id": "chatcmpl-guard",
        "model": "gpt-4o",
        "choices": [{"index": 0, "delta": {"content": "Hi"}, "finish_reason": null}]
        }));

        // finish_reason + usage: emits completion now.
        let finish_events = converter.convert_chunk(&json!({
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        }));
        let completions = finish_events
            .iter()
            .filter(|(ty, _)| ty == "response.completed")
            .count();
        assert_eq!(
            completions, 1,
            "exactly one completion expected: {finish_events:?}"
        );

        // Duplicate finish_reason chunk: must be a complete no-op.
        let dup = converter.convert_chunk(&json!({
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
        }));
        assert!(
            dup.is_empty(),
            "post-completion finish chunk must produce no events: {dup:?}"
        );

        // Stray content delta after completion: must NOT emit a
        // response.output_text.delta (spec violation) or any other event.
        let stray = converter.convert_chunk(&json!({
        "choices": [{"index": 0, "delta": {"content": "late"}, "finish_reason": null}]
        }));
        assert!(
            stray.is_empty(),
            "post-completion content delta must produce no events: {stray:?}"
        );

        // And [DONE] must not duplicate the completion.
        assert!(
            converter.finish_events().is_empty(),
            "completion already emitted; [DONE] must not duplicate it"
        );
    }

    /// C24 LOW: `build_responses_usage` must surface both cached_tokens and
    /// cache_write_tokens under input_tokens_details and nothing else (the
    /// dead `preserved` branch was removed).
    #[test]
    fn responses_usage_details_carry_only_cache_fields() {
        let usage = json!({
        "prompt_tokens": 1000,
        "completion_tokens": 5,
        "prompt_tokens_details": {"cached_tokens": 400, "cache_write_tokens": 600}
        });
        let out = build_responses_usage(Some(&usage));
        let details = &out["input_tokens_details"];
        assert_eq!(details["cached_tokens"], 400);
        assert_eq!(details["cache_write_tokens"], 600);
        assert_eq!(
            details.as_object().expect("object").len(),
            2,
            "no extraneous keys (e.g. preserved) should appear: {details}"
        );
    }

    /// C19 MEDIUM: a provider-supplied usage with `prompt_tokens` near
    /// `u64::MAX` must not panic debug builds or wrap release builds when
    /// `build_responses_usage` computes `total_tokens`. Mirrors the billing
    /// path's `extract_usage_saturates_total_on_overflow`.
    #[test]
    fn build_responses_usage_saturates_total_on_overflow() {
        let usage = json!({
            "prompt_tokens": u64::MAX,
            "completion_tokens": 1
        });
        let out = build_responses_usage(Some(&usage));
        assert_eq!(out["input_tokens"], u64::MAX);
        assert_eq!(out["output_tokens"], 1);
        assert_eq!(
            out["total_tokens"],
            u64::MAX,
            "total must saturate, not panic or wrap"
        );
    }

    // ================================================================
    // C24 LOW: custom_tool_call `input` is normalised to `arguments`
    // ================================================================

    /// A `custom_tool_call` input item carries its payload in `input` (raw
    /// string), not `arguments`. The converter must normalise the input to
    /// an arguments JSON string `{"content": ...}` (matching the reference) so the
    /// chat pipeline receives a non-empty payload. Without the fix, the
    /// payload vanished and the converted tool call had `arguments: ""`.
    #[test]
    fn responses_custom_tool_call_input_normalised_to_arguments() {
        let req = json!({
        "model": "gpt-4o",
        "input": [
        {"type": "custom_tool_call", "call_id": "c1", "name": "f", "input": "hello"}
        ]
        });
        let chat = responses_to_chat_request(&req);
        let messages = chat["messages"].as_array().expect("messages");
        let tool_call = &messages[0]["tool_calls"][0];
        assert_eq!(tool_call["id"], "c1");
        assert_eq!(tool_call["function"]["name"], "f");
        let args = tool_call["function"]["arguments"].as_str().expect("str");
        assert!(!args.is_empty(), "arguments must not be empty: {args}");
        let parsed: Value = serde_json::from_str(args).expect("must be valid JSON");
        assert_eq!(parsed["content"], "hello");
    }

    /// Sanity: a `function_call` (not custom) still uses `arguments` directly.
    #[test]
    fn responses_function_call_arguments_unchanged() {
        let req = json!({
        "model": "gpt-4o",
        "input": [
        {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{\"x\":1}"}
        ]
        });
        let chat = responses_to_chat_request(&req);
        let args = &chat["messages"][0]["tool_calls"][0]["function"]["arguments"];
        assert_eq!(args, "{\"x\":1}");
    }

    // ================================================================
    // C24 LOW: tool_result / custom_tool_call_output input items
    // ================================================================

    /// A `tool_result` (Anthropic/MCP spelling) input item must map to a
    /// tool message using `tool_use_id` and `content`. Previously it fell
    /// into the catchall and was wrapped as a malformed user message.
    #[test]
    fn responses_tool_result_input_item_mapped_to_tool_message() {
        let req = json!({
        "model": "gpt-4o",
        "input": [
        {"role": "user", "content": "hi"},
        {"type": "tool_result", "tool_use_id": "t1", "content": "ok"}
        ]
        });
        let chat = responses_to_chat_request(&req);
        let messages = chat["messages"].as_array().expect("messages");
        let tool_msg = messages
            .iter()
            .find(|m| m["role"] == "tool")
            .expect("tool message must be produced: {messages:?}");
        assert_eq!(tool_msg["tool_call_id"], "t1");
        assert_eq!(tool_msg["content"], "ok");
    }

    /// `custom_tool_call_output` maps to a tool message using `call_id` and
    /// `output` (the standard spelling). Verifies the new arm covers it
    /// instead of falling into the catchall.
    #[test]
    fn responses_custom_tool_call_output_mapped_to_tool_message() {
        let req = json!({
        "model": "gpt-4o",
        "input": [
        {"type": "custom_tool_call_output", "call_id": "c1", "output": "done"}
        ]
        });
        let chat = responses_to_chat_request(&req);
        let messages = chat["messages"].as_array().expect("messages");
        let tool_msg = messages
            .iter()
            .find(|m| m["role"] == "tool")
            .expect("tool message must be produced: {messages:?}");
        assert_eq!(tool_msg["tool_call_id"], "c1");
        assert_eq!(tool_msg["content"], "done");
    }

    /// `web_search_call` has no clean chat equivalent — it must be silently
    /// dropped, not wrapped as a user message.
    #[test]
    fn responses_web_search_call_input_item_dropped() {
        let req = json!({
        "model": "gpt-4o",
        "input": [
        {"role": "user", "content": "search?"},
        {"type": "web_search_call", "id": "ws_1", "status": "completed"}
        ]
        });
        let chat = responses_to_chat_request(&req);
        let messages = chat["messages"].as_array().expect("messages");
        assert_eq!(
            messages.len(),
            1,
            "web_search_call must be dropped, not wrapped: {messages:?}"
        );
        assert_eq!(messages[0]["role"], "user");
    }

    // ================================================================
    // C24 LOW: reasoning effort+summary precedence
    // ================================================================

    /// When `reasoning` carries both `effort` and `summary`, the converter
    /// must forward the full dict (so providers that use `summary` receive
    /// it), matching the reference precedence. Previously oxllm emitted just
    /// the effort string.
    #[test]
    fn responses_reasoning_with_summary_keeps_full_dict() {
        let req = json!({
        "model": "gpt-4o",
        "input": "Hi",
        "reasoning": {"effort": "high", "summary": "detailed"}
        });
        let chat = responses_to_chat_request(&req);
        let effort = &chat["reasoning_effort"];
        assert!(effort.is_object(), "effort must be the full dict: {effort}");
        assert_eq!(effort["effort"], "high");
        assert_eq!(effort["summary"], "detailed");
    }

    /// Sanity: dict with only `effort` still forwards the bare string.
    #[test]
    fn responses_reasoning_effort_only_forwards_string() {
        let req = json!({
        "model": "gpt-4o",
        "input": "Hi",
        "reasoning": {"effort": "low"}
        });
        let chat = responses_to_chat_request(&req);
        assert_eq!(chat["reasoning_effort"], "low");
    }

    /// Sanity: a bare string reasoning is forwarded as-is.
    #[test]
    fn responses_reasoning_bare_string_forwards_as_is() {
        let req = json!({
        "model": "gpt-4o",
        "input": "Hi",
        "reasoning": "minimal"
        });
        let chat = responses_to_chat_request(&req);
        assert_eq!(chat["reasoning_effort"], "minimal");
    }

    // ================================================================
    // C24 LOW: normalize_function_call_output preserves image parts
    // ================================================================

    /// A multimodal tool output array containing both text and image parts
    /// must promote the output to a list of `{"type":"text"|"image_url"}`
    /// blocks so the image is not lost. Mirrors the reference
    /// `_normalize_function_call_output_to_tool_content`.
    #[test]
    fn responses_function_call_output_preserves_image_parts() {
        let output = json!([
        {"type": "input_text", "text": "screenshot below"},
        {"type": "input_image", "image_url": {"url": "https://example.com/i.png"}}
        ]);
        let normalised = normalize_function_call_output(&output);
        let arr = normalised
            .as_array()
            .expect("multimodal output must be a content array");
        assert!(
            arr.iter()
                .any(|p| p["type"] == "text" && p["text"] == "screenshot below"),
            "text part must be preserved: {arr:?}"
        );
        assert!(
            arr.iter().any(|p| p["type"] == "image_url"
                && p["image_url"]["url"] == "https://example.com/i.png"),
            "image_url part must be preserved: {arr:?}"
        );
    }

    /// Sanity: a text-only array output still joins to a string (back-compat
    /// with the previous behaviour).
    #[test]
    fn responses_function_call_output_text_only_still_joins() {
        let output = json!([
        {"type": "input_text", "text": "hello "},
        {"type": "output_text", "text": "world"}
        ]);
        let normalised = normalize_function_call_output(&output);
        assert_eq!(normalised, "hello world");
    }

    /// C24 MEDIUM: a bare-string `image_url` (the canonical Responses
    /// `input_image` shape) must be preserved, not silently dropped. The
    /// object form (`{"image_url":{"url":"..."}}`) is already handled; this
    /// pins the string form.
    #[test]
    fn responses_function_call_output_bare_string_image_url_preserved() {
        let output = json!([
        {"type": "input_image", "image_url": "data:image/png;base64,iVBOR"}
        ]);
        let normalised = normalize_function_call_output(&output);
        let arr = normalised
            .as_array()
            .expect("bare-string image must promote to a content array");
        assert!(
            arr.iter().any(|p| p["type"] == "image_url"
                && p["image_url"]["url"] == "data:image/png;base64,iVBOR"),
            "bare-string image_url must be preserved: {arr:?}"
        );
    }

    /// End-to-end: a `function_call_output` whose `output` carries a
    /// bare-string `input_image` must produce a tool message whose `content`
    /// array contains an `image_url` part.
    #[test]
    fn responses_function_call_output_bare_string_image_roundtrips_to_tool_message() {
        let req = json!({
        "model": "gpt-4o",
        "input": [
        {"type": "function_call_output", "call_id": "c1", "output": [
            {"type": "input_image", "image_url": "data:image/png;base64,iVBOR"}
        ]}
        ]
        });
        let chat = responses_to_chat_request(&req);
        let messages = chat["messages"].as_array().expect("messages");
        let tool_msg = messages
            .iter()
            .find(|m| m["role"] == "tool")
            .expect("tool message must be produced: {messages:?}");
        let content = tool_msg["content"]
            .as_array()
            .expect("content must be an array for a multimodal tool output");
        assert!(
            content.iter().any(|p| p["type"] == "image_url"
                && p["image_url"]["url"] == "data:image/png;base64,iVBOR"),
            "image_url part must survive into the tool message: {content:?}"
        );
    }

    // ================================================================
    // C24 LOW: streaming tool call without id is not dropped
    // ================================================================

    /// A streaming tool call that arrives with `name` + `arguments` but no
    /// `id` must NOT be silently dropped from the completed output. The
    /// converter synthesises a fallback `call_<uuid>` id (matching the
    /// non-streaming path and the Anthropic converter's behaviour).
    #[test]
    fn responses_stream_tool_call_without_id_synthesises_fallback() {
        let mut converter = ResponsesStreamConverter::new();
        let events = converter.convert_chunk(&json!({
        "id": "chatcmpl-noid",
        "model": "gpt-4o",
        "choices": [{
        "index": 0,
        "delta": {
        "tool_calls": [{"index": 0, "function": {"name": "get_weather", "arguments": "{\"city\":\"SF\"}"}}]
        },
        "finish_reason": "tool_calls"
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        }));
        let completed = events
            .iter()
            .find(|(ty, _)| ty == "response.completed")
            .expect("response.completed must be emitted");
        let output = &completed.1["response"]["output"];
        let tool_call = output
            .as_array()
            .expect("output array")
            .iter()
            .find(|o| o["type"] == "function_call")
            .expect("function_call must be present in output: {output}");
        assert_eq!(tool_call["name"], "get_weather");
        assert_eq!(tool_call["arguments"], "{\"city\":\"SF\"}");
        assert!(
            !tool_call["call_id"].as_str().unwrap_or("").is_empty(),
            "synthesised call_id must not be empty: {tool_call}"
        );
    }

    // ================================================================
    // C24 LOW: output_item.added carries the function name
    // ================================================================

    /// When a streaming tool call delta arrives with both `id` and `name`,
    /// the `response.output_item.added` event must include the function
    /// `name` on the item, matching the OpenAI Responses streaming spec.
    #[test]
    fn responses_stream_output_item_added_includes_name_when_known() {
        let mut converter = ResponsesStreamConverter::new();
        let events = converter.convert_chunk(&json!({
        "id": "chatcmpl-name",
        "model": "gpt-4o",
        "choices": [{
        "index": 0,
        "delta": {
        "tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "get_weather"}}]
        },
        "finish_reason": null
        }]
        }));
        let added = events
            .iter()
            .find(|(ty, _)| ty == "response.output_item.added")
            .expect("output_item.added must be emitted");
        assert_eq!(added.1["item"]["name"], "get_weather");
        assert_eq!(added.1["item"]["call_id"], "call_1");
    }

    // ================================================================
    // C24 LOW: malformed SSE data: payloads are logged (and dropped)
    // ================================================================

    /// A malformed `data:` payload must yield no events (pins current
    /// drop-on-error behaviour). The accompanying `tracing::debug!` log is
    /// not asserted here — the report flagged the silent drop, and any
    /// observability hook (e.g. `tracing_subscriber`) can capture the line.
    #[test]
    fn responses_stream_malformed_data_payload_yields_no_events() {
        let mut converter = ResponsesStreamConverter::new();
        let events = process_openai_sse_event_responses(&mut converter, "data: not-json");
        assert!(
            events.is_empty(),
            "malformed payload must not produce events: {events:?}"
        );
    }

    // ================================================================
    // C19 MEDIUM: streaming delta.reasoning_content
    // ================================================================

    /// A reasoning model (e.g. DeepSeek-R1, o1/o3) that streams
    /// `delta.reasoning_content` must produce
    /// `response.reasoning_summary_text.delta` events and a `reasoning` output
    /// item in `response.completed`. Previously the reasoning content was
    /// silently dropped, producing zero reasoning events and no reasoning item
    /// in the completed response.
    #[test]
    fn responses_stream_reasoning_content_produces_delta_events_and_output_item() {
        let mut converter = ResponsesStreamConverter::new();

        let _ = converter.convert_chunk(&json!({
        "id": "chatcmpl-rs",
        "model": "deepseek-r1",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": null}]
        }));
        let reasoning_events = converter.convert_chunk(&json!({
        "choices": [{"index": 0, "delta": {"reasoning_content": "thinking..."}, "finish_reason": null}]
        }));
        let reasoning_delta = reasoning_events
            .iter()
            .find(|(ty, _)| ty == "response.reasoning_summary_text.delta")
            .expect("reasoning_summary_text.delta must be emitted for reasoning_content");
        assert_eq!(reasoning_delta.1["delta"], "thinking...");

        let content_events = converter.convert_chunk(&json!({
        "choices": [{"index": 0, "delta": {"content": "answer"}, "finish_reason": null}]
        }));
        assert!(
            content_events
                .iter()
                .any(|(ty, _)| ty == "response.output_text.delta"),
            "content delta must still be emitted: {content_events:?}"
        );

        let finish_events = converter.convert_chunk(&json!({
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }));
        let completed = finish_events
            .iter()
            .find(|(ty, _)| ty == "response.completed")
            .expect("response.completed must be emitted");
        let output = &completed.1["response"]["output"];
        let reasoning_item = output
            .as_array()
            .expect("output array")
            .iter()
            .find(|o| o["type"] == "reasoning")
            .expect("reasoning output item must be present in completed response");
        assert!(
            reasoning_item["summary"][0]["text"]
                .as_str()
                .unwrap_or("")
                .contains("thinking..."),
            "reasoning item summary must contain the accumulated reasoning: {reasoning_item}"
        );
        assert!(
            output
                .as_array()
                .unwrap()
                .iter()
                .any(|o| o["type"] == "message"),
            "message output item must also be present"
        );

        assert!(
            finish_events
                .iter()
                .any(|(ty, _)| ty == "response.reasoning_summary_text.done"),
            "reasoning_summary_text.done must be emitted on finalize: {finish_events:?}"
        );
    }

    // ================================================================
    // C19 LOW: intermediate streaming events + sequence_number
    // ================================================================

    /// The converter must emit `response.in_progress` immediately after
    /// `response.created`, and the full message-item lifecycle
    /// (`output_item.added` → `content_part.added` → deltas →
    /// `output_text.done` → `content_part.done` → `output_item.done`) for a
    /// simple text stream, matching the OpenAI Responses streaming spec.
    #[test]
    fn responses_stream_emits_intermediate_events_in_spec_order() {
        let mut converter = ResponsesStreamConverter::new();

        let init = converter.convert_chunk(&json!({
        "id": "chatcmpl-order",
        "model": "gpt-4o",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": null}]
        }));
        let init_types: Vec<&str> = init.iter().map(|(ty, _)| ty.as_str()).collect();
        assert_eq!(init_types, ["response.created", "response.in_progress"]);

        let delta = converter.convert_chunk(&json!({
        "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": null}]
        }));
        let delta_types: Vec<&str> = delta.iter().map(|(ty, _)| ty.as_str()).collect();
        assert_eq!(
            delta_types,
            [
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta"
            ]
        );

        let finish = converter.convert_chunk(&json!({
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        }));
        let finish_types: Vec<&str> = finish.iter().map(|(ty, _)| ty.as_str()).collect();
        assert!(
            finish_types
                .windows(4)
                .any(|w| w[0] == "response.output_text.done"
                    && w[1] == "response.content_part.done"
                    && w[2] == "response.output_item.done"
                    && w[3] == "response.completed"),
            "output_text.done → content_part.done → output_item.done → completed must appear in order: {finish_types:?}"
        );
    }

    /// Every streaming event must carry a `sequence_number` and they must be
    /// monotonically increasing across the full event stream.
    #[test]
    fn responses_stream_events_carry_monotonic_sequence_number() {
        let mut converter = ResponsesStreamConverter::new();
        let mut all_seq: Vec<u64> = Vec::new();

        for (_, payload) in converter.convert_chunk(&json!({
        "id": "chatcmpl-seq",
        "model": "gpt-4o",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": null}]
        })) {
            all_seq.push(payload["sequence_number"].as_u64().expect("sequence_number"));
        }
        for (_, payload) in converter.convert_chunk(&json!({
        "choices": [{"index": 0, "delta": {"content": "Hi"}, "finish_reason": null}]
        })) {
            all_seq.push(
                payload["sequence_number"]
                    .as_u64()
                    .expect("sequence_number"),
            );
        }
        for (_, payload) in converter.finish_events() {
            all_seq.push(
                payload["sequence_number"]
                    .as_u64()
                    .expect("sequence_number"),
            );
        }

        assert!(all_seq.len() > 3, "expected several events: {all_seq:?}");
        for i in 1..all_seq.len() {
            assert!(
                all_seq[i] > all_seq[i - 1],
                "sequence_number must be strictly increasing: {all_seq:?}"
            );
        }
    }
}
