//! Token estimation.
//!
//! For OpenAI models we use [`tiktoken_rs`] (the same BPE tokenizer OpenAI
//! ships). For other providers we fall back to `cl100k_base` — the same
//! default encoding the reference uses for unrecognised models (`_fix_model_name`
//! normalises unknown names to `gpt-3.5-turbo`, which is `cl100k_base`). Only
//! if `cl100k_base` itself fails to load do we degrade to a `byte length / 4`
//! ASCII heuristic. Estimation never panics on user input — tokenizer errors
//! return `0`.

use serde_json::Value;
use std::sync::{Mutex, OnceLock};

use std::collections::HashSet;

/// Decide which tokenizer family a model uses.
///
/// - `o200k_base` for GPT-4o / 4o-mini / o1-family (newer encoding).
/// - `cl100k_base` for GPT-4 / GPT-3.5 (legacy encoding).
/// - Returns `None` for non-OpenAI models; the caller falls back to
///   `cl100k_base` (matching the reference token counter), then to the byte
///   heuristic if cl100k is unavailable.
///
/// Ordering is significant: the `gpt-4.1` / `gpt-4.5` / `gpt-5` (and
/// `gpt-4o`) checks MUST precede the `gpt-4` catch-all below — those newer
/// models share the `gpt-4` prefix but use `o200k_base`, not `cl100k_base`.
/// Adding a future `gpt-4.*` model that uses a newer encoding must extend the
/// first arm, not rely on the `gpt-4` fallback.
fn tokenizer_for(model: &str) -> Option<&'static str> {
    // Aggregators can nest provider prefixes (openrouter/openai/gpt-4o).
    // The tokenizer family belongs to the final model segment.
    let bare = match model.rsplit_once('/') {
        Some((_, rest)) if !rest.is_empty() => rest,
        _ => model,
    };
    let lower = bare.to_ascii_lowercase();
    if lower.starts_with("gpt-4o")
        || lower.starts_with("gpt-4.1")
        || lower.starts_with("gpt-4.5")
        || lower.starts_with("gpt-5")
        || lower.starts_with("gpt-6")
        || lower.starts_with("o1")
        || lower.starts_with("o3")
        || lower.starts_with("o4")
        || lower.starts_with("chatgpt-4o")
    {
        Some("o200k_base")
    } else if lower.starts_with("gpt-4") || lower.starts_with("gpt-3.5") {
        Some("cl100k_base")
    } else {
        if looks_like_openai(&lower) {
            warn_unknown_openai_model_once(model);
        }
        None
    }
}

/// Heuristic for whether a model name *looks* OpenAI (so that a new prefix
/// we don't recognise yet is observable instead of silently heuristing).
fn looks_like_openai(lower: &str) -> bool {
    lower.starts_with("gpt-")
        || lower.starts_with("chatgpt-")
        || lower
            .as_bytes()
            .get(1)
            .is_some_and(|b| lower.starts_with('o') && b.is_ascii_digit())
}

/// Upper bound on the number of distinct unknown-model names we record for
/// dedup. A process running for weeks under an adversary spraying unique
/// OpenAI-looking model names (`gpt-<random>` per request) would otherwise leak
/// memory monotonically; legitimate traffic never approaches this many distinct
/// unknown prefixes.
const MAX_WARNED_MODELS: usize = 1024;

/// Record `model` in `set`, returning `true` only the first time a given name
/// is seen. The set is capped at [`MAX_WARNED_MODELS`] distinct entries: once
/// full, a novel name returns `false` (no warning) rather than growing the set.
/// Pure helper extracted so the dedup + cap logic is testable in isolation
/// from the global warned-set.
fn mark_warned(set: &mut HashSet<String>, model: &str) -> bool {
    if set.len() >= MAX_WARNED_MODELS && !set.contains(model) {
        return false;
    }
    set.insert(model.to_string())
}

/// Emit the "OpenAI-looking model did not match" warning at most once per
/// model name. A single request calls `tokenizer_for` once per message field
/// and once per tool; without dedup a 50-message request floods the log with
/// the same line 100+ times.
fn warn_unknown_openai_model_once(model: &str) {
    static WARNED: Mutex<Option<HashSet<String>>> = Mutex::new(None);
    let Ok(mut guard) = WARNED.lock() else {
        return;
    };
    let set = guard.get_or_insert_with(HashSet::new);
    if mark_warned(set, model) {
        tracing::warn!(
            model = model,
            "OpenAI-looking model did not match any known tokenizer prefix; \
             falling back to cl100k_base"
        );
    }
}

/// Cached o200k tokenizer (initialization is expensive).
fn o200k() -> Option<&'static tiktoken_rs::CoreBPE> {
    static TOK: OnceLock<Option<tiktoken_rs::CoreBPE>> = OnceLock::new();
    TOK.get_or_init(|| match tiktoken_rs::o200k_base() {
        Ok(bpe) => Some(bpe),
        Err(e) => {
            tracing::error!(error = %e, "failed to load o200k_base tokenizer");
            None
        }
    })
    .as_ref()
}

/// Cached cl100k tokenizer.
fn cl100k() -> Option<&'static tiktoken_rs::CoreBPE> {
    static TOK: OnceLock<Option<tiktoken_rs::CoreBPE>> = OnceLock::new();
    TOK.get_or_init(|| match tiktoken_rs::cl100k_base() {
        Ok(bpe) => Some(bpe),
        Err(e) => {
            tracing::error!(error = %e, "failed to load cl100k_base tokenizer");
            None
        }
    })
    .as_ref()
}

/// Estimate the number of tokens in `text` for the given model.
///
/// - Models with a known tokenizer prefix (`gpt-4o` family → `o200k_base`,
///   `gpt-4` / `gpt-3.5` → `cl100k_base`) are counted exactly via
///   `tiktoken_rs`.
/// - All other models (Claude, Gemini, Cohere, …) fall back to `cl100k_base`,
///   matching the reference token counter (which normalises unknown names to
///   `gpt-3.5-turbo`); the previous `byte length / 4` heuristic materially
///   mis-estimated both ASCII and CJK for those providers.
/// - Text is encoded with `encode_ordinary` (no special-token collapsing),
///   matching how the chat API tokenizes user content.
/// - Never panics: tokenizer errors fall back to the byte heuristic.
pub fn estimate_tokens(text: &str, model: &str) -> u64 {
    if text.is_empty() {
        return 0;
    }
    match tokenizer_for(model) {
        Some("o200k_base") => match o200k() {
            Some(bpe) => bpe.encode_ordinary(text).len() as u64,
            None => cl100k_or_heuristic(text),
        },
        _ => cl100k_or_heuristic(text),
    }
}

/// Report the `tokenizer_type` string for [`estimate_tokens`] on `model`,
/// matching the reference token-count response's `tokenizer_type` field.
/// The reference returns `str(_select_tokenizer(model)["type"])`, which is
/// `"openai_tokenizer"` or `"huggingface_tokenizer"` — never an encoding name
/// (`proxy_server.py` `tokenizer_used`, `utils.py _return_openai_tokenizer`).
/// oxllm counts every model with tiktoken BPE (o200k/cl100k), so the truthful
/// value is always `"openai_tokenizer"`; the concrete encoding remains visible
/// via the model prefix rules in `tokenizer_for`.
pub fn tokenizer_type_for(_model: &str) -> &'static str {
    "openai_tokenizer"
}

/// `cl100k_base`-based fallback used both for known cl100k models and for any
/// model without a dedicated tokenizer. Degrades to the byte heuristic only
/// if cl100k failed to load at process start.
fn cl100k_or_heuristic(text: &str) -> u64 {
    match cl100k() {
        Some(bpe) => bpe.encode_ordinary(text).len() as u64,
        None => (text.len() / 4) as u64,
    }
}

/// Force initialization of both cached tokenizers. The `OnceLock` constructors
/// parse the BPE ranks files (tens of milliseconds); without a pre-warm that
/// cost is paid lazily inside the first `estimate_tokens` call — potentially
/// on a tokio worker thread, stalling the first billed request. Callers should
/// invoke this from `spawn_blocking` during startup; the lazy path remains as
/// a fallback and is a no-op once warmed.
pub fn warm_tokenizers() {
    o200k();
    cl100k();
}

/// Minimum token contribution for any image content part. OpenAI's published
/// rate starts at 85 tokens (the low/auto-detail floor); counting at least
/// this keeps vision requests from scoring ~0 prompt tokens before
/// post-success settlement.
const PER_IMAGE_FLOOR_TOKENS: u64 = 85;

/// Conservative floor for a `detail:"high"` image. The reference
/// `calculate_img_tokens(mode="high")` returns `85 + (170 × tiles)` after
/// decoding the image dimensions; the minimum (a single tile) is therefore
/// `255`. We cannot decode dimensions on the hot path without a blocking fetch
/// / decode, so this 1-tile floor is still an under-estimate for large images,
/// but it is materially closer than the 85 floor and keeps high-detail vision
/// requests from being under-reserved by 3× during the in-flight window.
/// `post_success` re-bases to the provider's real usage afterward.
const PER_IMAGE_HIGH_DETAIL_FLOOR_TOKENS: u64 = 255;

/// Minimum token contribution for any audio content part (`input_audio`).
/// Audio is metered by duration upstream; without decoding the blob we apply a
/// conservative floor so audio-bearing requests reserve nonzero TPM.
const PER_AUDIO_FLOOR_TOKENS: u64 = 50;

/// Estimate the prompt-token contribution of a message's `content` field.
/// String content is tokenized directly; array content is summed across parts.
/// Mirrors the reference `_count_content_list` (`token_counter.py:692-740`): text,
/// image, audio, `tool_use`, `tool_result`, `thinking`, `tool_reference` and
/// bare-string elements are all counted, and any unrecognised block is
/// serialized so it never silently scores 0 on a billing path (the reference raises
/// here; we degrade to a conservative count instead of panicking).
fn count_content_tokens(content: &Value, model: &str) -> u64 {
    match content {
        Value::String(s) => estimate_tokens(s, model),
        Value::Array(parts) => {
            let mut total = 0u64;
            for p in parts {
                total = total.saturating_add(count_content_part(p, model));
            }
            total
        }
        _ => 0,
    }
}

/// Floor for a single image content part based on its `detail` field.
/// `image_url` may be an object (`{"url","detail"}`, chat completions form) or
/// a bare URL string (`input_image`, responses-API form) where `detail` lives
/// at the part top level.
fn image_part_tokens(p: &Value) -> u64 {
    let detail = p
        .get("image_url")
        .and_then(|v| v.get("detail"))
        .or_else(|| p.get("detail"))
        .and_then(|v| v.as_str());
    match detail {
        Some("high") => PER_IMAGE_HIGH_DETAIL_FLOOR_TOKENS,
        _ => PER_IMAGE_FLOOR_TOKENS,
    }
}

/// Token contribution of a single content-block element, covering every block
/// type the reference `_count_content_list` recognises. A bare-string element
/// (`isinstance(c, str)`) is counted as text; an unrecognised object type is
/// serialized so it contributes nonzero rather than being dropped.
fn count_content_part(p: &Value, model: &str) -> u64 {
    if let Some(s) = p.as_str() {
        return estimate_tokens(s, model);
    }
    match p.get("type").and_then(|t| t.as_str()) {
        Some("text") => p
            .get("text")
            .map(|t| count_arguments(t, model))
            .unwrap_or(0),
        Some("image_url") | Some("input_image") => image_part_tokens(p),
        Some("input_audio") => PER_AUDIO_FLOOR_TOKENS,
        Some("tool_use") => {
            let mut t = 0u64;
            if let Some(name) = p.get("name").and_then(|v| v.as_str()) {
                t = t.saturating_add(estimate_tokens(name, model));
            }
            if let Some(input) = p.get("input") {
                t = t.saturating_add(count_arguments(input, model));
            }
            t
        }
        Some("tool_result") => p
            .get("content")
            .map(|c| count_content_tokens(c, model))
            .unwrap_or(0),
        Some("thinking") => p
            .get("thinking")
            .and_then(|v| v.as_str())
            .map(|t| estimate_tokens(t, model))
            .unwrap_or(0),
        Some("tool_reference") => p
            .get("tool_name")
            .and_then(|v| v.as_str())
            .map(|t| estimate_tokens(t, model))
            .unwrap_or(0),
        _ => {
            let s = serde_json::to_string(p).unwrap_or_default();
            estimate_tokens(&s, model).max(1)
        }
    }
}

/// Per-message structural overhead (OpenAI convention: every message follows
/// `<|start|>{role/name}\n{content}<|end|>\n`). The reference `_MessageCountParams`
/// uses `3` for every model except the obsolete `gpt-3.5-turbo-0301` (which
/// used `4` / a negative `tokens_per_name`); that model has been deprecated
/// since the April 2021 cutoff and is not routed in practice, so its
/// special-case is intentionally not implemented here. The role name is
/// embedded in this constant and is NOT counted separately. Divergence from
/// the reference `_count_messages` (which BPE-counts *every* string field,
/// including `role` and unknown custom fields, plus Anthropic
/// `search_results`): those extra few tokens per message are intentionally
/// dropped here — the role text is covered by this constant and unknown
/// string fields / `search_results` traffic is rare enough that the
/// under-estimate stays within the post-success settlement correction.
const PER_MESSAGE_OVERHEAD_TOKENS: u64 = 3;

/// Additional overhead when a message carries a `name` field (the reference
/// `tokens_per_name = 1`). Applied on top of the `name` text itself.
const PER_NAME_OVERHEAD_TOKENS: u64 = 1;

/// Assistant-reply priming tokens added once for any message-bearing request.
/// The reference `_count_extra` (`token_counter.py:505`) starts with
/// `num_tokens = 3  # every reply is primed with <|start|>assistant<|message|>`
/// on the prompt-count path (`count_response_tokens is False`). This is
/// independent of the tool-formatting `+9`/`-4` adjustments (which oxllm omits
/// because it BPE-counts the serialized tool JSON instead), so the base `+3`
/// applies unconditionally whenever `messages` is present. Omitted on the
/// embeddings / rerank / prompt-only arms, where the reference `text` path applies.
const REPLY_PRIMING_TOKENS: u64 = 3;

/// Minimum token count for any non-empty tool definition: covers the
/// `function.name` + `.description` + `.parameters` JSON-schema scaffolding
/// that surrounds the user-supplied strings. Used as a floor so a tiny tool
/// still contributes meaningfully to the request-size estimate.
const PER_TOOL_FLOOR_TOKENS: u64 = 50;

/// Token count for a `tool_calls[].function.arguments`, legacy
/// `function_call.arguments`, or content-part `text` value. The OpenAI wire
/// format specifies both `arguments` and `text` as strings, but non-conforming
/// clients sometimes send a JSON object/array/number. The reference handles this
/// defensively via `str(...)` (`token_counter.py:432, 437, 700` — its text
/// branch counts `str(c.get("text", ""))`); we mirror that by stringifying any
/// non-null, non-string value before counting, so a large object-form payload
/// never silently scores 0 and under-reserves budget.
fn count_arguments(value: &Value, model: &str) -> u64 {
    match value {
        Value::String(s) => estimate_tokens(s, model),
        Value::Null => 0,
        other => {
            let s = serde_json::to_string(other).unwrap_or_default();
            estimate_tokens(&s, model)
        }
    }
}

/// Estimate the total prompt tokens of a chat completion request by summing
/// token estimates across all messages, then adding tool-schema overhead.
///
/// Components:
/// - Each message: `content` text + 3-token structural overhead (OpenAI
///   convention; the role name is folded into the constant, not BPE-counted
///   on top, matching the reference). Assistant messages also account for
///   `tool_calls` (and the legacy `function_call`) and the per-message `name`
///   field — `arguments` is frequently the largest field in an agent
///   conversation. Image/audio content parts each contribute a floor so
///   multimodal requests do not score ~0.
/// - `tools` array: each tool definition is estimated with the model's
///   tokenizer (falling back to cl100k for non-OpenAI models), floored at 50
///   tokens so the schema scaffolding alone counts even when
///   name/description are short.
/// - `tool_choice`: `"none"` → +1, object form → +7 + the function-name
///   tokens (the reference `_count_extra`).
/// - Embeddings (`{model, input}`) and rerank (`{model, query, documents}`)
///   bodies carry no `messages`; we count them too so TPM pre-throttling and
///   budget reservation engage before post-success settlement.
/// - Gemini-shaped `contents` (`[{role, parts: [{text}, ...]}]`) is counted as
///   a best-effort sum of its `parts[*].text` fields (non-text parts are
///   skipped); the reference routes this shape through Google's `/countTokens`
///   endpoint when `call_endpoint=True`, so the local count is an estimate.
pub fn estimate_request_tokens(request: &Value) -> u64 {
    let model = request.get("model").and_then(|v| v.as_str()).unwrap_or("");

    let mut total = 0u64;
    // Native Messages places system instructions outside the message array.
    // They still consume prompt tokens for admission and budget reservations.
    if let Some(system) = request.get("system") {
        total = total.saturating_add(count_content_tokens(system, model));
    }
    let mut message_count = 0u64;
    let mut has_messages = false;
    if let Some(messages) = request.get("messages").and_then(|v| v.as_array()) {
        has_messages = true;
        for msg in messages {
            // The reference `_count_messages` adds `tokens_per_message` for every
            // entry of the array unconditionally — even a degenerate empty
            // `{}` still consumes framing tokens.
            message_count = message_count.saturating_add(1);
            if let Some(content) = msg.get("content") {
                total = total.saturating_add(count_content_tokens(content, model));
            }
            // Assistant `tool_calls`: `content` is typically null here, so
            // the function name + arguments body would otherwise be dropped.
            if let Some(tool_calls) = msg.get("tool_calls").and_then(|v| v.as_array()) {
                for tc in tool_calls {
                    if let Some(id) = tc.get("id").and_then(|v| v.as_str()) {
                        total = total.saturating_add(estimate_tokens(id, model));
                    }
                    if let Some(func) = tc.get("function") {
                        if let Some(name) = func.get("name").and_then(|v| v.as_str()) {
                            total = total.saturating_add(estimate_tokens(name, model));
                        }
                        if let Some(args) = func.get("arguments") {
                            total = total.saturating_add(count_arguments(args, model));
                        }
                    }
                }
            }
            // Legacy single `function_call`.
            if let Some(fc) = msg.get("function_call").and_then(|v| v.as_object()) {
                if let Some(name) = fc.get("name").and_then(|v| v.as_str()) {
                    total = total.saturating_add(estimate_tokens(name, model));
                }
                if let Some(args) = fc.get("arguments") {
                    total = total.saturating_add(count_arguments(args, model));
                }
            }
            // Per-message `name` field used to disambiguate participants.
            // Counted as text plus the +1 structural bump (the reference
            // `tokens_per_name`).
            if let Some(name) = msg.get("name").and_then(|v| v.as_str()) {
                total = total.saturating_add(estimate_tokens(name, model));
                total = total.saturating_add(PER_NAME_OVERHEAD_TOKENS);
            }
        }
    }
    // OpenAI-style per-message structural overhead.
    total = total.saturating_add(message_count.saturating_mul(PER_MESSAGE_OVERHEAD_TOKENS));
    // Assistant-reply priming: the reference `_count_extra` adds an unconditional
    // base `+3` on the prompt-count path whenever `messages` is present. Omitted
    // for the embeddings / rerank / prompt-only arms (the `text` path).
    if has_messages {
        total = total.saturating_add(REPLY_PRIMING_TOKENS);
    }

    // Embeddings / rerank / Gemini-contents bodies carry no `messages`;
    // without this branch they would estimate to 0 and bypass TPM
    // pre-throttling and budget reservation until post-success settlement.
    // Gated on `messages` not being a usable *array* (not mere key absence):
    // a malformed non-array `messages` value contributes nothing to the
    // messages loop above, so the prompt/input arms must still run or such a
    // body silently estimates 0. The reference concatenates list inputs for
    // `text`-mode counting; we mirror that for `input` arrays.
    if request.get("messages").and_then(|v| v.as_array()).is_none() {
        if let Some(input) = request.get("input") {
            match input {
                Value::String(s) => total = total.saturating_add(estimate_tokens(s, model)),
                Value::Array(arr) => {
                    for item in arr {
                        if let Some(s) = item.as_str() {
                            total = total.saturating_add(estimate_tokens(s, model));
                        }
                    }
                }
                _ => {}
            }
        }
        if let Some(query) = request.get("query").and_then(|v| v.as_str()) {
            total = total.saturating_add(estimate_tokens(query, model));
        }
        if let Some(documents) = request.get("documents").and_then(|v| v.as_array()) {
            for doc in documents {
                if let Some(s) = doc.as_str() {
                    total = total.saturating_add(estimate_tokens(s, model));
                }
            }
        }
        // The reference `/utils/token_counter` accepts a top-level `prompt` string
        // (its `text` argument) as an alternative to `messages`. Counted as
        // raw text with no per-message overhead, matching the reference pure-text
        // count path. Only reached when `messages` is not a usable array so a
        // body carrying both is not double-counted.
        if let Some(prompt) = request.get("prompt").and_then(|v| v.as_str()) {
            total = total.saturating_add(estimate_tokens(prompt, model));
        }
        // Gemini-shaped `contents` (accepted by the reference `/utils/token_counter`
        // alongside `prompt`/`messages`). The reference routes this through Google's
        // `/countTokens` endpoint when `call_endpoint=True`; for local counting
        // we sum the `parts[*].text` fields as a best-effort estimate so a
        // `contents`-only body no longer silently reports `total_tokens: 0`.
        // Non-text parts (inline_data, function_call, etc.) are skipped — they
        // do not map cleanly to BPE tokens.
        if let Some(contents) = request.get("contents").and_then(|v| v.as_array()) {
            for content in contents {
                if let Some(parts) = content.get("parts").and_then(|v| v.as_array()) {
                    for part in parts {
                        if let Some(text) = part.get("text").and_then(|v| v.as_str()) {
                            total = total.saturating_add(estimate_tokens(text, model));
                        }
                    }
                }
            }
        }
    }

    // Tool definitions can easily account for thousands of tokens; ignoring
    // them was undercounting requests that carry a function/tool schema.
    // Serialize each tool and run it through the same estimator as message
    // content so OpenAI models get BPE-accurate counts; non-OpenAI models
    // transparently fall back to cl100k inside `estimate_tokens`.
    // Intentional divergence from the reference `_count_extra`, which counts a
    // compact TypeScript-ish rendering (`_format_function_definitions`) plus a
    // flat `+9` (`−4` with a system message): the JSON serialization includes
    // quotes/braces and typically counts ~1.2–2× higher. For the TPM/budget
    // reservation this over-estimate is the safe direction; the `/utils/
    // token_counter` handler strips `tools` when reference response parity is
    // required instead.
    if let Some(tools) = request.get("tools").and_then(|v| v.as_array()) {
        for tool in tools {
            let serialized = serde_json::to_string(tool).unwrap_or_default();
            let estimate = estimate_tokens(&serialized, model).max(PER_TOOL_FLOOR_TOKENS);
            total = total.saturating_add(estimate);
        }
    }

    // `tool_choice` framing (the reference `_count_extra`): `"none"` → +1, an
    // object naming a function → +7 + the function-name tokens. The flat
    // `+9`/system-`−4` adjustments in `_count_extra` apply to the reference
    // TypeScript-style formatted tool representation; oxllm serializes tools
    // as JSON instead, which already accounts for that framing via BPE, so
    // those two adjustments are intentionally omitted.
    if let Some(tc) = request.get("tool_choice") {
        match tc {
            Value::String(s) if s == "none" => {
                total = total.saturating_add(1);
            }
            Value::Object(_) if tc.get("function").is_some() => {
                total = total.saturating_add(7);
                if let Some(name) = tc
                    .get("function")
                    .and_then(|f| f.get("name"))
                    .and_then(|n| n.as_str())
                {
                    total = total.saturating_add(estimate_tokens(name, model));
                }
            }
            _ => {}
        }
    }

    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn native_system_instructions_contribute_to_prompt_reservations() {
        let mut request = json!({
            "model":"anthropic/claude-sonnet-4-20250514",
            "messages":[{"role":"user","content":"hello"}]
        });
        let baseline = estimate_request_tokens(&request);
        let text = "Follow these detailed system instructions.".repeat(500);
        let expected = estimate_tokens(&text, request["model"].as_str().unwrap());
        request["system"] = json!(text);
        assert_eq!(estimate_request_tokens(&request), baseline + expected);
        request["system"] =
            json!([{"type":"text","text":text,"cache_control":{"type":"ephemeral"}}]);
        assert_eq!(estimate_request_tokens(&request), baseline + expected);
    }

    #[test]
    fn estimate_tokens_for_openai_model_is_nonzero() {
        let n = estimate_tokens("hello world from an openai model", "gpt-4o");
        assert!(n > 0, "expected nonzero estimate, got {n}");
    }

    #[test]
    fn estimate_tokens_empty_returns_zero() {
        assert_eq!(estimate_tokens("", "gpt-4o"), 0);
    }

    /// Non-OpenAI models fall back to `cl100k_base`, matching the reference
    /// `token_counter` (which normalises unknown names to `gpt-3.5-turbo`).
    #[test]
    fn estimate_tokens_non_openai_uses_cl100k_fallback() {
        let text = "the quick brown fox jumps over the lazy dog";
        let n = estimate_tokens(text, "claude-3-opus");
        let expected = cl100k()
            .expect("cl100k must load for tests")
            .encode_ordinary(text)
            .len() as u64;
        assert_eq!(n, expected);
    }

    #[test]
    fn estimate_tokens_strips_provider_prefix() {
        // ASCII greetings often have identical counts in both encodings,
        // so use text that detects an accidental cl100k fallback.
        let text = "お誕生日おめでとう";
        let expected = o200k()
            .expect("o200k must load")
            .encode_ordinary(text)
            .len() as u64;
        let legacy = cl100k()
            .expect("cl100k must load")
            .encode_ordinary(text)
            .len() as u64;
        assert_ne!(expected, legacy, "fixture must distinguish encodings");

        for model in ["gpt-4o", "openai/gpt-4o", "openrouter/openai/gpt-4o"] {
            assert_eq!(estimate_tokens(text, model), expected, "model {model}");
            let request = json!({
                "model": model,
                "messages": [{"role": "user", "content": text}]
            });
            assert_eq!(
                estimate_request_tokens(&request),
                expected + PER_MESSAGE_OVERHEAD_TOKENS + REPLY_PRIMING_TOKENS,
                "request model {model}"
            );
        }
    }

    #[test]
    fn tokenizer_for_classifies_correctly() {
        assert_eq!(tokenizer_for("gpt-4o"), Some("o200k_base"));
        assert_eq!(tokenizer_for("gpt-4o-mini"), Some("o200k_base"));
        assert_eq!(tokenizer_for("openai/gpt-4o"), Some("o200k_base"));
        assert_eq!(
            tokenizer_for("openrouter/openai/gpt-4o"),
            Some("o200k_base")
        );
        assert_eq!(
            tokenizer_for("openrouter/openai/O3-mini"),
            Some("o200k_base")
        );
        assert_eq!(
            tokenizer_for("openrouter/openai/gpt-4"),
            Some("cl100k_base")
        );
        assert_eq!(tokenizer_for("openrouter/anthropic/claude-3-opus"), None);
        assert_eq!(tokenizer_for("openrouter/openai/"), None);
        assert_eq!(tokenizer_for("gpt-4"), Some("cl100k_base"));
        assert_eq!(tokenizer_for("gpt-3.5-turbo"), Some("cl100k_base"));
        assert_eq!(tokenizer_for("claude-3-opus"), None);
        assert_eq!(tokenizer_for("o1-preview"), Some("o200k_base"));
        assert_eq!(tokenizer_for("gpt-4.1"), Some("o200k_base"));
        assert_eq!(tokenizer_for("gpt-4.1-mini"), Some("o200k_base"));
        assert_eq!(tokenizer_for("gpt-4.5"), Some("o200k_base"));
        assert_eq!(tokenizer_for("gpt-5"), Some("o200k_base"));
        assert_eq!(tokenizer_for("gpt-5-mini"), Some("o200k_base"));
        assert_eq!(tokenizer_for("o4-mini"), Some("o200k_base"));
    }

    #[test]
    fn estimate_request_tokens_sums_messages() {
        let req = json!({
        "model": "gpt-4o",
        "messages": [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello there!"}
        ]
        });
        let total = estimate_request_tokens(&req);
        assert!(total > 0, "expected nonzero estimate, got {total}");
    }

    #[test]
    fn estimate_request_tokens_handles_array_content() {
        let req = json!({
        "model": "gpt-4o",
        "messages": [{
        "role": "user",
        "content": [
        {"type": "text", "text": "hello world"},
        {"type": "image_url", "image_url": {"url": "http://example.com/x.png"}}
        ]
        }]
        });
        let total = estimate_request_tokens(&req);
        assert!(total > 0, "expected nonzero estimate, got {total}");
    }

    #[test]
    fn estimate_request_tokens_handles_missing_fields() {
        let req = json!({"foo": "bar"});
        assert_eq!(estimate_request_tokens(&req), 0);

        let req2 = json!({"model": "gpt-4o", "messages": "not an array"});
        assert_eq!(estimate_request_tokens(&req2), 0);
    }

    /// Bug #53: a request carrying a `tools` array must produce a strictly
    /// higher estimate than the same request without tools — tool schemas
    /// routinely contribute thousands of tokens that were being silently
    /// dropped.
    #[test]
    fn estimate_request_tokens_counts_tools_overhead() {
        let base = json!({
        "model": "gpt-4o",
        "messages": [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is the weather in Paris?"}
        ]
        });
        let with_tools = json!({
        "model": "gpt-4o",
        "messages": [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is the weather in Paris?"}
        ],
        "tools": [
        {
        "type": "function",
        "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
        "type": "object",
        "properties": {
        "city": {"type": "string", "description": "The city to query."},
        "unit": {"type": "string", "enum": ["c", "f"]}
        },
        "required": ["city"]
        }
        }
        }
        ]
        });
        let without = estimate_request_tokens(&base);
        let with = estimate_request_tokens(&with_tools);
        assert!(
            with > without,
            "tools must raise the estimate: with={with}, without={without}"
        );
        // The tool schema above is hundreds of bytes → far above the 50 floor.
        assert!(
            with - without >= 50,
            "tool overhead must contribute at least the per-tool floor (50 tokens), got delta {}",
            with - without
        );
    }

    /// Per-message structural overhead (OpenAI's 3-token convention, matching
    /// the reference) must apply to each message in the array.
    #[test]
    fn estimate_request_tokens_adds_per_message_overhead() {
        let one_msg = json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}]
        });
        let three_msgs = json!({
        "model": "gpt-4o",
        "messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "hi"}
        ]
        });
        let one = estimate_request_tokens(&one_msg);
        let three = estimate_request_tokens(&three_msgs);
        // Two extra messages → 2 × (content "hi" + 3 overhead) more tokens.
        assert!(
            three >= one + 8,
            "expected at least 8 extra tokens from two extra messages, got one={one}, three={three}"
        );
    }

    /// Special-token literals must NOT be collapsed: `encode_with_special_tokens`
    /// counted each `<|endoftext|>` occurrence as one special token, so this
    /// string scored exactly 100. With `encode_ordinary` the literal is counted
    /// like ordinary text (matching the chat API's treatment of content).
    #[test]
    fn estimate_tokens_does_not_collapse_special_token_literals() {
        let special = "<|endoftext|>".repeat(100);
        let n = estimate_tokens(&special, "gpt-4o");
        assert!(
            n > 150,
            "special-token literal must not collapse to ~occurrence count (got {n})"
        );
        // The count must scale with input length, not occurrence count.
        let half = "<|endoftext|>".repeat(50);
        let n_half = estimate_tokens(&half, "gpt-4o");
        assert!(
            n > n_half,
            "doubling the literal must raise the count: n={n}, n_half={n_half}"
        );
    }

    /// Non-OpenAI models now use `cl100k_base` (matching the reference) instead of
    /// the `byte length / 4` heuristic, which under-counted CJK. Each CJK
    /// char is roughly one cl100k token, far above the old `4` for 18 bytes.
    #[test]
    fn estimate_tokens_non_openai_counts_cjk_via_cl100k() {
        let cjk = "你好你好你好"; // 6 chars, 18 bytes
        let n = estimate_tokens(cjk, "claude-3");
        let expected = cl100k()
            .expect("cl100k must load for tests")
            .encode_ordinary(cjk)
            .len() as u64;
        assert_eq!(n, expected);
        assert!(
            n > 4,
            "cl100k must beat the old byte/4 heuristic for CJK (was 4, got {n})"
        );
    }

    /// Assistant `tool_calls` (with null `content`) must contribute the
    /// function name + arguments body, which is often the largest field.
    #[test]
    fn estimate_request_tokens_counts_assistant_tool_calls() {
        let big_args = serde_json::to_string(&(0..50).collect::<Vec<_>>()).unwrap();
        let with = json!({
        "model": "gpt-4o",
        "messages": [
        {"role": "user", "content": "call the tool"},
        {"role": "assistant", "content": null, "tool_calls": [
        {"id": "call_1", "type": "function", "function": {
        "name": "search", "arguments": big_args
        }}
        ]}
        ]
        });
        let without = json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "call the tool"}]
        });
        let w = estimate_request_tokens(&with);
        let wo = estimate_request_tokens(&without);
        assert!(
            w > wo,
            "tool_calls arguments must raise the estimate: with={w}, without={wo}"
        );
        assert!(
            w - wo > 4,
            "tool_calls body must be counted beyond message overhead, delta={}",
            w - wo
        );
    }

    /// The legacy `function_call` field is counted the same way as `tool_calls`.
    #[test]
    fn estimate_request_tokens_counts_legacy_function_call() {
        let with = json!({
        "model": "gpt-4o",
        "messages": [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": null, "function_call": {
        "name": "lookup", "arguments": "{\"q\":\"a fairly long query string\"}"
        }}
        ]
        });
        let without = json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "go"}]
        });
        assert!(
            estimate_request_tokens(&with) > estimate_request_tokens(&without),
            "legacy function_call must contribute tokens"
        );
    }

    /// For an OpenAI model the serialized tool definition must be tokenized with
    /// BPE, not approximated as `bytes/4` (which is the non-OpenAI fallback).
    #[test]
    fn estimate_request_tokens_tokenizes_tool_definition_for_openai() {
        let description = "The quick brown fox jumps over the lazy dog. ".repeat(10);
        let tool = json!({
        "type": "function",
        "function": {
        "name": "do_thing",
        "description": description,
        "parameters": {"type": "object"}
        }
        });
        let serialized = serde_json::to_string(&tool).unwrap();
        let expected = estimate_tokens(&serialized, "gpt-4o").max(PER_TOOL_FLOOR_TOKENS);

        let base = json!({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]});
        let mut with_tool = base.clone();
        with_tool["tools"] = json!([tool]);
        let delta = estimate_request_tokens(&with_tool) - estimate_request_tokens(&base);
        assert_eq!(
            delta, expected,
            "tool contribution should be BPE-accurate for gpt-4o"
        );
    }

    /// An OpenAI-looking model name we don't recognise yet (`o9-*`) should fall
    /// through to the heuristic visibly rather than silently mis-classifying.
    #[test]
    fn tokenizer_for_unknown_openai_looking_model_falls_through() {
        assert_eq!(tokenizer_for("o9-preview"), None);
        assert!(looks_like_openai("o9-preview"));
        assert!(!looks_like_openai("claude-3-opus"));
        assert!(!looks_like_openai("opus"));
        assert!(looks_like_openai("gpt-6"));
        assert!(looks_like_openai("chatgpt-4o"));
    }

    /// The dedup set must record each unknown model name exactly once.
    #[test]
    fn mark_warned_dedupes_per_model() {
        let mut set = HashSet::new();
        assert!(mark_warned(&mut set, "o9-preview"));
        assert!(
            !mark_warned(&mut set, "o9-preview"),
            "same model must dedup"
        );
        assert!(
            mark_warned(&mut set, "o11-preview"),
            "a different model must warn again"
        );
    }

    /// An image content part must contribute at least the per-image floor (85
    /// tokens) ON TOP of any text in the same message; the delta vs. the
    /// text-only variant must be ≥ 85 so vision requests reserve real TPM.
    #[test]
    fn estimate_request_tokens_counts_image_content_floor() {
        let text_only = json!({
        "model": "gpt-4o",
        "messages": [{
        "role": "user",
        "content": [{"type": "text", "text": "what is in this picture"}]
        }]
        });
        let mut with_image = text_only.clone();
        with_image["messages"][0]["content"] = json!([
        {"type": "text", "text": "what is in this picture"},
        {"type": "image_url", "image_url": {"url": "http://example.com/cat.png"}}
        ]);
        let without = estimate_request_tokens(&text_only);
        let with = estimate_request_tokens(&with_image);
        assert!(
            with - without >= PER_IMAGE_FLOOR_TOKENS,
            "image part must add at least the floor ({}), got delta {}",
            PER_IMAGE_FLOOR_TOKENS,
            with - without
        );
        assert!(with >= 85, "vision request must not score ~0, got {with}");
    }

    /// An audio content part must contribute at least the per-audio floor.
    #[test]
    fn estimate_request_tokens_counts_audio_content_floor() {
        let base = json!({
        "model": "gpt-4o-audio-preview",
        "messages": [{"role": "user", "content": "transcribe this"}]
        });
        let mut with_audio = base.clone();
        with_audio["messages"][0]["content"] = json!([
        {"type": "text", "text": "transcribe this"},
        {"type": "input_audio", "input_audio": {"data": "base64", "format": "wav"}}
        ]);
        let delta = estimate_request_tokens(&with_audio) - estimate_request_tokens(&base);
        assert!(
            delta >= PER_AUDIO_FLOOR_TOKENS,
            "audio part must add at least the floor ({}), got delta {delta}",
            PER_AUDIO_FLOOR_TOKENS
        );
    }

    /// Embeddings bodies (`{model, input}`) must estimate nonzero prompt
    /// tokens and scale with input length — otherwise TPM pre-throttling and
    /// budget reservation are bypassed until post-success settlement.
    #[test]
    fn estimate_request_tokens_counts_embeddings_input() {
        let small = json!({"model": "text-embedding-3-small", "input": "hello"});
        let big = json!({
        "model": "text-embedding-3-small",
        "input": "hello ".repeat(100)
        });
        let small_n = estimate_request_tokens(&small);
        let big_n = estimate_request_tokens(&big);
        assert!(
            small_n > 0,
            "embeddings must not estimate to 0, got {small_n}"
        );
        assert!(
            big_n > small_n,
            "estimate must scale with input length: small={small_n}, big={big_n}"
        );

        // Array-of-strings input is concatenated (each entry counted).
        let arr = json!({
        "model": "text-embedding-3-small",
        "input": ["hello world", "another sentence"]
        });
        let arr_n = estimate_request_tokens(&arr);
        assert!(arr_n > 0, "array input must be counted, got {arr_n}");
    }

    /// Rerank bodies (`{model, query, documents}`) must estimate nonzero
    /// prompt tokens covering both the query and every document.
    #[test]
    fn estimate_request_tokens_counts_rerank_body() {
        let single = json!({
        "model": "rerank-english-v3.0",
        "query": "what is oxllm",
        "documents": ["oxllm is a gateway"]
        });
        let multi = json!({
        "model": "rerank-english-v3.0",
        "query": "what is oxllm",
        "documents": [
        "oxllm is a gateway",
        "it is written in rust",
        "it re-implements litellm semantics"
        ]
        });
        let single_n = estimate_request_tokens(&single);
        let multi_n = estimate_request_tokens(&multi);
        assert!(
            single_n > 0,
            "rerank must not estimate to 0, got {single_n}"
        );
        assert!(
            multi_n > single_n,
            "more documents must raise the estimate: single={single_n}, multi={multi_n}"
        );
    }

    /// Gemini-shaped `contents` (`[{role, parts: [{text}, ...]}]`) accepted by
    /// the reference `/utils/token_counter` must be counted locally as a
    /// best-effort sum of its `parts[*].text` fields, so a `contents`-only
    /// body no longer silently reports `total_tokens: 0`. Non-text parts are
    /// skipped; the estimate scales with text length.
    #[test]
    fn estimate_request_tokens_counts_gemini_contents() {
        let small = json!({
        "model": "gemini-1.5-pro",
        "contents": [{"role": "user", "parts": [{"text": "hello world"}]}]
        });
        let big = json!({
        "model": "gemini-1.5-pro",
        "contents": [
        {"role": "user", "parts": [{"text": "hello world "}, {"text": "this is longer"}]},
        {"role": "model", "parts": [{"text": "and a reply"}]}
        ]
        });
        let small_n = estimate_request_tokens(&small);
        let big_n = estimate_request_tokens(&big);
        assert!(
            small_n > 0,
            "gemini contents must not estimate to 0, got {small_n}"
        );
        assert!(
            big_n > small_n,
            "more text parts must raise the estimate: small={small_n}, big={big_n}"
        );

        // Non-text parts (inline_data, function_call, etc.) are skipped, so a
        // contents block with no text parts estimates to 0 — they do not map
        // cleanly to BPE tokens.
        let no_text = json!({
        "model": "gemini-1.5-pro",
        "contents": [{"role": "user", "parts": [{"inline_data": {"mime_type": "image/png", "data": "base64"}}]}]
        });
        assert_eq!(
            estimate_request_tokens(&no_text),
            0,
            "non-text parts must not contribute to the local estimate"
        );
    }

    /// An empty `{}` message still consumes the per-message structural
    /// overhead, matching the reference `_count_messages` (which adds
    /// `tokens_per_message` unconditionally per array entry). The reply-priming
    /// `+3` from `_count_extra` is also present whenever `messages` is.
    #[test]
    fn estimate_request_tokens_empty_message_still_has_overhead() {
        let req = json!({"model": "gpt-4o", "messages": [{}]});
        assert_eq!(
            estimate_request_tokens(&req),
            PER_MESSAGE_OVERHEAD_TOKENS + REPLY_PRIMING_TOKENS,
            "an empty message must still contribute structural + reply-priming overhead"
        );
    }

    /// `tool_choice` framing must contribute tokens per the reference
    /// `_count_extra`: an object naming a function adds `+7` plus the
    /// function-name tokens.
    #[test]
    fn estimate_request_tokens_counts_tool_choice() {
        let base = json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}]
        });
        let mut with_choice = base.clone();
        with_choice["tool_choice"] =
            json!({"type": "function", "function": {"name": "get_weather"}});
        let base_n = estimate_request_tokens(&base);
        let with_n = estimate_request_tokens(&with_choice);
        let name_tokens = estimate_tokens("get_weather", "gpt-4o");
        assert_eq!(
            with_n - base_n,
            7 + name_tokens,
            "tool_choice object must add 7 + name tokens"
        );

        let mut with_none = base.clone();
        with_none["tool_choice"] = json!("none");
        let none_n = estimate_request_tokens(&with_none);
        assert_eq!(
            none_n - base_n,
            1,
            "tool_choice:\"none\" must add exactly 1 token"
        );
    }

    /// A top-level `prompt` string (the reference `/utils/token_counter` `text`
    /// argument) must be counted as raw text when `messages` is absent.
    #[test]
    fn estimate_request_tokens_counts_prompt_field() {
        let prompt = json!({
        "model": "gpt-4o",
        "prompt": "hello world this is a counting test"
        });
        let n = estimate_request_tokens(&prompt);
        assert!(n > 0, "prompt-only body must count tokens, got {n}");
        // The prompt must be counted as pure text (no per-message overhead):
        // it equals the direct text estimate for the same model.
        assert_eq!(
            n,
            estimate_tokens("hello world this is a counting test", "gpt-4o"),
            "prompt field is counted as raw text with no message overhead"
        );
    }

    /// A body carrying both `messages` and `prompt` must not double-count:
    /// the prompt is only read in the `messages`-absent branch.
    #[test]
    fn estimate_request_tokens_does_not_double_count_prompt_with_messages() {
        let msgs_only = json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}]
        });
        let mut with_prompt = msgs_only.clone();
        with_prompt["prompt"] = json!("hello world this is extra text");
        assert_eq!(
            estimate_request_tokens(&msgs_only),
            estimate_request_tokens(&with_prompt),
            "prompt must not be counted when messages is present"
        );
    }

    /// The reference `tokenizer_type` is `str(_select_tokenizer(model)["type"])` —
    /// `"openai_tokenizer"` / `"huggingface_tokenizer"`, never an encoding
    /// name. oxllm counts every model with tiktoken BPE, so the truthful and
    /// parity-correct value is `"openai_tokenizer"` regardless of which
    /// encoding (`o200k_base` / `cl100k_base`) `tokenizer_for` selects.
    #[test]
    fn tokenizer_type_matches_litellm_openai_tokenizer_string() {
        assert_eq!(tokenizer_type_for("gpt-4o"), "openai_tokenizer");
        assert_eq!(tokenizer_type_for("openai/gpt-4o"), "openai_tokenizer");
        assert_eq!(tokenizer_type_for("gpt-4"), "openai_tokenizer");
        assert_eq!(tokenizer_type_for("gpt-3.5-turbo"), "openai_tokenizer");
        assert_eq!(tokenizer_type_for("claude-3-opus"), "openai_tokenizer");
        assert_eq!(tokenizer_type_for("some-unknown-model"), "openai_tokenizer");
    }

    /// Warming must make both cached encoders usable (the ranks files parsed
    /// and stored in the `OnceLock`s) so the first estimate on an async
    /// executor never pays initialization.
    #[test]
    fn warm_tokenizers_initializes_both_encoders() {
        warm_tokenizers();
        assert!(estimate_tokens("hello world", "gpt-4o") > 0);
        assert!(estimate_tokens("hello world", "claude-3") > 0);
    }

    /// Token-count parity: a single short user message must score exactly
    /// `tokens(content) + per_message_overhead + reply_priming`. Asserting the
    /// precise composition catches the missing `+3` reply-priming and any future
    /// drift in the per-message constant.
    #[test]
    fn estimate_request_tokens_reply_priming_parity() {
        let req = json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}]
        });
        let content = estimate_tokens("hi", "gpt-4o");
        assert_eq!(
            estimate_request_tokens(&req),
            content + PER_MESSAGE_OVERHEAD_TOKENS + REPLY_PRIMING_TOKENS,
            "single message must include per-message overhead and the +3 reply priming"
        );

        // The priming applies once per request, not once per message: doubling
        // the messages adds 2×(content+overhead) but only the single base +3.
        let two = json!({
        "model": "gpt-4o",
        "messages": [
        {"role": "user", "content": "hi"},
        {"role": "user", "content": "hi"}
        ]
        });
        assert_eq!(
            estimate_request_tokens(&two),
            2 * content + 2 * PER_MESSAGE_OVERHEAD_TOKENS + REPLY_PRIMING_TOKENS,
            "reply priming is per-request, not per-message"
        );
    }

    /// The `+3` reply priming must NOT apply on the prompt-only / embeddings
    /// arms (the reference `text` path has no `_count_extra`).
    #[test]
    fn estimate_request_tokens_no_priming_on_prompt_only_arm() {
        let prompt = json!({"model": "gpt-4o", "prompt": "hi"});
        assert_eq!(
            estimate_request_tokens(&prompt),
            estimate_tokens("hi", "gpt-4o"),
            "prompt-only arm has no reply priming or per-message overhead"
        );
    }

    /// A `tool_result` / `thinking` content block must raise the estimate vs.
    /// the same body without it (previously these were silently dropped to 0).
    #[test]
    fn estimate_request_tokens_counts_tool_result_and_thinking_blocks() {
        let base = json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": [
        {"type": "text", "text": "describe"}
        ]}]
        });

        let mut with_blocks = base.clone();
        with_blocks["messages"][0]["content"] = json!([
        {"type": "text", "text": "describe"},
        {"type": "tool_result", "tool_use_id": "tu_1", "content": "the weather is sunny with a chance of rain"},
        {"type": "thinking", "thinking": "let me reason carefully about the weather report I just received"}
        ]);
        let without = estimate_request_tokens(&base);
        let with = estimate_request_tokens(&with_blocks);
        assert!(
            with > without,
            "tool_result + thinking blocks must raise the estimate: with={with}, without={without}"
        );
    }

    /// A `tool_use` block must count its `name` + `input`, and a bare-string
    /// content element must be counted as text (`isinstance(c, str)` semantics).
    #[test]
    fn estimate_request_tokens_counts_tool_use_and_bare_string_parts() {
        let bare_string = estimate_request_tokens(&json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": ["just a plain string part"]}]
        }));
        let equivalent = estimate_request_tokens(&json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "just a plain string part"}]
        }));
        assert_eq!(
            bare_string, equivalent,
            "bare-string array element must count the same as string content"
        );

        let with_tool_use = estimate_request_tokens(&json!({
        "model": "gpt-4o",
        "messages": [{"role": "assistant", "content": [
        {"type": "tool_use", "id": "tu_1", "name": "get_weather", "input": {"city": "paris", "unit": "c"}}
        ]}]
        }));
        let empty = estimate_request_tokens(&json!({
        "model": "gpt-4o",
        "messages": [{"role": "assistant", "content": []}]
        }));
        assert!(
            with_tool_use > empty,
            "tool_use block must contribute its name + input tokens"
        );
    }

    /// A `detail:"high"` image must contribute strictly more than the low/auto
    /// floor of 85 (the reference `calculate_img_tokens(mode="high")`).
    #[test]
    fn estimate_request_tokens_high_detail_image_above_floor() {
        let text_only = json!({
        "model": "gpt-4o",
        "messages": [{
        "role": "user",
        "content": [{"type": "text", "text": "describe"}]
        }]
        });
        let mut low = text_only.clone();
        low["messages"][0]["content"] = json!([
        {"type": "text", "text": "describe"},
        {"type": "image_url", "image_url": {"url": "http://x/y.png", "detail": "low"}}
        ]);
        let mut high = text_only.clone();
        high["messages"][0]["content"] = json!([
        {"type": "text", "text": "describe"},
        {"type": "image_url", "image_url": {"url": "http://x/y.png", "detail": "high"}}
        ]);
        let base = estimate_request_tokens(&text_only);
        let low_delta = estimate_request_tokens(&low) - base;
        let high_delta = estimate_request_tokens(&high) - base;
        assert_eq!(
            low_delta, PER_IMAGE_FLOOR_TOKENS,
            "low detail uses the 85 floor"
        );
        assert!(
            high_delta > PER_IMAGE_FLOOR_TOKENS,
            "high detail must exceed the 85 floor, got {high_delta}"
        );
        assert_eq!(
            high_delta, PER_IMAGE_HIGH_DETAIL_FLOOR_TOKENS,
            "high detail applies the 255 (1-tile) floor"
        );

        // Responses-API `input_image` form carries `detail` at the part top
        // level (image_url is a bare URL string there).
        let mut input_high = text_only.clone();
        input_high["messages"][0]["content"] = json!([
        {"type": "text", "text": "describe"},
        {"type": "input_image", "image_url": "http://x/y.png", "detail": "high"}
        ]);
        assert_eq!(
            estimate_request_tokens(&input_high) - base,
            PER_IMAGE_HIGH_DETAIL_FLOOR_TOKENS,
            "input_image top-level detail must also be honoured"
        );
    }

    /// Object-form `arguments` (non-conforming clients) must be counted by
    /// stringifying the value, not silently dropped to 0.
    #[test]
    fn estimate_request_tokens_counts_object_form_arguments() {
        let without = json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "go"}]
        });
        let with = json!({
        "model": "gpt-4o",
        "messages": [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": null, "tool_calls": [
        {"id": "call_1", "type": "function", "function": {
        "name": "search", "arguments": {"q": "a fairly long query string", "page": 42, "filters": ["a", "b", "c"]}
        }}
        ]}
        ]
        });
        let w = estimate_request_tokens(&with);
        let wo = estimate_request_tokens(&without);
        assert!(
            w > wo + 8,
            "object-form arguments must be counted beyond message overhead, delta={}",
            w - wo
        );
    }

    /// The `WARNED` set must be capped: novel names past `MAX_WARNED_MODELS` no
    /// longer grow it, while a previously-recorded name still reports "seen".
    #[test]
    fn mark_warned_caps_unbounded_growth() {
        let mut set = HashSet::new();
        for i in 0..MAX_WARNED_MODELS {
            assert!(
                mark_warned(&mut set, &format!("model-{i}")),
                "novel names below the cap must be recorded"
            );
        }
        assert_eq!(set.len(), MAX_WARNED_MODELS);
        // At the cap, a brand-new name is rejected (no growth, no warning).
        assert!(
            !mark_warned(&mut set, "brand-new-adversarial-name"),
            "cap must stop unbounded growth"
        );
        assert_eq!(
            set.len(),
            MAX_WARNED_MODELS,
            "rejected name must not be stored"
        );
        // A previously-recorded name still returns false (already seen).
        assert!(!mark_warned(&mut set, "model-0"));
    }

    /// `tool_choice: "auto"` and `"required"` must add exactly 0 tokens
    /// (the reference `_count_extra` only special-cases `"none"` and object form).
    #[test]
    fn estimate_request_tokens_tool_choice_auto_and_required_add_zero() {
        let base = json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}]
        });
        let base_n = estimate_request_tokens(&base);
        for val in ["auto", "required"] {
            let mut req = base.clone();
            req["tool_choice"] = json!(val);
            assert_eq!(
                estimate_request_tokens(&req),
                base_n,
                "tool_choice:{val:?} must add exactly 0 tokens"
            );
        }
    }

    /// A malformed content part (missing `type`, or `text` with a non-string
    /// value) must never panic. A non-string `text` is stringified and counted
    /// (`str(c.get("text", ""))` semantics); an unrecognised object type is
    /// serialized so it contributes a nonzero floor (never silently 0 on a
    /// billing path). Per-message framing is unaffected.
    #[test]
    fn estimate_request_tokens_malformed_content_part_is_safe() {
        let req = json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": [
        {"no_type_field": true},
        {"type": "text", "text": 12345}
        ]}]
        });
        let n = estimate_request_tokens(&req);
        let framing = PER_MESSAGE_OVERHEAD_TOKENS + REPLY_PRIMING_TOKENS;
        assert!(
            n > framing,
            "unknown part must contribute a serialized floor (>=1), got {n}; framing is {framing}"
        );
    }

    /// A `{"type":"text","text":<non-string>}` part must contribute ≥ 1 token:
    /// The reference counts `str(c.get("text", ""))`, so a numeric/boolean text
    /// value still scores its stringified form instead of silently 0.
    #[test]
    fn estimate_request_tokens_stringifies_non_string_text_part() {
        let base = json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": []}]
        });
        let with = json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": [
        {"type": "text", "text": 1234567}
        ]}]
        });
        let delta = estimate_request_tokens(&with) - estimate_request_tokens(&base);
        assert!(
            delta >= 1,
            "non-string text must contribute its stringified count, got delta {delta}"
        );
    }

    /// A body whose `messages` exists but is not an array must still count the
    /// fallback `prompt`/`input` arms: the messages loop contributes nothing,
    /// so gating on mere key-absence silently estimated 0 for e.g.
    /// `{"messages":"oops","prompt":"<lots of text>"}` on both the
    /// `/utils/token_counter` endpoint and the pre-call TPM/budget estimate.
    #[test]
    fn estimate_request_tokens_malformed_messages_still_counts_prompt() {
        let req = json!({
        "model": "gpt-4o",
        "messages": "not an array",
        "prompt": "hello world this is a counting test"
        });
        assert_eq!(
            estimate_request_tokens(&req),
            estimate_tokens("hello world this is a counting test", "gpt-4o"),
            "malformed messages must not suppress the prompt arm"
        );

        // Same for the embeddings `input` arm.
        let embed = json!({
        "model": "text-embedding-3-small",
        "messages": 42,
        "input": "hello"
        });
        assert!(
            estimate_request_tokens(&embed) > 0,
            "malformed messages must not suppress the input arm"
        );

        // A *usable* messages array still suppresses the prompt arm (no
        // double counting).
        let both = json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
        "prompt": "hello world this is a counting test"
        });
        let msgs_only = json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}]
        });
        assert_eq!(
            estimate_request_tokens(&both),
            estimate_request_tokens(&msgs_only),
            "prompt must not be counted when messages is a usable array"
        );
    }

    /// Embeddings `input` given as an array of token arrays (`[[1,2],[3,4]]`)
    /// contributes 0 locally: only string entries are counted, matching the
    /// pin that non-string entries are skipped rather than serialized.
    #[test]
    fn estimate_request_tokens_input_token_arrays_estimate_zero() {
        let req = json!({
        "model": "text-embedding-3-small",
        "input": [[1, 2], [3, 4]]
        });
        assert_eq!(
            estimate_request_tokens(&req),
            0,
            "non-string input entries must be skipped, not serialized"
        );
    }

    /// An invalid `detail` value on an image part is treated leniently: the
    /// 85-token low/auto floor applies (the reference raises on unknown detail;
    /// oxllm never rejects during estimation).
    #[test]
    fn estimate_request_tokens_invalid_detail_uses_low_floor() {
        let base = json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "describe"}]}]
        });
        let mut with = base.clone();
        with["messages"][0]["content"] = json!([
        {"type": "text", "text": "describe"},
        {"type": "image_url", "image_url": {"url": "http://x/y.png", "detail": "ultra"}}
        ]);
        assert_eq!(
            estimate_request_tokens(&with) - estimate_request_tokens(&base),
            PER_IMAGE_FLOOR_TOKENS,
            "invalid detail must fall back to the 85 floor, not reject or over-reserve"
        );
    }
}
