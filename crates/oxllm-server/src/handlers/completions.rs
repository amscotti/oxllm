//! `POST /v1/completions` — legacy text completion endpoint.
//!
//! Most providers have deprecated or merged this into chat completions. The
//! handler is a thin forwarder: it wraps the request into a chat-shaped body
//! (a user message per `prompt` element, matching the reference
//! `text_completion` translation), routes through the normal chat pipeline,
//! then unwraps the response. Streaming responses are translated
//! chunk-by-chunk from `chat.completion.chunk` to `text_completion` so legacy
//! clients receive the shape they expect.

use axum::Extension;
use axum::body::Bytes;
use axum::extract::State;
use axum::response::sse::Event;
use axum::response::{IntoResponse, Response};
use serde_json::Value;
use tokio::sync::mpsc;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;

use crate::auth::AuthUser;
use crate::error::ApiError;
use crate::handlers::chat::{MAX_STREAM_BUFFER_BYTES, chat_completions_inner};
use crate::orchestration::stream::send_converted_event;
use crate::pipeline::preserved_response_headers;
use crate::sse::{SseFramer, extract_sse_data, pop_sse_event};
use crate::state::SharedState;

/// Legacy text completions handler.
///
/// Wraps the legacy `prompt` into a chat message and forwards to the chat
/// pipeline. Most upstream providers either deprecated `/v1/completions` or
/// accept the same model list, so a passthrough is sufficient for this endpoint.
pub async fn completions(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    headers: axum::http::HeaderMap,
    body: Bytes,
) -> Result<Response, ApiError> {
    let request_body: Value = serde_json::from_slice(&body)
        .map_err(|e| ApiError::bad_request(format!("invalid JSON body: {e}")))?;

    crate::security::check_json_depth(&request_body, crate::security::MAX_JSON_DEPTH)?;
    crate::orchestration::request_controls::validate(&request_body)?;
    let state = shared.load_full();

    let model = request_body
        .get("model")
        .and_then(Value::as_str)
        .filter(|model| !model.trim().is_empty())
        .ok_or_else(|| ApiError::bad_request("missing required field: 'model'"))?;
    crate::pipeline::check_request_model_acl(&state, &auth, model, &request_body, &headers).await?;

    validate_legacy_prompt(&request_body)?;

    let chat_body = wrap_prompt_as_chat(&request_body);

    // Read `stream` from the wrapped body so the converter path agrees with
    // what `chat_completions_inner` passes to the orchestrator.
    let is_stream = chat_body
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    // Serialize→re-parse is intentional: `chat_completions_inner` is the
    // canonical entrypoint shared with `/v1/messages` and `/v1/responses`.
    let bytes = serde_json::to_vec(&chat_body)
        .map_err(|e| ApiError::internal(format!("failed to serialize chat body: {e}")))?;

    let chat_response = chat_completions_inner(state, &auth, &headers, Bytes::from(bytes)).await?;
    if !chat_response.status().is_success() {
        return Ok(chat_response);
    }

    if is_stream {
        return convert_stream_to_completion(chat_response).await;
    }

    convert_unary_to_completion(chat_response).await
}

/// Translate the chat pipeline's non-streaming JSON response into the legacy
/// `text_completion` shape, preserving the `x-oxllm-*`, `x-ratelimit-*`, and
/// standard `x-request-id` headers stamped by the orchestrator. The streaming
/// branch (`convert_stream_to_completion`) preserves the same header families
/// via [`crate::pipeline::preserved_response_headers`]; this keeps the two
/// paths consistent so a legacy client sees cost & routing transparency
/// regardless of `stream`.
///
/// If the chat body cannot be parsed (over-cap body, malformed JSON, future
/// bug in [`chat_to_legacy_completion`]), the *original* chat body is surfaced
/// verbatim with an `x-oxllm-shape: conversion-failed` marker so the caller
/// still receives the bytes they paid for. By the time this runs,
/// `pipeline::post_success` has already debited budget / written a spend log,
/// so converting to a generic 500 would silently lose both the data and any
/// chance for the client to retry idempotently.
async fn convert_unary_to_completion(chat_response: Response) -> Result<Response, ApiError> {
    let preserved_headers = preserved_response_headers(chat_response.headers());
    let body = axum::body::to_bytes(
        chat_response.into_body(),
        crate::security::MAX_RESPONSE_BODY_BYTES,
    )
    .await
    .map_err(|e| ApiError::internal(format!("failed to read chat response: {e}")))?;
    let parsed = serde_json::from_slice::<Value>(&body).ok();
    match parsed {
        Some(chat) => {
            let legacy = chat_to_legacy_completion(&chat);
            let mut response = axum::Json(legacy).into_response();
            for (name, value) in preserved_headers {
                response.headers_mut().insert(name, value);
            }
            Ok(response)
        }
        None => {
            tracing::warn!(
                bytes = body.len(),
                "legacy text_completion conversion failed; surfacing original chat body"
            );
            let mut response = Response::new(axum::body::Body::from(body));
            response.headers_mut().insert(
                axum::http::header::CONTENT_TYPE,
                axum::http::HeaderValue::from_static("application/json"),
            );
            response.headers_mut().insert(
                "x-oxllm-shape",
                axum::http::HeaderValue::from_static("conversion-failed"),
            );
            for (name, value) in preserved_headers {
                response.headers_mut().insert(name, value);
            }
            Ok(response)
        }
    }
}

/// Translate the chat pipeline's streaming SSE response into the legacy
/// `text_completion` chunk shape, preserving the `x-oxllm-*`, `x-ratelimit-*`,
/// and standard `x-request-id` headers stamped by the orchestrator.
///
/// Each `data: {...}\n\n` event is parsed, run through
/// [`convert_chat_chunk_to_completion`], and re-serialized. The `[DONE]`
/// sentinel and upstream error chunks pass through unchanged so the stream
/// safety invariant (no fallback after the first byte) is preserved.
async fn convert_stream_to_completion(chat_response: Response) -> Result<Response, ApiError> {
    let preserved_headers = preserved_response_headers(chat_response.headers());

    let body = chat_response.into_body();
    let data_stream = body.into_data_stream();

    let (tx, rx) = mpsc::channel::<Result<Event, std::convert::Infallible>>(64);

    tokio::spawn(async move {
        let mut framer = SseFramer::new();
        let mut buffer = String::new();
        let mut stream = data_stream;
        // Sticky flag for the >cap partial-event recovery: set when an overflow
        // leaves a partial event in `buffer`, cleared on any successful
        // `pop_sse_event`. Lets a >cap event split across two reads complete
        // on the second read while still bounding growth for a misbehaving
        // upstream that never emits a separator.
        let mut overflow_pending = false;

        loop {
            tokio::select! {
            _ = tx.closed() => return,
            result = stream.next() => {
            match result {
            Some(Ok(bytes)) => {
            buffer.push_str(&framer.push(&bytes));
            while let Some(event_str) = pop_sse_event(&mut buffer) {
            overflow_pending = false;
            if let Some(transformed) = transform_completion_sse_event(&event_str) {
            let done = transformed == "[DONE]";
            let event = Event::default().data(transformed);
            if send_converted_event(&tx, event).await.is_err() {
            return;
            }
            if done {
            return;
            }
            }
            }
            if buffer.len() > MAX_STREAM_BUFFER_BYTES {
            // The drain loop above already exhausted every complete event,
            // so `buffer` now holds only a partial frame. Preserve it
            // across this overflow so a >cap event split across two reads
            // can complete on the next chunk (previously the partial was
            // force-parsed — which always fails JSON deserialization on a
            // truncated body — and then `buffer.clear()`-ed, losing the
            // event at the boundary). To avoid unbounded growth on a
            // genuinely misbehaving upstream that never emits a separator,
            // only preserve once: if the next read also overflows without
            // yielding a complete event, drop the accumulated partial.
            if overflow_pending {
            tracing::warn!(
            size = buffer.len(),
            "stream buffer overflowed again after preserving a partial event; dropping it"
            );
            buffer.clear();
            overflow_pending = false;
            } else {
            tracing::warn!(
            size = buffer.len(),
            "stream buffer exceeded cap without a complete SSE event; preserving partial for next read"
            );
            overflow_pending = !buffer.is_empty();
            }
            }
            }
            Some(Err(e)) => {
            tracing::warn!(error = %e, "stream read error");
            // M5: forward a terminal error event before returning so the
            // client's SSE stream is closed explicitly rather than truncated
            // silently. Without this the converter stream just ends and the
            // legacy completions client cannot distinguish a clean completion
            // from an upstream failure.
            if send_converted_event(&tx, Event::default().data(
                r#"{"error":{"message":"stream read error","type":"api_error"}}"#,
            )).await.is_err() {
            return;
            }
            // Mirror `orchestration/stream.rs`: a terminal error chunk is
            // always followed by the `[DONE]` sentinel so an OpenAI-shaped
            // SSE client that waits on `[DONE]` (instead of treating stream
            // close as completion) observes the failure cleanly.
            let _ = send_converted_event(&tx, Event::default().data("[DONE]")).await;
            return;
            }
            None => {
            match framer.finish() {
            Ok(remaining) => buffer.push_str(&remaining),
            Err(e) => {
            tracing::warn!(error = %e, "stream ended with undecodable trailing bytes; dropping them");
            }
            }
            if !buffer.trim().is_empty() {
            if let Some(transformed) = transform_completion_sse_event(&buffer) {
            let event = Event::default().data(transformed);
            let _ = send_converted_event(&tx, event).await;
            }
            }
            break;
            }
            }
            }
            }
        }
    });

    let mut response = axum::response::Sse::new(ReceiverStream::new(rx)).into_response();
    for (name, value) in preserved_headers {
        response.headers_mut().insert(name, value);
    }
    Ok(response)
}

/// Parse one chat SSE event and return the serialized `text_completion`
/// payload to forward. Returns `None` for events without a `data:` line or
/// with unparseable JSON (silently dropped, matching the sibling handlers).
/// The `[DONE]` sentinel is returned verbatim.
fn transform_completion_sse_event(raw: &str) -> Option<String> {
    let data_str = extract_sse_data(raw)?;
    if data_str == "[DONE]" {
        return Some("[DONE]".to_string());
    }
    let chunk: Value = serde_json::from_str(data_str).ok()?;
    let converted = convert_chat_chunk_to_completion(&chunk);
    Some(converted.to_string())
}

/// Validate the legacy `prompt` field shape before it is wrapped into a chat
/// message. A missing `prompt` is intentionally allowed (yields an empty user
/// message, pinned by `wrap_missing_prompt_yields_empty_message`). A string or
/// an array of strings is the supported shape. Token-ID arrays (integers) or
/// arrays of token-ID arrays are part of the OpenAI `/v1/completions` spec but
/// have no chat equivalent and would otherwise collapse to an empty message
/// (every element fails `as_str`), silently dispatching a no-op upstream call.
/// Reject those shapes upfront with a `400` instead.
fn validate_legacy_prompt(request_body: &Value) -> Result<(), ApiError> {
    match request_body.get("prompt") {
        None | Some(Value::String(_)) => Ok(()),
        Some(Value::Array(arr)) => {
            if arr.iter().all(|v| v.is_string()) {
                Ok(())
            } else {
                Err(ApiError::bad_request(
                    "field 'prompt' array must contain only strings; token-ID prompts are not supported",
                ))
            }
        }
        Some(_) => Err(ApiError::bad_request(
            "field 'prompt' must be a string or an array of strings",
        )),
    }
}

fn wrap_prompt_as_chat(request: &Value) -> Value {
    let mut out = serde_json::Map::new();
    if let Some(model) = request.get("model") {
        out.insert("model".to_string(), model.clone());
    }
    if let Some(stream) = request.get("stream") {
        out.insert("stream".to_string(), stream.clone());
    }
    if let Some(temperature) = request.get("temperature") {
        out.insert("temperature".to_string(), temperature.clone());
    }
    if let Some(max_tokens) = request.get("max_tokens") {
        out.insert("max_tokens".to_string(), max_tokens.clone());
    }
    if let Some(top_p) = request.get("top_p") {
        out.insert("top_p".to_string(), top_p.clone());
    }

    // Forward every remaining chat-compatible key. The legacy text-completion
    // API has a small set of params with no chat equivalent; denylist those so
    // a future OpenAI param (e.g. `response_format`,
    // `max_completion_tokens`, `stream_options`) reaches the upstream without
    // needing another edit here. `prompt` is consumed below to build the
    // wrapped user message; the rest (`echo`, `best_of`, `suffix`) have no
    // chat representation and would 400 from the upstream if forwarded.
    // `logprobs` is also denied here because the legacy int (0-5) is
    // translated to the chat bool + `top_logprobs` pair below — forwarding it
    // verbatim produces a type mismatch the chat API rejects.
    const TEXT_COMPLETION_ONLY: &[&str] = &["prompt", "echo", "best_of", "suffix", "logprobs"];
    if let Some(req_obj) = request.as_object() {
        for (key, value) in req_obj {
            if key == "model"
                || key == "stream"
                || key == "temperature"
                || key == "max_tokens"
                || key == "top_p"
            {
                continue;
            }
            if TEXT_COMPLETION_ONLY.contains(&key.as_str()) {
                tracing::debug!(
                    param = %key,
                    "dropping legacy text-completion param with no chat equivalent"
                );
                continue;
            }
            out.insert(key.clone(), value.clone());
        }
    }

    let messages: Vec<Value> = match request.get("prompt") {
        Some(Value::String(s)) => vec![serde_json::json!({"role": "user", "content": s})],
        Some(Value::Array(arr)) if !arr.is_empty() => arr
            .iter()
            .filter_map(|v| {
                v.as_str()
                    .map(|s| serde_json::json!({"role": "user", "content": s}))
            })
            .collect(),
        _ => vec![serde_json::json!({"role": "user", "content": ""})],
    };
    out.insert("messages".to_string(), Value::Array(messages));

    // Legacy text-completions `logprobs` is an integer count (0-5); the chat
    // API types `logprobs` as a boolean and uses a separate integer
    // `top_logprobs` (0-20). Forwarding the int verbatim (previously the
    // behaviour) is a type mismatch the chat upstream rejects with a 400, so
    // the legacy logprobs feature was broken through this endpoint. The reference
    // keeps the int because it routes text-completion models to the native
    // `/v1/completions` endpoint (`main.py`); oxllm always wraps as
    // chat, so the translation is required for parity. `0` (or a non-int
    // value) maps to the chat default of "no logprobs" — omit both fields
    // rather than emitting a contradictory `logprobs: false` +
    // `top_logprobs: N` pair (the client may have set `top_logprobs`
    // directly, which we scrub here so the upstream sees a clean "off").
    if let Some(logprobs) = request.get("logprobs").and_then(Value::as_i64) {
        if logprobs > 0 {
            out.insert("logprobs".to_string(), Value::Bool(true));
            out.insert("top_logprobs".to_string(), Value::from(logprobs));
        } else {
            out.remove("logprobs");
            out.remove("top_logprobs");
        }
    }

    Value::Object(out)
}

/// Flatten a chat `message.content` value into the legacy `text` string.
///
/// Mirrors `oxllm_core::router::complexity::extract_request_text` so the
/// legacy endpoint agrees with the rest of the gateway on how multipart
/// content becomes plain text:
/// - `string` → verbatim.
/// - `array`  → the `text` parts of every `{"type":"text","text":...}` element
///   concatenated with no separator (preserving chunk-level ordering).
/// - `null` / missing → empty string.
/// - any other scalar (number/bool) → `serde_json::to_string` so the bytes
///   survive rather than being silently emptied; a `tracing::warn!` flags the
///   unexpected shape so an operator can trace the drop.
fn flatten_chat_content(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|part| {
                if part.get("type").and_then(|t| t.as_str()) == Some("text") {
                    part.get("text").and_then(|t| t.as_str()).map(String::from)
                } else {
                    None
                }
            })
            .collect(),
        Value::Null => String::new(),
        other => {
            tracing::warn!(
                shape = %other,
                "non-string chat message.content flattened to JSON literal for legacy text_completion"
            );
            other.to_string()
        }
    }
}

fn chat_to_legacy_completion(chat: &Value) -> Value {
    let mut out = chat.clone();
    let Some(object) = out.as_object_mut() else {
        return out;
    };
    object.insert(
        "object".to_string(),
        Value::String("text_completion".to_string()),
    );
    if let Some(choices) = object.get_mut("choices").and_then(Value::as_array_mut) {
        for choice in choices {
            let Some(choice) = choice.as_object_mut() else {
                continue;
            };
            let text = choice
                .get("message")
                .and_then(|m| m.get("content"))
                .map(flatten_chat_content)
                .unwrap_or_default();
            choice.remove("message");
            choice.insert("text".to_string(), Value::String(text));
        }
    }
    out
}

/// Convert a single chat-completion streaming chunk into the legacy
/// `text_completion` shape.
///
/// - `object` is rewritten to `"text_completion"`.
/// - `choices[].delta.content` is hoisted to `choices[].text` and `delta`
///   is removed.
/// - `logprobs` is defaulted to `null` on each choice (legacy clients
///   expect the field to be present even when logprobs were not requested).
/// - Chunks that carry a top-level `error` object are passed through
///   unchanged so the stream pump's error shape reaches the client intact.
fn convert_chat_chunk_to_completion(chunk: &Value) -> Value {
    let mut result = chunk.clone();
    let Some(obj) = result.as_object_mut() else {
        return result;
    };
    if obj.contains_key("error") {
        return result;
    }
    obj.insert(
        "object".to_string(),
        Value::String("text_completion".to_string()),
    );
    if let Some(choices) = obj.get_mut("choices").and_then(|c| c.as_array_mut()) {
        for choice in choices {
            let Some(choice_obj) = choice.as_object_mut() else {
                continue;
            };
            if let Some(delta) = choice_obj.remove("delta") {
                if let Some(content) = delta.get("content") {
                    choice_obj.insert("text".to_string(), content.clone());
                }
                // The reference `convert_to_text_completion_object` carries
                // `delta.reasoning_content` onto `text_choices.reasoning_content`
                // so DeepSeek-R1 / o1-style traces survive the legacy endpoint.
                if let Some(rc) = delta.get("reasoning_content") {
                    choice_obj.insert("reasoning_content".to_string(), rc.clone());
                }
            }
            choice_obj
                .entry("logprobs".to_string())
                .or_insert(Value::Null);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn completion_converter_ignores_error_and_duplicate_done_tail() {
        let source = futures_util::stream::iter([
            Ok::<Bytes, std::io::Error>(Bytes::from_static(b"data: [DONE]\n\ndata: [DONE]\n\n")),
            Err(std::io::Error::other("late socket reset")),
        ]);
        let response =
            convert_stream_to_completion(Response::new(axum::body::Body::from_stream(source)))
                .await
                .unwrap();
        let bytes = axum::body::to_bytes(response.into_body(), 1024)
            .await
            .unwrap();
        let text = String::from_utf8_lossy(&bytes);
        assert_eq!(text.matches("[DONE]").count(), 1);
        assert!(!text.contains("error"), "{text}");
    }

    #[test]
    fn wrap_string_prompt_into_user_message() {
        let req = json!({
        "model": "gpt-3.5-turbo-instruct",
        "prompt": "Once upon a time",
        "max_tokens": 5,
        });
        let wrapped = wrap_prompt_as_chat(&req);
        assert_eq!(wrapped["model"], "gpt-3.5-turbo-instruct");
        assert_eq!(wrapped["messages"][0]["role"], "user");
        assert_eq!(wrapped["messages"][0]["content"], "Once upon a time");
        assert_eq!(wrapped["max_tokens"], 5);
    }

    #[test]
    fn wrap_forwards_additional_params() {
        let req = json!({
            "model": "m",
            "prompt": "hi",
            "stop": ["\\n"],
            "n": 2,
            "logprobs": 1,
            "logit_bias": {"50256": -100},
            "seed": 42,
            "presence_penalty": 0.5,
            "frequency_penalty": 0.3,
            "user": "u-1",
            "cache": {"insert": true}
        });
        let wrapped = wrap_prompt_as_chat(&req);
        assert_eq!(wrapped["stop"], json!(["\\n"]));
        assert_eq!(wrapped["n"], 2);
        // Legacy `logprobs: 1` (int) is translated to the chat pair
        // `logprobs: true` + `top_logprobs: 1` — forwarding the int verbatim
        // is a type mismatch the chat API rejects.
        assert_eq!(wrapped["logprobs"], true);
        assert_eq!(wrapped["top_logprobs"], 1);
        assert_eq!(wrapped["logit_bias"], json!({"50256": -100}));
        assert_eq!(wrapped["seed"], 42);
        assert_eq!(wrapped["presence_penalty"], 0.5);
        assert_eq!(wrapped["frequency_penalty"], 0.3);
        assert_eq!(wrapped["user"], "u-1");
        assert_eq!(wrapped["cache"]["insert"], true);
    }

    #[test]
    fn chat_to_legacy_completion_passes_non_object_through_unchanged() {
        let non_object = json!([1, 2, 3]);
        let out = chat_to_legacy_completion(&non_object);
        assert_eq!(out, non_object);
    }

    #[test]
    fn chat_to_legacy_completion_handles_array_response_gracefully() {
        let array_resp = json!([{"a": 1}]);
        let out = chat_to_legacy_completion(&array_resp);
        assert_eq!(out, array_resp);
    }

    /// A string-array `prompt` maps to one user message per element, matching
    /// The reference `text_completion` translation (`main.py`: "for p in
    /// prompt: messages.append({'role':'user','content':p})"). Previously the
    /// wrapper joined every element with `\n` into a single user message,
    /// diverging from the reference and yielding different model context.
    #[test]
    fn wrap_array_prompt_into_per_element_user_messages() {
        let req = json!({
        "model": "m",
        "prompt": ["Line 1", "Line 2"],
        });
        let wrapped = wrap_prompt_as_chat(&req);
        let messages = wrapped["messages"].as_array().expect("messages array");
        assert_eq!(
            messages.len(),
            2,
            "one user message per prompt element: {wrapped}"
        );
        for (msg, expected) in messages.iter().zip(["Line 1", "Line 2"]) {
            assert_eq!(msg["role"], "user");
            assert_eq!(msg["content"], expected);
        }
    }

    /// A three-element string-array `prompt` produces three user messages,
    /// preserving order and content verbatim (no joining, no separators).
    #[test]
    fn wrap_array_prompt_three_elements_into_three_user_messages() {
        let req = json!({"model": "m", "prompt": ["a", "b", "c"]});
        let wrapped = wrap_prompt_as_chat(&req);
        let messages = wrapped["messages"].as_array().expect("messages array");
        assert_eq!(
            messages.len(),
            3,
            "three prompt elements -> three messages: {wrapped}"
        );
        assert_eq!(messages[0]["content"], "a");
        assert_eq!(messages[1]["content"], "b");
        assert_eq!(messages[2]["content"], "c");
        assert!(
            messages.iter().all(|m| m["role"] == "user"),
            "every element must be a user message: {wrapped}"
        );
    }

    #[test]
    fn wrap_missing_prompt_yields_empty_message() {
        let req = json!({"model": "m"});
        let wrapped = wrap_prompt_as_chat(&req);
        assert_eq!(wrapped["messages"][0]["content"], "");
    }

    #[test]
    fn convert_content_chunk_moves_delta_content_to_text() {
        let chunk = json!({
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4o",
        "choices": [{
        "index": 0,
        "delta": {"content": "hello"},
        "finish_reason": null
        }]
        });
        let converted = convert_chat_chunk_to_completion(&chunk);
        assert_eq!(converted["object"], "text_completion");
        assert_eq!(converted["choices"][0]["text"], "hello");
        assert!(
            converted["choices"][0].get("delta").is_none(),
            "delta must be removed: {converted}"
        );
        assert!(converted["choices"][0]["logprobs"].is_null());
        assert_eq!(
            converted["choices"][0]["finish_reason"],
            serde_json::Value::Null
        );
        assert_eq!(converted["id"], "chatcmpl-1");
        assert_eq!(converted["model"], "gpt-4o");
    }

    #[test]
    fn convert_first_chunk_role_only_delta_yields_empty_text() {
        let chunk = json!({
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "choices": [{
        "index": 0,
        "delta": {"role": "assistant"},
        "finish_reason": null
        }]
        });
        let converted = convert_chat_chunk_to_completion(&chunk);
        assert_eq!(converted["object"], "text_completion");
        assert!(
            converted["choices"][0].get("delta").is_none(),
            "role-only delta must still be removed: {converted}"
        );
        assert!(
            converted["choices"][0].get("text").is_none(),
            "no content in delta means no text field: {converted}"
        );
        assert!(converted["choices"][0]["logprobs"].is_null());
    }

    #[test]
    fn convert_finish_chunk_preserves_finish_reason() {
        let chunk = json!({
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "choices": [{
        "index": 0,
        "delta": {},
        "finish_reason": "stop"
        }]
        });
        let converted = convert_chat_chunk_to_completion(&chunk);
        assert_eq!(converted["choices"][0]["finish_reason"], "stop");
        assert!(
            converted["choices"][0].get("delta").is_none(),
            "empty delta must be removed: {converted}"
        );
        assert!(converted["choices"][0]["logprobs"].is_null());
    }

    #[test]
    fn convert_finish_chunk_with_length_reason_preserved() {
        let chunk = json!({
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]
        });
        let converted = convert_chat_chunk_to_completion(&chunk);
        assert_eq!(converted["choices"][0]["finish_reason"], "length");
    }

    #[test]
    fn convert_usage_chunk_preserves_usage() {
        let chunk = json!({
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "choices": [],
        "usage": {"prompt_tokens": 7, "completion_tokens": 4, "total_tokens": 11}
        });
        let converted = convert_chat_chunk_to_completion(&chunk);
        assert_eq!(converted["object"], "text_completion");
        assert_eq!(converted["usage"]["prompt_tokens"], 7);
        assert_eq!(converted["usage"]["completion_tokens"], 4);
        assert_eq!(converted["usage"]["total_tokens"], 11);
    }

    #[test]
    fn convert_error_chunk_passes_through_unchanged() {
        let chunk = json!({
        "error": {
        "message": "capacity exhausted",
        "type": "api_error",
        "code": 429
        }
        });
        let converted = convert_chat_chunk_to_completion(&chunk);
        assert!(
            converted.get("object").is_none(),
            "error chunks must not be rewritten: {converted}"
        );
        assert_eq!(converted["error"]["message"], "capacity exhausted");
        assert_eq!(converted["error"]["code"], 429);
    }

    #[test]
    fn transform_event_passes_done_sentinel_through() {
        let out = transform_completion_sse_event("data: [DONE]\n\n");
        assert_eq!(out.as_deref(), Some("[DONE]"));
    }

    #[test]
    fn transform_event_drops_non_data_lines() {
        assert!(transform_completion_sse_event(": keepalive\n\n").is_none());
        assert!(transform_completion_sse_event("event: ping\n\n").is_none());
    }

    #[test]
    fn transform_event_drops_malformed_json() {
        let out = transform_completion_sse_event("data: {not json}\n\n");
        assert!(out.is_none(), "malformed JSON must be dropped, got {out:?}");
    }

    #[test]
    fn transform_event_rewrites_chat_chunk_to_text_completion() {
        let raw = "data: {\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\n";
        let out = transform_completion_sse_event(raw).expect("converted");
        assert!(out.contains("\"object\":\"text_completion\""), "out: {out}");
        assert!(out.contains("\"text\":\"hi\""), "out: {out}");
        assert!(!out.contains("delta"), "delta must be gone: {out}");
    }

    #[test]
    fn transform_event_handles_data_without_space() {
        let raw = "data:{\"object\":\"chat.completion.chunk\",\"choices\":[]}";
        let out = transform_completion_sse_event(raw).expect("converted");
        assert!(out.contains("\"object\":\"text_completion\""));
    }

    #[test]
    fn transform_event_preserves_error_payload_shape() {
        let raw =
            "data: {\"error\":{\"message\":\"boom\",\"type\":\"api_error\",\"code\":500}}\n\n";
        let out = transform_completion_sse_event(raw).expect("converted");
        assert!(
            !out.contains("text_completion"),
            "error chunks must not be rewritten: {out}"
        );
        assert!(out.contains("\"message\":\"boom\""));
    }

    #[test]
    fn extract_sse_data_handles_space_and_no_space() {
        assert_eq!(extract_sse_data("data: {\"a\":1}"), Some("{\"a\":1}"));
        assert_eq!(extract_sse_data("data:{\"a\":1}"), Some("{\"a\":1}"));
        assert_eq!(extract_sse_data("data: [DONE]"), Some("[DONE]"));
        assert_eq!(extract_sse_data("not data"), None);
    }

    // ---- Reference text-completion regression cases -----------------------
    //
    // The reference text-completion response conversion test validates the
    // full text-completion response shape (id, choices[].text/finish_reason/
    // logprobs, created, model, object, usage incl. token details). oxllm's
    // analogue is `chat_to_legacy_completion`, which rewrites a chat response
    // into the legacy text-completion shape. The port feeds the chat-shaped
    // equivalent of that fixture through the converter and asserts the same
    // fields survive. (The HuggingFace logprobs E2E test and the OpenAI
    // httpx-client wiring test have no oxllm equivalent and are omitted.)

    /// Ported from `test_convert_dict_to_text_completion_response`: a full
    /// chat response converts to the text-completion shape, preserving id,
    /// created, model, usage (with token detail sub-objects), and the choice
    /// logprobs, while hoisting `message.content` to `text`.
    #[test]
    fn chat_to_legacy_completion_maps_full_response_with_logprobs_and_usage() {
        let chat = json!({
        "id": "cmpl-ALVLPJgRkqpTomotoOMi3j0cAaL4L",
        "object": "chat.completion",
        "created": 1729688739,
        "model": "davinci-002",
        "system_fingerprint": null,
        "choices": [{
        "index": 0,
        "message": {"role": "assistant", "content": "hello crisp"},
        "finish_reason": "length",
        "logprobs": {
        "text_offset": [0, 5],
        "token_logprobs": [null, -12.203847],
        "tokens": ["hello", " crisp"],
        "top_logprobs": [null, {",": -2.1568563}],
        },
        }],
        "usage": {
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "total_tokens": 2,
        "prompt_tokens_details": null,
        "completion_tokens_details": null,
        },
        });
        let out = chat_to_legacy_completion(&chat);

        // Top-level passthrough.
        assert_eq!(out["id"], "cmpl-ALVLPJgRkqpTomotoOMi3j0cAaL4L");
        assert_eq!(out["model"], "davinci-002");
        assert_eq!(out["created"], 1729688739);
        assert_eq!(out["object"], "text_completion");
        assert_eq!(out["system_fingerprint"], serde_json::Value::Null);

        // Choice: message → text, finish_reason preserved.
        let choice = &out["choices"][0];
        assert_eq!(choice["index"], 0);
        assert_eq!(choice["finish_reason"], "length");
        assert_eq!(choice["text"], "hello crisp");
        assert!(
            choice.get("message").is_none(),
            "message must be removed after hoisting to text"
        );

        // logprobs survive verbatim (the converter does not touch them).
        assert_eq!(choice["logprobs"]["text_offset"], json!([0, 5]));
        assert_eq!(
            choice["logprobs"]["token_logprobs"],
            json!([null, -12.203847])
        );
        assert_eq!(choice["logprobs"]["tokens"], json!(["hello", " crisp"]));
        assert_eq!(
            choice["logprobs"]["top_logprobs"],
            json!([null, {",": -2.1568563}])
        );

        // Usage incl. token-detail sub-objects is preserved.
        assert_eq!(out["usage"]["prompt_tokens"], 1);
        assert_eq!(out["usage"]["completion_tokens"], 1);
        assert_eq!(out["usage"]["total_tokens"], 2);
        assert_eq!(
            out["usage"]["prompt_tokens_details"],
            serde_json::Value::Null
        );
        assert_eq!(
            out["usage"]["completion_tokens_details"],
            serde_json::Value::Null
        );
    }

    /// Ported from `test_convert_dict_to_text_completion_response` (usage
    /// detail variant): populated `prompt_tokens_details` /
    /// `completion_tokens_details` objects pass through unchanged.
    #[test]
    fn chat_to_legacy_completion_preserves_usage_token_details() {
        let chat = json!({
        "object": "chat.completion",
        "choices": [{
        "index": 0,
        "message": {"role": "assistant", "content": "hi"},
        "finish_reason": "stop",
        }],
        "usage": {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
        "prompt_tokens_details": {"cached_tokens": 12},
        "completion_tokens_details": {"reasoning_tokens": 7},
        },
        });
        let out = chat_to_legacy_completion(&chat);
        assert_eq!(out["usage"]["prompt_tokens"], 100);
        assert_eq!(out["usage"]["completion_tokens"], 50);
        assert_eq!(out["usage"]["total_tokens"], 150);
        assert_eq!(out["usage"]["prompt_tokens_details"]["cached_tokens"], 12);
        assert_eq!(
            out["usage"]["completion_tokens_details"]["reasoning_tokens"],
            7
        );
    }

    /// A chat choice carrying a populated `logprobs` object keeps it after
    /// conversion (the converter only rewrites `message` → `text`).
    #[test]
    fn chat_to_legacy_completion_preserves_logprobs_object_on_choice() {
        let chat = json!({
        "object": "chat.completion",
        "choices": [{
        "index": 0,
        "message": {"role": "assistant", "content": "hello"},
        "finish_reason": "stop",
        "logprobs": {
        "content": [{"token": "hello", "logprob": -0.1, "bytes": [104]}],
        },
        }],
        });
        let out = chat_to_legacy_completion(&chat);
        let logprobs = &out["choices"][0]["logprobs"];
        assert_eq!(logprobs["content"][0]["token"], "hello");
        assert_eq!(logprobs["content"][0]["logprob"], -0.1);
    }

    /// A legacy `logprobs: N` (int) request param is translated into the chat
    /// pair `logprobs: true` + `top_logprobs: N` so the upstream chat API
    /// accepts it (forwarding the int verbatim is a type mismatch the chat
    /// API rejects with a 400). Mirrors the reference text-completion code path,
    /// which keeps the int only because it routes those models to the native
    /// `/v1/completions` endpoint.
    #[test]
    fn wrap_prompt_forwards_logprobs_request_param() {
        let req = json!({
        "model": "gpt-3.5-turbo-instruct",
        "prompt": "Once upon a time",
        "logprobs": 1,
        });
        let wrapped = wrap_prompt_as_chat(&req);
        assert_eq!(wrapped["logprobs"], true);
        assert_eq!(wrapped["top_logprobs"], 1);
        assert!(
            wrapped.get("logprobs").and_then(|v| v.as_i64()).is_none(),
            "legacy int logprobs must NOT be forwarded verbatim: {wrapped}"
        );
        assert_eq!(wrapped["messages"][0]["content"], "Once upon a time");
    }

    /// A larger legacy `logprobs: 5` translates to `logprobs: true` +
    /// `top_logprobs: 5`.
    #[test]
    fn wrap_prompt_translates_logprobs_five_to_top_logprobs_five() {
        let req = json!({"model": "m", "prompt": "x", "logprobs": 5});
        let wrapped = wrap_prompt_as_chat(&req);
        assert_eq!(wrapped["logprobs"], true);
        assert_eq!(wrapped["top_logprobs"], 5);
    }

    /// Legacy `logprobs: 0` (the "off" value) is translated to "no logprobs"
    /// on the chat side: neither `logprobs` nor `top_logprobs` appears in the
    /// wrapped body. A client-provided `top_logprobs` is also scrubbed so the
    /// upstream does not see a contradictory `logprobs: false` +
    /// `top_logprobs: N` pair.
    #[test]
    fn wrap_prompt_logprobs_zero_omits_logprobs_fields() {
        let req = json!({"model": "m", "prompt": "x", "logprobs": 0, "top_logprobs": 3});
        let wrapped = wrap_prompt_as_chat(&req);
        assert!(
            wrapped.get("logprobs").is_none(),
            "logprobs: 0 must not surface as a chat field: {wrapped}"
        );
        assert!(
            wrapped.get("top_logprobs").is_none(),
            "client top_logprobs must be scrubbed when legacy logprobs is 0: {wrapped}"
        );
    }

    /// A stream that yields a read error must forward a terminal error event
    /// followed by the `[DONE]` sentinel, so the legacy completions client
    /// can distinguish an upstream failure from a clean completion (M5 parity
    /// with messages.rs and `orchestration/stream.rs`, which always append
    /// `[DONE]` after an error chunk for OpenAI-shaped SSE streams).
    #[tokio::test]
    async fn stream_read_error_forwards_terminal_error_event() {
        let chunk = "data: {\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\n";
        let error_stream = futures_util::stream::iter([
            Ok::<Bytes, std::io::Error>(Bytes::from(chunk)),
            Err(std::io::Error::other("upstream gone")),
        ]);
        let response = Response::new(axum::body::Body::from_stream(error_stream));
        let out = convert_stream_to_completion(response)
            .await
            .expect("convert");
        let body = axum::body::to_bytes(out.into_body(), 1024 * 1024)
            .await
            .expect("body");
        let text = String::from_utf8_lossy(&body);
        assert!(
            text.contains(r#"{"error":{"message":"stream read error","type":"api_error"}}"#),
            "stream read error must forward a terminal error event: {text}"
        );
        let error_idx = text
            .find(r#""stream read error""#)
            .expect("error event present");
        let done_idx = text.rfind("[DONE]").expect("[DONE] sentinel present");
        assert!(
            done_idx > error_idx,
            "[DONE] must follow the error event (error @ {error_idx}, done @ {done_idx}): {text}"
        );
    }

    // ---- C17 LOW: logit_bias forwarding ---------------------------------

    /// `logit_bias` is a valid chat-completions parameter with an identical
    /// shape on the legacy endpoint, so the wrapper must forward it. Dropping
    /// it would be a silent behavior change for any legacy client relying on
    /// token-biasing.
    #[test]
    fn wrap_prompt_forwards_logit_bias() {
        let req = json!({
            "model": "gpt-3.5-turbo-instruct",
            "prompt": "Once upon a time",
            "logit_bias": {"50256": -100, "314": 5},
        });
        let wrapped = wrap_prompt_as_chat(&req);
        assert_eq!(wrapped["logit_bias"], json!({"50256": -100, "314": 5}));
    }

    // ---- C17 LOW: token-ID / mixed-type prompt rejection ----------------

    #[test]
    fn validate_prompt_accepts_string() {
        assert!(validate_legacy_prompt(&json!({"prompt": "hi"})).is_ok());
    }

    #[test]
    fn validate_prompt_accepts_array_of_strings() {
        assert!(validate_legacy_prompt(&json!({"prompt": ["a", "b"]})).is_ok());
    }

    #[test]
    fn validate_prompt_accepts_missing_prompt() {
        // Missing prompt is intentional (yields an empty user message).
        assert!(validate_legacy_prompt(&json!({"model": "m"})).is_ok());
    }

    #[test]
    fn validate_prompt_rejects_token_id_array() {
        let err = validate_legacy_prompt(&json!({"prompt": [1, 2, 3]}))
            .expect_err("token-ID array must be rejected");
        assert_eq!(err.status, axum::http::StatusCode::BAD_REQUEST);
        assert!(err.message.contains("prompt"), "message: {}", err.message);
    }

    #[test]
    fn validate_prompt_rejects_mixed_type_array() {
        let err = validate_legacy_prompt(&json!({"prompt": ["ok", 42]}))
            .expect_err("mixed-type array must be rejected");
        assert_eq!(err.status, axum::http::StatusCode::BAD_REQUEST);
    }

    #[test]
    fn validate_prompt_rejects_non_string_scalar() {
        let err = validate_legacy_prompt(&json!({"prompt": 123}))
            .expect_err("scalar prompt must be rejected");
        assert_eq!(err.status, axum::http::StatusCode::BAD_REQUEST);
    }

    // ---- C17 MEDIUM: non-streaming header preservation ------------------

    /// Build a synthetic successful chat response carrying the full set of
    /// orchestrator-stamped transparency headers and a minimal chat body.
    fn chat_response_with_headers() -> Response {
        let body = Bytes::from(
            r#"{"id":"x","object":"chat.completion","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}]}"#,
        );
        let mut response = Response::new(axum::body::Body::from(body));
        response
            .headers_mut()
            .insert(crate::pipeline::REQUEST_ID_HEADER, "req-1".parse().unwrap());
        response.headers_mut().insert(
            crate::pipeline::STANDARD_REQUEST_ID_HEADER,
            "req-1".parse().unwrap(),
        );
        response
            .headers_mut()
            .insert(crate::pipeline::COST_HEADER, "0.0001".parse().unwrap());
        response
            .headers_mut()
            .insert(crate::pipeline::CACHE_HIT_HEADER, "false".parse().unwrap());
        response.headers_mut().insert(
            crate::orchestration::ROUTER_STRATEGY_HEADER,
            "simple-shuffle".parse().unwrap(),
        );
        response.headers_mut().insert(
            crate::orchestration::ROUTER_DECISION_HEADER,
            "deployment-A".parse().unwrap(),
        );
        response.headers_mut().insert(
            crate::pipeline::RATELIMIT_REMAINING_REQUESTS,
            "99".parse().unwrap(),
        );
        // A non-preserved header that must NOT survive the conversion.
        response
            .headers_mut()
            .insert("content-type", "application/json".parse().unwrap());
        response
    }

    /// The non-streaming conversion must carry every `x-oxllm-*`,
    /// `x-ratelimit-*`, and the standard `x-request-id` header onto the legacy
    /// response. Previously it re-stamped only `x-oxllm-request-id`, silently
    /// dropping cost / router / ratelimit transparency for non-streaming
    /// legacy clients (while the streaming path preserved them).
    #[tokio::test]
    async fn convert_unary_preserves_transparency_headers() {
        let response = chat_response_with_headers();
        let out = convert_unary_to_completion(response)
            .await
            .expect("convert");
        let headers = out.headers();
        assert_eq!(
            headers
                .get(crate::pipeline::REQUEST_ID_HEADER)
                .and_then(|v| v.to_str().ok()),
            Some("req-1")
        );
        assert_eq!(
            headers
                .get(crate::pipeline::STANDARD_REQUEST_ID_HEADER)
                .and_then(|v| v.to_str().ok()),
            Some("req-1"),
            "standard x-request-id must be preserved on the legacy response"
        );
        assert_eq!(
            headers
                .get(crate::pipeline::COST_HEADER)
                .and_then(|v| v.to_str().ok()),
            Some("0.0001"),
            "x-oxllm-response-cost must be preserved"
        );
        assert_eq!(
            headers
                .get(crate::orchestration::ROUTER_STRATEGY_HEADER)
                .and_then(|v| v.to_str().ok()),
            Some("simple-shuffle"),
            "x-oxllm-router-strategy must be preserved"
        );
        assert_eq!(
            headers
                .get(crate::orchestration::ROUTER_DECISION_HEADER)
                .and_then(|v| v.to_str().ok()),
            Some("deployment-A")
        );
        assert_eq!(
            headers
                .get(crate::pipeline::RATELIMIT_REMAINING_REQUESTS)
                .and_then(|v| v.to_str().ok()),
            Some("99"),
            "an x-ratelimit-* header must be preserved"
        );
        // The legacy body shape is still produced correctly.
        let body = axum::body::to_bytes(out.into_body(), 1024 * 1024)
            .await
            .expect("body");
        let legacy: Value = serde_json::from_slice(&body).expect("json");
        assert_eq!(legacy["object"], "text_completion");
        assert_eq!(legacy["choices"][0]["text"], "hi");
    }

    /// Regression pin: the streaming conversion must preserve the same header
    /// families as the non-streaming path (including `x-request-id`), so the
    /// two paths stay consistent.
    #[tokio::test]
    async fn convert_stream_preserves_transparency_headers() {
        let chunk = "data: {\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\n";
        let body = axum::body::Body::from(Bytes::from(chunk));
        let mut src = Response::new(body);
        src.headers_mut()
            .insert(crate::pipeline::COST_HEADER, "0.0002".parse().unwrap());
        src.headers_mut().insert(
            crate::pipeline::STANDARD_REQUEST_ID_HEADER,
            "req-2".parse().unwrap(),
        );
        src.headers_mut().insert(
            crate::pipeline::RATELIMIT_REMAINING_TOKENS,
            "500".parse().unwrap(),
        );
        let out = convert_stream_to_completion(src).await.expect("convert");
        let headers = out.headers();
        assert_eq!(
            headers
                .get(crate::pipeline::COST_HEADER)
                .and_then(|v| v.to_str().ok()),
            Some("0.0002")
        );
        assert_eq!(
            headers
                .get(crate::pipeline::STANDARD_REQUEST_ID_HEADER)
                .and_then(|v| v.to_str().ok()),
            Some("req-2")
        );
        assert_eq!(
            headers
                .get(crate::pipeline::RATELIMIT_REMAINING_TOKENS)
                .and_then(|v| v.to_str().ok()),
            Some("500")
        );
    }

    /// The non-streaming body read must accept chat responses above the old
    /// 10 MiB magic cap (now aligned with the upstream `MAX_RESPONSE_BODY_BYTES`
    /// 50 MiB cap). An 11 MiB body would previously fail with a generic
    /// "failed to read chat response" internal error.
    #[tokio::test]
    async fn convert_unary_reads_body_above_old_ten_mib_cap() {
        // Build an 11 MiB chat body whose choices content is a large padding
        // string so the total serialized size exceeds 10 MiB.
        let padding = "a".repeat(11 * 1024 * 1024);
        let chat = json!({
            "id": "big",
            "object": "chat.completion",
            "model": "m",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": padding},
                "finish_reason": "stop"
            }]
        });
        let body = Bytes::from(serde_json::to_vec(&chat).expect("serialize"));
        let response = Response::new(axum::body::Body::from(body));
        let out = convert_unary_to_completion(response)
            .await
            .expect("11 MiB body must succeed under the 50 MiB cap");
        let body_bytes = axum::body::to_bytes(out.into_body(), 60 * 1024 * 1024)
            .await
            .expect("body");
        let legacy: Value = serde_json::from_slice(&body_bytes).expect("json");
        assert_eq!(legacy["object"], "text_completion");
        assert_eq!(
            legacy["choices"][0]["text"].as_str().unwrap().len(),
            11 * 1024 * 1024
        );
    }

    // ---- C17 HIGH: non-string message.content must not silently empty text --

    /// A chat response whose `message.content` is the array-of-parts shape
    /// used by every vision/multimodal model must flatten its `text` parts
    /// into the legacy `text` field instead of silently producing `""`. The
    /// previous `unwrap_or("")` path dropped the entire completion for any
    /// non-string content.
    #[test]
    fn chat_to_legacy_completion_flattens_content_part_array() {
        let chat = json!({
            "id": "x",
            "object": "chat.completion",
            "model": "m",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "a"},
                        {"type": "image_url", "image_url": {"url": "data:..."}},
                        {"type": "text", "text": "b"}
                    ]
                },
                "finish_reason": "stop"
            }]
        });
        let out = chat_to_legacy_completion(&chat);
        assert_eq!(out["choices"][0]["text"], "ab");
        assert!(
            out["choices"][0].get("message").is_none(),
            "message must be removed after flattening: {out}"
        );
    }

    /// A null content (some providers emit this for tool-call-only responses)
    /// flattens to an empty string rather than panicking.
    #[test]
    fn chat_to_legacy_completion_handles_null_content() {
        let chat = json!({
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": null},
                "finish_reason": "stop"
            }]
        });
        let out = chat_to_legacy_completion(&chat);
        assert_eq!(out["choices"][0]["text"], "");
    }

    // ---- C17 MEDIUM: convert_unary surfaces original body on parse failure -

    /// When the chat body cannot be parsed as JSON (upstream returned
    /// malformed bytes, or a future bug in the converter), the handler must
    /// surface the *original* chat body verbatim with an
    /// `x-oxllm-shape: conversion-failed` marker rather than a generic 500.
    /// By this point `pipeline::post_success` has already debited budget, so
    /// the caller should still receive the bytes they paid for.
    #[tokio::test]
    async fn convert_unary_to_completion_propagates_original_body_on_parse_failure() {
        let original = Bytes::from_static(b"not valid json");
        let mut src = Response::new(axum::body::Body::from(original.clone()));
        src.headers_mut()
            .insert(crate::pipeline::COST_HEADER, "0.001".parse().unwrap());
        src.headers_mut().insert(
            crate::pipeline::STANDARD_REQUEST_ID_HEADER,
            "req-x".parse().unwrap(),
        );

        let out = convert_unary_to_completion(src).await.expect("not a 500");
        assert!(
            !out.status().is_server_error(),
            "conversion failure must not surface as a 5xx: {}",
            out.status()
        );
        assert_eq!(
            out.headers()
                .get("x-oxllm-shape")
                .and_then(|v| v.to_str().ok()),
            Some("conversion-failed"),
            "must mark the response so the caller can tell shape conversion failed"
        );
        assert_eq!(
            out.headers()
                .get(crate::pipeline::COST_HEADER)
                .and_then(|v| v.to_str().ok()),
            Some("0.001"),
            "cost header must survive the recovery path"
        );
        let body = axum::body::to_bytes(out.into_body(), 1024 * 1024)
            .await
            .expect("body");
        assert_eq!(
            &body[..],
            &original[..],
            "original chat body must be passed through"
        );
    }

    // ---- C17 MEDIUM: streaming preserves delta.reasoning_content -----------

    /// A streaming chunk carrying `delta.reasoning_content` (DeepSeek-R1 /
    /// o1-style reasoning models) must hoist the reasoning trace onto the
    /// choice level so legacy clients receive it. Mirrors the reference
    /// `convert_to_text_completion_object`.
    #[test]
    fn convert_chunk_preserves_reasoning_content() {
        let chunk = json!({
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "choices": [{
                "index": 0,
                "delta": {"reasoning_content": "thinking..."},
                "finish_reason": null
            }]
        });
        let converted = convert_chat_chunk_to_completion(&chunk);
        assert_eq!(converted["object"], "text_completion");
        assert!(
            converted["choices"][0].get("delta").is_none(),
            "delta must be removed: {converted}"
        );
        assert_eq!(
            converted["choices"][0]["reasoning_content"], "thinking...",
            "reasoning_content must be hoisted to the choice level"
        );
    }

    /// A chunk with both `content` and `reasoning_content` deltas hoists both.
    #[test]
    fn convert_chunk_preserves_reasoning_content_alongside_content() {
        let chunk = json!({
            "object": "chat.completion.chunk",
            "choices": [{
                "index": 0,
                "delta": {"content": "ans", "reasoning_content": "why"},
                "finish_reason": null
            }]
        });
        let converted = convert_chat_chunk_to_completion(&chunk);
        assert_eq!(converted["choices"][0]["text"], "ans");
        assert_eq!(converted["choices"][0]["reasoning_content"], "why");
    }

    // ---- C17 MEDIUM: wrap_prompt_as_chat forwards modern chat params -------

    /// The wrapper must forward `response_format` and `stream_options` (and
    /// `top_logprobs`, `max_completion_tokens`) so legacy clients using these
    /// OpenAI chat params reach the upstream. Previously the explicit field
    /// list silently dropped them.
    #[test]
    fn wrap_forwards_response_format_and_stream_options() {
        let req = json!({
            "model": "m",
            "prompt": "hi",
            "response_format": {"type": "json_object"},
            "stream_options": {"include_usage": true},
            "top_logprobs": 3,
            "max_completion_tokens": 128,
            "metadata": {"order_id": "o-1"}
        });
        let wrapped = wrap_prompt_as_chat(&req);
        assert_eq!(wrapped["response_format"], json!({"type": "json_object"}));
        assert_eq!(wrapped["stream_options"], json!({"include_usage": true}));
        assert_eq!(wrapped["top_logprobs"], 3);
        assert_eq!(wrapped["max_completion_tokens"], 128);
        assert_eq!(wrapped["metadata"]["order_id"], "o-1");
    }

    /// Legacy text-completion-only params (`echo`, `best_of`, `suffix`) have
    /// no chat equivalent and must NOT be forwarded (the upstream would 400).
    /// `prompt` is consumed to build the wrapped user message.
    #[test]
    fn wrap_drops_text_completion_only_params() {
        let req = json!({
            "model": "m",
            "prompt": "hi",
            "echo": true,
            "best_of": 4,
            "suffix": "tail"
        });
        let wrapped = wrap_prompt_as_chat(&req);
        assert!(
            wrapped.get("echo").is_none(),
            "echo must be dropped: {wrapped}"
        );
        assert!(
            wrapped.get("best_of").is_none(),
            "best_of must be dropped: {wrapped}"
        );
        assert!(
            wrapped.get("suffix").is_none(),
            "suffix must be dropped: {wrapped}"
        );
        assert!(
            wrapped.get("prompt").is_none(),
            "prompt must be consumed: {wrapped}"
        );
    }

    /// `stream: false` must round-trip through the wrapper so the converter
    /// picks the unary path. Previously only `stream: true` was implicitly
    /// pinned by the streaming E2E tests.
    #[test]
    fn wrap_round_trips_stream_false() {
        let req = json!({"model": "m", "prompt": "hi", "stream": false});
        let wrapped = wrap_prompt_as_chat(&req);
        assert_eq!(wrapped["stream"], false);
    }

    // ---- C17 LOW: stream buffer overflow drains all queued events ---------

    /// When the SSE buffer exceeds the cap with multiple complete events
    /// queued (e.g. a misbehaving upstream emits a long run of small frames
    /// without a trailing separator within a single read), the recovery path
    /// must drain every complete event via `pop_sse_event` before the
    /// single-shot fallback. Previously only the first `data:` line survived
    /// and the rest were dropped on `buffer.clear()`.
    #[tokio::test]
    async fn convert_stream_drains_all_events_on_buffer_overflow() {
        let mut events = String::new();
        for i in 0..50 {
            events.push_str(&format!(
                "data: {{\"object\":\"chat.completion.chunk\",\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"c{i}\"}},\"finish_reason\":null}}]}}\n\n"
            ));
        }
        let one_big_chunk = Bytes::from(events);
        let response = Response::new(axum::body::Body::from(one_big_chunk));
        let out = convert_stream_to_completion(response)
            .await
            .expect("convert");
        let body = axum::body::to_bytes(out.into_body(), 16 * 1024 * 1024)
            .await
            .expect("body");
        let text = String::from_utf8_lossy(&body);
        let forwarded = text.matches("text_completion").count();
        assert!(
            forwarded >= 50,
            "every complete event must survive the overflow recovery; saw {forwarded} in: {text}"
        );
    }

    // ---- C17 LOW: stream buffer overflow preserves a split >cap event ------

    /// A single SSE event larger than `MAX_STREAM_BUFFER_BYTES` (4 MiB), split
    /// across two reads such that the first read alone exceeds the cap without
    /// containing the `\n\n` terminator, must still be reassembled and emitted
    /// once the second read supplies the terminator. Previously the overflow
    /// branch force-parsed the partial (which always fails JSON
    /// deserialization) and then `buffer.clear()`-ed, dropping the event at
    /// the boundary — the trailing read had no `data:` prefix and was also
    /// lost.
    #[tokio::test]
    async fn convert_stream_preserves_split_event_across_overflow() {
        let payload = "a".repeat(MAX_STREAM_BUFFER_BYTES + 1024 * 1024);
        let event = format!(
            "data: {{\"object\":\"chat.completion.chunk\",\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"{payload}\"}},\"finish_reason\":null}}]}}\n\n"
        );
        assert!(
            event.len() > MAX_STREAM_BUFFER_BYTES,
            "test fixture must exceed the cap: {}",
            event.len()
        );
        // First read: everything except the trailing `\n\n`. This alone is
        // above the cap and contains no event separator, so the overflow
        // branch fires on the first chunk.
        let split_point = event.len() - 2;
        let chunk1 = Bytes::from(event[..split_point].to_string());
        let chunk2 = Bytes::from(event[split_point..].to_string());
        let stream = futures_util::stream::iter([Ok::<Bytes, std::io::Error>(chunk1), Ok(chunk2)]);
        let response = Response::new(axum::body::Body::from_stream(stream));
        let out = convert_stream_to_completion(response)
            .await
            .expect("convert");
        let body = axum::body::to_bytes(out.into_body(), 16 * 1024 * 1024)
            .await
            .expect("body");
        let text = String::from_utf8_lossy(&body);
        assert!(
            text.contains("\"object\":\"text_completion\""),
            "the split >cap event must survive the overflow recovery: {text}"
        );
        let preserved_payload_len = payload.len();
        assert!(
            text.matches('a').count() >= preserved_payload_len,
            "the full >cap content must be present in the converted stream: {text}",
        );
    }

    /// Bounding behaviour: if the buffer overflows twice in a row without
    /// yielding a complete event (a genuinely misbehaving upstream that never
    /// emits a separator), the partial is dropped on the second overflow so
    /// the buffer cannot grow unboundedly. A subsequent well-formed event
    /// still flows through cleanly.
    #[tokio::test]
    async fn convert_stream_drops_partial_after_two_consecutive_overflows() {
        // Chunk 1: >cap partial with no separator -> overflow, preserve.
        let huge_partial_1 = "data: {\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"";
        // 'Z' is chosen so it cannot appear in the converted well-formed
        // output (whose keys are fixed: choices/finish_reason/index/logprobs/
        // text/object), making the leak check below unambiguous.
        let padding_1 = "Z".repeat(MAX_STREAM_BUFFER_BYTES + 1024);
        let chunk1 = Bytes::from(format!("{huge_partial_1}{padding_1}"));
        // Chunk 2: more bytes, still no separator -> second overflow, drop.
        let chunk2 = Bytes::from("Z".repeat(1024));
        // Chunk 3: a fresh, well-formed small event must still convert.
        let good = "data: {\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":null}]}\n\n";
        let chunk3 = Bytes::from(good.to_string());
        let stream = futures_util::stream::iter([
            Ok::<Bytes, std::io::Error>(chunk1),
            Ok(chunk2),
            Ok(chunk3),
        ]);
        let response = Response::new(axum::body::Body::from_stream(stream));
        let out = convert_stream_to_completion(response)
            .await
            .expect("convert");
        let body = axum::body::to_bytes(out.into_body(), 16 * 1024 * 1024)
            .await
            .expect("body");
        let text = String::from_utf8_lossy(&body);
        assert!(
            text.contains("\"text\":\"ok\""),
            "the well-formed event after the dropped partials must survive: {text}"
        );
        // The dropped partials never formed a complete event, so their padding
        // must not appear in the converted output.
        assert!(
            !text.contains('Z'),
            "padding from dropped partials must not leak: {text}",
        );
    }
}
