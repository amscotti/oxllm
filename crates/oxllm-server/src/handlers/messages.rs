//! `POST /v1/messages` — Anthropic Messages input API.
//!
//! Accepts Anthropic-shaped requests from Anthropic SDK clients, translates
//! them to the internal OpenAI format, routes through the standard chat
//! pipeline, and converts the OpenAI response back to Anthropic format.
//!
//! This is a thin handler: all API-shape conversion lives in
//! `crate::handlers::convert::anthropic`.

use std::time::Duration;

use axum::Extension;
use axum::Json;
use axum::body::Bytes;
use axum::extract::State;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;

/// Interval between Anthropic SSE keepalive pings while the upstream is
/// silent (extended thinking).
const ANTHROPIC_SSE_PING_INTERVAL: Duration = Duration::from_secs(15);

use crate::auth::AuthUser;
use crate::error::ApiError;
use crate::handlers::anthropic_common::{
    anthropic_error_response, sanitise_forwarded_error_message, status_to_anthropic_error_type,
};
use crate::handlers::chat::chat_completions_inner_preserving_stream_usage;
use crate::handlers::convert::anthropic::{
    AnthropicStreamConverter, anthropic_to_openai_request, openai_to_anthropic_response,
    process_openai_sse_event, validate_anthropic_to_openai_request,
};
use crate::orchestration::stream::send_converted_event;
use crate::pipeline::preserved_response_headers;
use crate::sse::{SseFramer, pop_sse_event};
use crate::state::SharedState;

/// Anthropic Messages handler.
///
/// Converts the incoming Anthropic request to OpenAI format, forwards through
/// the chat completions pipeline, then converts the OpenAI response back to
/// Anthropic format.
pub async fn messages(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    headers: axum::http::HeaderMap,
    body: Bytes,
) -> Result<Response, ApiError> {
    match messages_inner(shared, auth, headers, body).await {
        Ok(response) => Ok(response),
        Err(error) => {
            let status = error.status;
            let message = error.message.clone();
            let preserved = preserved_response_headers(error.into_response().headers());
            Ok(anthropic_error_response(
                status,
                status_to_anthropic_error_type(status),
                &message,
                &preserved,
            ))
        }
    }
}

async fn messages_inner(
    shared: SharedState,
    auth: AuthUser,
    headers: axum::http::HeaderMap,
    body: Bytes,
) -> Result<Response, ApiError> {
    let state = shared.load_full();
    let request_body: Value = serde_json::from_slice(&body)
        .map_err(|e| ApiError::bad_request(format!("invalid JSON body: {e}")))?;

    // Depth-check the raw Anthropic body *before* conversion so a deeply
    // nested payload cannot burn CPU/alloc in `anthropic_to_openai_request`.
    // Responses API does the same; chat also checks the converted body.
    crate::security::check_json_depth(&request_body, crate::security::MAX_JSON_DEPTH)?;
    crate::orchestration::request_controls::validate(&request_body)?;

    let model = request_body
        .get("model")
        .and_then(|v| v.as_str())
        .filter(|model| !model.trim().is_empty())
        .ok_or_else(|| ApiError::bad_request("missing required field: 'model'"))?;
    crate::pipeline::check_request_model_acl(&state, &auth, model, &request_body, &headers).await?;

    validate_anthropic_to_openai_request(&request_body)?;
    let drop_params = state
        .router
        .deployments_for_model(&state.deployments, model)
        .into_iter()
        .next()
        .and_then(|deployment| deployment.oxllm_params.drop_params)
        .or(state.config.oxllm_settings.drop_params)
        .unwrap_or(false)
        || request_body.get("drop_params").and_then(Value::as_bool) == Some(true);
    oxllm_providers::anthropic::count_tokens::validate_tool_result_text(
        &request_body,
        drop_params,
    )?;
    let chat_body = anthropic_to_openai_request(&request_body);
    let chat_bytes = serde_json::to_vec(&chat_body)
        .map_err(|e| ApiError::internal(format!("failed to serialize request: {e}")))?;

    // Read `is_stream` from the converted OpenAI body so the converter path
    // and the orchestrator agree on a single source of truth (matching
    // completions.rs). Reading it from the Anthropic body and relying on the
    // converter to forward `stream` would diverge silently if the converter
    // ever changed.
    let is_stream = chat_body
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let chat_response = match chat_completions_inner_preserving_stream_usage(
        state,
        &auth,
        &headers,
        Bytes::from(chat_bytes),
    )
    .await
    {
        Ok(resp) if resp.status().is_success() => resp,
        Ok(resp) => {
            let status = resp.status();
            // Capture the orchestrator-stamped transparency headers
            // (`x-request-id`, `x-oxllm-*`, `x-ratelimit-*`) BEFORE
            // `into_body()` discards the header map, so an Anthropic SDK
            // client can correlate a 429/500 from this route with
            // logs/traces — matching the success path's
            // `preserved_response_headers` handling.
            let preserved = preserved_response_headers(resp.headers());
            let body_bytes =
                axum::body::to_bytes(resp.into_body(), crate::security::MAX_ERROR_BODY_BYTES)
                    .await
                    .map_err(|e| ApiError::internal(format!("failed to read error body: {e}")))?;
            let error_body: Value = serde_json::from_slice(&body_bytes).unwrap_or(Value::Null);
            let raw_message = error_body
                .get("error")
                .and_then(|e| e.get("message"))
                .and_then(|m| m.as_str())
                .unwrap_or("upstream error");
            tracing::debug!(
                status = status.as_u16(),
                raw_message = raw_message,
                "anthropic messages handler forwarding upstream error"
            );
            let message = sanitise_forwarded_error_message(raw_message);
            return Ok(anthropic_error_response(
                status,
                status_to_anthropic_error_type(status),
                &message,
                &preserved,
            ));
        }
        Err(e) => {
            let status = e.status;
            let message = e.message.clone();
            let error_type = status_to_anthropic_error_type(e.status);
            let preserved = preserved_response_headers(e.into_response().headers());
            return Ok(anthropic_error_response(
                status, error_type, &message, &preserved,
            ));
        }
    };

    if is_stream {
        convert_stream_to_anthropic(chat_response).await
    } else {
        convert_json_to_anthropic(chat_response).await
    }
}

async fn convert_json_to_anthropic(chat_response: Response) -> Result<Response, ApiError> {
    let preserved_headers = preserved_response_headers(chat_response.headers());
    let body_bytes = axum::body::to_bytes(
        chat_response.into_body(),
        crate::security::MAX_RESPONSE_BODY_BYTES,
    )
    .await
    .map_err(|e| ApiError::internal(format!("failed to read response body: {e}")))?;
    let openai_resp: Value = match serde_json::from_slice(&body_bytes) {
        Ok(v) => v,
        Err(_) => {
            // By this point `pipeline::post_success` has already debited
            // budget / written a spend log, so converting to a generic 500
            // would silently lose both the data and any chance for the client
            // to retry idempotently. Surface the original chat body verbatim
            // with an `x-oxllm-shape: conversion-failed` marker so the caller
            // still receives the bytes they paid for. Mirrors completions.rs.
            tracing::warn!(
                bytes = body_bytes.len(),
                "anthropic messages conversion failed; surfacing original chat body"
            );
            let mut response = Response::new(axum::body::Body::from(body_bytes));
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
            return Ok(response);
        }
    };
    let anthropic_resp = openai_to_anthropic_response(&openai_resp);
    let mut response = Json(anthropic_resp).into_response();
    for (name, value) in preserved_headers {
        response.headers_mut().insert(name, value);
    }
    Ok(response)
}

async fn convert_stream_to_anthropic(chat_response: Response) -> Result<Response, ApiError> {
    convert_stream_to_anthropic_with_ping_interval(chat_response, ANTHROPIC_SSE_PING_INTERVAL).await
}

async fn convert_stream_to_anthropic_with_ping_interval(
    chat_response: Response,
    ping_interval: Duration,
) -> Result<Response, ApiError> {
    let preserved_headers = preserved_response_headers(chat_response.headers());

    let body = chat_response.into_body();
    let data_stream = body.into_data_stream();

    let (tx, rx) = mpsc::channel::<Result<Event, std::convert::Infallible>>(64);

    tokio::spawn(async move {
        let mut converter = AnthropicStreamConverter::new();
        let mut framer = SseFramer::new();
        let mut buffer = String::new();
        let mut stream = data_stream;
        // Sticky flag for the >cap partial-event recovery: set when an overflow
        // leaves a partial event in `buffer`, cleared on any successful
        // `pop_sse_event`. Lets a >cap event split across two reads complete
        // on the second read while still bounding growth for a misbehaving
        // upstream that never emits a separator. Ported from `completions.rs`.
        let mut overflow_pending = false;

        loop {
            tokio::select! {
            _ = tx.closed() => return,
            result = tokio::time::timeout(ping_interval, stream.next()) => {
            match result {
            Err(_elapsed) => {
            let ping = json!({"type": "ping"});
            if send_converted_event(&tx, build_anthropic_sse_event(&ping)).await.is_err() {
            return;
            }
            }
            Ok(Some(Ok(bytes))) => {
            buffer.push_str(&framer.push(&bytes));
            while let Some(event_str) = pop_sse_event(&mut buffer) {
            overflow_pending = false;
            for event_value in process_openai_sse_event(&mut converter, &event_str) {
            let sse_event = build_anthropic_sse_event(&event_value);
            if send_converted_event(&tx, sse_event).await.is_err() {
            return;
            }
            }
            }
            if buffer.len() > crate::handlers::chat::MAX_STREAM_BUFFER_BYTES {
            // The drain loop above already exhausted every complete event,
            // so `buffer` now holds only a partial frame. Preserve it
            // across this overflow so a >cap event split across two reads
            // can complete on the next chunk (previously the partial was
            // force-parsed — which always fails JSON deserialization on a
            // truncated body — and then `buffer.clear()`-ed, losing the
            // event at the boundary). Only preserve once: if the next read
            // also overflows without yielding a complete event, drop the
            // accumulated partial to bound growth.
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
            Ok(Some(Err(e))) => {
            tracing::warn!(error = %e, "stream read error");
            // Close any open content block before the terminal error so the
            // event stream stays balanced (every `content_block_start` pairs
            // with a `content_block_stop`), mirroring the chunk-level error
            // arm in `convert_chunk`. Without this an Anthropic SDK client
            // that tracks block state would see an unbalanced start.
            for event_value in converter.close_open_block() {
            let sse_event = build_anthropic_sse_event(&event_value);
            if send_converted_event(&tx, sse_event).await.is_err() {
            return;
            }
            }
            // M5: forward a terminal error event before returning so the
            // client's SSE stream is closed explicitly rather than truncated
            // silently. Without this the converter stream just ends and the
            // Anthropic-shaped client cannot distinguish a clean completion
            // from an upstream failure.
            let error_value = json!({
            "type": "error",
            "error": {"type": "api_error", "message": "stream read error"}
            });
            let _ = send_converted_event(&tx, build_anthropic_sse_event(&error_value)).await;
            return;
            }
            Ok(None) => {
            match framer.finish() {
            Ok(remaining) => buffer.push_str(&remaining),
            Err(e) => {
            tracing::warn!(error = %e, "stream ended with undecodable trailing bytes; dropping them");
            }
            }
            if !buffer.trim().is_empty() {
            for event_value in process_openai_sse_event(&mut converter, &buffer) {
            let sse_event = build_anthropic_sse_event(&event_value);
            if send_converted_event(&tx, sse_event).await.is_err() {
            return;
            }
            }
            }
            for event_value in converter.finish_events() {
            let sse_event = build_anthropic_sse_event(&event_value);
            if send_converted_event(&tx, sse_event).await.is_err() {
            return;
            }
            }
            break;
            }
            }
            }
            }
        }
    });

    let mut response = Sse::new(ReceiverStream::new(rx)).into_response();
    for (name, value) in preserved_headers {
        response.headers_mut().insert(name, value);
    }
    Ok(response)
}

fn build_anthropic_sse_event(value: &Value) -> Event {
    let event_type = value.get("type").and_then(|v| v.as_str()).unwrap_or("");
    Event::default().event(event_type).data(value.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::StatusCode;

    /// Idle upstream (extended thinking) must emit Anthropic `event: ping`
    /// keepalives so proxies and SDKs do not drop the connection.
    #[tokio::test]
    async fn stream_idle_emits_anthropic_ping() {
        let (delay_tx, delay_rx) = mpsc::channel::<Result<Bytes, std::io::Error>>(1);
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(40)).await;
            let _ = delay_tx
                .send(Ok(Bytes::from(
                    "data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\ndata: [DONE]\n\n",
                )))
                .await;
        });
        let delayed = ReceiverStream::new(delay_rx);
        let response = Response::new(axum::body::Body::from_stream(delayed));
        let out =
            convert_stream_to_anthropic_with_ping_interval(response, Duration::from_millis(10))
                .await
                .expect("convert");
        let body = axum::body::to_bytes(out.into_body(), 1024 * 1024)
            .await
            .expect("body");
        let text = String::from_utf8_lossy(&body);
        assert!(
            text.contains("event: ping"),
            "keepalive ping must set the SSE event field: {text}"
        );
        assert!(
            text.contains(r#""type":"ping""#),
            "keepalive ping body must be Anthropic ping: {text}"
        );
    }

    /// A stream that yields a read error must forward a terminal `error`
    /// event that carries both the SSE `event: error` discriminator and the
    /// Anthropic-shaped `{"type":"error",...}` body, matching every other
    /// event in the stream (M5 parity with completions.rs).
    #[tokio::test]
    async fn stream_read_error_forwards_terminal_error_event_with_event_field() {
        let chunk = "data: {\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\n";
        let error_stream = futures_util::stream::iter([
            Ok::<Bytes, std::io::Error>(Bytes::from(chunk)),
            Err(std::io::Error::other("upstream gone")),
        ]);
        let response = Response::new(axum::body::Body::from_stream(error_stream));
        let out = convert_stream_to_anthropic(response)
            .await
            .expect("convert");
        let body = axum::body::to_bytes(out.into_body(), 1024 * 1024)
            .await
            .expect("body");
        let text = String::from_utf8_lossy(&body);
        assert!(
            text.contains("event: error"),
            "terminal error frame must set the SSE event: field: {text}"
        );
        assert!(
            text.contains(r#""type":"error""#),
            "terminal error body must carry the Anthropic error shape: {text}"
        );
        assert!(
            text.contains("stream read error"),
            "terminal error body must carry the stream-read-error message: {text}"
        );
        // The content chunk above opens a text block (`content_block_start`);
        // the read-error path must close it with a `content_block_stop`
        // BEFORE the terminal error so the block stream stays balanced
        // (matching the chunk-level error arm in the converter).
        let stop_pos = text.find("content_block_stop");
        let error_pos = text.find("event: error");
        assert!(
            stop_pos.is_some(),
            "open content block must be closed before the terminal error: {text}"
        );
        assert!(
            stop_path_is_before_error(stop_pos, error_pos),
            "content_block_stop must precede the terminal error event: {text}"
        );
    }

    fn stop_path_is_before_error(stop: Option<usize>, error: Option<usize>) -> bool {
        match (stop, error) {
            (Some(s), Some(e)) => s < e,
            _ => false,
        }
    }

    /// The non-streaming conversion must carry the orchestrator-stamped
    /// `x-oxllm-*`, `x-ratelimit-*`, and `x-request-id` headers onto the
    /// Anthropic-shaped response. Mirrors the parity test in completions.rs.
    #[tokio::test]
    async fn convert_json_preserves_transparency_headers() {
        let body = Bytes::from(
            r#"{"id":"x","object":"chat.completion","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}]}"#,
        );
        let mut src = Response::new(axum::body::Body::from(body));
        src.headers_mut()
            .insert(crate::pipeline::REQUEST_ID_HEADER, "req-1".parse().unwrap());
        src.headers_mut().insert(
            crate::pipeline::STANDARD_REQUEST_ID_HEADER,
            "req-1".parse().unwrap(),
        );
        src.headers_mut()
            .insert(crate::pipeline::COST_HEADER, "0.0001".parse().unwrap());
        src.headers_mut().insert(
            crate::orchestration::ROUTER_STRATEGY_HEADER,
            "simple-shuffle".parse().unwrap(),
        );
        src.headers_mut().insert(
            crate::pipeline::RATELIMIT_REMAINING_REQUESTS,
            "99".parse().unwrap(),
        );

        let out = convert_json_to_anthropic(src).await.expect("convert");
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
            Some("req-1")
        );
        assert_eq!(
            headers
                .get(crate::pipeline::COST_HEADER)
                .and_then(|v| v.to_str().ok()),
            Some("0.0001")
        );
        assert_eq!(
            headers
                .get(crate::orchestration::ROUTER_STRATEGY_HEADER)
                .and_then(|v| v.to_str().ok()),
            Some("simple-shuffle")
        );
        assert_eq!(
            headers
                .get(crate::pipeline::RATELIMIT_REMAINING_REQUESTS)
                .and_then(|v| v.to_str().ok()),
            Some("99")
        );

        let body = axum::body::to_bytes(out.into_body(), 1024 * 1024)
            .await
            .expect("body");
        let anthropic: Value = serde_json::from_slice(&body).expect("json");
        assert_eq!(anthropic["type"], "message");
        assert_eq!(anthropic["content"][0]["text"], "hi");
    }

    /// The non-streaming body read must accept chat responses above the old
    /// 10 MiB magic cap (now aligned with `MAX_RESPONSE_BODY_BYTES` = 50 MiB).
    /// An 11 MiB body would previously fail with a generic
    /// "failed to read response body" internal error.
    #[tokio::test]
    async fn convert_json_reads_body_above_old_ten_mib_cap() {
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
        let out = convert_json_to_anthropic(response)
            .await
            .expect("11 MiB body must succeed under the 50 MiB cap");
        let body_bytes = axum::body::to_bytes(out.into_body(), 60 * 1024 * 1024)
            .await
            .expect("body");
        let anthropic: Value = serde_json::from_slice(&body_bytes).expect("json");
        assert_eq!(anthropic["type"], "message");
        assert_eq!(
            anthropic["content"][0]["text"].as_str().unwrap().len(),
            11 * 1024 * 1024
        );
    }

    /// When the chat body cannot be parsed as JSON, the handler must surface
    /// the *original* chat body verbatim with an `x-oxllm-shape:
    /// conversion-failed` marker rather than a generic 500. By this point
    /// `pipeline::post_success` has already debited budget, so the caller
    /// should still receive the bytes they paid for. Mirrors the parity test
    /// in completions.rs.
    #[tokio::test]
    async fn convert_json_surfaces_original_body_on_parse_failure() {
        let original = Bytes::from_static(b"not valid json");
        let mut src = Response::new(axum::body::Body::from(original.clone()));
        src.headers_mut()
            .insert(crate::pipeline::COST_HEADER, "0.001".parse().unwrap());
        src.headers_mut().insert(
            crate::pipeline::STANDARD_REQUEST_ID_HEADER,
            "req-x".parse().unwrap(),
        );

        let out = convert_json_to_anthropic(src).await.expect("not a 500");
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

    // ---- C18 MEDIUM: stream buffer overflow preserves a split >cap event ---

    /// A single SSE event larger than `MAX_STREAM_BUFFER_BYTES` (4 MiB), split
    /// across two reads such that the first read alone exceeds the cap without
    /// containing the `\n\n` terminator, must still be reassembled and emitted
    /// once the second read supplies the terminator. Previously the overflow
    /// branch force-parsed the partial (which always fails JSON
    /// deserialization on a truncated body) and then `buffer.clear()`-ed,
    /// dropping the event at the boundary. Ported from `completions.rs`.
    #[tokio::test]
    async fn convert_stream_preserves_split_event_across_overflow() {
        let payload = "a".repeat(crate::handlers::chat::MAX_STREAM_BUFFER_BYTES + 1024 * 1024);
        let event = format!(
            "data: {{\"id\":\"x\",\"object\":\"chat.completion.chunk\",\"model\":\"m\",\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"{payload}\"}},\"finish_reason\":null}}]}}\n\n"
        );
        assert!(
            event.len() > crate::handlers::chat::MAX_STREAM_BUFFER_BYTES,
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
        let out = convert_stream_to_anthropic(response)
            .await
            .expect("convert");
        let body = axum::body::to_bytes(out.into_body(), 16 * 1024 * 1024)
            .await
            .expect("body");
        let text = String::from_utf8_lossy(&body);
        assert!(
            text.contains("content_block_start"),
            "the split >cap event must survive the overflow recovery: {text}"
        );
        assert!(
            text.matches('a').count() >= payload.len(),
            "the full >cap content must be present in the converted stream",
        );
    }

    // ---- C18 LOW: non-2xx error path preserves transparency headers --------

    /// A non-2xx upstream response must carry the orchestrator-stamped
    /// transparency headers (`x-request-id`, `x-oxllm-*`) onto the
    /// Anthropic-shaped error response, matching the success path. Previously
    /// `into_body()` discarded the header map before the error response was
    /// built, so an Anthropic SDK client could not correlate a 429/500.
    #[tokio::test]
    async fn anthropic_error_response_preserves_transparency_headers() {
        let preserved = vec![(
            crate::pipeline::STANDARD_REQUEST_ID_HEADER.parse().unwrap(),
            "req-9".parse().unwrap(),
        )];
        let resp = anthropic_error_response(
            StatusCode::TOO_MANY_REQUESTS,
            "rate_limit_error",
            "slow down",
            &preserved,
        );
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(
            resp.headers()
                .get(crate::pipeline::STANDARD_REQUEST_ID_HEADER)
                .and_then(|v| v.to_str().ok()),
            Some("req-9"),
            "x-request-id must survive onto the Anthropic-shaped error response"
        );
        let body = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .expect("body");
        let v: Value = serde_json::from_slice(&body).expect("json");
        assert_eq!(v["type"], "error");
        assert_eq!(v["error"]["type"], "rate_limit_error");
    }

    // ---- C18 LOW: forwarded upstream error message is sanitised ------------

    #[test]
    fn sanitise_redacts_api_key_and_bearer_and_url() {
        let msg =
            "auth failed for key sk-abcd1234EFgh at https://api.upstream.com/v1 Bearer xyz123";
        let out = sanitise_forwarded_error_message(msg);
        assert!(!out.contains("sk-abcd1234EFgh"), "api key leaked: {out}");
        assert!(!out.contains("xyz123"), "bearer token leaked: {out}");
        assert!(
            !out.contains("https://api.upstream.com/v1"),
            "url leaked: {out}"
        );
        assert!(
            out.contains("[redacted]"),
            "api key must be redacted: {out}"
        );
        assert!(
            out.contains("[redacted-url]"),
            "url must be redacted: {out}"
        );
    }

    #[test]
    fn sanitise_caps_length() {
        let msg = "e".repeat(2000);
        let out = sanitise_forwarded_error_message(&msg);
        assert!(
            out.chars().count() <= 501,
            "message must be capped (got {}): {out}",
            out.chars().count()
        );
    }

    #[test]
    fn sanitise_preserves_plain_message() {
        let msg = "model is overloaded";
        let out = sanitise_forwarded_error_message(msg);
        assert_eq!(out, msg, "plain error text must pass through unchanged");
    }
}
