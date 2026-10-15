//! `POST /v1/responses` — OpenAI Responses API.
//!
//! Accepts Responses API requests (`input` + `instructions`), translates them
//! to chat completions format, routes through the standard pipeline, and
//! converts the chat response back to Responses format.
//!
//! This is a thin handler: all API-shape conversion lives in
//! `crate::handlers::convert::responses`.

use axum::Extension;
use axum::Json;
use axum::body::Bytes;
use axum::extract::State;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use serde_json::Value;
use tokio::sync::mpsc;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;

use crate::auth::AuthUser;
use crate::error::ApiError;
use crate::handlers::chat::chat_completions_inner;
use crate::handlers::convert::responses::{
    ResponsesStreamConverter, chat_to_responses_response, process_openai_sse_event_responses,
    responses_to_chat_request,
};
use crate::orchestration::stream::send_converted_event;
use crate::pipeline::preserved_response_headers;
use crate::sse::{SseFramer, pop_sse_event};
use crate::state::SharedState;

/// Whether the Responses request body carries a usable `input` field. The
/// Responses API requires `input` as a string or a non-empty array; an absent,
/// null, or empty-array `input` would otherwise produce an empty `messages`
/// array and a chat-shaped "'messages'" 400 from the downstream layer.
fn responses_input_is_present(body: &Value) -> bool {
    match body.get("input") {
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Array(a)) => !a.is_empty(),
        _ => false,
    }
}

/// Responses API handler.
///
/// Converts the incoming Responses request to chat completions format,
/// forwards through the chat pipeline, then converts the chat response
/// back to Responses API format.
pub async fn responses(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    headers: axum::http::HeaderMap,
    body: Bytes,
) -> Result<Response, ApiError> {
    let state = shared.load_full();
    let request_body: Value = serde_json::from_slice(&body)
        .map_err(|e| ApiError::bad_request(format!("invalid JSON body: {e}")))?;

    crate::security::check_json_depth(&request_body, crate::security::MAX_JSON_DEPTH)?;
    crate::orchestration::request_controls::validate(&request_body)?;

    let model = request_body
        .get("model")
        .and_then(Value::as_str)
        .filter(|model| !model.trim().is_empty())
        .ok_or_else(|| ApiError::bad_request("missing required field: 'model'"))?;
    crate::pipeline::check_request_model_acl(&state, &auth, model, &request_body, &headers).await?;

    // `input` is
    // required by the Responses API — without it the converter produces an
    // empty `messages` array and the chat layer rejects with a confusing
    // "'messages'" error referencing a field the caller never sent.
    if !responses_input_is_present(&request_body) {
        return Err(ApiError::bad_request("missing required field: 'input'"));
    }

    if let Some(prev_id) = request_body.get("previous_response_id") {
        if !prev_id.is_null() {
            return Err(ApiError::bad_request(
                "previous_response_id is not supported: server-side response storage is not configured",
            ));
        }
    }

    let is_stream = request_body
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let chat_body = match responses_to_chat_request(&request_body) {
        Ok(body) => body,
        Err(error) => {
            let error = ApiError::from(error);
            crate::pipeline::record_http_request_rejection(
                &state,
                &auth,
                model,
                &request_body,
                &headers,
                &error,
            )
            .await;
            return Err(error);
        }
    };
    let chat_bytes = serde_json::to_vec(&chat_body)
        .map_err(|e| ApiError::internal(format!("failed to serialize request: {e}")))?;

    let chat_response =
        chat_completions_inner(state, &auth, &headers, Bytes::from(chat_bytes)).await?;

    if !chat_response.status().is_success() {
        return Ok(chat_response);
    }

    if is_stream {
        convert_stream_to_responses(chat_response).await
    } else {
        convert_json_to_responses(chat_response).await
    }
}

async fn convert_json_to_responses(chat_response: Response) -> Result<Response, ApiError> {
    let preserved_headers = preserved_response_headers(chat_response.headers());
    let body_bytes = axum::body::to_bytes(
        chat_response.into_body(),
        crate::security::MAX_RESPONSE_BODY_BYTES,
    )
    .await
    .map_err(|e| ApiError::internal(format!("failed to read response body: {e}")))?;
    let chat_resp: Value = serde_json::from_slice(&body_bytes)
        .map_err(|e| ApiError::internal(format!("failed to parse response JSON: {e}")))?;
    let responses_resp = chat_to_responses_response(&chat_resp);
    let mut response = Json(responses_resp).into_response();
    for (name, value) in preserved_headers {
        response.headers_mut().insert(name, value);
    }
    Ok(response)
}

async fn convert_stream_to_responses(chat_response: Response) -> Result<Response, ApiError> {
    let preserved_headers = preserved_response_headers(chat_response.headers());

    let body = chat_response.into_body();
    let data_stream = body.into_data_stream();

    let (tx, rx) = mpsc::channel::<Result<Event, std::convert::Infallible>>(64);

    tokio::spawn(async move {
        let mut converter = ResponsesStreamConverter::new();
        let mut framer = SseFramer::new();
        let mut buffer = String::new();
        let mut stream = data_stream;

        loop {
            tokio::select! {
            _ = tx.closed() => return,
            // The upstream pump owns the deployment-specific idle timeout.
            // A second fixed 60s deadline here would cut off valid long-idle
            // streams even when the operator configured a larger allowance.
            result = stream.next() => {
            match result {
            Some(Ok(bytes)) => {
            buffer.push_str(&framer.push(&bytes));
            while let Some(event_str) = pop_sse_event(&mut buffer) {
            for (event_type, event_value) in process_openai_sse_event_responses(&mut converter, &event_str) {
            let sse_event = Event::default()
            .event(&event_type)
            .data(event_value.to_string());
            if send_converted_event(&tx, sse_event).await.is_err() {
            return;
            }
            }
            }
            if buffer.len() > crate::handlers::chat::MAX_STREAM_BUFFER_BYTES {
            tracing::warn!(
            size = buffer.len(),
            "stream buffer exceeded cap without a complete SSE event; flushing best-effort"
            );
            for (event_type, event_value) in process_openai_sse_event_responses(&mut converter, &buffer) {
            let sse_event = Event::default()
            .event(&event_type)
            .data(event_value.to_string());
            if send_converted_event(&tx, sse_event).await.is_err() {
            return;
            }
            }
            buffer.clear();
            }
            }
            Some(Err(e)) => {
            tracing::warn!(error = %e, "stream read error");
            // Close any started output items before the terminal error so the
            // item lifecycle stays balanced (every `response.output_item.added`
            // pairs with a `response.output_item.done`), mirroring the
            // chunk-level error arm in `convert_chunk` and the Anthropic
            // handler's stream-read-error path.
            let mut events = converter.close_open_items();
            events.extend(converter.convert_chunk(&serde_json::json!({
                "error":{"type":"api_error","message":"stream read error"}
            })));
            for (event_type, event_value) in events {
            let sse_event = Event::default()
            .event(&event_type)
            .data(event_value.to_string());
            if send_converted_event(&tx, sse_event).await.is_err() {
            return;
            }
            }
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
            for (event_type, event_value) in process_openai_sse_event_responses(&mut converter, &buffer) {
            let sse_event = Event::default()
            .event(&event_type)
            .data(event_value.to_string());
            if send_converted_event(&tx, sse_event).await.is_err() {
            return;
            }
            }
            }
            for (event_type, event_value) in converter.finish_events() {
            let sse_event = Event::default()
            .event(&event_type)
            .data(event_value.to_string());
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handlers::chat::MAX_STREAM_BUFFER_BYTES;
    use serde_json::json;

    #[tokio::test(start_paused = true)]
    async fn responses_converter_does_not_override_upstream_idle_budget() {
        let source = futures_util::stream::once(async {
            tokio::time::sleep(std::time::Duration::from_secs(120)).await;
            Ok::<Bytes, std::io::Error>(Bytes::from_static(
                b"data: {\"id\":\"x\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"late\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"
            ))
        });
        let response =
            convert_stream_to_responses(Response::new(axum::body::Body::from_stream(source)))
                .await
                .unwrap();
        let body = axum::body::to_bytes(response.into_body(), 1024 * 1024)
            .await
            .unwrap();
        let text = String::from_utf8_lossy(&body);
        assert!(text.contains("response.completed"), "{text}");
        assert!(text.contains("late"), "{text}");
        assert!(!text.contains("stream idle timeout"), "{text}");
    }

    #[tokio::test(start_paused = true)]
    async fn stalled_responses_client_releases_the_input_stream() {
        let (tx, rx) = mpsc::channel::<Result<Bytes, std::io::Error>>(1);
        let producer = tokio::spawn(async move {
            for _ in 0..256 {
                if tx.send(Ok(Bytes::from_static(
                    b"data: {\"id\":\"x\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"x\"},\"finish_reason\":null}]}\n\n"
                ))).await.is_err() { return true; }
            }
            false
        });
        let response = convert_stream_to_responses(Response::new(axum::body::Body::from_stream(
            ReceiverStream::new(rx),
        )))
        .await
        .unwrap();
        assert!(
            tokio::time::timeout(std::time::Duration::from_secs(61), producer)
                .await
                .expect("converter must abandon a full channel")
                .unwrap()
        );
        drop(response);
    }

    /// When the upstream emits a chunk larger than `MAX_STREAM_BUFFER_BYTES`
    /// with no complete SSE event (no `\n\n`), the buffer overflow path must
    /// flush/clear it rather than accumulate forever — and the converter must
    /// still process subsequent well-formed events normally.
    #[tokio::test]
    async fn stream_buffer_overflow_flushes_and_recovers() {
        let oversized = format!(
            "data: {{\"junk\":\"{}",
            "x".repeat(MAX_STREAM_BUFFER_BYTES + 1)
        );
        let valid = "data: {\"id\":\"chatcmpl-1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}\n\ndata: [DONE]\n\n";

        let stream = futures_util::stream::iter([
            Ok::<Bytes, std::io::Error>(Bytes::from(oversized)),
            Ok(Bytes::from(valid)),
        ]);
        let response = Response::new(axum::body::Body::from_stream(stream));
        let out = convert_stream_to_responses(response)
            .await
            .expect("convert");
        let body = axum::body::to_bytes(out.into_body(), 16 * 1024 * 1024)
            .await
            .expect("body");
        let text = String::from_utf8_lossy(&body);
        assert!(
            text.contains("response.created"),
            "after an oversized buffer the converter must still emit response.created: {text}"
        );
        assert!(
            text.contains("response.completed"),
            "the stream must complete normally after the overflow flush: {text}"
        );
        assert!(
            !text.contains("junk"),
            "the oversized partial payload must not leak into any emitted event: {text}"
        );
    }

    /// A stream read error on the Responses path must forward a terminal
    /// `error` event (M5 parity with messages.rs / completions.rs).
    #[tokio::test]
    async fn stream_read_error_forwards_terminal_error_event() {
        let chunk = "data: {\"id\":\"chatcmpl-1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\n";
        let error_stream = futures_util::stream::iter([
            Ok::<Bytes, std::io::Error>(Bytes::from(chunk)),
            Err(std::io::Error::other("upstream gone")),
        ]);
        let response = Response::new(axum::body::Body::from_stream(error_stream));
        let out = convert_stream_to_responses(response)
            .await
            .expect("convert");
        let body = axum::body::to_bytes(out.into_body(), 1024 * 1024)
            .await
            .expect("body");
        let text = String::from_utf8_lossy(&body);
        assert!(
            text.contains("stream read error"),
            "stream read error must forward a terminal error event: {text}"
        );
        assert!(
            text.contains(r#""type":"error""#),
            "error event must carry type=error: {text}"
        );
        let events: Vec<serde_json::Value> = text
            .lines()
            .filter_map(|line| serde_json::from_str(line.strip_prefix("data: ")?).ok())
            .collect();
        let error = events
            .iter()
            .find(|event| event["type"] == "error")
            .expect("terminal error");
        assert_eq!(error["error"]["type"], "api_error");
        let sequence: Vec<u64> = events
            .iter()
            .map(|event| {
                event["sequence_number"]
                    .as_u64()
                    .expect("every event has a sequence number")
            })
            .collect();
        assert!(sequence.windows(2).all(|pair| pair[0] < pair[1]));
    }

    /// The non-streaming conversion must carry the orchestrator-stamped
    /// `x-oxllm-*`, `x-ratelimit-*`, and `x-request-id` headers onto the
    /// Responses-shaped response. Mirrors the parity tests in completions.rs
    /// and messages.rs.
    #[tokio::test]
    async fn convert_json_preserves_transparency_headers() {
        let body = Bytes::from(
            r#"{"id":"chatcmpl-1","object":"chat.completion","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}]}"#,
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
        src.headers_mut()
            .insert(crate::pipeline::CACHE_HIT_HEADER, "false".parse().unwrap());
        src.headers_mut().insert(
            crate::orchestration::ROUTER_STRATEGY_HEADER,
            "simple-shuffle".parse().unwrap(),
        );
        src.headers_mut().insert(
            crate::orchestration::ROUTER_DECISION_HEADER,
            "deployment-A".parse().unwrap(),
        );
        src.headers_mut().insert(
            crate::pipeline::RATELIMIT_REMAINING_REQUESTS,
            "99".parse().unwrap(),
        );

        let out = convert_json_to_responses(src).await.expect("convert");
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
                .get(crate::pipeline::CACHE_HIT_HEADER)
                .and_then(|v| v.to_str().ok()),
            Some("false")
        );
        assert_eq!(
            headers
                .get(crate::orchestration::ROUTER_STRATEGY_HEADER)
                .and_then(|v| v.to_str().ok()),
            Some("simple-shuffle")
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
            Some("99")
        );

        let body = axum::body::to_bytes(out.into_body(), 1024 * 1024)
            .await
            .expect("body");
        let responses: Value = serde_json::from_slice(&body).expect("json");
        assert_eq!(responses["object"], "response");
        assert_eq!(responses["output"][0]["content"][0]["text"], "hi");
    }

    /// Regression pin: the streaming conversion must preserve the same header
    /// families as the non-streaming path so the two paths stay consistent.
    #[tokio::test]
    async fn convert_stream_preserves_transparency_headers() {
        let chunk = "data: {\"id\":\"chatcmpl-1\",\"object\":\"chat.completion.chunk\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\n";
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
        let out = convert_stream_to_responses(src).await.expect("convert");
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
    /// 10 MiB magic cap (now aligned with `MAX_RESPONSE_BODY_BYTES` = 50 MiB).
    /// An 11 MiB body would previously fail with a generic
    /// "failed to read response body" internal error.
    #[tokio::test]
    async fn convert_json_reads_body_above_old_ten_mib_cap() {
        let padding = "a".repeat(11 * 1024 * 1024);
        let chat = serde_json::json!({
            "id": "chatcmpl-big",
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
        let out = convert_json_to_responses(response)
            .await
            .expect("11 MiB body must succeed under the 50 MiB cap");
        let body_bytes = axum::body::to_bytes(out.into_body(), 60 * 1024 * 1024)
            .await
            .expect("body");
        let responses: Value = serde_json::from_slice(&body_bytes).expect("json");
        assert_eq!(responses["object"], "response");
        assert_eq!(
            responses["output"][0]["content"][0]["text"]
                .as_str()
                .unwrap()
                .len(),
            11 * 1024 * 1024
        );
    }

    /// C19 LOW: a Responses request without a usable `input` field must be
    /// detected by the handler so the caller receives a Responses-shaped 400
    /// naming `input`, rather than a confusing chat-shaped "'messages'" error
    /// from the downstream layer.
    #[test]
    fn responses_input_validation_detects_missing_input() {
        assert!(
            !responses_input_is_present(&json!({"model": "gpt-4o"})),
            "absent input must be rejected"
        );
        assert!(
            !responses_input_is_present(&json!({"model": "gpt-4o", "input": null})),
            "null input must be rejected"
        );
        assert!(
            !responses_input_is_present(&json!({"model": "gpt-4o", "input": []})),
            "empty array input must be rejected"
        );
        assert!(
            !responses_input_is_present(&json!({"model": "gpt-4o", "input": 42})),
            "non-string/array input must be rejected"
        );
        assert!(
            responses_input_is_present(&json!({"model": "gpt-4o", "input": "Hi"})),
            "string input must be accepted"
        );
        assert!(
            responses_input_is_present(&json!({
                "model": "gpt-4o",
                "input": [{"role": "user", "content": "Hi"}]
            })),
            "non-empty array input must be accepted"
        );
    }

    /// C19 LOW: an empty-string `input: ""` must be rejected so the caller
    /// gets a clear Responses-shaped 400 instead of a semantically meaningless
    /// request forwarded to the upstream provider.
    #[test]
    fn responses_empty_string_input_is_rejected() {
        assert!(
            !responses_input_is_present(&json!({"model": "gpt-4o", "input": ""})),
            "empty string input must be rejected"
        );
    }
}
