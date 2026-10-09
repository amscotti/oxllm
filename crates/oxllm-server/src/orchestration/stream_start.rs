//! Hold stream metadata until content or a normal terminator makes it safe to commit.

use std::pin::Pin;
use std::time::Duration;

use axum::body::Bytes;
use axum::http::HeaderMap;
use futures_util::{Stream, StreamExt, stream};
use oxllm_core::provider::ChatProvider;
use serde_json::Value;

use super::outcomes::{
    UpstreamError, is_content_policy_error, is_context_window_error, is_retryable_status,
};
use super::stream::{MAX_RAW_BUFFER_BYTES, append_and_decode_bedrock};
use crate::error::ApiError;
use crate::sse::{SseFramer, extract_sse_data, pop_sse_event};

pub(super) type ByteStream = Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>>;

pub(super) async fn prepare(
    response: reqwest::Response,
    chat: &dyn ChatProvider,
    is_bedrock: bool,
    request: &Value,
    timeout: Duration,
) -> Result<ByteStream, UpstreamError> {
    let headers = response.headers().clone();
    let mut bytes = Box::pin(response.bytes_stream());
    let mut replay = Vec::new();
    let mut retained = 0usize;
    let mut framer = SseFramer::new();
    let mut buffer = String::new();
    let mut raw_buffer = Vec::new();
    let mut events = 0;
    let mut state = None;

    loop {
        let next = tokio::time::timeout(timeout, bytes.next())
            .await
            .map_err(|_| {
                UpstreamError::Timeout("upstream stream timed out before content".into())
            })?;
        match next {
            Some(Ok(chunk)) => {
                retained = retained.saturating_add(chunk.len());
                if retained > MAX_RAW_BUFFER_BYTES {
                    return Err(UpstreamError::Retryable {
                        status: 502,
                        message: "upstream stream prelude exceeds the buffer limit".into(),
                        headers: Some(headers),
                    });
                }
                if is_bedrock {
                    append_and_decode_bedrock(&mut raw_buffer, &mut buffer, &chunk, &mut events)
                        .map_err(|error| classify(error, &headers))?;
                } else {
                    buffer.push_str(&framer.push(&chunk));
                }
                // Coalesce bytes so tiny metadata chunks cannot grow an
                // unbounded list of allocation/stream-item bookkeeping.
                replay.extend_from_slice(&chunk);
                while let Some(event) = pop_sse_event(&mut buffer) {
                    if ready(&event, chat, &mut state, request, &headers)? {
                        return Ok(Box::pin(
                            stream::iter([Ok(Bytes::from(replay))]).chain(bytes),
                        ));
                    }
                }
            }
            Some(Err(error)) => {
                return Err(UpstreamError::Network(error.without_url().to_string()));
            }
            None => {
                if !is_bedrock {
                    buffer.push_str(&framer.finish().map_err(|_| {
                        UpstreamError::Network("upstream ended with incomplete UTF-8".into())
                    })?);
                }
                if !buffer.trim().is_empty() && ready(&buffer, chat, &mut state, request, &headers)?
                {
                    return Ok(Box::pin(
                        stream::iter([Ok(Bytes::from(replay))]).chain(bytes),
                    ));
                }
                return Err(UpstreamError::Network(
                    "upstream stream ended before content".into(),
                ));
            }
        }
    }
}

fn ready(
    event: &str,
    chat: &dyn ChatProvider,
    state: &mut Option<Box<dyn std::any::Any + Send + Sync>>,
    request: &Value,
    headers: &HeaderMap,
) -> Result<bool, UpstreamError> {
    let data = extract_sse_data(event);
    if data == Some("[DONE]") {
        return Ok(true);
    }
    let body = data.and_then(|data| serde_json::from_str::<Value>(data).ok());
    let value = chat
        .transform_stream_chunk_stateful(event, state, request)
        .map_err(|error| {
            let oxllm_core::Error::Upstream {
                mut status,
                message,
            } = error
            else {
                return classify(error, headers);
            };
            let message = crate::security::truncate_error_message(&message, 500);
            if let Some(body) = &body {
                if body.get("type").and_then(Value::as_str) == Some("error")
                    && body.pointer("/error/type").and_then(Value::as_str)
                        == Some("overloaded_error")
                {
                    status = 529;
                }
                if is_context_window_error(body) {
                    return UpstreamError::ContextWindow {
                        status,
                        message,
                        headers: Some(headers.clone()),
                    };
                }
                if is_content_policy_error(body) {
                    return UpstreamError::ContentPolicy {
                        status,
                        message,
                        headers: Some(headers.clone()),
                    };
                }
            }
            classify(oxllm_core::Error::Upstream { status, message }, headers)
        })?;
    Ok(value.as_ref().is_some_and(has_content_or_finish))
}

fn has_content_or_finish(value: &Value) -> bool {
    value
        .get("choices")
        .and_then(Value::as_array)
        .is_some_and(|choices| {
            choices.iter().any(|choice| {
                choice
                    .get("finish_reason")
                    .is_some_and(|reason| !reason.is_null())
                    || choice.get("delta").is_some_and(|delta| {
                        ["content", "reasoning_content", "reasoning", "refusal"]
                            .iter()
                            .any(|field| {
                                delta
                                    .get(*field)
                                    .and_then(Value::as_str)
                                    .is_some_and(|text| !text.is_empty())
                            })
                            || delta
                                .get("tool_calls")
                                .and_then(Value::as_array)
                                .is_some_and(|tools| !tools.is_empty())
                            || delta
                                .get("function_call")
                                .is_some_and(|call| !call.is_null())
                    })
            })
        })
}

fn classify(error: oxllm_core::Error, headers: &HeaderMap) -> UpstreamError {
    match error {
        oxllm_core::Error::Network(message) => UpstreamError::Network(message),
        oxllm_core::Error::Transform(detail) | oxllm_core::Error::Serialization(detail) => {
            tracing::warn!(%detail, "invalid upstream stream prelude");
            UpstreamError::Retryable {
                status: 502,
                message: "invalid upstream stream prelude".into(),
                headers: Some(headers.clone()),
            }
        }
        oxllm_core::Error::Upstream { status, message }
            if reqwest::StatusCode::from_u16(status).is_ok_and(is_retryable_status) =>
        {
            UpstreamError::Retryable {
                status,
                message,
                headers: Some(headers.clone()),
            }
        }
        error => {
            let error = ApiError::from(error);
            UpstreamError::Client {
                status: error.status.as_u16(),
                message: error.message,
                headers: Some(headers.clone()),
            }
        }
    }
}
