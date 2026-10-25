//! Streaming response handling for the chat orchestrator.
//!
//! Owns the SSE pump that reads from an upstream `reqwest::Response`,
//! transforms each SSE event through the provider's [`ChatProvider`]
//! implementation, forwards the transformed chunks to the client over a
//! channel, accumulates token usage across the stream, and runs the
//! post-success / post-failure hook chain when the stream terminates.
//!
//! # Safety invariant: no fallback after the first stream byte
//!
//! Stream startup buffers metadata until content or a normal terminator arrives.
//! Once [`super::chat::execute_request_chain`] returns a streaming
//! [`Response`] to the client, the retry/fallback machinery has committed
//! to this upstream. Any error that occurs **after** the first SSE
//! event has been forwarded to the client is surfaced as a truncated stream
//! (terminated with `[DONE]`); it is **never** retried against another model.

use std::sync::Arc;
use std::time::Duration;

use axum::response::sse::Event;
use axum::response::{IntoResponse, Response};
use oxllm_core::CallContext;
use oxllm_core::deployment::Deployment;
use oxllm_core::router::deployment_key;
use tokio::sync::mpsc;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;

use super::MAX_STREAM_BUFFER_BYTES;
use super::chat::{HEALTH_CHECK_UNHEALTHY_TOTAL, allowed_fails_limit_reached};
use crate::error::ApiError;
use crate::orchestration::usage::{extract_usage, merge_streaming_usage};
use crate::pipeline;
use crate::spend_writer::tags_json;
use crate::sse::{SseFramer, extract_sse_data, pop_sse_event};
use crate::state::AppState;

/// Default stream idle timeout if none configured.
pub(crate) const DEFAULT_STREAM_TIMEOUT_SECS: u64 = 60;

/// Provider-idle timeout used to distinguish a stalled stream from a
/// connection-health failure. The client sees a 504 envelope containing
/// "timeout", but the deployment is not cooled for slow generation.
const STREAM_IDLE_TIMEOUT_MESSAGE: &str = "upstream stream idle timeout";

/// Cap on how long the pump waits for the client to drain a full SSE
/// channel before abandoning the stream. A client that keeps the
/// connection open but stops reading backfills the 64-event channel and
/// would otherwise park the pump on `tx.send` forever — the idle timeout
/// can never fire, the upstream connection, in-flight slot, and budget
/// reservation stay pinned. On stall the stream is dropped with
/// client-disconnect semantics (the client is at fault; the debit stands).
const CLIENT_SEND_STALL_TIMEOUT: Duration = Duration::from_secs(60);

/// Cap on the accumulated completion text used for zero-usage estimation,
/// bounding the memory the estimator retains for pathological streams.
const MAX_ESTIMATED_COMPLETION_BYTES: usize = 4 * 1024 * 1024;

/// Outcome of a client-facing send, bounded by
/// [`CLIENT_SEND_STALL_TIMEOUT`].
enum ClientSend {
    Sent,
    /// The client went away (channel closed).
    Closed,
    /// The client stopped reading for longer than the stall timeout.
    Stalled,
}

async fn send_event(
    tx: &mpsc::Sender<Result<Event, std::convert::Infallible>>,
    event: Event,
) -> ClientSend {
    match tokio::time::timeout(CLIENT_SEND_STALL_TIMEOUT, tx.send(Ok(event))).await {
        Ok(Ok(())) => ClientSend::Sent,
        Ok(Err(_)) => ClientSend::Closed,
        Err(_) => ClientSend::Stalled,
    }
}

/// Converted API streams share the pump's client-stall deadline. A stalled
/// converter drops its input body, propagating disconnect to upstream billing.
pub(crate) async fn send_converted_event(
    tx: &mpsc::Sender<Result<Event, std::convert::Infallible>>,
    event: Event,
) -> Result<(), ()> {
    match send_event(tx, event).await {
        ClientSend::Sent => Ok(()),
        ClientSend::Closed | ClientSend::Stalled => Err(()),
    }
}

/// Whether a transformed chunk is the OpenAI usage-only final chunk:
/// empty `choices` array plus a `usage` object. Delivered by compliant
/// upstreams when we injected `stream_options.include_usage`.
fn is_usage_only_chunk(value: &serde_json::Value) -> bool {
    let empty_choices = value
        .get("choices")
        .and_then(serde_json::Value::as_array)
        .is_some_and(std::vec::Vec::is_empty);
    empty_choices && extract_usage(value).is_some()
}

/// Usage is metered internally on every chunk, but clients must see it only in
/// the single terminal usage chunk. Content/finish chunks never carry usage.
fn strip_usage_from_content_chunk(value: &mut serde_json::Value) {
    if !is_usage_only_chunk(value)
        && let Some(object) = value.as_object_mut()
    {
        object.remove("usage");
    }
}

/// Append a transformed chunk's `delta.content` fragments to the
/// accumulated completion text (capped), used to estimate tokens for
/// providers that never deliver a usage chunk.
fn accumulate_completion_text(completion_text: &mut String, value: &serde_json::Value) {
    if completion_text.len() >= MAX_ESTIMATED_COMPLETION_BYTES {
        return;
    }
    let Some(choices) = value.get("choices").and_then(serde_json::Value::as_array) else {
        return;
    };
    for choice in choices {
        for pointer in [
            "/delta/content",
            "/delta/reasoning_content",
            "/delta/reasoning",
            "/delta/function_call/arguments",
        ] {
            if let Some(text) = choice.pointer(pointer).and_then(serde_json::Value::as_str) {
                append_estimated_text(completion_text, text);
            }
        }
        if let Some(tools) = choice
            .pointer("/delta/tool_calls")
            .and_then(serde_json::Value::as_array)
        {
            for tool in tools {
                if let Some(text) = tool
                    .pointer("/function/arguments")
                    .and_then(serde_json::Value::as_str)
                {
                    append_estimated_text(completion_text, text);
                }
            }
        }
    }
}

fn append_estimated_text(accumulated: &mut String, text: &str) {
    let mut end = text
        .len()
        .min(MAX_ESTIMATED_COMPLETION_BYTES.saturating_sub(accumulated.len()));
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    accumulated.push_str(&text[..end]);
}

/// Estimate stream usage when the upstream finished without ever
/// delivering a usage chunk (providers that ignore
/// `stream_options.include_usage`): prompt tokens from the pre-call
/// estimate on the call context (or a fresh request estimate), completion
/// tokens tokenized from the accumulated delta text.
///
/// Mirrors the reference `stream_chunk_builder` fallback
/// (`main.py:8511`, `ChunkProcessor.calculate_usage`), which
/// always bills something for a completed stream instead of letting a
/// zero-usage stream be free to the caller.
fn estimate_stream_usage(
    call_ctx: &oxllm_core::CallContext,
    response_model: &str,
    completion_text: &str,
) -> Option<oxllm_core::UsageSummary> {
    let prompt_tokens = call_ctx
        .token_estimate
        .get()
        .copied()
        .unwrap_or_else(|| oxllm_core::tokens::estimate_request_tokens(&call_ctx.request_body));
    let completion_tokens = oxllm_core::tokens::estimate_tokens(completion_text, response_model);
    if prompt_tokens == 0 && completion_tokens == 0 {
        return None;
    }
    Some(oxllm_core::UsageSummary {
        prompt_tokens,
        completion_tokens,
        total_tokens: prompt_tokens.saturating_add(completion_tokens),
        ..Default::default()
    })
}

/// Fill in usage missing from an incomplete stream. A prompt-only snapshot
/// must not suppress completion estimation: reported prompt/cache counters are
/// preserved while already-received text supplies the missing output count.
fn estimate_incomplete_stream_usage(
    accumulated_usage: &mut Option<oxllm_core::UsageSummary>,
    call_ctx: &oxllm_core::CallContext,
    response_model: &str,
    completion_text: &str,
) {
    if accumulated_usage.is_none() {
        *accumulated_usage = estimate_stream_usage(call_ctx, response_model, completion_text);
    }
    let Some(usage) = accumulated_usage.as_mut() else {
        return;
    };
    if usage.completion_tokens != 0 || completion_text.is_empty() {
        return;
    }
    let completion_tokens = oxllm_core::tokens::estimate_tokens(completion_text, response_model);
    if completion_tokens != 0 {
        usage.completion_tokens = completion_tokens;
        usage.total_tokens = usage.prompt_tokens.saturating_add(completion_tokens);
    }
}

/// Maximum bytes the Bedrock streaming path will retain in `raw_buffer`
/// between chunk arrivals without a complete eventstream frame. AWS
/// eventstream frames declare their `total_len` as a 4-byte big-endian
/// header; a malicious or buggy upstream could declare `total_len = u32::MAX`
/// (~4 GB) and trickle bytes faster than the idle timeout, growing
/// `raw_buffer` without bound (the frame never completes, so it is never
/// drained into the text `buffer` and the `MAX_STREAM_BUFFER_BYTES` text cap
/// never engages). 16 MB is well above any legitimate AWS eventstream frame
/// and bounds the slowloris-style memory exhaustion vector.
pub(crate) const MAX_RAW_BUFFER_BYTES: usize = 16 * 1024 * 1024;

/// How the stream pump's read loop terminated. Extracted to module scope so
/// [`finalize_stream_exit`] — the accounting for each exit — is directly
/// callable from tests (the pump itself lives in a fire-and-forget
/// `tokio::spawn` with no join handle).
#[derive(Debug)]
pub(crate) enum StreamExit {
    Done,
    /// The local read-idle deadline expired. This is distinct from a
    /// provider-supplied 504 so only genuine local stalls skip health cooldown.
    IdleTimeout,
    Error(oxllm_core::Error),
    /// The HTTP client went away mid-stream (dropped the response body
    /// reader, closed the connection, etc.). Upstream tokens were very
    /// likely already consumed on our side, so the budget reservation
    /// made in `pre_call` must be kept as the debit — calling
    /// `post_failure` would release it and make partial streams "free"
    /// (Bug #33).
    ClientDisconnected,
}

/// Handle a streaming success response (2xx from upstream).
///
/// Applies `stream_timeout` as the max idle time between chunks, and aborts
/// the upstream stream when the client disconnects.
///
/// Token usage is accumulated across stream chunks (notably the final chunk
/// that carries a `usage` object). When the stream completes normally, the
/// post-success hook chain fires with the aggregated usage so that budget
/// debit, rate-limit increment, cost log, and cache store all run — matching
/// the non-streaming path.
///
/// `budget_reservation` is the router-budget reservation made at pick time;
/// it is settled to the actual cost when the stream completes normally, to
/// the partial cost when the stream terminates early with usage on record,
/// and released when the stream errors before any usage was accumulated.
///
/// `model` is the requested model group name, used for cooldown-duration
/// resolution when a mid-stream failure accounts against the deployment.
/// The in-flight slot the chat loop reserved is held for the whole stream
/// lifetime (`LeastBusy` counts active streams) and released here on every
/// exit; mid-stream failures are accounted against the stable per-deployment
/// key ([`crate::router::Router::cooldown_key`]) for allowed-fails /
/// percentage cooldown / health.
pub(crate) async fn handle_stream_success(
    upstream_response: super::stream_start::ByteStream,
    deployment: Deployment,
    provider: Arc<dyn oxllm_core::Provider>,
    state: AppState,
    mut call_ctx: CallContext,
    budget_reservation: super::budget::RouterBudgetReservation,
    model: String,
) -> Result<Response, ApiError> {
    // Reuse the request/deployment idle budget that bounded the header wait,
    // now applied per chunk without a total stream deadline.
    let stream_timeout = super::upstream::resolve_stream_idle_timeout_for_request(
        &deployment,
        &state.config,
        &call_ctx.request_body,
    );
    // The stable per-deployment key: the same identity the chat loop reserved
    // its in-flight slot under and that cooldown/health/failure accounting use.
    let cooldown_key = state.router.cooldown_key(&deployment);
    // Non-secret `{model_name}:{provider_model}` identity for metric labels:
    // the cooldown key embeds a digest of credential material and must not
    // surface on `/metrics`, which ordinary virtual-key holders can scrape.
    let dep_key = oxllm_core::router::deployment_key(&deployment);

    let (tx, rx) = mpsc::channel::<Result<Event, std::convert::Infallible>>(64);

    tokio::spawn(async move {
        let Some(chat) = provider.chat() else {
            tracing::error!(provider = ?provider.id(), "provider lost chat capability mid-stream");
            // Release the router-budget reservation made at pick time —
            // returning without settle/release would leak it for the whole
            // budget window (TTL), same as any other failure branch.
            budget_reservation.release(&state.cache).await;
            // The chat loop kept this stream's in-flight slot reserved
            // (streams hold it until the task exits); release it here.
            state.router.record_in_flight(&cooldown_key, -1);
            return;
        };
        let is_bedrock = provider.id() == "bedrock";
        let mut byte_stream = upstream_response;
        let mut framer = SseFramer::new();
        let mut buffer = String::new();
        let mut raw_buffer: Vec<u8> = Vec::new();
        let mut bedrock_events: u64 = 0;
        let mut accumulated_usage: Option<oxllm_core::UsageSummary> = None;
        let mut response_model = deployment.model_name.clone();
        let mut bytes_sent_to_client = false;
        // Accumulated `delta.content` text, used to estimate completion
        // tokens when the upstream never delivers a usage chunk.
        let mut completion_text = String::new();
        // Set when a client-facing send stalls past the stall timeout; the
        // select loop breaks with client-disconnect semantics.
        let mut client_stalled = false;
        // H14: stream_options.include_usage. When the client requests it,
        // synthesize a final usage chunk (empty choices + usage) from the
        // accumulated usage, emitted once right before the `[DONE]` marker.
        let include_usage = call_ctx
            .request_body
            .get("stream_options")
            .and_then(|s| s.get("include_usage"))
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let mut usage_injection = UsageInjection {
            include_usage,
            public_model: Some(pipeline::public_response_model(&call_ctx).to_string()),
            emitted: false,
            reported_cost: None,
            served_service_tier: None,
            reported_usage: None,
            strip_content_usage: !call_ctx.preserve_stream_usage_for_conversion,
        };
        // Tracks whether the OpenAI `[DONE]` terminator has already been
        // emitted (either inline via `process_buffer` when the upstream
        // sends `[DONE]`, or in the post-loop drain below). Non-OpenAI
        // providers (Anthropic `message_stop`, Bedrock eventstream) never
        // send `[DONE]`, so after the loop we synthesize it unconditionally
        // when this flag is still false.
        let mut done_sent = false;

        // Per-stream provider state. Created here (host-owned) and threaded
        // through every chunk so it follows this task across worker-thread
        // migrations. `None` for providers whose default
        // `transform_stream_chunk_stateful` ignores it; Anthropic downcasts
        // it to its concrete accumulator.
        let mut stream_state: Option<Box<dyn std::any::Any + Send + Sync>> = None;
        let original_request = call_ctx.request_body.clone();

        let exit: StreamExit = loop {
            tokio::select! {
            _ = tx.closed() => {
            tracing::info!(
            "client disconnected during streaming, aborting upstream"
            );
            break StreamExit::ClientDisconnected;
            }
            result = tokio::time::timeout(
            stream_timeout,
            byte_stream.next()) => {
            match result {
            Ok(Some(Ok(bytes))) => {
            if is_bedrock {
                match append_and_decode_bedrock(
                    &mut raw_buffer,
                    &mut buffer,
                    &bytes,
                    &mut bedrock_events,
                ) {
                    Ok(()) => {}
                    Err(err) => break StreamExit::Error(err),
                }
            } else {
                buffer.push_str(&framer.push(&bytes));
            }
            // U22: OpenRouter emits the usage
            // chunk (carrying provider-reported cost) AFTER the
            // `finish_reason` chunk. oxllm is safe here: the stream pump
            // keeps calling `process_buffer` for every chunk until
            // `byte_stream.next()` returns `None` (the `Ok(None)` arm
            // above). There is no early-exit on `finish_reason`, so
            // post-finish usage chunks flow through `process_buffer` →
            // `extract_usage` → `merge_streaming_usage` and the
            // `accumulated_usage` (used by the post-success hook for cost
            // accounting) captures them. (The reference bug was a
            // `raise StopIteration` inside the chunk handler the moment
            // `finish_reason` appeared; oxllm has no such generator-style
            // early termination.)
            if let Err(err) = process_buffer(
            &mut buffer,
            chat,
            &tx,
            &mut accumulated_usage,
            &mut response_model,
            &mut usage_injection,
            &mut done_sent,
            &mut bytes_sent_to_client,
            &mut stream_state,
            &original_request,
            &mut completion_text,
            &mut client_stalled)
            .await
            {
                break StreamExit::Error(err);
            }
            if client_stalled {
            tracing::warn!(
            "client stopped reading the stream; aborting (stall timeout)"
            );
            break StreamExit::ClientDisconnected;
            }
            if done_sent {
            break StreamExit::Done;
            }
            if buffer.len() > MAX_STREAM_BUFFER_BYTES {
            // Bug #41: never forward the raw buffer as an SSE event — for
            // non-OpenAI upstreams this leaks provider-native partial JSON
            // (and even for OpenAI it sends malformed/incomplete chunks).
            // Drop the incomplete content and trust the upstream to re-send
            // a complete event; the client sees a gap rather than garbage.
            tracing::warn!(
            size = buffer.len(),
            "dropping oversized incomplete SSE event"
            );
            buffer.clear();
            }
            }
            Ok(Some(Err(e))) => {
            // A transport error AFTER the OpenAI `[DONE]` terminator is a
            // clean stream (the provider already finished; some aggregators
            // reset the socket immediately). Before `[DONE]`, a reset is
            // still a real failure (U8).
            break stream_exit_on_upstream_read_error(done_sent, e);
            }
            Ok(None) => {
            if is_bedrock {
                if let Some(err) = bedrock_stream_close_error(bedrock_events, !raw_buffer.is_empty())
                {
                    break StreamExit::Error(err);
                }
            }
            break StreamExit::Done;
            }
            Err(_elapsed) => {
            tracing::warn!(
            timeout_secs = stream_timeout.as_secs(),
            bytes_already_sent = bytes_sent_to_client,
            "stream idle timeout, aborting; no fallback after bytes sent"
            );
            break StreamExit::IdleTimeout;
            }
            }
            }
            }
        };

        let mut exit = exit;
        if matches!(&exit, StreamExit::Done) && !done_sent {
            match framer.finish() {
                Ok(remaining) => buffer.push_str(&remaining),
                Err(err) => {
                    exit = StreamExit::Error(oxllm_core::Error::Network(format!(
                        "invalid UTF-8 in upstream stream: {err}"
                    )));
                }
            }
        }
        if matches!(&exit, StreamExit::Done) && !done_sent && !buffer.trim().is_empty() {
            match chat.transform_stream_chunk_stateful(
                &buffer,
                &mut stream_state,
                &original_request,
            ) {
                Ok(Some(mut value)) => {
                    if let Some(u) = extract_usage(&value) {
                        merge_streaming_usage(&mut accumulated_usage, u);
                    }
                    capture_stream_usage_cost(&mut usage_injection, &value);
                    if let Some(m) = value.get("model").and_then(|v| v.as_str()) {
                        response_model = m.to_string();
                    }
                    accumulate_completion_text(&mut completion_text, &value);
                    normalize_stream_chunk(&mut value, &original_request, &usage_injection);
                    if usage_injection.strip_content_usage {
                        strip_usage_from_content_chunk(&mut value);
                    }
                    // The upstream's own usage-only chunk is forwarded only
                    // when the client asked for usage; when it is forwarded,
                    // `emitted` suppresses the synthesized duplicate.
                    let forward = !is_usage_only_chunk(&value)
                        || (usage_injection.include_usage && !usage_injection.emitted);
                    if forward {
                        if is_usage_only_chunk(&value) {
                            usage_injection.emitted = true;
                        }
                        match send_event(&tx, Event::default().data(value.to_string())).await {
                            ClientSend::Sent => bytes_sent_to_client = true,
                            ClientSend::Closed | ClientSend::Stalled => {}
                        }
                    }
                }
                Ok(None) => {
                    // M4: gate on `!done_sent` so a malformed SSE upstream that
                    // already triggered an inline `[DONE]` (via `process_buffer`,
                    // which sets `done_sent`) cannot force a SECOND terminator
                    // here. Without this guard, a leftover `[DONE]` fragment in
                    // the post-loop buffer would be emitted again, violating the
                    // OpenAI SSE contract (exactly one terminator per stream).
                    // Hard to unit test: the duplicate only surfaces from the
                    // interaction between the inline `process_buffer` path and
                    // this post-loop drain inside the spawned stream task.
                    if !done_sent && extract_sse_data(&buffer) == Some("[DONE]") {
                        // The helper reads `emitted` (set during the streaming
                        // loop) to avoid a second emission here. This terminal
                        // block runs at most once, so it does not need to flip
                        // the flag back.
                        if let Some(mut chunk) = build_usage_chunk_if_requested(
                            &usage_injection,
                            accumulated_usage.as_ref(),
                        ) {
                            normalize_stream_chunk(&mut chunk, &original_request, &usage_injection);
                            usage_injection.emitted = true;
                            if matches!(
                                send_event(&tx, Event::default().data(chunk.to_string())).await,
                                ClientSend::Sent
                            ) {
                                bytes_sent_to_client = true;
                            }
                        }
                        if matches!(
                            send_event(&tx, Event::default().data("[DONE]")).await,
                            ClientSend::Sent
                        ) {
                            bytes_sent_to_client = true;
                        }
                        done_sent = true;
                    }
                }
                Err(err) => {
                    tracing::warn!(
                    error = %err,
                    chunk_len = buffer.len(),
                    "stream chunk transform error (content redacted)"
                    );
                    exit = StreamExit::Error(err);
                }
            }
        }

        // Unconditional `[DONE]` for normal completion (OpenAI SSE contract).
        // Anthropic (`message_stop`), Bedrock (eventstream), and other
        // non-OpenAI providers never emit `[DONE]` upstream, so we synthesize
        // it here when the OpenAI inline path did not already do so. Guarded
        // by `done_sent` to avoid a duplicate terminator. The flag is set
        // after emission so a future second `[DONE]`-emitting block added
        // below this one cannot produce a duplicate — matching the pattern in
        // `process_buffer` and the post-loop `Ok(None)` drain.
        if matches!(&exit, StreamExit::Done) && !done_sent {
            if let Some(mut chunk) =
                build_usage_chunk_if_requested(&usage_injection, accumulated_usage.as_ref())
            {
                normalize_stream_chunk(&mut chunk, &original_request, &usage_injection);
                if matches!(
                    send_event(&tx, Event::default().data(chunk.to_string())).await,
                    ClientSend::Sent
                ) {
                    bytes_sent_to_client = true;
                }
            }
            if matches!(
                send_event(&tx, Event::default().data("[DONE]")).await,
                ClientSend::Sent
            ) {
                bytes_sent_to_client = true;
            }
            // Set for robustness against a future second `[DONE]`-emitting
            // block. No code reads the flag after this point (the task returns
            // once the `match exit` below completes), so the assignment would
            // otherwise be flagged `unused_assignments`; the allow keeps the
            // future-proofing intent explicit.
            #[allow(unused_assignments)]
            {
                done_sent = true;
            }
        }

        call_ctx.served_service_tier = usage_injection.served_service_tier;
        finalize_stream_exit(
            &state,
            &deployment,
            &call_ctx,
            &budget_reservation,
            &model,
            &cooldown_key,
            &dep_key,
            &tx,
            exit,
            &mut accumulated_usage,
            &response_model,
            bytes_sent_to_client,
            &completion_text,
            usage_injection.reported_cost,
        )
        .await;
    });

    Ok(axum::response::Sse::new(ReceiverStream::new(rx)).into_response())
}

/// Run the exit accounting for a terminated stream pump: in-flight release,
/// usage estimation, post-success/post-failure hooks, router-budget
/// settle/release, and (where a debit stands) the partial spend row.
///
/// Extracted from the fire-and-forget `tokio::spawn` in
/// [`handle_stream_success`] (which owns no join handle) so the disconnect /
/// error accounting is directly unit-testable.
#[expect(clippy::too_many_arguments)]
async fn finalize_stream_exit(
    state: &AppState,
    deployment: &Deployment,
    call_ctx: &CallContext,
    budget_reservation: &super::budget::RouterBudgetReservation,
    model: &str,
    cooldown_key: &str,
    dep_key: &str,
    tx: &mpsc::Sender<Result<Event, std::convert::Infallible>>,
    exit: StreamExit,
    accumulated_usage: &mut Option<oxllm_core::UsageSummary>,
    response_model: &str,
    bytes_sent_to_client: bool,
    completion_text: &str,
    reported_cost: Option<f64>,
) {
    let (exit, idle_timeout) = match exit {
        StreamExit::IdleTimeout => (
            StreamExit::Error(oxllm_core::Error::Upstream {
                status: 504,
                message: STREAM_IDLE_TIMEOUT_MESSAGE.to_string(),
            }),
            true,
        ),
        other => (other, false),
    };
    match exit {
        StreamExit::Done => {
            // Release the in-flight slot the chat loop reserved for the
            // whole stream lifetime (LeastBusy counts active streams).
            state.router.record_in_flight(cooldown_key, -1);
            // Zero-usage streams (providers that ignore the injected
            // `stream_options.include_usage`) must still be billed:
            // estimate from the pre-call prompt estimate and the
            // accumulated delta text, matching the reference
            // `stream_chunk_builder` fallback. Without this the
            // post-success budget settle treats the streamed tokens as
            // free to the caller.
            if bytes_sent_to_client {
                let missing_completion = accumulated_usage
                    .as_ref()
                    .is_none_or(|usage| usage.completion_tokens == 0);
                estimate_incomplete_stream_usage(
                    accumulated_usage,
                    call_ctx,
                    response_model,
                    completion_text,
                );
                if missing_completion && let Some(estimated) = accumulated_usage.as_ref() {
                    tracing::info!(
                        request_id = %call_ctx.request_id,
                        prompt_tokens = estimated.prompt_tokens,
                        completion_tokens = estimated.completion_tokens,
                        "stream usage was incomplete; billing estimated usage"
                    );
                }
            }
            // Aggregators report measured spend on late usage chunks, including
            // dynamic routing/search fees. Only that receipt is threaded through
            // the dedicated
            // `body_cost` field — NOT synthesized into a `{"usage":{"cost":…}}`
            // body, because `CacheHook::post_success` stores `body` verbatim
            // for cache-enabled requests and the next identical request
            // would short-circuit with that fragment instead of a chat
            // completion.
            let cost = pipeline::post_success(
                state,
                call_ctx,
                response_model,
                accumulated_usage.as_ref(),
                None,
                reported_cost,
                deployment.provider_slug(),
            )
            .await;
            // Router budget limiting: settle the deployment/provider
            // spend counters with the actual cost (reservation made at
            // pick time; a configured budget with unknown cost records
            // a nominal increment and logs a warning).
            super::budget::settle_router_spend(
                state,
                deployment,
                &deployment_key(deployment),
                cost,
                budget_reservation,
            )
            .await;
            if let Some(u) = accumulated_usage.as_ref() {
                pipeline::record_deployment_tpm(state, &deployment_key(deployment), u.total_tokens)
                    .await;
            }
            tracing::info!(
                request_id = %call_ctx.request_id,
                bytes_already_sent = bytes_sent_to_client,
                cost = ?cost,
                "stream completed normally"
            );
        }
        StreamExit::ClientDisconnected => {
            // Release the in-flight slot (the stream has ended).
            state.router.record_in_flight(cooldown_key, -1);
            if !bytes_sent_to_client && accumulated_usage.is_none() {
                // The client vanished BEFORE the first SSE event reached it
                // and no usage was on record: the provider almost certainly
                // generated nothing. The reference cancels the task on disconnect
                // and never completes spend tracking — nothing is charged —
                // so release the router reservation and run the failure
                // hooks (which release the key RPM/TPM/budget reservations)
                // instead of keeping the worst-case estimate as the debit.
                // No partial spend row: no debit stands, nothing to
                // reconcile. The keep-the-debit anti-abuse policy (Bug #33)
                // only applies once bytes (or usage) exist — see below.
                budget_reservation.release(&state.cache).await;
                pipeline::post_failure(
                    state,
                    call_ctx,
                    &oxllm_core::Error::Network(
                        "client disconnected before any stream bytes were sent".to_string(),
                    ),
                )
                .await;
                tracing::info!(
                    request_id = %call_ctx.request_id,
                    "client disconnected before the first byte; reservations released"
                );
                return;
            }
            // The client went away mid-stream. Estimate missing usage before
            // writing the partial row: the provider had already accepted the
            // prompt and may have generated the deltas sent to the client.
            if bytes_sent_to_client {
                estimate_incomplete_stream_usage(
                    accumulated_usage,
                    call_ctx,
                    response_model,
                    completion_text,
                );
            }
            // Keep the budget/RPM reservation as the debit (`post_failure`
            // would refund it and make disconnect-to-avoid-charge work) but
            // always release the no-TTL `max_parallel_requests` gauge so a
            // disconnect storm cannot lock the key until restart.
            pipeline::release_concurrency(state, call_ctx).await;
            // When `accumulated_usage` is available, account deployment
            // TPM and settle the router-level budget reservation to the
            // computed partial cost. When no usage was accumulated, the
            // reservation simply stands as the debit.
            //
            // Note: key-level budget accounting on these partial
            // streams is approximate by design and over-counts — the
            // key-level reservation is NOT released (it debits the
            // conservative full estimate) while the router-level
            // counters settle to the computed partial cost below, so
            // the two scopes can diverge.
            if let Some(u) = accumulated_usage.as_ref() {
                pipeline::record_deployment_tpm(state, &deployment_key(deployment), u.total_tokens)
                    .await;
            }
            // M5: same keep-the-debit policy for the router-level
            // deployment/provider spend counters — settle the
            // reservation to the partial cost instead of leaking it.
            // With no usage on record the reservation simply stands as
            // the debit, but a `status: "partial"` zero-usage spend row
            // is still queued so the DB spend ledger matches the budget
            // counters (reconciliation between `/spend/logs` and key
            // spend totals).
            record_partial_stream_spend(
                state,
                deployment,
                call_ctx,
                response_model,
                accumulated_usage.as_ref(),
                budget_reservation,
                reported_cost,
            )
            .await;
            tracing::info!(
                request_id = %call_ctx.request_id,
                bytes_already_sent = bytes_sent_to_client,
                prompt_tokens = accumulated_usage
                    .as_ref()
                    .map(|u| u.prompt_tokens)
                    .unwrap_or(0),
                "client disconnected mid-stream; budget reservation kept as debit"
            );
        }
        StreamExit::Error(err) => {
            // Release the in-flight slot (the stream has ended).
            state.router.record_in_flight(cooldown_key, -1);
            if idle_timeout && bytes_sent_to_client {
                estimate_incomplete_stream_usage(
                    accumulated_usage,
                    call_ctx,
                    response_model,
                    completion_text,
                );
            }
            let (status, message) = match &err {
                oxllm_core::Error::Upstream { status, message } => (*status, message.as_str()),
                _ => (500, "upstream stream failed"),
            };
            // Route the streaming error `code`/`type` through the same
            // helpers as the non-streaming `ApiError::upstream` path so a
            // client switching on `code` sees the same value for the same
            // upstream condition regardless of `stream: true` (previously
            // this path emitted OpenAI-native codes like
            // `rate_limit_exceeded` while the non-streaming path emitted
            // `upstream_rate_limited`). The status is synthesized first
            // (out-of-range → 502) so the derived code tracks what the
            // client would see on the non-streaming path. `code` is
            // omitted entirely when `None`, matching the non-streaming
            // envelope shape.
            let synthesized = crate::error::synthesize_upstream_status(status);
            // Build the error object via the shared `error_envelope_object`
            // helper (the single source of truth also used by the
            // non-streaming `ApiError::into_response`) so a client sees a
            // byte-identical envelope on both paths regardless of
            // `stream: true`. The status is synthesized first
            // (out-of-range -> 502) so the derived `type`/`code` track
            // what the client would see on the non-streaming path.
            let error_code = if idle_timeout {
                Some(crate::error::UPSTREAM_TIMEOUT_CODE)
            } else {
                crate::error::code_for_upstream_status(synthesized.as_u16())
            };
            let error_obj = crate::error::error_envelope_object(
                message,
                crate::error::error_type_for_status(synthesized.as_u16()),
                error_code,
            );
            let error_event = serde_json::json!({ "error": error_obj });
            match send_event(tx, Event::default().data(error_event.to_string())).await {
                ClientSend::Sent => {}
                // Client already gone/stalled: nothing more to deliver.
                ClientSend::Closed | ClientSend::Stalled => {}
            }
            let _ = send_event(tx, Event::default().data("[DONE]")).await;
            if let Some(u) = accumulated_usage.as_ref() {
                pipeline::record_deployment_tpm(state, &deployment_key(deployment), u.total_tokens)
                    .await;
            }
            // M5: usage accumulated before the error represents tokens
            // the upstream already generated and billed; settle the
            // router-level reservation to the partial cost instead of
            // leaking it. With no usage on record, release the
            // reservation — mirroring the key-level `post_failure`
            // below, which releases the key reservation.
            //
            // Intentional asymmetry with `StreamExit::ClientDisconnected`
            // (Bug #33): the two exit paths
            // treat the KEY-level budget reservation in opposite
            // directions, and the divergence is deliberate:
            //
            //   * Error path (here): the upstream failed mid-stream. That
            //     is the server's fault, so the caller is refunded —
            //     `pipeline::post_failure` below releases the FULL key
            //     reservation (the errored stream is free to the key).
            //     The operator still owes the provider for the tokens it
            //     already generated, so the router-level reservation
            //     settles to the PARTIAL cost above. Net: the caller pays
            //     nothing, the operator pays for what was generated.
            //
            //   * ClientDisconnected path: the client went away. That is
            //     the client's choice, so the key reservation is KEPT as
            //     the debit (full estimate) to prevent abuse (a client
            //     could otherwise disconnect early to avoid being
            //     charged). The router scope likewise settles to partial.
            //
            // The principle in both arms: tokens the provider already
            // generated are billed to the OPERATOR (router scope →
            // partial settle). The KEY scope differs by fault: upstream
            // fault → refund; client fault → charge.
            if accumulated_usage.is_some() || reported_cost.is_some() {
                record_partial_stream_spend(
                    state,
                    deployment,
                    call_ctx,
                    response_model,
                    accumulated_usage.as_ref(),
                    budget_reservation,
                    reported_cost,
                )
                .await;
            } else {
                budget_reservation.release(&state.cache).await;
            }
            // Router-side accounting for the mid-stream failure — the gap
            // that left deployments whose streams die mid-body permanently
            // healthy (the 2xx headers already recorded a connection
            // success, and nothing re-degraded them). Health is degraded
            // only by transport errors, mirroring the non-stream
            // `UpstreamError::Network` arm. Allowed-fails / percentage
            // cooldown are counted for transport-class errors only —
            // deterministic transform/parse errors would be reproduced by
            // a sibling and say nothing about the deployment's health.
            // Idle read timeouts are excluded below for the same reason:
            // slow generation is not evidence of a failed connection.
            if let oxllm_core::Error::Network(msg) = &err {
                if state.router.record_connection_failure(cooldown_key, msg) {
                    ::metrics::counter!(
                        HEALTH_CHECK_UNHEALTHY_TOTAL,
                        "deployment" => dep_key.to_string()
                    )
                    .increment(1);
                }
            }
            let fail_status: Option<u16> = if idle_timeout {
                None
            } else {
                match &err {
                    oxllm_core::Error::Network(_) => Some(0),
                    oxllm_core::Error::Upstream { status, .. } => Some(*status),
                    _ => None,
                }
            };
            if let Some(status) = fail_status
                && super::outcomes::terminal_failure_affects_deployment(state, status)
                && allowed_fails_limit_reached(state, cooldown_key, status, model)
            {
                state.router.mark_cooldown_for(cooldown_key, model);
            }
            tracing::info!(
                request_id = %call_ctx.request_id,
                error = %err,
                bytes_already_sent = bytes_sent_to_client,
                prompt_tokens = accumulated_usage
                    .as_ref()
                    .map(|u| u.prompt_tokens)
                    .unwrap_or(0),
                "stream ended with error; no fallback (bytes already sent: {})",
                bytes_sent_to_client
            );
            pipeline::post_failure(state, call_ctx, &err).await;
        }
        StreamExit::IdleTimeout => {
            unreachable!("idle timeouts are normalized before exit accounting")
        }
    }
}

/// Settle a byte-preserving native SSE relay using the same success, failure
/// and disconnect policy as transformed chat streams. Native wire bytes are
/// owned by the relay, so error events from this accounting helper are dropped.
#[expect(clippy::too_many_arguments)]
pub(crate) async fn finish_passthrough_stream(
    state: &AppState,
    deployment: &Deployment,
    ctx: &CallContext,
    reservation: &super::budget::RouterBudgetReservation,
    exit: StreamExit,
    usage: &mut Option<oxllm_core::UsageSummary>,
    response_model: &str,
    bytes_sent: bool,
    completion_text: &str,
    reported_cost: Option<f64>,
) {
    let (tx, rx) = mpsc::channel(1);
    drop(rx);
    finalize_stream_exit(
        state,
        deployment,
        ctx,
        reservation,
        &ctx.model,
        &state.router.cooldown_key(deployment),
        &deployment_key(deployment),
        &tx,
        exit,
        usage,
        response_model,
        bytes_sent,
        completion_text,
        reported_cost,
    )
    .await;
}

/// Record router-level deployment/provider spend for a stream that never
/// reached the post-success chain (client disconnect or mid-stream error).
///
/// Aligns with the deliberate keep-the-debit policy for key budgets (Bug
/// #33): usage accumulated before termination represents tokens the upstream
/// already generated and billed to us, so the router-level reservation made
/// at pick time is settled to the partial cost (via
/// [`crate::orchestration::budget::settle_router_spend`]) instead of leaking.
/// With no usage on record the counters are left untouched — the caller
/// decides whether the reservation stands (client disconnect with bytes
/// sent: kept as the debit) or is released (stream error without usage, and
/// disconnect-before-first-byte: released, mirroring `post_failure`).
///
/// Contract: a `status: "partial"` spend-log row is queued for every caller
/// that keeps a debit standing — including the zero-usage client-disconnect
/// path, so the ledger matches the budget counters. The ONE exception is
/// the stream-error-without-usage path, which releases the reservation
/// instead (no debit stands → no row to reconcile) and never calls this
/// function; see the `StreamExit::Error` arm of [`finalize_stream_exit`].
async fn record_partial_stream_spend(
    state: &AppState,
    deployment: &Deployment,
    call_ctx: &CallContext,
    response_model: &str,
    accumulated_usage: Option<&oxllm_core::UsageSummary>,
    budget_reservation: &super::budget::RouterBudgetReservation,
    reported_cost: Option<f64>,
) {
    if !call_ctx.claim_spend_record() {
        return;
    }
    let cost = crate::hooks::cost_fallback::resolve_cost(
        call_ctx,
        response_model,
        accumulated_usage,
        None,
        reported_cost,
        deployment.provider_slug(),
    )
    .or_else(|| deployment_cost_fallback(deployment, call_ctx, accumulated_usage));
    let tags = tags_json(&call_ctx.request_tags);
    tracing::info!(event = "spend", status = "partial", request_id = %call_ctx.request_id,
        session_id = call_ctx.session_id.as_deref(),
        model = response_model, provider = deployment.provider_slug(), cost, cost_known = cost.is_some(),
        deployment_id = call_ctx.deployment_id.as_deref(), key_id = call_ctx.key_id.as_deref(),
        prompt_tokens = accumulated_usage.map(|usage| usage.prompt_tokens),
        completion_tokens = accumulated_usage.map(|usage| usage.completion_tokens),
        tags = %tags.as_deref().unwrap_or("[]"), "partial stream spend recorded");
    if accumulated_usage.is_some() || cost.is_some() {
        super::budget::settle_router_spend(
            state,
            deployment,
            &deployment_key(deployment),
            cost,
            budget_reservation,
        )
        .await;
    }

    // Queue a spend-log row for partial streams (disconnect / mid-stream
    // error). Success streams write via CostTrackingHook on post_success;
    // these exit paths never reach that hook. The row carries zeros when no
    // usage was accumulated — the key-scope debit stands via the un-released
    // reservation, and the row keeps the ledger reconcilable with it.
    if let Some(tx) = state.spend_tx.as_ref() {
        let status = "partial";
        let tokens = |v: u64| i64::try_from(v).unwrap_or(i64::MAX);
        let entry = crate::spend_writer::SpendEntry {
            timestamp: jiff::Timestamp::now().to_string(),
            request_id: Some(call_ctx.request_id.clone()),
            session_id: call_ctx.session_id.clone(),
            model: response_model.to_string(),
            provider: deployment.provider_slug().to_string(),
            deployment_id: call_ctx.deployment_id.clone(),
            requested_model: Some(call_ctx.model.clone()),
            model_group: call_ctx.serving_model_group.clone(),
            prompt_tokens: Some(
                accumulated_usage
                    .map(|u| tokens(u.prompt_tokens))
                    .unwrap_or(0),
            ),
            completion_tokens: Some(
                accumulated_usage
                    .map(|u| tokens(u.completion_tokens))
                    .unwrap_or(0),
            ),
            total_tokens: Some(
                accumulated_usage
                    .map(|u| tokens(u.total_tokens))
                    .unwrap_or(0),
            ),
            cost,
            key_id: call_ctx.key_id.clone(),
            team_id: call_ctx.team_id.clone(),
            user_id: call_ctx.user_id.clone(),
            end_user_id: call_ctx.end_user_id.clone(),
            request_tag: call_ctx.request_tag.clone(),
            request_tags: tags,
            status: Some(status.to_string()),
            latency_ms: call_ctx.start_time.map(|s| s.elapsed().as_millis() as i64),
        };
        crate::spend_writer::queue_partial_spend(tx, entry).await;
    }
}

/// Some streams (notably Converse) identify themselves with the public alias.
/// Keep that identity in logs/chunks, but retain the resolved model as a cost
/// fallback so an unpriced alias does not turn a priced deployment into free use.
fn deployment_cost_fallback(
    deployment: &Deployment,
    call_ctx: &CallContext,
    usage: Option<&oxllm_core::UsageSummary>,
) -> Option<f64> {
    crate::hooks::cost_fallback::resolve_cost(
        call_ctx,
        &deployment.oxllm_params.model,
        usage,
        None,
        None,
        deployment.provider_slug(),
    )
}

/// Drain complete AWS eventstream frames from `raw_buffer`, converting each
/// frame's JSON payload into an SSE `data: {...}\n\n` event appended to
/// `buffer`.
///
/// Bedrock's `invoke-with-response-stream` (and `converseStream`) return
/// binary `application/vnd.amazon.eventstream` frames, not SSE text — feeding
/// those bytes through the UTF-8 `SseFramer` silently produces nothing, so
/// no chunks ever reach the client (the original Bug: Bedrock streaming
/// produced no output). Each frame's payload is a JSON object in the vendor's
/// native event shape; we wrap it as an SSE `data:` line so the downstream
/// `pop_sse_event` / `transform_stream_chunk` pipeline works unchanged.
///
/// Append incoming Bedrock bytes to `raw_buffer`, drain any complete
/// eventstream frames into the text `buffer`, and enforce the
/// [`MAX_RAW_BUFFER_BYTES`] cap on the leftover undecoded tail. Returns an
/// error when the incomplete tail exceeds the cap, which bounds memory
/// against a malicious/buggy upstream that declares a huge frame and trickles
/// bytes (the declared `total_len` is read from the frame header by
/// [`oxllm_providers::bedrock::decode_eventstream_frame`], so a never-complete
/// frame would otherwise grow `raw_buffer` without limit).
/// A Bedrock stream that ends before a single event decodes is an upstream
/// failure, even when the HTTP status was 200 and the body was empty.
/// Leftover bytes are a truncated frame, which is a transport error.
fn bedrock_stream_close_error(events: u64, raw_leftover: bool) -> Option<oxllm_core::Error> {
    if raw_leftover {
        return Some(oxllm_core::Error::Network(
            "upstream ended with an incomplete or invalid Bedrock eventstream frame".to_string(),
        ));
    }
    if events == 0 {
        return Some(oxllm_core::Error::Upstream {
            status: 502,
            message: "Bedrock answered the stream with HTTP 200 but its body decoded to no events"
                .to_string(),
        });
    }
    None
}

pub(super) fn append_and_decode_bedrock(
    raw_buffer: &mut Vec<u8>,
    buffer: &mut String,
    incoming: &[u8],
    events: &mut u64,
) -> oxllm_core::Result<()> {
    raw_buffer.extend_from_slice(incoming);
    *events = events.saturating_add(decode_bedrock_eventstream_into_buffer(raw_buffer, buffer));
    if raw_buffer.len() > MAX_RAW_BUFFER_BYTES {
        return Err(oxllm_core::Error::Network(format!(
            "bedrock eventstream raw_buffer exceeded {MAX_RAW_BUFFER_BYTES} byte cap without a complete frame"
        )));
    }
    Ok(())
}

/// Incomplete frames (fewer bytes than `total_len`) are left in `raw_buffer`
/// for the next chunk arrival.
fn decode_bedrock_eventstream_into_buffer(raw_buffer: &mut Vec<u8>, buffer: &mut String) -> u64 {
    let mut consumed_total = 0usize;
    let mut events = 0u64;
    while let Some((consumed, payload)) =
        oxllm_providers::bedrock::decode_eventstream_frame(&raw_buffer[consumed_total..])
    {
        events = events.saturating_add(1);
        let event_type =
            bedrock_eventstream_type(&raw_buffer[consumed_total..consumed_total + consumed]);
        let raw_str = match std::str::from_utf8(payload) {
            Ok(s) => s.trim(),
            Err(_) => {
                tracing::warn!(
                    payload_len = payload.len(),
                    "bedrock eventstream payload failed UTF-8 validation; dropping frame"
                );
                consumed_total += consumed;
                continue;
            }
        };
        consumed_total += consumed;
        if raw_str.is_empty() {
            continue;
        }
        // The shared codec has already skipped the 12-byte prelude and typed
        // headers. Header bytes must never be treated as a JSON prefix.
        let json_str = raw_str;
        // Some Bedrock models wrap output in {"bytes": "<base64>"}.
        let final_str = if let Ok(mut v) = serde_json::from_str::<serde_json::Value>(json_str) {
            if let Some(kind) = event_type.filter(|kind| is_bedrock_event_wrapper(kind)) {
                if v.get(kind).is_none() {
                    v = serde_json::json!({kind: v});
                }
            }
            if let Some(b) = v.get("bytes").and_then(|b| b.as_str()) {
                use base64::Engine;
                if let Ok(decoded) = base64::engine::general_purpose::STANDARD.decode(b) {
                    compact_json_or_raw(String::from_utf8_lossy(&decoded).trim().to_string())
                } else {
                    v.to_string()
                }
            } else {
                v.to_string()
            }
        } else {
            json_str.to_string()
        };
        if !final_str.is_empty() {
            buffer.push_str("data: ");
            buffer.push_str(&final_str);
            buffer.push_str("\n\n");
        }
    }
    raw_buffer.drain(..consumed_total);
    events
}

/// Read the event/exception discriminator from an already CRC-validated frame.
/// AWS headers carry typed values; skip other standard types without treating
/// their bytes as strings. Exception type takes precedence over event type.
fn bedrock_eventstream_type(frame: &[u8]) -> Option<&str> {
    let header_len = u32::from_be_bytes(frame.get(4..8)?.try_into().ok()?) as usize;
    let mut headers = frame.get(12..12usize.checked_add(header_len)?)?;
    let mut event_type = None;
    let mut exception_type = None;
    while !headers.is_empty() {
        let name_len = *headers.first()? as usize;
        headers = headers.get(1..)?;
        let name = headers.get(..name_len)?;
        headers = headers.get(name_len..)?;
        let kind = *headers.first()?;
        headers = headers.get(1..)?;
        let value_len = match kind {
            0 | 1 => 0,
            2 => 1,
            3 => 2,
            4 => 4,
            5 | 8 => 8,
            9 => 16,
            6 | 7 => {
                let len = u16::from_be_bytes(headers.get(..2)?.try_into().ok()?) as usize;
                headers = headers.get(2..)?;
                len
            }
            _ => return None,
        };
        let value = headers.get(..value_len)?;
        if kind == 7 {
            if name == b":event-type" {
                event_type = std::str::from_utf8(value).ok();
            } else if name == b":exception-type" {
                exception_type = std::str::from_utf8(value).ok();
            }
        }
        headers = headers.get(value_len..)?;
    }
    exception_type.or(event_type)
}

fn is_bedrock_event_wrapper(kind: &str) -> bool {
    matches!(
        kind,
        "messageStart"
            | "contentBlockStart"
            | "contentBlockDelta"
            | "contentBlockStop"
            | "messageStop"
            | "metadata"
            | "internalServerException"
            | "modelStreamErrorException"
            | "modelTimeoutException"
            | "throttlingException"
            | "serviceUnavailableException"
            | "validationException"
    )
}

/// Re-serialize `text` compactly when it parses as JSON, so a payload
/// containing newlines (pretty-printed vendor JSON, embedded whitespace)
/// cannot split the SSE event framing downstream (`pop_sse_event` splits on
/// `\n\n` and would truncate the event to its first line). Non-JSON text
/// passes through unchanged.
fn compact_json_or_raw(text: String) -> String {
    match serde_json::from_str::<serde_json::Value>(&text) {
        Ok(v) => v.to_string(),
        Err(_) => text,
    }
}

/// Process complete SSE events from the buffer, forwarding transformed chunks
/// to the client while accumulating token usage and the reported model name.
///
/// When `stream_options.include_usage` was requested
/// (`usage_injection.include_usage`), a synthesized usage chunk is emitted
/// exactly once, immediately before the `[DONE]` marker — matching OpenAI's
/// `stream_options.include_usage` contract (final chunk carries empty
/// `choices` + the accumulated `usage`). The `usage_injection.emitted` guard
/// prevents a second emission if `[DONE]` were to appear more than once, and
/// also suppresses the synthesized chunk when the upstream's own usage-only
/// chunk was already forwarded.
///
/// Usage-only chunks the upstream emits (because we injected
/// `include_usage` upstream) are forwarded only when the client requested
/// usage; a client that did not ask never sees one (OpenAI contract).
///
/// Sends are bounded by [`CLIENT_SEND_STALL_TIMEOUT`]: on a client stall the
/// `client_stalled` flag is set and the caller aborts the stream.
#[allow(clippy::too_many_arguments)]
async fn process_buffer(
    buffer: &mut String,
    chat: &dyn oxllm_core::provider::ChatProvider,
    tx: &mpsc::Sender<Result<Event, std::convert::Infallible>>,
    accumulated_usage: &mut Option<oxllm_core::UsageSummary>,
    response_model: &mut String,
    usage_injection: &mut UsageInjection,
    done_sent: &mut bool,
    bytes_sent_to_client: &mut bool,
    stream_state: &mut Option<Box<dyn std::any::Any + Send + Sync>>,
    original_request: &serde_json::Value,
    completion_text: &mut String,
    client_stalled: &mut bool,
) -> oxllm_core::Result<()> {
    if *done_sent {
        buffer.clear();
        return Ok(());
    }
    while let Some(event_str) = pop_sse_event(buffer) {
        // Use the stateful variant so providers that accumulate per-stream
        // state (e.g. Anthropic) carry it in a host-owned slot that travels
        // with this tokio task across worker-thread migrations, instead of a
        // provider-side `thread_local!` that gets orphaned on reschedule.
        match chat.transform_stream_chunk_stateful(&event_str, stream_state, original_request) {
            Ok(Some(mut value)) => {
                if let Some(u) = extract_usage(&value) {
                    merge_streaming_usage(accumulated_usage, u);
                }
                capture_stream_usage_cost(usage_injection, &value);
                if let Some(m) = value.get("model").and_then(|v| v.as_str()) {
                    *response_model = m.to_string();
                }
                accumulate_completion_text(completion_text, &value);
                normalize_stream_chunk(&mut value, original_request, usage_injection);
                if usage_injection.strip_content_usage {
                    strip_usage_from_content_chunk(&mut value);
                }
                if is_usage_only_chunk(&value) {
                    if !usage_injection.include_usage {
                        // The client never asked for usage chunks; this one
                        // exists only because we injected `include_usage`
                        // upstream for metering. Suppress it (usage was
                        // still merged into the accumulator above).
                        continue;
                    }
                    // The client asked: forward the upstream's own usage
                    // chunk exactly once; `emitted` suppresses the later
                    // synthesized duplicate (exactly one usage chunk).
                    if usage_injection.emitted {
                        continue;
                    }
                    usage_injection.emitted = true;
                }
                match send_event(tx, Event::default().data(value.to_string())).await {
                    ClientSend::Sent => *bytes_sent_to_client = true,
                    ClientSend::Closed => return Ok(()),
                    ClientSend::Stalled => {
                        *client_stalled = true;
                        return Ok(());
                    }
                }
            }
            Ok(None) => {
                if extract_sse_data(&event_str) == Some("[DONE]") {
                    if let Some(mut chunk) =
                        build_usage_chunk_if_requested(usage_injection, accumulated_usage.as_ref())
                    {
                        normalize_stream_chunk(&mut chunk, original_request, usage_injection);
                        usage_injection.emitted = true;
                        match send_event(tx, Event::default().data(chunk.to_string())).await {
                            ClientSend::Sent => *bytes_sent_to_client = true,
                            ClientSend::Closed => return Ok(()),
                            ClientSend::Stalled => {
                                *client_stalled = true;
                                return Ok(());
                            }
                        }
                    }
                    match send_event(tx, Event::default().data("[DONE]")).await {
                        ClientSend::Sent => *bytes_sent_to_client = true,
                        ClientSend::Closed => return Ok(()),
                        ClientSend::Stalled => {
                            *client_stalled = true;
                            return Ok(());
                        }
                    }
                    *done_sent = true;
                    buffer.clear();
                    return Ok(());
                }
            }
            Err(e) => {
                tracing::warn!(
                error = %e,
                chunk_len = event_str.len(),
                    "stream chunk transform error (content redacted)"
                );
                return Err(e);
            }
        }
    }
    Ok(())
}

/// State for the `stream_options.include_usage` synthesized usage chunk.
///
/// `include_usage` is captured once at the start of the stream task from the
/// request body; `emitted` guards against emitting the synthesized chunk more
/// than once (defensive — `[DONE]` normally appears exactly once).
struct UsageInjection {
    public_model: Option<String>,
    /// Whether the client requested `stream_options.include_usage: true`.
    include_usage: bool,
    /// Whether the synthesized usage chunk has already been emitted.
    emitted: bool,
    /// Scalar or complete receipt captured from a stream chunk. Local rates
    /// normally win; Perplexity receipts include dynamic search/request fees.
    reported_cost: Option<f64>,
    /// Last served-tier observation; independent of whether usage was sent.
    served_service_tier: Option<String>,
    /// Provider usage metadata merged across chunks so complementary fields
    /// survive after usage is stripped from content chunks.
    reported_usage: Option<serde_json::Value>,
    /// Whether content/finish chunks should have usage removed for a direct
    /// OpenAI client. Native converters keep the metadata for their own events.
    strip_content_usage: bool,
}

fn normalize_stream_chunk(
    value: &mut serde_json::Value,
    request: &serde_json::Value,
    injection: &UsageInjection,
) {
    if let Some(model) = injection
        .public_model
        .as_deref()
        .or_else(|| request.get("model").and_then(serde_json::Value::as_str))
    {
        pipeline::normalize_client_response(value, model);
    }
}

/// Merge nested provider usage metadata. Later scalar values win, while
/// complementary fields from earlier snapshots are retained.
fn merge_usage_metadata(current: &mut serde_json::Value, incoming: &serde_json::Value) {
    let Some(incoming) = incoming.as_object() else {
        *current = incoming.clone();
        return;
    };
    if !current.is_object() {
        *current = serde_json::Value::Object(serde_json::Map::new());
    }
    for (key, value) in incoming {
        if let Some(existing) = current.get_mut(key) {
            if existing.is_object() && value.is_object() {
                merge_usage_metadata(existing, value);
                continue;
            }
        }
        current[key] = value.clone();
    }
}

/// Capture usage metadata from a transformed stream chunk. Later chunks win
/// on conflicting values; complementary fields are merged across snapshots.
fn capture_stream_usage_cost(usage_injection: &mut UsageInjection, value: &serde_json::Value) {
    if let Some(usage) = value.get("usage").filter(|usage| usage.is_object()) {
        match usage_injection.reported_usage.as_mut() {
            Some(current) => merge_usage_metadata(current, usage),
            None => usage_injection.reported_usage = Some(usage.clone()),
        }
    }
    if let Some(c) = pipeline::extract_body_usage_cost(value) {
        usage_injection.reported_cost = Some(c);
    }
    if let Some(tier) = oxllm_core::response_service_tier(value) {
        if usage_injection.served_service_tier.as_deref() != Some(tier) {
            usage_injection.served_service_tier = Some(tier.to_string());
        }
    }
}

/// Classify an upstream body-read error. After `[DONE]` the provider has
/// already finished; a trailing connection reset is treated as success.
/// Before `[DONE]` it remains a stream failure.
fn stream_exit_on_upstream_read_error(done_sent: bool, err: impl std::fmt::Display) -> StreamExit {
    if done_sent {
        tracing::info!(
            error = %err,
            "upstream stream error after [DONE]; treating as clean completion"
        );
        StreamExit::Done
    } else {
        tracing::warn!(
            error = %err,
            "upstream stream error; no fallback after bytes sent"
        );
        StreamExit::Error(oxllm_core::Error::Network(format!(
            "upstream stream error: {err}"
        )))
    }
}

/// Build the synthesized usage chunk for `stream_options.include_usage`, when
/// the client requested it and usage was accumulated and it has not already
/// been emitted.
///
/// The chunk matches OpenAI's contract: empty `choices` array plus a
/// `usage` object populated from the accumulated token counts. Returns `None`
/// when `include_usage` is false, no usage is on record, or the chunk was
/// already emitted.
fn build_usage_chunk_if_requested(
    usage_injection: &UsageInjection,
    accumulated_usage: Option<&oxllm_core::UsageSummary>,
) -> Option<serde_json::Value> {
    if !usage_injection.include_usage || usage_injection.emitted {
        return None;
    }
    accumulated_usage.map(|u| {
        // Start with the provider's latest usage object so fields outside our
        // billing summary (for example Anthropic server_tool_use) survive.
        // Accumulated counts overwrite the canonical fields below.
        let mut usage = usage_injection
            .reported_usage
            .clone()
            .filter(|usage| usage.is_object())
            .unwrap_or_else(|| serde_json::json!({}));
        usage["prompt_tokens"] = serde_json::json!(u.prompt_tokens);
        usage["completion_tokens"] = serde_json::json!(u.completion_tokens);
        usage["total_tokens"] = serde_json::json!(u.total_tokens);
        // Preserve a late provider receipt when the client requested usage.
        // In particular, Perplexity's dynamic fees cannot be reconstructed
        // from token counts alone in a subsequent /spend/calculate preview.
        if let Some(cost) = usage_injection.reported_cost {
            usage["cost"] = serde_json::json!(cost);
        }
        // H6: surface `completion_tokens_details.reasoning_tokens` so
        // streaming clients of reasoning models (o3 / o4-mini) see the same
        // breakdown as the non-streaming path. The field is a subset of
        // `completion_tokens` and is priced at the completion rate.
        usage["completion_tokens_details"]["reasoning_tokens"] =
            serde_json::json!(u.reasoning_tokens.unwrap_or(0));
        if let Some(value) = u.cache_creation_input_tokens {
            usage["cache_creation_input_tokens"] = serde_json::json!(value);
        }
        for (name, value) in [
            ("cached_tokens", u.cache_read_input_tokens),
            (
                "ephemeral_1h_input_tokens",
                u.cache_creation_1h_input_tokens,
            ),
            ("audio_tokens", u.audio_input_tokens),
        ] {
            if let Some(value) = value {
                usage["prompt_tokens_details"][name] = serde_json::json!(value);
            }
        }
        let mut chunk = serde_json::json!({"choices": [], "usage": usage});
        if let Some(tier) = &usage_injection.served_service_tier {
            chunk["service_tier"] = serde_json::json!(tier);
        }
        chunk
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxllm_core::Deployment;
    use oxllm_providers::openai::OpenAiProvider;
    use std::sync::{Arc, Mutex};

    #[tokio::test]
    async fn duration_priced_partial_stream_is_metered_without_token_usage() {
        let config = crate::config::parse_config_str("model_list: []").unwrap();
        let mut state =
            AppState::with_registry(config, oxllm_providers::default_registry()).unwrap();
        let (tx, mut rx) = mpsc::channel(1);
        state.spend_tx = Some(tx);
        let deployment = Deployment {
            model_name: "duration-model".into(),
            oxllm_params: oxllm_core::OxllmParams {
                model: "openai/private-duration-model".into(),
                max_budget: Some(100.0),
                ..Default::default()
            },
            ..Default::default()
        };
        let ctx = CallContext {
            deployment_pricing: Some(oxllm_core::DeploymentPricing {
                cost_per_second_rate: Some(0.01),
                ..Default::default()
            }),
            start_time: Some(std::time::Instant::now() - Duration::from_secs(1)),
            upstream_start_time: Some(std::time::Instant::now() - Duration::from_secs(1)),
            ..CallContext::for_test("duration-model")
        };
        record_partial_stream_spend(
            &state,
            &deployment,
            &ctx,
            "private-duration-model",
            None,
            &super::super::budget::RouterBudgetReservation::default(),
            None,
        )
        .await;
        let entry = rx.try_recv().unwrap();
        assert_eq!(entry.status.as_deref(), Some("partial"));
        assert_eq!(entry.total_tokens, Some(0));
        assert!(entry.cost.unwrap() >= 0.01);
        let key = crate::hooks::spend_keys::deployment_spend_key(&deployment_key(&deployment));
        assert_eq!(state.cache.counter_value(&key).await, entry.cost);
    }

    #[tokio::test]
    async fn disconnect_billing_releases_before_content_and_keeps_partial_debits() {
        use crate::config::types::{GeneralSettings, ProxyConfig};
        use crate::hooks::spend_keys::{deployment_spend_key, spend_key};
        use crate::hooks::{BudgetHook, RateLimitHook};

        for (bytes_sent, has_usage) in [(false, false), (true, false), (false, true), (true, true)]
        {
            let config = ProxyConfig {
                model_list: vec![],
                oxllm_settings: Default::default(),
                general_settings: GeneralSettings::default(),
                router_settings: Default::default(),
                extra: serde_json::Map::new(),
            };
            let mut state =
                AppState::with_registry(config, oxllm_providers::default_registry()).unwrap();
            state.hooks = vec![
                Arc::new(BudgetHook::with_duration(
                    state.cache.clone(),
                    Some(100.0),
                    Duration::from_secs(3600),
                )),
                Arc::new(RateLimitHook::new(
                    state.cache.clone(),
                    Some(100),
                    Some(10_000),
                )),
            ];
            let (spend_tx, mut spend_rx) = mpsc::channel(8);
            state.spend_tx = Some(spend_tx);
            let deployment = Deployment {
                model_name: "gpt-4o".to_string(),
                oxllm_params: oxllm_core::deployment::OxllmParams {
                    model: "openai/gpt-4o".to_string(),
                    max_budget: Some(100.0),
                    ..Default::default()
                },
                ..Default::default()
            };
            let mut ctx = CallContext {
                max_budget: Some(100.0),
                session_id: Some("partial-stream-session".into()),
                max_parallel_requests: Some(1),
                request_body: Arc::new(serde_json::json!({
                    "model":"gpt-4o","messages":[{"role":"user","content":"hello"}],"max_tokens":100
                })),
                ..CallContext::for_test("gpt-4o")
            };
            crate::hooks::run_pre_call_hooks(&state.hooks, &mut ctx)
                .await
                .unwrap();
            let dep_key = deployment_key(&deployment);
            let reservation =
                super::super::budget::reserve_router_budgets(&state, &deployment, &dep_key, &ctx)
                    .await
                    .unwrap();
            let key_spend = spend_key(&ctx.api_key_hash);
            let dep_spend = deployment_spend_key(&dep_key);
            let reserved_key = state.cache.counter_value(&key_spend).await.unwrap();
            let reserved_dep = state.cache.counter_value(&dep_spend).await.unwrap();
            assert!(reserved_key > 0.0 && reserved_dep > 0.0);
            assert_eq!(state.cache.counter_value("parallel:abc").await, Some(1.0));
            let mut usage = has_usage.then_some(oxllm_core::UsageSummary {
                prompt_tokens: 2,
                completion_tokens: 1,
                total_tokens: 3,
                ..Default::default()
            });
            let cooldown_key = state.router.cooldown_key(&deployment);
            state.router.record_in_flight(&cooldown_key, 1);
            let (tx, _rx) = mpsc::channel(8);
            finalize_stream_exit(
                &state,
                &deployment,
                &ctx,
                &reservation,
                "gpt-4o",
                &cooldown_key,
                &dep_key,
                &tx,
                StreamExit::ClientDisconnected,
                &mut usage,
                "public-model-alias",
                bytes_sent,
                "",
                None,
            )
            .await;
            assert_eq!(state.cache.counter_value("parallel:abc").await, Some(0.0));
            let actual_key = state.cache.counter_value(&key_spend).await.unwrap_or(0.0);
            let actual_dep = state.cache.counter_value(&dep_spend).await.unwrap_or(0.0);
            if bytes_sent || has_usage {
                assert!(
                    (actual_key - reserved_key).abs() < 1e-9,
                    "partial key debit must stand"
                );
                let expected_dep = usage
                    .as_ref()
                    .and_then(|u| oxllm_core::calculate_cost("gpt-4o", u))
                    .unwrap_or(reserved_dep);
                assert!(
                    (actual_dep - expected_dep).abs() < 1e-9,
                    "router spend must settle partial usage"
                );
                assert_eq!(state.cache.counter_value("rpm:abc").await, Some(1.0));
                let entry = spend_rx.try_recv().expect("partial debit must be logged");
                assert_eq!(entry.status.as_deref(), Some("partial"));
                assert_eq!(entry.session_id.as_deref(), Some("partial-stream-session"));
                assert_eq!(
                    entry.total_tokens,
                    usage.as_ref().map(|usage| usage.total_tokens as i64)
                );
            } else {
                assert!(actual_key.abs() < 1e-9 && actual_dep.abs() < 1e-9);
                assert_eq!(state.cache.counter_value("rpm:abc").await, Some(0.0));
                assert!(
                    spend_rx.try_recv().is_err(),
                    "no debit should produce no partial ledger entry"
                );
            }
        }
    }

    #[tokio::test]
    async fn process_buffer_stops_at_first_done_before_error_tail() {
        let (tx, rx) = mpsc::channel(8);
        let mut buffer = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n",
            "data: [DONE]\n\ndata: [DONE]\n\n",
            "data: {\"error\":{\"message\":\"late failure\"}}\n\n"
        )
        .to_string();
        let mut usage = None;
        let mut model = String::new();
        let mut injection = UsageInjection {
            public_model: None,
            include_usage: false,
            emitted: false,
            reported_cost: None,
            served_service_tier: None,
            reported_usage: None,
            strip_content_usage: true,
        };
        let mut done = false;
        process_buffer(
            &mut buffer,
            &OpenAiProvider,
            &tx,
            &mut usage,
            &mut model,
            &mut injection,
            &mut done,
            &mut false,
            &mut None,
            &serde_json::Value::Null,
            &mut String::new(),
            &mut false,
        )
        .await
        .unwrap();
        assert!(done);
        assert!(buffer.is_empty());
        drop(tx);
        let body = drain_sse_to_string(rx).await;
        assert_eq!(body.matches("[DONE]").count(), 1);
        assert!(!body.contains("error"));
    }

    #[tokio::test]
    async fn dropping_stream_body_keeps_debit_and_cancels_live_upstream() {
        use crate::hooks::{BudgetHook, RateLimitHook};
        use axum::body::{Body, Bytes};

        let (source_tx, source_rx) = mpsc::channel::<Result<Bytes, std::io::Error>>(1);
        source_tx.send(Ok(Bytes::from_static(
            b"data: {\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"content\":\"partial\"},\"finish_reason\":null}]}\n\n"
        ))).await.unwrap();
        let source = Arc::new(Mutex::new(Some(source_rx)));
        let upstream_app = axum::Router::new().route(
            "/",
            axum::routing::get(move || {
                let source = source.clone();
                async move {
                    let rx = source.lock().unwrap().take().unwrap();
                    (
                        [("content-type", "text/event-stream")],
                        Body::from_stream(ReceiverStream::new(rx)),
                    )
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            axum::serve(listener, upstream_app).await.unwrap();
        });
        let upstream = reqwest::Client::new()
            .get(format!("http://{address}/"))
            .send()
            .await
            .unwrap();
        let config = serde_json::from_value(serde_json::json!({"model_list":[]})).unwrap();
        let mut state =
            AppState::with_registry(config, oxllm_providers::default_registry()).unwrap();
        state.hooks = vec![
            Arc::new(BudgetHook::with_duration(
                state.cache.clone(),
                Some(100.0),
                Duration::from_secs(3600),
            )),
            Arc::new(RateLimitHook::new(
                state.cache.clone(),
                Some(100),
                Some(10_000),
            )),
        ];
        let (spend_tx, mut spend_rx) = mpsc::channel(1);
        state.spend_tx = Some(spend_tx);
        let mut ctx = CallContext {
            max_budget: Some(100.0),
            max_parallel_requests: Some(1),
            request_body: Arc::new(serde_json::json!({
                "model":"gpt-4o","messages":[{"role":"user","content":"hello"}],"max_tokens":100
            })),
            ..CallContext::for_test("gpt-4o")
        };
        crate::hooks::run_pre_call_hooks(&state.hooks, &mut ctx)
            .await
            .unwrap();
        let reserved = state.cache.counter_value("spend:abc").await.unwrap();
        let deployment = Deployment {
            model_name: "gpt-4o".to_string(),
            oxllm_params: oxllm_core::deployment::OxllmParams {
                model: "openai/gpt-4o".to_string(),
                ..Default::default()
            },
            ..Default::default()
        };
        state
            .router
            .record_in_flight(&state.router.cooldown_key(&deployment), 1);
        let response = handle_stream_success(
            Box::pin(upstream.bytes_stream()),
            deployment,
            Arc::new(OpenAiProvider),
            state.clone(),
            ctx,
            super::super::budget::RouterBudgetReservation::default(),
            "gpt-4o".to_string(),
        )
        .await
        .unwrap();
        let mut client = response.into_body().into_data_stream();
        let first = tokio::time::timeout(Duration::from_secs(5), client.next())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert!(String::from_utf8_lossy(&first).contains("partial"));
        drop(client);
        // A partial-spend row is emitted after settlement and concurrency
        // release. Await that signal rather than sleeping/polling a task.
        let entry = tokio::time::timeout(Duration::from_secs(5), spend_rx.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(entry.status.as_deref(), Some("partial"));
        assert!(entry.prompt_tokens.unwrap() > 0);
        assert!(entry.completion_tokens.unwrap() > 0);
        assert_eq!(state.cache.counter_value("spend:abc").await, Some(reserved));
        assert_eq!(state.cache.counter_value("parallel:abc").await, Some(0.0));
        tokio::time::timeout(Duration::from_secs(5), source_tx.closed())
            .await
            .expect("dropping the client must cancel the live upstream body");
        server.abort();
    }

    /// M4 duplicate-terminator guard, end to end through the spawned pump:
    /// an upstream error event that trails a clean `[DONE]` in a *separate*
    /// body chunk must not produce a second terminator or leak the error to
    /// the client. The inline `[DONE]` sets `done_sent`, the pump exits
    /// `Done`, and the post-loop drain must stay gated on `!done_sent`.
    /// `process_buffer_stops_at_first_done_before_error_tail` pins the
    /// function-level guard; this test pins the pump wiring around it.
    #[tokio::test]
    async fn trailing_error_after_done_emits_exactly_one_done() {
        use axum::body::{Body, Bytes};

        let (source_tx, source_rx) = mpsc::channel::<Result<Bytes, std::io::Error>>(4);
        source_tx
            .send(Ok(Bytes::from_static(
                b"data: {\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n\n",
            )))
            .await
            .unwrap();
        // Trailing error event in a second chunk: a regression that keeps
        // reading past the terminator (or replays the tail in the post-loop
        // drain) would deliver it and/or a second `[DONE]`.
        source_tx
            .send(Ok(Bytes::from_static(
                b"data: {\"error\":{\"message\":\"late failure\"}}\n\n",
            )))
            .await
            .unwrap();
        let source = Arc::new(Mutex::new(Some(source_rx)));
        let upstream_app = axum::Router::new().route(
            "/",
            axum::routing::get(move || {
                let source = source.clone();
                async move {
                    let rx = source.lock().unwrap().take().unwrap();
                    (
                        [("content-type", "text/event-stream")],
                        Body::from_stream(ReceiverStream::new(rx)),
                    )
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            axum::serve(listener, upstream_app).await.unwrap();
        });
        let upstream = reqwest::Client::new()
            .get(format!("http://{address}/"))
            .send()
            .await
            .unwrap();
        let config = serde_json::from_value(serde_json::json!({"model_list":[]})).unwrap();
        let state = AppState::with_registry(config, oxllm_providers::default_registry()).unwrap();
        let ctx = CallContext::for_test("gpt-4o");
        let deployment = Deployment {
            model_name: "gpt-4o".to_string(),
            oxllm_params: oxllm_core::deployment::OxllmParams {
                model: "openai/gpt-4o".to_string(),
                ..Default::default()
            },
            ..Default::default()
        };
        state
            .router
            .record_in_flight(&state.router.cooldown_key(&deployment), 1);
        let response = handle_stream_success(
            Box::pin(upstream.bytes_stream()),
            deployment,
            Arc::new(OpenAiProvider),
            state.clone(),
            ctx,
            super::super::budget::RouterBudgetReservation::default(),
            "gpt-4o".to_string(),
        )
        .await
        .unwrap();
        let body = tokio::time::timeout(
            Duration::from_secs(5),
            axum::body::to_bytes(response.into_body(), 1024 * 1024),
        )
        .await
        .expect("stream must terminate")
        .expect("sse body");
        let body = String::from_utf8_lossy(&body);
        assert!(
            body.contains("\"content\":\"ok\""),
            "the pre-terminator content must reach the client: {body}"
        );
        assert_eq!(
            body.matches("[DONE]").count(),
            1,
            "exactly one [DONE] after a trailing upstream error: {body}"
        );
        assert!(
            !body.contains("late failure") && !body.contains("\"error\""),
            "the post-[DONE] error must never reach the client: {body}"
        );
        server.abort();
    }

    struct CapturingWriter {
        buf: Arc<Mutex<String>>,
    }

    impl std::io::Write for CapturingWriter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.buf
                .lock()
                .unwrap()
                .push_str(&String::from_utf8_lossy(bytes));
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    fn make_writer(buf: Arc<Mutex<String>>) -> CapturingWriter {
        CapturingWriter { buf }
    }

    struct FailingChunkProvider;

    impl oxllm_core::Provider for FailingChunkProvider {
        fn id(&self) -> oxllm_core::ProviderId {
            "failing-mock"
        }
        fn supports(&self, _: oxllm_core::Endpoint) -> bool {
            false
        }
        fn chat(&self) -> Option<&dyn oxllm_core::provider::ChatProvider> {
            Some(self)
        }
    }
    impl oxllm_core::provider::ChatProvider for FailingChunkProvider {
        fn endpoint_url(&self, _: &Deployment) -> String {
            String::new()
        }
        fn apply_auth(
            &self,
            _: &Deployment,
            _: &oxllm_core::provider::ResolvedCredentials,
            _: &mut std::collections::HashMap<String, String>,
        ) {
        }
        fn transform_request(
            &self,
            req: &serde_json::Value,
            _: &Deployment,
        ) -> oxllm_core::Result<serde_json::Value> {
            Ok(req.clone())
        }
        fn transform_response(
            &self,
            resp: &serde_json::Value,
            _: &str,
        ) -> oxllm_core::Result<serde_json::Value> {
            Ok(resp.clone())
        }
        fn transform_stream_chunk(&self, _: &str) -> oxllm_core::Result<Option<serde_json::Value>> {
            Err(oxllm_core::Error::Transform("bad chunk".to_string()))
        }
    }

    #[tokio::test]
    async fn stream_chunk_error_log_does_not_leak_content() {
        let captured: Arc<Mutex<String>> = Arc::new(Mutex::new(String::new()));
        let buf = captured.clone();
        let subscriber = tracing_subscriber::FmtSubscriber::builder()
            .with_writer(move || make_writer(buf.clone()))
            .with_max_level(tracing::Level::WARN)
            .without_time()
            .with_target(false)
            .finish();
        let _guard = tracing::subscriber::set_default(subscriber);

        let (tx, mut rx) = mpsc::channel::<Result<Event, std::convert::Infallible>>(64);
        let provider = FailingChunkProvider;
        let mut usage: Option<oxllm_core::UsageSummary> = None;
        let mut model = String::new();
        let mut usage_injection = UsageInjection {
            include_usage: false,
            public_model: None,
            emitted: false,
            reported_cost: None,
            served_service_tier: None,
            reported_usage: None,
            strip_content_usage: true,
        };
        let secret = "SUPER_SECRET_CONTENT_42_this_must_not_be_logged";
        let mut buffer = format!("data: {secret}\n\n");
        let mut done_sent = false;
        let mut bytes_sent = false;
        let result = process_buffer(
            &mut buffer,
            &provider,
            &tx,
            &mut usage,
            &mut model,
            &mut usage_injection,
            &mut done_sent,
            &mut bytes_sent,
            &mut None,
            &serde_json::Value::Null,
            &mut String::new(),
            &mut false,
        )
        .await;
        assert!(result.is_err());
        drop(tx);
        while rx.recv().await.is_some() {}

        let log_output = captured.lock().unwrap().clone();
        assert!(
            log_output.contains("stream chunk transform error"),
            "expected warn log, got: {log_output}"
        );
        assert!(
            log_output.contains("chunk_len"),
            "log should use chunk_len field, got: {log_output}"
        );
        assert!(
            !log_output.contains(secret),
            "log must NOT leak chunk content, got: {log_output}"
        );
    }

    #[tokio::test]
    async fn process_buffer_forwards_transformed_chunk() {
        let (tx, mut rx) = mpsc::channel::<Result<Event, std::convert::Infallible>>(64);
        let provider = OpenAiProvider;
        let mut usage: Option<oxllm_core::UsageSummary> = None;
        let mut model = String::new();
        let mut usage_injection = UsageInjection {
            include_usage: false,
            public_model: None,
            emitted: false,
            reported_cost: None,
            served_service_tier: None,
            reported_usage: None,
            strip_content_usage: true,
        };
        let chunk = "data: {\"choices\":[]}\n\n";
        let mut buffer = chunk.to_string();
        let mut done_sent = false;
        let mut bytes_sent = false;
        process_buffer(
            &mut buffer,
            &provider,
            &tx,
            &mut usage,
            &mut model,
            &mut usage_injection,
            &mut done_sent,
            &mut bytes_sent,
            &mut None,
            &serde_json::Value::Null,
            &mut String::new(),
            &mut false,
        )
        .await
        .expect("valid chunk");
        let received = rx.recv().await;
        assert!(received.is_some(), "transformed chunk should be forwarded");
    }

    // ---- H14: stream_options.include_usage --------------------------------

    /// Reads an SSE `Event`'s data payload back as a string by rendering the
    /// full SSE response body (the `Event` struct has no public getter).
    async fn drain_sse_to_string(
        rx: mpsc::Receiver<Result<Event, std::convert::Infallible>>,
    ) -> String {
        use axum::response::IntoResponse;
        let resp = axum::response::Sse::new(ReceiverStream::new(rx)).into_response();
        let bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .expect("sse body");
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// With `stream_options.include_usage: true`, a synthesized usage chunk
    /// (empty `choices` + `usage` from the accumulated counts) is emitted
    /// exactly once, immediately before `[DONE]`.
    #[tokio::test]
    async fn process_buffer_emits_usage_chunk_before_done_when_include_usage() {
        let (tx, rx) = mpsc::channel::<Result<Event, std::convert::Infallible>>(64);
        let provider = OpenAiProvider;
        let mut usage: Option<oxllm_core::UsageSummary> = None;
        let mut model = String::new();
        let mut usage_injection = UsageInjection {
            include_usage: true,
            public_model: None,
            emitted: false,
            reported_cost: None,
            served_service_tier: None,
            reported_usage: None,
            strip_content_usage: true,
        };
        // Usage attached to a content chunk must move to the terminal chunk,
        // retaining provider-specific usage fields.
        let mut buffer = concat!(
            "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"\"}}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":0,\"total_tokens\":10,\"prompt_tokens_details\":{\"text_tokens\":10}}}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":20,\"total_tokens\":30,\"server_tool_use\":{\"web_search_requests\":2}}}\n\n",
            "data: [DONE]\n\n",
        )
        .to_string();
        let mut done_sent = false;
        let mut bytes_sent = false;
        process_buffer(
            &mut buffer,
            &provider,
            &tx,
            &mut usage,
            &mut model,
            &mut usage_injection,
            &mut done_sent,
            &mut bytes_sent,
            &mut None,
            &serde_json::Value::Null,
            &mut String::new(),
            &mut false,
        )
        .await
        .expect("valid buffer");
        drop(tx);
        let body = drain_sse_to_string(rx).await;

        // The synthesized usage chunk carries the accumulated token counts.
        assert!(
            body.contains("\"prompt_tokens\":10"),
            "usage chunk must be present: {body}"
        );
        assert!(body.contains("\"completion_tokens\":20"), "body: {body}");
        assert!(body.contains("\"total_tokens\":30"), "body: {body}");
        assert!(body.contains("\"choices\":[]"), "body: {body}");
        assert!(
            body.contains("\"server_tool_use\":{\"web_search_requests\":2}"),
            "provider-specific usage fields must survive: {body}"
        );
        assert!(
            body.contains("\"text_tokens\":10"),
            "complementary fields from the prompt snapshot must survive: {body}"
        );
        assert!(body.contains("\"content\":\"hi\""), "body: {body}");
        // It must precede [DONE].
        let usage_pos = body.find("\"prompt_tokens\":10").unwrap();
        let done_pos = body.find("[DONE]").unwrap();
        assert!(
            usage_pos < done_pos,
            "usage chunk must precede [DONE]: {body}"
        );
        // Emitted exactly once.
        assert_eq!(
            body.matches("\"prompt_tokens\":10").count(),
            1,
            "usage chunk must be emitted once: {body}"
        );
        assert!(
            usage_injection.emitted,
            "emitted flag must be set after emission"
        );
    }

    /// Without `include_usage`, no synthesized usage chunk is emitted — the
    /// stream is the plain content + `[DONE]`.
    #[tokio::test]
    async fn process_buffer_omits_usage_chunk_when_not_requested() {
        let (tx, rx) = mpsc::channel::<Result<Event, std::convert::Infallible>>(64);
        let provider = OpenAiProvider;
        let mut usage: Option<oxllm_core::UsageSummary> = None;
        let mut model = String::new();
        let mut usage_injection = UsageInjection {
            include_usage: false,
            public_model: None,
            emitted: false,
            reported_cost: None,
            served_service_tier: None,
            reported_usage: None,
            strip_content_usage: true,
        };
        let mut buffer = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":20,\"total_tokens\":30}}\n\n",
            "data: [DONE]\n\n",
        )
        .to_string();
        let mut done_sent = false;
        let mut bytes_sent = false;
        process_buffer(
            &mut buffer,
            &provider,
            &tx,
            &mut usage,
            &mut model,
            &mut usage_injection,
            &mut done_sent,
            &mut bytes_sent,
            &mut None,
            &serde_json::Value::Null,
            &mut String::new(),
            &mut false,
        )
        .await
        .expect("valid buffer");
        drop(tx);
        let body = drain_sse_to_string(rx).await;
        assert!(
            !body.contains("prompt_tokens"),
            "no usage chunk when include_usage is false: {body}"
        );
        assert!(body.contains("\"content\":\"hi\""), "body: {body}");
        assert!(body.contains("[DONE]"));
        assert!(!usage_injection.emitted);
        assert_eq!(usage.expect("usage is still metered").total_tokens, 30);
    }

    /// The usage chunk is emitted at most once even if `[DONE]` appears
    /// multiple times in the buffer (defensive).
    #[tokio::test]
    async fn process_buffer_emits_usage_chunk_at_most_once() {
        let (tx, rx) = mpsc::channel::<Result<Event, std::convert::Infallible>>(64);
        let provider = OpenAiProvider;
        let mut usage: Option<oxllm_core::UsageSummary> = Some(oxllm_core::UsageSummary {
            prompt_tokens: 1,
            completion_tokens: 2,
            total_tokens: 3,
            ..Default::default()
        });
        let mut model = String::new();
        let mut usage_injection = UsageInjection {
            include_usage: true,
            public_model: None,
            emitted: false,
            reported_cost: None,
            served_service_tier: None,
            reported_usage: None,
            strip_content_usage: true,
        };
        let mut buffer = "data: [DONE]\n\ndata: [DONE]\n\n".to_string();
        let mut done_sent = false;
        let mut bytes_sent = false;
        process_buffer(
            &mut buffer,
            &provider,
            &tx,
            &mut usage,
            &mut model,
            &mut usage_injection,
            &mut done_sent,
            &mut bytes_sent,
            &mut None,
            &serde_json::Value::Null,
            &mut String::new(),
            &mut false,
        )
        .await
        .expect("valid buffer");
        drop(tx);
        let body = drain_sse_to_string(rx).await;
        assert_eq!(
            body.matches("\"prompt_tokens\":1").count(),
            1,
            "usage chunk must be emitted exactly once: {body}"
        );
    }

    /// No usage on record → no usage chunk, even when `include_usage` is set.
    #[tokio::test]
    async fn build_usage_chunk_returns_none_without_usage() {
        let inj = UsageInjection {
            public_model: None,
            include_usage: true,
            emitted: false,
            reported_cost: None,
            served_service_tier: None,
            reported_usage: None,
            strip_content_usage: true,
        };
        assert!(build_usage_chunk_if_requested(&inj, None).is_none());
    }

    /// Once emitted, the helper returns None (idempotency guard).
    #[tokio::test]
    async fn build_usage_chunk_returns_none_after_emitted() {
        let usage = oxllm_core::UsageSummary {
            prompt_tokens: 1,
            completion_tokens: 1,
            total_tokens: 2,
            ..Default::default()
        };
        let inj = UsageInjection {
            public_model: None,
            include_usage: true,
            emitted: true,
            reported_cost: None,
            served_service_tier: None,
            reported_usage: None,
            strip_content_usage: true,
        };
        assert!(build_usage_chunk_if_requested(&inj, Some(&usage)).is_none());
    }

    /// The synthesized chunk carries the full usage breakdown and empty
    /// choices, matching the OpenAI `stream_options.include_usage` shape.
    #[tokio::test]
    async fn build_usage_chunk_shape_matches_openai_contract() {
        let usage = oxllm_core::UsageSummary {
            prompt_tokens: 100,
            completion_tokens: 50,
            total_tokens: 150,
            ..Default::default()
        };
        let inj = UsageInjection {
            public_model: None,
            include_usage: true,
            emitted: false,
            reported_cost: None,
            served_service_tier: None,
            reported_usage: None,
            strip_content_usage: true,
        };
        let chunk = build_usage_chunk_if_requested(&inj, Some(&usage)).expect("some chunk");
        assert!(chunk["choices"].is_array());
        assert_eq!(chunk["choices"].as_array().unwrap().len(), 0);
        assert_eq!(chunk["usage"]["prompt_tokens"], 100);
        assert_eq!(chunk["usage"]["completion_tokens"], 50);
        assert_eq!(chunk["usage"]["total_tokens"], 150);
    }

    // ---- H6: streaming reasoning_tokens / reasoning_content ---------------

    /// H6: when `stream_options.include_usage` is requested and the
    /// accumulated usage carries `reasoning_tokens` (populated from the
    /// upstream final usage chunk's `completion_tokens_details.reasoning_tokens`),
    /// the synthesized usage chunk must surface them under
    /// `completion_tokens_details.reasoning_tokens` — matching the OpenAI
    /// contract so streaming clients of reasoning models (o3 / o4-mini) see
    /// the same breakdown as the non-streaming path.
    #[tokio::test]
    async fn build_usage_chunk_includes_reasoning_tokens_when_present() {
        let usage = oxllm_core::UsageSummary {
            prompt_tokens: 100,
            completion_tokens: 80,
            total_tokens: 180,
            reasoning_tokens: Some(60),
            citation_tokens: None,
            web_search_requests: None,
            browser_open_requests: None,
            ..Default::default()
        };
        let inj = UsageInjection {
            public_model: None,
            include_usage: true,
            emitted: false,
            reported_cost: None,
            served_service_tier: None,
            reported_usage: None,
            strip_content_usage: true,
        };
        let chunk = build_usage_chunk_if_requested(&inj, Some(&usage)).expect("some chunk");
        assert_eq!(
            chunk["usage"]["completion_tokens_details"]["reasoning_tokens"], 60,
            "reasoning_tokens must appear under completion_tokens_details"
        );
        // Standard counters still present.
        assert_eq!(chunk["usage"]["completion_tokens"], 80);
        assert_eq!(chunk["usage"]["total_tokens"], 180);
    }

    /// Non-reasoning streams expose an explicit zero reasoning detail for
    /// clients that consume a normalized usage shape.
    #[tokio::test]
    async fn build_usage_chunk_defaults_reasoning_details_when_absent() {
        let usage = oxllm_core::UsageSummary {
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15,
            reasoning_tokens: None,
            citation_tokens: None,
            web_search_requests: None,
            browser_open_requests: None,
            ..Default::default()
        };
        let inj = UsageInjection {
            public_model: None,
            include_usage: true,
            emitted: false,
            reported_cost: None,
            served_service_tier: None,
            reported_usage: None,
            strip_content_usage: true,
        };
        let chunk = build_usage_chunk_if_requested(&inj, Some(&usage)).expect("some chunk");
        assert_eq!(
            chunk["usage"]["completion_tokens_details"]["reasoning_tokens"],
            0
        );
    }

    /// H6: a streaming delta carrying `delta.reasoning_content` (the
    /// DeepSeek/Grok chain-of-thought field) is forwarded verbatim to the
    /// client. The OpenAI provider's identity transform passes deltas
    /// through unchanged; this test pins that behavior so a future
    /// refactor that starts filtering delta keys cannot silently drop
    /// reasoning content mid-stream.
    #[tokio::test]
    async fn process_buffer_forwards_reasoning_content_delta() {
        let (tx, rx) = mpsc::channel::<Result<Event, std::convert::Infallible>>(64);
        let provider = OpenAiProvider;
        let mut usage: Option<oxllm_core::UsageSummary> = None;
        let mut model = String::new();
        let mut usage_injection = UsageInjection {
            include_usage: false,
            public_model: None,
            emitted: false,
            reported_cost: None,
            served_service_tier: None,
            reported_usage: None,
            strip_content_usage: true,
        };
        // OpenAI o-series emits reasoning_content deltas inline with the
        // other content deltas.
        let mut buffer =
            "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"thinking...\"}}]}\n\n"
                .to_string();
        let mut done_sent = false;
        let mut bytes_sent = false;
        process_buffer(
            &mut buffer,
            &provider,
            &tx,
            &mut usage,
            &mut model,
            &mut usage_injection,
            &mut done_sent,
            &mut bytes_sent,
            &mut None,
            &serde_json::Value::Null,
            &mut String::new(),
            &mut false,
        )
        .await
        .expect("valid buffer");
        drop(tx);
        let body = drain_sse_to_string(rx).await;
        assert!(
            body.contains("reasoning_content"),
            "reasoning_content delta must be forwarded: {body}"
        );
        assert!(
            body.contains("thinking..."),
            "reasoning_content text must survive: {body}"
        );
    }

    /// H6: a streaming finish chunk whose `usage` carries
    /// `completion_tokens_details.reasoning_tokens` (OpenAI o-series with
    /// `stream_options.include_usage`) populates the aggregated accumulator
    /// with reasoning_tokens via `extract_usage` → `merge_streaming_usage`.
    /// This is the end-to-end streaming fix: reasoning token counts no
    /// longer vanish from the streaming billing path.
    #[tokio::test]
    async fn process_buffer_aggregates_reasoning_tokens_from_finish_chunk() {
        let (tx, _rx) = mpsc::channel::<Result<Event, std::convert::Infallible>>(64);
        let provider = OpenAiProvider;
        let mut usage: Option<oxllm_core::UsageSummary> = None;
        let mut model = String::new();
        let mut usage_injection = UsageInjection {
            include_usage: false,
            public_model: None,
            emitted: false,
            reported_cost: None,
            served_service_tier: None,
            reported_usage: None,
            strip_content_usage: true,
        };
        // Final usage chunk from OpenAI streaming: empty choices + usage
        // carrying reasoning tokens.
        let mut buffer = "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":50,\"completion_tokens\":120,\"total_tokens\":170,\"completion_tokens_details\":{\"reasoning_tokens\":90}}}\n\n".to_string();
        let mut done_sent = false;
        let mut bytes_sent = false;
        process_buffer(
            &mut buffer,
            &provider,
            &tx,
            &mut usage,
            &mut model,
            &mut usage_injection,
            &mut done_sent,
            &mut bytes_sent,
            &mut None,
            &serde_json::Value::Null,
            &mut String::new(),
            &mut false,
        )
        .await
        .expect("valid buffer");
        let merged = usage.expect("usage accumulated");
        assert_eq!(merged.prompt_tokens, 50);
        assert_eq!(merged.completion_tokens, 120);
        assert_eq!(
            merged.reasoning_tokens,
            Some(90),
            "reasoning_tokens must be aggregated from the finish chunk"
        );
    }

    /// U22 regression: OpenRouter emits the
    /// provider-reported usage chunk AFTER the `finish_reason` chunk. The
    /// stream pump must keep processing chunks past `finish_reason` so the
    /// late usage is captured by `accumulated_usage` (which feeds cost
    /// accounting). This test pins that behaviour at the `process_buffer`
    /// level: feeding a finish chunk then a *separate* usage-only chunk
    /// must leave the accumulator populated with the late usage, not the
    /// zero/empty state from the finish chunk.
    #[tokio::test]
    async fn process_buffer_accumulates_usage_from_chunk_after_finish_reason() {
        let (tx, _rx) = mpsc::channel::<Result<Event, std::convert::Infallible>>(64);
        let provider = OpenAiProvider;
        let mut usage: Option<oxllm_core::UsageSummary> = None;
        let mut model = String::new();
        let mut usage_injection = UsageInjection {
            include_usage: false,
            public_model: None,
            emitted: false,
            reported_cost: None,
            served_service_tier: None,
            reported_usage: None,
            strip_content_usage: true,
        };
        let mut done_sent = false;
        let mut bytes_sent = false;

        // 1. Finish chunk arrives FIRST, with no usage.
        let mut buffer =
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n".to_string();
        process_buffer(
            &mut buffer,
            &provider,
            &tx,
            &mut usage,
            &mut model,
            &mut usage_injection,
            &mut done_sent,
            &mut bytes_sent,
            &mut None,
            &serde_json::Value::Null,
            &mut String::new(),
            &mut false,
        )
        .await
        .expect("finish chunk");
        // No usage on the finish chunk — accumulator must still be empty.
        assert!(
            usage.is_none(),
            "finish chunk without usage must not populate accumulator"
        );

        // 2. Usage chunk arrives AFTER finish_reason (the OpenRouter shape).
        let mut buffer =
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":13,\"total_tokens\":20}}\n\n".to_string();
        process_buffer(
            &mut buffer,
            &provider,
            &tx,
            &mut usage,
            &mut model,
            &mut usage_injection,
            &mut done_sent,
            &mut bytes_sent,
            &mut None,
            &serde_json::Value::Null,
            &mut String::new(),
            &mut false,
        )
        .await
        .expect("usage chunk");
        let merged = usage.expect("post-finish usage must be accumulated");
        assert_eq!(merged.prompt_tokens, 7);
        assert_eq!(merged.completion_tokens, 13);
        assert_eq!(merged.total_tokens, 20);
    }

    /// OpenRouter's usage-only chunk carries `usage.cost`. That value must
    /// be captured so `post_success` can bill unpriced aggregator models.
    #[tokio::test]
    async fn process_buffer_captures_usage_cost_from_chunk() {
        let (tx, _rx) = mpsc::channel::<Result<Event, std::convert::Infallible>>(64);
        let provider = OpenAiProvider;
        let mut usage: Option<oxllm_core::UsageSummary> = None;
        let mut model = String::new();
        let mut usage_injection = UsageInjection {
            include_usage: false,
            public_model: None,
            emitted: false,
            reported_cost: None,
            served_service_tier: None,
            reported_usage: None,
            strip_content_usage: true,
        };
        let mut buffer = "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1,\"total_tokens\":2,\"cost\":0.0023}}\n\n".to_string();
        process_buffer(
            &mut buffer,
            &provider,
            &tx,
            &mut usage,
            &mut model,
            &mut usage_injection,
            &mut false,
            &mut false,
            &mut None,
            &serde_json::Value::Null,
            &mut String::new(),
            &mut false,
        )
        .await
        .expect("usage chunk");
        assert_eq!(usage_injection.reported_cost, Some(0.0023));
        let merged = usage.expect("tokens accumulated");
        assert_eq!(merged.prompt_tokens, 1);
    }

    #[test]
    fn stream_exit_after_done_is_clean_completion() {
        match stream_exit_on_upstream_read_error(true, "connection reset") {
            StreamExit::Done => {}
            other => panic!("expected Done after [DONE], got {other:?}"),
        }
    }

    #[test]
    fn stream_exit_before_done_is_error() {
        match stream_exit_on_upstream_read_error(false, "connection reset") {
            StreamExit::Error(_) => {}
            other => panic!("expected Error before [DONE], got {other:?}"),
        }
    }

    /// Regression: Bedrock streaming returns binary AWS eventstream frames,
    /// not SSE text. `decode_bedrock_eventstream_into_buffer` must extract
    /// each frame's JSON payload and wrap it as an SSE `data:` event so the
    /// downstream `pop_sse_event` / `transform_stream_chunk` pipeline can fire.
    /// Feeding raw binary bytes through the UTF-8 `SseFramer` silently drops
    /// everything — the original symptom was "Bedrock streaming produced no
    /// output".
    #[test]
    fn decode_bedrock_eventstream_into_buffer_extracts_payloads_as_sse() {
        let payload1 = br#"{"type":"message_start","message":{"id":"msg_1"}}"#;
        let payload2 = br#"{"type":"content_block_delta","index":0,"delta":{"text":"hi"}}"#;
        // Two complete frames back-to-back, plus a truncated third.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&oxllm_providers::bedrock::encode_eventstream_frame(
            payload1,
        ));
        bytes.extend_from_slice(&oxllm_providers::bedrock::encode_eventstream_frame(
            payload2,
        ));
        // Truncated frame prefix (4-byte total_len only) — must be retained.
        bytes.extend_from_slice(&0x00_00_00_40u32.to_be_bytes());

        let mut raw_buffer = bytes;
        let mut buffer = String::new();
        decode_bedrock_eventstream_into_buffer(&mut raw_buffer, &mut buffer);

        // Both complete frames decoded into SSE events.
        assert_eq!(
            buffer.matches("data: ").count(),
            2,
            "both complete frames should be decoded"
        );
        assert!(buffer.contains("message_start"));
        assert!(buffer.contains("content_block_delta"));
        // Truncated frame retained for the next chunk arrival.
        assert!(
            !raw_buffer.is_empty(),
            "truncated frame bytes must remain in raw_buffer"
        );
    }

    #[test]
    fn bedrock_exception_header_reaches_provider_with_exact_status() {
        use oxllm_core::provider::ChatProvider;
        // Independent Smithy/struct/zlib fixture, including the 12-byte
        // prelude, typed exception headers and message CRC.
        let frame = b"\x00\x00\x00g\x00\x00\x00@\x38\xec\x35\xa4\x0d:message-type\x07\x00\x09exception\x0f:exception-type\x07\x00\x13throttlingException{\"message\":\"slow down\"}\x46\xf2\x43\xd2";
        let mut raw = frame.to_vec();
        let mut buffer = String::new();
        decode_bedrock_eventstream_into_buffer(&mut raw, &mut buffer);
        assert!(raw.is_empty());
        let event = pop_sse_event(&mut buffer).unwrap();
        let error = oxllm_providers::bedrock::BedrockProvider
            .transform_stream_chunk_stateful(&event, &mut None, &serde_json::json!({"model":"m"}))
            .unwrap_err();
        assert!(
            matches!(error, oxllm_core::Error::Upstream { status: 429, message } if message == "slow down")
        );
    }

    #[test]
    fn bedrock_event_header_and_payload_use_real_prelude_offsets() {
        let mut raw = hex::decode(concat!(
            "0000006000000020c77e88ec0b3a6576656e742d74797065070011",
            "636f6e74656e74426c6f636b44656c7461",
            "7b22636f6e74656e74426c6f636b496e646578223a302c2264656c7461223a7b2274657874223a2268656c6c6f227d7d92c77f75"
        )).unwrap();
        let mut buffer = String::new();
        decode_bedrock_eventstream_into_buffer(&mut raw, &mut buffer);
        assert!(raw.is_empty());
        let event = pop_sse_event(&mut buffer).unwrap();
        let body: serde_json::Value =
            serde_json::from_str(extract_sse_data(&event).unwrap()).unwrap();
        assert_eq!(body["contentBlockDelta"]["delta"]["text"], "hello");
    }

    /// MEDIUM: a Bedrock frame that declares `total_len = u32::MAX` but only
    /// delivers a trickle of bytes must not grow `raw_buffer` without bound.
    /// `append_and_decode_bedrock` enforces the `MAX_RAW_BUFFER_BYTES` cap on
    /// the undecoded tail; once the trickled bytes exceed the cap it errors
    /// rather than continuing to accumulate.
    #[test]
    fn append_and_decode_bedrock_caps_oversized_incomplete_frame() {
        // Frame header declares total_len = u32::MAX, headers_len = 0; the
        // body that follows is irrelevant because decode_eventstream_frame
        // waits for `total_len` bytes (4 GB) that never arrive.
        let mut oversized = vec![0u8; MAX_RAW_BUFFER_BYTES + 1];
        oversized[0..4].copy_from_slice(&u32::MAX.to_be_bytes());
        oversized[4..8].copy_from_slice(&0u32.to_be_bytes());
        // Independent zlib.crc32(ffffffff00000000), for the real 12-byte
        // prelude. A bad prelude checksum would test corruption, not a valid
        // frame whose declared 4 GB body never finishes.
        oversized[8..12].copy_from_slice(&0xffff_ffffu32.to_be_bytes());

        let mut raw_buffer = Vec::new();
        let mut buffer = String::new();
        let err = append_and_decode_bedrock(&mut raw_buffer, &mut buffer, &oversized, &mut 0)
            .expect_err("oversized incomplete frame must error");
        assert!(
            matches!(err, oxllm_core::Error::Network(_)),
            "expected network error for oversized raw_buffer, got {err:?}"
        );
        // No complete frame was decoded (buffer stays empty); the cap fires.
        assert!(buffer.is_empty());
    }

    /// A complete frame larger than the text cap but well under the raw cap
    /// is still drained normally — the raw cap only trips on the UNDECODED
    /// tail, so legitimate large frames that decode are not rejected.
    #[test]
    fn append_and_decode_bedrock_drains_complete_frame_under_raw_cap() {
        let payload = br#"{"type":"message_start"}"#;
        let frame = oxllm_providers::bedrock::encode_eventstream_frame(payload);
        let mut raw_buffer = Vec::new();
        let mut buffer = String::new();
        let mut events = 0;
        append_and_decode_bedrock(&mut raw_buffer, &mut buffer, &frame, &mut events)
            .expect("complete frame ok");
        assert_eq!(events, 1, "a complete frame counts as one decoded event");
        assert!(buffer.contains("message_start"), "complete frame decoded");
        assert!(
            raw_buffer.is_empty(),
            "complete frame drained from raw_buffer"
        );
    }

    #[test]
    fn bedrock_stream_close_error_rejects_an_empty_200() {
        let err = bedrock_stream_close_error(0, false).expect("empty body is an upstream failure");
        match err {
            oxllm_core::Error::Upstream { status, .. } => assert_eq!(status, 502),
            other => panic!("expected upstream 502, got {other:?}"),
        }
        assert!(bedrock_stream_close_error(1, false).is_none());
        assert!(matches!(
            bedrock_stream_close_error(2, true),
            Some(oxllm_core::Error::Network(_))
        ));
    }

    /// LOW (C11): a frame whose JSON payload is PRETTY-PRINTED (contains
    /// newlines) must still decode into exactly ONE SSE event whose data
    /// round-trips whole — splicing the raw payload between `data: ` and
    /// `\n\n` would let `pop_sse_event` split inside the JSON and hand the
    /// transformer a truncated first line. The fix re-serializes parsed
    /// payloads compactly, which also normalizes any embedded whitespace.
    #[test]
    fn decode_bedrock_eventstream_pretty_json_survives_sse_framing() {
        let pretty = "{\n  \"type\": \"message_start\",\n  \"message\": {\"id\": \"msg_1\"}\n}";
        let frame = oxllm_providers::bedrock::encode_eventstream_frame(pretty.as_bytes());
        let mut raw_buffer = frame;
        let mut buffer = String::new();
        decode_bedrock_eventstream_into_buffer(&mut raw_buffer, &mut buffer);

        assert_eq!(
            buffer.matches("data: ").count(),
            1,
            "pretty payload must produce exactly one SSE event: {buffer:?}"
        );
        assert!(
            !buffer.contains("\\n"),
            "payload newlines must be normalized away: {buffer:?}"
        );
        let event = pop_sse_event(&mut buffer).expect("one complete event");
        assert!(
            buffer.trim().is_empty(),
            "no trailing bytes after the single event: {buffer:?}"
        );
        let data = event.trim().strip_prefix("data: ").unwrap_or(event.trim());
        let value: serde_json::Value =
            serde_json::from_str(data).expect("the whole payload must round-trip");
        assert_eq!(value["type"], "message_start");
        assert_eq!(value["message"]["id"], "msg_1");
    }

    /// LOW (C11): the `{"bytes": "<base64>"}` wrapper path gets the same
    /// treatment — decoded content that parses as JSON is re-serialized
    /// compactly, so newline-bearing inner payloads cannot split framing.
    #[test]
    fn decode_bedrock_eventstream_bytes_wrapper_compacts_inner_newlines() {
        let inner = "{\n  \"type\": \"content_block_delta\",\n  \"delta\": {\"text\": \"hi\"}\n}";
        let wrapped = format!("{{\"bytes\": \"{}\"}}", {
            use base64::Engine;
            base64::engine::general_purpose::STANDARD.encode(inner.as_bytes())
        });
        let frame = oxllm_providers::bedrock::encode_eventstream_frame(wrapped.as_bytes());
        let mut raw_buffer = frame;
        let mut buffer = String::new();
        decode_bedrock_eventstream_into_buffer(&mut raw_buffer, &mut buffer);

        assert_eq!(buffer.matches("data: ").count(), 1, "one event: {buffer:?}");
        let event = pop_sse_event(&mut buffer).expect("one complete event");
        let data = event.trim().strip_prefix("data: ").unwrap_or(event.trim());
        let value: serde_json::Value =
            serde_json::from_str(data).expect("inner payload must round-trip whole");
        assert_eq!(value["type"], "content_block_delta");
        assert_eq!(value["delta"]["text"], "hi");
    }

    /// The streaming SSE error event sources its `code` from the shared
    /// [`crate::error::code_for_upstream_status`] helper, the same source of
    /// truth as the non-streaming `ApiError::upstream` body. This pins the
    /// MEDIUM-issue parity contract: an upstream 429 produces the same
    /// `code` string on both paths.
    #[test]
    fn streaming_error_code_helper_matches_non_streaming_codes() {
        // A provider 429 reports `upstream_rate_limited` on BOTH paths.
        assert_eq!(
            crate::error::code_for_upstream_status(
                crate::error::synthesize_upstream_status(429).as_u16()
            ),
            Some("upstream_rate_limited")
        );
        assert_eq!(
            ApiError::upstream(429, "throttled").code.as_deref(),
            Some("upstream_rate_limited")
        );
        // 5xx reports `upstream_error` on both paths (NOT the OpenAI-native
        // `internal_server_error`/`bad_gateway` the streaming path used to
        // emit).
        for status in [500u16, 502, 503, 504] {
            assert_eq!(
                crate::error::code_for_upstream_status(
                    crate::error::synthesize_upstream_status(status).as_u16()
                ),
                Some("upstream_error"),
                "status {status}"
            );
            assert_eq!(
                ApiError::upstream(status, "boom").code.as_deref(),
                Some("upstream_error"),
                "status {status}"
            );
        }
    }

    /// Partial-stream spend accounting: usage accumulated before a stream
    /// terminates early must settle the router-level deployment + provider
    /// spend counters to the computed partial cost.
    #[tokio::test]
    async fn record_partial_stream_spend_records_deployment_and_provider() {
        use crate::config::types::{GeneralSettings, OxllmSettings, ProxyConfig, RouterSettings};
        let budgets = std::collections::HashMap::from([(
            "openai".to_string(),
            oxllm_core::router::ProviderBudgetConfig {
                budget_limit: 100.0,
                time_period: "1d".to_string(),
            },
        )]);
        let config = ProxyConfig {
            model_list: vec![],
            oxllm_settings: OxllmSettings::default(),
            general_settings: GeneralSettings::default(),
            router_settings: RouterSettings {
                provider_budget_config: Some(budgets),
                ..Default::default()
            },
            extra: serde_json::Map::new(),
        };
        let state =
            AppState::with_registry(config, oxllm_providers::default_registry()).expect("state");
        let deployment = Deployment {
            model_name: "gpt-4o".to_string(),
            oxllm_params: oxllm_core::deployment::OxllmParams {
                model: "openai/gpt-4o".to_string(),
                max_budget: Some(10.0),
                ..Default::default()
            },
            ..Default::default()
        };
        let call_ctx = CallContext::for_test("gpt-4o");
        let usage = oxllm_core::UsageSummary {
            prompt_tokens: 1_000,
            completion_tokens: 500,
            total_tokens: 1_500,
            cache_read_input_tokens: None,
            cache_creation_input_tokens: None,
            audio_input_tokens: None,
            reasoning_tokens: None,
            citation_tokens: None,
            web_search_requests: None,
            browser_open_requests: None,
            cache_creation_1h_input_tokens: None,
        };
        record_partial_stream_spend(
            &state,
            &deployment,
            &call_ctx,
            "gpt-4o",
            Some(&usage),
            &crate::orchestration::budget::RouterBudgetReservation::default(),
            None,
        )
        .await;
        let expected = oxllm_core::calculate_cost("gpt-4o", &usage).expect("gpt-4o priced");
        let dep_spend = state
            .cache
            .counter_value("spend:dep:gpt-4o:openai/gpt-4o")
            .await
            .unwrap_or(0.0);
        let provider_spend = state
            .cache
            .counter_value("spend:provider:openai")
            .await
            .unwrap_or(0.0);
        assert!(
            (dep_spend - expected).abs() < 1e-9,
            "deployment spend must equal the partial cost {expected}, got {dep_spend}"
        );
        assert!(
            (provider_spend - expected).abs() < 1e-9,
            "provider spend must equal the partial cost {expected}, got {provider_spend}"
        );
    }

    #[tokio::test]
    async fn partial_stream_error_has_one_terminal_row_and_one_router_settlement() {
        let mut state = AppState::with_registry(
            crate::config::types::ProxyConfig {
                model_list: vec![],
                oxllm_settings: Default::default(),
                general_settings: Default::default(),
                router_settings: Default::default(),
                extra: serde_json::Map::new(),
            },
            oxllm_providers::default_registry(),
        )
        .unwrap();
        let (spend_tx, mut ledger) = mpsc::channel(8);
        state.spend_tx = Some(spend_tx.clone());
        state.hooks = vec![std::sync::Arc::new(
            crate::hooks::CostTrackingHook::with_spend_writer(Some(spend_tx)),
        )];
        let deployment = Deployment {
            model_name: "public-alias".into(),
            oxllm_params: oxllm_core::deployment::OxllmParams {
                model: "openai/gpt-4o".into(),
                max_budget: Some(10.0),
                ..Default::default()
            },
            ..Default::default()
        };
        let ctx = CallContext::for_test("public-alias");
        let mut usage = Some(oxllm_core::UsageSummary {
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15,
            ..Default::default()
        });
        let expected = oxllm_core::calculate_cost("gpt-4o", usage.as_ref().unwrap()).unwrap();
        let reservation = crate::orchestration::budget::RouterBudgetReservation::default();
        let dep_key = deployment_key(&deployment);
        let cooldown_key = state.router.cooldown_key(&deployment);
        let (events, _receiver) = mpsc::channel(8);
        finalize_stream_exit(
            &state,
            &deployment,
            &ctx,
            &reservation,
            &ctx.model,
            &cooldown_key,
            &dep_key,
            &events,
            StreamExit::Error(oxllm_core::Error::Network("disconnected".into())),
            &mut usage,
            "gpt-4o",
            true,
            "partial output",
            None,
        )
        .await;
        let entry = ledger.try_recv().unwrap();
        assert_eq!(entry.status.as_deref(), Some("partial"));
        assert_eq!(entry.request_id.as_ref(), Some(&ctx.request_id));
        assert_eq!(entry.total_tokens, Some(15));
        assert!((entry.cost.unwrap() - expected).abs() < 1e-12);
        assert!(
            ledger.try_recv().is_err(),
            "terminal failure must not add a second row"
        );
        record_partial_stream_spend(
            &state,
            &deployment,
            &ctx.clone(),
            "gpt-4o",
            usage.as_ref(),
            &reservation,
            None,
        )
        .await;
        assert!(
            ledger.try_recv().is_err(),
            "cloned contexts share the partial-spend claim"
        );
        let settled = state
            .cache
            .counter_value(&crate::hooks::spend_keys::deployment_spend_key(&dep_key))
            .await
            .unwrap();
        assert!(
            (settled - expected).abs() < 1e-12,
            "a repeated partial observer cannot settle twice"
        );
    }

    #[tokio::test]
    async fn idle_stream_timeout_is_504_and_does_not_cool_the_deployment() {
        let mut state = AppState::with_registry(
            crate::config::types::ProxyConfig {
                model_list: vec![],
                oxllm_settings: Default::default(),
                general_settings: Default::default(),
                router_settings: crate::config::types::RouterSettings {
                    routing: crate::config::types::RoutingKnobs {
                        allowed_fails: Some(0),
                        ..Default::default()
                    },
                    ..Default::default()
                },
                extra: serde_json::Map::new(),
            },
            oxllm_providers::default_registry(),
        )
        .unwrap();
        let (spend_tx, mut ledger) = mpsc::channel(8);
        state.spend_tx = Some(spend_tx);
        let deployment = Deployment {
            model_name: "public-alias".into(),
            oxllm_params: oxllm_core::deployment::OxllmParams {
                model: "openai/gpt-4o".into(),
                ..Default::default()
            },
            ..Default::default()
        };
        let ctx = CallContext::for_test("public-alias");
        let _ = ctx.token_estimate.set(120);
        let dep_key = deployment_key(&deployment);
        let cooldown_key = state.router.cooldown_key(&deployment);
        let (events, receiver) = mpsc::channel(8);
        let mut usage = Some(oxllm_core::UsageSummary {
            prompt_tokens: 100,
            cache_read_input_tokens: Some(40),
            ..Default::default()
        });
        finalize_stream_exit(
            &state,
            &deployment,
            &ctx,
            &Default::default(),
            &ctx.model,
            &cooldown_key,
            &dep_key,
            &events,
            StreamExit::IdleTimeout,
            &mut usage,
            "gpt-4o",
            true,
            "partial output",
            None,
        )
        .await;
        assert!(
            !state.router.cooldown_store().is_cooling(&cooldown_key),
            "a slow stream must not count as a connection-health failure"
        );
        drop(events);
        let body = drain_sse_to_string(receiver).await;
        assert!(body.contains("\"code\":\"upstream_timeout\""), "{body}");
        assert!(body.contains(STREAM_IDLE_TIMEOUT_MESSAGE), "{body}");
        let row = ledger.try_recv().expect("estimated partial spend row");
        assert_eq!(row.status.as_deref(), Some("partial"));
        assert_eq!(row.prompt_tokens, Some(100));
        assert!(row.completion_tokens.unwrap() > 0);
        assert_eq!(usage.unwrap().cache_read_input_tokens, Some(40));
    }

    #[tokio::test(start_paused = true)]
    async fn partial_stream_survives_transient_ledger_backpressure() {
        let mut state = AppState::with_registry(
            serde_json::from_value(serde_json::json!({"model_list":[]})).unwrap(),
            oxllm_providers::default_registry(),
        )
        .unwrap();
        let (spend_tx, mut ledger) = mpsc::channel(1);
        spend_tx
            .try_send(
                serde_json::from_value(serde_json::json!({
                    "timestamp":"2026-10-11T00:00:00Z","model":"filler","provider":"test"
                }))
                .unwrap(),
            )
            .unwrap();
        state.spend_tx = Some(spend_tx.clone());
        state.hooks = vec![std::sync::Arc::new(
            crate::hooks::CostTrackingHook::with_spend_writer(Some(spend_tx)),
        )];
        let receiver = tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            assert_eq!(ledger.recv().await.unwrap().model, "filler");
            let row = tokio::time::timeout(std::time::Duration::from_millis(500), ledger.recv())
                .await
                .expect("partial record must recover when queue capacity returns")
                .unwrap();
            assert!(ledger.try_recv().is_err(), "one terminal ledger row");
            row
        });
        let deployment: Deployment = serde_json::from_value(serde_json::json!({
            "model_name":"public-alias","oxllm_params":{"model":"openai/gpt-4o","max_budget":10}
        }))
        .unwrap();
        let ctx = CallContext::for_test("public-alias");
        let mut usage = Some(oxllm_core::UsageSummary {
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15,
            ..Default::default()
        });
        let expected = oxllm_core::calculate_cost("gpt-4o", usage.as_ref().unwrap()).unwrap();
        let dep_key = deployment_key(&deployment);
        let (events, _events_rx) = mpsc::channel(8);
        finalize_stream_exit(
            &state,
            &deployment,
            &ctx,
            &Default::default(),
            &ctx.model,
            &state.router.cooldown_key(&deployment),
            &dep_key,
            &events,
            StreamExit::Error(oxllm_core::Error::Network("closed".into())),
            &mut usage,
            "gpt-4o",
            true,
            "partial output",
            None,
        )
        .await;
        let row = receiver.await.unwrap();
        assert_eq!(row.status.as_deref(), Some("partial"));
        assert_eq!(row.total_tokens, Some(15));
        assert!((row.cost.unwrap() - expected).abs() < 1e-12);
        assert!(
            (state
                .cache
                .counter_value(&crate::hooks::spend_keys::deployment_spend_key(&dep_key))
                .await
                .unwrap()
                - expected)
                .abs()
                < 1e-12
        );
    }

    /// Without accumulated usage, partial-stream spend recording is a no-op.
    #[tokio::test]
    async fn record_partial_stream_spend_without_usage_is_noop() {
        use crate::config::types::{GeneralSettings, OxllmSettings, ProxyConfig, RouterSettings};
        let config = ProxyConfig {
            model_list: vec![],
            oxllm_settings: OxllmSettings::default(),
            general_settings: GeneralSettings::default(),
            router_settings: RouterSettings::default(),
            extra: serde_json::Map::new(),
        };
        let state =
            AppState::with_registry(config, oxllm_providers::default_registry()).expect("state");
        let deployment = Deployment {
            model_name: "gpt-4o".to_string(),
            oxllm_params: oxllm_core::deployment::OxllmParams {
                model: "openai/gpt-4o".to_string(),
                max_budget: Some(10.0),
                ..Default::default()
            },
            ..Default::default()
        };
        let call_ctx = CallContext::for_test("gpt-4o");
        record_partial_stream_spend(
            &state,
            &deployment,
            &call_ctx,
            "gpt-4o",
            None,
            &crate::orchestration::budget::RouterBudgetReservation::default(),
            None,
        )
        .await;
        assert!(
            state
                .cache
                .counter_value("spend:dep:gpt-4o:openai/gpt-4o")
                .await
                .is_none(),
            "no usage → no spend recorded"
        );
    }

    /// LOW (C11): a client disconnect with no accumulated usage must still
    /// queue a `status: "partial"` zero-usage spend row so the DB ledger
    /// matches the budget counters (the key-scope debit stands via the
    /// un-released reservation).
    #[tokio::test]
    async fn record_partial_stream_spend_without_usage_queues_partial_row() {
        use crate::config::types::{GeneralSettings, OxllmSettings, ProxyConfig, RouterSettings};
        let config = ProxyConfig {
            model_list: vec![],
            oxllm_settings: OxllmSettings::default(),
            general_settings: GeneralSettings::default(),
            router_settings: RouterSettings::default(),
            extra: serde_json::Map::new(),
        };
        let mut state =
            AppState::with_registry(config, oxllm_providers::default_registry()).expect("state");
        let (tx, mut rx) = mpsc::channel::<crate::spend_writer::SpendEntry>(8);
        state.spend_tx = Some(tx);
        let deployment = Deployment {
            model_name: "gpt-4o".to_string(),
            oxllm_params: oxllm_core::deployment::OxllmParams {
                model: "openai/gpt-4o".to_string(),
                max_budget: Some(10.0),
                ..Default::default()
            },
            ..Default::default()
        };
        let call_ctx = CallContext::for_test("gpt-4o");
        record_partial_stream_spend(
            &state,
            &deployment,
            &call_ctx,
            "gpt-4o",
            None,
            &crate::orchestration::budget::RouterBudgetReservation::default(),
            None,
        )
        .await;
        let entry = rx.recv().await.expect("partial spend row must be queued");
        assert_eq!(entry.status.as_deref(), Some("partial"));
        assert_eq!(entry.total_tokens, Some(0));
        assert_eq!(entry.prompt_tokens, Some(0));
        assert_eq!(entry.completion_tokens, Some(0));
    }

    /// Partial-stream spend rows dropped by the spend-writer channel must be
    /// counted on the matching `reason` label of
    /// `oxllm_spend_entries_dropped_total`, mirroring `CostTrackingHook`'s
    /// `queue_spend` (`hooks/cost.rs`). Without this a disconnect or
    /// mid-stream error could lose the ledger row silently while
    /// `/spend/logs` diverges from the budget counters.
    #[tokio::test]
    #[serial_test::serial]
    async fn record_partial_stream_spend_counts_full_and_closed_drops() {
        use crate::config::types::{GeneralSettings, OxllmSettings, ProxyConfig, RouterSettings};

        let _ = crate::metrics::install();
        let config = ProxyConfig {
            model_list: vec![],
            oxllm_settings: OxllmSettings::default(),
            general_settings: GeneralSettings::default(),
            router_settings: RouterSettings::default(),
            extra: serde_json::Map::new(),
        };
        let mut state =
            AppState::with_registry(config, oxllm_providers::default_registry()).expect("state");
        let deployment = Deployment {
            model_name: "gpt-4o".to_string(),
            oxllm_params: oxllm_core::deployment::OxllmParams {
                model: "openai/gpt-4o".to_string(),
                max_budget: Some(10.0),
                ..Default::default()
            },
            ..Default::default()
        };
        let reservation = crate::orchestration::budget::RouterBudgetReservation::default();

        // `full`: capacity-1 channel with no draining receiver. The first
        // partial row fills it; the second must land on `reason="full"`.
        let (tx, _rx) = mpsc::channel::<crate::spend_writer::SpendEntry>(1);
        state.spend_tx = Some(tx);
        record_partial_stream_spend(
            &state,
            &deployment,
            &CallContext::for_test("gpt-4o"),
            "gpt-4o",
            None,
            &reservation,
            None,
        )
        .await;
        let full_before = spend_dropped_count("full");
        record_partial_stream_spend(
            &state,
            &deployment,
            &CallContext::for_test("gpt-4o"),
            "gpt-4o",
            None,
            &reservation,
            None,
        )
        .await;
        let full_after = spend_dropped_count("full");
        assert_eq!(
            full_after.saturating_sub(full_before),
            1,
            "a full spend channel must count one 'full' drop: before={full_before} after={full_after}"
        );

        // `closed`: receiver dropped; every queue attempt must count.
        let (tx, rx) = mpsc::channel::<crate::spend_writer::SpendEntry>(8);
        drop(rx);
        state.spend_tx = Some(tx);
        let closed_before = spend_dropped_count("closed");
        record_partial_stream_spend(
            &state,
            &deployment,
            &CallContext::for_test("gpt-4o"),
            "gpt-4o",
            None,
            &reservation,
            None,
        )
        .await;
        let closed_after = spend_dropped_count("closed");
        assert_eq!(
            closed_after.saturating_sub(closed_before),
            1,
            "a closed spend channel must count one 'closed' drop: before={closed_before} after={closed_after}"
        );
    }

    /// Parse the current value of
    /// `oxllm_spend_entries_dropped_total{reason=...}` out of the rendered
    /// Prometheus exposition text (0 when the series is absent). Mirrors the
    /// helper in `hooks::cost::tests`.
    fn spend_dropped_count(reason: &str) -> u64 {
        let needle = format!("reason=\"{reason}\"");
        for line in crate::metrics::render().lines() {
            if line.starts_with("oxllm_spend_entries_dropped_total{") && line.contains(&needle) {
                if let Some(v) = line.rsplit(' ').next() {
                    if let Ok(n) = v.parse::<u64>() {
                        return n;
                    }
                }
            }
        }
        0
    }

    // ---- LOW (C11): exactly-one usage chunk under each include_usage setting ----

    /// The upstream's own usage-only chunk must NOT reach a client that did
    /// not request `include_usage` (it exists only because oxllm injects
    /// the flag upstream for metering).
    #[tokio::test]
    async fn process_buffer_suppresses_upstream_usage_chunk_when_not_requested() {
        let (tx, rx) = mpsc::channel::<Result<Event, std::convert::Infallible>>(64);
        let provider = OpenAiProvider;
        let mut usage: Option<oxllm_core::UsageSummary> = None;
        let mut model = String::new();
        let mut usage_injection = UsageInjection {
            include_usage: false,
            public_model: None,
            emitted: false,
            reported_cost: None,
            served_service_tier: None,
            reported_usage: None,
            strip_content_usage: true,
        };
        let mut buffer = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":7,\"total_tokens\":12}}\n\n",
            "data: [DONE]\n\n",
        )
        .to_string();
        let mut done_sent = false;
        let mut bytes_sent = false;
        process_buffer(
            &mut buffer,
            &provider,
            &tx,
            &mut usage,
            &mut model,
            &mut usage_injection,
            &mut done_sent,
            &mut bytes_sent,
            &mut None,
            &serde_json::Value::Null,
            &mut String::new(),
            &mut false,
        )
        .await
        .expect("valid buffer");
        drop(tx);
        let body = drain_sse_to_string(rx).await;
        assert!(
            !body.contains("prompt_tokens"),
            "usage chunk must not reach a client that did not request it: {body}"
        );
        assert!(body.contains("\"content\":\"hi\""), "content flows: {body}");
        assert!(body.contains("[DONE]"));
        // Metering still captured the usage from the suppressed chunk.
        let merged = usage.expect("usage must be merged even when suppressed");
        assert_eq!(merged.total_tokens, 12);
    }

    /// When the client DID request `include_usage` and the upstream already
    /// delivered its own usage chunk, that chunk is forwarded and the
    /// synthesized duplicate is suppressed — exactly one usage chunk.
    #[tokio::test]
    async fn process_buffer_forwards_upstream_usage_chunk_once_when_requested() {
        let (tx, rx) = mpsc::channel::<Result<Event, std::convert::Infallible>>(64);
        let provider = OpenAiProvider;
        let mut usage: Option<oxllm_core::UsageSummary> = None;
        let mut model = String::new();
        let mut usage_injection = UsageInjection {
            include_usage: true,
            public_model: None,
            emitted: false,
            reported_cost: None,
            served_service_tier: None,
            reported_usage: None,
            strip_content_usage: true,
        };
        let mut buffer = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":7,\"total_tokens\":12}}\n\n",
            "data: [DONE]\n\n",
        )
        .to_string();
        let mut done_sent = false;
        let mut bytes_sent = false;
        process_buffer(
            &mut buffer,
            &provider,
            &tx,
            &mut usage,
            &mut model,
            &mut usage_injection,
            &mut done_sent,
            &mut bytes_sent,
            &mut None,
            &serde_json::Value::Null,
            &mut String::new(),
            &mut false,
        )
        .await
        .expect("valid buffer");
        drop(tx);
        let body = drain_sse_to_string(rx).await;
        assert_eq!(
            body.matches("\"total_tokens\":12").count(),
            1,
            "exactly one usage chunk must reach the client: {body}"
        );
        assert!(body.contains("[DONE]"));
    }

    // ---- LOW (C11): client-stall timeout on the pump sends ----

    /// A client that stops reading backfills the channel; the bounded send
    /// must trip the stall flag instead of parking forever. Paused clock:
    /// the stall timeout auto-advances.
    #[tokio::test(start_paused = true)]
    async fn process_buffer_flags_client_stall_when_channel_full() {
        // Capacity 1: the first event fills it, the second send stalls.
        let (tx, _rx) = mpsc::channel::<Result<Event, std::convert::Infallible>>(1);
        let provider = OpenAiProvider;
        let mut usage: Option<oxllm_core::UsageSummary> = None;
        let mut model = String::new();
        let mut usage_injection = UsageInjection {
            include_usage: false,
            public_model: None,
            emitted: false,
            reported_cost: None,
            served_service_tier: None,
            reported_usage: None,
            strip_content_usage: true,
        };
        let mut buffer = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"b\"}}]}\n\n",
        )
        .to_string();
        let mut done_sent = false;
        let mut bytes_sent = false;
        let mut completion_text = String::new();
        let mut client_stalled = false;
        let start = tokio::time::Instant::now();
        process_buffer(
            &mut buffer,
            &provider,
            &tx,
            &mut usage,
            &mut model,
            &mut usage_injection,
            &mut done_sent,
            &mut bytes_sent,
            &mut None,
            &serde_json::Value::Null,
            &mut completion_text,
            &mut client_stalled,
        )
        .await
        .expect("valid buffer");
        assert!(client_stalled, "a full channel must trip the stall flag");
        assert!(
            start.elapsed() >= CLIENT_SEND_STALL_TIMEOUT,
            "the stall must have waited for the bounded timeout, got {:?}",
            start.elapsed()
        );
    }

    // ---- MEDIUM (C11): zero-usage stream estimation ----

    /// With a pre-call prompt estimate on the context, the estimator bills
    /// prompt + tokenized completion; with neither signal it yields None.
    #[test]
    fn estimate_stream_usage_uses_prompt_estimate_and_delta_text() {
        let ctx = CallContext::for_test("gpt-4o");
        let _ = ctx.token_estimate.set(120);
        let est = estimate_stream_usage(&ctx, "gpt-4o", "hello world, how are you?")
            .expect("estimated usage");
        assert_eq!(est.prompt_tokens, 120);
        assert!(est.completion_tokens > 0, "completion must be estimated");
        assert_eq!(est.total_tokens, 120 + est.completion_tokens);

        // No prompt estimate: falls back to estimating the request body.
        let mut ctx2 = CallContext::for_test("gpt-4o");
        ctx2.request_body = std::sync::Arc::new(serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "estimate me please"}]
        }));
        let est2 = estimate_stream_usage(
            &ctx2,
            "gpt-4o",
            "hello world, how are you? surely more than a single token",
        )
        .expect("estimated usage via request fallback");
        assert!(est2.prompt_tokens > 0, "request fallback must estimate");
        // Neither prompt nor completion → None (nothing to bill).
        let ctx3 = CallContext::for_test("gpt-4o");
        assert!(estimate_stream_usage(&ctx3, "gpt-4o", "").is_none());
    }
}
