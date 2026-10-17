//! `GET /v1/realtime` — WebSocket proxy to the upstream provider's realtime API.
//!
//! A bidirectional proxy with a pluggable [`RealtimeProvider`] transform
//! pipeline. The client opens a WS to `/v1/realtime?model=gpt-4o-realtime`;
//! oxllm resolves the model to a deployment, selects a realtime provider
//! (OpenAI-compatible → identity passthrough; Bedrock Nova Sonic →
//! translating provider, added in a later batch), opens a WS to the
//! upstream provider's realtime endpoint, and forwards frames both ways
//! through the provider's `transform_client_event` /
//! `transform_upstream_event` hooks.
//!
//! # Auth
//!
//! Auth is enforced by the shared auth middleware, which runs before the
//! WebSocket upgrade. For browser WS clients that cannot set headers during
//! the HTTP upgrade, the middleware accepts `?authorization=` as a fallback
//! *only* on `/v1/realtime*` (see `auth::extract_auth_token` /
//! `auth::is_realtime_path`); the realtime handler itself never parses auth.
//! Because query strings can appear in access logs, treat the realtime path
//! as sensitive when logging request URLs.
//!
//! # Metering
//!
//! Pre-call metering (rate-limit + budget reservation) and per-deployment
//! RPM/TPM checks run before the WS upgrade so virtual keys cannot open
//! unlimited sessions. During the session, each upstream text frame is
//! offered to the provider's `extract_usage` hook; when the provider reports
//! token usage (e.g. OpenAI Realtime `response.done`), the session settles
//! the actual cost via the post-success hook chain (budget debit, spend log,
//! deployment TPM) at session close. Disconnects get a bounded cancel/drain
//! to recover usage. Token-priced control-only sessions release reservations;
//! output still lacking usage retains the existing estimate, like partial chat
//! streams. Duration-priced sessions settle their known elapsed charge even
//! without final tokens (no invented token counts).
//!
//! # Frame handling
//!
//! - **Text** frames carry one OpenAI realtime JSON event each and are
//!   pumped through the provider's transform methods (which may expand one
//!   frame into many, or drop it).
//! - **Binary** frames carry raw PCM audio and are forwarded verbatim — the
//!   trait is string/JSON-scoped and does not re-encode audio.

pub mod azure;
pub mod bedrock;
mod bedrock_http;
mod connect;
mod lifecycle;
pub mod openai;
mod routing;
mod session;
pub mod xai;

pub use azure::AzureRealtimeProvider;
pub use openai::OpenAIRealtimeProvider;
pub use xai::XaiRealtimeProvider;

use std::time::Duration;

use axum::Extension;
use axum::extract::ws::{Message as AxumMessage, WebSocket, WebSocketUpgrade};
use axum::extract::{Query, State};
use axum::http::{HeaderMap, HeaderName, HeaderValue};
use axum::response::Response;
use futures_util::{SinkExt, StreamExt};
use oxllm_core::RealtimeProvider;
use oxllm_core::hook::UsageSummary;
use serde::Deserialize;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::protocol::Message as TungsteniteMessage;

use crate::auth::AuthUser;
use crate::error::ApiError;
use crate::state::SharedState;
use lifecycle::{RealtimeRouterBudgetGuard, UpgradeAbortGuard, await_settlement};
use routing::RealtimeRoutingGuard;
use session::{SessionEnd, SessionMeter, SessionOutcome};

/// Timeout for the upstream WebSocket handshake.
const UPSTREAM_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// Best-effort usage recovery after a client disconnect: cancel/end input,
/// then drain the upstream without forwarding more output to the client.
pub(super) const USAGE_DRAIN_TIMEOUT: Duration = Duration::from_secs(2);
pub(super) const WS_SEND_TIMEOUT: Duration = Duration::from_secs(5);

/// Wall-clock cap on a single realtime session. A client that opens the WS
/// and goes silent (or a malicious client holding the session open) would
/// otherwise pin one upstream socket, one client socket, the budget
/// reservation, and a worker task indefinitely. The reference does not enforce
/// this either (deployments typically sit behind a load balancer with an
/// idle timeout), but oxllm-as-edge provides one.
const REALTIME_MAX_SESSION_DURATION: Duration = Duration::from_secs(600);

/// Duration billing currently measures from connection start through teardown.
/// Reserve the live cap plus successful connect, an in-flight bounded write,
/// usage drain and socket close, not an unrelated HTTP timeout. This remains an
/// estimate (scheduling and multi-event transforms can add time), not a new cap.
const REALTIME_DURATION_ESTIMATE: Duration = REALTIME_MAX_SESSION_DURATION
    .saturating_add(UPSTREAM_CONNECT_TIMEOUT)
    .saturating_add(WS_SEND_TIMEOUT)
    .saturating_add(USAGE_DRAIN_TIMEOUT.saturating_mul(2));

/// Cap on a single WebSocket message size (reassembled) for both the client
/// and upstream sides of the realtime proxy. Tungstenite's default is 64 MiB;
/// realtime audio frames are typically a few KiB and JSON events are smaller,
/// so 1 MiB is generous for any sane realtime event while bounding per-session
/// worker memory (a malicious/buggy client cannot pin ~64 MiB per inflight
/// session by sending a single huge frame).
pub(super) const REALTIME_MAX_MESSAGE_SIZE: usize = 1024 * 1024;

/// Build the upstream `WebSocketConfig` with the same message/frame caps
/// applied to the client side, so an oversized upstream frame cannot grow the
/// worker buffer without bound either.
pub(super) fn upstream_ws_config() -> tokio_tungstenite::tungstenite::protocol::WebSocketConfig {
    use tokio_tungstenite::tungstenite::protocol::WebSocketConfig;
    WebSocketConfig::default()
        .max_message_size(Some(REALTIME_MAX_MESSAGE_SIZE))
        .max_frame_size(Some(REALTIME_MAX_MESSAGE_SIZE))
}

#[expect(clippy::too_many_arguments)]
async fn settle_reserved_realtime_session(
    state: &crate::state::AppState,
    ctx: &oxllm_core::CallContext,
    deployment: &oxllm_core::Deployment,
    model: &str,
    provider: &str,
    dep_key: &str,
    outcome: oxllm_core::Result<SessionOutcome>,
    guard: RealtimeRouterBudgetGuard,
) {
    // This function is driven by a detached settlement task, so its sequence
    // of monetary operations survives cancellation of the on_upgrade owner.
    // Take ownership before the first debit; Drop must never refund a partial
    // settlement a second time.
    let reservation = guard.into_reservation();
    let usage = outcome
        .as_ref()
        .ok()
        .and_then(|outcome| outcome.usage.as_ref());
    let mut settlement_ctx = ctx.clone();
    let duration_pricing = realtime_duration_pricing(ctx, model, provider);
    let duration_priced = duration_pricing.is_some();
    if let Some(pricing) = duration_pricing {
        // Pin the effective duration card for every later hook/exporter, not
        // just this first resolution. In particular, unavailable timing must
        // stay unknown rather than fall back to token prices on a public alias.
        settlement_ctx.deployment_pricing = Some(pricing);
    }
    let cost = outcome.as_ref().ok().and_then(|_| {
        crate::hooks::cost_fallback::resolve_cost(
            &settlement_ctx,
            model,
            usage,
            None,
            None,
            provider,
        )
    });
    // A setup/connection failure is not a live session. Explicitly freeze its
    // zero charge as well, so a failure observer cannot price setup elapsed time.
    settlement_ctx.cost_override = if outcome.is_err() {
        Some(0.0)
    } else {
        cost.or(ctx.cost_override)
    };
    let retain_unknown_estimate = outcome.as_ref().is_ok_and(|outcome| {
        outcome.unmetered_output && (!duration_priced || !outcome.end.is_failure())
    });
    // Tokens cannot replace missing duration evidence, including via the
    // router's nominal-unknown-token-cost fallback. Retain or release the same
    // estimate as the key scopes when no elapsed charge can be established.
    if cost.is_some() || (usage.is_some() && !duration_priced) {
        crate::orchestration::budget::settle_router_spend(
            state,
            deployment,
            dep_key,
            cost,
            &reservation,
        )
        .await;
    } else if !retain_unknown_estimate {
        reservation.release(&state.cache).await;
    }
    settle_realtime_session(state, &settlement_ctx, model, provider, dep_key, outcome).await;
}

/// Rewrite an HTTP(S) api_base to its WebSocket scheme equivalent, warning
/// when the result would be cleartext (`ws://`) for a non-local host. A
/// deployment configured with `api_base: "http://..."` (misconfiguration or a
/// plaintext internal host) would otherwise send the bearer token / Azure
/// `api-key` in cleartext over an unencrypted WS without any warning.
///
/// `http://localhost`, `http://127.0.0.1`, and `http://[::1]` are allowed
/// without warning (local debug deployments). Non-localhost `http://` / `ws://`
/// triggers a `tracing::warn` so the credential-disclosure risk is visible to
/// operators.
pub(super) fn scheme_rewrite_for_ws(base: &str) -> String {
    if base.starts_with("https://") {
        return base.replacen("https://", "wss://", 1);
    }
    if base.starts_with("wss://") {
        return base.to_string();
    }
    if base.starts_with("http://") || base.starts_with("ws://") {
        let host = base
            .trim_start_matches("http://")
            .trim_start_matches("ws://");
        let is_local = is_localhost_host(host);
        let rewritten = if base.starts_with("http://") {
            base.replacen("http://", "ws://", 1)
        } else {
            base.to_string()
        };
        if !is_local {
            tracing::warn!(
                api_base = %redact_api_base_host(base),
                "realtime api_base uses a cleartext scheme (ws://) for a non-local host; \
                 credentials will be sent unencrypted"
            );
        }
        return rewritten;
    }
    base.to_string()
}

/// Build an upstream realtime WebSocket URL by appending a path suffix (e.g.
/// `/realtime`, `/openai/realtime`) and the `model` query param to a
/// scheme-rewritten, trailing-slash-trimmed `api_base`.
///
/// The base is parsed with `url::Url` **first** so that an existing query
/// string on `api_base` (e.g. Azure's `?api-version=…`) is preserved and the
/// path suffix lands in the URL path rather than being swallowed into the
/// query. This mirrors the reference `OpenAIRealtime._construct_url`, which uses
/// `url.copy_with(path=…)` to set the path explicitly **before** applying
/// query params — naive `format!("{base}/realtime")` produces a malformed URL
/// when `base` already carries a `?` (the suffix ends up inside the query
/// string and the upstream rejects the upgrade).
///
/// Falls back to string concat if parsing fails (very unusual `api_base`) so
/// the upgrade is never blocked.
pub(super) fn build_realtime_ws_url(base: &str, path_suffix: &str, model: &str) -> String {
    match url::Url::parse(base) {
        Ok(mut parsed) => {
            // Preserve any existing path (trimming a trailing slash so
            // `/v1/` + `/realtime` → `/v1/realtime`, not `/v1//realtime`):
            // `/v1` → `/v1/realtime`, `/`/empty → `/realtime`.
            let trimmed = parsed.path().trim_end_matches('/');
            let new_path = if trimmed.is_empty() {
                path_suffix.to_string()
            } else {
                format!("{trimmed}{path_suffix}")
            };
            parsed.set_path(&new_path);
            // api_base may carry a template selector. Replace it, rather than
            // leaving a leading `model=*` / `model=provider/id` for an upstream
            // that resolves the first duplicate query value.
            let query: Vec<_> = parsed
                .query_pairs()
                .filter(|(name, _)| name != "model")
                .map(|(name, value)| (name.into_owned(), value.into_owned()))
                .collect();
            parsed.set_query(None);
            parsed
                .query_pairs_mut()
                .extend_pairs(query)
                .append_pair("model", model);
            parsed.to_string()
        }
        Err(e) => {
            tracing::warn!(
                base = %base,
                error = %e,
                "failed to parse realtime upstream url; falling back to string concat"
            );
            format!("{base}{path_suffix}?model={model}")
        }
    }
}

/// Whether a `host[:port]/path` string refers to the loopback address.
///
/// Bracketed IPv6 literals (`[::1]:8080`) are handled specially: the colon
/// inside the address must not be treated as the port separator, so the
/// literal runs to the closing `]` rather than the first `:`.
fn is_localhost_host(host: &str) -> bool {
    let host_part = if let Some(rest) = host.strip_prefix('[') {
        match rest.find(']') {
            Some(end) => &host[..end + 2],
            None => host.split('/').next().unwrap_or(host),
        }
    } else {
        host.split([':', '/']).next().unwrap_or(host)
    };
    host_part == "localhost"
        || host_part == "127.0.0.1"
        || host_part == "[::1]"
        || host_part == "::1"
}

/// Redact the host in an api_base for logging, keeping only the scheme and
/// a placeholder so the warning identifies the misconfiguration without
/// leaking the internal hostname.
fn redact_api_base_host(base: &str) -> String {
    if let Some(rest) = base.strip_prefix("http://") {
        format!(
            "http://<host>{}",
            rest.find('/').map(|i| &rest[i..]).unwrap_or("")
        )
    } else if let Some(rest) = base.strip_prefix("ws://") {
        format!(
            "ws://<host>{}",
            rest.find('/').map(|i| &rest[i..]).unwrap_or("")
        )
    } else {
        base.to_string()
    }
}

/// Build a `Close(1011)` message for an upstream-connect failure / timeout.
///
/// RFC 6455 §7.4 reserves 1011 (Internal Error) for "a generic error status
/// code… the endpoint will no longer be available". The proxy's normal session
/// teardown uses `Close(None)` (→ code 1000, Normal Closure), which is correct
/// when one side closed cleanly — but an upstream-connect failure / timeout is
/// *not* a normal closure. Surfacing 1011 (with a reason) lets the client
/// distinguish "session ended normally" from "upstream failed, retry / surface
/// an error" instead of being misled by a 1000.
pub(super) fn upstream_failure_close(reason: &'static str) -> AxumMessage {
    AxumMessage::Close(Some(axum::extract::ws::CloseFrame {
        code: 1011,
        reason: reason.into(),
    }))
}

/// Query parameters accepted by `/v1/realtime`.
///
/// `model` is required (selects the deployment). Auth is handled entirely by
/// the shared auth middleware (which runs before the WS upgrade and accepts
/// `?authorization=` as a fallback only on `/v1/realtime*`), so this handler
/// never parses auth itself.
#[derive(Debug, Deserialize)]
pub struct RealtimeQuery {
    /// The model name to route to (e.g. `gpt-4o-realtime`).
    pub model: Option<String>,
}

/// Collect the upstream auth headers for a realtime deployment, failing
/// closed when the provider produced none.
///
/// The [`RealtimeProvider::ws_auth_headers`] contract says a deployment
/// whose `api_key` is empty/resolver-only "must fail closed rather than
/// connect unauthenticated" — every shipped provider returns an empty vec
/// exactly when `api_key` is unset, and attaching nothing would silently
/// connect unauthenticated (a late upstream 401 after the client WS
/// upgrade at best; unauthenticated sessions against permissive
/// OpenAI-compatible `api_base` endpoints at worst). This mirrors the chat
/// tier's `StaticKeyResolver` fail-closed behavior
/// (`Error::Config("missing api_key for deployment")`), so the rejection is
/// early, typed, and the pre-call reservation can be released.
fn realtime_auth_headers(
    provider: &dyn RealtimeProvider,
    deployment: &oxllm_core::Deployment,
) -> Result<Vec<(String, String)>, ApiError> {
    let headers = provider.ws_auth_headers(deployment);
    if headers.is_empty() {
        tracing::warn!(
            model = %deployment.oxllm_params.model,
            "realtime deployment has no api_key; refusing to connect unauthenticated (fail closed)"
        );
        return Err(oxllm_core::Error::Config(
            "missing api_key for realtime deployment; refusing to connect unauthenticated"
                .to_string(),
        )
        .into());
    }
    Ok(headers)
}

/// `GET /v1/realtime` — accept the client WebSocket upgrade, open an
/// upstream WebSocket to the resolved provider, and bidirectionally proxy
/// frames through the provider's transform pipeline.
pub async fn handle_realtime_upgrade(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Query(query): Query<RealtimeQuery>,
    ws: WebSocketUpgrade,
    headers: HeaderMap,
) -> Result<Response, ApiError> {
    let state = shared.load_full();

    // Cap the client WS frame/message size well below tungstenite's 64 MiB
    // default so a single oversized client frame cannot pin ~64 MiB of worker
    // memory per inflight realtime session. 1 MiB is generous for any sane
    // realtime JSON event or audio chunk.
    let ws = ws
        .max_message_size(REALTIME_MAX_MESSAGE_SIZE)
        .max_frame_size(REALTIME_MAX_MESSAGE_SIZE);

    let model = query
        .model
        .as_deref()
        .filter(|model| !model.trim().is_empty())
        .ok_or_else(|| ApiError::bad_request("missing required query parameter: 'model'"))?;

    let session_id = crate::pipeline::extract_session_id(&headers);
    let tags = crate::pipeline::extract_tags(&headers, None);
    // Pre-upgrade metering: rate-limit/budget pre-call + deployment RPM/TPM.
    // Rejects before we open the upstream WS or upgrade the client.
    let request_body = serde_json::json!({ "model": model, "stream": true });
    let mut pipeline = crate::pipeline::EndpointPipeline::for_request_with_headers(
        &state,
        &auth,
        model,
        request_body,
        &headers,
    )
    .await?;
    let selected = match pick_realtime_deployment(&state, model, session_id, &tags) {
        Ok(selected) => selected,
        Err(error) => {
            crate::pipeline::record_request_rejection(&state, &pipeline.call_ctx, &error).await;
            return Err(error.for_inference());
        }
    };
    // Reserve at pick, before any other await can let least-busy select this
    // deployment as idle. The original identity remains stable for wildcards.
    let routing = RealtimeRoutingGuard::new(state.clone(), &selected, model);
    let deployment = match routing::concrete_deployment(&selected, model) {
        Ok(deployment) => deployment,
        Err(error) => {
            crate::pipeline::record_request_rejection(&state, &pipeline.call_ctx, &error).await;
            return Err(error);
        }
    };
    if let Err(error) = crate::pipeline::prepare_deployment_pricing_with_duration_estimate(
        &state,
        &mut pipeline.call_ctx,
        &deployment,
        &serde_json::json!({"model":model}),
        REALTIME_DURATION_ESTIMATE,
    )
    .await
    {
        let error = ApiError::from(error);
        crate::pipeline::record_request_rejection(&state, &pipeline.call_ctx, &error).await;
        return Err(error);
    }
    pipeline.apply_request_tags(&tags).await?;
    let admission_state = state.clone();
    let admission_deployment = deployment.clone();
    let call_ctx = pipeline.call_ctx;
    // Admission itself may contain several cache debits. Drive it independently
    // so cancellation cannot strand a partially reserved key/router scope. If
    // the handler disappears, the returned guard is dropped by the detached
    // task and cleans up the completed admission exactly once.
    let admission = tokio::spawn(async move {
        let mut pipeline = crate::pipeline::EndpointPipeline {
            state: &admission_state,
            call_ctx,
        };
        if pipeline.pre_call().await?.is_some() {
            // Realtime proceeds through cache short-circuits; it must still be
            // metered as an upstream session rather than a zero-rated hit.
            pipeline.call_ctx.cache_hit = false;
        }
        let dep_key = routing.dep_key.clone();
        let store_key = routing.cooldown_key.clone();
        let mut guard = UpgradeAbortGuard::new(
            admission_state.clone(),
            pipeline.call_ctx.clone(),
            admission_deployment.clone(),
            routing,
        );
        if let Err(error) = pipeline.check_limits(&admission_deployment, &dep_key).await {
            await_settlement(guard.settle(Err(crate::pipeline::core_error_from_api(&error)))).await;
            return Err(error);
        }
        crate::orchestration::chat::record_deployment_usage(&admission_state, &dep_key, &store_key)
            .await;
        let reservation = match crate::orchestration::budget::reserve_router_budgets(
            &admission_state,
            &admission_deployment,
            &dep_key,
            &pipeline.call_ctx,
        )
        .await
        {
            Ok(reservation) => reservation,
            Err(error) => {
                await_settlement(
                    guard.settle(Err(crate::pipeline::core_error_from_api(&error.api_error))),
                )
                .await;
                return Err(error.api_error);
            }
        };
        guard.set_router_budget(RealtimeRouterBudgetGuard::new(
            admission_state.cache.clone(),
            reservation,
        ));
        Ok(guard)
    });
    let mut guard = admission
        .await
        .map_err(|_| ApiError::internal("realtime admission task failed"))??;
    let allow_private = state
        .config
        .general_settings
        .allow_private_api_base
        .unwrap_or(false);

    // Bedrock's native API is a signed HTTP2 POST with duplex eventstream I/O.
    // Explicit WS(S) bridge endpoints are also adapter-owned. Neither fits the
    // generic upstream WebSocket text proxy, so delegate the full lifecycle.
    if matches!(deployment.provider_slug(), "bedrock" | "bedrock_mantle") {
        let adapter = match bedrock::BedrockRealtimeAdapter::from_deployment(&deployment) {
            Ok(a) => a,
            Err(e) => {
                await_settlement(guard.settle(Err(oxllm_core::Error::Config(e.message.clone()))))
                    .await;
                return Err(e);
            }
        };
        return Ok(ws.on_upgrade(move |socket| async move {
            let started = guard.start_connection();
            let meter = guard.meter();
            let outcome = adapter
                .connect_and_proxy(socket, allow_private, meter, |result| {
                    guard.connection_result(result, started);
                })
                .await;
            await_settlement(guard.settle(outcome)).await;
        }));
    }

    let provider = match resolve_realtime_provider(&deployment) {
        Ok(p) => p,
        Err(e) => {
            await_settlement(guard.settle(Err(oxllm_core::Error::BadRequest(e.message.clone()))))
                .await;
            return Err(e);
        }
    };
    let upstream_url = provider.ws_url(&deployment);
    let mut request = match upstream_url.as_str().into_client_request() {
        Ok(r) => r,
        Err(e) => {
            let err = ApiError::internal(format!("invalid upstream url: {e}"));
            await_settlement(guard.settle(Err(oxllm_core::Error::Config(
                "invalid realtime upstream URL".into(),
            ))))
            .await;
            return Err(err);
        }
    };
    let auth_headers = match realtime_auth_headers(provider.as_ref(), &deployment) {
        Ok(h) => h,
        Err(e) => {
            await_settlement(guard.settle(Err(oxllm_core::Error::Config(e.message.clone())))).await;
            return Err(e);
        }
    };
    for (name, value) in auth_headers {
        match (
            HeaderName::from_bytes(name.as_bytes()),
            HeaderValue::from_str(&value),
        ) {
            (Ok(n), Ok(v)) => {
                request.headers_mut().insert(n, v);
            }
            _ => {
                tracing::warn!(
                    header_name = %name,
                    "realtime upstream auth header is malformed; refusing to connect"
                );
                let err =
                    ApiError::internal("invalid auth header for upstream realtime connection");
                await_settlement(guard.settle(Err(oxllm_core::Error::Config(
                    "invalid auth header for upstream realtime connection".into(),
                ))))
                .await;
                return Err(err);
            }
        }
    }

    tracing::info!(
        url = %crate::security::sanitize_url(&upstream_url),
        model = %model,
        "realtime upgrade; opening upstream"
    );

    // The upstream connection must happen INSIDE the on_upgrade callback,
    // not before it. If we connect before returning on_upgrade, axum holds
    // the HTTP 101 response until our await completes — the client times out.
    Ok(ws.on_upgrade(move |mut socket| async move {
        let started = guard.start_connection();
        let upstream = match connect::connect(request, allow_private).await {
            Ok(upstream) => {
                guard.connection_result(Ok(()), started);
                upstream
            }
            Err(e) => {
                tracing::warn!(error = %e, "upstream WS connect failed");
                guard.connection_result(Err(&e), started);
                let settlement = guard.settle(Err(e.error));
                send_client(
                    &mut socket,
                    upstream_failure_close("upstream connect failed"),
                )
                .await;
                await_settlement(settlement).await;
                return;
            }
        };
        let outcome = proxy(socket, upstream, provider, guard.meter()).await;
        await_settlement(guard.settle(Ok(outcome))).await;
    }))
}

/// Settle a realtime session's metering after the pump returns.
///
/// Setup failures always run failure hooks. Successfully connected sessions
/// with recovered usage settle the reported amount. If output remains
/// unmetered after the bounded drain, token-priced sessions keep the existing
/// reservation as a conservative estimate, matching chat's partial policy.
/// A duration-priced session's elapsed cost is complete independently of token
/// usage, so settle that snapshot without turning a failure into a success.
async fn settle_realtime_session(
    state: &crate::state::AppState,
    call_ctx: &oxllm_core::CallContext,
    model: &str,
    provider_slug: &str,
    dep_key: &str,
    outcome: oxllm_core::Result<SessionOutcome>,
) {
    let outcome = match outcome {
        Ok(outcome) => outcome,
        Err(error) => {
            crate::pipeline::post_failure(state, call_ctx, &error).await;
            return;
        }
    };
    tracing::info!(
        request_id = %call_ctx.request_id,
        usage_reported = outcome.usage.is_some(),
        unmetered_output = outcome.unmetered_output,
        "realtime session ended; settling metering"
    );
    if let Some(usage) = &outcome.usage {
        crate::pipeline::record_deployment_tpm(state, dep_key, usage.total_tokens).await;
    }
    let failure = match &outcome.end {
        SessionEnd::UpstreamError(error) => Some(error.clone()),
        SessionEnd::TaskAborted => Some(oxllm_core::Error::Network(
            "realtime session task aborted by infrastructure".into(),
        )),
        _ => None,
    };
    let mut partial_ctx = call_ctx.clone();
    let duration_cost = call_ctx.cost_override.filter(|cost| {
        cost.is_finite()
            && *cost >= 0.0
            && realtime_duration_pricing(call_ctx, model, provider_slug).is_some()
    });
    if (failure.is_some() || outcome.unmetered_output)
        && let Some(cost) = duration_cost
    {
        crate::pipeline::settle_partial_budget(state, &mut partial_ctx, cost).await;
    }
    let call_ctx = &partial_ctx;
    if let Some(error) = failure {
        // Token-priced faults retain their existing refund policy. Known
        // duration charges were already settled, so failure hooks cannot undo
        // measured work; even a connected control-only fault has elapsed cost.
        if outcome.usage.is_some() || outcome.unmetered_output || duration_cost.is_some() {
            record_partial_realtime(
                state,
                call_ctx,
                model,
                provider_slug,
                outcome.usage.as_ref(),
            )
            .await;
        }
        // Preserve the partial record first; the cost observer skips its
        // claimed terminal row, while all failure callbacks still run.
        crate::pipeline::post_failure(state, call_ctx, &error).await;
    } else if outcome.unmetered_output {
        crate::pipeline::release_concurrency(state, call_ctx).await;
        record_partial_realtime(
            state,
            call_ctx,
            model,
            provider_slug,
            outcome.usage.as_ref(),
        )
        .await;
        if duration_cost.is_some() {
            tracing::info!(request_id = %call_ctx.request_id, cost = duration_cost,
                "realtime output lacks final tokens; settled known duration cost");
        } else {
            tracing::warn!(
                request_id = %call_ctx.request_id,
                reservation_retained = ?call_ctx.budget_estimate,
                "realtime output lacks final usage; retaining existing estimate, actual cost unknown"
            );
        }
    } else {
        crate::pipeline::post_success(
            state,
            call_ctx,
            model,
            outcome.usage.as_ref(),
            None,
            None,
            provider_slug,
        )
        .await;
    }
}

/// Explicit token rates take precedence over duration, including on catalog
/// voice models. Only duration pricing makes the *whole* partial charge known;
/// reported tokens from one response must not settle another unmetered one.
fn realtime_duration_pricing(
    ctx: &oxllm_core::CallContext,
    model: &str,
    provider: &str,
) -> Option<oxllm_core::DeploymentPricing> {
    if let Some(pricing) = &ctx.deployment_pricing {
        return pricing.is_per_second_priced().then(|| pricing.clone());
    }
    [
        Some(model),
        ctx.pricing_model.as_deref(),
        ctx.deployment_model.as_deref(),
        Some(ctx.model.as_str()),
    ]
    .into_iter()
    .flatten()
    .find_map(|model| oxllm_core::lookup_pricing_with_provider(model, provider))
    .filter(|pricing| pricing.is_per_second_priced())
    .map(|pricing| oxllm_core::DeploymentPricing {
        cost_per_second_rate: pricing.cost_per_second_rate,
        input_cost_per_second: pricing.input_cost_per_second,
        output_cost_per_second: pricing.output_cost_per_second,
        ..Default::default()
    })
}

async fn record_partial_realtime(
    state: &crate::state::AppState,
    call_ctx: &oxllm_core::CallContext,
    model: &str,
    provider: &str,
    usage: Option<&UsageSummary>,
) {
    if !call_ctx.claim_spend_record() {
        return;
    }
    let cost =
        crate::hooks::cost_fallback::resolve_cost(call_ctx, model, usage, None, None, provider);
    tracing::info!(event = "spend", status = "partial", request_id = %call_ctx.request_id,
        session_id = call_ctx.session_id.as_deref(),
        model, provider, cost, cost_known = cost.is_some(), key_id = call_ctx.key_id.as_deref(),
        prompt_tokens = usage.map(|usage| usage.prompt_tokens),
        completion_tokens = usage.map(|usage| usage.completion_tokens),
        "partial realtime spend recorded");
    let Some(tx) = &state.spend_tx else {
        return;
    };
    let tokens = |n| i64::try_from(n).unwrap_or(i64::MAX);
    let entry = crate::spend_writer::SpendEntry {
        timestamp: jiff::Timestamp::now().to_string(),
        request_id: Some(call_ctx.request_id.clone()),
        session_id: call_ctx.session_id.clone(),
        model: model.to_string(),
        provider: provider.to_string(),
        deployment_id: call_ctx.deployment_id.clone(),
        requested_model: Some(call_ctx.model.clone()),
        model_group: call_ctx.serving_model_group.clone(),
        prompt_tokens: usage.map(|u| tokens(u.prompt_tokens)),
        completion_tokens: usage.map(|u| tokens(u.completion_tokens)),
        total_tokens: usage.map(|u| tokens(u.total_tokens)),
        cost,
        key_id: call_ctx.key_id.clone(),
        team_id: call_ctx.team_id.clone(),
        user_id: call_ctx.user_id.clone(),
        end_user_id: call_ctx.end_user_id.clone(),
        request_tag: call_ctx.request_tag.clone(),
        request_tags: crate::spend_writer::tags_json(&call_ctx.request_tags),
        status: Some("partial".to_string()),
        latency_ms: call_ctx
            .start_time
            .map(|s| i64::try_from(s.elapsed().as_millis()).unwrap_or(i64::MAX)),
    };
    crate::spend_writer::queue_partial_spend(tx, entry).await;
}

type UpstreamStream =
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>;

/// Bidirectional realtime proxy with a provider transform pipeline.
///
/// Both directions are multiplexed with `select!`; every write is bounded.
/// After disconnect, stop forwarding output, cancel generation and recover
/// final usage before closing the upstream (within USAGE_DRAIN_TIMEOUT).
async fn proxy(
    mut client: WebSocket,
    mut upstream: UpstreamStream,
    provider: Box<dyn RealtimeProvider>,
    mut meter: SessionMeter,
) -> SessionOutcome {
    let deadline = tokio::time::Instant::now() + REALTIME_MAX_SESSION_DURATION;
    let end = 'pump: loop {
        tokio::select! {
            msg = client.next() => match msg {
                Some(Ok(AxumMessage::Text(text))) => {
                    match provider.transform_client_event(text.as_str()) {
                        Ok(events) => for event in events {
                            if !send_upstream(&mut upstream, TungsteniteMessage::Text(event.into())).await {
                                break 'pump upstream_send_error();
                            }
                        },
                        Err(error) => {
                            if !send_client(&mut client, client_event_error(text.as_str(), &error)).await {
                                break SessionEnd::ClientDisconnected;
                            }
                        }
                    }
                }
                Some(Ok(AxumMessage::Binary(bytes))) => {
                    if !send_upstream(&mut upstream, TungsteniteMessage::Binary(bytes)).await {
                        break upstream_send_error();
                    }
                }
                Some(Ok(AxumMessage::Ping(_) | AxumMessage::Pong(_))) => {}
                _ => break SessionEnd::ClientDisconnected,
            },
            msg = upstream.next() => match msg {
                Some(Ok(TungsteniteMessage::Text(text))) => {
                    meter.observe(provider.as_ref(), text.as_str());
                    let events = match provider.transform_upstream_event(text.as_str()) {
                        Ok(events) => events,
                        Err(error) => break SessionEnd::UpstreamError(error),
                    };
                    for event in events {
                        if !send_client(&mut client, AxumMessage::Text(event.clone().into())).await {
                            break 'pump SessionEnd::ClientDisconnected;
                        }
                        meter.delivered_text(&event);
                    }
                }
                Some(Ok(TungsteniteMessage::Binary(bytes))) => {
                    let has_output = !bytes.is_empty();
                    if !send_client(&mut client, AxumMessage::Binary(bytes)).await {
                        break SessionEnd::ClientDisconnected;
                    }
                    if has_output {
                        meter.delivered_binary();
                    }
                }
                Some(Ok(TungsteniteMessage::Close(close))) => break upstream_close_end(close.as_ref()),
                None => break SessionEnd::UpstreamClosed,
                Some(Err(_)) => break SessionEnd::UpstreamError(oxllm_core::Error::Network("realtime upstream read failed".into())),
                Some(Ok(_)) => {}
            },
            _ = tokio::time::sleep_until(deadline) => break SessionEnd::TimedOut,
        }
    };
    meter.ending(&end);
    if end.should_drain() {
        let _ = tokio::time::timeout(USAGE_DRAIN_TIMEOUT, async {
            'cancel: for cancel in meter.cancel_events() {
                if let Ok(events) = provider.transform_client_event(&cancel) {
                    for event in events {
                        if upstream
                            .send(TungsteniteMessage::Text(event.into()))
                            .await
                            .is_err()
                        {
                            // A final usage event may already be readable even
                            // if the peer stopped accepting client events.
                            break 'cancel;
                        }
                    }
                }
            }
            while let Some(Ok(message)) = upstream.next().await {
                match message {
                    TungsteniteMessage::Text(text) => {
                        if meter.observe(provider.as_ref(), text.as_str()) {
                            break;
                        }
                    }
                    TungsteniteMessage::Close(_) => break,
                    _ => {}
                }
            }
        })
        .await;
    }
    close_sockets(&mut client, &mut upstream, &end).await;
    meter.finish(end)
}

fn upstream_send_error() -> SessionEnd {
    SessionEnd::UpstreamError(oxllm_core::Error::Network(
        "realtime upstream write failed or stalled".into(),
    ))
}

pub(super) async fn send_client(client: &mut WebSocket, message: AxumMessage) -> bool {
    matches!(
        tokio::time::timeout(WS_SEND_TIMEOUT, client.send(message)).await,
        Ok(Ok(()))
    )
}

pub(super) async fn send_upstream(
    upstream: &mut UpstreamStream,
    message: TungsteniteMessage,
) -> bool {
    matches!(
        tokio::time::timeout(WS_SEND_TIMEOUT, upstream.send(message)).await,
        Ok(Ok(()))
    )
}

async fn close_sockets(client: &mut WebSocket, upstream: &mut UpstreamStream, end: &SessionEnd) {
    let _ = tokio::time::timeout(USAGE_DRAIN_TIMEOUT, async {
        let close = if matches!(end, SessionEnd::UpstreamError(_)) {
            upstream_failure_close("upstream realtime session failed")
        } else {
            AxumMessage::Close(None)
        };
        let _ = client.send(close).await;
        let _ = upstream.close(None).await;
    })
    .await;
}

fn upstream_close_end(
    close: Option<&tokio_tungstenite::tungstenite::protocol::CloseFrame>,
) -> SessionEnd {
    if let Some(close) = close {
        let code = u16::from(close.code);
        if !matches!(code, 1000 | 1001) {
            return SessionEnd::UpstreamError(oxllm_core::Error::Network(format!(
                "realtime upstream closed with error code {code}"
            )));
        }
    }
    SessionEnd::UpstreamClosed
}

pub(super) fn client_event_error(event: &str, error: &oxllm_core::Error) -> AxumMessage {
    let (code, message) = match error {
        oxllm_core::Error::BadRequest(message) => ("unsupported_event", message.as_str()),
        _ => ("invalid_event", "invalid realtime client event"),
    };
    let event_id = serde_json::from_str::<serde_json::Value>(event)
        .ok()
        .and_then(|v| v.get("event_id").cloned());
    AxumMessage::Text(serde_json::json!({
        "type":"error",
        "error":{"type":"invalid_request_error","code":code,"message":message,"event_id":event_id}
    }).to_string().into())
}

/// Resolve a deployment to its realtime provider.
///
/// OpenAI-compatible providers (openai, xai, azure, …) resolve to their
/// respective identity/normalizing provider. An `azure` deployment without
/// `api_base` is rejected here (fail closed) rather than falling back to an
/// OpenAI-bound URL with Azure credentials. Bedrock / bedrock_mantle are
/// **not** handled here: the live connect path short-circuits to
/// [`bedrock::BedrockRealtimeAdapter`] in [`handle_realtime_upgrade`] before
/// this resolver is called, because Bedrock realtime requires query-string
/// SigV4 signing and binary eventstream framing that the generic
/// text-passthrough `proxy` cannot express. Keeping Bedrock out of this
/// resolver avoids a dead arm that unit tests exercise but production never
/// reaches (divergence risk).
fn resolve_realtime_provider(
    deployment: &oxllm_core::Deployment,
) -> Result<Box<dyn RealtimeProvider>, ApiError> {
    let slug = deployment.provider_slug();
    match slug {
        "xai" => Ok(Box::new(XaiRealtimeProvider::new())),
        "azure" => {
            // Azure realtime has no meaningful default base: without
            // `api_base` the URL would fall back to OpenAI's endpoint and
            // send the Azure `api-key` header to the wrong vendor. Fail
            // closed with a config error naming the missing field instead
            // (mirroring the chat tier's stance on incomplete Azure config).
            if deployment.oxllm_params.api_base.is_none() {
                tracing::warn!(
                    model = %deployment.oxllm_params.model,
                    "azure realtime deployment sets no api_base; refusing to \
                     default to the OpenAI endpoint with Azure credentials"
                );
                return Err(ApiError::bad_request(
                    "azure realtime deployment requires api_base (the resource \
                     endpoint, e.g. https://<resource>.openai.azure.com; the \
                     documented v1 base …/openai/v1/ is also accepted)",
                ));
            }
            Ok(Box::new(AzureRealtimeProvider::new()))
        }
        // OpenAI-compatible realtime. The bare-model default slug is also
        // "openai" (see `Deployment::provider_slug`).
        "openai" => Ok(Box::new(OpenAIRealtimeProvider)),
        other => {
            // Refuse to silently route an unsupported slug to OpenAI: an
            // unknown provider prefix (typo, leading-slash model id, or a
            // provider without a realtime impl) must surface as a clean error
            // rather than attempting an OpenAI-shaped WS connection with
            // mismatched credentials. Bedrock slugs are handled earlier in
            // `handle_realtime_upgrade` and never reach here.
            tracing::warn!(
                provider = %other,
                model = %deployment.oxllm_params.model,
                "no realtime provider registered for slug; refusing to silently route to OpenAI"
            );
            Err(ApiError::bad_request(format!(
                "provider '{other}' does not support the realtime endpoint"
            )))
        }
    }
}

/// Pick a deployment for a realtime model using the shared router.
fn pick_realtime_deployment(
    state: &crate::state::AppState,
    model: &str,
    session_id: Option<String>,
    tags: &[String],
) -> Result<oxllm_core::Deployment, ApiError> {
    let mut ctx = oxllm_core::RequestContext::new(model);
    ctx.session_id = session_id;
    ctx.tag = tags.first().cloned();
    ctx.tags = tags.to_vec();
    crate::pipeline::routing::apply_default_tag(state, &mut ctx);
    let idx = crate::pipeline::routing::try_pick_deployment_index(state, &ctx)?
        .ok_or_else(|| crate::pipeline::routing::no_available_deployment(state, model))?;
    Ok(state.deployments[idx].clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxllm_core::deployment::OxllmParams;

    fn dep(slug_model: &str) -> oxllm_core::Deployment {
        let mut d = oxllm_core::Deployment {
            model_name: slug_model.to_string(),
            oxllm_params: OxllmParams {
                model: slug_model.to_string(),
                api_key: Some("sk-test".to_string()),
                ..Default::default()
            },
            ..Default::default()
        };
        if d.provider_slug() == "azure" {
            d.oxllm_params.api_base = Some("https://my.resource.openai.azure.com".to_string());
        }
        d
    }

    // ---- resolve_realtime_provider -----------------------------------------

    #[test]
    fn resolve_openai_provider_for_openai_deployment() {
        let p = resolve_realtime_provider(&dep("openai/gpt-4o-realtime")).unwrap();
        // Identity transform: one event out per event in.
        assert_eq!(
            p.transform_client_event("x")
                .expect("identity never errors"),
            vec!["x".to_string()]
        );
    }

    #[test]
    fn resolve_openai_provider_for_bare_model() {
        // A model with no provider prefix defaults to "openai".
        let p = resolve_realtime_provider(&dep("gpt-4o-realtime")).unwrap();
        assert_eq!(
            p.transform_upstream_event("y")
                .expect("identity never errors"),
            vec!["y".to_string()]
        );
    }

    #[test]
    fn resolve_xai_provider_uses_normalizer() {
        let p = resolve_realtime_provider(&dep("xai/grok-realtime")).unwrap();
        // The xAI provider normalizes wire quirks: it drops ping events,
        // whereas the identity provider passes them through. Use this to
        // confirm we got the normalizing provider, not the identity one.
        assert!(
            p.transform_upstream_event(r#"{"type":"ping"}"#)
                .expect("xai never errors")
                .is_empty(),
            "xai must drop ping events"
        );
        // Bearer auth (delegated to the identity provider).
        assert_eq!(p.ws_auth_headers(&dep("xai/grok-realtime")).len(), 1);
    }

    #[test]
    fn resolve_azure_provider_uses_api_key_header() {
        let p = resolve_realtime_provider(&dep("azure/gpt-4o-realtime")).unwrap();
        // Azure authenticates with `api-key:`, not `Authorization: Bearer`.
        let headers = p.ws_auth_headers(&dep("azure/gpt-4o-realtime"));
        assert_eq!(headers.len(), 1);
        assert_eq!(headers[0].0, "api-key");
        // Azure is otherwise wire-compatible (identity transforms).
        let event = r#"{"type":"response.audio.delta","delta":"x"}"#;
        assert_eq!(
            p.transform_upstream_event(event)
                .expect("identity never errors"),
            vec![event.to_string()]
        );
    }

    #[test]
    fn resolve_bedrock_slug_not_handled_by_resolver() {
        // MEDIUM #7: the Bedrock arm was removed from resolve_realtime_provider
        // because the live path short-circuits to BedrockRealtimeAdapter before
        // the resolver is called. The resolver must surface a clean error for
        // bedrock slugs rather than silently succeeding — proving the dead arm
        // is gone and keeping the resolver as the single source of truth for
        // non-Bedrock slugs.
        let res = resolve_realtime_provider(&dep("bedrock/amazon.nova-sonic-v1:0"));
        assert!(
            res.is_err(),
            "bedrock must not resolve through resolve_realtime_provider"
        );
    }

    #[test]
    fn resolve_unknown_provider_slug_returns_error_not_openai() {
        // An admin-configured deployment whose slug has no realtime provider
        // must surface a typed error rather than silently routing to OpenAI
        // (which would attempt an OpenAI-shaped WS connection with mismatched
        // credentials).
        let res = resolve_realtime_provider(&dep("acme/rt"));
        let err = match res {
            Err(e) => e,
            Ok(_) => panic!("unknown slug must error, not resolve to a provider"),
        };
        assert_eq!(err.status, axum::http::StatusCode::BAD_REQUEST);
        assert!(
            err.message.contains("acme"),
            "error should name the unsupported slug, got: {}",
            err.message
        );
    }

    /// LOW (C10): an azure realtime deployment without `api_base` must be
    /// rejected with a typed config error naming the missing field, instead
    /// of building an OpenAI-bound URL that carries the Azure `api-key`
    /// header to the wrong vendor (fail-late, confusing upstream 401).
    #[test]
    fn resolve_azure_without_api_base_fails_closed() {
        let deployment = dep_with_key("azure/gpt-4o-realtime", Some("az-test"));
        let mut no_base = deployment.clone();
        no_base.oxllm_params.api_base = None;
        let err = match resolve_realtime_provider(&no_base) {
            Err(e) => e,
            Ok(_) => panic!("azure without api_base must fail closed"),
        };
        assert_eq!(err.status, axum::http::StatusCode::BAD_REQUEST);
        assert!(
            err.message.contains("api_base"),
            "error should name the missing api_base, got: {}",
            err.message
        );
        assert!(
            !err.message.contains("az-test"),
            "error must not echo the api key"
        );
        // With api_base set the same deployment resolves.
        assert!(resolve_realtime_provider(&deployment).is_ok());
    }

    // ---- is_localhost_host / scheme_rewrite_for_ws -------------------------

    /// LOW (C10): a bracketed IPv6 loopback api_base (`http://[::1]:8080`)
    /// must be classified local — the colons inside the literal are not port
    /// separators — so no spurious cleartext-credential warning fires for a
    /// loopback debug deployment.
    #[test]
    fn localhost_detection_handles_bracketed_ipv6() {
        for host in ["[::1]:8080", "[::1]", "[::1]:9000/v1"] {
            assert!(
                is_localhost_host(host),
                "{host} is a loopback host and must be classified local"
            );
        }
        // Non-loopback IPv6 stays non-local.
        assert!(!is_localhost_host("[fe80::1]:9000"));
        // The plain forms keep working.
        assert!(is_localhost_host("localhost"));
        assert!(is_localhost_host("localhost:8080"));
        assert!(is_localhost_host("127.0.0.1:9000"));
        assert!(!is_localhost_host("api.openai.com"));
    }

    #[test]
    fn scheme_rewrite_accepts_ipv6_loopback_http_base() {
        assert_eq!(
            scheme_rewrite_for_ws("http://[::1]:9000/v1"),
            "ws://[::1]:9000/v1"
        );
    }

    // ---- realtime_auth_headers (fail closed on missing api_key) ----------

    fn dep_with_key(slug_model: &str, api_key: Option<&str>) -> oxllm_core::Deployment {
        let mut d = oxllm_core::Deployment {
            model_name: slug_model.to_string(),
            oxllm_params: OxllmParams {
                model: slug_model.to_string(),
                api_key: api_key.map(String::from),
                ..Default::default()
            },
            ..Default::default()
        };
        if d.provider_slug() == "azure" {
            d.oxllm_params.api_base = Some("https://my.resource.openai.azure.com".to_string());
        }
        d
    }

    /// MEDIUM (C10): the documented `ws_auth_headers` contract — a
    /// deployment whose `api_key` is empty "must fail closed rather than
    /// connect unauthenticated" — must be enforced. A missing key must
    /// surface a typed rejection BEFORE the client WS upgrade / upstream
    /// connect, mirroring the chat tier's `StaticKeyResolver`.
    #[test]
    fn realtime_auth_headers_fail_closed_when_api_key_missing() {
        let deployment = dep_with_key("openai/gpt-4o-realtime", None);
        let provider = resolve_realtime_provider(&deployment).expect("openai resolves");
        let err = match realtime_auth_headers(provider.as_ref(), &deployment) {
            Err(e) => e,
            Ok(headers) => panic!(
                "missing api_key must fail closed, got {} header(s): {headers:?}",
                headers.len()
            ),
        };
        assert_eq!(
            err.status,
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            "mirrors the chat tier's Error::Config mapping"
        );
    }

    #[test]
    fn realtime_auth_headers_pass_through_when_api_key_present() {
        let deployment = dep_with_key("azure/gpt-4o-realtime", Some("az-test"));
        let provider = resolve_realtime_provider(&deployment).expect("azure resolves");
        let headers =
            realtime_auth_headers(provider.as_ref(), &deployment).expect("headers present");
        assert_eq!(headers.len(), 1);
        assert_eq!(headers[0].0, "api-key");
        assert_eq!(headers[0].1, "az-test");
    }

    #[test]
    fn realtime_auth_headers_fail_closed_for_every_openai_compatible_slug() {
        for model in [
            "gpt-4o-realtime",
            "xai/grok-realtime",
            "azure/gpt-4o-realtime",
        ] {
            let deployment = dep_with_key(model, None);
            let provider = resolve_realtime_provider(&deployment)
                .unwrap_or_else(|e| panic!("{model} must resolve: {e:?}"));
            assert!(
                realtime_auth_headers(provider.as_ref(), &deployment).is_err(),
                "{model}: no api_key must fail closed for every provider whose \
                 ws_auth_headers derive from the deployment key"
            );
        }
    }

    // ---- session usage accumulation --------------------------------------

    /// A realtime provider that surfaces every `UsageSummary` field on
    /// `response.done`. Used to prove the live session meter folds the full set
    /// (in particular the cache-creation buckets the hand-rolled accumulator
    /// previously dropped) across multiple events.
    struct FullUsageProvider;
    impl RealtimeProvider for FullUsageProvider {
        fn transform_client_event(&self, e: &str) -> oxllm_core::error::Result<Vec<String>> {
            Ok(vec![e.to_string()])
        }
        fn transform_upstream_event(&self, e: &str) -> oxllm_core::error::Result<Vec<String>> {
            Ok(vec![e.to_string()])
        }
        fn extract_usage(&self, upstream_event: &str) -> Option<UsageSummary> {
            if !upstream_event.contains("response.done") {
                return None;
            }
            let v: serde_json::Value = serde_json::from_str(upstream_event).ok()?;
            let usage = v.get("usage")?;
            Some(UsageSummary {
                prompt_tokens: usage
                    .get("prompt_tokens")
                    .and_then(|t| t.as_u64())
                    .unwrap_or(0),
                completion_tokens: usage
                    .get("completion_tokens")
                    .and_then(|t| t.as_u64())
                    .unwrap_or(0),
                total_tokens: usage
                    .get("total_tokens")
                    .and_then(|t| t.as_u64())
                    .unwrap_or(0),
                cache_read_input_tokens: usage
                    .get("cache_read_input_tokens")
                    .and_then(|t| t.as_u64()),
                cache_creation_input_tokens: usage
                    .get("cache_creation_input_tokens")
                    .and_then(|t| t.as_u64()),
                cache_creation_1h_input_tokens: usage
                    .get("cache_creation_1h_input_tokens")
                    .and_then(|t| t.as_u64()),
                audio_input_tokens: usage.get("audio_input_tokens").and_then(|t| t.as_u64()),
                reasoning_tokens: usage.get("reasoning_tokens").and_then(|t| t.as_u64()),
                citation_tokens: None,
                web_search_requests: None,
                browser_open_requests: None,
            })
        }
        fn ws_url(&self, _: &oxllm_core::Deployment) -> String {
            "wss://x".to_string()
        }
        fn ws_auth_headers(&self, _: &oxllm_core::Deployment) -> Vec<(String, String)> {
            vec![]
        }
    }

    #[test]
    fn session_meter_folds_all_usage_fields_across_events() {
        let mut meter = SessionMeter::default();
        let event = r#"{"type":"response.done","usage":{
            "prompt_tokens":5,"completion_tokens":7,"total_tokens":12,
            "cache_read_input_tokens":10,"cache_creation_input_tokens":20,
            "cache_creation_1h_input_tokens":5,"audio_input_tokens":3,
            "reasoning_tokens":8}}"#;
        meter.observe(&FullUsageProvider, event);
        meter.observe(&FullUsageProvider, event);

        let acc = meter
            .finish(SessionEnd::UpstreamClosed)
            .usage
            .expect("accumulated usage");
        assert_eq!(acc.prompt_tokens, 10);
        assert_eq!(acc.completion_tokens, 14);
        assert_eq!(acc.total_tokens, 24);
        assert_eq!(acc.cache_read_input_tokens, Some(20));
        // These two fields were silently dropped by the previous hand-rolled
        // accumulator (regression guard).
        assert_eq!(acc.cache_creation_input_tokens, Some(40));
        assert_eq!(acc.cache_creation_1h_input_tokens, Some(10));
        assert_eq!(acc.audio_input_tokens, Some(6));
        assert_eq!(acc.reasoning_tokens, Some(16));
    }

    #[test]
    fn session_meter_skips_events_without_usage() {
        let mut meter = SessionMeter::default();
        // A non-response.done event: extract_usage returns None, accumulator
        // stays empty.
        meter.observe(
            &FullUsageProvider,
            r#"{"type":"response.audio.delta","delta":"x"}"#,
        );
        assert!(meter.finish(SessionEnd::UpstreamClosed).usage.is_none());
    }

    // ---- REALTIME_MAX_SESSION_DURATION (MEDIUM #4) ------------------------

    /// A loopback socket pair for driving the real [`proxy`] loop: returns the
    /// server-side (axum) [`WebSocket`] plus the connected tokio-tungstenite
    /// peer stream (the caller owns the peer so it can play upstream).
    pub(super) async fn session_socket_pair()
    -> (WebSocket, UpstreamStream, tokio::task::JoinHandle<()>) {
        use axum::extract::ws::WebSocketUpgrade;
        let (tx, rx) = tokio::sync::oneshot::channel();
        let tx = std::sync::Arc::new(std::sync::Mutex::new(Some(tx)));
        let app = axum::Router::new().route(
            "/",
            axum::routing::get(move |upgrade: WebSocketUpgrade| {
                let tx = tx.clone();
                async move {
                    upgrade.on_upgrade(move |socket| async move {
                        let sender = tx.lock().unwrap().take().unwrap();
                        let _ = sender.send(socket);
                    })
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("ws://{}/", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let (peer, _) = tokio_tungstenite::connect_async(url).await.unwrap();
        (rx.await.unwrap(), peer, server)
    }

    /// MEDIUM #4: the wall-clock cap is the `sleep_until(deadline)` arm inside
    /// the production [`proxy`] loop, not a pure helper. Drive the real loop
    /// over loopback sockets under paused time so a regression in the deadline
    /// wiring (wrong constant, an arm that never fires, respawned deadline)
    /// fails this test; a test-only threshold helper could not.
    #[tokio::test(start_paused = true)]
    async fn live_proxy_times_out_an_idle_session_at_the_max_duration() {
        let (client, _client_peer, client_server) = session_socket_pair().await;
        // The proxy's `upstream` is the tokio-tungstenite side of a second
        // pair; its axum peer stays alive (and idle) for the session.
        let (_upstream_peer, upstream, upstream_server) = session_socket_pair().await;
        let provider = resolve_realtime_provider(&dep("openai/gpt-4o-realtime")).unwrap();
        let pump = tokio::spawn(proxy(client, upstream, provider, SessionMeter::default()));
        // Let the spawned proxy register its deadline before advancing the
        // clock, otherwise it would compute `now + cap` after the jump.
        tokio::task::yield_now().await;
        tokio::time::advance(REALTIME_MAX_SESSION_DURATION).await;
        let outcome = tokio::time::timeout(Duration::from_secs(30), pump)
            .await
            .expect("proxy must return after the wall-clock cap")
            .expect("proxy task must not panic");
        assert!(
            matches!(outcome.end, SessionEnd::TimedOut),
            "an idle session must end via the production max-duration arm, got {:?}",
            outcome.end
        );
        client_server.abort();
        upstream_server.abort();
    }

    #[test]
    fn upstream_error_close_is_not_a_successful_idle_session() {
        use tokio_tungstenite::tungstenite::protocol::{CloseFrame, frame::coding::CloseCode};
        let close = CloseFrame {
            code: CloseCode::Error,
            reason: "failed".into(),
        };
        assert!(matches!(
            upstream_close_end(Some(&close)),
            SessionEnd::UpstreamError(_)
        ));
        assert!(matches!(
            upstream_close_end(None),
            SessionEnd::UpstreamClosed
        ));
    }

    // ---- settle_realtime_session unmetered path (HIGH #1) ----------------

    /// A connected control-only session has no delivered output to meter and
    /// releases the reservation. The probe test below also checks hook status.
    #[tokio::test]
    async fn settle_unmetered_records_success_not_failure() {
        use crate::config::types::{GeneralSettings, OxllmSettings, ProxyConfig};
        let state = crate::state::AppState::with_registry(
            ProxyConfig {
                model_list: vec![],
                oxllm_settings: OxllmSettings {
                    max_budget: Some(1.0),
                    ..Default::default()
                },
                general_settings: GeneralSettings::default(),
                router_settings: Default::default(),
                extra: serde_json::Map::new(),
            },
            oxllm_providers::default_registry(),
        )
        .expect("state built");

        // Simulate a pre-call reservation: reserve some budget against the
        // test principal's spend key. CallContext::for_test uses api_key_hash
        // "abc", so the spend key is "spend:abc".
        let spend_key = "spend:abc".to_string();
        let _ = state.cache.incr(&spend_key, 0.01, None).await;

        let ctx = oxllm_core::CallContext {
            budget_estimate: Some(0.01),
            ..oxllm_core::CallContext::for_test("gpt-4o-realtime")
        };

        // Settle as unmetered (None usage).
        settle_realtime_session(
            &state,
            &ctx,
            "gpt-4o-realtime",
            "openai",
            "dep-key",
            Ok(SessionOutcome {
                usage: None,
                unmetered_output: false,
                end: SessionEnd::UpstreamClosed,
            }),
        )
        .await;

        // The budget reservation must be released (spend back to ~0).
        let spend = state.cache.counter_value(&spend_key).await.unwrap_or(0.0);
        assert!(
            spend.abs() < 1e-9,
            "unmetered session must release the budget reservation, got spend={spend}"
        );
    }

    // ---- MEDIUM: UpgradeAbortGuard releases reservation on upgrade abort ----

    /// Shared helper: build a test AppState with a budget and pre-reserve
    /// `0.01` against the test principal's spend key.
    async fn state_with_reservation() -> (std::sync::Arc<crate::state::AppState>, String) {
        use crate::config::types::{GeneralSettings, OxllmSettings, ProxyConfig};
        let state = std::sync::Arc::new(
            crate::state::AppState::with_registry(
                ProxyConfig {
                    model_list: vec![],
                    oxllm_settings: OxllmSettings {
                        max_budget: Some(1.0),
                        ..Default::default()
                    },
                    general_settings: GeneralSettings::default(),
                    router_settings: Default::default(),
                    extra: serde_json::Map::new(),
                },
                oxllm_providers::default_registry(),
            )
            .expect("state built"),
        );
        let spend_key = "spend:abc".to_string();
        let _ = state.cache.incr(&spend_key, 0.01, None).await;
        (state, spend_key)
    }

    #[tokio::test]
    async fn realtime_partial_failure_records_once_and_still_refunds_the_key() {
        for end in [
            SessionEnd::UpstreamError(oxllm_core::Error::Network("closed".into())),
            SessionEnd::TaskAborted,
        ] {
            let (state, spend_key) = state_with_reservation().await;
            let mut state = std::sync::Arc::try_unwrap(state)
                .ok()
                .expect("sole test owner");
            let (tx, mut ledger) = tokio::sync::mpsc::channel(8);
            state.spend_tx = Some(tx.clone());
            let probe = std::sync::Arc::new(SettlementProbe::default());
            state.hooks = vec![
                std::sync::Arc::new(crate::hooks::BudgetHook::new(
                    state.cache.clone(),
                    Some(1.0),
                    "30d",
                )),
                std::sync::Arc::new(crate::hooks::CostTrackingHook::with_spend_writer(Some(tx))),
                probe.clone(),
            ];
            let ctx = oxllm_core::CallContext {
                budget_estimate: Some(0.01),
                session_id: Some("realtime-session".into()),
                ..oxllm_core::CallContext::for_test("gpt-4o-realtime")
            };
            let usage = UsageSummary {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
                ..Default::default()
            };
            settle_realtime_session(
                &state,
                &ctx,
                "gpt-4o-realtime",
                "openai",
                "dep-key",
                Ok(SessionOutcome {
                    usage: Some(usage.clone()),
                    unmetered_output: false,
                    end,
                }),
            )
            .await;
            let entry = ledger.try_recv().unwrap();
            assert_eq!(entry.status.as_deref(), Some("partial"));
            assert_eq!(entry.session_id.as_deref(), Some("realtime-session"));
            assert_eq!(entry.total_tokens, Some(15));
            assert_eq!(entry.request_id.as_ref(), Some(&ctx.request_id));
            assert!(
                ledger.try_recv().is_err(),
                "a realtime session has one terminal audit row"
            );
            assert_eq!(probe.failures.load(std::sync::atomic::Ordering::SeqCst), 1);
            assert!(state.cache.counter_value(&spend_key).await.unwrap().abs() < 1e-12);
            record_partial_realtime(
                &state,
                &ctx.clone(),
                "gpt-4o-realtime",
                "openai",
                Some(&usage),
            )
            .await;
            assert!(
                ledger.try_recv().is_err(),
                "cloned contexts share realtime reporting claims"
            );
        }
    }

    #[tokio::test(start_paused = true)]
    async fn realtime_partial_survives_transient_ledger_backpressure_and_refunds() {
        let (state, spend_key) = state_with_reservation().await;
        let mut state = std::sync::Arc::try_unwrap(state).ok().unwrap();
        let (tx, mut ledger) = tokio::sync::mpsc::channel(1);
        tx.try_send(
            serde_json::from_value(serde_json::json!({
                "timestamp":"2026-10-11T00:00:00Z","model":"filler","provider":"test"
            }))
            .unwrap(),
        )
        .unwrap();
        state.spend_tx = Some(tx.clone());
        state.hooks = vec![
            std::sync::Arc::new(crate::hooks::BudgetHook::new(
                state.cache.clone(),
                Some(1.0),
                "30d",
            )),
            std::sync::Arc::new(crate::hooks::CostTrackingHook::with_spend_writer(Some(tx))),
        ];
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
        let ctx = oxllm_core::CallContext {
            budget_estimate: Some(0.01),
            deployment_model: Some("openai/gpt-4o".into()),
            ..oxllm_core::CallContext::for_test("gpt-4o-realtime")
        };
        settle_realtime_session(
            &state,
            &ctx,
            "gpt-4o-realtime",
            "openai",
            "dep-key",
            Ok(SessionOutcome {
                usage: Some(UsageSummary {
                    prompt_tokens: 10,
                    completion_tokens: 5,
                    total_tokens: 15,
                    ..Default::default()
                }),
                unmetered_output: false,
                end: SessionEnd::UpstreamError(oxllm_core::Error::Network("closed".into())),
            }),
        )
        .await;
        let row = receiver.await.unwrap();
        assert_eq!(row.status.as_deref(), Some("partial"));
        assert_eq!(row.total_tokens, Some(15));
        assert!(row.cost.unwrap() > 0.0);
        assert!(state.cache.counter_value(&spend_key).await.unwrap().abs() < 1e-12);
    }

    #[tokio::test]
    async fn realtime_partial_ledger_matches_reported_router_cost() {
        let (state, spend_key) = state_with_reservation().await;
        let mut state = std::sync::Arc::try_unwrap(state).ok().unwrap();
        let (tx, mut ledger) = tokio::sync::mpsc::channel(8);
        state.spend_tx = Some(tx.clone());
        let probe = std::sync::Arc::new(SettlementProbe::default());
        state.hooks = vec![
            std::sync::Arc::new(crate::hooks::BudgetHook::new(
                state.cache.clone(),
                Some(1.0),
                "30d",
            )),
            std::sync::Arc::new(crate::hooks::CostTrackingHook::with_spend_writer(Some(tx))),
            probe.clone(),
        ];
        let deployment = oxllm_core::Deployment {
            model_name: "priced-session".into(),
            oxllm_params: OxllmParams {
                model: "openai/gpt-4o".into(),
                max_budget: Some(1.0),
                input_cost_per_token: Some(2e-6),
                output_cost_per_token: Some(4e-6),
                ..Default::default()
            },
            ..Default::default()
        };
        let ctx = oxllm_core::CallContext {
            budget_estimate: Some(0.01),
            deployment_pricing: deployment.oxllm_params.pricing_override(),
            ..oxllm_core::CallContext::for_test("priced-session")
        };
        let usage = UsageSummary {
            prompt_tokens: 10,
            completion_tokens: 5,
            total_tokens: 15,
            ..Default::default()
        };
        let dep_key = oxllm_core::router::deployment_key(&deployment);
        settle_reserved_realtime_session(
            &state,
            &ctx,
            &deployment,
            "gpt-4o",
            "openai",
            &dep_key,
            Ok(SessionOutcome {
                usage: Some(usage),
                unmetered_output: false,
                end: SessionEnd::UpstreamError(oxllm_core::Error::Network("closed".into())),
            }),
            RealtimeRouterBudgetGuard::new(state.cache.clone(), Default::default()),
        )
        .await;
        let row = ledger.try_recv().unwrap();
        let cost = row
            .cost
            .expect("reported priced usage remains in the partial ledger");
        assert!((cost - 4e-5).abs() < 1e-12);
        assert_eq!(row.status.as_deref(), Some("partial"));
        let settled = state
            .cache
            .counter_value(&crate::hooks::spend_keys::deployment_spend_key(&dep_key))
            .await
            .unwrap();
        assert!((settled - cost).abs() < 1e-12);
        assert!(state.cache.counter_value(&spend_key).await.unwrap().abs() < 1e-12);
        assert_eq!(probe.failures.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert!(ledger.try_recv().is_err());
    }

    #[derive(Default)]
    struct SettlementProbe {
        successes: std::sync::atomic::AtomicUsize,
        failures: std::sync::atomic::AtomicUsize,
        concurrency_releases: std::sync::atomic::AtomicUsize,
    }

    #[async_trait::async_trait]
    impl oxllm_core::Hook for SettlementProbe {
        async fn post_success(
            &self,
            _: &oxllm_core::CallContext,
            _: &oxllm_core::hook::ResponseSummary,
        ) {
            self.successes
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }

        async fn post_failure(&self, _: &oxllm_core::CallContext, _: &oxllm_core::Error) {
            self.failures
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }

        async fn release_concurrency(&self, _: &oxllm_core::CallContext) {
            self.concurrency_releases
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
    }

    #[tokio::test]
    async fn settlement_distinguishes_setup_failures_idle_and_unmetered_output() {
        use std::sync::atomic::Ordering::SeqCst;
        let cases = [
            (
                Err(oxllm_core::Error::Config("presign failed".into())),
                0,
                1,
                false,
            ),
            (
                Err(oxllm_core::Error::Network("connect failed".into())),
                0,
                1,
                false,
            ),
            (
                Ok(SessionOutcome {
                    usage: None,
                    unmetered_output: false,
                    end: SessionEnd::UpstreamError(oxllm_core::Error::Network(
                        "upstream closed with 1011".into(),
                    )),
                }),
                0,
                1,
                false,
            ),
            (
                Ok(SessionOutcome {
                    usage: None,
                    unmetered_output: false,
                    end: SessionEnd::UpstreamClosed,
                }),
                1,
                0,
                false,
            ),
            (
                Ok(SessionOutcome {
                    usage: None,
                    unmetered_output: true,
                    end: SessionEnd::ClientDisconnected,
                }),
                0,
                0,
                true,
            ),
        ];
        for (outcome, successes, failures, partial) in cases {
            let (mut state, spend_key) = state_with_reservation().await;
            let probe = std::sync::Arc::new(SettlementProbe::default());
            let (tx, mut rx) = tokio::sync::mpsc::channel(4);
            let mutable = std::sync::Arc::get_mut(&mut state).unwrap();
            mutable.hooks.push(probe.clone());
            mutable.spend_tx = Some(tx);
            let ctx = oxllm_core::CallContext {
                budget_estimate: Some(0.01),
                ..oxllm_core::CallContext::for_test("gpt-4o")
            };
            settle_realtime_session(&state, &ctx, "gpt-4o", "bedrock", "dep", outcome).await;
            assert_eq!(probe.successes.load(SeqCst), successes);
            assert_eq!(probe.failures.load(SeqCst), failures);
            let spend = state.cache.counter_value(&spend_key).await.unwrap_or(0.0);
            if partial {
                assert!((spend - 0.01).abs() < 1e-9, "existing estimate must stand");
                assert!(probe.concurrency_releases.load(SeqCst) > 0);
                let entry = rx.try_recv().expect("partial ledger entry");
                assert_eq!(entry.status.as_deref(), Some("partial"));
                assert!(
                    entry.cost.is_none(),
                    "unknown usage must not fabricate a charge"
                );
                assert!(
                    entry.total_tokens.is_none(),
                    "unknown is not reported zero usage"
                );
            } else {
                assert!(
                    spend.abs() < 1e-9,
                    "setup/idle reservation must be released"
                );
            }
        }
    }

    #[tokio::test]
    async fn realtime_propagates_checked_router_tag_rejection() {
        let (state, _) = state_with_reservation().await;
        let mut state = (*state).clone();
        state.deployments = vec![dep("openai/gpt-4o")];
        state.router = std::sync::Arc::new(oxllm_core::router::Router::new(
            oxllm_core::router::RouterConfig {
                routing_strategy: oxllm_core::RoutingStrategyKind::TagBased,
                ..Default::default()
            },
        ));
        let error =
            pick_realtime_deployment(&state, "openai/gpt-4o", None, &["&missing".to_string()])
                .unwrap_err();
        assert_eq!(error.status, axum::http::StatusCode::BAD_REQUEST);
        assert!(error.message.contains("routing tags"));
    }
}
