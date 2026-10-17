//! Native Anthropic `/v1/messages` passthrough.
//!
//! The existing [`crate::handlers::messages::messages`] handler at
//! `POST /v1/messages` translates Anthropic-shaped requests to the internal
//! OpenAI format, routes through the chat pipeline, and converts the OpenAI
//! response back to Anthropic format (Anthropic → OpenAI → Anthropic). That
//! round-trip is lossy for features with no OpenAI equivalent (prompt caching
//! hints, tool-choice variants, `anthropic-beta` governed fields, etc.).
//!
//! This module provides a native passthrough that forwards the Anthropic body
//! in its native shape to the upstream provider and returns the response as-is
//! — no conversion in either direction. It runs the same hook chain as the
//! other LLM endpoints (auth via the proxy middleware, rate-limit, budget,
//! spend tracking, deployment RPM/TPM) and resolves a deployment through the
//! router. Model aliases are rewritten and configured request guardrails mask
//! or reject prompt text while preserving native content blocks.
//!
//! ## Routing
//!
//! The native mode is exposed at a dedicated path, `POST /anthropic/v1/messages`,
//! rather than `/v1/messages` (which is owned by the conversion handler). It
//! shadows the global provider catch-all `/{*path}` for that exact path, so
//! clients that want the unconverted path can target it directly. The default
//! `POST /v1/messages` continues to use the conversion path.

use axum::Extension;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};

use crate::auth::AuthUser;
use crate::error::ApiError;
use crate::handlers::anthropic_common::{
    anthropic_error_response, extract_request_model, status_to_anthropic_error_type,
};
use crate::pipeline::{
    PassthroughMode, PassthroughRequest, forward_with_mode, preserved_response_headers,
    vetted_client_headers,
};
use crate::state::SharedState;

/// Native Anthropic Messages passthrough handler.
///
/// Forwards the Anthropic-format request body in its native shape to the upstream
/// provider's `messages` endpoint and returns the upstream response as-is.
/// Streaming requests (`"stream": true`) pass through unchanged because the
/// shared passthrough tier does not buffer the body.
///
/// Flow (via [`forward_with_mode`] with [`PassthroughMode::HookedRaw`]):
/// 1. Parse the body and extract the `model` (required for routing).
/// 2. Pre-call hooks (rate-limit, budget, cache lookup).
/// 3. Router picks a deployment (`pick_deployment_index`).
/// 4. Deployment RPM/TPM limits + router spend caps.
/// 5. Forward the body verbatim to `{api_base}/messages` with provider-native
///    auth headers (e.g. Anthropic `x-api-key` + `anthropic-version`).
/// 6. Post-success/post-failure hooks (spend tracking, metrics).
///
/// # Usage metering
///
/// **Non-streaming** responses are metered: usage is parsed from the
/// Anthropic response body (`usage.input_tokens`/`output_tokens`/`cache_*`)
/// and settled against key/team/model budgets by `post_success`, matching
/// the reference passthrough spend tracking.
///
/// **Streaming** responses relay native SSE bytes incrementally while the
/// shared relay accumulates usage and settles spend on completion. Disconnects
/// and provider errors follow the same partial-accounting policy as chat.
/// This native route makes one attempt; translated `/v1/messages` uses the chat
/// retry/fallback chain before client commitment.
pub async fn messages_passthrough(
    State(state): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, ApiError> {
    // Local failures (validation, ACL, hooks, routing) render the Anthropic
    // error envelope so an Anthropic SDK client on this native route never
    // has to parse the gateway's default OpenAI shape — matching the
    // reference anthropic endpoints, which wrap every handler-raised
    // exception via `transform_to_anthropic_error`. Upstream failures are
    // unaffected: the hooked-raw tier relays provider responses verbatim as
    // `Ok` values and never reaches this wrapper.
    match messages_passthrough_inner(state, auth, headers, body).await {
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

async fn messages_passthrough_inner(
    state: SharedState,
    auth: AuthUser,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, ApiError> {
    let state = state.load_full();
    let model = extract_request_model(&body)?;
    forward_with_mode(
        &state,
        &auth,
        PassthroughMode::HookedRaw {
            upstream_path: "messages".to_string(),
        },
        PassthroughRequest::new(
            axum::http::Method::POST,
            Some(&model),
            None,
            body,
            Some("application/json"),
        )
        .with_request_headers(&headers)
        .with_provider_hint("anthropic")
        .with_forward_headers(vetted_client_headers(&headers)),
    )
    .await
}
