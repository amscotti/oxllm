//! `POST /v1/messages/count_tokens` — Anthropic Count Tokens API.
//!
//! Anthropic exposes `/v1/messages/count_tokens` to return the exact token
//! count for a given message set without generating a completion. The request
//! and response are already Anthropic-shaped, so — unlike `/v1/messages` —
//! there is nothing to convert: the body is forwarded verbatim and the
//! upstream response is returned as-is.
//!
//! The handler is a thin wrapper around
//! [`crate::pipeline::forward_with_mode`] with [`PassthroughMode::HookedRaw`],
//! so the request still runs the full pre/post hook chain (auth via the proxy
//! middleware, rate-limit, budget, spend tracking, deployment RPM/TPM) and
//! resolves a deployment through the router (`pick_deployment_index`) exactly
//! like the other LLM endpoints.

use axum::Extension;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::Response;

use crate::auth::AuthUser;
use crate::error::ApiError;
use crate::handlers::anthropic_common::extract_request_model;
use crate::pipeline::{
    PassthroughMode, PassthroughRequest, forward_with_mode, vetted_client_headers,
};
use crate::state::SharedState;

/// Anthropic Count Tokens handler.
///
/// Flow:
/// 1. Parse the Anthropic-format body and extract the `model` field (required
///    — without it the upstream count_tokens call is meaningless).
/// 2. Delegate to [`forward_with_mode`] in [`PassthroughMode::HookedRaw`]:
///    pre-call hooks (rate-limit, budget, cache lookup) → router picks a
///    deployment → deployment RPM/TPM limits → forward the body verbatim to
///    `{api_base}/messages/count_tokens` with provider-native auth →
///    post-success/post-failure hooks (spend tracking, metrics).
/// 3. Return the upstream Anthropic response directly (no conversion).
///
/// Usage metering on this route is effectively a no-op for *settled* spend:
/// count_tokens is free on Anthropic, and even though the passthrough tier
/// conservatively reserves a prompt-based budget estimate for the duration of
/// the upstream call (`BudgetHook::pre_call`), `post_success` records
/// `cost = None` and the reservation is released to zero. The transient
/// reservation can briefly appear to consume budget under heavy concurrency,
/// but no permanent charge accrues. (This is a real billing gap only on the
/// sibling `/anthropic/v1/messages` passthrough — see
/// [`crate::handlers::anthropic_messages_passthrough`].)
pub async fn count_tokens(
    State(state): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, ApiError> {
    let model = extract_request_model(&body)?;
    forward_with_mode(
        &state.load_full(),
        &auth,
        PassthroughMode::HookedRaw {
            upstream_path: "messages/count_tokens".to_string(),
        },
        PassthroughRequest::new(
            axum::http::Method::POST,
            Some(&model),
            None,
            body,
            Some("application/json"),
        )
        .with_request_headers(&headers)
        .with_forward_headers(vetted_client_headers(&headers)),
    )
    .await
}
