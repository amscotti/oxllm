//! Anthropic Message Batches API — `/v1/messages/batches`.
//!
//! Anthropic exposes a native batches surface at `/v1/messages/batches` that
//! differs from OpenAI's `/v1/batches`: the request/response shapes are
//! Anthropic-native, the lifecycle status values are `in_progress`,
//! `canceling`, and `ended` (not OpenAI's `completed`/`failed`/`cancelled`),
//! and batch results are streamed as Anthropic JSONL. Because the wire format
//! is already Anthropic-shaped there is nothing to convert — the body is
//! forwarded verbatim and the upstream response is returned as-is.
//!
//! All requests flow through [`PassthroughMode::HookedRaw`], so the full
//! pre/post hook chain (auth via the proxy middleware, rate-limit, budget,
//! spend tracking, deployment RPM/TPM) fires for every batch operation and a
//! deployment is resolved through the router exactly like the other LLM
//! endpoints.
//!
//! Status mapping: Anthropic statuses (`in_progress`, `canceling`, `ended`)
//! are returned verbatim — this is an Anthropic-native endpoint, so we do NOT
//! normalize to OpenAI's status vocabulary. Callers consuming the response
//! should expect Anthropic's `processing_status` field.

use axum::Extension;
use axum::body::Bytes;
use axum::extract::{OriginalUri, Path, Query, State};
use axum::http::{HeaderMap, Method};

use crate::auth::AuthUser;
use crate::error::ApiError;
use crate::handlers::passthrough::ModelQuery;
use crate::handlers::platform_common::{content_type_for, validate_path_param};
use crate::pipeline::{
    PassthroughMode, PassthroughRequest, forward_with_mode, vetted_client_headers,
};
use crate::state::SharedState;

/// `POST /v1/messages/batches` — create a batch (forward to upstream
/// `/v1/messages/batches`).
///
/// The Anthropic batches body nests the model under
/// `requests[0].params.model`, which the pipeline's `resolve_model`
/// extracts as a fallback after parsing the body once for the depth guard
/// — no double parse. When no model is present the `anthropic` provider
/// hint picks the first anthropic-provider deployment.
pub async fn create_batch(
    State(state): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<axum::response::Response, ApiError> {
    let app_state = state.load_full();
    forward_with_mode(
        &app_state,
        &auth,
        PassthroughMode::HookedRaw {
            upstream_path: "messages/batches".to_string(),
        },
        PassthroughRequest::new(
            Method::POST,
            None,
            None,
            body,
            content_type_for(&headers).or(Some("application/json")),
        )
        .with_request_headers(&headers)
        .with_forward_headers(vetted_client_headers(&headers))
        .with_provider_hint("anthropic"),
    )
    .await
}

/// `GET /v1/messages/batches` — list batches.
pub async fn list_batches(
    State(state): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    OriginalUri(original): OriginalUri,
    Query(query): Query<ModelQuery>,
    headers: HeaderMap,
) -> Result<axum::response::Response, ApiError> {
    let app_state = state.load_full();
    forward_with_mode(
        &app_state,
        &auth,
        PassthroughMode::HookedRaw {
            upstream_path: "messages/batches".to_string(),
        },
        PassthroughRequest::new(
            Method::GET,
            query.model.as_deref(),
            original.query(),
            Bytes::new(),
            None,
        )
        .with_request_headers(&headers)
        .with_forward_headers(vetted_client_headers(&headers))
        .with_provider_hint("anthropic"),
    )
    .await
}

/// `GET /v1/messages/batches/{batch_id}` — retrieve a batch.
pub async fn retrieve_batch(
    State(state): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    OriginalUri(original): OriginalUri,
    Path(batch_id): Path<String>,
    Query(query): Query<ModelQuery>,
    headers: HeaderMap,
) -> Result<axum::response::Response, ApiError> {
    let app_state = state.load_full();
    validate_path_param(&batch_id)?;
    let path = format!("messages/batches/{batch_id}");
    forward_with_mode(
        &app_state,
        &auth,
        PassthroughMode::HookedRaw {
            upstream_path: path,
        },
        PassthroughRequest::new(
            Method::GET,
            query.model.as_deref(),
            original.query(),
            Bytes::new(),
            None,
        )
        .with_request_headers(&headers)
        .with_forward_headers(vetted_client_headers(&headers))
        .with_provider_hint("anthropic"),
    )
    .await
}

/// `POST /v1/messages/batches/{batch_id}/cancel` — cancel a batch.
pub async fn cancel_batch(
    State(state): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Path(batch_id): Path<String>,
    Query(query): Query<ModelQuery>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<axum::response::Response, ApiError> {
    let app_state = state.load_full();
    validate_path_param(&batch_id)?;
    let path = format!("messages/batches/{batch_id}/cancel");
    forward_with_mode(
        &app_state,
        &auth,
        PassthroughMode::HookedRaw {
            upstream_path: path,
        },
        PassthroughRequest::new(
            Method::POST,
            query.model.as_deref(),
            None,
            body,
            content_type_for(&headers).or(Some("application/json")),
        )
        .with_request_headers(&headers)
        .with_forward_headers(vetted_client_headers(&headers))
        .with_provider_hint("anthropic"),
    )
    .await
}

/// `GET /v1/messages/batches/{batch_id}/results` — download a batch's results
/// as Anthropic JSONL. The fifth endpoint of the Anthropic Message Batches API
/// (`litellm/llms/anthropic/files/handler.py:94` constructs the same upstream
/// path). Relayed verbatim like the other batch operations — the response body
/// is already Anthropic-shaped JSONL.
pub async fn retrieve_batch_results(
    State(state): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    OriginalUri(original): OriginalUri,
    Path(batch_id): Path<String>,
    Query(query): Query<ModelQuery>,
    headers: HeaderMap,
) -> Result<axum::response::Response, ApiError> {
    let app_state = state.load_full();
    validate_path_param(&batch_id)?;
    let path = format!("messages/batches/{batch_id}/results");
    forward_with_mode(
        &app_state,
        &auth,
        PassthroughMode::HookedRaw {
            upstream_path: path,
        },
        PassthroughRequest::new(
            Method::GET,
            query.model.as_deref(),
            original.query(),
            Bytes::new(),
            None,
        )
        .with_request_headers(&headers)
        .with_forward_headers(vetted_client_headers(&headers))
        .with_provider_hint("anthropic"),
    )
    .await
}
