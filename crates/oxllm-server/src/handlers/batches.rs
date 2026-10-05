//! `POST/GET /v1/batches` and `POST /v1/batches/{id}/cancel` — OpenAI Batches
//! API passthrough.
//!
//! All requests flow through [`PassthroughMode::HookedRaw`], so the full
//! pre/post hook chain (rate-limit, budget, deployment RPM/TPM, spend log)
//! fires for every batch operation. The model used to resolve the deployment
//! comes from the request body (POST — read by the pipeline from the parsed
//! JSON) or `?model=` query (GET/cancel). When neither is present the
//! pipeline's `resolve_model` picks the first deployment whose provider slug
//! matches the `openai` hint stamped here — matching the `handlers::passthrough`
//! images/audio convention — so a multi-provider config whose first deployment
//! is e.g. `anthropic/claude` does not route an OpenAI batches body to the
//! Anthropic upstream.

use axum::Extension;
use axum::body::Bytes;
use axum::extract::{OriginalUri, Path, Query, State};
use axum::http::Method;

use crate::auth::AuthUser;
use crate::error::ApiError;
use crate::handlers::passthrough::ModelQuery;
use crate::handlers::platform_common::{content_type_for, validate_path_param};
use crate::pipeline::{PassthroughMode, PassthroughRequest, forward_with_mode};
use crate::state::SharedState;

/// `POST /v1/batches` — create a batch (forward to upstream `/batches`).
pub async fn create_batch(
    State(state): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Query(query): Query<ModelQuery>,
    headers: axum::http::HeaderMap,
    body: Bytes,
) -> Result<axum::response::Response, ApiError> {
    let app_state = state.load_full();
    forward_with_mode(
        &app_state,
        &auth,
        PassthroughMode::HookedRaw {
            upstream_path: "batches".to_string(),
        },
        PassthroughRequest::new(
            Method::POST,
            query.model.as_deref(),
            None,
            body,
            content_type_for(&headers),
        )
        .with_request_headers(&headers)
        .with_provider_hint("openai"),
    )
    .await
}

/// `GET /v1/batches` — list batches.
pub async fn list_batches(
    State(state): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    OriginalUri(original): OriginalUri,
    Query(query): Query<ModelQuery>,
    headers: axum::http::HeaderMap,
) -> Result<axum::response::Response, ApiError> {
    let app_state = state.load_full();
    forward_with_mode(
        &app_state,
        &auth,
        PassthroughMode::HookedRaw {
            upstream_path: "batches".to_string(),
        },
        PassthroughRequest::new(
            Method::GET,
            query.model.as_deref(),
            original.query(),
            Bytes::new(),
            None,
        )
        .with_request_headers(&headers)
        .with_provider_hint("openai"),
    )
    .await
}

/// `GET /v1/batches/{batch_id}` — retrieve a batch.
pub async fn retrieve_batch(
    State(state): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    OriginalUri(original): OriginalUri,
    Path(batch_id): Path<String>,
    Query(query): Query<ModelQuery>,
    headers: axum::http::HeaderMap,
) -> Result<axum::response::Response, ApiError> {
    let app_state = state.load_full();
    validate_path_param(&batch_id)?;
    let path = format!("batches/{batch_id}");
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
        .with_provider_hint("openai"),
    )
    .await
}

/// `POST /v1/batches/{batch_id}/cancel` — cancel a batch.
pub async fn cancel_batch(
    State(state): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Path(batch_id): Path<String>,
    Query(query): Query<ModelQuery>,
    headers: axum::http::HeaderMap,
    body: Bytes,
) -> Result<axum::response::Response, ApiError> {
    let app_state = state.load_full();
    validate_path_param(&batch_id)?;
    let path = format!("batches/{batch_id}/cancel");
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
            content_type_for(&headers),
        )
        .with_request_headers(&headers)
        .with_provider_hint("openai"),
    )
    .await
}
