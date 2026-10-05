//! `POST/GET /v1/fine_tuning/jobs` and `POST /v1/fine_tuning/jobs/{id}/cancel`
//! — OpenAI Fine-tuning API passthrough.
//!
//! All requests flow through [`PassthroughMode::HookedRaw`], so the full
//! pre/post hook chain (rate-limit, budget, deployment RPM/TPM, spend log)
//! fires for every fine-tune operation. The model used to resolve the
//! deployment comes from the request body (POST create — read by the pipeline
//! from the parsed JSON) or `?model=` query (GET/cancel); when neither is
//! present the pipeline picks the first `openai`-provider deployment so a
//! multi-provider config does not route an OpenAI fine-tune body to the wrong
//! upstream.

use axum::Extension;
use axum::body::Bytes;
use axum::extract::{OriginalUri, Path, Query, State};
use axum::http::{HeaderMap, Method};

use crate::auth::AuthUser;
use crate::error::ApiError;
use crate::handlers::passthrough::ModelQuery;
use crate::handlers::platform_common::{content_type_for, validate_path_param};
use crate::pipeline::{PassthroughMode, PassthroughRequest, forward_with_mode};
use crate::state::SharedState;

/// `POST /v1/fine_tuning/jobs` — create a fine-tune job.
pub async fn create_fine_tuning_job(
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
            upstream_path: "fine_tuning/jobs".to_string(),
        },
        PassthroughRequest::new(Method::POST, None, None, body, content_type_for(&headers))
            .with_request_headers(&headers)
            .with_provider_hint("openai"),
    )
    .await
}

/// `GET /v1/fine_tuning/jobs` — list fine-tune jobs.
pub async fn list_fine_tuning_jobs(
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
            upstream_path: "fine_tuning/jobs".to_string(),
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

/// `GET /v1/fine_tuning/jobs/{job_id}` — retrieve a fine-tune job.
pub async fn retrieve_fine_tuning_job(
    State(state): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    OriginalUri(original): OriginalUri,
    Path(job_id): Path<String>,
    Query(query): Query<ModelQuery>,
    headers: HeaderMap,
) -> Result<axum::response::Response, ApiError> {
    let app_state = state.load_full();
    validate_path_param(&job_id)?;
    let path = format!("fine_tuning/jobs/{job_id}");
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

/// `POST /v1/fine_tuning/jobs/{job_id}/cancel` — cancel a fine-tune job.
pub async fn cancel_fine_tuning_job(
    State(state): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Path(job_id): Path<String>,
    Query(query): Query<ModelQuery>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<axum::response::Response, ApiError> {
    let app_state = state.load_full();
    validate_path_param(&job_id)?;
    let path = format!("fine_tuning/jobs/{job_id}/cancel");
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
