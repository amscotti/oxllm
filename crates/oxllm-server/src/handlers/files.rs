//! `POST/GET/DELETE /v1/files`, `GET /v1/files/{id}/content` — OpenAI Files
//! API passthrough.
//!
//! All requests flow through [`PassthroughMode::HookedRaw`], so the full
//! pre/post hook chain (rate-limit, budget, deployment RPM/TPM, spend log)
//! fires for every file operation. Multipart upload bodies are preserved
//! (the content-type header — including the multipart boundary — is forwarded
//! verbatim). The `?model=` query selects the deployment when one isn't
//! present in the body; when neither is present the pipeline's
//! selects a deployment only when the OpenAI namespace is unambiguous.
//! A file operation must resolve to one deployment; otherwise specify an
//! explicit, single-deployment `?model=` alias. This route is always OpenAI:
//! Anthropic clients use `/anthropic/v1/files`, regardless of the proxy
//! authentication header spelling (`Authorization` / `x-api-key` / `api-key`).

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

/// `POST /v1/files` — upload a file (multipart passthrough).
pub async fn upload_file(
    State(state): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Query(query): Query<ModelQuery>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<axum::response::Response, ApiError> {
    let app_state = state.load_full();
    forward_with_mode(
        &app_state,
        &auth,
        PassthroughMode::HookedRaw {
            upstream_path: "files".to_string(),
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

/// `GET /v1/files` — list files.
pub async fn list_files(
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
            upstream_path: "files".to_string(),
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

/// `GET /v1/files/{file_id}` — retrieve file metadata.
pub async fn retrieve_file(
    State(state): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    OriginalUri(original): OriginalUri,
    Path(file_id): Path<String>,
    Query(query): Query<ModelQuery>,
    headers: HeaderMap,
) -> Result<axum::response::Response, ApiError> {
    let app_state = state.load_full();
    validate_path_param(&file_id)?;
    let path = format!("files/{file_id}");
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

/// `DELETE /v1/files/{file_id}` — delete a file.
pub async fn delete_file(
    State(state): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Path(file_id): Path<String>,
    Query(query): Query<ModelQuery>,
    headers: HeaderMap,
) -> Result<axum::response::Response, ApiError> {
    let app_state = state.load_full();
    validate_path_param(&file_id)?;
    let path = format!("files/{file_id}");
    forward_with_mode(
        &app_state,
        &auth,
        PassthroughMode::HookedRaw {
            upstream_path: path,
        },
        PassthroughRequest::new(
            Method::DELETE,
            query.model.as_deref(),
            None,
            Bytes::new(),
            None,
        )
        .with_request_headers(&headers)
        .with_provider_hint("openai"),
    )
    .await
}

/// `GET /v1/files/{file_id}/content` — download file content.
pub async fn retrieve_file_content(
    State(state): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    OriginalUri(original): OriginalUri,
    Path(file_id): Path<String>,
    Query(query): Query<ModelQuery>,
    headers: HeaderMap,
) -> Result<axum::response::Response, ApiError> {
    let app_state = state.load_full();
    validate_path_param(&file_id)?;
    let path = format!("files/{file_id}/content");
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
