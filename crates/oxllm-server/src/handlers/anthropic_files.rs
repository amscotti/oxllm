//! Anthropic Files API — `/anthropic/v1/files`.
//!
//! Anthropic exposes a native files surface at `/v1/files` (beta) for use
//! with the Messages API, message batches, and other features. It mirrors
//! the OpenAI Files CRUD shape but lives on a different upstream host and
//! uses Anthropic-native auth (`x-api-key` + `anthropic-version`, applied
//! automatically by the provider registry via [`PassthroughMode::HookedRaw`]).
//!
//! Route prefix: Anthropic files are mounted at `/anthropic/v1/files` rather
//! than `/v1/files` to avoid colliding with the OpenAI Files API registered
//! under the `platform-apis` feature. The path mirrors the existing
//! `/anthropic/v1/messages` native-passthrough convention.
//! Proxy authentication headers never select this namespace. With multiple
//! Anthropic deployments, use a `?model=` alias resolving to one deployment so
//! file IDs cannot be sent to a different account on a later operation.
//!
//! All requests flow through [`PassthroughMode::HookedRaw`], so the full
//! pre/post hook chain (auth via the proxy middleware, rate-limit, budget,
//! spend tracking, deployment RPM/TPM) fires for every file operation.
//! Multipart upload bodies are preserved (the content-type header — including
//! the multipart boundary — is forwarded verbatim). File content downloads
//! stream the upstream body directly, which may be binary or JSONL.
//!
//! Managed file IDs: this implementation passes the Anthropic `file_id`
//! through directly. Multi-provider file routing (a managed ID namespace
//! that decouples client-visible IDs from provider-specific storage) is a
//! future enhancement.

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

/// `POST /anthropic/v1/files` — upload a file (multipart passthrough).
pub async fn upload_file(
    State(state): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Query(query): Query<ModelQuery>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<axum::response::Response, ApiError> {
    let app_state = state.load_full();
    // Forward the inbound content-type verbatim — it carries the form boundary
    // the client generated, which the upstream needs to parse the body. When
    // the client omitted the header, pass `None` (matching the OpenAI
    // [`crate::handlers::files`] sibling) rather than a bare
    // `multipart/form-data` without a `boundary=` parameter: RFC 7578 requires
    // the boundary in the content-type for multipart bodies, so a synthesized
    // bare value would only guarantee an upstream rejection.
    let content_type = content_type_for(&headers);
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
            content_type,
        )
        .with_request_headers(&headers)
        .with_forward_headers(vetted_client_headers(&headers))
        .with_provider_hint("anthropic"),
    )
    .await
}

/// `GET /anthropic/v1/files` — list files.
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
        .with_forward_headers(vetted_client_headers(&headers))
        .with_provider_hint("anthropic"),
    )
    .await
}

/// `GET /anthropic/v1/files/{file_id}` — retrieve file metadata.
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
        .with_forward_headers(vetted_client_headers(&headers))
        .with_provider_hint("anthropic"),
    )
    .await
}

/// `GET /anthropic/v1/files/{file_id}/content` — download file content.
///
/// The upstream body may be binary (e.g. a PDF) or JSONL (e.g. batch
/// results); it is streamed back to the client unchanged.
pub async fn download_file_content(
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
        .with_forward_headers(vetted_client_headers(&headers))
        .with_provider_hint("anthropic"),
    )
    .await
}

/// `DELETE /anthropic/v1/files/{file_id}` — delete a file.
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
        .with_forward_headers(vetted_client_headers(&headers))
        .with_provider_hint("anthropic"),
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upload_file_with_no_content_type_forwards_none() {
        // Regression for the bare `multipart/form-data` default: when the
        // client omits the header, `content_type_for` returns `None` and the
        // handler forwards that rather than synthesising an invalid value.
        let headers = HeaderMap::new();
        assert_eq!(content_type_for(&headers), None);
    }

    #[test]
    fn upload_file_preserves_inbound_boundary() {
        let mut headers = HeaderMap::new();
        headers.insert(
            "content-type",
            "multipart/form-data; boundary=xyz".parse().unwrap(),
        );
        assert_eq!(
            content_type_for(&headers),
            Some("multipart/form-data; boundary=xyz")
        );
    }
}
