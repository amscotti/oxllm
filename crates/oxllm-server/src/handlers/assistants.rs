//! `POST/GET /v1/assistants`, `POST/DELETE /v1/assistants/{id}` — OpenAI
//! Assistants API passthrough.
//!
//! All requests flow through [`PassthroughMode::HookedRaw`], so the full
//! pre/post hook chain (rate-limit, budget, deployment RPM/TPM, spend log)
//! fires for every assistant operation. The model used to resolve the
//! deployment comes from the request body (POST create/modify — read by the
//! pipeline from the parsed JSON) or `?model=` query (GET/list/delete); when
//! neither is present the pipeline picks the first `openai`-provider
//! deployment so a multi-provider config does not route an OpenAI assistants
//! body to the wrong upstream.

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

/// `POST /v1/assistants` — create an assistant.
pub async fn create_assistant(
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
            upstream_path: "assistants".to_string(),
        },
        PassthroughRequest::new(Method::POST, None, None, body, content_type_for(&headers))
            .with_request_headers(&headers)
            .with_provider_hint("openai"),
    )
    .await
}

/// `GET /v1/assistants` — list assistants.
pub async fn list_assistants(
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
            upstream_path: "assistants".to_string(),
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

/// `GET /v1/assistants/{assistant_id}` — retrieve an assistant.
pub async fn retrieve_assistant(
    State(state): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    OriginalUri(original): OriginalUri,
    Path(assistant_id): Path<String>,
    Query(query): Query<ModelQuery>,
    headers: HeaderMap,
) -> Result<axum::response::Response, ApiError> {
    let app_state = state.load_full();
    validate_path_param(&assistant_id)?;
    let path = format!("assistants/{assistant_id}");
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

/// `POST /v1/assistants/{assistant_id}` — modify an assistant.
pub async fn modify_assistant(
    State(state): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Path(assistant_id): Path<String>,
    Query(query): Query<ModelQuery>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<axum::response::Response, ApiError> {
    let app_state = state.load_full();
    validate_path_param(&assistant_id)?;
    let path = format!("assistants/{assistant_id}");
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

/// `DELETE /v1/assistants/{assistant_id}` — delete an assistant.
pub async fn delete_assistant(
    State(state): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Path(assistant_id): Path<String>,
    Query(query): Query<ModelQuery>,
    headers: HeaderMap,
) -> Result<axum::response::Response, ApiError> {
    let app_state = state.load_full();
    validate_path_param(&assistant_id)?;
    let path = format!("assistants/{assistant_id}");
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
