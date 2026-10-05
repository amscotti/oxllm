//! `/v1/threads` and sub-routes — OpenAI Threads API passthrough.
//!
//! All requests flow through [`PassthroughMode::HookedRaw`], so the full
//! pre/post hook chain (rate-limit, budget, deployment RPM/TPM, spend log)
//! fires for every thread / message / run operation. The model used to
//! resolve the deployment comes from the request body when present (notably
//! runs, which carry `model`) or `?model=` query; when neither is present
//! the pipeline picks the first `openai`-provider deployment so a
//! multi-provider config does not route an OpenAI threads body to the wrong
//! upstream. Threads and thread messages do not carry a model field of their
//! own — they rely on the provider-scoped fallback.
//!
//! # Metering (runs)
//!
//! The threads **runs** endpoints (`POST /v1/threads/{id}/runs`,
//! `POST /v1/threads/runs`, `POST /v1/threads/runs/stream`) are real LLM
//! inference calls whose JSON / SSE response bodies carry a `usage` object.
//! For non-streaming JSON responses, the hooked-raw path recovers usage from
//! the response body (`usage.prompt_tokens` / `completion_tokens` /
//! `total_tokens`) via the `recover_openai_threads_usage` helper in the
//! passthrough pipeline so
//! spend is recorded and the deployment TPM window is charged. Streaming
//! (`text/event-stream`) runs are relayed incrementally and usage metering
//! for the terminal `thread.run.completed` SSE event is not yet implemented
//! — operators should treat streaming runs spend as unmetered until then.

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

#[expect(clippy::too_many_arguments)]
async fn forward(
    app_state: &crate::state::AppState,
    auth: &AuthUser,
    method: Method,
    upstream_path: String,
    query: Option<&str>,
    body: Bytes,
    content_type: Option<&str>,
    headers: &HeaderMap,
    query_model: Option<&str>,
) -> Result<axum::response::Response, ApiError> {
    for segment in upstream_path.split('/') {
        if !segment.is_empty() {
            validate_path_param(segment)?;
        }
    }
    // Pass `query_model` (None for POST routes that don't accept `?model=`)
    // as the explicit routing input. The pipeline reads the `model` field
    // from a JSON POST body itself, so we deliberately do NOT pre-parse the
    // body here — that avoids the double JSON parse the platform handlers
    // used to perform and lets a single parse in `hooked_raw` feed both the
    // depth guard and the routing resolution. When neither source supplies
    // a model the provider hint stamps the fallback so the first
    // `openai`-provider deployment is selected rather than
    // `state.deployments.first()` (which may belong to a different provider
    // in a multi-provider config).
    forward_with_mode(
        app_state,
        auth,
        PassthroughMode::HookedRaw { upstream_path },
        PassthroughRequest::new(method, query_model, query, body, content_type)
            .with_request_headers(headers)
            .with_provider_hint("openai"),
    )
    .await
}

/// `POST /v1/threads` — create a thread.
pub async fn create_thread(
    State(state): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Query(query): Query<ModelQuery>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<axum::response::Response, ApiError> {
    let app_state = state.load_full();
    forward(
        &app_state,
        &auth,
        Method::POST,
        "threads".to_string(),
        None,
        body,
        content_type_for(&headers),
        &headers,
        query.model.as_deref(),
    )
    .await
}

/// `GET /v1/threads/{thread_id}` — retrieve a thread.
pub async fn retrieve_thread(
    State(state): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    OriginalUri(original): OriginalUri,
    Path(thread_id): Path<String>,
    Query(query): Query<ModelQuery>,
    headers: HeaderMap,
) -> Result<axum::response::Response, ApiError> {
    let app_state = state.load_full();
    validate_path_param(&thread_id)?;
    let path = format!("threads/{thread_id}");
    forward(
        &app_state,
        &auth,
        Method::GET,
        path,
        original.query(),
        Bytes::new(),
        None,
        &headers,
        query.model.as_deref(),
    )
    .await
}

/// `POST /v1/threads/{thread_id}` — modify a thread.
pub async fn modify_thread(
    State(state): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Path(thread_id): Path<String>,
    Query(query): Query<ModelQuery>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<axum::response::Response, ApiError> {
    let app_state = state.load_full();
    validate_path_param(&thread_id)?;
    let path = format!("threads/{thread_id}");
    forward(
        &app_state,
        &auth,
        Method::POST,
        path,
        None,
        body,
        content_type_for(&headers),
        &headers,
        query.model.as_deref(),
    )
    .await
}

/// `DELETE /v1/threads/{thread_id}` — delete a thread.
pub async fn delete_thread(
    State(state): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Path(thread_id): Path<String>,
    Query(query): Query<ModelQuery>,
    headers: HeaderMap,
) -> Result<axum::response::Response, ApiError> {
    let app_state = state.load_full();
    validate_path_param(&thread_id)?;
    let path = format!("threads/{thread_id}");
    forward(
        &app_state,
        &auth,
        Method::DELETE,
        path,
        None,
        Bytes::new(),
        None,
        &headers,
        query.model.as_deref(),
    )
    .await
}

/// `POST /v1/threads/{thread_id}/messages` — create a message in a thread.
pub async fn create_message(
    State(state): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Path(thread_id): Path<String>,
    Query(query): Query<ModelQuery>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<axum::response::Response, ApiError> {
    let app_state = state.load_full();
    validate_path_param(&thread_id)?;
    let path = format!("threads/{thread_id}/messages");
    forward(
        &app_state,
        &auth,
        Method::POST,
        path,
        None,
        body,
        content_type_for(&headers),
        &headers,
        query.model.as_deref(),
    )
    .await
}

/// `GET /v1/threads/{thread_id}/messages` — list messages in a thread.
pub async fn list_messages(
    State(state): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    OriginalUri(original): OriginalUri,
    Path(thread_id): Path<String>,
    Query(query): Query<ModelQuery>,
    headers: HeaderMap,
) -> Result<axum::response::Response, ApiError> {
    let app_state = state.load_full();
    validate_path_param(&thread_id)?;
    let path = format!("threads/{thread_id}/messages");
    forward(
        &app_state,
        &auth,
        Method::GET,
        path,
        original.query(),
        Bytes::new(),
        None,
        &headers,
        query.model.as_deref(),
    )
    .await
}

/// `GET /v1/threads/{thread_id}/messages/{message_id}` — retrieve a message.
pub async fn retrieve_message(
    State(state): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    OriginalUri(original): OriginalUri,
    Path((thread_id, message_id)): Path<(String, String)>,
    Query(query): Query<ModelQuery>,
    headers: HeaderMap,
) -> Result<axum::response::Response, ApiError> {
    let app_state = state.load_full();
    validate_path_param(&thread_id)?;
    validate_path_param(&message_id)?;
    let path = format!("threads/{thread_id}/messages/{message_id}");
    forward(
        &app_state,
        &auth,
        Method::GET,
        path,
        original.query(),
        Bytes::new(),
        None,
        &headers,
        query.model.as_deref(),
    )
    .await
}

/// `POST /v1/threads/{thread_id}/messages/{message_id}` — modify a message.
pub async fn modify_message(
    State(state): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Path((thread_id, message_id)): Path<(String, String)>,
    Query(query): Query<ModelQuery>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<axum::response::Response, ApiError> {
    let app_state = state.load_full();
    validate_path_param(&thread_id)?;
    validate_path_param(&message_id)?;
    let path = format!("threads/{thread_id}/messages/{message_id}");
    forward(
        &app_state,
        &auth,
        Method::POST,
        path,
        None,
        body,
        content_type_for(&headers),
        &headers,
        query.model.as_deref(),
    )
    .await
}

/// `DELETE /v1/threads/{thread_id}/messages/{message_id}` — delete a message.
pub async fn delete_message(
    State(state): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Path((thread_id, message_id)): Path<(String, String)>,
    Query(query): Query<ModelQuery>,
    headers: HeaderMap,
) -> Result<axum::response::Response, ApiError> {
    let app_state = state.load_full();
    validate_path_param(&thread_id)?;
    validate_path_param(&message_id)?;
    let path = format!("threads/{thread_id}/messages/{message_id}");
    forward(
        &app_state,
        &auth,
        Method::DELETE,
        path,
        None,
        Bytes::new(),
        None,
        &headers,
        query.model.as_deref(),
    )
    .await
}

/// `POST /v1/threads/runs` — create a run (thread_id in body).
///
/// Note: this is a real LLM inference call. Token usage in the non-streaming
/// JSON response body is metered via body-level usage recovery; see the
/// module-level "Metering (runs)" doc-comment.
pub async fn create_run_v2(
    State(state): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<axum::response::Response, ApiError> {
    let app_state = state.load_full();
    forward(
        &app_state,
        &auth,
        Method::POST,
        "threads/runs".to_string(),
        None,
        body,
        content_type_for(&headers),
        &headers,
        None,
    )
    .await
}

/// `POST /v1/threads/runs/stream` — create a streaming run (thread_id in body).
///
/// Note: this is a real LLM inference call. Usage in the terminal
/// `thread.run.completed` SSE event is not yet metered; see the module-level
/// "Metering (runs)" doc-comment.
pub async fn create_run_stream(
    State(state): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<axum::response::Response, ApiError> {
    let app_state = state.load_full();
    forward(
        &app_state,
        &auth,
        Method::POST,
        "threads/runs/stream".to_string(),
        None,
        body,
        content_type_for(&headers),
        &headers,
        None,
    )
    .await
}

/// `POST /v1/threads/{thread_id}/runs` — create a run.
///
/// Note: this is a real LLM inference call. Token usage in the non-streaming
/// JSON response body is metered via body-level usage recovery; see the
/// module-level "Metering (runs)" doc-comment.
pub async fn create_run(
    State(state): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Path(thread_id): Path<String>,
    Query(query): Query<ModelQuery>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<axum::response::Response, ApiError> {
    let app_state = state.load_full();
    validate_path_param(&thread_id)?;
    let path = format!("threads/{thread_id}/runs");
    forward(
        &app_state,
        &auth,
        Method::POST,
        path,
        None,
        body,
        content_type_for(&headers),
        &headers,
        query.model.as_deref(),
    )
    .await
}

/// `GET /v1/threads/{thread_id}/runs` — list runs.
pub async fn list_runs(
    State(state): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    OriginalUri(original): OriginalUri,
    Path(thread_id): Path<String>,
    Query(query): Query<ModelQuery>,
    headers: HeaderMap,
) -> Result<axum::response::Response, ApiError> {
    let app_state = state.load_full();
    validate_path_param(&thread_id)?;
    let path = format!("threads/{thread_id}/runs");
    forward(
        &app_state,
        &auth,
        Method::GET,
        path,
        original.query(),
        Bytes::new(),
        None,
        &headers,
        query.model.as_deref(),
    )
    .await
}

/// `GET /v1/threads/{thread_id}/runs/{run_id}` — retrieve a run.
pub async fn retrieve_run(
    State(state): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    OriginalUri(original): OriginalUri,
    Path((thread_id, run_id)): Path<(String, String)>,
    Query(query): Query<ModelQuery>,
    headers: HeaderMap,
) -> Result<axum::response::Response, ApiError> {
    let app_state = state.load_full();
    validate_path_param(&thread_id)?;
    validate_path_param(&run_id)?;
    let path = format!("threads/{thread_id}/runs/{run_id}");
    forward(
        &app_state,
        &auth,
        Method::GET,
        path,
        original.query(),
        Bytes::new(),
        None,
        &headers,
        query.model.as_deref(),
    )
    .await
}

/// `POST /v1/threads/{thread_id}/runs/{run_id}` — modify a run status.
pub async fn modify_run(
    State(state): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Path((thread_id, run_id)): Path<(String, String)>,
    Query(query): Query<ModelQuery>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<axum::response::Response, ApiError> {
    let app_state = state.load_full();
    validate_path_param(&thread_id)?;
    validate_path_param(&run_id)?;
    let path = format!("threads/{thread_id}/runs/{run_id}");
    forward(
        &app_state,
        &auth,
        Method::POST,
        path,
        None,
        body,
        content_type_for(&headers),
        &headers,
        query.model.as_deref(),
    )
    .await
}

/// `POST /v1/threads/{thread_id}/runs/{run_id}/cancel` — cancel a run.
pub async fn cancel_run(
    State(state): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Path((thread_id, run_id)): Path<(String, String)>,
    Query(query): Query<ModelQuery>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<axum::response::Response, ApiError> {
    let app_state = state.load_full();
    validate_path_param(&thread_id)?;
    validate_path_param(&run_id)?;
    let path = format!("threads/{thread_id}/runs/{run_id}/cancel");
    forward(
        &app_state,
        &auth,
        Method::POST,
        path,
        None,
        body,
        content_type_for(&headers),
        &headers,
        query.model.as_deref(),
    )
    .await
}
