//! Azure OpenAI legacy route aliases.
//!
//! Azure SDK clients hit deployment-named paths instead of the standard
//! `/v1/chat/completions` / `/v1/embeddings` shape. These thin handlers
//! inject the `{model}` from the path into the request body (replacing
//! whatever `model` was supplied, if any) and delegate to the standard
//! chat / embeddings pipelines.

use axum::Extension;
use axum::body::Bytes;
use axum::extract::{Path, State};
use axum::response::Response;
use serde_json::Value;

use crate::auth::AuthUser;
use crate::error::ApiError;
use crate::handlers::chat::chat_completions_inner;
use crate::handlers::completions::completions as completions_handler;
use crate::handlers::embeddings::embeddings_inner_with_headers;
use crate::state::SharedState;

fn inject_model(body: Bytes, model: &str) -> Result<Bytes, ApiError> {
    let mut value: Value = serde_json::from_slice(&body)
        .map_err(|e| ApiError::bad_request(format!("invalid JSON body: {e}")))?;
    let obj = value.as_object_mut().ok_or_else(|| {
        ApiError::bad_request("expected a JSON object body for Azure legacy route")
    })?;
    obj.insert("model".to_string(), Value::String(model.to_string()));
    let bytes = serde_json::to_vec(&value)
        .map_err(|e| ApiError::internal(format!("failed to serialize body: {e}")))?;
    Ok(Bytes::from(bytes))
}

/// `POST /openai/deployments/{model}/chat/completions`
pub async fn deployments_chat_completions(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    headers: axum::http::HeaderMap,
    Path(model): Path<String>,
    body: Bytes,
) -> Result<Response, ApiError> {
    let state = shared.load_full();
    let body = inject_model(body, &model)?;
    chat_completions_inner(state, &auth, &headers, body).await
}

/// `POST /engines/{model}/chat/completions`
pub async fn engines_chat_completions(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    headers: axum::http::HeaderMap,
    Path(model): Path<String>,
    body: Bytes,
) -> Result<Response, ApiError> {
    let state = shared.load_full();
    let body = inject_model(body, &model)?;
    chat_completions_inner(state, &auth, &headers, body).await
}

/// `POST /openai/deployments/{model}/embeddings`
pub async fn deployments_embeddings(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    headers: axum::http::HeaderMap,
    Path(model): Path<String>,
    body: Bytes,
) -> Result<Response, ApiError> {
    let state = shared.load_full();
    let body = inject_model(body, &model)?;
    embeddings_inner_with_headers(state, &auth, body, &headers).await
}

/// `POST /openai/deployments/{model}/completions`
pub async fn deployments_completions(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    headers: axum::http::HeaderMap,
    Path(model): Path<String>,
    body: Bytes,
) -> Result<Response, ApiError> {
    let body = inject_model(body, &model)?;
    completions_handler(State(shared), Extension(auth), headers, body).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inject_model_adds_when_missing() {
        let body = Bytes::from(r#"{"messages":[]}"#);
        let out = inject_model(body, "my-dep").expect("ok");
        let v: Value = serde_json::from_slice(&out).expect("valid json");
        assert_eq!(v["model"], "my-dep");
    }

    #[test]
    fn inject_model_overrides_existing() {
        let body = Bytes::from(r#"{"model":"other","messages":[]}"#);
        let out = inject_model(body, "my-dep").expect("ok");
        let v: Value = serde_json::from_slice(&out).expect("valid json");
        assert_eq!(v["model"], "my-dep");
    }

    #[test]
    fn inject_model_rejects_invalid_json() {
        let body = Bytes::from("not json");
        let err = inject_model(body, "x").expect_err("should error");
        match err {
            ApiError {
                status: axum::http::StatusCode::BAD_REQUEST,
                ..
            } => {}
            other => panic!("expected BAD_REQUEST, got {other:?}"),
        }
    }

    #[test]
    fn inject_model_rejects_non_object_json() {
        let body = Bytes::from("[1,2,3]");
        let err = inject_model(body, "x").expect_err("non-object must error");
        assert_eq!(err.status, axum::http::StatusCode::BAD_REQUEST);

        let scalar = inject_model(Bytes::from("42"), "x").expect_err("scalar must error");
        assert_eq!(scalar.status, axum::http::StatusCode::BAD_REQUEST);
    }
}
