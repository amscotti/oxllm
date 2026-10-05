//! Error type for the admin server.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use minijinja::Error as TemplateError;

/// Admin-server error. Converts into an HTTP response for handlers.
#[derive(Debug, thiserror::Error)]
pub enum AdminError {
    #[error("upstream proxy error: {0}")]
    Proxy(String),
    #[error("template render error: {0}")]
    Template(#[from] TemplateError),
    #[error("internal error: {0}")]
    Internal(String),
}

impl AdminError {
    pub fn proxy(msg: impl Into<String>) -> Self {
        AdminError::Proxy(msg.into())
    }

    pub fn internal(msg: impl Into<String>) -> Self {
        AdminError::Internal(msg.into())
    }
}

impl IntoResponse for AdminError {
    fn into_response(self) -> Response {
        let status = match &self {
            AdminError::Proxy(_) => StatusCode::BAD_GATEWAY,
            AdminError::Template(_) | AdminError::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        };
        tracing::warn!(error = %self, "admin request failed");
        (status, self.to_string()).into_response()
    }
}
