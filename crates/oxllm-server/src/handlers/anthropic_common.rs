//! Small helpers shared by the Anthropic-shaped handlers
//! ([`crate::handlers::messages`], [`crate::handlers::anthropic_count_tokens`],
//! [`crate::handlers::anthropic_messages_passthrough`]). Kept here rather than
//! duplicated in each handler so a future change (e.g. validating
//! `max_tokens`, rejecting non-string `model` fields) lands in one place.

use axum::body::Bytes;
use axum::http::StatusCode;
use axum::response::IntoResponse;

use crate::error::ApiError;

/// Parse the `model` field from an Anthropic-shaped request body.
///
/// Returns a [`ApiError::bad_request`] (400) when the body is not valid JSON,
/// exceeds the allowed nesting depth, or is missing the required `model`
/// field the router needs to select a deployment. Used by the count_tokens
/// and messages passthrough handlers — both forward the body verbatim but
/// still need a model up front for routing.
pub(crate) fn extract_request_model(body: &Bytes) -> Result<String, ApiError> {
    let value: serde_json::Value = serde_json::from_slice(body)
        .map_err(|e| ApiError::bad_request(format!("invalid JSON body: {e}")))?;
    crate::security::check_json_depth(&value, crate::security::MAX_JSON_DEPTH)?;
    crate::orchestration::request_controls::validate(&value)?;
    value
        .get("model")
        .and_then(|v| v.as_str())
        .filter(|model| !model.trim().is_empty())
        .map(str::to_string)
        .ok_or_else(|| ApiError::bad_request("missing required field: 'model'"))
}

/// Map an HTTP status code to the canonical Anthropic `error.type` value.
///
/// Anthropic SDK clients use `error.type` to drive retry behaviour (retry on
/// `rate_limit_error` / `overloaded_error`, surface `authentication_error` to
/// the user). The `/v1/messages` conversion handler surfaces upstream errors
/// through this mapping so an Anthropic SDK client sees the same retry
/// semantics it would against `api.anthropic.com` directly. Unknown status
/// codes fall back to the generic `api_error`.
pub(crate) fn status_to_anthropic_error_type(status: StatusCode) -> &'static str {
    match status {
        StatusCode::BAD_REQUEST => "invalid_request_error",
        StatusCode::UNAUTHORIZED => "authentication_error",
        StatusCode::FORBIDDEN => "permission_error",
        StatusCode::NOT_FOUND => "not_found_error",
        StatusCode::TOO_MANY_REQUESTS => "rate_limit_error",
        // 529 "Overloaded" is Anthropic-specific and has no named variant in
        // the `http` crate, so match on the raw code.
        s if s.as_u16() == 529 => "overloaded_error",
        _ => "api_error",
    }
}

/// Build an Anthropic-shaped error response (`{"type":"error","error":{…}}`)
/// from a local [`ApiError`]. Anthropic SDK clients on every Anthropic-native
/// route must see this envelope, not the OpenAI one, so local validation
/// failures match what the upstream API would have returned.
pub(crate) fn anthropic_error_response(
    status: StatusCode,
    error_type: &str,
    message: &str,
    preserved: &[(axum::http::HeaderName, axum::http::HeaderValue)],
) -> axum::response::Response {
    let body = serde_json::json!({
        "type": "error",
        "error": {
            "type": error_type,
            "message": message,
        }
    });
    let mut response = (status, axum::Json(body)).into_response();
    for (name, value) in preserved {
        response.headers_mut().insert(name.clone(), value.clone());
    }
    response
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_request_model_reads_model() {
        let body = Bytes::from(
            r#"{"model":"claude-3-5-sonnet","messages":[{"role":"user","content":"hi"}]}"#,
        );
        assert_eq!(extract_request_model(&body).unwrap(), "claude-3-5-sonnet");
    }

    #[test]
    fn extract_request_model_rejects_invalid_json() {
        let body = Bytes::from(r#"not json"#);
        let err = extract_request_model(&body).expect_err("invalid json");
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
    }

    #[test]
    fn extract_request_model_rejects_missing_model() {
        let body = Bytes::from(r#"{"messages":[]}"#);
        let err = extract_request_model(&body).expect_err("no model");
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
        assert!(err.message.contains("model"));
    }

    #[test]
    fn status_to_anthropic_error_type_maps_canonical_codes() {
        assert_eq!(
            status_to_anthropic_error_type(StatusCode::BAD_REQUEST),
            "invalid_request_error"
        );
        assert_eq!(
            status_to_anthropic_error_type(StatusCode::UNAUTHORIZED),
            "authentication_error"
        );
        assert_eq!(
            status_to_anthropic_error_type(StatusCode::FORBIDDEN),
            "permission_error"
        );
        assert_eq!(
            status_to_anthropic_error_type(StatusCode::NOT_FOUND),
            "not_found_error"
        );
        assert_eq!(
            status_to_anthropic_error_type(StatusCode::TOO_MANY_REQUESTS),
            "rate_limit_error"
        );
    }

    #[test]
    fn status_to_anthropic_error_type_maps_overloaded_529() {
        let status: StatusCode = StatusCode::from_u16(529).unwrap();
        assert_eq!(status_to_anthropic_error_type(status), "overloaded_error");
    }

    #[test]
    fn status_to_anthropic_error_type_defaults_to_api_error() {
        assert_eq!(
            status_to_anthropic_error_type(StatusCode::INTERNAL_SERVER_ERROR),
            "api_error"
        );
        assert_eq!(
            status_to_anthropic_error_type(StatusCode::BAD_GATEWAY),
            "api_error"
        );
        assert_eq!(
            status_to_anthropic_error_type(StatusCode::SERVICE_UNAVAILABLE),
            "api_error"
        );
    }
}
