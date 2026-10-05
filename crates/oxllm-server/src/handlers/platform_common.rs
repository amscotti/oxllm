//! Small helpers shared by the platform-API handlers
//! ([`crate::handlers::batches`], [`crate::handlers::files`],
//! [`crate::handlers::fine_tuning`], [`crate::handlers::assistants`],
//! [`crate::handlers::threads`], [`crate::handlers::anthropic_batches`],
//! [`crate::handlers::anthropic_files`]). Kept here rather than duplicated in
//! each handler so a future tightening (e.g. a length cap, additional rejected
//! characters) lands in one place.

use axum::http::HeaderMap;

use crate::error::ApiError;

/// Strict allowlist for client-controlled `{id}` path segments. Rejects `/`,
/// `.`, and any decoded traversal sequence before it can be formatted into the
/// upstream path (Axum's `Path<String>` decodes `%2F` to `/`, so without this
/// an authenticated client could escape the deployment's base path via the
/// pooled credential). The pattern is `^[A-Za-z0-9_-]+$`.
pub(crate) fn validate_path_param(value: &str) -> Result<(), ApiError> {
    if value
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
        && !value.is_empty()
    {
        Ok(())
    } else {
        Err(ApiError::bad_request(format!(
            "invalid path parameter: must match ^[A-Za-z0-9_-]+$ (got {value})"
        )))
    }
}

/// Read the inbound `content-type` header, returning `None` when it is absent
/// or empty. Used by platform handlers that forward the header verbatim (e.g.
/// multipart uploads whose boundary the upstream needs to parse the body).
pub(crate) fn content_type_for(headers: &HeaderMap) -> Option<&str> {
    headers
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .filter(|s| !s.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_path_param_accepts_alphanumerics() {
        assert!(validate_path_param("batch_abc-123").is_ok());
        assert!(validate_path_param("file-1").is_ok());
        assert!(validate_path_param("asst_X").is_ok());
    }

    #[test]
    fn validate_path_param_rejects_traversal_attempts() {
        assert!(validate_path_param("../v1/models").is_err());
        assert!(validate_path_param("..").is_err());
        assert!(validate_path_param("a/b").is_err());
        assert!(validate_path_param("a\\b").is_err());
        assert!(validate_path_param("%2e%2e").is_err());
        assert!(validate_path_param("").is_err());
    }

    #[test]
    fn content_type_for_reads_header() {
        let mut headers = HeaderMap::new();
        headers.insert("content-type", "application/json".parse().unwrap());
        assert_eq!(content_type_for(&headers), Some("application/json"));
    }

    #[test]
    fn content_type_for_returns_none_when_absent() {
        let headers = HeaderMap::new();
        assert_eq!(content_type_for(&headers), None);
    }

    #[test]
    fn content_type_for_skips_empty_value() {
        let mut headers = HeaderMap::new();
        headers.insert("content-type", "".parse().unwrap());
        assert_eq!(content_type_for(&headers), None);
    }
}
