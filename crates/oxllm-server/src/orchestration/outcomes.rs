//! Outcome classification for the chat retry/fallback loop.
//!
//! These types describe how the orchestrator should treat a single upstream
//! attempt — success, retryable failure, non-retryable client error, or
//! fallback-eligible error (context window / content policy).

use axum::http::HeaderMap;
use axum::response::Response;

use crate::error::ApiError;

/// Outcome of a model's retry loop.
pub(crate) enum ModelOutcome {
    /// A successful response was produced.
    Success(Response),
    /// A context-length-exceeded error was returned (eligible for context-window
    /// fallbacks).
    ContextWindow(ApiError),
    /// A content-policy / content-filter error was returned (eligible for
    /// content-policy fallbacks).
    ContentPolicy(ApiError),
    /// A retryable error (5xx, 429, timeout, network) exhausted all retries.
    Retryable(ApiError),
    /// A non-retryable client error (4xx, not context-length).
    Client(ApiError),
}

impl ModelOutcome {
    /// Collapse a non-success outcome into its [`ApiError`]. Success outcomes
    /// have no meaningful error and yield a synthetic internal error.
    pub(crate) fn into_api_error(self) -> ApiError {
        match self {
            ModelOutcome::ContextWindow(e)
            | ModelOutcome::ContentPolicy(e)
            | ModelOutcome::Retryable(e)
            | ModelOutcome::Client(e) => e,
            ModelOutcome::Success(_) => ApiError::internal("unexpected success outcome"),
        }
    }
}

/// Classifies an upstream HTTP error response.
///
/// The HTTP-sourced variants (`ContextWindow`, `ContentPolicy`, `Retryable`,
/// `Client`) carry the upstream response [`HeaderMap`] so that debugging
/// metadata (`x-request-id`) and rate-limit signals (`retry-after`,
/// `x-ratelimit-*`) can be forwarded to the client on the error path. The
/// `Timeout` and `Network` variants carry no headers — no HTTP response ever
/// arrived, so there is nothing to preserve.
#[derive(Debug)]
pub(crate) enum UpstreamError {
    /// 4xx with a context-length-exceeded code.
    ContextWindow {
        status: u16,
        message: String,
        headers: Option<HeaderMap>,
    },
    /// 4xx with a content-policy / content-filter code.
    ContentPolicy {
        status: u16,
        message: String,
        headers: Option<HeaderMap>,
    },
    /// 5xx or 429 (an HTTP response arrived, so the deployment is
    /// reachable — including a genuine upstream 504).
    Retryable {
        status: u16,
        message: String,
        headers: Option<HeaderMap>,
    },
    /// Locally-synthesized client-side timeout: no response ever arrived
    /// from the upstream. Deliberately distinct from `Retryable { 504 }`:
    /// a genuine upstream 504 proves reachability, a local timeout proves
    /// nothing, so health tracking must treat this as a connection
    /// failure, not a success.
    Timeout(String),
    /// Other 4xx.
    Client {
        status: u16,
        message: String,
        headers: Option<HeaderMap>,
    },
    /// Network error (DNS/TLS/connect).
    Network(String),
}

/// Parse the upstream `Retry-After` header (seconds or HTTP-date form)
/// into whole seconds to wait before the next attempt. Returns `None` when
/// the header is absent, malformed, or resolves to a non-positive delay.
///
/// Mirrors the reference retry-after header parsing
/// (`utils.py`): a numeric value is used verbatim; an HTTP-date is
/// converted to `date - now`. Callers treat a non-positive result as "no
/// signal" and fall back to exponential backoff (the reference gates on
/// `0 < retry_after <= 60` in `_calculate_retry_after`).
pub(crate) fn retry_after_secs(headers: Option<&HeaderMap>) -> Option<u64> {
    let raw = headers?.get("retry-after")?.to_str().ok()?.trim();
    if raw.is_empty() {
        return None;
    }
    if let Ok(secs) = raw.parse::<i64>() {
        return u64::try_from(secs).ok();
    }
    let target = jiff::fmt::rfc2822::parse(raw).ok()?.timestamp();
    let delta_ms = target
        .as_millisecond()
        .saturating_sub(jiff::Timestamp::now().as_millisecond())
        .max(0);
    Some(u64::try_from((delta_ms + 999) / 1000).unwrap_or(0))
}

/// Check if an HTTP status code is retryable.
///
/// Matches the reference retry classifier (`utils.py`): **408** (Request
/// Timeout), **409** (Conflict / lock timeout), **429** (Too Many Requests),
/// and **>= 500** (server errors) are retried. All other 4xx are terminal
/// client errors. Keeping this classifier in lock-step with the chat path's
/// `is_server_reflecting_status` (which already treats 408 as reflecting)
/// avoids the bug where a 408 inflates a deployment's failure ratio via the
/// percentage-cooldown path but never triggers the retry that would "spend"
/// that failure credit.
pub(crate) fn is_retryable_status(status: reqwest::StatusCode) -> bool {
    status == reqwest::StatusCode::REQUEST_TIMEOUT
        || status == reqwest::StatusCode::CONFLICT
        || status == reqwest::StatusCode::TOO_MANY_REQUESTS
        || status.is_server_error()
}

/// Whether a terminal 4xx status indicates a **deployment** fault rather
/// than a fault in the caller's request content: 401 (stale/invalid vendor
/// credential), 403 (credential not entitled for this model), 404 (model
/// decommissioned at the provider). These are exactly the statuses for
/// which the reference `async_function_with_fallbacks`
/// (`router.py:6435`) walks the configured `fallbacks` chain after
/// the primary fails — an operator's `fallbacks: [{primary: [secondary]}]`
/// exists to rescue precisely this scenario.
///
/// Request-content 4xx (400/413/422/…) are terminal by default. An explicit
/// zero allowed-fails entry opts that status into sibling/configured fallback.
/// Retry aliases never manufacture a cooldown policy.
pub(crate) fn is_deployment_fault_status(status: u16) -> bool {
    matches!(status, 401 | 403 | 404)
}

/// Generic request-content failures do not describe deployment health. Only
/// deployment faults or an operator's explicit per-status cooldown policy count;
/// global thresholds and content-policy retry categories cannot cool a typo.
pub(crate) fn client_failure_affects_deployment(
    state: &crate::state::AppState,
    status: u16,
) -> bool {
    is_deployment_fault_status(status)
        || state
            .router
            .config()
            .allowed_fails_policy
            .as_ref()
            .is_some_and(|policy| policy.contains_key(&status.to_string()))
}

/// Whether a terminal relay/stream failure with `status` may count toward
/// deployment health. Non-client statuses (network, 5xx, retryable client
/// codes) always may; terminal client statuses count only for deployment
/// faults or an explicit per-status policy. Context-window/content-policy
/// errors are classified before this point and keep their dedicated cooldown
/// path.
pub(crate) fn terminal_failure_affects_deployment(
    state: &crate::state::AppState,
    status: u16,
) -> bool {
    !reqwest::StatusCode::from_u16(status)
        .is_ok_and(|status| status.is_client_error() && !is_retryable_status(status))
        || client_failure_affects_deployment(state, status)
}

/// An explicit zero allowed-fails entry opts a client status into immediate
/// deployment failover. Retry counts and fallback eligibility are independent;
/// other request-content errors remain terminal.
pub(crate) fn explicitly_allows_client_fallback(
    state: &crate::state::AppState,
    status: u16,
) -> bool {
    state
        .router
        .config()
        .allowed_fails_policy
        .as_ref()
        .and_then(|policy| policy.get(&status.to_string()))
        == Some(&0)
}

/// Check if an error response body indicates a context-length-exceeded
/// error. Matches the OpenAI-shaped `"context_length"` / `"context_window"`
/// codes, Anthropic's `"prompt is too long"` message, and Gemini's
/// `"exceeds context"` / `"token limit"` messages. Also catches generic 400
/// errors whose message mentions context, token-limit, or too-long prompts.
pub(crate) fn is_context_window_error(body: &serde_json::Value) -> bool {
    let Some(error) = body.get("error") else {
        return false;
    };
    let code = error.get("code").and_then(|c| c.as_str()).unwrap_or("");
    let typ = error.get("type").and_then(|t| t.as_str()).unwrap_or("");
    let message = error.get("message").and_then(|m| m.as_str()).unwrap_or("");
    let code_lower = code.to_lowercase();
    let type_lower = typ.to_lowercase();
    let msg_lower = message.to_lowercase();

    if code_lower.contains("context_length") || code_lower.contains("context_window") {
        return true;
    }
    if type_lower.contains("context_length") {
        return true;
    }
    if msg_lower.contains("prompt is too long") {
        return true;
    }
    if msg_lower.contains("exceeds context") || msg_lower.contains("token limit") {
        return true;
    }
    if msg_lower.contains("context window") || msg_lower.contains("maximum context") {
        return true;
    }
    if msg_lower.contains("too long") && type_lower.contains("invalid_request") {
        return true;
    }
    false
}

/// Check if an error response body indicates a content-policy / content-filter
/// rejection. Matches OpenAI's `"content_policy"` /
/// `"content_filter"` codes and messages, Anthropic's content-policy messages
/// (often `invalid_request_error` with a `content` / `content policy` /
/// `content filtering` mention), and Azure's `ResponsibleAIPolicyViolation`.
pub(crate) fn is_content_policy_error(body: &serde_json::Value) -> bool {
    let Some(error) = body.get("error") else {
        return false;
    };
    let code = error.get("code").and_then(|c| c.as_str()).unwrap_or("");
    let typ = error.get("type").and_then(|t| t.as_str()).unwrap_or("");
    let message = error.get("message").and_then(|m| m.as_str()).unwrap_or("");
    let code_lower = code.to_lowercase();
    let type_lower = typ.to_lowercase();
    let msg_lower = message.to_lowercase();

    if code_lower == "cyber_policy"
        || code_lower.contains("content_policy")
        || code_lower.contains("content_filter")
    {
        return true;
    }
    if type_lower == "cyber_policy"
        || type_lower.contains("content_policy")
        || type_lower.contains("content_filter")
    {
        return true;
    }
    if msg_lower.contains("content policy") || msg_lower.contains("content filtering") {
        return true;
    }
    if msg_lower.contains("content filter") {
        return true;
    }
    if code_lower.contains("responsibleai") || msg_lower.contains("responsible ai") {
        return true;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderMap;
    use serde_json::json;

    #[test]
    fn exact_policy_code_is_specialized_without_promoting_generic_invalid_requests() {
        assert!(is_content_policy_error(
            &json!({"error":{"code":"cyber_policy","message":"request declined"}})
        ));
        assert!(!is_content_policy_error(
            &json!({"error":{"code":"invalid_prompt","message":"request declined"}})
        ));
        assert!(!is_content_policy_error(
            &json!({"error":{"code":"not_cyber_policy","message":"request declined"}})
        ));
    }

    fn headers_with(retry_after: &str) -> Option<HeaderMap> {
        let mut h = HeaderMap::new();
        h.insert(
            "retry-after",
            axum::http::HeaderValue::from_str(retry_after).expect("valid header value"),
        );
        Some(h)
    }

    #[test]
    fn retry_after_secs_parses_numeric_form() {
        assert_eq!(retry_after_secs(headers_with("5").as_ref()), Some(5));
        assert_eq!(retry_after_secs(headers_with("0").as_ref()), Some(0));
        assert_eq!(retry_after_secs(headers_with("30").as_ref()), Some(30));
    }

    #[test]
    fn retry_after_secs_parses_http_date_form() {
        let future = jiff::Timestamp::now()
            .checked_add(jiff::SignedDuration::from_secs(90))
            .expect("future ts");
        let http_date = jiff::fmt::rfc2822::to_string(&future.to_zoned(jiff::tz::TimeZone::UTC))
            .expect("rfc2822 string");
        let parsed =
            retry_after_secs(headers_with(&http_date).as_ref()).expect("http-date must parse");
        assert!(
            (85..=90).contains(&parsed),
            "future date ~90s out must parse to ~90s, got {parsed}"
        );
    }

    #[test]
    fn retry_after_secs_http_date_in_past_is_zero() {
        let past = jiff::Timestamp::now()
            .checked_sub(jiff::SignedDuration::from_secs(600))
            .expect("past ts");
        let http_date = jiff::fmt::rfc2822::to_string(&past.to_zoned(jiff::tz::TimeZone::UTC))
            .expect("rfc2822 string");
        assert_eq!(retry_after_secs(headers_with(&http_date).as_ref()), Some(0));
    }

    #[test]
    fn retry_after_secs_garbage_and_missing_return_none() {
        assert_eq!(retry_after_secs(headers_with("soon").as_ref()), None);
        assert_eq!(retry_after_secs(headers_with("").as_ref()), None);
        assert_eq!(retry_after_secs(None), None);
        let mut other = HeaderMap::new();
        other.insert("x-request-id", axum::http::HeaderValue::from_static("r"));
        assert_eq!(retry_after_secs(Some(&other)), None);
    }

    #[test]
    fn is_retryable_status_retries_429_and_5xx() {
        assert!(is_retryable_status(reqwest::StatusCode::TOO_MANY_REQUESTS));
        assert!(is_retryable_status(
            reqwest::StatusCode::INTERNAL_SERVER_ERROR
        ));
        assert!(is_retryable_status(reqwest::StatusCode::BAD_GATEWAY));
        assert!(is_retryable_status(
            reqwest::StatusCode::SERVICE_UNAVAILABLE
        ));
        assert!(is_retryable_status(reqwest::StatusCode::GATEWAY_TIMEOUT));
    }

    #[test]
    fn is_retryable_status_retries_408_and_409() {
        assert!(
            is_retryable_status(reqwest::StatusCode::REQUEST_TIMEOUT),
            "408 Request Timeout must be retryable (LiteLLM _should_retry)"
        );
        assert!(
            is_retryable_status(reqwest::StatusCode::CONFLICT),
            "409 Conflict must be retryable (LiteLLM _should_retry)"
        );
    }

    #[test]
    fn is_retryable_status_rejects_other_4xx() {
        assert!(!is_retryable_status(reqwest::StatusCode::BAD_REQUEST));
        assert!(!is_retryable_status(reqwest::StatusCode::UNAUTHORIZED));
        assert!(!is_retryable_status(reqwest::StatusCode::FORBIDDEN));
        assert!(!is_retryable_status(reqwest::StatusCode::NOT_FOUND));
        assert!(!is_retryable_status(reqwest::StatusCode::PAYMENT_REQUIRED));
        assert!(!is_retryable_status(reqwest::StatusCode::GONE));
    }

    #[test]
    fn is_context_window_error_openai_code() {
        let body = json!({
        "error": {"code": "context_length_exceeded", "message": "too long"}
        });
        assert!(is_context_window_error(&body));
    }

    #[test]
    fn is_context_window_error_openai_context_window_code() {
        let body = json!({
        "error": {"code": "context_window_reached", "message": "too long"}
        });
        assert!(is_context_window_error(&body));
    }

    #[test]
    fn is_context_window_error_anthropic_prompt_too_long() {
        let body = json!({
        "error": {
        "type": "invalid_request_error",
        "message": "prompt is too long: 300000 tokens > 200000 maximum"
        }
        });
        assert!(is_context_window_error(&body));
    }

    #[test]
    fn is_context_window_error_gemini_exceeds_context() {
        let body = json!({
        "error": {
        "message": "Request exceeds context limit: 1000000 tokens",
        "status": "INVALID_ARGUMENT"
        }
        });
        assert!(is_context_window_error(&body));
    }

    #[test]
    fn is_context_window_error_gemini_token_limit() {
        let body = json!({
        "error": {
        "message": "The request exceeds the token limit for this model."
        }
        });
        assert!(is_context_window_error(&body));
    }

    #[test]
    fn is_context_window_error_generic_context_message() {
        let body = json!({
        "error": {
        "type": "invalid_request_error",
        "message": "This model's context window is 8192 tokens."
        }
        });
        assert!(is_context_window_error(&body));
    }

    #[test]
    fn is_context_window_error_generic_too_long_with_invalid_request() {
        let body = json!({
        "error": {
        "type": "invalid_request_error",
        "message": "Your input is too long for this model."
        }
        });
        assert!(is_context_window_error(&body));
    }

    #[test]
    fn is_context_window_error_generic_maximum_context() {
        let body = json!({
        "error": {"message": "Request exceeds the maximum context length."}
        });
        assert!(is_context_window_error(&body));
    }

    #[test]
    fn is_context_window_error_unrelated_400_is_not_context_window() {
        let body = json!({
        "error": {
        "type": "invalid_request_error",
        "message": "Invalid model name."
        }
        });
        assert!(!is_context_window_error(&body));
    }

    #[test]
    fn is_context_window_error_no_error_field() {
        assert!(!is_context_window_error(&json!({"foo": "bar"})));
    }

    #[test]
    fn is_content_policy_error_openai_code() {
        let body = json!({
        "error": {"code": "content_policy_violation", "message": "blocked"}
        });
        assert!(is_content_policy_error(&body));
    }

    #[test]
    fn is_content_policy_error_azure_content_filter() {
        let body = json!({
        "error": {
        "code": "content_filter",
        "message": "Output blocked by content filtering policy"
        }
        });
        assert!(is_content_policy_error(&body));
    }

    #[test]
    fn is_content_policy_error_anthropic_message() {
        let body = json!({
        "error": {
        "type": "invalid_request_error",
        "message": "Output blocked by content filtering policy"
        }
        });
        assert!(is_content_policy_error(&body));
    }

    #[test]
    fn is_content_policy_error_azure_responsibleai() {
        let body = json!({
        "error": {
        "code": "ResponsibleAIPolicyViolation",
        "message": "flagged"
        }
        });
        assert!(is_content_policy_error(&body));
    }

    #[test]
    fn is_content_policy_error_unrelated_400_is_not_content_policy() {
        let body = json!({
        "error": {
        "type": "invalid_request_error",
        "message": "Invalid model name."
        }
        });
        assert!(!is_content_policy_error(&body));
    }

    #[test]
    fn is_content_policy_error_context_window_is_not_content_policy() {
        let body = json!({
        "error": {"code": "context_length_exceeded", "message": "too long"}
        });
        assert!(!is_content_policy_error(&body));
    }

    #[test]
    fn is_content_policy_error_no_error_field() {
        assert!(!is_content_policy_error(&json!({"foo": "bar"})));
    }
}
