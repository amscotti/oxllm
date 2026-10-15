//! OpenAI-compatible error responses.

use axum::Json;
use axum::http::{HeaderMap, HeaderName, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use serde_json::json;

/// Header advising the client how long (in seconds) to wait before retrying.
const RETRY_AFTER_HEADER: HeaderName = HeaderName::from_static("retry-after");

/// Default `Retry-After` value (seconds) for 429 responses without one.
const DEFAULT_RETRY_AFTER_SECS: u64 = 60;

/// Error `code` for spend-budget rejections. Budget windows reset on much
/// longer horizons (daily/monthly via `budget_duration`) than the per-minute
/// RPM window the 60s default targets, so a budget-capped 429 must NOT
/// inherit the blanket `Retry-After`: clients would hot-retry every 60s and
/// be re-rejected. Omitting the header lets clients back off per their own
/// policy.
const BUDGET_EXCEEDED_CODE: &str = "budget_exceeded";

/// Error `code` marking a 502 synthesized by the *gateway* for a transport
/// failure (connection refused, DNS, TLS) — the request never reached the
/// provider. Distinct from a provider-served 502 (`upstream_error`), matching
/// the reference `APIConnectionError` vs `APIStatusError` split. Inverse mappers
/// (`crate::pipeline::core_error_from_api`) keys off this code to
/// re-classify the failure as [`oxllm_core::Error::Network`] so failure hooks
/// (e.g. the rate-limit RPM/TPM release) treat it as "never reached the
/// provider".
pub(crate) const UPSTREAM_UNREACHABLE_CODE: &str = "upstream_unreachable";

/// Error `code` marking a 504 synthesized by the *gateway* for a locally
/// expired request timeout — no response ever arrived. Same inverse-mapping
/// contract as [`UPSTREAM_UNREACHABLE_CODE`].
pub(crate) const UPSTREAM_TIMEOUT_CODE: &str = "upstream_timeout";

/// Upper bound (seconds) applied to a *numeric* forwarded upstream
/// `retry-after`. A misbehaving or compromised upstream could otherwise push
/// every client into an arbitrarily long backoff (`retry-after: 99999999`) —
/// a cheap client-side DoS amplifier. HTTP-date values are forwarded
/// verbatim (parity with the reference, which forwards `exc.headers` as-is).
const MAX_FORWARDED_RETRY_AFTER_SECS: u64 = 3600;

/// Maximum length of a client-supplied model name echoed into an error
/// message. The request-body limit (10 MB) otherwise bounds the echo, so a
/// pathological `model` field could produce a multi-megabyte error response.
const MAX_MODEL_ECHO_CHARS: usize = 256;

/// Bound a client-supplied model name before echoing it into an error
/// message so a pathological `model` field cannot amplify the error body.
pub(crate) fn truncate_model_for_echo(model: &str) -> &str {
    match model.char_indices().nth(MAX_MODEL_ECHO_CHARS) {
        Some((idx, _)) => &model[..idx],
        None => model,
    }
}

/// Upstream response headers that may be forwarded to the client on the error
/// path. This allowlist is the security boundary: arbitrary upstream headers
/// never reach the client (mirroring the `strip_leaking_headers` posture on
/// the success path). `x-request-id` is deliberately included here because its
/// debugging value on errors (correlating with provider support tickets)
/// outweighs the leak risk, and only this curated set is ever exposed.
const FORWARD_ON_ERROR: &[&str] = &[
    "x-request-id",
    "retry-after",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-remaining-tokens",
    "x-ratelimit-reset-requests",
    "x-ratelimit-reset-tokens",
    "x-ratelimit-limit-requests",
    "x-ratelimit-limit-tokens",
];

/// An error that renders as an OpenAI-shaped error envelope.
///
/// `#[non_exhaustive]` forces construction through the [`ApiError::bad_request`]
/// / [`ApiError::upstream`] / ... constructors for *downstream* crates, which
/// enforce the invariants a struct literal would otherwise let callers bypass
/// (correct `error_type` for the status, correct `code` derivation,
/// `headers: None` default). Within this crate the fields are visible, so
/// construction by literal is *conventionally* forbidden — every in-crate
/// site must use a constructor. `#[non_exhaustive]` also makes adding a field
/// later non-breaking for any such literal — there is none today, and this
/// keeps it that way.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ApiError {
    /// HTTP status code.
    pub status: StatusCode,
    /// Human-readable error message.
    pub message: String,
    /// Error type slug (e.g. `"invalid_request_error"`).
    pub error_type: String,
    /// Optional machine-readable code (e.g. `"invalid_api_key"`).
    pub code: Option<String>,
    /// Upstream response headers preserved from a provider error, if any.
    /// An allowlist is forwarded in [`ApiError::into_response`]; `None` for
    /// proxy-synthesized errors with no upstream response. Boxed to keep
    /// `ApiError` small — a [`HeaderMap`] is ~150 bytes and would otherwise
    /// trip `clippy::result_large_err` on the many `Result<_, ApiError>`
    /// return sites.
    pub headers: Option<Box<HeaderMap>>,
}

impl ApiError {
    /// 400 Bad Request.
    pub fn bad_request(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: msg.into(),
            error_type: "invalid_request_error".to_string(),
            code: None,
            headers: None,
        }
    }

    /// 401 Unauthorized (invalid API key).
    pub fn unauthorized() -> Self {
        Self {
            status: StatusCode::UNAUTHORIZED,
            message: "invalid api key".to_string(),
            error_type: "invalid_request_error".to_string(),
            code: Some("invalid_api_key".to_string()),
            headers: None,
        }
    }

    /// 403 Forbidden.
    pub fn forbidden(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::FORBIDDEN,
            message: msg.into(),
            error_type: "permission_error".to_string(),
            code: None,
            headers: None,
        }
    }

    /// 404 Not Found (generic).
    pub fn not_found(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            message: msg.into(),
            error_type: "invalid_request_error".to_string(),
            code: None,
            headers: None,
        }
    }

    /// 404 Not Found (model not configured).
    pub fn model_not_found(model: &str) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            message: format!(
                "The model '{}' does not exist",
                truncate_model_for_echo(model)
            ),
            error_type: "invalid_request_error".to_string(),
            code: Some("model_not_found".to_string()),
            headers: None,
        }
    }

    /// Render a terminal, locally unresolved inference model as an invalid
    /// model parameter. Invoke at the HTTP boundary, after fallback handling;
    /// resource lookups and genuine provider 404s retain their own semantics.
    pub(crate) fn for_inference(mut self) -> Self {
        if self.status == StatusCode::NOT_FOUND && self.code.as_deref() == Some("model_not_found") {
            self.status = StatusCode::BAD_REQUEST;
        }
        self
    }

    /// 429 Too Many Requests.
    pub fn rate_limited(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::TOO_MANY_REQUESTS,
            message: msg.into(),
            error_type: "rate_limit_error".to_string(),
            code: Some("proxy_rate_limited".to_string()),
            headers: None,
        }
    }

    /// 429 Too Many Requests (spend budget exhausted).
    ///
    /// Same status as [`ApiError::rate_limited`] but a distinct
    /// `budget_exceeded` code so clients can tell a budget-cap rejection
    /// apart from request-rate limiting (`proxy_rate_limited`) and provider
    /// 429s (`upstream_rate_limited`).
    pub fn budget_exceeded(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::TOO_MANY_REQUESTS,
            message: msg.into(),
            error_type: "rate_limit_error".to_string(),
            code: Some("budget_exceeded".to_string()),
            headers: None,
        }
    }

    /// Upstream error (status passed through from the provider response).
    ///
    /// The `code` disambiguates the error source so clients can distinguish a
    /// proxy-enforced limit (`proxy_rate_limited`) from a provider 429
    /// (`upstream_rate_limited`), a stale upstream credential
    /// (`upstream_authentication_error`), or any other provider 5xx
    /// (`upstream_error`).
    ///
    /// Only the standard 4xx–5xx error range is ever sent to the client.
    /// Informational (1xx), success (2xx), redirect (3xx), and non-existent
    /// (≥600 or sub-100) statuses are synthesized to 502 Bad Gateway so the
    /// client never sees a non-error status paired with an error body (a
    /// buggy/malicious upstream returning e.g. 200 is reported as a gateway
    /// error rather than a nonsensical "200 OK error"). The `code` and
    /// `error_type` are derived from that *synthesized* status.
    pub fn upstream(status: u16, message: impl Into<String>) -> Self {
        let status_code = synthesize_upstream_status(status);
        Self {
            status: status_code,
            message: message.into(),
            error_type: error_type_for_status(status_code.as_u16()).to_string(),
            code: code_for_upstream_status(status_code.as_u16()).map(str::to_string),
            headers: None,
        }
    }

    /// 502 Bad Gateway synthesized by the gateway for a transport failure
    /// (connection refused, DNS, TLS): the request never reached the
    /// provider. Carries the `UPSTREAM_UNREACHABLE_CODE` marker so inverse
    /// mapping (`crate::pipeline::core_error_from_api`) can recover
    /// [`oxllm_core::Error::Network`] and failure hooks release RPM/TPM
    /// reservations that a provider-served 502 would keep. Same client shape
    /// as a genuine provider 502 except the more precise `code`.
    pub fn network_unreachable() -> Self {
        Self {
            status: StatusCode::BAD_GATEWAY,
            message: "upstream error".to_string(),
            error_type: "api_error".to_string(),
            code: Some(UPSTREAM_UNREACHABLE_CODE.to_string()),
            headers: None,
        }
    }

    /// 504 Gateway Timeout synthesized by the gateway for a locally expired
    /// request timeout: no response ever arrived from the provider. Carries
    /// the `UPSTREAM_TIMEOUT_CODE` marker with the same inverse-mapping
    /// contract as [`ApiError::network_unreachable`].
    pub fn gateway_timeout() -> Self {
        Self {
            status: StatusCode::GATEWAY_TIMEOUT,
            message: "upstream request timeout".to_string(),
            error_type: "api_error".to_string(),
            code: Some(UPSTREAM_TIMEOUT_CODE.to_string()),
            headers: None,
        }
    }

    /// 500 Internal Server Error.
    pub fn internal(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: msg.into(),
            error_type: "api_error".to_string(),
            code: None,
            headers: None,
        }
    }

    /// Attach preserved upstream response headers to this error.
    ///
    /// [`ApiError::into_response`] forwards an allowlist from `headers` onto
    /// the client-facing response; an upstream `retry-after`, when present,
    /// overrides the hardcoded 60s default for 429s. The headers are boxed
    /// internally to keep `ApiError` small.
    pub fn with_headers(mut self, headers: Option<HeaderMap>) -> Self {
        self.headers = headers.map(Box::new);
        self
    }
}

/// Map an upstream HTTP status to the OpenAI `error.type` slug. OpenAI (and
/// the reference `openai_exception_handler`) disambiguates errors by `type`:
/// `invalid_request_error` for 400s, `permission_error` for 403s,
/// `rate_limit_error` for 429s, and `api_error` for 5xx. Clients that switch
/// on `error.type` (rather than the status) would otherwise misclassify a
/// provider 403 as a plain invalid request and a provider 4xx as a 5xx.
pub(crate) fn error_type_for_status(status: u16) -> &'static str {
    match status {
        403 => "permission_error",
        429 => "rate_limit_error",
        400..=499 => "invalid_request_error",
        500..=599 => "api_error",
        _ => "api_error",
    }
}

/// Synthesize an upstream HTTP status to the [`StatusCode`] actually sent to
/// the client on the error path. Only the standard 4xx–5xx error range is
/// passed through; informational (1xx), success (2xx), redirect (3xx), and
/// non-existent (≥600 or sub-100) statuses are synthesized to 502 Bad
/// Gateway so the client never sees a non-error status paired with an error
/// body. A buggy/malicious upstream returning such a status is reported as a
/// gateway error rather than a nonsensical "200 OK error".
///
/// Shared by [`ApiError::upstream`] (non-streaming) and the streaming SSE
/// error event so both paths apply the identical synthesis.
pub(crate) fn synthesize_upstream_status(status: u16) -> StatusCode {
    if (400..600).contains(&status) {
        StatusCode::from_u16(status).unwrap_or(StatusCode::BAD_GATEWAY)
    } else {
        StatusCode::BAD_GATEWAY
    }
}

/// Derive the disambiguating OpenAI-style `code` string for an upstream HTTP
/// status, or `None` when no code applies. The input must be the *synthesized*
/// status (see [`synthesize_upstream_status`]): an out-of-range input that
/// became 502 reports `upstream_error`, not the raw value.
///
/// This is the single source of truth for the upstream error `code` field,
/// used by both the non-streaming [`ApiError::upstream`] constructor and the
/// streaming SSE error event. Routing both through it guarantees a client
/// switching on `code` sees the same value for the same upstream condition
/// regardless of `stream: true` (previously the streaming path emitted
/// OpenAI-native codes like `rate_limit_exceeded` while the non-streaming
/// path emitted `upstream_rate_limited`).
pub(crate) fn code_for_upstream_status(status: u16) -> Option<&'static str> {
    match status {
        401 => Some("upstream_authentication_error"),
        403 => Some("upstream_permission_error"),
        429 => Some("upstream_rate_limited"),
        500..=599 => Some("upstream_error"),
        _ => None,
    }
}

/// Build the OpenAI-shaped error object (`{message, type, param, code}`)
/// shared by the non-streaming [`ApiError::into_response`] body and the
/// streaming SSE error event.
///
/// OpenAI's error object always carries (nullable) `param` and `code` fields;
/// SDK/agent code that reads them gets `null` instead of `undefined` when we
/// have nothing to report, so both are rendered explicitly as `null` and
/// `code` is overwritten only when a value is present. Routing both error
/// paths through this helper guarantees a client sees a byte-identical
/// envelope for the same inputs regardless of `stream: true`, so the two
/// shapes cannot silently diverge.
pub(crate) fn error_envelope_object(
    message: &str,
    error_type: &str,
    code: Option<&str>,
) -> serde_json::Value {
    let mut error = json!({
        "message": message,
        "type": error_type,
        "param": null,
        "code": null,
    });
    if let Some(code) = code {
        error["code"] = json!(code);
    }
    error
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let error = error_envelope_object(&self.message, &self.error_type, self.code.as_deref());
        let body = Json(json!({ "error": error }));
        let mut response = (self.status, body).into_response();

        // Forward the curated allowlist of upstream headers (debugging +
        // rate-limit metadata) when this error originated from a provider
        // response. Only these names are ever exposed — arbitrary upstream
        // headers stay private, preserving the strip_leaking_headers posture.
        // A numeric `retry-after` is clamped to
        // MAX_FORWARDED_RETRY_AFTER_SECS so a misbehaving upstream cannot
        // push clients into an arbitrarily long backoff; HTTP-date values
        // are forwarded verbatim.
        if let Some(upstream_headers) = &self.headers {
            for &name in FORWARD_ON_ERROR {
                if let Some(value) = upstream_headers.get(name) {
                    if name == RETRY_AFTER_HEADER.as_str() {
                        if let Some(trimmed) = value.to_str().ok().map(str::trim) {
                            let forwarded = match trimmed.parse::<u64>() {
                                Ok(secs) if secs > MAX_FORWARDED_RETRY_AFTER_SECS => {
                                    MAX_FORWARDED_RETRY_AFTER_SECS.to_string()
                                }
                                _ => trimmed.to_string(),
                            };
                            if let Ok(value) = HeaderValue::from_str(&forwarded) {
                                response.headers_mut().insert(RETRY_AFTER_HEADER, value);
                            }
                        }
                        continue;
                    }
                    if let Ok(hname) = HeaderName::from_bytes(name.as_bytes()) {
                        response.headers_mut().insert(hname, value.clone());
                    }
                }
            }
        }

        // `retry-after`: a forwarded upstream value (now in the response if
        // present) takes precedence; otherwise fall back to the hardcoded
        // default for 429s. This lets a vendor's backoff signal reach the
        // client instead of being clobbered by our 60s. Budget-exceeded 429s
        // are excluded: their reset window is far longer than 60s, so the
        // default would mislead clients into hot-retrying.
        if self.status == StatusCode::TOO_MANY_REQUESTS
            && self.code.as_deref() != Some(BUDGET_EXCEEDED_CODE)
            && response.headers().get(RETRY_AFTER_HEADER).is_none()
        {
            if let Ok(value) = HeaderValue::from_str(&DEFAULT_RETRY_AFTER_SECS.to_string()) {
                response.headers_mut().insert(RETRY_AFTER_HEADER, value);
            }
        }
        response
    }
}

impl From<oxllm_core::Error> for ApiError {
    fn from(e: oxllm_core::Error) -> Self {
        match &e {
            oxllm_core::Error::ModelNotFound(m) => ApiError::model_not_found(m),
            oxllm_core::Error::Unauthorized => ApiError::unauthorized(),
            oxllm_core::Error::BadRequest(msg) => ApiError::bad_request(msg.clone()),
            oxllm_core::Error::RateLimited(msg) => ApiError::rate_limited(msg),
            oxllm_core::Error::BudgetExceeded(msg) => ApiError::budget_exceeded(msg),
            // Header-less by design: the core `Error` type carries no HTTP
            // header payload, so this conversion cannot preserve an upstream
            // `retry-after` / `x-request-id` (a real upstream 429 through here
            // would fall back to the hardcoded 60s default). Today this arm
            // only carries provider-*synthesized* errors (no live HTTP
            // response to preserve); every live HTTP-error path routes
            // through the `UpstreamError` taxonomy and calls
            // `ApiError::upstream(..).with_headers(..)` directly, preserving
            // headers. A future change that funnels a real upstream response
            // through `Error::Upstream` must either extend the variant with an
            // optional `HeaderMap` or use `ApiError::upstream(..).with_headers(..)`
            // at the call site to avoid silently regressing `retry-after`.
            oxllm_core::Error::Upstream { status, message } => ApiError::upstream(*status, message),
            // A failure to reach the upstream (connection refused, DNS, TLS)
            // is an upstream-side problem: render 502 Bad Gateway — matching
            // genuine upstream 502s — with a generic message so no internal
            // topology reaches the client. The `upstream_unreachable` code
            // preserves the transport/provider distinction across the
            // boundary so inverse mappings recover `Error::Network`. Detail
            // is logged server-side.
            oxllm_core::Error::Network(msg) => {
                tracing::error!(
                    error.kind = "network",
                    detail = %msg,
                    "network error reaching upstream"
                );
                ApiError::network_unreachable()
            }
            // Serialization failures are ambiguous between client- and
            // provider-side parse errors; render a generic 500 rather than
            // 400, which would mis-blame the client for a malformed provider
            // response. Detail is logged server-side.
            oxllm_core::Error::Serialization(msg) => {
                tracing::error!(
                    error.kind = "serialization",
                    detail = %msg,
                    "serialization error"
                );
                ApiError::internal("internal error")
            }
            // Operational internals (credential-resolver messages, SigV4
            // signing failures, transform internals) must never reach the
            // client body: redact to a generic message and log server-side.
            oxllm_core::Error::Config(msg) => {
                tracing::error!(error.kind = "config", detail = %msg, "config error");
                ApiError::internal("internal error")
            }
            oxllm_core::Error::Transform(msg) => {
                tracing::error!(error.kind = "transform", detail = %msg, "transform error");
                ApiError::internal("internal error")
            }
            // A pre-call-check rejection is operator-configured and often
            // client-caused (budget cap, prompt-cache pin): surface it as a
            // 400 that preserves the operator-defined reason rather than a
            // generic 500.
            oxllm_core::Error::PreCallCheckFailed(msg) => ApiError::bad_request(msg.clone()),
            oxllm_core::Error::Internal => ApiError::internal("internal error"),
            // `oxllm_core::Error` is `#[non_exhaustive]`; new variants added
            // upstream must map to a generic internal error here rather than
            // breaking the conversion. Concrete arms above preserve specific
            // status codes for known error kinds.
            _ => ApiError::internal("internal error"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bad_request_renders_correct_shape() {
        let err = ApiError::bad_request("missing field");
        let resp = err.into_response();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[test]
    fn unauthorized_has_correct_code() {
        let err = ApiError::unauthorized();
        assert_eq!(err.status, StatusCode::UNAUTHORIZED);
        assert_eq!(err.code.as_deref(), Some("invalid_api_key"));
        assert_eq!(err.error_type, "invalid_request_error");
        assert_eq!(err.message, "invalid api key");
    }

    #[test]
    fn model_not_found_has_correct_code() {
        let err = ApiError::model_not_found("gpt-99");
        assert_eq!(err.status, StatusCode::NOT_FOUND);
        assert_eq!(err.code.as_deref(), Some("model_not_found"));
        assert!(err.message.contains("gpt-99"));
    }

    #[test]
    fn inference_model_errors_preserve_the_local_marker_and_provider_resource_statuses() {
        let error = ApiError::model_not_found("missing").for_inference();
        assert_eq!(error.status, StatusCode::BAD_REQUEST);
        assert_eq!(error.code.as_deref(), Some("model_not_found"));
        assert!(matches!(
            crate::pipeline::core_error_from_api(&error),
            oxllm_core::Error::ModelNotFound(_)
        ));
        assert_eq!(
            ApiError::upstream(404, "provider model missing")
                .for_inference()
                .status,
            StatusCode::NOT_FOUND
        );
        assert_eq!(
            ApiError::not_found("resource missing")
                .for_inference()
                .status,
            StatusCode::NOT_FOUND
        );
        assert_eq!(
            ApiError::gateway_timeout().for_inference().status,
            StatusCode::GATEWAY_TIMEOUT
        );
    }

    #[test]
    fn upstream_status_mapped() {
        let err = ApiError::upstream(429, "rate limited by provider");
        assert_eq!(err.status, StatusCode::TOO_MANY_REQUESTS);
        // OpenAI's wire format: a provider 429 is a `rate_limit_error`, not
        // the generic `api_error` bucket.
        assert_eq!(err.error_type, "rate_limit_error");
    }

    #[test]
    fn rate_limited_carries_proxy_rate_limited_code() {
        // Issue #10: proxy-enforced rejections must be distinguishable from
        // provider 429s.
        let err = ApiError::rate_limited("too many requests");
        assert_eq!(err.status, StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(err.code.as_deref(), Some("proxy_rate_limited"));
    }

    #[test]
    fn budget_exceeded_carries_budget_exceeded_code() {
        // Budget-cap rejections share HTTP 429 with rate limiting but must
        // carry a distinct code so clients can tell them apart.
        let err = ApiError::budget_exceeded("deployment budget (10 USD) exceeded");
        assert_eq!(err.status, StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(err.error_type, "rate_limit_error");
        assert_eq!(err.code.as_deref(), Some("budget_exceeded"));
    }

    #[test]
    fn upstream_429_carries_upstream_rate_limited_code() {
        // Issue #10: provider 429s must report a distinct code so clients can
        // tell "our limit" from "provider limit".
        let err = ApiError::upstream(429, "provider throttled");
        assert_eq!(err.status, StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(err.code.as_deref(), Some("upstream_rate_limited"));
    }

    #[test]
    fn upstream_500_carries_upstream_error_code() {
        // Issue #10: provider 5xx errors report `upstream_error`.
        let err = ApiError::upstream(500, "internal");
        assert_eq!(err.status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(err.code.as_deref(), Some("upstream_error"));
    }

    #[test]
    fn upstream_503_carries_upstream_error_code() {
        let err = ApiError::upstream(503, "unavailable");
        assert_eq!(err.status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(err.code.as_deref(), Some("upstream_error"));
    }

    #[test]
    fn upstream_400_has_no_code() {
        // Non-429, non-5xx upstream statuses get no disambiguating code.
        let err = ApiError::upstream(400, "bad request from provider");
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
        assert!(err.code.is_none());
    }

    #[test]
    fn upstream_invalid_status_defaults_to_bad_gateway() {
        // Out-of-range (>999), 1xx (informational — a body cannot be served),
        // and ≥600 (not real HTTP statuses) must all synthesize to 502 rather
        // than pass through a non-standard status with an error body.
        for status in [1000, 199, 600, 750, 99] {
            let err = ApiError::upstream(status, "weird");
            assert_eq!(
                err.status,
                StatusCode::BAD_GATEWAY,
                "status {status} must synthesize to 502"
            );
            assert_eq!(err.error_type, "api_error");
        }
        // A real 5xx keeps its own status and type.
        let err = ApiError::upstream(503, "unavailable");
        assert_eq!(err.status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(err.error_type, "api_error");
    }

    #[test]
    fn upstream_2xx_synthesizes_to_bad_gateway() {
        // `ApiError::upstream` is semantically an *error* constructor; a 2xx
        // success status paired with an error body is a nonsensical
        // combination. Narrowing the accepted range to 4xx–5xx forces 2xx
        // (and 3xx) inputs to synthesize to 502, so a buggy/malicious caller
        // or upstream cannot produce a "200 OK error". The synthesized 502
        // then derives `api_error` type and `upstream_error` code, matching a
        // genuine 502.
        for status in [200u16, 201, 204, 226, 300, 301, 308] {
            let err = ApiError::upstream(status, "ok-but-error");
            assert_eq!(
                err.status,
                StatusCode::BAD_GATEWAY,
                "2xx/3xx status {status} must synthesize to 502"
            );
            assert_eq!(err.error_type, "api_error");
            assert_eq!(
                err.code.as_deref(),
                Some("upstream_error"),
                "synthesized 502 must carry upstream_error code"
            );
        }
    }

    #[test]
    fn upstream_401_carries_authentication_code() {
        // A provider 401 (rejected upstream credential) must carry a code
        // distinct from `None` (which a generic 400 keeps) so clients can
        // detect a stale upstream credential and distinguish it from a
        // malformed request body. This mirrors the reference
        // `AuthenticationError` / `BadRequestError` disambiguation.
        let err = ApiError::upstream(401, "invalid api key from provider");
        assert_eq!(err.status, StatusCode::UNAUTHORIZED);
        assert_eq!(err.error_type, "invalid_request_error");
        assert_eq!(err.code.as_deref(), Some("upstream_authentication_error"));
    }

    #[test]
    fn upstream_403_carries_permission_code() {
        // A provider 403 (forbidden) already reports `permission_error` type;
        // it now also carries `upstream_permission_error` code so clients can
        // attribute it to the upstream rather than the proxy.
        let err = ApiError::upstream(403, "forbidden by provider");
        assert_eq!(err.status, StatusCode::FORBIDDEN);
        assert_eq!(err.error_type, "permission_error");
        assert_eq!(err.code.as_deref(), Some("upstream_permission_error"));
    }

    #[test]
    fn upstream_400_and_404_remain_codeless() {
        // Non-auth, non-rate-limit, non-5xx upstream statuses keep no
        // disambiguating code — they remain indistinguishable from a generic
        // 4xx by design (no reference exception type maps to a distinct code
        // here).
        for status in [400u16, 404, 418, 422] {
            let err = ApiError::upstream(status, "misc");
            assert!(err.code.is_none(), "status {status} should have no code");
        }
    }

    #[test]
    fn synthesize_upstream_status_boundary_values() {
        // Only the 4xx–5xx error range passes through; everything else
        // synthesizes to 502.
        for pass_through in [400u16, 401, 403, 404, 429, 499, 500, 502, 599] {
            assert_eq!(
                synthesize_upstream_status(pass_through).as_u16(),
                pass_through,
                "status {pass_through} should pass through"
            );
        }
        for synthesized in [
            0u16, 99, 100, 199, 200, 201, 204, 226, 300, 301, 399, 600, 700, 999, 1000,
        ] {
            assert_eq!(
                synthesize_upstream_status(synthesized),
                StatusCode::BAD_GATEWAY,
                "status {synthesized} should synthesize to 502"
            );
        }
    }

    #[test]
    fn error_type_for_status_boundary_values() {
        // Guards the match-arm ordering: a regression like swapping the `403`
        // and `400..=499` arms is only caught if a boundary case is present.
        assert_eq!(error_type_for_status(399), "api_error");
        assert_eq!(error_type_for_status(400), "invalid_request_error");
        assert_eq!(error_type_for_status(402), "invalid_request_error");
        assert_eq!(error_type_for_status(403), "permission_error");
        assert_eq!(error_type_for_status(404), "invalid_request_error");
        assert_eq!(error_type_for_status(428), "invalid_request_error");
        assert_eq!(error_type_for_status(429), "rate_limit_error");
        assert_eq!(error_type_for_status(430), "invalid_request_error");
        assert_eq!(error_type_for_status(499), "invalid_request_error");
        assert_eq!(error_type_for_status(500), "api_error");
        assert_eq!(error_type_for_status(503), "api_error");
        assert_eq!(error_type_for_status(599), "api_error");
        assert_eq!(error_type_for_status(600), "api_error");
        assert_eq!(error_type_for_status(0), "api_error");
    }

    #[test]
    fn code_for_upstream_status_matches_non_streaming_constructor() {
        // Parity contract: the streaming SSE error event and the
        // non-streaming `ApiError::upstream` body both source their `code`
        // from `code_for_upstream_status`. Asserting that
        // `ApiError::upstream(s).code` equals
        // `code_for_upstream_status(synthesize_upstream_status(s))` for every
        // representative status pins the parity so the two paths cannot
        // silently diverge again (the original MEDIUM issue).
        for status in [
            0u16, 200, 204, 301, 400, 401, 403, 404, 418, 429, 430, 500, 502, 503, 504, 599, 600,
            999,
        ] {
            let synthesized = synthesize_upstream_status(status).as_u16();
            let expected = code_for_upstream_status(synthesized);
            let actual = ApiError::upstream(status, "x").code;
            assert_eq!(
                actual.as_deref(),
                expected,
                "status {status} (synthesized {synthesized}): constructor must match helper"
            );
        }
    }

    #[test]
    fn upstream_invalid_status_code_matches_synthesized_status() {
        // An out-of-range status is synthesized to 502; the disambiguating
        // `code` must track that synthesized 502, not the raw input, so
        // clients switching on `code` recognise it as an upstream error.
        let err = ApiError::upstream(1000, "weird");
        assert_eq!(err.status, StatusCode::BAD_GATEWAY);
        assert_eq!(err.code.as_deref(), Some("upstream_error"));
    }

    #[test]
    fn core_error_model_not_found_maps() {
        let core_err = oxllm_core::Error::ModelNotFound("foo".to_string());
        let api_err: ApiError = core_err.into();
        assert_eq!(api_err.status, StatusCode::NOT_FOUND);
    }

    #[test]
    fn core_error_unauthorized_maps() {
        let core_err = oxllm_core::Error::Unauthorized;
        let api_err: ApiError = core_err.into();
        assert_eq!(api_err.status, StatusCode::UNAUTHORIZED);
    }

    #[test]
    fn rate_limited_has_correct_type() {
        let err = ApiError::rate_limited("too many requests");
        assert_eq!(err.status, StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(err.error_type, "rate_limit_error");
    }

    #[test]
    fn rate_limited_response_has_retry_after_header() {
        let err = ApiError::rate_limited("too many requests");
        let resp = err.into_response();
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
        let retry_after = resp
            .headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u64>().ok());
        assert_eq!(retry_after, Some(60));
    }

    #[test]
    fn non_rate_limited_response_has_no_retry_after_header() {
        let err = ApiError::bad_request("nope");
        let resp = err.into_response();
        assert!(resp.headers().get("retry-after").is_none());
    }

    #[test]
    fn with_headers_forwards_allowlist() {
        // A provider 429's headers carry debugging (x-request-id) and
        // rate-limit (retry-after, x-ratelimit-*) metadata that the client
        // must see. Only the curated allowlist is forwarded; everything
        // else stays private (strip_leaking_headers posture).
        let mut upstream = axum::http::HeaderMap::new();
        upstream.insert("x-request-id", "req_abc123".parse().unwrap());
        upstream.insert("retry-after", "5".parse().unwrap());
        upstream.insert("x-ratelimit-remaining-requests", "10".parse().unwrap());
        // Not in the allowlist — must NOT leak to the client.
        upstream.insert("server", "vendor/1.0".parse().unwrap());
        upstream.insert("x-vendor-internal", "secret".parse().unwrap());

        let err = ApiError::upstream(429, "rate limited").with_headers(Some(upstream));
        let resp = err.into_response();

        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(
            resp.headers()
                .get("x-request-id")
                .and_then(|v| v.to_str().ok()),
            Some("req_abc123")
        );
        assert_eq!(
            resp.headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok()),
            Some("5")
        );
        assert_eq!(
            resp.headers()
                .get("x-ratelimit-remaining-requests")
                .and_then(|v| v.to_str().ok()),
            Some("10")
        );
        // Non-allowlisted headers are never forwarded.
        assert!(resp.headers().get("server").is_none());
        assert!(resp.headers().get("x-vendor-internal").is_none());
    }

    #[test]
    fn upstream_retry_after_overrides_default() {
        // The vendor's backoff signal must reach the client instead of being
        // clobbered by the hardcoded 60s default.
        let mut upstream = axum::http::HeaderMap::new();
        upstream.insert("retry-after", "5".parse().unwrap());
        let err = ApiError::upstream(429, "rate limited").with_headers(Some(upstream));
        let resp = err.into_response();
        let retry_after = resp
            .headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u64>().ok());
        assert_eq!(retry_after, Some(5));
    }

    #[test]
    fn no_headers_keeps_default_retry_after() {
        // Backward compat: a proxy-synthesized 429 (no upstream response)
        // still emits the 60s default.
        let err = ApiError::rate_limited("too many requests");
        assert!(err.headers.is_none());
        let resp = err.into_response();
        let retry_after = resp
            .headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u64>().ok());
        assert_eq!(retry_after, Some(60));
    }

    #[test]
    fn non_429_with_headers_forwards_x_request_id_but_no_retry_after() {
        // A 5xx upstream error should still surface x-request-id (debugging)
        // but must not synthesize a retry-after.
        let mut upstream = axum::http::HeaderMap::new();
        upstream.insert("x-request-id", "req_500".parse().unwrap());
        let err = ApiError::upstream(500, "boom").with_headers(Some(upstream));
        let resp = err.into_response();
        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(
            resp.headers()
                .get("x-request-id")
                .and_then(|v| v.to_str().ok()),
            Some("req_500")
        );
        assert!(resp.headers().get("retry-after").is_none());
    }

    #[test]
    fn budget_exceeded_omits_default_retry_after() {
        // Budget windows reset on daily/monthly horizons; the 60s default
        // targets per-minute RPM and would mislead clients into hot-retrying
        // a budget-capped request. The header must be omitted so clients
        // fall back to their own backoff policy.
        let resp = ApiError::budget_exceeded("over").into_response();
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
        assert!(
            resp.headers().get("retry-after").is_none(),
            "budget_exceeded must not synthesize a misleading Retry-After"
        );
    }

    #[tokio::test]
    async fn core_error_config_redacts_detail_from_body() {
        // Internal variants (Config/Transform/Network/Serialization) must
        // never interpolate their raw detail into the client-visible body —
        // a future `Error::Config(format!("... {secret}"))` would otherwise
        // flow straight to the response with no redaction gate.
        let core_err = oxllm_core::Error::Config("contains sensitive: sk-abc123".to_string());
        let resp: axum::response::Response = ApiError::from(core_err).into_response();
        let bytes = axum::body::to_bytes(resp.into_body(), 8 * 1024)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let message = body["error"]["message"].as_str().unwrap();
        assert_eq!(message, "internal error");
        assert!(
            !body.to_string().contains("sk-abc123"),
            "internal detail must not leak into the response body"
        );
    }

    #[test]
    fn core_error_network_maps_to_bad_gateway() {
        // A failure to reach the upstream is an upstream-side problem (502),
        // matching genuine upstream 502s, not a proxy-internal 500. The
        // `upstream_unreachable` code marks it as gateway-synthesized so the
        // transport/provider distinction survives the boundary.
        let api_err: ApiError =
            ApiError::from(oxllm_core::Error::Network("conn refused".to_string()));
        assert_eq!(api_err.status, StatusCode::BAD_GATEWAY);
        assert_eq!(api_err.code.as_deref(), Some("upstream_unreachable"));
        assert_eq!(api_err.message, "upstream error");
    }

    #[test]
    fn network_unreachable_constructor_shape() {
        let err = ApiError::network_unreachable();
        assert_eq!(err.status, StatusCode::BAD_GATEWAY);
        assert_eq!(err.error_type, "api_error");
        assert_eq!(err.code.as_deref(), Some("upstream_unreachable"));
        assert_eq!(err.message, "upstream error");
        assert!(err.headers.is_none());
    }

    #[test]
    fn gateway_timeout_constructor_shape() {
        let err = ApiError::gateway_timeout();
        assert_eq!(err.status, StatusCode::GATEWAY_TIMEOUT);
        assert_eq!(err.error_type, "api_error");
        assert_eq!(err.code.as_deref(), Some("upstream_timeout"));
        assert_eq!(err.message, "upstream request timeout");
        assert!(err.headers.is_none());
    }

    #[test]
    fn transport_codes_distinct_from_provider_served_statuses() {
        // A genuine provider 502/504 (via `ApiError::upstream`) must keep the
        // generic `upstream_error` code — only gateway-synthesized transport
        // failures carry the marker codes.
        assert_eq!(
            ApiError::upstream(502, "provider down").code.as_deref(),
            Some("upstream_error")
        );
        assert_eq!(
            ApiError::upstream(504, "provider slow").code.as_deref(),
            Some("upstream_error")
        );
    }

    #[test]
    fn model_not_found_truncates_huge_model_echo() {
        // A request-controlled `model` field must not amplify the error body:
        // the echo is bounded at MAX_MODEL_ECHO_CHARS even when the request
        // body allows a multi-megabyte model string.
        let err = ApiError::model_not_found(&"m".repeat(10_000));
        assert!(err.message.len() < 512, "message must stay bounded");
        assert!(err.message.contains("does not exist"));
        assert_eq!(
            truncate_model_for_echo(&"m".repeat(10_000)).chars().count(),
            256
        );
    }

    #[test]
    fn model_echo_truncation_respects_char_boundaries() {
        let model: String = "é".repeat(300);
        let truncated = truncate_model_for_echo(&model);
        assert_eq!(truncated.chars().count(), 256);
    }

    #[tokio::test]
    async fn forwarded_absurd_numeric_retry_after_is_clamped() {
        // A misbehaving upstream sending `retry-after: 99999999` must not
        // push clients into an arbitrarily long backoff: numeric values are
        // clamped to MAX_FORWARDED_RETRY_AFTER_SECS.
        let mut upstream = axum::http::HeaderMap::new();
        upstream.insert("retry-after", "99999999".parse().unwrap());
        let err = ApiError::upstream(429, "throttled").with_headers(Some(upstream));
        let resp = err.into_response();
        assert_eq!(
            resp.headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok()),
            Some("3600")
        );
    }

    #[tokio::test]
    async fn forwarded_small_numeric_retry_after_passes_through() {
        let mut upstream = axum::http::HeaderMap::new();
        upstream.insert("retry-after", "11".parse().unwrap());
        let err = ApiError::upstream(429, "throttled").with_headers(Some(upstream));
        let resp = err.into_response();
        assert_eq!(
            resp.headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok()),
            Some("11")
        );
    }

    #[test]
    fn core_error_pre_call_check_failed_maps_to_bad_request() {
        // A pre-call-check rejection is operator-configured and often
        // client-caused; surface it as a 400 that preserves the reason.
        let api_err: ApiError = ApiError::from(oxllm_core::Error::PreCallCheckFailed(
            "too many tokens".to_string(),
        ));
        assert_eq!(api_err.status, StatusCode::BAD_REQUEST);
        assert!(api_err.message.contains("too many tokens"));
    }

    #[test]
    fn core_error_from_maps_every_variant() {
        // Table-driven coverage of `From<oxllm_core::Error>` so each arm has
        // a resulting (status, error_type, code) assertion.
        use oxllm_core::Error;
        let cases: Vec<(Error, StatusCode, &str, Option<&str>)> = vec![
            (
                Error::ModelNotFound("foo".into()),
                StatusCode::NOT_FOUND,
                "invalid_request_error",
                Some("model_not_found"),
            ),
            (
                Error::Unauthorized,
                StatusCode::UNAUTHORIZED,
                "invalid_request_error",
                Some("invalid_api_key"),
            ),
            (
                Error::BadRequest("bad".into()),
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                None,
            ),
            (
                Error::RateLimited("slow".into()),
                StatusCode::TOO_MANY_REQUESTS,
                "rate_limit_error",
                Some("proxy_rate_limited"),
            ),
            (
                Error::BudgetExceeded("budget cap".into()),
                StatusCode::TOO_MANY_REQUESTS,
                "rate_limit_error",
                Some("budget_exceeded"),
            ),
            (
                Error::Upstream {
                    status: 500,
                    message: "boom".into(),
                },
                StatusCode::INTERNAL_SERVER_ERROR,
                "api_error",
                Some("upstream_error"),
            ),
            // OpenAI's `error.type` disambiguates by status: 400s are
            // `invalid_request_error`, 403s `permission_error`, 429s
            // `rate_limit_error`, 5xx `api_error`.
            (
                Error::Upstream {
                    status: 400,
                    message: "bad".into(),
                },
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                None,
            ),
            (
                Error::Upstream {
                    status: 403,
                    message: "forbidden".into(),
                },
                StatusCode::FORBIDDEN,
                "permission_error",
                Some("upstream_permission_error"),
            ),
            (
                Error::Upstream {
                    status: 429,
                    message: "throttled".into(),
                },
                StatusCode::TOO_MANY_REQUESTS,
                "rate_limit_error",
                Some("upstream_rate_limited"),
            ),
            (
                Error::Network("x".into()),
                StatusCode::BAD_GATEWAY,
                "api_error",
                Some("upstream_unreachable"),
            ),
            (
                Error::Serialization("x".into()),
                StatusCode::INTERNAL_SERVER_ERROR,
                "api_error",
                None,
            ),
            (
                Error::Config("x".into()),
                StatusCode::INTERNAL_SERVER_ERROR,
                "api_error",
                None,
            ),
            (
                Error::Transform("x".into()),
                StatusCode::INTERNAL_SERVER_ERROR,
                "api_error",
                None,
            ),
            (
                Error::PreCallCheckFailed("x".into()),
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                None,
            ),
            (
                Error::Internal,
                StatusCode::INTERNAL_SERVER_ERROR,
                "api_error",
                None,
            ),
        ];
        for (err, status, error_type, code) in cases {
            let api: ApiError = err.into();
            assert_eq!(api.status, status, "status mismatch");
            assert_eq!(api.error_type, error_type, "error_type mismatch");
            assert_eq!(api.code.as_deref(), code, "code mismatch");
        }
    }

    #[tokio::test]
    async fn bad_request_renders_full_openai_envelope() {
        // Assert the complete OpenAI-shaped body, not just the status.
        let resp = ApiError::bad_request("missing field").into_response();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let bytes = axum::body::to_bytes(resp.into_body(), 8 * 1024)
            .await
            .unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["error"]["message"].as_str(), Some("missing field"));
        assert_eq!(
            body["error"]["type"].as_str(),
            Some("invalid_request_error")
        );
        // OpenAI's error object always carries a (nullable) `code` field,
        // symmetric with `param`: rendered explicitly as `null` (not omitted)
        // so consumers doing a presence check (`"code" in err`) see the same
        // shape as a real OpenAI response.
        assert_eq!(body["error"].get("code"), Some(&serde_json::Value::Null));
        // OpenAI's error object always carries a (nullable) `param` field.
        assert_eq!(body["error"].get("param"), Some(&serde_json::Value::Null));
    }

    #[test]
    fn bad_request_renders_code_null() {
        // A code-less error must still render `"code": null` (not omit the
        // key), symmetric with `param`, so consumers doing a presence check
        // (`"code" in err`) see the same shape as a real OpenAI response.
        let err = ApiError::bad_request("missing field");
        assert_eq!(err.code, None);
        let value = error_envelope_object(&err.message, &err.error_type, err.code.as_deref());
        assert_eq!(value.get("code"), Some(&serde_json::Value::Null));
        assert_eq!(value.get("param"), Some(&serde_json::Value::Null));
    }

    #[test]
    fn error_envelope_object_code_overwrites_null_when_present() {
        let value = error_envelope_object("x", "rate_limit_error", Some("upstream_rate_limited"));
        assert_eq!(value["code"].as_str(), Some("upstream_rate_limited"));
        assert_eq!(value["param"], serde_json::Value::Null);
    }

    #[tokio::test]
    async fn streaming_and_non_streaming_error_envelopes_match() {
        // Parity contract: the streaming SSE error event and the
        // non-streaming `ApiError::upstream` body both source their envelope
        // from `error_envelope_object`. Asserting the helper's output equals
        // the rendered non-streaming body for representative statuses pins the
        // full `{message,type,param,code}` shape (not just `code`) so the two
        // paths cannot silently diverge again.
        for status in [
            400u16, 401, 403, 404, 418, 422, 429, 430, 500, 502, 503, 504, 599,
        ] {
            let resp = ApiError::upstream(status, "boom").into_response();
            let bytes = axum::body::to_bytes(resp.into_body(), 8 * 1024)
                .await
                .unwrap();
            let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();

            let synth = synthesize_upstream_status(status);
            let expected = error_envelope_object(
                "boom",
                error_type_for_status(synth.as_u16()),
                code_for_upstream_status(synth.as_u16()),
            );
            assert_eq!(
                body["error"], expected,
                "status {status}: streaming helper must match non-streaming body"
            );
        }
    }

    #[tokio::test]
    async fn from_upstream_drops_headers_and_uses_default_retry_after() {
        // The `From<oxllm_core::Error>` conversion is intentionally
        // header-less (see the doc-note on the arm): the core `Error` type
        // carries no header payload, so a 429 through it falls back to the
        // hardcoded 60s default and exposes no `x-request-id`. This pins the
        // accepted behaviour so a future regression (e.g. routing a real
        // upstream 429 through `Error::Upstream`) is caught.
        let api: ApiError = oxllm_core::Error::Upstream {
            status: 429,
            message: "throttled".into(),
        }
        .into();
        assert!(api.headers.is_none(), "From-conversion carries no headers");
        let resp = api.into_response();
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
        assert!(resp.headers().get("x-request-id").is_none());
        let retry_after = resp
            .headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.parse::<u64>().ok());
        assert_eq!(retry_after, Some(60));
    }

    #[tokio::test]
    async fn budget_exceeded_with_headers_forwards_x_request_id_but_omits_retry_after() {
        // Defence-in-depth: a `budget_exceeded` 429 built with upstream headers
        // attached must still forward the allowlisted `x-request-id` while
        // omitting the synthesized `retry-after` default (budget windows reset
        // on daily/monthly horizons; the 60s default would mislead clients).
        // Today `budget_exceeded` is always proxy-synthesized so `headers` is
        // `None`, but the combination is worth pinning: the omission guard
        // keys off `code == budget_exceeded` regardless of attached headers.
        let mut upstream = axum::http::HeaderMap::new();
        upstream.insert("x-request-id", "req_bud".parse().unwrap());
        let err = ApiError::budget_exceeded("over").with_headers(Some(upstream));
        let resp = err.into_response();
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(
            resp.headers()
                .get("x-request-id")
                .and_then(|v| v.to_str().ok()),
            Some("req_bud")
        );
        assert!(
            resp.headers().get("retry-after").is_none(),
            "budget_exceeded must omit the synthesized retry-after default"
        );
    }

    #[tokio::test]
    async fn upstream_http_date_retry_after_survives() {
        // The precedence check treats any forwarded `retry-after` `HeaderValue`
        // as present, including an HTTP-date format, and so skips the numeric
        // 60s default. Confirm the date string reaches the client verbatim.
        let mut upstream = axum::http::HeaderMap::new();
        upstream.insert(
            "retry-after",
            "Wed, 21 Oct 2026 07:28:00 GMT".parse().unwrap(),
        );
        let err = ApiError::upstream(429, "throttled").with_headers(Some(upstream));
        let resp = err.into_response();
        assert_eq!(
            resp.headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok()),
            Some("Wed, 21 Oct 2026 07:28:00 GMT")
        );
    }

    #[tokio::test]
    async fn forbidden_not_found_internal_render_full_envelopes() {
        // `forbidden` / `not_found` / `internal` previously had only structural
        // (field-level) assertions; drive them through `into_response` to pin
        // the full envelope shape including the now-always-present nullable
        // `code`.
        let cases = [
            (
                ApiError::forbidden("nope"),
                StatusCode::FORBIDDEN,
                "permission_error",
            ),
            (
                ApiError::not_found("nope"),
                StatusCode::NOT_FOUND,
                "invalid_request_error",
            ),
            (
                ApiError::internal("nope"),
                StatusCode::INTERNAL_SERVER_ERROR,
                "api_error",
            ),
        ];
        for (err, status, error_type) in cases {
            let resp = err.into_response();
            assert_eq!(resp.status(), status);
            let bytes = axum::body::to_bytes(resp.into_body(), 8 * 1024)
                .await
                .unwrap();
            let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(
                body["error"]["message"].as_str(),
                Some("nope"),
                "status {status}"
            );
            assert_eq!(
                body["error"]["type"].as_str(),
                Some(error_type),
                "status {status}"
            );
            assert_eq!(
                body["error"].get("code"),
                Some(&serde_json::Value::Null),
                "status {status}: codeless error renders null code"
            );
            assert_eq!(
                body["error"].get("param"),
                Some(&serde_json::Value::Null),
                "status {status}"
            );
        }
    }
}
