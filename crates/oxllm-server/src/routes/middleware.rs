//! Shared middleware layers and span builders used by every route group.
//!
//! These helpers exist so that the auth / allowed-IPs / auth-brake trio and
//! the global request-limit / smuggling / tracing layers are applied from a
//! single place. Route modules never call `.layer(...)` directly with these
//! guards — they go through the helpers here, which keeps the ordering
//! identical to the original single-file router.

use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::middleware;
use axum::response::Response;
use std::time::Instant;
use tower_http::limit::RequestBodyLimitLayer;
use tower_http::trace::{MakeSpan, TraceLayer};
use tracing::Span;

use crate::auth::auth_middleware;
use crate::metrics;
use crate::security::{allowed_ips_guard, auth_brake_guard, smuggling_guard};
use crate::state::SharedState;

/// A `MakeSpan` that records the HTTP method and URI but never records
/// header values, ensuring secrets like `authorization`, `x-api-key`,
/// `api-key`, and `cookie` never appear in tracing spans.
#[derive(Clone, Copy)]
pub(crate) struct RedactingMakeSpan;

impl<B> MakeSpan<B> for RedactingMakeSpan {
    fn make_span(&mut self, request: &axum::http::Request<B>) -> Span {
        tracing::info_span!(
 "http.request",
 method = %request.method(),
 uri = %redact_uri(request.uri()),
  version = ?request.version(),
  request_id = tracing::field::Empty)
    }
}

/// Log path (and non-secret query) only — strip known secret query params such
/// as `authorization` used by the realtime WebSocket auth fallback.
///
/// Classification delegates to [`crate::security::is_secret_key`] (the
/// documented chokepoint for secret-bearing name classification, which
/// over-matches on purpose) and additionally redacts the bare keys `key` and
/// `sk` that some SDKs use as query-param names for credentials. Redaction
/// only affects log output; over-redacting a benign key is harmless.
fn redact_uri(uri: &axum::http::Uri) -> String {
    let path = uri.path();
    let Some(query) = uri.query() else {
        return path.to_string();
    };
    let mut kept = Vec::new();
    for pair in query.split('&') {
        if pair.is_empty() {
            continue;
        }
        let key = pair.split('=').next().unwrap_or(pair);
        let key_lower = key.to_ascii_lowercase();
        if crate::security::is_secret_key(&key_lower) || key_lower == "key" || key_lower == "sk" {
            kept.push(format!("{key}=[REDACTED]"));
        } else {
            kept.push(pair.to_string());
        }
    }
    if kept.is_empty() {
        path.to_string()
    } else {
        format!("{path}?{}", kept.join("&"))
    }
}

/// Apply the standard proxy auth stack — auth middleware, allowed-IPs guard,
/// and auth-brake guard — to a route group. The order matches the original
/// router: auth is innermost (closest to the handler), auth-brake is
/// outermost (first to see a request).
///
/// Uses [`auth_with_preflight`], which bypasses auth for `OPTIONS` requests
/// so a browser's CORS preflight can reach the inner CORS layer. This is only
/// correct for route groups that carry a CORS layer (e.g. `v1_routes`); for
/// CORS-less groups use [`apply_proxy_auth_no_preflight`] instead, otherwise
/// an unauthenticated `OPTIONS` would slip past auth into a handler whose
/// `Extension<AuthUser>` extractor then fails with 500.
pub(crate) fn apply_proxy_auth(
    router: Router<SharedState>,
    shared: SharedState,
) -> Router<SharedState> {
    router
        .layer(middleware::from_fn_with_state(
            shared.clone(),
            auth_with_preflight,
        ))
        .layer(middleware::from_fn_with_state(
            shared.clone(),
            allowed_ips_guard,
        ))
        .layer(middleware::from_fn_with_state(shared, auth_brake_guard))
}

/// Apply the standard proxy auth stack without the CORS-preflight bypass.
///
/// This is the correct wrapper for route groups that have no inner CORS layer
/// (notably the global provider catch-all `/{*path}`). For such groups the
/// preflight bypass in [`auth_with_preflight`] is purely harmful — an
/// unauthenticated `OPTIONS` would be forwarded past [`auth_middleware`] into
/// a handler whose `Extension<AuthUser>` extractor is missing, producing a
/// 500 instead of the correct 401. Mirrors [`apply_proxy_auth`] in every
/// other respect (same layer ordering).
pub(crate) fn apply_proxy_auth_no_preflight(
    router: Router<SharedState>,
    shared: SharedState,
) -> Router<SharedState> {
    router
        .layer(middleware::from_fn_with_state(
            shared.clone(),
            auth_middleware,
        ))
        .layer(middleware::from_fn_with_state(
            shared.clone(),
            allowed_ips_guard,
        ))
        .layer(middleware::from_fn_with_state(shared, auth_brake_guard))
}

/// [`auth_middleware`] wrapped with a CORS preflight (OPTIONS) bypass.
///
/// On the proxy LLM routes the auth stack sits *outside* the CORS layer (see
/// [`build_router`](crate::routes::build_router): CORS is applied before
/// [`apply_proxy_auth`]). Without a bypass, a browser's preflight `OPTIONS`
/// request — which by design carries no `Authorization` header — would be
/// rejected with 401 by [`auth_middleware`] before the inner CORS layer could
/// attach the preflight response headers, breaking every credentialed
/// cross-origin request whenever auth is enabled.
///
/// Only `OPTIONS` is passed through; every `GET`/`POST`/`PUT`/`DELETE`/`PATCH`
/// still requires valid credentials via [`auth_middleware`].
async fn auth_with_preflight(
    state: axum::extract::State<SharedState>,
    headers: axum::http::HeaderMap,
    request: axum::extract::Request,
    next: middleware::Next,
) -> Response {
    if request.method() == axum::http::Method::OPTIONS {
        // Let the inner CORS layer handle the preflight — no auth required.
        return next.run(request).await;
    }
    auth_middleware(state, headers, request, next).await
}

/// Apply the global layers that wrap the entire application router: the
/// request body limit, the request-smuggling guard, the request-timing
/// middleware (records `oxllm_request_duration_seconds`), and the redacting
/// HTTP trace layer. Generic over the router state so it can be applied before
/// `with_state` finalizes the router.
pub(crate) fn apply_global_layers<S>(router: Router<S>, body_limit: usize) -> Router<S>
where
    S: Clone + Send + Sync + 'static,
{
    router
        .layer(DefaultBodyLimit::max(body_limit))
        .layer(RequestBodyLimitLayer::new(body_limit))
        .layer(middleware::from_fn(smuggling_guard))
        .layer(middleware::from_fn(request_timing))
        .layer(TraceLayer::new_for_http().make_span_with(RedactingMakeSpan))
}

/// Build the LLM-route CORS layer from the configured allow-list. Returns
/// `None` when CORS should be disabled entirely (no allow-list configured).
/// Mirrors the management CORS builder but scopes the allowed methods to the
/// verbs the proxy v1 routes actually expose: `GET`, `POST`, `DELETE` (Anthropic
/// Files / platform assistants, threads, messages), and `OPTIONS` (preflight).
/// No `PUT`/`PATCH` routes exist in the v1 group today.
///
/// `allow_headers` is a wildcard (matching the reference `proxy_server.py:1852`) so
/// that browser clients using Anthropic-style auth (`x-api-key`, `api-key`) and
/// the Anthropic SDK headers (`anthropic-version`, `anthropic-beta`) are not
/// blocked by a narrow preflight allow-list. This is safe because
/// `AllowOrigin::list` already restricts origins to a finite operator-supplied
/// set — the header wildcard cannot be abused by arbitrary origins.
///
/// `expose_headers` lists the `x-oxllm-*` / `x-ratelimit-*` / `retry-after` /
/// `x-request-id` response headers the pipeline and orchestration layers stamp,
/// so browser JS can read routing observability, cost, and rate-limit metadata
/// from `fetch()` responses. Mirrors the reference
/// management-UI exposed-header list (`proxy_server.py:1853`), scoped
/// to oxllm's own header family.
pub(crate) fn build_proxy_cors(
    origins: &Option<Vec<String>>,
) -> Option<tower_http::cors::CorsLayer> {
    use tower_http::cors::{AllowOrigin, Any, CorsLayer};

    let list = origins.as_ref()?;
    if list.is_empty() {
        return None;
    }
    let parsed: Vec<axum::http::HeaderValue> = list
        .iter()
        .filter_map(|o| axum::http::HeaderValue::from_str(o).ok())
        .collect();
    if parsed.is_empty() {
        return None;
    }
    Some(
        CorsLayer::new()
            .allow_origin(AllowOrigin::list(parsed))
            .allow_methods([
                axum::http::Method::GET,
                axum::http::Method::POST,
                axum::http::Method::DELETE,
                axum::http::Method::OPTIONS,
            ])
            .allow_headers(Any)
            .expose_headers([
                // Oxllm-native transparency headers.
                axum::http::HeaderName::from_static("x-oxllm-request-id"),
                axum::http::HeaderName::from_static("x-oxllm-response-cost"),
                axum::http::HeaderName::from_static("x-oxllm-total-tokens"),
                axum::http::HeaderName::from_static("x-oxllm-router-strategy"),
                axum::http::HeaderName::from_static("x-oxllm-router-decision"),
                axum::http::HeaderName::from_static("x-oxllm-cache-hit"),
                axum::http::HeaderName::from_static("x-oxllm-fallback-used"),
                axum::http::HeaderName::from_static("x-oxllm-deployment-id"),
                axum::http::HeaderName::from_static("x-oxllm-model-id"),
                axum::http::HeaderName::from_static("x-oxllm-model-name"),
                axum::http::HeaderName::from_static("x-oxllm-model-group"),
                axum::http::HeaderName::from_static("x-oxllm-requested-model"),
                axum::http::HeaderName::from_static("x-oxllm-attempted-retries"),
                axum::http::HeaderName::from_static("x-oxllm-attempted-fallbacks"),
                axum::http::HeaderName::from_static("x-oxllm-max-retries"),
                axum::http::HeaderName::from_static("x-oxllm-fallback-errors"),
                axum::http::HeaderName::from_static("x-oxllm-response-duration-ms"),
                axum::http::HeaderName::from_static("x-oxllm-upstream-duration-ms"),
                axum::http::HeaderName::from_static("x-oxllm-overhead-duration-ms"),
                // Additive compatibility response aliases.
                axum::http::HeaderName::from_static("x-litellm-model-id"),
                axum::http::HeaderName::from_static("x-litellm-model-name"),
                axum::http::HeaderName::from_static("x-litellm-model-group"),
                axum::http::HeaderName::from_static("x-litellm-attempted-retries"),
                axum::http::HeaderName::from_static("x-litellm-attempted-fallbacks"),
                axum::http::HeaderName::from_static("x-litellm-max-retries"),
                axum::http::HeaderName::from_static("x-litellm-fallback-errors"),
                axum::http::HeaderName::from_static("x-litellm-response-duration-ms"),
                axum::http::HeaderName::from_static("x-litellm-overhead-duration-ms"),
                // OpenAI-standard alias stamped alongside x-oxllm-request-id.
                axum::http::HeaderName::from_static("x-request-id"),
                // Client-side throttling headers (IETF draft pattern).
                axum::http::HeaderName::from_static("x-ratelimit-limit-requests"),
                axum::http::HeaderName::from_static("x-ratelimit-limit-tokens"),
                axum::http::HeaderName::from_static("x-ratelimit-remaining-requests"),
                axum::http::HeaderName::from_static("x-ratelimit-remaining-tokens"),
                axum::http::HeaderName::from_static("x-ratelimit-reset-requests"),
                axum::http::HeaderName::from_static("x-ratelimit-reset-tokens"),
                // Backoff advisory on 429s.
                axum::http::HeaderName::from_static("retry-after"),
            ]),
    )
}

/// Records [`crate::metrics::REQUEST_DURATION`] for every request that
/// traverses the global layer, labelled with the HTTP status family.
///
/// For streaming (SSE) responses the value captures time-to-first-byte, not
/// full stream duration: `next.run(request).await` returns as soon as the
/// handler produces the `Response` with a streaming body, which is consumed
/// asynchronously afterward. Operators investigating long-running SSE latency
/// should correlate with streaming-completion metrics rather than this
/// histogram.
async fn request_timing(request: axum::extract::Request, next: middleware::Next) -> Response {
    let started = Instant::now();
    // Start correlation outside auth/body-limit/extractor handling. The
    // pipeline reuses this id, so an early rejection and a late upstream
    // failure are traceable using the same headers as a successful call.
    let request_id = uuid::Uuid::new_v4().to_string();
    Span::current().record("request_id", request_id.as_str());
    let mut response = crate::pipeline::HTTP_REQUEST_START
        .scope(
            started,
            crate::pipeline::HTTP_REQUEST_ID.scope(request_id.clone(), next.run(request)),
        )
        .await;
    crate::pipeline::stamp_request_id_header(&mut response, &request_id);
    let elapsed = started.elapsed().as_secs_f64();
    let total_ms = elapsed * 1000.0;
    // Include conversion/admission work up to the outer response boundary.
    let previous = response
        .headers()
        .get("x-oxllm-response-duration-ms")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<f64>().ok());
    let overhead = response
        .headers()
        .get("x-oxllm-overhead-duration-ms")
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<f64>().ok());
    if let (Some(previous), Some(overhead)) = (previous, overhead) {
        let corrected = (overhead + (total_ms - previous).max(0.0)).to_string();
        for header in [
            "x-oxllm-overhead-duration-ms",
            "x-litellm-overhead-duration-ms",
        ] {
            crate::pipeline::set_header(&mut response, header, &corrected);
        }
    }
    for header in [
        "x-oxllm-response-duration-ms",
        "x-litellm-response-duration-ms",
    ] {
        crate::pipeline::set_header(&mut response, header, &total_ms.to_string());
    }
    let status_family = match response.status().as_u16() {
        200..=299 => "2xx",
        400..=499 => "4xx",
        500..=599 => "5xx",
        other if other < 200 => "1xx",
        other if other < 400 => "3xx",
        _ => "other",
    };
    metrics::record_request_duration(status_family, elapsed);
    response
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{HeaderValue, Request, StatusCode};
    use tower::ServiceExt;

    #[test]
    fn redact_uri_strips_authorization_query() {
        let uri: axum::http::Uri = "/v1/realtime?model=gpt-4o&authorization=sk-secret"
            .parse()
            .unwrap();
        let redacted = redact_uri(&uri);
        assert!(redacted.contains("authorization=[REDACTED]"));
        assert!(!redacted.contains("sk-secret"));
        assert!(redacted.contains("model=gpt-4o"));
    }

    #[test]
    fn redact_uri_strips_common_secret_bearing_query_keys() {
        // Covers the keys `is_secret_key` matches (`apikey`, `api-key`,
        // `api_key`, `token`, `access_token`, `password`, `secret`) plus the
        // bare `key` / `sk` fallbacks added in `redact_uri`. Each secret value
        // uses a distinctive marker that is not a substring of any key name,
        // so the leakage check is exact.
        let pairs = [
            ("apikey", "LEAK1"),
            ("api-key", "LEAK2"),
            ("api_key", "LEAK3"),
            ("x-api-key", "LEAK4"),
            ("token", "LEAK5"),
            ("access_token", "LEAK6"),
            ("password", "LEAK7"),
            ("secret", "LEAK8"),
            ("key", "LEAK9"),
            ("sk", "LEAK10"),
        ];
        let query: String = pairs
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join("&");
        let uri: axum::http::Uri = format!("/v1/x?{query}&model=gpt-4o").parse().unwrap();
        let redacted = redact_uri(&uri);
        for (_, leaked) in pairs {
            assert!(
                !redacted.contains(leaked),
                "secret value {leaked:?} should not appear in redacted uri: {redacted}"
            );
        }
        for (key, _) in pairs {
            assert!(
                redacted.contains(&format!("{key}=[REDACTED]")),
                "key {key:?} should be redacted in: {redacted}"
            );
        }
        assert!(redacted.contains("model=gpt-4o"));
    }

    #[test]
    fn redact_uri_redacts_valueless_secret_key() {
        // `?authorization` with no `=` still classifies as a secret key and is
        // rewritten to `authorization=[REDACTED]` (cosmetic; the original had
        // no value to leak).
        let uri: axum::http::Uri = "/v1/x?authorization".parse().unwrap();
        let redacted = redact_uri(&uri);
        assert!(redacted.contains("authorization=[REDACTED]"));
    }

    #[tokio::test]
    async fn redacting_make_span_omits_authorization_value() {
        let mut make_span = RedactingMakeSpan;
        let request = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("authorization", "Bearer sk-secret-xyz")
            .header("x-api-key", "keyval")
            .header("cookie", "session=abc")
            .body(())
            .expect("request");

        let span = make_span.make_span(&request);
        assert_eq!(span.metadata().map(|m| m.name()), Some("http.request"));
        let _guard = span.enter();
    }

    #[tokio::test]
    async fn bytes_extractor_obeys_configured_limit_above_two_mebibytes() {
        let limit = 3 * 1024 * 1024;
        let app = apply_global_layers(
            axum::Router::new().route(
                "/upload",
                axum::routing::post(
                    |body: axum::body::Bytes| async move { body.len().to_string() },
                ),
            ),
            limit,
        );
        for (size, expected) in [
            (2 * 1024 * 1024 + 1, StatusCode::OK),
            (limit + 1, StatusCode::PAYLOAD_TOO_LARGE),
        ] {
            let request = Request::builder()
                .method("POST")
                .uri("/upload")
                .body(Body::from(vec![b'x'; size]))
                .expect("request");
            assert_eq!(
                app.clone()
                    .oneshot(request)
                    .await
                    .expect("response")
                    .status(),
                expected
            );
        }
    }

    #[test]
    fn redacting_make_span_does_not_record_header_fields() {
        let mut make_span = RedactingMakeSpan;
        let request = Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header(
                "authorization",
                HeaderValue::from_static("Bearer sk-super-secret"),
            )
            .header("x-api-key", "keyval")
            .header("cookie", "session=abc")
            .body(())
            .expect("request");
        let span = make_span.make_span(&request);
        assert_eq!(span.metadata().map(|m| m.name()), Some("http.request"));
    }

    /// MEDIUM (C25): the proxy CORS layer must list `DELETE` in
    /// `allow_methods` so that browser preflights for the Anthropic Files and
    /// platform DELETE routes (`/anthropic/v1/files/{file_id}`,
    /// `/v1/files/{file_id}`, `/v1/assistants/{id}`, `/v1/threads/{id}`, …)
    /// are answered with `Access-Control-Allow-Methods: DELETE`. Without it,
    /// the browser blocks the actual `DELETE`.
    #[tokio::test]
    async fn proxy_cors_preflight_allows_delete_method() {
        let cors = build_proxy_cors(&Some(vec!["http://localhost:3000".to_string()]))
            .expect("cors layer with a valid origin");
        let app = axum::Router::new()
            .route(
                "/anthropic/v1/files/{file_id}",
                axum::routing::delete(|| async { "deleted" }),
            )
            .layer(cors);

        let req = Request::builder()
            .method("OPTIONS")
            .uri("/anthropic/v1/files/file-123")
            .header("origin", "http://localhost:3000")
            .header("access-control-request-method", "DELETE")
            .body(Body::empty())
            .expect("request");
        let resp = app.oneshot(req).await.expect("response");
        assert_eq!(resp.status(), StatusCode::OK);
        let methods = resp
            .headers()
            .get("access-control-allow-methods")
            .expect("preflight must include Access-Control-Allow-Methods")
            .to_str()
            .expect("header value");
        assert!(
            methods.contains("DELETE"),
            "CORS preflight must allow DELETE, got: {methods}"
        );
    }

    /// Guard against the proxy CORS layer being silently narrowed back to a
    /// method set that omits `DELETE` (regression of the MEDIUM fix above).
    #[test]
    fn proxy_cors_returns_none_when_no_origins() {
        assert!(build_proxy_cors(&None).is_none());
        assert!(build_proxy_cors(&Some(vec![])).is_none());
    }

    /// MEDIUM (C25): the proxy CORS preflight must accept arbitrary request
    /// headers, in particular the Anthropic SDK's `x-api-key` / `api-key` and
    /// `anthropic-version` / `anthropic-beta` headers. A browser client using
    /// Anthropic-style auth sends a preflight with
    /// `Access-Control-Request-Headers: x-api-key, anthropic-version`; if the
    /// CORS layer does not allow those, the browser blocks the actual request.
    /// We use a wildcard (`allow_headers(Any)`, matching the reference
    /// `proxy_server.py:1852`), which is safe because the origin allow-list is
    /// already finite.
    #[tokio::test]
    async fn proxy_cors_preflight_allows_anthropic_sdk_headers() {
        let cors = build_proxy_cors(&Some(vec!["http://localhost:3000".to_string()]))
            .expect("cors layer with a valid origin");
        let app = axum::Router::new()
            .route("/v1/messages", axum::routing::post(|| async { "ok" }))
            .layer(cors);

        let req = Request::builder()
            .method("OPTIONS")
            .uri("/v1/messages")
            .header("origin", "http://localhost:3000")
            .header("access-control-request-method", "POST")
            .header(
                "access-control-request-headers",
                "x-api-key, anthropic-version",
            )
            .body(Body::empty())
            .expect("request");
        let resp = app.oneshot(req).await.expect("response");
        assert_eq!(resp.status(), StatusCode::OK);
        // The wildcard is sent as `*` (since we do not enable credentials).
        let allow_headers = resp
            .headers()
            .get("access-control-allow-headers")
            .expect("preflight must include Access-Control-Allow-Headers")
            .to_str()
            .expect("header value");
        // A wildcard (or a reflected list) must not reject the SDK headers.
        // `*` is accepted by browsers when credentials mode is off.
        assert!(
            allow_headers == "*"
                || (allow_headers.contains("x-api-key")
                    && allow_headers.contains("anthropic-version")),
            "CORS must allow x-api-key and anthropic-version via wildcard or explicit list, \
             got: {allow_headers}"
        );
    }

    /// MEDIUM (C25): the proxy CORS layer must expose the `x-oxllm-*` /
    /// `x-ratelimit-*` / `retry-after` response headers so browser JS can read
    /// routing observability, cost, and rate-limit metadata from `fetch()`
    /// responses. Without `expose_headers`, only the CORS-safelisted headers
    /// are readable and `response.headers.get("x-oxllm-request-id")` returns
    /// `null`.
    ///
    /// `Access-Control-Expose-Headers` is attached to actual (non-preflight)
    /// responses, so we send a real POST with an `Origin` header and inspect
    /// the response.
    #[tokio::test]
    async fn proxy_cors_exposes_oxllm_and_ratelimit_headers() {
        let cors = build_proxy_cors(&Some(vec!["http://localhost:3000".to_string()]))
            .expect("cors layer with a valid origin");
        let app = axum::Router::new()
            .route("/v1/messages", axum::routing::post(|| async { "ok" }))
            .layer(cors);

        let req = Request::builder()
            .method("POST")
            .uri("/v1/messages")
            .header("origin", "http://localhost:3000")
            .header("content-type", "application/json")
            .body(Body::empty())
            .expect("request");
        let resp = app.oneshot(req).await.expect("response");
        assert_eq!(resp.status(), StatusCode::OK);
        let expose = resp
            .headers()
            .get("access-control-expose-headers")
            .expect("CORS must include Access-Control-Expose-Headers on actual responses");
        let expose = expose.to_str().expect("header value");
        assert!(
            expose.contains("x-oxllm-request-id"),
            "expose-headers must include x-oxllm-request-id, got: {expose}"
        );
        assert!(
            expose.contains("x-ratelimit-remaining-tokens"),
            "expose-headers must include x-ratelimit-remaining-tokens, got: {expose}"
        );
        assert!(
            expose.contains("retry-after"),
            "expose-headers must include retry-after, got: {expose}"
        );
    }

    /// LOW (C25): CORS must sit OUTSIDE the auth stack so that auth/ip/brake
    /// failures (401/403/429) pick up `Access-Control-Allow-Origin` on the way
    /// out — otherwise browsers report a generic CORS error instead of the
    /// real status, degrading the developer experience. This test models the
    /// layering with a middleware that always returns 401 (mimicking
    /// `auth_middleware` rejecting a bad token), wrapped OUTER by the CORS
    /// layer. The 401 response must carry the allow-origin header.
    #[tokio::test]
    async fn proxy_cors_attaches_allow_origin_on_auth_failure() {
        use axum::middleware::Next;
        use axum::response::IntoResponse;

        async fn always_401(_req: axum::extract::Request, _next: Next) -> Response {
            StatusCode::UNAUTHORIZED.into_response()
        }

        let cors =
            build_proxy_cors(&Some(vec!["http://localhost:3000".to_string()])).expect("cors layer");
        // CORS applied AFTER the 401 middleware => CORS is outermost.
        let app = axum::Router::new()
            .route("/v1/messages", axum::routing::post(|| async { "ok" }))
            .layer(middleware::from_fn(always_401))
            .layer(cors);

        let req = Request::builder()
            .method("POST")
            .uri("/v1/messages")
            .header("origin", "http://localhost:3000")
            .header("authorization", "Bearer sk-wrong")
            .body(Body::empty())
            .expect("request");
        let resp = app.oneshot(req).await.expect("response");
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
        let allow_origin = resp
            .headers()
            .get("access-control-allow-origin")
            .expect("401 from auth must carry Access-Control-Allow-Origin when CORS is outermost");
        assert_eq!(
            allow_origin.to_str().unwrap(),
            "http://localhost:3000",
            "allow-origin must echo the configured origin"
        );
    }
}
