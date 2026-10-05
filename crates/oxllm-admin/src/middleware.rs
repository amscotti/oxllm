//! Cross-Site Request Forgery protection and browser response headers.
//!
//! On state-changing requests (POST, PUT, PATCH, DELETE) the
//! [`require_csrf`] middleware compares the `X-CSRF-Token` header against the
//! CSRF token stored on the verified session (attached as a request extension
//! by [`crate::auth::require_session`]). GET/HEAD/OPTIONS requests are exempt since
//! they perform no server-side mutation.

use axum::extract::{Request, State};
use axum::http::{HeaderValue, Method, StatusCode, header};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};

use crate::app::AppState;
use crate::auth::{CSRF_HEADER, SessionCsrf, ct_eq};

// The standard Alpine build evaluates attribute expressions with Function,
// and existing templates use inline scripts/styles/event handlers. Keep those
// working while restricting network sources and forbidding framing. This is
// defense in depth, not a nonce-based XSS policy.
pub(crate) const CONTENT_SECURITY_POLICY: &str = "default-src 'self'; \
script-src 'self' 'unsafe-inline' 'unsafe-eval'; \
style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; \
font-src 'self' https://fonts.gstatic.com; img-src 'self' data:; \
connect-src 'self'; object-src 'none'; base-uri 'none'; \
form-action 'self'; frame-ancestors 'none'";

/// Covers login, authenticated pages, fragments, errors and redirects. Only
/// successful public static assets may be cached; session/CSRF/key-bearing
/// HTML and CSV exports must not be stored by browser or intermediary caches.
pub(crate) async fn security_headers(req: Request, next: Next) -> Response {
    let is_static = req.uri().path().starts_with("/static/");
    let mut response = next.run(req).await;
    let cacheable = is_static && response.status().is_success();
    let headers = response.headers_mut();
    headers.insert(
        header::CONTENT_SECURITY_POLICY,
        HeaderValue::from_static(CONTENT_SECURITY_POLICY),
    );
    headers.insert(header::X_FRAME_OPTIONS, HeaderValue::from_static("DENY"));
    headers.insert(
        header::X_CONTENT_TYPE_OPTIONS,
        HeaderValue::from_static("nosniff"),
    );
    headers.insert(
        header::REFERRER_POLICY,
        HeaderValue::from_static("no-referrer"),
    );
    if !cacheable {
        headers.insert(
            header::CACHE_CONTROL,
            HeaderValue::from_static("no-store, private"),
        );
    }
    response
}

fn is_state_changing(method: &Method) -> bool {
    matches!(
        *method,
        Method::POST | Method::PUT | Method::PATCH | Method::DELETE
    )
}

/// Reject state-changing requests whose `X-CSRF-Token` header does not match
/// the session's CSRF token. Must be layered **after** `require_session`.
pub async fn require_csrf(State(_state): State<AppState>, req: Request, next: Next) -> Response {
    if !is_state_changing(req.method()) {
        return next.run(req).await;
    }

    let Some(SessionCsrf(token)) = req.extensions().get::<SessionCsrf>().cloned() else {
        return csrf_error();
    };

    let header_ok = req
        .headers()
        .get(CSRF_HEADER)
        .and_then(|v| v.to_str().ok())
        .map(|h| ct_eq(h.as_bytes(), token.as_bytes()))
        .unwrap_or(false);

    if header_ok {
        next.run(req).await
    } else {
        csrf_error()
    }
}

fn csrf_error() -> Response {
    (
        StatusCode::FORBIDDEN,
        [("content-type", "text/plain; charset=utf-8")],
        "csrf token missing or invalid",
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::SessionCsrf;
    use axum::middleware::from_fn_with_state;
    use axum::routing::post;
    use axum::{Router, body::Body, http::Request};
    use tower::ServiceExt;

    fn app() -> Router {
        Router::new()
            .route("/echo", post(|| async { "ok" }))
            .route_layer(from_fn_with_state(
                crate::app::test_state_for_middleware(),
                require_csrf,
            ))
    }

    #[tokio::test]
    async fn get_request_passes_through() {
        let app = app();
        let req = Request::builder().uri("/echo").body(Body::empty()).unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_ne!(resp.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn post_without_extension_is_forbidden() {
        let app = app();
        let req = Request::builder()
            .method("POST")
            .uri("/echo")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn post_with_matching_header_passes() {
        let app = app();
        let mut req = Request::builder()
            .method("POST")
            .uri("/echo")
            .header(CSRF_HEADER, "token-abc")
            .body(Body::empty())
            .unwrap();
        req.extensions_mut()
            .insert(SessionCsrf("token-abc".to_string()));
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn post_with_mismatched_header_is_forbidden() {
        let app = app();
        let mut req = Request::builder()
            .method("POST")
            .uri("/echo")
            .header(CSRF_HEADER, "wrong")
            .body(Body::empty())
            .unwrap();
        req.extensions_mut()
            .insert(SessionCsrf("token-abc".to_string()));
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    }
}
