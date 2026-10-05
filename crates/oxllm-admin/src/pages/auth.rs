//! Login/logout, health check, and the index redirect.

use axum::Form;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{Html, IntoResponse, Response};
use serde::Deserialize;
use serde_json::json;

use crate::app::AppState;
use crate::auth::{
    clear_cookie_header, client_ip, create_login_csrf, create_session, login_csrf_ok,
    session_cookie_header,
};
use crate::error::AdminError;

/// Log out. POST-only so the CSRF middleware covers it; a plain GET (e.g. a
/// cross-site `<img src="/logout">`) cannot force a logout.
pub(crate) async fn logout(State(state): State<AppState>) -> Response {
    let clear = clear_cookie_header(state.config.cookie_secure);
    (
        StatusCode::SEE_OTHER,
        [("location", "/login"), ("set-cookie", clear.as_str())],
    )
        .into_response()
}

/// Re-render the login page with a fresh pre-login CSRF token (and its
/// cookie) and the given error message.
fn login_error_page(
    state: &AppState,
    message: &str,
    status: StatusCode,
) -> Result<Response, AdminError> {
    let (token, cookie) =
        create_login_csrf(&state.config.session_secret, state.config.cookie_secure);
    let html = state.templates.render(
        "login.html",
        &json!({
            "error": message,
            "login_csrf_token": token,
            "cookie_secure": state.config.cookie_secure,
        }),
    )?;
    Ok((status, [("set-cookie", cookie.as_str())], Html(html)).into_response())
}

pub(crate) async fn health() -> impl IntoResponse {
    (
        StatusCode::OK,
        [("content-type", "application/json")],
        r#"{"status":"healthy","service":"oxllm-admin"}"#,
    )
}

pub(crate) async fn login_form(State(state): State<AppState>) -> Result<Response, AdminError> {
    let (token, cookie) =
        create_login_csrf(&state.config.session_secret, state.config.cookie_secure);
    let html = state.templates.render(
        "login.html",
        &json!({
            "error": null,
            "login_csrf_token": token,
            "cookie_secure": state.config.cookie_secure,
        }),
    )?;
    Ok((
        StatusCode::OK,
        [("set-cookie", cookie.as_str())],
        Html(html),
    )
        .into_response())
}

#[derive(Debug, Deserialize)]
pub(crate) struct LoginForm {
    password: String,
    /// Pre-login CSRF token echoed from the login form's hidden field.
    #[serde(default)]
    csrf_token: String,
}

/// Optional connection peer. Present when the server is started with
/// `into_make_service_with_connect_info`; absent in unit tests that drive
/// the router via `oneshot`.
pub(crate) struct OptionalPeer(Option<std::net::SocketAddr>);

impl<S> axum::extract::FromRequestParts<S> for OptionalPeer
where
    S: Send + Sync,
{
    type Rejection = std::convert::Infallible;

    async fn from_request_parts(
        parts: &mut axum::http::request::Parts,
        _state: &S,
    ) -> Result<Self, Self::Rejection> {
        let addr = parts
            .extensions
            .get::<axum::extract::ConnectInfo<std::net::SocketAddr>>()
            .map(|c| c.0);
        Ok(OptionalPeer(addr))
    }
}

pub(crate) async fn login(
    State(state): State<AppState>,
    headers: HeaderMap,
    peer: OptionalPeer,
    Form(form): Form<LoginForm>,
) -> Result<Response, AdminError> {
    // Trust the connection peer unless it lands inside the configured
    // `trusted_proxies` allowlist — in which case honour the rightmost
    // non-trusted `X-Forwarded-For` entry so per-client attribution survives a
    // reverse proxy / load balancer topology.
    let trusted = crate::auth::parse_trusted_proxies(&state.config.trusted_proxies);
    let ip = client_ip(&headers, peer.0, &trusted);

    // Pre-login CSRF check: without it, a cross-site page can force the
    // victim into an attacker-known/blank session (login CSRF). The token
    // is minted by `login_form` in a signed cookie and echoed by the form.
    if !login_csrf_ok(&headers, &state.config.session_secret, &form.csrf_token) {
        return login_error_page(
            &state,
            "Session expired or the login cookie was blocked — reload this page and try again.",
            StatusCode::FORBIDDEN,
        );
    }

    if state.login_brake.is_blocked(&ip) {
        let retry = state.login_brake.retry_after(&ip).max(1);
        tracing::warn!(ip = %ip, retry_after = retry, "login blocked by rate limiter");
        return Ok((
            StatusCode::TOO_MANY_REQUESTS,
            [
                ("content-type", "text/html; charset=utf-8"),
                ("retry-after", retry.to_string().as_str()),
            ],
            Html("Too many login attempts. Try again later.".to_string()),
        )
            .into_response());
    }

    let authorized = match state.config.login_password.as_deref() {
        Some(expected) => crate::auth::ct_eq(form.password.as_bytes(), expected.as_bytes()),
        None => true,
    };

    if !authorized {
        state.login_brake.record_failure(&ip);
        let status = if state.login_brake.is_blocked(&ip) {
            StatusCode::TOO_MANY_REQUESTS
        } else {
            StatusCode::UNAUTHORIZED
        };
        return login_error_page(&state, "Incorrect password", status);
    }

    let session = create_session(&state.config.session_secret, state.config.session_ttl_hours);
    let header = session_cookie_header(&session, state.config.cookie_secure);
    Ok((
        StatusCode::SEE_OTHER,
        [("location", "/dashboard"), ("set-cookie", header.as_str())],
    )
        .into_response())
}

pub(crate) async fn index() -> impl IntoResponse {
    (StatusCode::SEE_OTHER, [("location", "/dashboard")])
}
