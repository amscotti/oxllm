//! Axum application: router, shared state, and cross-page render helpers.
//! Page handlers live under `crate::pages`.

use std::sync::Arc;

use axum::Router;
use axum::http::{HeaderMap, StatusCode};
use axum::middleware;
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use serde_json::{Value, json};
use tower_cookies::CookieManagerLayer;

use crate::auth::{LoginBrake, csrf_token_from_headers, require_session};
use crate::config::AdminConfig;
use crate::error::AdminError;
use crate::middleware::{require_csrf, security_headers};
use crate::pages::{
    auth, calculator, dashboard, keys, models, orgs, projects, spend, teams, users,
};
use crate::proxy_client::ProxyClient;
use crate::templates::Templates;

#[cfg(test)]
mod tests;

/// Shared application state cloned into every request.
#[derive(Clone)]
pub struct AppState {
    pub config: Arc<AdminConfig>,
    pub proxy: Arc<ProxyClient>,
    pub templates: Arc<Templates>,
    pub login_brake: Arc<LoginBrake>,
}

/// Build the configured Axum router.
pub fn build(state: AppState) -> Router {
    let public = Router::new()
        .route("/health", get(auth::health))
        .route("/login", get(auth::login_form).post(auth::login))
        .route("/static/{*path}", get(crate::assets::serve));

    let protected = Router::new()
        .route("/", get(auth::index))
        .route("/logout", post(auth::logout))
        .route("/dashboard", get(dashboard::dashboard))
        .route("/keys", get(keys::keys_page))
        .route("/keys/create", post(keys::keys_create))
        .route("/keys/{key_id}/delete", post(keys::keys_delete))
        .route("/keys/{key_id}/enable", post(keys::keys_enable))
        .route("/keys/{key_id}/purge", post(keys::keys_purge))
        .route("/keys/{key_id}/rotate", post(keys::keys_rotate))
        .route("/keys/{key_id}/update", post(keys::keys_update))
        .route("/teams", get(teams::teams_page))
        .route("/teams/create", post(teams::teams_create))
        .route("/teams/{team_id}/update", post(teams::teams_update))
        .route("/teams/{team_id}/delete", post(teams::teams_delete))
        .route("/users", get(users::users_page))
        .route("/users/create", post(users::users_create))
        .route("/users/{user_id}/update", post(users::users_update))
        .route("/users/{user_id}/delete", post(users::users_delete))
        .route("/organizations", get(orgs::organizations_page))
        .route("/organizations/create", post(orgs::organizations_create))
        .route(
            "/organizations/{org_id}/update",
            post(orgs::organizations_update),
        )
        .route(
            "/organizations/{org_id}/delete",
            post(orgs::organizations_delete),
        )
        .route("/projects", get(projects::projects_page))
        .route("/projects/create", post(projects::projects_create))
        .route(
            "/projects/{project_id}/update",
            post(projects::projects_update),
        )
        .route(
            "/projects/{project_id}/delete",
            post(projects::projects_delete),
        )
        .route("/spend", get(spend::spend_page))
        .route("/spend/export.csv", get(spend::spend_export_csv))
        .route("/calculator", get(calculator::calculator_page))
        .route("/calculator/run", post(calculator::calculator_run))
        .route("/models", get(models::models_page))
        .route("/models/create", post(models::models_create))
        .route("/models/{model_id}/delete", post(models::models_delete))
        .route_layer(middleware::from_fn_with_state(state.clone(), require_csrf))
        .route_layer(middleware::from_fn_with_state(
            state.clone(),
            require_session,
        ));

    public
        .merge(protected)
        .layer(CookieManagerLayer::new())
        .layer(middleware::from_fn(security_headers))
        .with_state(state)
}

pub(crate) fn is_htmx(headers: &HeaderMap) -> bool {
    headers
        .get("HX-Request")
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.eq_ignore_ascii_case("true"))
}

pub(crate) fn is_proxy_unauthorized(err: &anyhow::Error) -> bool {
    // Match the typed `ProxyStatusError` carried in the chain rather than
    // grepping the error string: a 5xx body containing the literal "status
    // 401" must NOT trigger a spurious login redirect, and a future rewording
    // of the bail! message must not silently break the redirect.
    err.downcast_ref::<crate::proxy_client::ProxyStatusError>()
        .map(|e| e.status == 401)
        .unwrap_or(false)
}

pub(crate) fn htmx_redirect(location: &str) -> Response {
    (
        StatusCode::OK,
        [("HX-Redirect", location)],
        Html(String::new()),
    )
        .into_response()
}

pub(crate) fn error_response(state: &AppState, is_htmx: bool, message: &str) -> Response {
    // Both the HTMX and non-HTMX paths render the same error alert partial.
    // The HTMX path must stay HTTP 200 so htmx swaps the fragment; the
    // full-page path returns a real error status so failures are visible to
    // the browser and to monitoring instead of a silently-OK error page.
    error_response_with_status(state, is_htmx, message, StatusCode::BAD_REQUEST)
}

pub(crate) fn error_response_with_status(
    state: &AppState,
    is_htmx: bool,
    message: &str,
    status: StatusCode,
) -> Response {
    let ctx = json!({ "session_active": true, "flash": message, "message": message });
    let status = if is_htmx { StatusCode::OK } else { status };
    match state.templates.render("partials/error_alert.html", &ctx) {
        Ok(html) if is_htmx => (
            status,
            [
                ("HX-Retarget", "#flash"),
                ("HX-Reswap", "innerHTML"),
                ("X-Oxllm-Error", "true"),
            ],
            Html(html),
        )
            .into_response(),
        Ok(html) => (status, Html(html)).into_response(),
        Err(e) => {
            tracing::warn!(error = %e, "failed to render error_alert");
            (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response()
        }
    }
}

pub(crate) fn proxy_error_response(
    state: &AppState,
    is_htmx: bool,
    err: &anyhow::Error,
) -> Response {
    if is_proxy_unauthorized(err) {
        if is_htmx {
            return htmx_redirect("/login");
        }
        // Non-HTMX (plain browser navigation): an HX-Redirect header would be
        // ignored, so send a real redirect to the login page.
        return (StatusCode::SEE_OTHER, [("location", "/login")]).into_response();
    }
    tracing::warn!(error = %err, "proxy call failed");
    error_response_with_status(state, is_htmx, &err.to_string(), StatusCode::BAD_GATEWAY)
}

pub(crate) fn render_page_or_partial(
    state: &AppState,
    headers: &HeaderMap,
    is_htmx: bool,
    page: &str,
    partial: &str,
    ctx: &Value,
) -> Result<Response, AdminError> {
    let name = if is_htmx { partial } else { page };
    if is_htmx {
        let html = state.templates.render(name, ctx)?;
        Ok(Html(html).into_response())
    } else {
        let csrf =
            csrf_token_from_headers(headers, &state.config.session_secret).unwrap_or_default();
        let mut map = match ctx {
            Value::Object(m) => m.clone(),
            _ => serde_json::Map::new(),
        };
        map.insert("csrf_token".to_string(), json!(csrf));
        let html = state.templates.render(name, &Value::Object(map))?;
        Ok(Html(html).into_response())
    }
}

#[cfg(test)]
pub(crate) fn test_state_for_middleware() -> AppState {
    use crate::auth::LoginBrake;
    AppState {
        config: Arc::new(AdminConfig {
            listen: "127.0.0.1:0".to_string(),
            proxy_url: "http://127.0.0.1:0".to_string(),
            master_key: "sk-test".to_string(),
            session_secret: "unit-test-secret-value".to_string(),
            session_ttl_hours: 1,
            login_password: None,
            trusted_proxies: Vec::new(),
            cookie_secure: false,
            login_max_fails: 5,
            login_window_secs: 60,
            login_block_secs: 300,
        }),
        proxy: Arc::new(ProxyClient::new("http://127.0.0.1:0", "sk-test")),
        templates: Arc::new(Templates::new()),
        login_brake: Arc::new(LoginBrake::new(5, 60, 300)),
    }
}
