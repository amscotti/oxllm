//! Users page and user CRUD handlers.

use axum::Form;
use axum::extract::{Path, State};
use axum::http::HeaderMap;
use axum::response::{Html, IntoResponse, Response};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::app::{AppState, error_response, is_htmx, proxy_error_response, render_page_or_partial};
use crate::error::AdminError;
use crate::pages::spend::enrich_with_budget_spend;
use crate::pages::{normalize_models_for_display, parse_csv_models, parse_optional_f64};

pub(crate) async fn users_page(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Response, AdminError> {
    let is_htmx = is_htmx(&headers);
    let mut users = match state.proxy.list_users().await {
        Ok(u) => u,
        Err(e) => return Ok(proxy_error_response(&state, is_htmx, &e)),
    };
    normalize_models_for_display(&mut users.items);
    enrich_with_budget_spend(&state.proxy, &mut users.items, "user_id").await;
    let ctx =
        json!({ "session_active": true, "users": users.items, "users_truncated": users.truncated });
    render_page_or_partial(
        &state,
        &headers,
        is_htmx,
        "pages/users.html",
        "partials/users_table.html",
        &ctx,
    )
}

#[derive(Debug, Deserialize, Default)]
pub(crate) struct UserForm {
    #[serde(default)]
    user_name: String,
    #[serde(default)]
    max_budget: String,
    #[serde(default)]
    budget_duration: String,
    /// Comma-separated allowed-model patterns.
    #[serde(default)]
    models: String,
}

impl UserForm {
    fn to_payload(&self) -> Result<Value, String> {
        let mut m = serde_json::Map::new();
        let name = if self.user_name.trim().is_empty() {
            "untitled".to_string()
        } else {
            self.user_name.trim().to_string()
        };
        m.insert("user_name".into(), name.into());
        if let Some(v) = parse_optional_f64(&self.max_budget, "max_budget")? {
            m.insert("max_budget".into(), json!(v));
        }
        if !self.budget_duration.is_empty() {
            m.insert(
                "budget_duration".into(),
                self.budget_duration.clone().into(),
            );
        }
        if let Some(models) = parse_csv_models(&self.models) {
            m.insert("models".into(), json!(models));
        }
        Ok(Value::Object(m))
    }
}

pub(crate) async fn users_create(
    State(state): State<AppState>,
    headers: HeaderMap,
    Form(form): Form<UserForm>,
) -> Result<Response, AdminError> {
    let is_htmx = is_htmx(&headers);
    let payload = match form.to_payload() {
        Ok(p) => p,
        Err(msg) => return Ok(error_response(&state, is_htmx, &msg)),
    };
    if let Err(e) = state.proxy.create_user(payload).await {
        return Ok(proxy_error_response(&state, is_htmx, &e));
    }
    let mut users = match state.proxy.list_users().await {
        Ok(u) => u,
        Err(e) => return Ok(proxy_error_response(&state, is_htmx, &e)),
    };
    normalize_models_for_display(&mut users.items);
    enrich_with_budget_spend(&state.proxy, &mut users.items, "user_id").await;
    let ctx = json!({ "session_active": true, "users": users.items, "users_truncated": users.truncated, "created": "User created." });
    let html = state.templates.render("partials/users_table.html", &ctx)?;
    Ok(Html(html).into_response())
}

pub(crate) async fn users_update(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(user_id): Path<String>,
    Form(form): Form<UserForm>,
) -> Result<Response, AdminError> {
    let is_htmx = is_htmx(&headers);
    let mut body = match form.to_payload() {
        Ok(p) => p,
        Err(msg) => return Ok(error_response(&state, is_htmx, &msg)),
    };
    if let Some(obj) = body.as_object_mut() {
        obj.insert("user_id".into(), json!(user_id));
        obj.insert(
            "models".into(),
            json!(parse_csv_models(&form.models).unwrap_or_default()),
        );
        // Blank means "clear" per the proxy's /user/update contract: null /
        // omission leaves the value unchanged, so the budget needs the
        // explicit clear flag and budget_duration needs an empty string.
        // Create still omits these on empty.
        if !obj.contains_key("max_budget") {
            obj.insert("clear_max_budget".into(), json!(true));
        }
        if form.budget_duration.trim().is_empty() {
            obj.insert("budget_duration".into(), json!(""));
        }
    }
    if let Err(e) = state.proxy.update_user(body).await {
        return Ok(proxy_error_response(&state, is_htmx, &e));
    }
    let mut users = match state.proxy.list_users().await {
        Ok(u) => u,
        Err(e) => return Ok(proxy_error_response(&state, is_htmx, &e)),
    };
    normalize_models_for_display(&mut users.items);
    enrich_with_budget_spend(&state.proxy, &mut users.items, "user_id").await;
    let ctx = json!({ "session_active": true, "users": users.items, "users_truncated": users.truncated, "created": "User updated." });
    let html = state.templates.render("partials/users_table.html", &ctx)?;
    Ok(Html(html).into_response())
}

pub(crate) async fn users_delete(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(user_id): Path<String>,
) -> Result<Response, AdminError> {
    let is_htmx = is_htmx(&headers);
    if let Err(e) = state.proxy.delete_user(&user_id).await {
        return Ok(proxy_error_response(&state, is_htmx, &e));
    }
    let mut users = match state.proxy.list_users().await {
        Ok(u) => u,
        Err(e) => return Ok(proxy_error_response(&state, is_htmx, &e)),
    };
    normalize_models_for_display(&mut users.items);
    enrich_with_budget_spend(&state.proxy, &mut users.items, "user_id").await;
    let ctx = json!({
        "session_active": true,
        "users": users.items,
        "users_truncated": users.truncated,
        "created": format!("User {user_id} deleted."),
    });
    let html = state.templates.render("partials/users_table.html", &ctx)?;
    Ok(Html(html).into_response())
}
