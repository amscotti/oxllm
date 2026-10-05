//! Teams page and team CRUD handlers.

use axum::Form;
use axum::extract::{Path, State};
use axum::http::HeaderMap;
use axum::response::{Html, IntoResponse, Response};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::app::{AppState, error_response, is_htmx, proxy_error_response, render_page_or_partial};
use crate::error::AdminError;
use crate::pages::spend::enrich_with_budget_spend;
use crate::pages::{
    normalize_models_for_display, parse_csv_models, parse_optional_f64, parse_optional_i64,
};

pub(crate) async fn teams_page(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Response, AdminError> {
    let is_htmx = is_htmx(&headers);
    let mut teams = match state.proxy.list_teams().await {
        Ok(t) => t,
        Err(e) => return Ok(proxy_error_response(&state, is_htmx, &e)),
    };
    normalize_models_for_display(&mut teams.items);
    enrich_with_budget_spend(&state.proxy, &mut teams.items, "team_id").await;
    let ctx =
        json!({ "session_active": true, "teams": teams.items, "teams_truncated": teams.truncated });
    render_page_or_partial(
        &state,
        &headers,
        is_htmx,
        "pages/teams.html",
        "partials/teams_table.html",
        &ctx,
    )
}

#[derive(Debug, Deserialize, Default)]
pub(crate) struct TeamForm {
    #[serde(default)]
    team_name: String,
    #[serde(default)]
    max_budget: String,
    #[serde(default)]
    budget_duration: String,
    #[serde(default)]
    rpm: String,
    #[serde(default)]
    tpm: String,
    /// Comma-separated allowed-model patterns.
    #[serde(default)]
    models: String,
}

impl TeamForm {
    fn to_payload(&self) -> Result<Value, String> {
        let mut m = serde_json::Map::new();
        let name = if self.team_name.trim().is_empty() {
            "untitled".to_string()
        } else {
            self.team_name.trim().to_string()
        };
        m.insert("team_name".into(), name.into());
        if let Some(v) = parse_optional_f64(&self.max_budget, "max_budget")? {
            m.insert("max_budget".into(), json!(v));
        }
        if !self.budget_duration.is_empty() {
            m.insert(
                "budget_duration".into(),
                self.budget_duration.clone().into(),
            );
        }
        if let Some(v) = parse_optional_i64(&self.rpm, "rpm")? {
            m.insert("rpm".into(), json!(v));
        }
        if let Some(v) = parse_optional_i64(&self.tpm, "tpm")? {
            m.insert("tpm".into(), json!(v));
        }
        if let Some(models) = parse_csv_models(&self.models) {
            m.insert("models".into(), json!(models));
        }
        Ok(Value::Object(m))
    }
}

pub(crate) async fn teams_create(
    State(state): State<AppState>,
    headers: HeaderMap,
    Form(form): Form<TeamForm>,
) -> Result<Response, AdminError> {
    let is_htmx = is_htmx(&headers);
    let payload = match form.to_payload() {
        Ok(p) => p,
        Err(msg) => return Ok(error_response(&state, is_htmx, &msg)),
    };
    if let Err(e) = state.proxy.create_team(payload).await {
        return Ok(proxy_error_response(&state, is_htmx, &e));
    }
    let mut teams = match state.proxy.list_teams().await {
        Ok(t) => t,
        Err(e) => return Ok(proxy_error_response(&state, is_htmx, &e)),
    };
    normalize_models_for_display(&mut teams.items);
    enrich_with_budget_spend(&state.proxy, &mut teams.items, "team_id").await;
    let ctx = json!({ "session_active": true, "teams": teams.items, "teams_truncated": teams.truncated, "created": "Team created." });
    let html = state.templates.render("partials/teams_table.html", &ctx)?;
    Ok(Html(html).into_response())
}

pub(crate) async fn teams_update(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(team_id): Path<String>,
    Form(form): Form<TeamForm>,
) -> Result<Response, AdminError> {
    let is_htmx = is_htmx(&headers);
    let mut body = match form.to_payload() {
        Ok(p) => p,
        Err(msg) => return Ok(error_response(&state, is_htmx, &msg)),
    };
    if let Some(obj) = body.as_object_mut() {
        obj.insert("team_id".into(), json!(team_id));
        // Always send models so an empty input clears the field.
        obj.insert(
            "models".into(),
            json!(parse_csv_models(&form.models).unwrap_or_default()),
        );
        // Blank means "clear" per the proxy's /team/update contract: null /
        // omission leaves the value unchanged, so caps need the explicit
        // clear_* flags and budget_duration needs an empty string. Create
        // still omits these on empty.
        if form.rpm.trim().is_empty() {
            obj.insert("clear_rpm".into(), json!(true));
        }
        if form.tpm.trim().is_empty() {
            obj.insert("clear_tpm".into(), json!(true));
        }
        // `to_payload` only inserts max_budget when the field parsed to a
        // valid number, so its absence here means "blanked -> clear".
        if !obj.contains_key("max_budget") {
            obj.insert("clear_max_budget".into(), json!(true));
        }
        if form.budget_duration.trim().is_empty() {
            obj.insert("budget_duration".into(), json!(""));
        }
    }
    if let Err(e) = state.proxy.update_team(body).await {
        return Ok(proxy_error_response(&state, is_htmx, &e));
    }
    let mut teams = match state.proxy.list_teams().await {
        Ok(t) => t,
        Err(e) => return Ok(proxy_error_response(&state, is_htmx, &e)),
    };
    normalize_models_for_display(&mut teams.items);
    enrich_with_budget_spend(&state.proxy, &mut teams.items, "team_id").await;
    let ctx = json!({ "session_active": true, "teams": teams.items, "teams_truncated": teams.truncated, "created": "Team updated." });
    let html = state.templates.render("partials/teams_table.html", &ctx)?;
    Ok(Html(html).into_response())
}

pub(crate) async fn teams_delete(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(team_id): Path<String>,
) -> Result<Response, AdminError> {
    let is_htmx = is_htmx(&headers);
    if let Err(e) = state.proxy.delete_team(&team_id).await {
        return Ok(proxy_error_response(&state, is_htmx, &e));
    }
    let mut teams = match state.proxy.list_teams().await {
        Ok(t) => t,
        Err(e) => return Ok(proxy_error_response(&state, is_htmx, &e)),
    };
    normalize_models_for_display(&mut teams.items);
    enrich_with_budget_spend(&state.proxy, &mut teams.items, "team_id").await;
    let ctx = json!({
        "session_active": true,
        "teams": teams.items,
        "teams_truncated": teams.truncated,
        "created": format!("Team {team_id} deleted."),
    });
    let html = state.templates.render("partials/teams_table.html", &ctx)?;
    Ok(Html(html).into_response())
}
