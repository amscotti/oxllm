//! Projects page and project CRUD handlers.

use axum::Form;
use axum::extract::{Path, State};
use axum::http::HeaderMap;
use axum::response::{Html, IntoResponse, Response};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::app::{AppState, error_response, is_htmx, proxy_error_response, render_page_or_partial};
use crate::error::AdminError;
use crate::pages::{parse_optional_f64, parse_optional_i64};
use crate::proxy_client::ListResponse;

pub(crate) async fn projects_page(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Response, AdminError> {
    let is_htmx = is_htmx(&headers);
    let (projects_res, organizations, teams) = fetch_projects_context(&state.proxy).await;
    let projects = match projects_res {
        Ok(p) => p,
        Err(e) => return Ok(proxy_error_response(&state, is_htmx, &e)),
    };
    let ctx = json!({
        "session_active": true,
        "projects": projects.items,
        "projects_truncated": projects.truncated,
        "organizations": organizations.items,
        "organizations_truncated": organizations.truncated,
        "organizations_unavailable": organizations.unavailable,
        "teams": teams.items,
        "teams_truncated": teams.truncated,
        "teams_unavailable": teams.unavailable,
    });
    render_page_or_partial(
        &state,
        &headers,
        is_htmx,
        "pages/projects.html",
        "partials/projects_table.html",
        &ctx,
    )
}

/// Fetch the projects list plus the org/team feeds that populate the edit
/// drawer's association selects. Shared by the page render and every
/// post-mutation table re-render (`projects_create`/`update`/`delete` swap
/// `partials/projects_table.html` back in) so the selects never degrade to
/// "(none)"-only after a swap — which, combined with the clear-on-empty
/// submit contract, would silently strip associations on the next save.
///
/// Fan the independent list calls out concurrently so worst-case latency is
/// a single request-timeout window rather than 3x (matching the dashboard /
/// keys handlers). Missing/partial feeds are labeled, and edit selects retain
/// an existing association even when it isn't among the returned options.
async fn fetch_projects_context(
    proxy: &crate::proxy_client::ProxyClient,
) -> (anyhow::Result<ListResponse>, ListResponse, ListResponse) {
    tokio::join!(
        proxy.list_projects(),
        async {
            proxy.list_organizations().await.unwrap_or_else(|e| {
                tracing::warn!(error = %e, "list_organizations for projects page failed");
                ListResponse::unavailable()
            })
        },
        async {
            proxy.list_teams().await.unwrap_or_else(|e| {
                tracing::warn!(error = %e, "list_teams for projects page failed");
                ListResponse::unavailable()
            })
        },
    )
}

#[derive(Debug, Deserialize, Default)]
pub(crate) struct ProjectForm {
    #[serde(default)]
    project_name: String,
    #[serde(default)]
    org_id: String,
    #[serde(default)]
    team_id: String,
    #[serde(default)]
    max_budget: String,
    #[serde(default)]
    budget_duration: String,
    #[serde(default)]
    rpm: String,
    #[serde(default)]
    tpm: String,
}

impl ProjectForm {
    fn to_payload(&self) -> Result<Value, String> {
        let mut m = serde_json::Map::new();
        let name = if self.project_name.trim().is_empty() {
            "untitled".to_string()
        } else {
            self.project_name.trim().to_string()
        };
        m.insert("project_name".into(), name.into());
        if !self.org_id.is_empty() {
            m.insert("org_id".into(), self.org_id.clone().into());
        }
        if !self.team_id.is_empty() {
            m.insert("team_id".into(), self.team_id.clone().into());
        }
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
        Ok(Value::Object(m))
    }
}

pub(crate) async fn projects_create(
    State(state): State<AppState>,
    headers: HeaderMap,
    Form(form): Form<ProjectForm>,
) -> Result<Response, AdminError> {
    let is_htmx = is_htmx(&headers);
    let payload = match form.to_payload() {
        Ok(p) => p,
        Err(msg) => return Ok(error_response(&state, is_htmx, &msg)),
    };
    if let Err(e) = state.proxy.create_project(payload).await {
        return Ok(proxy_error_response(&state, is_htmx, &e));
    }
    let (projects_res, organizations, teams) = fetch_projects_context(&state.proxy).await;
    let projects = match projects_res {
        Ok(p) => p,
        Err(e) => return Ok(proxy_error_response(&state, is_htmx, &e)),
    };
    let ctx = json!({
        "session_active": true,
        "projects": projects.items,
        "projects_truncated": projects.truncated,
        "organizations": organizations.items,
        "organizations_truncated": organizations.truncated,
        "organizations_unavailable": organizations.unavailable,
        "teams": teams.items,
        "teams_truncated": teams.truncated,
        "teams_unavailable": teams.unavailable,
        "created": "Project created.",
    });
    let html = state
        .templates
        .render("partials/projects_table.html", &ctx)?;
    Ok(Html(html).into_response())
}

pub(crate) async fn projects_update(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(project_id): Path<String>,
    Form(form): Form<ProjectForm>,
) -> Result<Response, AdminError> {
    let is_htmx = is_htmx(&headers);
    let mut body = match form.to_payload() {
        Ok(p) => p,
        Err(msg) => return Ok(error_response(&state, is_htmx, &msg)),
    };
    if let Some(obj) = body.as_object_mut() {
        obj.insert("project_id".into(), json!(project_id));
        // Blank means "clear" per the proxy's /project/update contract:
        // associations need the explicit clear_org_id/clear_team_id flags,
        // caps need clear_max_budget/clear_rpm/clear_tpm, and
        // budget_duration needs an empty string — null / omission leaves
        // every one of them unchanged. Create still omits these on empty.
        if form.team_id.trim().is_empty() {
            obj.insert("clear_team_id".into(), json!(true));
        }
        if form.org_id.trim().is_empty() {
            obj.insert("clear_org_id".into(), json!(true));
        }
        if form.rpm.trim().is_empty() {
            obj.insert("clear_rpm".into(), json!(true));
        }
        if form.tpm.trim().is_empty() {
            obj.insert("clear_tpm".into(), json!(true));
        }
        if !obj.contains_key("max_budget") {
            obj.insert("clear_max_budget".into(), json!(true));
        }
        if form.budget_duration.trim().is_empty() {
            obj.insert("budget_duration".into(), json!(""));
        }
    }
    if let Err(e) = state.proxy.update_project(body).await {
        return Ok(proxy_error_response(&state, is_htmx, &e));
    }
    let (projects_res, organizations, teams) = fetch_projects_context(&state.proxy).await;
    let projects = match projects_res {
        Ok(p) => p,
        Err(e) => return Ok(proxy_error_response(&state, is_htmx, &e)),
    };
    let ctx = json!({
        "session_active": true,
        "projects": projects.items,
        "projects_truncated": projects.truncated,
        "organizations": organizations.items,
        "organizations_truncated": organizations.truncated,
        "organizations_unavailable": organizations.unavailable,
        "teams": teams.items,
        "teams_truncated": teams.truncated,
        "teams_unavailable": teams.unavailable,
        "created": "Project updated.",
    });
    let html = state
        .templates
        .render("partials/projects_table.html", &ctx)?;
    Ok(Html(html).into_response())
}

pub(crate) async fn projects_delete(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(project_id): Path<String>,
) -> Result<Response, AdminError> {
    let is_htmx = is_htmx(&headers);
    if let Err(e) = state.proxy.delete_project(&project_id).await {
        return Ok(proxy_error_response(&state, is_htmx, &e));
    }
    let (projects_res, organizations, teams) = fetch_projects_context(&state.proxy).await;
    let projects = match projects_res {
        Ok(p) => p,
        Err(e) => return Ok(proxy_error_response(&state, is_htmx, &e)),
    };
    let ctx = json!({
        "session_active": true,
        "projects": projects.items,
        "projects_truncated": projects.truncated,
        "organizations": organizations.items,
        "organizations_truncated": organizations.truncated,
        "organizations_unavailable": organizations.unavailable,
        "teams": teams.items,
        "teams_truncated": teams.truncated,
        "teams_unavailable": teams.unavailable,
        "created": format!("Project {project_id} deleted."),
    });
    let html = state
        .templates
        .render("partials/projects_table.html", &ctx)?;
    Ok(Html(html).into_response())
}
