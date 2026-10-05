//! API keys page and key lifecycle handlers.

use axum::Form;
use axum::extract::{Path, State};
use axum::http::HeaderMap;
use axum::response::{Html, IntoResponse, Response};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::app::error_response;
use crate::app::{AppState, is_htmx, proxy_error_response, render_page_or_partial};
use crate::error::AdminError;
use crate::pages::spend::enrich_with_budget_spend;
use crate::pages::{
    normalize_models_for_display, parse_csv_models, parse_optional_f64, parse_optional_i64,
};
use crate::proxy_client::ListResponse;

pub(crate) async fn keys_page(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Response, AdminError> {
    let is_htmx = is_htmx(&headers);
    // Fan the independent list calls out concurrently so worst-case latency
    // is a single request-timeout window rather than 5x (matching the
    // dashboard handler). `list_keys` surfaces its error; the dropdown feeds
    // degrade gracefully to empty on failure.
    let (keys_res, mut teams, mut users, mut organizations, mut projects) = tokio::join!(
        state.proxy.list_keys(),
        async {
            state.proxy.list_teams().await.unwrap_or_else(|e| {
                tracing::warn!(error = %e, "list_teams for key form failed");
                ListResponse::unavailable()
            })
        },
        async {
            state.proxy.list_users().await.unwrap_or_else(|e| {
                tracing::warn!(error = %e, "list_users for key form failed");
                ListResponse::unavailable()
            })
        },
        async {
            state.proxy.list_organizations().await.unwrap_or_else(|e| {
                tracing::warn!(error = %e, "list_organizations for key form failed");
                ListResponse::unavailable()
            })
        },
        async {
            state.proxy.list_projects().await.unwrap_or_else(|e| {
                tracing::warn!(error = %e, "list_projects for key form failed");
                ListResponse::unavailable()
            })
        },
    );
    let mut keys = match keys_res {
        Ok(k) => k,
        Err(e) => return Ok(proxy_error_response(&state, is_htmx, &e)),
    };
    normalize_models_for_display(&mut keys.items);
    normalize_models_for_display(&mut teams.items);
    normalize_models_for_display(&mut users.items);
    normalize_models_for_display(&mut organizations.items);
    normalize_models_for_display(&mut projects.items);
    enrich_with_budget_spend(&state.proxy, &mut keys.items, "key_id").await;
    let ctx = json!({
        "session_active": true,
        "keys": keys.items,
        "keys_truncated": keys.truncated,
        "teams": teams.items,
        "teams_truncated": teams.truncated,
        "teams_unavailable": teams.unavailable,
        "users": users.items,
        "users_truncated": users.truncated,
        "users_unavailable": users.unavailable,
        "organizations": organizations.items,
        "organizations_truncated": organizations.truncated,
        "organizations_unavailable": organizations.unavailable,
        "projects": projects.items,
        "projects_truncated": projects.truncated,
        "projects_unavailable": projects.unavailable,
    });
    render_page_or_partial(
        &state,
        &headers,
        is_htmx,
        "pages/keys.html",
        "partials/keys_table.html",
        &ctx,
    )
}

#[derive(Debug, Deserialize, Default)]
pub(crate) struct KeyForm {
    #[serde(default)]
    key_alias: String,
    #[serde(default)]
    team_id: String,
    #[serde(default)]
    user_id: String,
    #[serde(default)]
    org_id: String,
    #[serde(default)]
    project_id: String,
    #[serde(default)]
    max_budget: String,
    #[serde(default)]
    budget_duration: String,
    #[serde(default)]
    rpm: String,
    #[serde(default)]
    tpm: String,
    /// Comma-separated allowed-model patterns (e.g. `gpt-4o, claude-*`).
    #[serde(default)]
    models: String,
}

impl KeyForm {
    fn to_payload(&self) -> Result<Value, String> {
        let mut m = serde_json::Map::new();
        if !self.key_alias.is_empty() {
            m.insert("key_alias".into(), self.key_alias.clone().into());
        }
        if !self.team_id.is_empty() {
            m.insert("team_id".into(), self.team_id.clone().into());
        }
        if !self.user_id.is_empty() {
            m.insert("user_id".into(), self.user_id.clone().into());
        }
        if !self.org_id.is_empty() {
            m.insert("org_id".into(), self.org_id.clone().into());
        }
        if !self.project_id.is_empty() {
            m.insert("project_id".into(), self.project_id.clone().into());
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
        if let Some(models) = parse_csv_models(&self.models) {
            m.insert("models".into(), json!(models));
        }
        Ok(Value::Object(m))
    }
}

pub(crate) async fn keys_create(
    State(state): State<AppState>,
    headers: HeaderMap,
    Form(form): Form<KeyForm>,
) -> Result<Response, AdminError> {
    let is_htmx = is_htmx(&headers);
    let payload = match form.to_payload() {
        Ok(p) => p,
        Err(msg) => return Ok(error_response(&state, is_htmx, &msg)),
    };
    let resp = match state.proxy.generate_key(payload).await {
        Ok(v) => v,
        Err(e) => return Ok(proxy_error_response(&state, is_htmx, &e)),
    };

    let key = resp
        .get("key")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let key_id = resp
        .get("key_id")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    if key.is_empty() {
        return Ok(proxy_error_response(
            &state,
            is_htmx,
            &anyhow::anyhow!("proxy returned no key in response"),
        ));
    }

    let ctx = json!({ "key": key, "key_id": key_id });
    let html = state.templates.render("partials/key_created.html", &ctx)?;
    Ok((
        [(
            axum::http::header::CACHE_CONTROL,
            axum::http::HeaderValue::from_static("no-store, private"),
        )],
        Html(html),
    )
        .into_response())
}

pub(crate) async fn keys_delete(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(key_id): Path<String>,
) -> Result<Response, AdminError> {
    let is_htmx = is_htmx(&headers);
    if let Err(e) = state.proxy.delete_key(&key_id).await {
        return Ok(proxy_error_response(&state, is_htmx, &e));
    }
    // Soft-delete keeps the row in the DB as inactive. Refresh the table so the
    // UI shows Disabled instead of removing a row that reappears on reload.
    let mut keys = match state.proxy.list_keys().await {
        Ok(k) => k,
        Err(e) => return Ok(proxy_error_response(&state, is_htmx, &e)),
    };
    normalize_models_for_display(&mut keys.items);
    enrich_with_budget_spend(&state.proxy, &mut keys.items, "key_id").await;
    let ctx = json!({
        "session_active": true,
        "keys": keys.items,
        "keys_truncated": keys.truncated,
        "created": format!("Key {key_id} disabled."),
    });
    let html = state.templates.render("partials/keys_table.html", &ctx)?;
    Ok(Html(html).into_response())
}

pub(crate) async fn keys_enable(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(key_id): Path<String>,
) -> Result<Response, AdminError> {
    let is_htmx = is_htmx(&headers);
    if let Err(e) = state.proxy.enable_key(&key_id).await {
        return Ok(proxy_error_response(&state, is_htmx, &e));
    }
    let mut keys = match state.proxy.list_keys().await {
        Ok(k) => k,
        Err(e) => return Ok(proxy_error_response(&state, is_htmx, &e)),
    };
    normalize_models_for_display(&mut keys.items);
    enrich_with_budget_spend(&state.proxy, &mut keys.items, "key_id").await;
    let ctx = json!({
        "session_active": true,
        "keys": keys.items,
        "keys_truncated": keys.truncated,
        "created": format!("Key {key_id} re-enabled."),
    });
    let html = state.templates.render("partials/keys_table.html", &ctx)?;
    Ok(Html(html).into_response())
}

pub(crate) async fn keys_purge(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(key_id): Path<String>,
) -> Result<Response, AdminError> {
    let is_htmx = is_htmx(&headers);
    if let Err(e) = state.proxy.purge_key(&key_id).await {
        return Ok(proxy_error_response(&state, is_htmx, &e));
    }
    let mut keys = match state.proxy.list_keys().await {
        Ok(k) => k,
        Err(e) => return Ok(proxy_error_response(&state, is_htmx, &e)),
    };
    normalize_models_for_display(&mut keys.items);
    enrich_with_budget_spend(&state.proxy, &mut keys.items, "key_id").await;
    let ctx = json!({
        "session_active": true,
        "keys": keys.items,
        "keys_truncated": keys.truncated,
        "created": format!("Key {key_id} permanently removed."),
    });
    let html = state.templates.render("partials/keys_table.html", &ctx)?;
    Ok(Html(html).into_response())
}

pub(crate) async fn keys_rotate(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(key_id): Path<String>,
) -> Result<Response, AdminError> {
    let is_htmx = is_htmx(&headers);
    let resp = match state.proxy.rotate_key(&key_id).await {
        Ok(v) => v,
        Err(e) => return Ok(proxy_error_response(&state, is_htmx, &e)),
    };
    let key = resp
        .get("key")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    if key.is_empty() {
        return Ok(proxy_error_response(
            &state,
            is_htmx,
            &anyhow::anyhow!("proxy returned no key on rotate"),
        ));
    }
    let ctx = json!({
        "key": key,
        "key_id": key_id,
        "rotated": true,
    });
    let html = state.templates.render("partials/key_created.html", &ctx)?;
    Ok((
        [(
            axum::http::header::CACHE_CONTROL,
            axum::http::HeaderValue::from_static("no-store, private"),
        )],
        Html(html),
    )
        .into_response())
}

#[derive(Debug, Deserialize, Default)]
pub(crate) struct KeyUpdateForm {
    #[serde(default)]
    key_alias: String,
    #[serde(default)]
    team_id: String,
    #[serde(default)]
    user_id: String,
    #[serde(default)]
    org_id: String,
    #[serde(default)]
    project_id: String,
    #[serde(default)]
    max_budget: String,
    #[serde(default)]
    budget_duration: String,
    #[serde(default)]
    rpm: String,
    #[serde(default)]
    tpm: String,
    /// Comma-separated allowed-model patterns. Empty string clears the field.
    #[serde(default)]
    models: String,
}

pub(crate) async fn keys_update(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(key_id): Path<String>,
    Form(form): Form<KeyUpdateForm>,
) -> Result<Response, AdminError> {
    let is_htmx = is_htmx(&headers);
    // Validate numeric fields up front: a non-numeric value must be reported,
    // not silently dropped (which previously kept the old budget/rpm/tpm
    // while still claiming "Key updated.").
    let max_budget = match parse_optional_f64(&form.max_budget, "max_budget") {
        Ok(v) => v,
        Err(msg) => return Ok(error_response(&state, is_htmx, &msg)),
    };
    let rpm = match parse_optional_i64(&form.rpm, "rpm") {
        Ok(v) => v,
        Err(msg) => return Ok(error_response(&state, is_htmx, &msg)),
    };
    let tpm = match parse_optional_i64(&form.tpm, "tpm") {
        Ok(v) => v,
        Err(msg) => return Ok(error_response(&state, is_htmx, &msg)),
    };
    let mut body = serde_json::Map::new();
    body.insert("key_id".into(), json!(key_id));
    body.insert("key_alias".into(), json!(form.key_alias));
    // Association clears follow the proxy's /key/update contract: an empty
    // string clears the field, while JSON null / omission leaves it
    // unchanged. Sending null here would silently keep the old association
    // while the UI reports "Key updated.".
    if form.team_id.trim().is_empty() {
        body.insert("team_id".into(), json!(""));
    } else {
        body.insert("team_id".into(), json!(form.team_id));
    }
    if form.user_id.trim().is_empty() {
        body.insert("user_id".into(), json!(""));
    } else {
        body.insert("user_id".into(), json!(form.user_id));
    }
    if form.org_id.trim().is_empty() {
        body.insert("org_id".into(), json!(""));
    } else {
        body.insert("org_id".into(), json!(form.org_id));
    }
    if form.project_id.trim().is_empty() {
        body.insert("project_id".into(), json!(""));
    } else {
        body.insert("project_id".into(), json!(form.project_id));
    }
    // Budget/rate caps: the proxy treats null/omitted as leave-unchanged;
    // removing a cap requires the explicit clear_* flags. A blanked field
    // in the edit drawer means "clear", not "keep".
    match max_budget {
        Some(v) => {
            body.insert("max_budget".into(), json!(v));
        }
        None => {
            body.insert("clear_max_budget".into(), json!(true));
        }
    }
    body.insert("budget_duration".into(), json!(form.budget_duration));
    match rpm {
        Some(v) => {
            body.insert("rpm".into(), json!(v));
        }
        None => {
            body.insert("clear_rpm".into(), json!(true));
        }
    }
    match tpm {
        Some(v) => {
            body.insert("tpm".into(), json!(v));
        }
        None => {
            body.insert("clear_tpm".into(), json!(true));
        }
    }
    // Always send models (even if empty) so the update can clear the field.
    body.insert(
        "models".into(),
        json!(parse_csv_models(&form.models).unwrap_or_default()),
    );
    if let Err(e) = state.proxy.update_key(Value::Object(body)).await {
        return Ok(proxy_error_response(&state, is_htmx, &e));
    }
    let mut keys = match state.proxy.list_keys().await {
        Ok(k) => k,
        Err(e) => return Ok(proxy_error_response(&state, is_htmx, &e)),
    };
    normalize_models_for_display(&mut keys.items);
    enrich_with_budget_spend(&state.proxy, &mut keys.items, "key_id").await;
    let ctx = json!({
        "session_active": true,
        "keys": keys.items,
        "keys_truncated": keys.truncated,
        "created": "Key updated.",
    });
    let html = state.templates.render("partials/keys_table.html", &ctx)?;
    Ok(Html(html).into_response())
}
