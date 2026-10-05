//! Project management endpoints.

use axum::extract::{Query, State};
use axum::response::{IntoResponse, Response};
use axum::{Extension, Json};
use serde::Deserialize;
use serde_json::json;

use crate::auth::{AuthUser, invalidate_cache_key};
use crate::error::ApiError;
use crate::state::SharedState;

use super::{
    db_err, load_db, map_create_err, random_id, require_master, truncate_scope_list,
    validate_budget_duration, validate_limits, validate_non_empty_name,
};

#[derive(Debug, Deserialize)]
pub struct AddProjectRequest {
    pub project_name: String,
    #[serde(default)]
    pub project_id: Option<String>,
    #[serde(default)]
    pub org_id: Option<String>,
    #[serde(default)]
    pub team_id: Option<String>,
    #[serde(default)]
    pub max_budget: Option<f64>,
    #[serde(default)]
    pub budget_duration: Option<String>,
    #[serde(default)]
    pub rpm: Option<i64>,
    #[serde(default)]
    pub tpm: Option<i64>,
}

pub async fn add_project(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Json(req): Json<AddProjectRequest>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    validate_non_empty_name(&req.project_name, "project_name")?;
    validate_limits(&req.rpm, &req.tpm, req.max_budget)?;
    validate_budget_duration(&req.budget_duration)?;
    let (state, db) = load_db(&shared, &auth).await?;

    let project_id = req
        .project_id
        .unwrap_or_else(|| format!("project-{}", random_id()));
    if db.get_project(&project_id).await.map_err(db_err)?.is_some() {
        return Err(ApiError::bad_request(format!(
            "project '{project_id}' already exists"
        )));
    }
    if let Some(org_id) = &req.org_id {
        if !org_id.is_empty() && db.get_organization(org_id).await.map_err(db_err)?.is_none() {
            return Err(ApiError::bad_request(format!("org '{org_id}' not found")));
        }
    }
    if let Some(team_id) = &req.team_id {
        if !team_id.is_empty() && db.get_team(team_id).await.map_err(db_err)?.is_none() {
            return Err(ApiError::bad_request(format!("team '{team_id}' not found")));
        }
    }
    let now = oxllm_db::now_iso();
    let record = oxllm_db::ProjectRecord {
        project_id: project_id.clone(),
        project_name: req.project_name,
        org_id: req.org_id,
        team_id: req.team_id,
        max_budget: req.max_budget,
        budget_duration: req.budget_duration,
        rpm: req.rpm,
        tpm: req.tpm,
        created_at: now.clone(),
        updated_at: now,
    };
    db.create_project(&record).await.map_err(|e| {
        map_create_err(e, || {
            ApiError::bad_request(format!("project '{project_id}' already exists"))
        })
    })?;
    invalidate_cache_key(&state, &format!("vk_project:{project_id}")).await;

    Ok(Json(json!({ "project_id": project_id })).into_response())
}

pub async fn list_projects(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    let (_state, db) = load_db(&shared, &auth).await?;

    let mut projects = db
        .list_projects(Some((super::SCOPE_LIST_LIMIT as i64) + 1))
        .await
        .map_err(db_err)?;
    let truncated = truncate_scope_list(&mut projects);
    Ok(Json(json!({ "projects": projects, "truncated": truncated })).into_response())
}

#[derive(Debug, Deserialize)]
pub struct ProjectInfoQuery {
    pub project_id: String,
}

pub async fn project_info(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Query(q): Query<ProjectInfoQuery>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    let (_state, db) = load_db(&shared, &auth).await?;

    let record = db
        .get_project(&q.project_id)
        .await
        .map_err(db_err)?
        .ok_or_else(|| ApiError::not_found(format!("project '{}' not found", q.project_id)))?;

    Ok(Json(record).into_response())
}

#[derive(Debug, Deserialize)]
pub struct DeleteProjectRequest {
    pub project_id: String,
}

pub async fn delete_project(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Json(req): Json<DeleteProjectRequest>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    let (state, db) = load_db(&shared, &auth).await?;

    let deleted = db
        .delete_project_and_detach(&req.project_id)
        .await
        .map_err(db_err)?;
    if !deleted.deleted {
        return Err(ApiError::not_found(format!(
            "project '{}' not found",
            req.project_id
        )));
    }
    for hash in deleted.key_hashes {
        invalidate_cache_key(&state, &format!("vk:{hash}")).await;
    }
    invalidate_cache_key(&state, &format!("vk_project:{}", req.project_id)).await;

    Ok(Json(json!({ "status": "deleted", "project_id": req.project_id })).into_response())
}

#[derive(Debug, Deserialize)]
pub struct UpdateProjectRequest {
    pub project_id: String,
    #[serde(default)]
    pub project_name: Option<String>,
    #[serde(default)]
    pub org_id: Option<String>,
    #[serde(default)]
    pub team_id: Option<String>,
    #[serde(default)]
    pub max_budget: Option<f64>,
    #[serde(default)]
    pub budget_duration: Option<String>,
    #[serde(default)]
    pub rpm: Option<i64>,
    #[serde(default)]
    pub tpm: Option<i64>,
    #[serde(default)]
    pub clear_max_budget: bool,
    #[serde(default)]
    pub clear_org_id: bool,
    #[serde(default)]
    pub clear_team_id: bool,
    /// When true, clear `rpm` (set to `None` / unlimited). Mirrors
    /// `clear_rpm` on the team/key update endpoints: a JSON `null`/omitted
    /// `rpm` leaves the value unchanged, so an explicit flag is required to
    /// remove the cap entirely.
    #[serde(default)]
    pub clear_rpm: bool,
    /// When true, clear `tpm` (set to `None` / unlimited). See [`Self::clear_rpm`].
    #[serde(default)]
    pub clear_tpm: bool,
}

pub async fn update_project(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Json(req): Json<UpdateProjectRequest>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    validate_limits(&req.rpm, &req.tpm, req.max_budget)?;
    validate_budget_duration(&req.budget_duration)?;
    let (state, db) = load_db(&shared, &auth).await?;

    let mut record = db
        .get_project(&req.project_id)
        .await
        .map_err(db_err)?
        .ok_or_else(|| ApiError::not_found(format!("project '{}' not found", req.project_id)))?;

    // Referential integrity: a reparent to a non-existent org/team is
    // rejected here, mirroring `add_project`. There is no FK constraint in
    // the schema, so without this check a project could be silently
    // reparented to a ghost id and later surface as a budget-scope miss.
    // `clear_org_id` / `clear_team_id` take precedence over a supplied id,
    // so the existence check is skipped when the field is being cleared.
    if let Some(org_id) = &req.org_id {
        if !org_id.is_empty()
            && !req.clear_org_id
            && db.get_organization(org_id).await.map_err(db_err)?.is_none()
        {
            return Err(ApiError::bad_request(format!("org '{org_id}' not found")));
        }
    }
    if let Some(team_id) = &req.team_id {
        if !team_id.is_empty()
            && !req.clear_team_id
            && db.get_team(team_id).await.map_err(db_err)?.is_none()
        {
            return Err(ApiError::bad_request(format!("team '{team_id}' not found")));
        }
    }

    if let Some(name) = req.project_name {
        let name = name.trim().to_string();
        if name.is_empty() {
            return Err(ApiError::bad_request("project_name must not be empty"));
        }
        record.project_name = name;
    }
    if let Some(v) = req.org_id {
        record.org_id = if v.is_empty() { None } else { Some(v) };
    }
    if req.clear_org_id {
        record.org_id = None;
    }
    if let Some(v) = req.team_id {
        record.team_id = if v.is_empty() { None } else { Some(v) };
    }
    if req.clear_team_id {
        record.team_id = None;
    }
    if req.clear_max_budget {
        record.max_budget = None;
    } else if req.max_budget.is_some() {
        record.max_budget = req.max_budget;
    }
    if let Some(v) = req.budget_duration {
        record.budget_duration = if v.is_empty() { None } else { Some(v) };
    }
    if req.clear_rpm {
        record.rpm = None;
    } else if req.rpm.is_some() {
        record.rpm = req.rpm;
    }
    if req.clear_tpm {
        record.tpm = None;
    } else if req.tpm.is_some() {
        record.tpm = req.tpm;
    }
    record.updated_at = oxllm_db::now_iso();

    let updated = db.update_project(&record).await.map_err(db_err)?;
    if !updated {
        return Err(ApiError::not_found(format!(
            "project '{}' not found",
            record.project_id
        )));
    }
    invalidate_cache_key(&state, &format!("vk_project:{}", record.project_id)).await;
    Ok(Json(record).into_response())
}
