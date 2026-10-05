//! Skill management endpoints (`/skill/*`).
//!
//! Anthropic Skills are versioned instruction files (SKILL.md content) that
//! extend a model's capabilities. These endpoints provide DB-backed CRUD for
//! skills; the `content` field holds the SKILL.md body.
//!
//! At request time, [`crate::skills::inject_skills_into_request`] loads every
//! skill whose `model_pattern` matches the requested model (or is unset) and
//! whose optional `team_id` is either empty or equals the caller's team,
//! then prepends the concatenated contents to the chat system prompt. See
//! that module for injection details and fail-open behaviour.

use axum::extract::{Path, State};
use axum::response::{IntoResponse, Response};
use axum::{Extension, Json};
use serde::Deserialize;
use serde_json::json;

use crate::auth::AuthUser;
use crate::error::ApiError;
use crate::state::SharedState;

use super::{db_err, load_db, random_id, require_master, truncate_scope_list};

/// Request body for `POST /skill/new`.
#[derive(Debug, Deserialize)]
pub struct NewSkillRequest {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    pub content: String,
    #[serde(default)]
    pub version: Option<String>,
    #[serde(default)]
    pub model_pattern: Option<String>,
    #[serde(default)]
    pub team_id: Option<String>,
}

/// `POST /skill/new`
pub async fn new_skill(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Json(req): Json<NewSkillRequest>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    let (_state, db) = load_db(&shared, &auth).await?;

    let name = req.name.trim().to_string();
    if name.is_empty() {
        return Err(ApiError::bad_request("name is required"));
    }
    if req.content.trim().is_empty() {
        return Err(ApiError::bad_request("content is required"));
    }
    // Referential integrity: a non-empty `team_id` must resolve to an
    // existing team (mirrors `add_project`). The schema has no FK, so
    // without this check a skill could be silently attached to a ghost
    // team and `list_skills_by_team` would never find it.
    if let Some(team_id) = &req.team_id {
        if !team_id.is_empty() && db.get_team(team_id).await.map_err(db_err)?.is_none() {
            return Err(ApiError::bad_request(format!("team '{team_id}' not found")));
        }
    }

    let id = format!("skill-{}", random_id());
    let now = oxllm_db::now_iso();
    let record = oxllm_db::types::SkillRecord {
        id: id.clone(),
        name,
        description: req.description,
        content: req.content,
        version: req.version.unwrap_or_else(|| "1.0".to_string()),
        model_pattern: req.model_pattern,
        team_id: req.team_id,
        created_at: now.clone(),
        updated_at: now,
    };
    db.create_skill(&record).await.map_err(db_err)?;

    tracing::info!(skill_id = %id, "skill added via management API");
    Ok(Json(json!({ "id": id, "status": "created" })).into_response())
}

/// Request body for `POST /skill/update`.
#[derive(Debug, Deserialize)]
pub struct UpdateSkillRequest {
    pub id: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub version: Option<String>,
    #[serde(default)]
    pub model_pattern: Option<String>,
    #[serde(default)]
    pub team_id: Option<String>,
    /// When true, clear the stored `model_pattern`.
    #[serde(default)]
    pub clear_model_pattern: bool,
    /// When true, clear the stored `team_id`.
    #[serde(default)]
    pub clear_team_id: bool,
    /// When true, clear the stored `description`.
    #[serde(default)]
    pub clear_description: bool,
}

/// `POST /skill/update`
pub async fn update_skill(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Json(req): Json<UpdateSkillRequest>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    let (_state, db) = load_db(&shared, &auth).await?;

    let mut record = db
        .get_skill(&req.id)
        .await
        .map_err(db_err)?
        .ok_or_else(|| ApiError::not_found(format!("skill '{}' not found", req.id)))?;

    if let Some(name) = req.name {
        let name = name.trim().to_string();
        if name.is_empty() {
            return Err(ApiError::bad_request("name must be non-empty"));
        }
        record.name = name;
    }
    if req.clear_description {
        record.description = None;
    } else if let Some(v) = req.description {
        record.description = if v.is_empty() { None } else { Some(v) };
    }
    if let Some(content) = req.content {
        if content.trim().is_empty() {
            return Err(ApiError::bad_request("content must be non-empty"));
        }
        record.content = content;
    }
    if let Some(version) = req.version {
        let version = version.trim().to_string();
        if version.is_empty() {
            return Err(ApiError::bad_request("version must be non-empty"));
        }
        record.version = version;
    }
    if req.clear_model_pattern {
        record.model_pattern = None;
    } else if let Some(v) = req.model_pattern {
        record.model_pattern = if v.is_empty() { None } else { Some(v) };
    }
    // Validate a newly-assigned `team_id` resolves (mirrors `add_project`).
    // `clear_team_id` takes precedence, so the check is skipped when the
    // field is being cleared.
    if req.clear_team_id {
        record.team_id = None;
    } else if let Some(v) = req.team_id {
        let v = v.trim().to_string();
        if !v.is_empty() && db.get_team(&v).await.map_err(db_err)?.is_none() {
            return Err(ApiError::bad_request(format!("team '{v}' not found")));
        }
        record.team_id = if v.is_empty() { None } else { Some(v) };
    }
    record.updated_at = oxllm_db::now_iso();

    let updated = db
        .update_skill(&req.id, &record)
        .await
        .map_err(db_err)?
        .ok_or_else(|| ApiError::not_found(format!("skill '{}' not found", req.id)))?;
    Ok(Json(updated).into_response())
}

/// Request body for `POST /skill/delete`.
#[derive(Debug, Deserialize)]
pub struct DeleteSkillRequest {
    pub id: String,
}

/// `POST /skill/delete`
pub async fn delete_skill(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Json(req): Json<DeleteSkillRequest>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    let (_state, db) = load_db(&shared, &auth).await?;

    let deleted = db.delete_skill(&req.id).await.map_err(db_err)?;
    if !deleted {
        return Err(ApiError::not_found(format!("skill '{}' not found", req.id)));
    }
    tracing::info!(skill_id = %req.id, "skill deleted via management API");
    Ok(Json(json!({ "id": req.id, "status": "deleted" })).into_response())
}

/// `GET /skill/{skill_id}`
pub async fn skill_info(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Path(skill_id): Path<String>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    let (_state, db) = load_db(&shared, &auth).await?;

    let record = db
        .get_skill(&skill_id)
        .await
        .map_err(db_err)?
        .ok_or_else(|| ApiError::not_found(format!("skill '{}' not found", skill_id)))?;
    Ok(Json(record).into_response())
}

/// `GET /skill/list`
pub async fn list_skills(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    let (_state, db) = load_db(&shared, &auth).await?;
    let mut skills = db
        .list_skills(Some((super::SCOPE_LIST_LIMIT as i64) + 1))
        .await
        .map_err(db_err)?;
    let truncated = truncate_scope_list(&mut skills);
    Ok(Json(json!({ "skills": skills, "truncated": truncated })).into_response())
}
