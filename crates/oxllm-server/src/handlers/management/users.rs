//! User management endpoints.

use axum::extract::{Query, State};
use axum::response::{IntoResponse, Response};
use axum::{Extension, Json};
use serde::Deserialize;
use serde_json::json;

use crate::auth::{AuthUser, invalidate_cache_key};
use crate::error::ApiError;
use crate::state::SharedState;

use super::{
    db_err, load_db, map_create_err, models_match_stored_policy, random_id, require_master,
    truncate_scope_list, validate_budget_duration, validate_limits, validate_non_empty_name,
};

/// Request body for `POST /user/add`.
#[derive(Debug, Deserialize)]
pub struct AddUserRequest {
    pub user_name: String,
    #[serde(default)]
    pub user_id: Option<String>,
    #[serde(default)]
    pub max_budget: Option<f64>,
    #[serde(default)]
    pub budget_duration: Option<String>,
    /// Allowed-model patterns for keys attached to this user.
    #[serde(default)]
    pub models: Option<Vec<String>>,
}

/// `POST /user/add` — create a user.
pub async fn add_user(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Json(req): Json<AddUserRequest>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    validate_non_empty_name(&req.user_name, "user_name")?;
    validate_limits(&None, &None, req.max_budget)?;
    validate_budget_duration(&req.budget_duration)?;
    let (_state, db) = load_db(&shared, &auth).await?;

    let user_id = req
        .user_id
        .unwrap_or_else(|| format!("user-{}", random_id()));
    let now = oxllm_db::now_iso();
    if db.get_user(&user_id).await.map_err(db_err)?.is_some() {
        return Err(ApiError::bad_request(format!(
            "user '{user_id}' already exists"
        )));
    }
    let record = oxllm_db::UserRecord {
        user_id: user_id.clone(),
        user_name: req.user_name,
        max_budget: req.max_budget,
        budget_duration: req.budget_duration,
        created_at: now.clone(),
        updated_at: now,
        models: serialize_models(req.models),
    };
    db.create_user(&record).await.map_err(|e| {
        map_create_err(e, || {
            ApiError::bad_request(format!("user '{user_id}' already exists"))
        })
    })?;

    Ok(Json(json!({ "user_id": user_id })).into_response())
}

/// `GET /user/list` — list all users.
///
/// The result is capped at `SCOPE_LIST_LIMIT` rows (1000) as a safety net for
/// unbounded tables, mirroring `keys::list_keys`. When the cap kicks in the
/// response carries `"truncated": true` so callers can detect an incomplete
/// list rather than silently receiving a shorter-than-expected array.
pub async fn list_users(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    let (_state, db) = load_db(&shared, &auth).await?;

    let mut users = db
        .list_users(Some((super::SCOPE_LIST_LIMIT as i64) + 1))
        .await
        .map_err(db_err)?;
    let truncated = truncate_scope_list(&mut users);
    Ok(Json(json!({ "users": users, "truncated": truncated })).into_response())
}

#[derive(Debug, Deserialize)]
pub struct UserInfoQuery {
    pub user_id: String,
}

/// `GET /user/info?user_id=...` — get user details.
pub async fn user_info(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Query(q): Query<UserInfoQuery>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    let (_state, db) = load_db(&shared, &auth).await?;

    let record = db
        .get_user(&q.user_id)
        .await
        .map_err(db_err)?
        .ok_or_else(|| ApiError::not_found(format!("user '{}' not found", q.user_id)))?;

    Ok(Json(record).into_response())
}

/// Request body for `POST /user/update`.
#[derive(Debug, Deserialize)]
pub struct UpdateUserRequest {
    pub user_id: String,
    #[serde(default)]
    pub user_name: Option<String>,
    #[serde(default)]
    pub max_budget: Option<f64>,
    #[serde(default)]
    pub budget_duration: Option<String>,
    #[serde(default)]
    pub clear_max_budget: bool,
    /// Replacement allowed-model patterns. `[]` clears; `null`/omitted unchanged.
    #[serde(default)]
    pub models: Option<Vec<String>>,
}

/// `POST /user/update` — update user fields.
pub async fn update_user(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Json(req): Json<UpdateUserRequest>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    if let Some(name) = &req.user_name {
        validate_non_empty_name(name, "user_name")?;
    }
    validate_limits(&None, &None, req.max_budget)?;
    validate_budget_duration(&req.budget_duration)?;
    let (state, db) = load_db(&shared, &auth).await?;

    let mut record = db
        .get_user(&req.user_id)
        .await
        .map_err(db_err)?
        .ok_or_else(|| ApiError::not_found(format!("user '{}' not found", req.user_id)))?;

    if let Some(name) = req.user_name {
        if !name.trim().is_empty() {
            record.user_name = name;
        }
    }
    if req.clear_max_budget {
        record.max_budget = None;
    } else if req.max_budget.is_some() {
        record.max_budget = req.max_budget;
    }
    if let Some(v) = req.budget_duration {
        record.budget_duration = if v.is_empty() { None } else { Some(v) };
    }
    if let Some(v) = req.models
        && !models_match_stored_policy(record.models.as_deref(), &v)
    {
        record.models = serialize_models(Some(v));
    }
    record.updated_at = oxllm_db::now_iso();

    let updated = db.update_user(&record).await.map_err(db_err)?;
    if !updated {
        return Err(ApiError::not_found(format!(
            "user '{}' not found",
            record.user_id
        )));
    }
    invalidate_cache_key(&state, &format!("vk_user:{}", record.user_id)).await;
    Ok(Json(record).into_response())
}

/// Request body for `POST /user/delete`.
#[derive(Debug, Deserialize)]
pub struct DeleteUserRequest {
    pub user_id: String,
}

/// Encode an allowed-models list as the JSON string stored in the `models`
/// column. `None` or empty list becomes `None` (no restriction).
fn serialize_models(models: Option<Vec<String>>) -> Option<String> {
    let filtered: Vec<String> = models
        .unwrap_or_default()
        .into_iter()
        .filter_map(|s| {
            let trimmed = s.trim().to_string();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed)
            }
        })
        .collect();
    if filtered.is_empty() {
        None
    } else {
        serde_json::to_string(&filtered).ok()
    }
}

/// `POST /user/delete` — hard-delete a user row.
///
/// The DB deletes the user and clears member keys' `user_id` atomically;
/// affected auth entries are invalidated after commit. Keys retain their
/// active state and other limits. Removing the deleted user's scope is the
/// intended administrative semantics, rather than key revocation.
pub async fn delete_user(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Json(req): Json<DeleteUserRequest>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    let (state, db) = load_db(&shared, &auth).await?;

    let deleted = db
        .delete_user_and_detach(&req.user_id)
        .await
        .map_err(db_err)?;
    if !deleted.deleted {
        return Err(ApiError::not_found(format!(
            "user '{}' not found",
            req.user_id
        )));
    }
    for hash in deleted.key_hashes {
        invalidate_cache_key(&state, &format!("vk:{hash}")).await;
    }
    invalidate_cache_key(&state, &format!("vk_user:{}", req.user_id)).await;

    Ok(Json(json!({ "status": "deleted", "user_id": req.user_id })).into_response())
}
