//! Tag management endpoints (`/tag/*`).

use axum::extract::{Query, State};
use axum::response::{IntoResponse, Response};
use axum::{Extension, Json};
use serde::Deserialize;
use serde_json::json;

use crate::auth::{AuthUser, invalidate_cache_key, tag_cache_key};
use crate::error::ApiError;
use crate::state::SharedState;

use super::{
    db_err, load_db, map_create_err, require_master, serialize_json, serialize_string_list,
    truncate_scope_list, validate_limits,
};

/// Reject tag names that break the `spend:tag:{name}` key format.
fn validate_tag_name(name: &str) -> Result<(), ApiError> {
    if name.contains(':') {
        return Err(ApiError::bad_request(
            "tag name must not contain ':' (reserved as a spend-key separator)",
        ));
    }
    Ok(())
}

/// Request body for `POST /tag/new`.
#[derive(Debug, Deserialize)]
pub struct NewTagRequest {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub models: Option<Vec<String>>,
    #[serde(default)]
    pub model_info: Option<serde_json::Value>,
    #[serde(default)]
    pub max_budget: Option<f64>,
}

/// `POST /tag/new`
pub async fn new_tag(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Json(req): Json<NewTagRequest>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    validate_limits(&None, &None, req.max_budget)?;
    let (_state, db) = load_db(&shared, &auth).await?;

    let name = req.name.trim().to_string();
    if name.is_empty() {
        return Err(ApiError::bad_request("name is required"));
    }
    validate_tag_name(&name)?;
    if db.get_tag(&name).await.map_err(db_err)?.is_some() {
        return Err(ApiError::bad_request(format!(
            "tag '{name}' already exists"
        )));
    }

    let now = oxllm_db::now_iso();
    let record = oxllm_db::TagRecord {
        name: name.clone(),
        description: req.description,
        models: serialize_string_list(req.models),
        model_info: serialize_json(req.model_info),
        spend: Some(0.0),
        max_budget: req.max_budget,
        created_at: now.clone(),
        updated_at: now,
    };
    db.create_tag(&record).await.map_err(|e| {
        map_create_err(e, || {
            ApiError::bad_request(format!("tag '{name}' already exists"))
        })
    })?;
    // Drop negative-cache tombstone so the next request sees the new row.
    let state = shared.load_full();
    invalidate_cache_key(&state, &tag_cache_key(&name)).await;
    Ok(Json(json!({ "name": name, "status": "created" })).into_response())
}

/// Request body for `POST /tag/update`.
#[derive(Debug, Deserialize)]
pub struct UpdateTagRequest {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub models: Option<Vec<String>>,
    #[serde(default)]
    pub model_info: Option<serde_json::Value>,
    #[serde(default)]
    pub max_budget: Option<f64>,
    #[serde(default)]
    pub clear_max_budget: bool,
}

/// `POST /tag/update`
pub async fn update_tag(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Json(req): Json<UpdateTagRequest>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    validate_limits(&None, &None, req.max_budget)?;
    let (_state, db) = load_db(&shared, &auth).await?;

    let mut record = db
        .get_tag(&req.name)
        .await
        .map_err(db_err)?
        .ok_or_else(|| ApiError::not_found(format!("tag '{}' not found", req.name)))?;

    if let Some(v) = req.description {
        record.description = if v.is_empty() { None } else { Some(v) };
    }
    if let Some(v) = req.models {
        record.models = serialize_string_list(Some(v));
    }
    if let Some(v) = req.model_info {
        record.model_info = serialize_json(Some(v));
    }
    if req.clear_max_budget {
        record.max_budget = None;
    } else if req.max_budget.is_some() {
        record.max_budget = req.max_budget;
    }
    record.updated_at = oxllm_db::now_iso();

    let updated = db.update_tag(&record).await.map_err(db_err)?;
    if !updated {
        return Err(ApiError::not_found(format!(
            "tag '{}' not found",
            record.name
        )));
    }
    let state = shared.load_full();
    invalidate_cache_key(&state, &tag_cache_key(&record.name)).await;
    Ok(Json(record).into_response())
}

/// Request body for `POST /tag/delete`.
#[derive(Debug, Deserialize)]
pub struct DeleteTagRequest {
    pub name: String,
}

/// `POST /tag/delete`
pub async fn delete_tag(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Json(req): Json<DeleteTagRequest>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    let (_state, db) = load_db(&shared, &auth).await?;

    if db.get_tag(&req.name).await.map_err(db_err)?.is_none() {
        return Err(ApiError::not_found(format!("tag '{}' not found", req.name)));
    }
    db.delete_tag(&req.name).await.map_err(db_err)?;
    let state = shared.load_full();
    invalidate_cache_key(&state, &tag_cache_key(&req.name)).await;
    Ok(Json(json!({ "status": "deleted", "name": req.name })).into_response())
}

#[derive(Debug, Deserialize)]
pub struct TagInfoQuery {
    pub name: String,
}

/// `GET /tag/info?name=...`
pub async fn tag_info(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Query(q): Query<TagInfoQuery>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    let (_state, db) = load_db(&shared, &auth).await?;

    let record = db
        .get_tag(&q.name)
        .await
        .map_err(db_err)?
        .ok_or_else(|| ApiError::not_found(format!("tag '{}' not found", q.name)))?;
    Ok(Json(record).into_response())
}

/// `GET /tag/list`
pub async fn list_tags(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    let (_state, db) = load_db(&shared, &auth).await?;
    let mut tags = db
        .list_tags(Some((super::SCOPE_LIST_LIMIT as i64) + 1))
        .await
        .map_err(db_err)?;
    let truncated = truncate_scope_list(&mut tags);
    Ok(Json(json!({ "tags": tags, "truncated": truncated })).into_response())
}

#[cfg(all(test, feature = "db"))]
mod tests {
    use super::{UpdateTagRequest, serialize_string_list};

    #[test]
    fn serialize_string_list_ok() {
        assert_eq!(
            serialize_string_list(Some(vec!["a".into(), "".into()])),
            Some(r#"["a"]"#.to_string())
        );
    }

    /// `spend` must not be a settable field on update — it is an accumulator
    /// owned by the spend pipeline. Providing it is silently dropped (serde
    /// ignores unknown fields), so the stored spend is never overwritten via
    /// the update endpoint.
    #[test]
    fn update_tag_request_ignores_spend() {
        let json = r#"{"name":"x","spend":0.0}"#;
        let req: UpdateTagRequest = serde_json::from_str(json).expect("parses, spend ignored");
        assert_eq!(req.name, "x");
    }
}
