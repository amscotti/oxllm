//! End-user / customer management endpoints (`/customer/*`).

use axum::extract::{Query, State};
use axum::response::{IntoResponse, Response};
use axum::{Extension, Json};
use serde::Deserialize;
use serde_json::json;

use crate::auth::{AuthUser, customer_cache_key, invalidate_cache_key};
use crate::error::ApiError;
use crate::state::SharedState;

use super::{
    SCOPE_LIST_LIMIT, db_err, load_db, map_create_err, require_master, serialize_string_list,
    truncate_scope_list, validate_budget_duration, validate_limits,
};

/// Reject end-user ids that break the `spend:customer:{id}` key format
/// (colon is the scope separator in spend keys / the reset scheduler).
fn validate_end_user_id(id: &str) -> Result<(), ApiError> {
    if id.contains(':') {
        return Err(ApiError::bad_request(
            "end_user_id must not contain ':' (reserved as a spend-key separator)",
        ));
    }
    Ok(())
}

/// Request body for `POST /customer/new`.
#[derive(Debug, Deserialize)]
pub struct NewCustomerRequest {
    pub end_user_id: String,
    #[serde(default)]
    pub alias: Option<String>,
    #[serde(default)]
    pub email: Option<String>,
    #[serde(default)]
    pub max_budget: Option<f64>,
    #[serde(default)]
    pub budget_duration: Option<String>,
    #[serde(default)]
    pub model_blocklist: Option<Vec<String>>,
}

/// `POST /customer/new` — create an end-user / customer.
pub async fn new_customer(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Json(req): Json<NewCustomerRequest>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    validate_limits(&None, &None, req.max_budget)?;
    validate_budget_duration(&req.budget_duration)?;
    let (_state, db) = load_db(&shared, &auth).await?;

    let end_user_id = req.end_user_id.trim().to_string();
    if end_user_id.is_empty() {
        return Err(ApiError::bad_request("end_user_id is required"));
    }
    validate_end_user_id(&end_user_id)?;
    if db
        .get_customer(&end_user_id)
        .await
        .map_err(db_err)?
        .is_some()
    {
        return Err(ApiError::bad_request(format!(
            "customer '{end_user_id}' already exists"
        )));
    }

    let now = oxllm_db::now_iso();
    let record = oxllm_db::CustomerRecord {
        end_user_id: end_user_id.clone(),
        alias: req.alias,
        email: req.email,
        max_budget: req.max_budget,
        spend: 0.0,
        budget_duration: req.budget_duration,
        blocked: false,
        model_blocklist: serialize_string_list(req.model_blocklist),
        created_at: now.clone(),
        updated_at: now,
    };
    db.create_customer(&record).await.map_err(|e| {
        map_create_err(e, || {
            ApiError::bad_request(format!("customer '{end_user_id}' already exists"))
        })
    })?;
    // Drop negative-cache tombstone so the next request sees the new row.
    let state = shared.load_full();
    invalidate_cache_key(&state, &customer_cache_key(&end_user_id)).await;
    Ok(Json(json!({ "end_user_id": end_user_id, "status": "created" })).into_response())
}

/// Request body for `POST /customer/update`.
#[derive(Debug, Deserialize)]
pub struct UpdateCustomerRequest {
    pub end_user_id: String,
    #[serde(default)]
    pub alias: Option<String>,
    #[serde(default)]
    pub email: Option<String>,
    #[serde(default)]
    pub max_budget: Option<f64>,
    #[serde(default)]
    pub clear_max_budget: bool,
    #[serde(default)]
    pub budget_duration: Option<String>,
    #[serde(default)]
    pub model_blocklist: Option<Vec<String>>,
    #[serde(default)]
    pub blocked: Option<bool>,
}

/// `POST /customer/update` — update customer fields.
pub async fn update_customer(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Json(req): Json<UpdateCustomerRequest>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    validate_limits(&None, &None, req.max_budget)?;
    validate_budget_duration(&req.budget_duration)?;
    let (_state, db) = load_db(&shared, &auth).await?;

    let mut record = db
        .get_customer(&req.end_user_id)
        .await
        .map_err(db_err)?
        .ok_or_else(|| ApiError::not_found(format!("customer '{}' not found", req.end_user_id)))?;

    if let Some(v) = req.alias {
        record.alias = if v.is_empty() { None } else { Some(v) };
    }
    if let Some(v) = req.email {
        record.email = if v.is_empty() { None } else { Some(v) };
    }
    if req.clear_max_budget {
        record.max_budget = None;
    } else if req.max_budget.is_some() {
        record.max_budget = req.max_budget;
    }
    if let Some(v) = req.budget_duration {
        record.budget_duration = if v.is_empty() { None } else { Some(v) };
    }
    if let Some(v) = req.model_blocklist {
        record.model_blocklist = serialize_string_list(Some(v));
    }
    if let Some(b) = req.blocked {
        record.blocked = b;
    }
    record.updated_at = oxllm_db::now_iso();

    let updated = db.update_customer(&record).await.map_err(db_err)?;
    if !updated {
        return Err(ApiError::not_found(format!(
            "customer '{}' not found",
            record.end_user_id
        )));
    }
    let state = shared.load_full();
    invalidate_cache_key(&state, &customer_cache_key(&record.end_user_id)).await;
    Ok(Json(record).into_response())
}

/// Request body for `POST /customer/delete`.
///
/// Mirrors [`CustomerBlockRequest`]: the legacy single `end_user_id` is
/// still accepted for back-compat, and the reference bulk shape
/// (`user_ids: List[str]`) is accepted alongside it. When `user_ids` is
/// present the request is treated as bulk and the reference all-or-404
/// existence semantics apply (every id must exist or the request fails
/// with a 404 naming the missing ids, before any row is deleted).
#[derive(Debug, Deserialize)]
pub struct DeleteCustomerRequest {
    #[serde(default)]
    pub end_user_id: Option<String>,
    #[serde(default)]
    pub user_ids: Option<Vec<String>>,
}

/// `POST /customer/delete` — hard-delete one or more customers.
///
/// - If only `end_user_id` is provided, deletes the single row and returns
///   the legacy single-id response shape (`status`, `end_user_id`). A
///   missing id is a 404.
/// - If `user_ids` is provided (with or without `end_user_id`), the ids are
///   deduped/trimmed/capped at [`SCOPE_LIST_LIMIT`] and the reference
///   all-or-404 semantics apply: every requested id must already exist or
///   the request fails with a 404 listing the missing ids, and nothing is
///   deleted. On success returns the bulk response shape
///   (`status`, `deleted_customers`).
pub async fn delete_customer(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Json(req): Json<DeleteCustomerRequest>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    let (_state, db) = load_db(&shared, &auth).await?;

    // Bulk path: reference-shaped `user_ids` (optionally combined with the
    // legacy `end_user_id`). All-or-404 existence check, then a single
    // bulk delete.
    let has_user_ids = req
        .user_ids
        .as_ref()
        .map(|v| v.iter().any(|s| !s.trim().is_empty()))
        .unwrap_or(false);
    if has_user_ids {
        let mut ids: Vec<String> = req
            .user_ids
            .clone()
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
        if let Some(id) = req.end_user_id.as_deref() {
            let id = id.trim();
            if !id.is_empty() && !ids.iter().any(|x| x == id) {
                ids.push(id.to_string());
            }
        }
        if ids.len() > SCOPE_LIST_LIMIT {
            return Err(ApiError::bad_request(format!(
                "too many ids (max {SCOPE_LIST_LIMIT})"
            )));
        }
        // All-or-404: every requested id must exist.
        let mut missing: Vec<String> = Vec::new();
        for id in &ids {
            if db.get_customer(id).await.map_err(db_err)?.is_none() {
                missing.push(id.clone());
            }
        }
        if !missing.is_empty() {
            return Err(ApiError::not_found(format!(
                "customer(s) not found: {}",
                missing.join(", ")
            )));
        }
        // Atomic bulk delete: a single chunked transaction so a mid-batch
        // failure can never leave a partial delete behind, and the reported
        // count is derived from `rows_affected` (not a request-side counter
        // that would over-report on a racy no-op between the existence
        // pre-check above and the delete).
        let deleted = db.delete_customers_bulk(&ids).await.map_err(db_err)?;
        let state = shared.load_full();
        for id in &ids {
            invalidate_cache_key(&state, &customer_cache_key(id)).await;
        }
        return Ok(Json(json!({
            "status": "deleted",
            "deleted_customers": deleted,
        }))
        .into_response());
    }

    // Single-id path (back-compat with the legacy oxllm shape).
    let end_user_id = req
        .end_user_id
        .map(|s| s.trim().to_string())
        .ok_or_else(|| ApiError::bad_request("end_user_id or user_ids is required"))?;
    if end_user_id.is_empty() {
        return Err(ApiError::bad_request("end_user_id or user_ids is required"));
    }
    if db
        .get_customer(&end_user_id)
        .await
        .map_err(db_err)?
        .is_none()
    {
        return Err(ApiError::not_found(format!(
            "customer '{end_user_id}' not found"
        )));
    }
    db.delete_customer(&end_user_id).await.map_err(db_err)?;
    let state = shared.load_full();
    invalidate_cache_key(&state, &customer_cache_key(&end_user_id)).await;
    Ok(Json(json!({ "status": "deleted", "end_user_id": end_user_id })).into_response())
}

#[derive(Debug, Deserialize)]
pub struct CustomerInfoQuery {
    pub end_user_id: String,
}

/// `GET /customer/info?end_user_id=...`
pub async fn customer_info(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Query(q): Query<CustomerInfoQuery>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    let (_state, db) = load_db(&shared, &auth).await?;

    let record = db
        .get_customer(&q.end_user_id)
        .await
        .map_err(db_err)?
        .ok_or_else(|| ApiError::not_found(format!("customer '{}' not found", q.end_user_id)))?;
    Ok(Json(record).into_response())
}

/// `GET /customer/list`
pub async fn list_customers(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    let (_state, db) = load_db(&shared, &auth).await?;
    let mut customers = db
        .list_customers(Some((SCOPE_LIST_LIMIT as i64) + 1))
        .await
        .map_err(db_err)?;
    let truncated = truncate_scope_list(&mut customers);
    Ok(Json(json!({ "customers": customers, "truncated": truncated })).into_response())
}

/// Request body for `POST /customer/block` and `POST /customer/unblock`.
#[derive(Debug, Deserialize)]
pub struct CustomerBlockRequest {
    /// Single id (reference shape).
    #[serde(default)]
    pub end_user_id: Option<String>,
    /// Bulk ids.
    #[serde(default)]
    pub user_ids: Option<Vec<String>>,
}

/// `POST /customer/block`
pub async fn block_customer(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Json(req): Json<CustomerBlockRequest>,
) -> Result<Response, ApiError> {
    set_customers_blocked(&shared, &auth, &req, true).await
}

/// `POST /customer/unblock`
pub async fn unblock_customer(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Json(req): Json<CustomerBlockRequest>,
) -> Result<Response, ApiError> {
    set_customers_blocked(&shared, &auth, &req, false).await
}

async fn set_customers_blocked(
    shared: &SharedState,
    auth: &AuthUser,
    req: &CustomerBlockRequest,
    blocked: bool,
) -> Result<Response, ApiError> {
    require_master(auth)?;
    let (_state, db) = load_db(shared, auth).await?;

    // Trim and drop blanks so `{"user_ids": ["", "  "]}` cannot slip past
    // the empty-list / cap gates (mirroring `serialize_string_list`).
    let mut ids: Vec<String> = req
        .user_ids
        .clone()
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
    if let Some(id) = &req.end_user_id {
        let id = id.trim();
        if !id.is_empty() && !ids.iter().any(|x| x == id) {
            ids.push(id.to_string());
        }
    }
    if ids.is_empty() {
        return Err(ApiError::bad_request("end_user_id or user_ids is required"));
    }
    if ids.len() > SCOPE_LIST_LIMIT {
        return Err(ApiError::bad_request(format!(
            "too many ids (max {SCOPE_LIST_LIMIT})"
        )));
    }

    let now = oxllm_db::now_iso();
    let affected = db
        .block_customers(&ids, blocked, &now)
        .await
        .map_err(db_err)?;
    let state = shared.load_full();
    for id in &ids {
        invalidate_cache_key(&state, &customer_cache_key(id)).await;
    }
    Ok(Json(json!({
        "status": "ok",
        "blocked": blocked,
        "customers_affected": affected,
    }))
    .into_response())
}

#[cfg(all(test, feature = "db"))]
mod tests {
    use super::{CustomerBlockRequest, DeleteCustomerRequest, serialize_string_list};

    #[test]
    fn serialize_string_list_filters_empty() {
        assert_eq!(serialize_string_list(None), None);
        assert_eq!(serialize_string_list(Some(vec![])), None);
        assert_eq!(
            serialize_string_list(Some(vec!["  ".into(), "gpt-4o".into()])),
            Some(r#"["gpt-4o"]"#.to_string())
        );
    }

    /// C28 MEDIUM: `/customer/delete` accepts the reference bulk shape
    /// (`user_ids: List[str]`) in addition to the legacy single
    /// `end_user_id`. Both fields are optional at the serde level so the
    /// dispatcher can branch on whichever was supplied.
    #[test]
    fn delete_customer_request_accepts_legacy_single_id() {
        let req: DeleteCustomerRequest =
            serde_json::from_str(r#"{"end_user_id": "eu-1"}"#).expect("parse legacy");
        assert_eq!(req.end_user_id.as_deref(), Some("eu-1"));
        assert!(req.user_ids.is_none());
    }

    #[test]
    fn delete_customer_request_accepts_litellm_bulk_ids() {
        let req: DeleteCustomerRequest =
            serde_json::from_str(r#"{"user_ids": ["a", "b"]}"#).expect("parse bulk");
        assert!(req.end_user_id.is_none());
        assert_eq!(req.user_ids.as_deref().unwrap(), &["a", "b"]);
    }

    #[test]
    fn delete_customer_request_accepts_both_fields() {
        let req: DeleteCustomerRequest =
            serde_json::from_str(r#"{"end_user_id": "x", "user_ids": ["y"]}"#)
                .expect("parse mixed");
        assert_eq!(req.end_user_id.as_deref(), Some("x"));
        assert_eq!(req.user_ids.as_deref().unwrap(), &["y"]);
    }

    /// C28 LOW: `set_customers_blocked` now trims/dedupes blanks from
    /// `user_ids` (mirroring `serialize_string_list`). Verify the
    /// `CustomerBlockRequest` shape still accepts both spellings.
    #[test]
    fn customer_block_request_accepts_bulk_and_single() {
        let req: CustomerBlockRequest =
            serde_json::from_str(r#"{"end_user_id": "a", "user_ids": ["b", ""]}"#).expect("parse");
        assert_eq!(req.end_user_id.as_deref(), Some("a"));
        assert_eq!(
            req.user_ids.as_deref().unwrap(),
            &["b".to_string(), "".to_string()]
        );
    }
}
