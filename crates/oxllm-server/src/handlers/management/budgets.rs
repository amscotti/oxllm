//! Budget plan management endpoints (`/budget/*`).

use axum::extract::{Query, State};
use axum::response::{IntoResponse, Response};
use axum::{Extension, Json};
use serde::Deserialize;
use serde_json::json;

use crate::auth::AuthUser;
use crate::error::ApiError;
use crate::state::SharedState;

use super::{
    db_err, load_db, map_create_err, random_id, require_master, serialize_json,
    validate_budget_duration, validate_limits, validate_parallel_limit,
};

/// Request body for `POST /budget/new`.
#[derive(Debug, Default, Deserialize)]
pub struct NewBudgetRequest {
    #[serde(default)]
    pub budget_id: Option<String>,
    #[serde(default)]
    pub max_budget: Option<f64>,
    #[serde(default)]
    pub soft_budget: Option<f64>,
    #[serde(default)]
    pub max_parallel_requests: Option<i64>,
    #[serde(default)]
    pub tpm_limit: Option<i64>,
    #[serde(default)]
    pub rpm_limit: Option<i64>,
    #[serde(default)]
    pub budget_duration: Option<String>,
    #[serde(default)]
    pub model_max_budget: Option<serde_json::Value>,
}

/// Reject a non-finite or negative `soft_budget`. `soft_budget` is
/// budget-specific (it has no place in [`super::validate_limits`], which is
/// shared across keys, teams, users, orgs, and projects) so it gets a
/// dedicated guard here. Mirrors the `max_budget` check inside
/// `validate_limits`: NaN / Infinity silently disable soft-budget alerting
/// (`hooks/alerting.rs` gates on `soft_budget > 0.0`, which is `false` for
/// NaN, and the trigger `current >= soft_budget.unwrap_or(INFINITY)` never
/// fires for `+Inf`), and a negative value would be silently coerced away at
/// enforcement time. LiteLLM applies the identical `math.isfinite(...) and
/// ... >= 0` check (`budget_management_endpoints.py`).
fn validate_soft_budget(soft_budget: Option<f64>) -> Result<(), ApiError> {
    if let Some(b) = soft_budget {
        if !b.is_finite() {
            return Err(ApiError::bad_request("soft_budget must be a finite number"));
        }
        if b < 0.0 {
            return Err(ApiError::bad_request("soft_budget must be non-negative"));
        }
    }
    Ok(())
}

/// Reject structurally malformed `model_max_budget` payloads at the API
/// boundary. LiteLLM parses each value as a `BudgetConfig` Pydantic model
/// (`budget_management_endpoints.py` → `validate_model_max_budget`); oxllm
/// stores the field as opaque JSON, so without a structural check a stray
/// string, number, or `{"oops": "garbage"}` object would persist silently
/// and surface as misleading data in `/budget/info`. The accepted shape is
/// `{"<model-name>": {"max_budget"?: number, "budget_duration"?: string,
/// "tpm_limit"?: int, "rpm_limit"?: int}, ...}` — a top-level object whose
/// every value is itself an object. We do not enforce the per-key numeric
/// constraints here (`max_budget` non-finite / negative, etc.) because the
/// field is operator-authored, master-gated input and the enforcement path
/// does a tolerant lookup.
fn validate_model_max_budget(model_max_budget: &Option<serde_json::Value>) -> Result<(), ApiError> {
    let Some(v) = model_max_budget else {
        return Ok(());
    };
    // `null` is treated as "absent" by `serialize_json`, so accept it here
    // for symmetry (LiteLLM's `if model_max_budget is not None` also skips
    // validation on null/None).
    if v.is_null() {
        return Ok(());
    }
    let serde_json::Value::Object(map) = v else {
        return Err(ApiError::bad_request(
            "model_max_budget must be an object keyed by model name",
        ));
    };
    for (model, inner) in map {
        if !inner.is_object() {
            return Err(ApiError::bad_request(format!(
                "model_max_budget['{model}'] must be an object"
            )));
        }
    }
    Ok(())
}

/// `POST /budget/new`
pub async fn new_budget(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Json(req): Json<NewBudgetRequest>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    validate_limits(&req.rpm_limit, &req.tpm_limit, req.max_budget)?;
    validate_parallel_limit(req.max_parallel_requests)?;
    validate_soft_budget(req.soft_budget)?;
    validate_model_max_budget(&req.model_max_budget)?;
    validate_budget_duration(&req.budget_duration)?;
    let (_state, db) = load_db(&shared, &auth).await?;

    let budget_id = req
        .budget_id
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| format!("budget-{}", random_id()));

    if db.get_budget(&budget_id).await.map_err(db_err)?.is_some() {
        return Err(ApiError::bad_request(format!(
            "budget '{budget_id}' already exists"
        )));
    }

    let now = oxllm_db::now_iso();
    let record = oxllm_db::BudgetRecord {
        budget_id: budget_id.clone(),
        max_budget: req.max_budget,
        soft_budget: req.soft_budget,
        max_parallel_requests: req.max_parallel_requests,
        tpm_limit: req.tpm_limit,
        rpm_limit: req.rpm_limit,
        budget_duration: req.budget_duration,
        model_max_budget: serialize_json(req.model_max_budget),
        created_at: now.clone(),
        updated_at: now,
    };
    db.create_budget(&record).await.map_err(|e| {
        map_create_err(e, || {
            ApiError::bad_request(format!("budget '{budget_id}' already exists"))
        })
    })?;
    Ok(Json(json!({ "budget_id": budget_id, "status": "created" })).into_response())
}

/// Request body for `POST /budget/update`.
#[derive(Debug, Default, Deserialize)]
pub struct UpdateBudgetRequest {
    pub budget_id: String,
    #[serde(default)]
    pub max_budget: Option<f64>,
    #[serde(default)]
    pub clear_max_budget: bool,
    #[serde(default)]
    pub soft_budget: Option<f64>,
    #[serde(default)]
    pub clear_soft_budget: bool,
    #[serde(default)]
    pub max_parallel_requests: Option<i64>,
    /// Reset `max_parallel_requests` to `None` (unlimited). Mirrors
    /// `clear_max_budget`/`clear_soft_budget`; required because
    /// `Some(0)` means "block all", not "unlimited", so the field
    /// cannot be cleared by sending a zero.
    #[serde(default)]
    pub clear_max_parallel_requests: bool,
    #[serde(default)]
    pub tpm_limit: Option<i64>,
    /// Reset `tpm_limit` to `None` (unlimited). See
    /// [`Self::clear_max_parallel_requests`].
    #[serde(default)]
    pub clear_tpm_limit: bool,
    #[serde(default)]
    pub rpm_limit: Option<i64>,
    /// Reset `rpm_limit` to `None` (unlimited). See
    /// [`Self::clear_max_parallel_requests`].
    #[serde(default)]
    pub clear_rpm_limit: bool,
    #[serde(default)]
    pub budget_duration: Option<String>,
    #[serde(default)]
    pub model_max_budget: Option<serde_json::Value>,
}

/// `POST /budget/update`
pub async fn update_budget(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Json(req): Json<UpdateBudgetRequest>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    validate_limits(&req.rpm_limit, &req.tpm_limit, req.max_budget)?;
    validate_parallel_limit(req.max_parallel_requests)?;
    validate_soft_budget(req.soft_budget)?;
    validate_model_max_budget(&req.model_max_budget)?;
    validate_budget_duration(&req.budget_duration)?;
    let (_state, db) = load_db(&shared, &auth).await?;

    let mut record = db
        .get_budget(&req.budget_id)
        .await
        .map_err(db_err)?
        .ok_or_else(|| ApiError::not_found(format!("budget '{}' not found", req.budget_id)))?;

    if req.clear_max_budget {
        record.max_budget = None;
    } else if req.max_budget.is_some() {
        record.max_budget = req.max_budget;
    }
    if req.clear_soft_budget {
        record.soft_budget = None;
    } else if req.soft_budget.is_some() {
        record.soft_budget = req.soft_budget;
    }
    if req.clear_max_parallel_requests {
        record.max_parallel_requests = None;
    } else if req.max_parallel_requests.is_some() {
        record.max_parallel_requests = req.max_parallel_requests;
    }
    if req.clear_tpm_limit {
        record.tpm_limit = None;
    } else if req.tpm_limit.is_some() {
        record.tpm_limit = req.tpm_limit;
    }
    if req.clear_rpm_limit {
        record.rpm_limit = None;
    } else if req.rpm_limit.is_some() {
        record.rpm_limit = req.rpm_limit;
    }
    if let Some(v) = req.budget_duration {
        record.budget_duration = if v.is_empty() { None } else { Some(v) };
    }
    if let Some(v) = req.model_max_budget {
        record.model_max_budget = serialize_json(Some(v));
    }
    record.updated_at = oxllm_db::now_iso();

    let updated = db.update_budget(&record).await.map_err(db_err)?;
    if !updated {
        return Err(ApiError::not_found(format!(
            "budget '{}' not found",
            record.budget_id
        )));
    }
    Ok(Json(record).into_response())
}

/// Request body for `POST /budget/delete` (single or bulk).
#[derive(Debug, Deserialize)]
pub struct DeleteBudgetRequest {
    #[serde(default)]
    pub budget_id: Option<String>,
    #[serde(default)]
    pub budget_ids: Option<Vec<String>>,
}

/// `POST /budget/delete` — delete one or many budgets.
pub async fn delete_budget(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Json(req): Json<DeleteBudgetRequest>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    let (_state, db) = load_db(&shared, &auth).await?;

    let mut ids = req.budget_ids.unwrap_or_default();
    if let Some(id) = req.budget_id {
        if !id.is_empty() && !ids.iter().any(|x| x == &id) {
            ids.push(id);
        }
    }
    if ids.is_empty() {
        return Err(ApiError::bad_request("budget_id or budget_ids is required"));
    }

    // Single-id path returns 404 when missing (mirrors other delete handlers).
    if ids.len() == 1 {
        let id = &ids[0];
        if db.get_budget(id).await.map_err(db_err)?.is_none() {
            return Err(ApiError::not_found(format!("budget '{id}' not found")));
        }
        db.delete_budget(id).await.map_err(db_err)?;
        return Ok(Json(json!({ "status": "deleted", "budget_id": id })).into_response());
    }

    let n = db.delete_budgets(&ids).await.map_err(db_err)?;
    Ok(Json(json!({
        "status": "deleted",
        "budgets_deleted": n,
        "budget_ids": ids,
    }))
    .into_response())
}

#[derive(Debug, Deserialize)]
pub struct BudgetInfoQuery {
    pub budget_id: String,
}

/// `GET /budget/info?budget_id=...`
pub async fn budget_info(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Query(q): Query<BudgetInfoQuery>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    let (_state, db) = load_db(&shared, &auth).await?;

    let record = db
        .get_budget(&q.budget_id)
        .await
        .map_err(db_err)?
        .ok_or_else(|| ApiError::not_found(format!("budget '{}' not found", q.budget_id)))?;
    Ok(Json(record).into_response())
}

/// Request body for bulk `POST /budget/info` (LiteLLM-compatible).
#[derive(Debug, Deserialize)]
pub struct BudgetInfoBody {
    pub budgets: Vec<String>,
}

/// `POST /budget/info` — fetch multiple budgets by id.
pub async fn budget_info_bulk(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Json(req): Json<BudgetInfoBody>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    let (_state, db) = load_db(&shared, &auth).await?;
    let budgets = db.get_budgets_by_ids(&req.budgets).await.map_err(db_err)?;
    Ok(Json(json!({ "budgets": budgets })).into_response())
}

/// `GET /budget/list`
pub async fn list_budgets(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    let (_state, db) = load_db(&shared, &auth).await?;
    let budgets = db.list_budgets(None).await.map_err(db_err)?;
    Ok(Json(json!({ "budgets": budgets })).into_response())
}

#[cfg(all(test, feature = "db"))]
mod tests {
    use super::{
        NewBudgetRequest, UpdateBudgetRequest, new_budget, update_budget,
        validate_model_max_budget, validate_soft_budget,
    };
    use axum::extract::State;
    use axum::http::StatusCode;
    use axum::{Extension, Json};
    use std::sync::Arc;

    use arc_swap::ArcSwap;

    use crate::auth::AuthUser;
    use crate::state::SharedState;

    #[test]
    fn validate_soft_budget_rejects_negative() {
        let err = validate_soft_budget(Some(-1.0)).expect_err("must reject");
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
        assert!(err.message.contains("soft_budget"));
    }

    #[test]
    fn validate_soft_budget_accepts_none_zero_positive() {
        assert!(validate_soft_budget(None).is_ok());
        assert!(validate_soft_budget(Some(0.0)).is_ok());
        assert!(validate_soft_budget(Some(10.0)).is_ok());
    }

    #[test]
    fn validate_soft_budget_rejects_nan() {
        let err = validate_soft_budget(Some(f64::NAN)).expect_err("must reject NaN");
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
        assert!(err.message.contains("finite"));
    }

    #[test]
    fn validate_soft_budget_rejects_infinity() {
        let err = validate_soft_budget(Some(f64::INFINITY)).expect_err("must reject Inf");
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
        assert!(err.message.contains("finite"));
    }

    #[test]
    fn validate_soft_budget_rejects_neg_infinity() {
        let err = validate_soft_budget(Some(f64::NEG_INFINITY)).expect_err("must reject -Inf");
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
        assert!(err.message.contains("finite"));
    }

    #[test]
    fn validate_model_max_budget_accepts_none() {
        assert!(validate_model_max_budget(&None).is_ok());
    }

    #[test]
    fn validate_model_max_budget_accepts_empty_object() {
        let v = serde_json::json!({});
        assert!(validate_model_max_budget(&Some(v)).is_ok());
    }

    #[test]
    fn validate_model_max_budget_accepts_well_formed() {
        let v = serde_json::json!({
            "openai/gpt-4o-mini": {"max_budget": 100.0, "budget_duration": "1d", "tpm_limit": 100000, "rpm_limit": 100000}
        });
        assert!(validate_model_max_budget(&Some(v)).is_ok());
    }

    #[test]
    fn validate_model_max_budget_rejects_bare_string() {
        let v = serde_json::json!("not-an-object");
        let err = validate_model_max_budget(&Some(v)).expect_err("must reject string");
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
        assert!(err.message.contains("model_max_budget"));
    }

    #[test]
    fn validate_model_max_budget_rejects_array() {
        let v = serde_json::json!([1, 2, 3]);
        assert!(validate_model_max_budget(&Some(v)).is_err());
    }

    #[test]
    fn validate_model_max_budget_rejects_non_object_inner_value() {
        let v = serde_json::json!({"gpt-4o": "oops"});
        let err = validate_model_max_budget(&Some(v)).expect_err("must reject inner string");
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
        assert!(err.message.contains("gpt-4o"));
    }

    #[test]
    fn validate_model_max_budget_accepts_null_as_absent() {
        // null maps to None at serialization time (see serialize_json) and
        // LiteLLM's validator skips on None, so accept it here.
        assert!(validate_model_max_budget(&Some(serde_json::Value::Null)).is_ok());
    }

    // ---- clear_* flags on UpdateBudgetRequest (C29 MEDIUM) -----------------
    //
    // rpm_limit / tpm_limit / max_parallel_requests are Option<i64> where
    // Some(0) means "block all" (not "unlimited"), so the field cannot be
    // cleared by sending a zero. The clear_* flags default to false so
    // existing callers (who omit them) leave the values unchanged.

    #[test]
    fn update_budget_request_clear_flags_default_false() {
        let req: UpdateBudgetRequest =
            serde_json::from_str(r#"{"budget_id":"b-1"}"#).expect("parse");
        assert!(!req.clear_max_parallel_requests);
        assert!(!req.clear_tpm_limit);
        assert!(!req.clear_rpm_limit);
    }

    #[test]
    fn update_budget_request_clear_flags_parse_true() {
        let req: UpdateBudgetRequest = serde_json::from_str(
            r#"{"budget_id":"b-1","clear_rpm_limit":true,"clear_tpm_limit":true,"clear_max_parallel_requests":true}"#,
        )
        .expect("parse");
        assert!(req.clear_rpm_limit);
        assert!(req.clear_tpm_limit);
        assert!(req.clear_max_parallel_requests);
    }

    // ---- C29 MEDIUM: budget_duration validated at the API boundary ----------
    //
    // `budgets.rs` was the lone scope handler that did not call
    // `validate_budget_duration`, so a typo like "30days" would persist into
    // the row and silently fall back to the 30d default at enforcement time.
    // The validation fires before `load_db`, so the rejection is observable
    // without needing a populated database.

    fn master_auth() -> AuthUser {
        AuthUser::master(crate::auth::hash_key(b"sk-master"))
    }

    async fn shared_with_db() -> (SharedState, Arc<oxllm_db::Database>) {
        let db = oxllm_db::Database::connect("sqlite::memory:")
            .await
            .expect("connect");
        db.migrate().await.expect("migrate");
        let state = crate::state::AppState::with_registry_cache_db(
            crate::config::types::ProxyConfig {
                model_list: vec![],
                oxllm_settings: Default::default(),
                general_settings: Default::default(),
                router_settings: Default::default(),
                extra: serde_json::Map::new(),
            },
            oxllm_providers::default_registry(),
            Arc::new(crate::cache::InMemoryCache::default()),
            None,
            Some(db.clone()),
        )
        .expect("state");
        (Arc::new(ArcSwap::from_pointee(state)), db)
    }

    #[tokio::test]
    async fn new_budget_rejects_malformed_budget_duration() {
        let (shared, _db) = shared_with_db().await;
        let req = NewBudgetRequest {
            budget_duration: Some("30days".to_string()),
            ..Default::default()
        };
        let err = new_budget(State(shared), Extension(master_auth()), Json(req))
            .await
            .expect_err("must reject malformed budget_duration");
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
        assert!(
            err.message.contains("budget_duration"),
            "message should name the offending field: {}",
            err.message
        );
    }

    #[tokio::test]
    async fn update_budget_rejects_malformed_budget_duration() {
        let (shared, _db) = shared_with_db().await;
        let req = UpdateBudgetRequest {
            budget_id: "budget-1".to_string(),
            budget_duration: Some("30days".to_string()),
            ..Default::default()
        };
        let err = update_budget(State(shared), Extension(master_auth()), Json(req))
            .await
            .expect_err("must reject malformed budget_duration");
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
        assert!(
            err.message.contains("budget_duration"),
            "message should name the offending field: {}",
            err.message
        );
    }

    // `Some("")` means "clear the field" on update; it must be accepted
    // (validate_budget_duration treats empty as valid) so the clear path is
    // not accidentally blocked by the new guard.
    #[tokio::test]
    async fn update_budget_accepts_empty_budget_duration_as_clear() {
        let (shared, db) = shared_with_db().await;
        let record = oxllm_db::BudgetRecord {
            budget_id: "budget-1".to_string(),
            max_budget: None,
            soft_budget: None,
            max_parallel_requests: None,
            tpm_limit: None,
            rpm_limit: None,
            budget_duration: Some("30d".to_string()),
            model_max_budget: None,
            created_at: oxllm_db::now_iso(),
            updated_at: oxllm_db::now_iso(),
        };
        db.create_budget(&record).await.expect("create");

        let req = UpdateBudgetRequest {
            budget_id: "budget-1".to_string(),
            budget_duration: Some(String::new()),
            ..Default::default()
        };
        let resp = update_budget(State(shared), Extension(master_auth()), Json(req))
            .await
            .expect("handler");
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .expect("body");
        let body: serde_json::Value = serde_json::from_slice(&bytes).expect("json");
        assert!(
            body["budget_duration"].is_null(),
            "empty string must clear the field, got {:?}",
            body["budget_duration"]
        );
    }
}
