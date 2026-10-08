//! Organization management endpoints.

use axum::extract::{Query, State};
use axum::response::{IntoResponse, Response};
use axum::{Extension, Json};
use serde::Deserialize;
use serde_json::json;

use crate::auth::{AuthUser, invalidate_cache_key};
use crate::error::ApiError;
use crate::state::SharedState;

use super::{
    db_err, load_db, map_create_err, random_id, require_master, serialize_json,
    serialize_string_list, truncate_scope_list, validate_budget_duration, validate_limits,
    validate_non_empty_name, validate_parallel_limit,
};

/// Request body for `POST /organization/add`.
///
/// Accepts both the legacy field names and the reference
/// `OrganizationUpdateRequestV2` aliases (`organization_alias`, `tpm_limit`,
/// `rpm_limit`) for fields the `organizations` table actually stores. The
/// v2 fields (`models`, `metadata`, `soft_budget`, `max_parallel_requests`,
/// `model_max_budget`) are persisted onto the v4 columns; `object_permission`
/// is still tolerated and ignored (serde drops unknown fields by default).
#[derive(Debug, Deserialize)]
pub struct AddOrgRequest {
    #[serde(alias = "organization_alias")]
    pub org_name: String,
    #[serde(default)]
    pub org_id: Option<String>,
    #[serde(default)]
    pub max_budget: Option<f64>,
    #[serde(default)]
    pub budget_duration: Option<String>,
    #[serde(default, alias = "rpm_limit")]
    pub rpm: Option<i64>,
    #[serde(default, alias = "tpm_limit")]
    pub tpm: Option<i64>,
    #[serde(default)]
    pub models: Option<Vec<String>>,
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
    #[serde(default)]
    pub soft_budget: Option<f64>,
    #[serde(default)]
    pub max_parallel_requests: Option<i64>,
    #[serde(default)]
    pub model_max_budget: Option<serde_json::Value>,
}

pub async fn add_org(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Json(req): Json<AddOrgRequest>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    validate_non_empty_name(&req.org_name, "org_name")?;
    validate_limits(&req.rpm, &req.tpm, req.max_budget)?;
    validate_parallel_limit(req.max_parallel_requests)?;
    validate_budget_duration(&req.budget_duration)?;
    let (state, db) = load_db(&shared, &auth).await?;

    let org_id = req.org_id.unwrap_or_else(|| format!("org-{}", random_id()));
    if db
        .get_organization(&org_id)
        .await
        .map_err(db_err)?
        .is_some()
    {
        return Err(ApiError::bad_request(format!(
            "org '{org_id}' already exists"
        )));
    }
    let now = oxllm_db::now_iso();
    let record = oxllm_db::OrganizationRecord {
        org_id: org_id.clone(),
        org_name: req.org_name,
        max_budget: req.max_budget,
        budget_duration: req.budget_duration,
        rpm: req.rpm,
        tpm: req.tpm,
        created_at: now.clone(),
        updated_at: now,
        models: serialize_string_list(req.models),
        metadata: serialize_json(req.metadata),
        soft_budget: req.soft_budget,
        max_parallel_requests: req.max_parallel_requests,
        model_max_budget: serialize_json(req.model_max_budget),
    };
    db.create_organization(&record).await.map_err(|e| {
        map_create_err(e, || {
            ApiError::bad_request(format!("org '{org_id}' already exists"))
        })
    })?;
    invalidate_cache_key(&state, &format!("vk_org:{org_id}")).await;

    Ok(Json(json!({ "org_id": org_id })).into_response())
}

pub async fn list_orgs(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    let (_state, db) = load_db(&shared, &auth).await?;

    let mut orgs = db
        .list_organizations(Some((super::SCOPE_LIST_LIMIT as i64) + 1))
        .await
        .map_err(db_err)?;
    let truncated = truncate_scope_list(&mut orgs);
    Ok(Json(json!({ "organizations": orgs, "truncated": truncated })).into_response())
}

#[derive(Debug, Deserialize)]
pub struct OrgInfoQuery {
    pub org_id: String,
}

pub async fn org_info(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Query(q): Query<OrgInfoQuery>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    let (_state, db) = load_db(&shared, &auth).await?;

    let record = db
        .get_organization(&q.org_id)
        .await
        .map_err(db_err)?
        .ok_or_else(|| ApiError::not_found(format!("org '{}' not found", q.org_id)))?;

    Ok(Json(record).into_response())
}

#[derive(Debug, Deserialize)]
pub struct DeleteOrgRequest {
    pub org_id: String,
}

pub async fn delete_org(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Json(req): Json<DeleteOrgRequest>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    let (state, db) = load_db(&shared, &auth).await?;

    let deleted = db
        .delete_organization_and_detach(&req.org_id)
        .await
        .map_err(db_err)?;
    if !deleted.deleted {
        return Err(ApiError::not_found(format!(
            "org '{}' not found",
            req.org_id
        )));
    }
    for hash in deleted.key_hashes {
        invalidate_cache_key(&state, &format!("vk:{hash}")).await;
    }
    for project_id in deleted.project_ids {
        invalidate_cache_key(&state, &format!("vk_project:{project_id}")).await;
    }
    invalidate_cache_key(&state, &format!("vk_org:{}", req.org_id)).await;

    Ok(Json(json!({ "status": "deleted", "org_id": req.org_id })).into_response())
}

/// Request body for `POST /organization/update`.
///
/// The reference v2 (`OrganizationUpdateRequestV2`) renamed several fields; both
/// spellings are accepted here and persisted onto the existing columns:
/// `organization_alias` → `org_name`, `tpm_limit` → `tpm`,
/// `rpm_limit` → `rpm`. The v2 fields (`models`, `metadata`, `soft_budget`,
/// `max_parallel_requests`, `model_max_budget`) are persisted onto the v4
/// columns; `object_permission` is still tolerated and ignored (serde drops
/// unknown fields by default).
#[derive(Debug, Deserialize)]
pub struct UpdateOrgRequest {
    pub org_id: String,
    #[serde(default, alias = "organization_alias")]
    pub org_name: Option<String>,
    #[serde(default)]
    pub max_budget: Option<f64>,
    #[serde(default)]
    pub budget_duration: Option<String>,
    #[serde(default, alias = "rpm_limit")]
    pub rpm: Option<i64>,
    #[serde(default, alias = "tpm_limit")]
    pub tpm: Option<i64>,
    #[serde(default)]
    pub clear_max_budget: bool,
    /// When true, clear `rpm` (set to `None` / unlimited). Mirrors
    /// `clear_rpm` on the team/key update endpoints: a JSON `null`/omitted
    /// `rpm` leaves the value unchanged, so an explicit flag is required to
    /// remove the cap entirely.
    #[serde(default)]
    pub clear_rpm: bool,
    /// When true, clear `tpm` (set to `None` / unlimited). See [`Self::clear_rpm`].
    #[serde(default)]
    pub clear_tpm: bool,
    #[serde(default)]
    pub models: Option<Vec<String>>,
    #[serde(default)]
    pub metadata: Option<serde_json::Value>,
    #[serde(default)]
    pub soft_budget: Option<f64>,
    #[serde(default)]
    pub max_parallel_requests: Option<i64>,
    #[serde(default)]
    pub model_max_budget: Option<serde_json::Value>,
    /// When true, reset `models` to `None`.
    #[serde(default)]
    pub clear_models: bool,
    /// When true, reset `metadata` to `None`.
    #[serde(default)]
    pub clear_metadata: bool,
    /// When true, reset `soft_budget` to `None`.
    #[serde(default)]
    pub clear_soft_budget: bool,
    /// When true, reset `max_parallel_requests` to `None`.
    #[serde(default)]
    pub clear_max_parallel_requests: bool,
    /// When true, reset `model_max_budget` to `None`.
    #[serde(default)]
    pub clear_model_max_budget: bool,
}

pub async fn update_org(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Json(req): Json<UpdateOrgRequest>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    validate_limits(&req.rpm, &req.tpm, req.max_budget)?;
    validate_parallel_limit(req.max_parallel_requests)?;
    validate_budget_duration(&req.budget_duration)?;
    let (state, db) = load_db(&shared, &auth).await?;

    let mut record = db
        .get_organization(&req.org_id)
        .await
        .map_err(db_err)?
        .ok_or_else(|| ApiError::not_found(format!("org '{}' not found", req.org_id)))?;

    if let Some(name) = req.org_name {
        let name = name.trim().to_string();
        if name.is_empty() {
            return Err(ApiError::bad_request("org_name must not be empty"));
        }
        record.org_name = name;
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
    if req.clear_models {
        record.models = None;
    } else if let Some(v) = req.models {
        record.models = serialize_string_list(Some(v));
    }
    if req.clear_metadata {
        record.metadata = None;
    } else if let Some(v) = req.metadata {
        record.metadata = serialize_json(Some(v));
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
    if req.clear_model_max_budget {
        record.model_max_budget = None;
    } else if let Some(v) = req.model_max_budget {
        record.model_max_budget = serialize_json(Some(v));
    }
    record.updated_at = oxllm_db::now_iso();

    let updated = db.update_organization(&record).await.map_err(db_err)?;
    if !updated {
        return Err(ApiError::not_found(format!(
            "org '{}' not found",
            record.org_id
        )));
    }
    invalidate_cache_key(&state, &format!("vk_org:{}", record.org_id)).await;
    Ok(Json(record).into_response())
}

#[cfg(all(test, feature = "db"))]
mod tests {
    use super::{AddOrgRequest, UpdateOrgRequest, serialize_json, serialize_string_list};

    /// Each v2 alias maps onto an existing column and round-trips through
    /// serde into the legacy field.
    #[test]
    fn update_request_accepts_v2_aliases() {
        let json = r#"{
            "org_id": "org-1",
            "organization_alias": "New Name",
            "tpm_limit": 500,
            "rpm_limit": 100,
            "max_budget": 25.5,
            "budget_duration": "30d"
        }"#;
        let req: UpdateOrgRequest = serde_json::from_str(json).expect("parse v2 update");
        assert_eq!(req.org_id, "org-1");
        assert_eq!(req.org_name.as_deref(), Some("New Name"));
        assert_eq!(req.tpm, Some(500));
        assert_eq!(req.rpm, Some(100));
        assert_eq!(req.max_budget, Some(25.5));
        assert_eq!(req.budget_duration.as_deref(), Some("30d"));
    }

    /// v4 fields are now backed by columns and round-trip through serde.
    /// `object_permission` (still no backing column) must be silently
    /// dropped so existing v2 clients remain compatible.
    #[test]
    fn update_request_accepts_v4_fields_and_drops_object_permission() {
        let json = r#"{
            "org_id": "org-1",
            "models": ["gpt-4o"],
            "metadata": {"team": "x"},
            "soft_budget": 10.0,
            "max_parallel_requests": 4,
            "model_max_budget": {"gpt-4o": 5.0},
            "object_permission": {}
        }"#;
        let req: UpdateOrgRequest = serde_json::from_str(json).expect("v4 fields must be parsed");
        assert_eq!(req.org_id, "org-1");
        assert!(req.org_name.is_none());
        assert!(req.tpm.is_none() && req.rpm.is_none());
        assert!(req.max_budget.is_none());
        assert_eq!(req.models.as_ref().unwrap()[0], "gpt-4o");
        assert_eq!(req.metadata.as_ref().unwrap()["team"], "x");
        assert_eq!(req.soft_budget, Some(10.0));
        assert_eq!(req.max_parallel_requests, Some(4));
        assert_eq!(req.model_max_budget.as_ref().unwrap()["gpt-4o"], 5.0);
    }

    /// Legacy clients using the original field names keep working unchanged.
    #[test]
    fn update_request_accepts_legacy_field_names() {
        let json = r#"{"org_id":"o","org_name":"N","tpm":1,"rpm":2}"#;
        let req: UpdateOrgRequest = serde_json::from_str(json).expect("parse legacy update");
        assert_eq!(req.org_name.as_deref(), Some("N"));
        assert_eq!(req.tpm, Some(1));
        assert_eq!(req.rpm, Some(2));
    }

    /// Mixing v2 and legacy names in one payload: explicit field wins, no
    /// panic when both spellings are absent for optional fields.
    #[test]
    fn update_request_v2_alias_for_org_name_satisfies_required_lookup() {
        let json = r#"{"org_id":"o","organization_alias":"AliasOnly"}"#;
        let req: UpdateOrgRequest = serde_json::from_str(json).expect("parse alias-only");
        assert_eq!(req.org_name.as_deref(), Some("AliasOnly"));
    }

    /// Create endpoint accepts the same v2 aliases.
    #[test]
    fn add_request_accepts_v2_aliases() {
        let json = r#"{
            "organization_alias": "Acme",
            "tpm_limit": 9,
            "rpm_limit": 8,
            "max_budget": 1.0
        }"#;
        let req: AddOrgRequest = serde_json::from_str(json).expect("parse v2 create");
        assert_eq!(req.org_name, "Acme");
        assert_eq!(req.tpm, Some(9));
        assert_eq!(req.rpm, Some(8));
        assert_eq!(req.max_budget, Some(1.0));
        assert!(req.org_id.is_none());
    }

    /// Create still rejects a payload missing both `org_name` and its alias.
    #[test]
    fn add_request_requires_org_name_or_alias() {
        let json = r#"{"tpm_limit": 9}"#;
        assert!(serde_json::from_str::<AddOrgRequest>(json).is_err());
    }

    /// Create endpoint accepts the v4 fields alongside the v2 aliases.
    #[test]
    fn add_request_accepts_v4_fields() {
        let json = r#"{
            "organization_alias": "Acme",
            "models": ["gpt-4o", "claude-*"],
            "metadata": {"tier": "enterprise"},
            "soft_budget": 50.0,
            "max_parallel_requests": 8,
            "model_max_budget": {"gpt-4o": 10.0}
        }"#;
        let req: AddOrgRequest = serde_json::from_str(json).expect("parse v4 create");
        assert_eq!(req.org_name, "Acme");
        assert_eq!(req.models.as_ref().unwrap()[1], "claude-*");
        assert_eq!(req.metadata.as_ref().unwrap()["tier"], "enterprise");
        assert_eq!(req.soft_budget, Some(50.0));
        assert_eq!(req.max_parallel_requests, Some(8));
        assert_eq!(req.model_max_budget.as_ref().unwrap()["gpt-4o"], 10.0);
    }

    /// C28 LOW: `models` is typed as `Vec<String>` so an object payload is
    /// rejected at the deserialization boundary rather than being persisted
    /// verbatim into a column whose contract is an array of model-pattern
    /// strings. The neighbouring `model_max_budget` column legitimately
    /// accepts an object, so only `models` is narrowed.
    #[test]
    fn add_request_rejects_non_array_models() {
        let json = r#"{
            "organization_alias": "Acme",
            "models": {"gpt-4o": true}
        }"#;
        assert!(
            serde_json::from_str::<AddOrgRequest>(json).is_err(),
            "object payload for models must be rejected"
        );
    }

    #[test]
    fn update_request_rejects_non_array_models() {
        let json = r#"{"org_id": "o", "models": 42}"#;
        assert!(
            serde_json::from_str::<UpdateOrgRequest>(json).is_err(),
            "scalar payload for models must be rejected"
        );
    }

    /// C28 LOW: org `models` is now serialized via `serialize_string_list`
    /// (consistent with the customer `model_blocklist` and tag `models`
    /// fields): blanks are trimmed/dropped and an empty result is stored as
    /// `None`.
    #[test]
    fn org_models_round_trips_through_serialize_string_list() {
        assert_eq!(serialize_string_list(None), None);
        assert_eq!(
            serialize_string_list(Some(vec!["gpt-4o".into(), "  ".into()])),
            Some(r#"["gpt-4o"]"#.to_string())
        );
    }

    /// `serialize_json`: None and explicit JSON null both become None;
    /// real values are serialized to a compact JSON string.
    #[test]
    fn serialize_json_null_becomes_none() {
        assert_eq!(serialize_json(None), None);
        assert_eq!(serialize_json(Some(serde_json::json!(null))), None);
        assert_eq!(
            serialize_json(Some(serde_json::json!(["gpt-4o"]))),
            Some(r#"["gpt-4o"]"#.to_string())
        );
        assert_eq!(
            serialize_json(Some(serde_json::json!({"gpt-4o": 1.0}))),
            Some(r#"{"gpt-4o":1.0}"#.to_string())
        );
    }

    /// C28 LOW: the v4 fields now expose `clear_*` flags so they can be
    /// reset to `None` (previously they were sticky once set). All five
    /// flags default to `false` and parse from a JSON payload.
    #[test]
    fn update_request_accepts_v4_clear_flags() {
        let json = r#"{
            "org_id": "org-1",
            "clear_models": true,
            "clear_metadata": true,
            "clear_soft_budget": true,
            "clear_max_parallel_requests": true,
            "clear_model_max_budget": true
        }"#;
        let req: UpdateOrgRequest = serde_json::from_str(json).expect("parse clear flags");
        assert!(req.clear_models);
        assert!(req.clear_metadata);
        assert!(req.clear_soft_budget);
        assert!(req.clear_max_parallel_requests);
        assert!(req.clear_model_max_budget);
    }

    /// C28 LOW: the `clear_*` flags default to `false` so omitting them is
    /// not interpreted as a reset.
    #[test]
    fn update_request_v4_clear_flags_default_false() {
        let req: UpdateOrgRequest =
            serde_json::from_str(r#"{"org_id": "org-1"}"#).expect("parse minimal");
        assert!(!req.clear_models);
        assert!(!req.clear_metadata);
        assert!(!req.clear_soft_budget);
        assert!(!req.clear_max_parallel_requests);
        assert!(!req.clear_model_max_budget);
    }
}
