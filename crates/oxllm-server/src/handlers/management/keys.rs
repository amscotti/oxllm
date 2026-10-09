//! Virtual API key management endpoints.

use axum::extract::{Query, State};
use axum::response::{IntoResponse, Response};
use axum::{Extension, Json};
use rand::Rng;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::auth::{
    AuthUser, hash_key_hex, invalidate_cache_key, parse_cache_controls_column,
    validate_allowed_cache_controls,
};
use crate::error::ApiError;
use crate::state::SharedState;

use super::{
    db_err, load_db, random_id, require_master, validate_budget_duration, validate_expires,
    validate_limits,
};

/// The prefix used for all generated virtual keys.
const KEY_PREFIX: &str = "sk-ox-";

/// Maximum number of rows returned by [`list_keys`] before the result is
/// truncated. Acts as a backstop against unbounded scans of the `keys` table
/// until proper offset/limit pagination is added (#39).
const KEY_LIST_LIMIT: usize = 1000;

async fn validate_key_scopes(
    db: &oxllm_db::Database,
    record: &oxllm_db::KeyRecord,
) -> Result<(), ApiError> {
    for (scope, missing) in [
        (
            "team_id",
            match record.team_id.as_deref() {
                Some(id) => db.get_team(id).await.map_err(db_err)?.is_none(),
                None => false,
            },
        ),
        (
            "user_id",
            match record.user_id.as_deref() {
                Some(id) => db.get_user(id).await.map_err(db_err)?.is_none(),
                None => false,
            },
        ),
        (
            "org_id",
            match record.org_id.as_deref() {
                Some(id) => db.get_organization(id).await.map_err(db_err)?.is_none(),
                None => false,
            },
        ),
        (
            "project_id",
            match record.project_id.as_deref() {
                Some(id) => db.get_project(id).await.map_err(db_err)?.is_none(),
                None => false,
            },
        ),
    ] {
        if missing {
            return Err(ApiError::bad_request(format!(
                "{scope} must reference an existing scope"
            )));
        }
    }
    Ok(())
}

async fn invalidate_mutated_key(
    state: &crate::state::AppState,
    db: &oxllm_db::Database,
    record: &oxllm_db::KeyRecord,
) {
    invalidate_cache_key(state, &format!("vk:{}", record.key_hash)).await;
    if let Ok(Some(current)) = db.get_key(&record.key_id).await {
        if current.key_hash != record.key_hash {
            invalidate_cache_key(state, &format!("vk:{}", current.key_hash)).await;
        }
    }
}

/// Request body for `POST /key/generate`.
#[derive(Debug, Deserialize)]
pub struct GenerateKeyRequest {
    #[serde(default)]
    pub key_alias: Option<String>,
    #[serde(default)]
    pub team_id: Option<String>,
    #[serde(default)]
    pub user_id: Option<String>,
    #[serde(default)]
    pub org_id: Option<String>,
    #[serde(default)]
    pub project_id: Option<String>,
    #[serde(default)]
    pub max_budget: Option<f64>,
    #[serde(default)]
    pub budget_duration: Option<String>,
    #[serde(default)]
    pub rpm: Option<i64>,
    #[serde(default)]
    pub tpm: Option<i64>,
    #[serde(default)]
    pub expires: Option<String>,
    /// Allowed-model patterns (e.g. `["gpt-4o", "claude-*"]`). When omitted
    /// or empty, no model restriction is applied to the key.
    #[serde(default)]
    pub models: Option<Vec<String>>,
    /// Cache-control names this key may send. Omitted/null is unrestricted;
    /// [] denies every explicit cache control.
    #[serde(default)]
    pub allowed_cache_controls: Option<Vec<String>>,
}

/// Response body for `POST /key/generate`.
#[derive(Debug, Serialize)]
pub struct GenerateKeyResponse {
    pub key: String,
    pub key_id: String,
    pub expires: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_cache_controls: Option<Vec<String>>,
}

/// Reject negative `rpm` / `tpm` / `max_budget` values up front. Negative
/// values would otherwise be silently coerced to `None` by `u32::try_from(v)`
/// / `f64` comparisons at enforcement time, making the limit disappear without
/// warning. `max_budget == 0.0` is valid (means "block all spend"). Thin
/// wrapper over the shared [`super::validate_limits`] so the key path reports
/// the same errors as the team/user paths.
#[cfg(test)]
fn validate_key_limits(
    rpm: &Option<i64>,
    tpm: &Option<i64>,
    max_budget: Option<f64>,
) -> Result<(), ApiError> {
    validate_limits(rpm, tpm, max_budget)
}

/// `POST /key/generate` — create a new virtual API key.
///
/// Generates a random key, stores only its SHA-256 hash in the DB, and returns
/// the raw key exactly once.
pub async fn generate_key(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Json(req): Json<GenerateKeyRequest>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    validate_limits(&req.rpm, &req.tpm, req.max_budget)?;
    validate_expires(&req.expires)?;
    validate_budget_duration(&req.budget_duration)?;
    let allowed_cache_controls = serialize_cache_controls(req.allowed_cache_controls.as_deref())?;
    let (state, db) = load_db(&shared, &auth).await?;

    let raw_key = generate_random_key();
    let key_hash = hash_key_hex(raw_key.as_bytes());
    let key_id = format!("key-{}", random_id());

    let now = oxllm_db::now_iso();
    let record = oxllm_db::KeyRecord {
        key_id: key_id.clone(),
        key_hash,
        key_alias: req.key_alias,
        team_id: req.team_id,
        user_id: req.user_id,
        org_id: req.org_id,
        project_id: req.project_id,
        max_budget: req.max_budget,
        budget_duration: req.budget_duration,
        rpm: req.rpm,
        tpm: req.tpm,
        created_at: now.clone(),
        updated_at: now,
        expires_at: req.expires.clone(),
        is_active: true,
        models: serialize_models(req.models),
        allowed_cache_controls,
    };
    validate_key_scopes(&db, &record).await?;
    db.create_key(&record).await.map_err(db_err)?;
    invalidate_cache_key(&state, &format!("vk:{}", record.key_hash)).await;

    tracing::info!(key_id = %key_id, "virtual key generated");

    Ok(Json(GenerateKeyResponse {
        key: raw_key,
        key_id,
        expires: req.expires,
        allowed_cache_controls: req.allowed_cache_controls,
    })
    .into_response())
}

/// `GET /key/list` — list all keys. Hashes are never included.
///
/// The result is capped at `KEY_LIST_LIMIT` rows (1000) as a safety net for
/// unbounded tables; callers that need paging should add explicit `limit` /
/// `offset` query params in a future iteration. When the cap kicks in a
/// warning is logged so operators notice.
pub async fn list_keys(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    let (_state, db) = load_db(&shared, &auth).await?;

    let mut keys = db
        .list_keys(Some((KEY_LIST_LIMIT as i64) + 1))
        .await
        .map_err(db_err)?;
    // Known limitation (L18): there is no real pagination here — the result is
    // hard-capped at KEY_LIST_LIMIT rows. Surface the truncation in the
    // response body so callers can detect that the list is incomplete rather
    // than silently receiving a shorter-than-expected array. We request one
    // extra row from the DB as a sentinel: if it comes back, the table has
    // more rows than the cap. Full `limit` / `offset` paging is deferred to a
    // future iteration.
    let truncated = keys.len() > KEY_LIST_LIMIT;
    if truncated {
        tracing::warn!(
            total = keys.len(),
            limit = KEY_LIST_LIMIT,
            "/key/list returned more than KEY_LIST_LIMIT rows; truncating. \
             Add explicit pagination to view all keys."
        );
        keys.truncate(KEY_LIST_LIMIT);
    }
    let keys = keys
        .into_iter()
        .map(key_response)
        .collect::<Result<Vec<_>, _>>()?;

    Ok(Json(json!({ "keys": keys, "truncated": truncated })).into_response())
}

#[derive(Debug, Deserialize)]
pub struct KeyInfoQuery {
    pub key_id: String,
}

/// `GET /key/info?key_id=...` — get a single key's details (no hash).
pub async fn key_info(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Query(q): Query<KeyInfoQuery>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    let (_state, db) = load_db(&shared, &auth).await?;

    let record = db
        .get_key(&q.key_id)
        .await
        .map_err(db_err)?
        .ok_or_else(|| ApiError::not_found(format!("key '{}' not found", q.key_id)))?;
    Ok(Json(key_response(record)?).into_response())
}

/// Request body for `POST /key/delete`.
#[derive(Debug, Deserialize)]
pub struct DeleteKeyRequest {
    pub key_id: String,
}

/// `POST /key/delete` — soft-delete a key (marks it inactive).
///
/// Returns 404 only when the `key_id` has never existed (no row at all). A
/// second call against an already-soft-deleted key returns `"deleted"` again,
/// preserving idempotency. This mirrors the behavior of `delete_user` while
/// accounting for the fact that `delete_key` is a soft-delete rather than a
/// hard row removal.
pub async fn delete_key(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Json(req): Json<DeleteKeyRequest>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    let (state, db) = load_db(&shared, &auth).await?;

    let existing = db.get_key(&req.key_id).await.map_err(db_err)?;
    let Some(record) = existing else {
        return Err(ApiError::not_found(format!(
            "key '{}' not found",
            req.key_id
        )));
    };

    db.delete_key(&req.key_id, &oxllm_db::now_iso())
        .await
        .map_err(db_err)?;

    invalidate_cache_key(&state, &format!("vk:{}", record.key_hash)).await;
    // Re-read after the mutation to catch a concurrent `rotate_key` that
    // changed `key_hash` between our initial `get_key` and the soft-delete.
    // Without this, a request already cached under the new hash would keep
    // being served from cache as active for up to VK_POSITIVE_TTL (5 min)
    // after the delete. If the row is gone (concurrent purge) or the hash
    // is unchanged, this is a harmless no-op.
    if let Ok(Some(post)) = db.get_key(&req.key_id).await {
        if post.key_hash != record.key_hash {
            invalidate_cache_key(&state, &format!("vk:{}", post.key_hash)).await;
        }
    }

    Ok(Json(json!({ "status": "deleted", "key_id": req.key_id })).into_response())
}

/// Request body for `POST /key/purge`.
#[derive(Debug, Deserialize)]
pub struct PurgeKeyRequest {
    pub key_id: String,
}

/// `POST /key/purge` — hard-delete a key row permanently.
pub async fn purge_key(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Json(req): Json<PurgeKeyRequest>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    let (state, db) = load_db(&shared, &auth).await?;

    let existing = db
        .get_key(&req.key_id)
        .await
        .map_err(db_err)?
        .ok_or_else(|| ApiError::not_found(format!("key '{}' not found", req.key_id)))?;

    db.purge_key(&req.key_id).await.map_err(db_err)?;

    invalidate_cache_key(&state, &format!("vk:{}", existing.key_hash)).await;
    // Re-read after the mutation to catch a concurrent `rotate_key` (same
    // rationale as `delete_key`). After a purge the row is gone, so this
    // will typically be `None` — but if a concurrent rotate+purge raced,
    // the new hash (if any row still exists under this key_id) is covered.
    if let Ok(Some(post)) = db.get_key(&req.key_id).await {
        if post.key_hash != existing.key_hash {
            invalidate_cache_key(&state, &format!("vk:{}", post.key_hash)).await;
        }
    }

    Ok(Json(json!({ "status": "purged", "key_id": req.key_id })).into_response())
}

/// Request body for `POST /key/update`.
#[derive(Debug, Deserialize)]
pub struct UpdateKeyRequest {
    pub key_id: String,
    #[serde(default)]
    pub key_alias: Option<String>,
    #[serde(default)]
    pub team_id: Option<String>,
    #[serde(default)]
    pub user_id: Option<String>,
    #[serde(default)]
    pub org_id: Option<String>,
    #[serde(default)]
    pub project_id: Option<String>,
    #[serde(default)]
    pub max_budget: Option<f64>,
    /// When true, clear `max_budget` (set to `None` / unlimited). Mirrors the
    /// `clear_max_budget` flag on the user/team update endpoints — `Some(0.0)`
    /// means "block all spend", so `None` is required to remove the cap, and a
    /// JSON `null`/omitted field deserializes to `None` which leaves the value
    /// unchanged.
    #[serde(default)]
    pub clear_max_budget: bool,
    #[serde(default)]
    pub budget_duration: Option<String>,
    #[serde(default)]
    pub rpm: Option<i64>,
    #[serde(default)]
    pub tpm: Option<i64>,
    /// When true, clear `rpm` (set to `None` / unlimited). Mirrors
    /// `clear_max_budget`: a rate limit, once set, can only be lowered to `0`
    /// (block all traffic) without this flag — `null`/omitted leaves the value
    /// unchanged, so `clear_rpm` is required to remove the cap entirely.
    #[serde(default)]
    pub clear_rpm: bool,
    /// When true, clear `tpm` (set to `None` / unlimited). See [`Self::clear_rpm`].
    #[serde(default)]
    pub clear_tpm: bool,
    #[serde(default)]
    pub is_active: Option<bool>,
    /// Replacement expiry timestamp (ISO-8601). Pass an empty string `""` or
    /// `null` to clear the expiry (key never expires). Omit the field to leave
    /// it unchanged. Validated via the internal `validate_expires` helper when a
    /// non-empty, non-`null` value is supplied. Mirrors the reference mutable
    /// `duration` field on `/key/update`.
    #[serde(default)]
    pub expires: Option<String>,
    /// Replacement allowed-model patterns. Pass an empty array `[]` to clear
    /// the restriction. Omit the field (or `null`) to leave it unchanged.
    #[serde(default)]
    pub models: Option<Vec<String>>,
    /// Replace the cache-control allowlist. [] denies all explicit controls;
    /// omitted/null leaves the policy unchanged.
    #[serde(default)]
    pub allowed_cache_controls: Option<Vec<String>>,
    /// Remove the policy entirely (unrestricted). Cannot be combined with a
    /// replacement `allowed_cache_controls` list.
    #[serde(default)]
    pub clear_allowed_cache_controls: bool,
}

/// `POST /key/update` — update key metadata (not the secret).
///
/// Concurrency: like every management mutation this performs a
/// read-modify-write (`get_key` → mutate → `update_key`) with no
/// `SELECT … FOR UPDATE` or version column. Two concurrent master-key
/// operations on the same `key_id` can therefore clobber each other (the
/// last writer wins, silently reverting the other's field change). The
/// window is small and the actors are master-key operators, so impact is
/// bounded; a complete fix needs an optimistic-concurrency column checked in
/// the `WHERE` clause (409 on stale read). The team bulk helper
/// (`reassign_user_keys`) sidesteps this for the `team_id` field by using a
/// single scoped bulk `UPDATE`.
///
/// U12: the raw caller-supplied key value must never
/// land in audit trails or logs. oxllm is safe here by design: this handler
/// operates on `key_id` (a non-secret identifier like `key-abc123`), never the
/// raw `sk-ox-...` secret. The only `tracing::info!` / `tracing::warn!` calls
/// in this module reference `key_id`; the secret is never in scope (it lives
/// only in the `Authorization` header used by the auth middleware). The
/// `key_hash` (a SHA-256 digest) is cleared from the returned record before
/// serialization. There is no reference-style `object_id` audit write that
/// could capture the raw key.
pub async fn update_key(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Json(req): Json<UpdateKeyRequest>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    validate_limits(&req.rpm, &req.tpm, req.max_budget)?;
    validate_budget_duration(&req.budget_duration)?;
    if req.clear_allowed_cache_controls && req.allowed_cache_controls.is_some() {
        return Err(ApiError::bad_request(
            "clear_allowed_cache_controls cannot be combined with allowed_cache_controls",
        ));
    }
    let allowed_cache_controls = serialize_cache_controls(req.allowed_cache_controls.as_deref())?;
    if let Some(ref exp) = req.expires {
        if !exp.is_empty() {
            validate_expires(&Some(exp.clone()))?;
        }
    }
    let (state, db) = load_db(&shared, &auth).await?;

    let mut record = db
        .get_key(&req.key_id)
        .await
        .map_err(db_err)?
        .ok_or_else(|| ApiError::not_found(format!("key '{}' not found", req.key_id)))?;

    if let Some(v) = req.key_alias {
        record.key_alias = if v.is_empty() { None } else { Some(v) };
    }
    if let Some(v) = req.team_id {
        record.team_id = if v.is_empty() { None } else { Some(v) };
    }
    if let Some(v) = req.user_id {
        record.user_id = if v.is_empty() { None } else { Some(v) };
    }
    if let Some(v) = req.org_id {
        record.org_id = if v.is_empty() { None } else { Some(v) };
    }
    if let Some(v) = req.project_id {
        record.project_id = if v.is_empty() { None } else { Some(v) };
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
    if let Some(v) = req.is_active {
        record.is_active = v;
    }
    if let Some(v) = req.expires {
        record.expires_at = if v.is_empty() { None } else { Some(v) };
    }
    if let Some(v) = req.models {
        record.models = serialize_models(Some(v));
    }
    if req.clear_allowed_cache_controls {
        record.allowed_cache_controls = None;
    } else if req.allowed_cache_controls.is_some() {
        record.allowed_cache_controls = allowed_cache_controls;
    }
    record.updated_at = oxllm_db::now_iso();

    validate_key_scopes(&db, &record).await?;
    let updated = db.update_key(&record).await.map_err(db_err)?;
    if !updated {
        return Err(ApiError::not_found(format!(
            "key '{}' not found",
            record.key_id
        )));
    }

    invalidate_mutated_key(&state, &db, &record).await;

    Ok(Json(key_response(record)?).into_response())
}

/// Request body for `POST /key/block` and `POST /key/unblock`.
#[derive(Debug, Deserialize)]
pub struct BlockKeyRequest {
    pub key_id: String,
}

/// `POST /key/block` — mark a key inactive so it stops authenticating.
pub async fn block_key(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Json(req): Json<BlockKeyRequest>,
) -> Result<Response, ApiError> {
    set_key_active(&shared, &auth, &req.key_id, false).await?;
    Ok(Json(json!({ "status": "ok" })).into_response())
}

/// `POST /key/unblock` — reactivate a previously blocked key.
pub async fn unblock_key(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Json(req): Json<BlockKeyRequest>,
) -> Result<Response, ApiError> {
    set_key_active(&shared, &auth, &req.key_id, true).await?;
    Ok(Json(json!({ "status": "ok" })).into_response())
}

/// Flip the `is_active` flag on a key, persist it, and invalidate the key's
/// auth cache entry so the change takes effect immediately.
async fn set_key_active(
    shared: &SharedState,
    auth: &AuthUser,
    key_id: &str,
    active: bool,
) -> Result<(), ApiError> {
    require_master(auth)?;
    let (state, db) = load_db(shared, auth).await?;

    let mut record = db
        .get_key(key_id)
        .await
        .map_err(db_err)?
        .ok_or_else(|| ApiError::not_found(format!("key '{key_id}' not found")))?;

    record.is_active = active;
    record.updated_at = oxllm_db::now_iso();
    let updated = db.update_key(&record).await.map_err(db_err)?;
    if !updated {
        return Err(ApiError::not_found(format!("key '{key_id}' not found")));
    }

    invalidate_mutated_key(&state, &db, &record).await;
    Ok(())
}

/// Request body for `POST /key/rotate`.
#[derive(Debug, Deserialize)]
pub struct RotateKeyRequest {
    pub key_id: String,
}

/// `POST /key/rotate` — issue a new secret for an existing key_id.
/// The previous secret stops working immediately. Raw key returned once.
pub async fn rotate_key(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Json(req): Json<RotateKeyRequest>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    let (state, db) = load_db(&shared, &auth).await?;

    let raw_key = generate_random_key();
    let new_hash = hash_key_hex(raw_key.as_bytes());
    let now = oxllm_db::now_iso();

    let prev_hash = db
        .rotate_key(&req.key_id, &new_hash, &now)
        .await
        .map_err(db_err)?
        .ok_or_else(|| ApiError::not_found(format!("key '{}' not found", req.key_id)))?;

    invalidate_cache_key(&state, &format!("vk:{prev_hash}")).await;
    invalidate_cache_key(&state, &format!("vk:{new_hash}")).await;

    tracing::info!(key_id = %req.key_id, "virtual key rotated");

    Ok(Json(json!({
        "key": raw_key,
        "key_id": req.key_id,
        "status": "rotated",
    }))
    .into_response())
}

/// Encode an allowed-models list as the JSON string stored in the `models`
/// column. `None` or an empty/whitespace-only list becomes `None` so no
/// restriction is applied (backward-compatible default).
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

/// Unlike model ACLs, an empty cache-control list has a restrictive meaning.
/// Preserve it as JSON TEXT "[]"; only None removes the policy.
fn serialize_cache_controls(controls: Option<&[String]>) -> Result<Option<String>, ApiError> {
    controls
        .map(|controls| {
            validate_allowed_cache_controls(controls)?;
            serde_json::to_string(controls)
                .map_err(|_| ApiError::internal("unable to encode cache-control policy"))
        })
        .transpose()
}

/// The DB stores JSON TEXT, but management clients receive a typed JSON array
/// (or null for unrestricted), consistently on info, list, and update.
fn key_response(mut record: oxllm_db::KeyRecord) -> Result<Value, ApiError> {
    let controls = parse_cache_controls_column(record.allowed_cache_controls.as_deref());
    record.key_hash.clear();
    let mut value = serde_json::to_value(record)
        .map_err(|_| ApiError::internal("unable to encode key information"))?;
    let object = value
        .as_object_mut()
        .ok_or_else(|| ApiError::internal("unable to encode key information"))?;
    object.insert("allowed_cache_controls".to_string(), json!(controls));
    Ok(value)
}

/// Generate a random virtual key: `sk-ox-<40 hex chars>`.
fn generate_random_key() -> String {
    let mut rng = rand::rng();
    let bytes: [u8; 20] = rng.random();
    let hex: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
    format!("{KEY_PREFIX}{hex}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_key_has_prefix_and_sufficient_entropy() {
        let key = generate_random_key();
        assert!(key.starts_with(KEY_PREFIX));
        // prefix (7) + 40 hex chars = 47 total. Pin the exact length so a
        // regression that dropped entropy below 40 hex chars while keeping a
        // `>= 40` total assertion could not slip through.
        assert_eq!(key.len(), KEY_PREFIX.len() + 40);
    }

    #[test]
    fn random_ids_differ() {
        let a = random_id();
        let b = random_id();
        assert_ne!(a, b);
    }

    // ---- clear_max_budget on UpdateKeyRequest (C27/L2) ---------------------
    //
    // Mirrors the `clear_max_budget` flag on the user/team update endpoints.
    // `Some(0.0)` means "block all spend", so `None` is required to remove the
    // cap entirely, and a JSON `null`/omitted field deserializes to `None`
    // which leaves the value unchanged. `clear_max_budget` defaults to false
    // so existing callers are unaffected.

    #[test]
    fn update_key_request_clear_max_budget_defaults_false() {
        let req: UpdateKeyRequest = serde_json::from_str(r#"{"key_id":"k-1"}"#).expect("parse");
        assert!(!req.clear_max_budget);
        assert!(req.max_budget.is_none());
    }

    #[test]
    fn update_key_request_clear_max_budget_parses_true() {
        let req: UpdateKeyRequest =
            serde_json::from_str(r#"{"key_id":"k-1","clear_max_budget":true}"#).expect("parse");
        assert!(req.clear_max_budget);
    }

    // ---- clear_rpm / clear_tpm on UpdateKeyRequest (C27/LOW) ---------------
    //
    // Mirrors `clear_max_budget`: `Some(0)` means "block all traffic", so
    // `None` is required to remove the rate cap entirely. `clear_rpm` /
    // `clear_tpm` default to false so existing callers are unaffected.

    #[test]
    fn update_key_request_clear_rpm_tpm_default_false() {
        let req: UpdateKeyRequest = serde_json::from_str(r#"{"key_id":"k-1"}"#).expect("parse");
        assert!(!req.clear_rpm);
        assert!(!req.clear_tpm);
    }

    #[test]
    fn update_key_request_clear_rpm_tpm_parses_true() {
        let req: UpdateKeyRequest =
            serde_json::from_str(r#"{"key_id":"k-1","clear_rpm":true,"clear_tpm":true}"#)
                .expect("parse");
        assert!(req.clear_rpm);
        assert!(req.clear_tpm);
    }

    // ---- expires on UpdateKeyRequest (C27/L8: mutable expiry) ---------------

    #[test]
    fn update_key_request_expires_defaults_none() {
        let req: UpdateKeyRequest = serde_json::from_str(r#"{"key_id":"k-1"}"#).expect("parse");
        assert!(req.expires.is_none(), "omitted expires -> None (unchanged)");
    }

    #[test]
    fn update_key_request_expires_parses_value() {
        let req: UpdateKeyRequest =
            serde_json::from_str(r#"{"key_id":"k-1","expires":"2999-01-01T00:00:00Z"}"#)
                .expect("parse");
        assert_eq!(req.expires.as_deref(), Some("2999-01-01T00:00:00Z"));
    }

    #[test]
    fn update_key_request_expires_parses_empty_to_clear() {
        let req: UpdateKeyRequest =
            serde_json::from_str(r#"{"key_id":"k-1","expires":""}"#).expect("parse");
        assert_eq!(req.expires.as_deref(), Some(""));
    }

    #[test]
    fn update_key_request_expires_parses_null_to_none() {
        let req: UpdateKeyRequest =
            serde_json::from_str(r#"{"key_id":"k-1","expires":null}"#).expect("parse");
        assert!(req.expires.is_none());
    }

    // ---- validate_key_limits (#26b) ----------------------------------------

    #[test]
    fn validate_key_limits_accepts_none_limits() {
        assert!(validate_key_limits(&None, &None, None).is_ok());
    }

    #[test]
    fn validate_key_limits_accepts_zero_limits() {
        // 0 is valid: rpm=0 means "no requests", max_budget=0 means "no spend".
        assert!(validate_key_limits(&Some(0), &Some(0), Some(0.0)).is_ok());
    }

    #[test]
    fn validate_key_limits_accepts_positive_limits() {
        assert!(validate_key_limits(&Some(100), &Some(10_000), Some(50.0)).is_ok());
    }

    #[test]
    fn validate_key_limits_rejects_negative_rpm() {
        let err = validate_key_limits(&Some(-1), &None, None).expect_err("must reject");
        assert_eq!(err.status, axum::http::StatusCode::BAD_REQUEST);
        assert!(err.message.contains("rpm"));
    }

    #[test]
    fn validate_key_limits_rejects_negative_tpm() {
        let err = validate_key_limits(&None, &Some(-5), None).expect_err("must reject");
        assert_eq!(err.status, axum::http::StatusCode::BAD_REQUEST);
        assert!(err.message.contains("tpm"));
    }

    #[test]
    fn validate_key_limits_rejects_negative_max_budget() {
        let err = validate_key_limits(&None, &None, Some(-0.01)).expect_err("must reject");
        assert_eq!(err.status, axum::http::StatusCode::BAD_REQUEST);
        assert!(err.message.contains("max_budget"));
    }

    // ---- U12: raw key never leaks into audit trails / hashes ----------------
    //
    // The reference wrote the raw caller-supplied key into the
    // audit log's `object_id`. oxllm is safe by design: the only identifier
    // used in logs and the audit/spend trail is `key_id` (e.g. `key-abc123`),
    // and the raw `sk-ox-...` secret is never persisted — only its SHA-256
    // hash. These tests pin that invariant so a future change cannot regress
    // it.

    #[test]
    fn u12_key_hash_never_contains_raw_key() {
        // The stored hash must be a one-way digest: the raw key string must
        // not appear as a substring of the hash.
        let raw_key = generate_random_key();
        let hash = hash_key_hex(raw_key.as_bytes());
        assert!(!hash.contains(&raw_key), "hash must not leak the raw key");
        assert!(
            !hash.contains(KEY_PREFIX),
            "hash must not contain the key prefix"
        );
    }

    #[test]
    fn u12_generated_key_hash_is_sha256_length() {
        // A SHA-256 hex digest is exactly 64 chars — confirming the stored
        // credential is a hash, not the raw key (which is 47 chars).
        let raw_key = generate_random_key();
        let hash = hash_key_hex(raw_key.as_bytes());
        assert_eq!(hash.len(), 64, "stored credential must be a 256-bit hash");
        assert_ne!(hash, raw_key);
    }

    #[test]
    fn u12_key_id_is_non_secret_identifier() {
        // `key_id` (the only value logged by key management operations and
        // stored in the spend/audit trail) is a non-secret identifier. It
        // must never resemble the raw secret format `sk-ox-...`.
        let key_id = format!("key-{}", random_id());
        assert!(!key_id.contains(KEY_PREFIX));
        assert!(key_id.starts_with("key-"));
    }
}
