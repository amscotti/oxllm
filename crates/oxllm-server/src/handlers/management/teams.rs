//! Team management endpoints.

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

/// Request body for `POST /team/add`.
#[derive(Debug, Deserialize)]
pub struct AddTeamRequest {
    pub team_name: String,
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
    /// Allowed-model patterns for keys attached to this team.
    #[serde(default)]
    pub models: Option<Vec<String>>,
}

/// `POST /team/add` — create a team.
pub async fn add_team(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Json(req): Json<AddTeamRequest>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    validate_non_empty_name(&req.team_name, "team_name")?;
    validate_limits(&req.rpm, &req.tpm, req.max_budget)?;
    validate_budget_duration(&req.budget_duration)?;
    let (state, db) = load_db(&shared, &auth).await?;

    let team_id = req
        .team_id
        .unwrap_or_else(|| format!("team-{}", random_id()));
    if db.get_team(&team_id).await.map_err(db_err)?.is_some() {
        return Err(ApiError::bad_request(format!(
            "team '{team_id}' already exists"
        )));
    }
    let now = oxllm_db::now_iso();
    let record = oxllm_db::TeamRecord {
        team_id: team_id.clone(),
        team_name: req.team_name,
        max_budget: req.max_budget,
        budget_duration: req.budget_duration,
        rpm: req.rpm,
        tpm: req.tpm,
        created_at: now.clone(),
        updated_at: now,
        models: serialize_models(req.models),
        blocked: false,
    };
    db.create_team(&record).await.map_err(|e| {
        map_create_err(e, || {
            ApiError::bad_request(format!("team '{team_id}' already exists"))
        })
    })?;
    invalidate_cache_key(&state, &format!("vk_team:{team_id}")).await;

    Ok(Json(json!({ "team_id": team_id })).into_response())
}

/// `GET /team/list` — list all teams.
///
/// The result is capped at `SCOPE_LIST_LIMIT` rows (1000) as a safety net for
/// unbounded tables, mirroring `keys::list_keys`. When the cap kicks in the
/// response carries `"truncated": true` so callers can detect an incomplete
/// list rather than silently receiving a shorter-than-expected array.
pub async fn list_teams(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    let (_state, db) = load_db(&shared, &auth).await?;

    let mut teams = db
        .list_teams(Some((super::SCOPE_LIST_LIMIT as i64) + 1))
        .await
        .map_err(db_err)?;
    let truncated = truncate_scope_list(&mut teams);
    Ok(Json(json!({ "teams": teams, "truncated": truncated })).into_response())
}

#[derive(Debug, Deserialize)]
pub struct TeamInfoQuery {
    pub team_id: String,
}

/// `GET /team/info?team_id=...` — get team details.
pub async fn team_info(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Query(q): Query<TeamInfoQuery>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    let (_state, db) = load_db(&shared, &auth).await?;

    let record = db
        .get_team(&q.team_id)
        .await
        .map_err(db_err)?
        .ok_or_else(|| ApiError::not_found(format!("team '{}' not found", q.team_id)))?;

    Ok(Json(record).into_response())
}

/// Request body for `POST /team/delete`.
#[derive(Debug, Deserialize)]
pub struct DeleteTeamRequest {
    pub team_id: String,
}

/// `POST /team/delete` — delete a team. Returns 404 if the team never
/// existed (mirrors [`crate::handlers::management::users::delete_user`]).
///
/// To avoid leaving keys pointing at a removed scope (which would otherwise
/// silently drop the team-level budget / rate enforcement on the next auth),
/// this cascades: `team_id` is cleared on every key that currently
/// references the team, and each affected key's auth cache entry is
/// invalidated. Keys keep working under their own (and any other scope's)
/// limits; only the now-deleted team's aggregate cap is removed.
pub async fn delete_team(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Json(req): Json<DeleteTeamRequest>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    let (state, db) = load_db(&shared, &auth).await?;

    let deleted = db
        .delete_team_and_detach(&req.team_id)
        .await
        .map_err(db_err)?;
    if !deleted.deleted {
        return Err(ApiError::not_found(format!(
            "team '{}' not found",
            req.team_id
        )));
    }
    for hash in deleted.key_hashes {
        invalidate_cache_key(&state, &format!("vk:{hash}")).await;
    }
    for project_id in deleted.project_ids {
        invalidate_cache_key(&state, &format!("vk_project:{project_id}")).await;
    }
    invalidate_cache_key(&state, &format!("vk_team:{}", req.team_id)).await;

    Ok(Json(json!({ "status": "deleted", "team_id": req.team_id })).into_response())
}

/// Request body for `POST /team/update`.
#[derive(Debug, Deserialize)]
pub struct UpdateTeamRequest {
    pub team_id: String,
    #[serde(default)]
    pub team_name: Option<String>,
    #[serde(default)]
    pub max_budget: Option<f64>,
    #[serde(default)]
    pub budget_duration: Option<String>,
    #[serde(default)]
    pub rpm: Option<i64>,
    #[serde(default)]
    pub tpm: Option<i64>,
    /// When true, clear max_budget (set null).
    #[serde(default)]
    pub clear_max_budget: bool,
    /// When true, clear `rpm` (set to `None` / unlimited). Mirrors
    /// `clear_max_budget`: a rate limit, once set, can only be lowered to `0`
    /// (block all traffic) without this flag.
    #[serde(default)]
    pub clear_rpm: bool,
    /// When true, clear `tpm` (set to `None` / unlimited). See [`Self::clear_rpm`].
    #[serde(default)]
    pub clear_tpm: bool,
    /// Replacement allowed-model patterns. `[]` clears the restriction;
    /// `null`/omitted leaves it unchanged.
    #[serde(default)]
    pub models: Option<Vec<String>>,
}

/// `POST /team/update` — update team fields.
pub async fn update_team(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Json(req): Json<UpdateTeamRequest>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    if let Some(name) = &req.team_name {
        validate_non_empty_name(name, "team_name")?;
    }
    validate_limits(&req.rpm, &req.tpm, req.max_budget)?;
    validate_budget_duration(&req.budget_duration)?;
    let (state, db) = load_db(&shared, &auth).await?;

    let mut record = db
        .get_team(&req.team_id)
        .await
        .map_err(db_err)?
        .ok_or_else(|| ApiError::not_found(format!("team '{}' not found", req.team_id)))?;

    if let Some(name) = req.team_name {
        if !name.trim().is_empty() {
            record.team_name = name;
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
    if let Some(v) = req.models {
        record.models = serialize_models(Some(v));
    }
    record.updated_at = oxllm_db::now_iso();

    let updated = db.update_team(&record).await.map_err(db_err)?;
    if !updated {
        return Err(ApiError::not_found(format!(
            "team '{}' not found",
            record.team_id
        )));
    }
    invalidate_cache_key(&state, &format!("vk_team:{}", record.team_id)).await;
    Ok(Json(record).into_response())
}

// ---------------------------------------------------------------------------
// Team block / unblock
// ---------------------------------------------------------------------------

/// Request body for `POST /team/block` and `POST /team/unblock`.
#[derive(Debug, Deserialize)]
pub struct TeamBlockRequest {
    pub team_id: String,
}

/// `POST /team/block` — block a team so every key in it stops
/// authenticating. Mirrors LiteLLM's `team.blocked` flag: the block lives on
/// the team row, not on the individual keys, so `/key/block` applied to a
/// specific key for a different reason (compromise, rotation, abuse) is
/// preserved across a subsequent `/team/unblock`.
pub async fn block_team(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Json(req): Json<TeamBlockRequest>,
) -> Result<Response, ApiError> {
    set_team_blocked(&shared, &auth, &req.team_id, true).await?;
    Ok(Json(json!({ "status": "ok" })).into_response())
}

/// `POST /team/unblock` — clear a team's blocked flag. Keys that were
/// individually blocked via `/key/block` remain inactive — this handler only
/// flips `team.blocked`, not per-key `is_active`.
pub async fn unblock_team(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Json(req): Json<TeamBlockRequest>,
) -> Result<Response, ApiError> {
    set_team_blocked(&shared, &auth, &req.team_id, false).await?;
    Ok(Json(json!({ "status": "ok" })).into_response())
}

/// Set the `blocked` flag on a team row and invalidate the team-limits auth
/// cache entry so the change takes effect immediately. Returns 404 if the
/// team does not exist. Does not touch any key's `is_active` flag.
async fn set_team_blocked(
    shared: &SharedState,
    auth: &AuthUser,
    team_id: &str,
    blocked: bool,
) -> Result<(), ApiError> {
    require_master(auth)?;
    let (state, db) = load_db(shared, auth).await?;

    let mut record = db
        .get_team(team_id)
        .await
        .map_err(db_err)?
        .ok_or_else(|| ApiError::not_found(format!("team '{team_id}' not found")))?;

    record.blocked = blocked;
    record.updated_at = oxllm_db::now_iso();
    let updated = db.update_team(&record).await.map_err(db_err)?;
    if !updated {
        return Err(ApiError::not_found(format!("team '{team_id}' not found")));
    }

    invalidate_cache_key(&state, &format!("vk_team:{team_id}")).await;
    Ok(())
}

// ---------------------------------------------------------------------------
// Team member management
// ---------------------------------------------------------------------------

/// Request body for `POST /team/member/*`.
#[derive(Debug, Deserialize)]
pub struct MemberRequest {
    pub team_id: String,
    pub user_id: String,
}

/// `POST /team/member/add` — add a user to a team.
///
/// There is no `team_members` table; a user's team membership is expressed
/// through their keys (a key carries both `user_id` and `team_id`). Adding a
/// member therefore sets `team_id` on **every** key owned by the user.
///
/// **Destructive across teams:** this operation moves the user's keys from
/// any current team (including `None`) into the new team. A user can only
/// belong to one team at a time in this model (unlike LiteLLM, which has a
/// separate `LiteLLM_TeamMembership` table allowing multi-team membership).
/// If the user has keys in team A and you call `/team/member/add(user, B)`,
/// all of the user's keys — including those in team A — are reassigned to
/// team B, dropping team A's budget/rate enforcement on those keys. Use
/// `/team/member/delete` first if you need to move only specific keys.
pub async fn member_add(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Json(req): Json<MemberRequest>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    let (state, db) = load_db(&shared, &auth).await?;

    if db.get_team(&req.team_id).await.map_err(db_err)?.is_none() {
        return Err(ApiError::not_found(format!(
            "team '{}' not found",
            req.team_id
        )));
    }
    if db.get_user(&req.user_id).await.map_err(db_err)?.is_none() {
        return Err(ApiError::not_found(format!(
            "user '{}' not found",
            req.user_id
        )));
    }

    let n = reassign_user_keys(&state, &db, &req.user_id, None, Some(&req.team_id)).await?;
    invalidate_cache_key(&state, &format!("vk_team:{}", req.team_id)).await;
    Ok(Json(json!({ "status": "ok", "keys_affected": n })).into_response())
}

/// `POST /team/member/update` — placeholder. No per-member fields (budget,
/// role) exist yet, so this is not implemented. Returns 501 so callers do not
/// silently rely on a no-op that appears to succeed.
pub async fn member_update(
    State(_shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Json(_req): Json<MemberRequest>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    Err(ApiError {
        status: axum::http::StatusCode::NOT_IMPLEMENTED,
        message: "team/member/update is not implemented".to_string(),
        error_type: "api_error".to_string(),
        code: Some("not_implemented".to_string()),
        headers: None,
    })
}

/// `POST /team/member/delete` — remove a user from a team. Clears `team_id`
/// on every key owned by the user that currently belongs to this team.
pub async fn member_delete(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Json(req): Json<MemberRequest>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    let (state, db) = load_db(&shared, &auth).await?;

    if db.get_team(&req.team_id).await.map_err(db_err)?.is_none() {
        return Err(ApiError::not_found(format!(
            "team '{}' not found",
            req.team_id
        )));
    }
    if db.get_user(&req.user_id).await.map_err(db_err)?.is_none() {
        return Err(ApiError::not_found(format!(
            "user '{}' not found",
            req.user_id
        )));
    }

    let n = reassign_user_keys(&state, &db, &req.user_id, Some(&req.team_id), None).await?;
    invalidate_cache_key(&state, &format!("vk_team:{}", req.team_id)).await;
    Ok(Json(json!({ "status": "ok", "keys_affected": n })).into_response())
}

/// Reassign (or clear) the team on a user's keys using a single scoped bulk
/// `UPDATE` (atomic, no unbounded `list_keys()` scan).
///
/// - `filter_team = Some(t)`: only touch keys whose current `team_id` is `t`
///   (used by `member/delete` so only this team's membership is dropped).
/// - `filter_team = None`: touch every key owned by the user (used by
///   `member/add`).
/// - `new_team = Some(t)`: set the key's `team_id` to `t`.
/// - `new_team = None`: clear the key's `team_id`.
async fn reassign_user_keys(
    state: &std::sync::Arc<crate::state::AppState>,
    db: &oxllm_db::Database,
    user_id: &str,
    filter_team: Option<&str>,
    new_team: Option<&str>,
) -> Result<u64, ApiError> {
    let now = oxllm_db::now_iso();
    let mut hashes = db
        .list_key_hashes_for_user(user_id, filter_team)
        .await
        .map_err(db_err)?;
    let affected = db
        .set_team_for_user_keys(user_id, filter_team, new_team, &now)
        .await
        .map_err(db_err)?;

    // Invalidate only AFTER the bulk UPDATE has committed. Otherwise a miss
    // can refill the old team assignment during the select/update window.
    // Cover both pre/post hashes so rotations or new keys concurrent with the
    // bulk update receive the same eviction/broadcast. The DB-versioned auth
    // protocol fences late fills and remains correct even if this cleanup's
    // post-read, cache deletion, or pub/sub broadcast is unavailable.
    match db.list_key_hashes_for_user(user_id, None).await {
        Ok(current) => hashes.extend(current),
        Err(error) => {
            tracing::warn!(
                error = %error,
                user_id,
                "unable to enumerate post-reassignment auth cache cleanup"
            );
        }
    }
    hashes.sort_unstable();
    hashes.dedup();
    for hash in hashes {
        invalidate_cache_key(state, &format!("vk:{hash}")).await;
    }
    Ok(affected)
}

// ---------------------------------------------------------------------------
// Team model allowlist add / delete
// ---------------------------------------------------------------------------

/// Request body for `POST /team/model/add` and `POST /team/model/delete`.
#[derive(Debug, Deserialize)]
pub struct TeamModelsRequest {
    pub team_id: String,
    pub models: Vec<String>,
}

/// `POST /team/model/add` — append unique models to the team's allowed list.
///
/// Concurrency: like [`update_team`], this performs a read-modify-write
/// (`get_team` → merge → `update_team`) with no `SELECT … FOR UPDATE` or
/// version column. Two concurrent `team/model/add` calls (or an add racing a
/// delete) can clobber each other's change (last writer wins). The window is
/// small and the actors are master-key operators, so impact is bounded; a
/// complete fix needs an atomic SQL merge (`json_insert`) or optimistic
/// concurrency (version column checked in the `WHERE`, 409 on stale).
pub async fn team_model_add(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Json(req): Json<TeamModelsRequest>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    let (state, db) = load_db(&shared, &auth).await?;

    let mut record = db
        .get_team(&req.team_id)
        .await
        .map_err(db_err)?
        .ok_or_else(|| ApiError::not_found(format!("team '{}' not found", req.team_id)))?;

    let mut current = parse_models_json(record.models.as_deref());
    for m in normalize_models(&req.models) {
        if !current.iter().any(|x| x == &m) {
            current.push(m);
        }
    }
    record.models = serialize_models(Some(current.clone()));
    record.updated_at = oxllm_db::now_iso();
    let updated = db.update_team(&record).await.map_err(db_err)?;
    if !updated {
        return Err(ApiError::not_found(format!(
            "team '{}' not found",
            record.team_id
        )));
    }
    invalidate_cache_key(&state, &format!("vk_team:{}", record.team_id)).await;

    Ok(Json(json!({
        "status": "ok",
        "team_id": record.team_id,
        "models": current,
    }))
    .into_response())
}

/// `POST /team/model/delete` — remove models from the team's allowed list.
///
/// Concurrency: same read-modify-write caveat as [`team_model_add`] — two
/// concurrent calls can clobber each other (last writer wins).
pub async fn team_model_delete(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Json(req): Json<TeamModelsRequest>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    let (state, db) = load_db(&shared, &auth).await?;

    let mut record = db
        .get_team(&req.team_id)
        .await
        .map_err(db_err)?
        .ok_or_else(|| ApiError::not_found(format!("team '{}' not found", req.team_id)))?;

    let remove: Vec<String> = normalize_models(&req.models);
    let current: Vec<String> = parse_models_json(record.models.as_deref())
        .into_iter()
        .filter(|m| !remove.iter().any(|r| r == m))
        .collect();
    record.models = serialize_models(Some(current.clone()));
    record.updated_at = oxllm_db::now_iso();
    let updated = db.update_team(&record).await.map_err(db_err)?;
    if !updated {
        return Err(ApiError::not_found(format!(
            "team '{}' not found",
            record.team_id
        )));
    }
    invalidate_cache_key(&state, &format!("vk_team:{}", record.team_id)).await;

    Ok(Json(json!({
        "status": "ok",
        "team_id": record.team_id,
        "models": current,
    }))
    .into_response())
}

fn parse_models_json(raw: Option<&str>) -> Vec<String> {
    let Some(s) = raw else {
        return Vec::new();
    };
    match serde_json::from_str::<Vec<String>>(s) {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!(raw = %s, error = %e, "corrupt models JSON column, treating as empty");
            Vec::new()
        }
    }
}

fn normalize_models(models: &[String]) -> Vec<String> {
    models
        .iter()
        .filter_map(|s| {
            let trimmed = s.trim().to_string();
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed)
            }
        })
        .collect()
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

#[cfg(all(test, feature = "db"))]
mod tests {
    use super::{normalize_models, parse_models_json, serialize_models};

    #[test]
    fn parse_and_serialize_roundtrip() {
        let raw = serialize_models(Some(vec!["gpt-4o".into(), "claude-*".into()]));
        let parsed = parse_models_json(raw.as_deref());
        assert_eq!(parsed, vec!["gpt-4o", "claude-*"]);
    }

    #[test]
    fn normalize_models_trims() {
        assert_eq!(
            normalize_models(&["  a ".into(), "".into(), "b".into()]),
            vec!["a", "b"]
        );
    }

    // C27/LOW: a corrupt `models` JSON column must not panic or silently
    // overwrite the intended list — it should be treated as an empty base so
    // the merge proceeds deterministically. (The corresponding `tracing::warn!`
    // is not asserted here; it is exercised via the integration suite.)
    #[test]
    fn parse_models_json_treats_corrupt_input_as_empty() {
        assert!(parse_models_json(None).is_empty());
        assert!(parse_models_json(Some("")).is_empty());
        assert!(parse_models_json(Some("not-json")).is_empty());
        assert!(parse_models_json(Some("[1,2,3]")).is_empty());
        assert_eq!(
            parse_models_json(Some(r#"["gpt-4o"]"#)),
            vec!["gpt-4o".to_string()]
        );
    }

    // C27/LOW: clear_rpm / clear_tpm on UpdateTeamRequest default to false so
    // existing callers are unaffected, and parse to true when supplied.
    #[test]
    fn update_team_request_clear_rpm_tpm_default_false() {
        let req: super::UpdateTeamRequest =
            serde_json::from_str(r#"{"team_id":"t-1"}"#).expect("parse");
        assert!(!req.clear_rpm);
        assert!(!req.clear_tpm);
    }

    #[test]
    fn update_team_request_clear_rpm_tpm_parses_true() {
        let req: super::UpdateTeamRequest =
            serde_json::from_str(r#"{"team_id":"t-1","clear_rpm":true,"clear_tpm":true}"#)
                .expect("parse");
        assert!(req.clear_rpm);
        assert!(req.clear_tpm);
    }
}
