//! Audit log and request log management endpoints.
//!
//! - `GET /audit/logs` — paginated audit trail (sourced from spend_logs)
//! - `GET /request/logs/{request_id}` — single spend log by request_id

use axum::extract::{Path, Query, State};
use axum::response::{IntoResponse, Response};
use axum::{Extension, Json};
use serde::Deserialize;
use serde_json::json;

use crate::auth::AuthUser;
use crate::error::ApiError;
use crate::state::SharedState;

use super::{db_err, load_db, require_master};

/// Maximum page size accepted by `GET /audit/logs`. User-supplied `limit`
/// values are clamped into `1..=MAX_LIMIT` so a negative or huge limit can
/// never reach the SQL layer (SQLite `LIMIT -1` removes the row cap
/// entirely → full-table dump).
const MAX_LIMIT: i64 = 500;

/// Default page size when `limit` is not supplied.
const DEFAULT_LIMIT: i64 = 100;

#[derive(Debug, Deserialize)]
pub struct AuditLogsQuery {
    #[serde(default)]
    pub start_date: Option<String>,
    #[serde(default)]
    pub end_date: Option<String>,
    #[serde(default)]
    pub key_id: Option<String>,
    #[serde(default)]
    pub team_id: Option<String>,
    #[serde(default)]
    pub limit: Option<i64>,
    #[serde(default)]
    pub offset: Option<i64>,
}

impl From<&AuditLogsQuery> for oxllm_db::SpendFilter {
    fn from(q: &AuditLogsQuery) -> Self {
        oxllm_db::SpendFilter {
            start_date: q.start_date.clone(),
            end_date: q.end_date.clone(),
            key_id: q.key_id.clone(),
            team_id: q.team_id.clone(),
            model: None,
            tag_name: None,
            limit: q.limit,
            offset: q.offset,
            sort_by: None,
            sort_order: None,
            request_id: None,
        }
    }
}

/// `GET /audit/logs` — paginated audit trail from spend_logs.
pub async fn list_audit_logs(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Query(q): Query<AuditLogsQuery>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    let (_state, db) = load_db(&shared, &auth).await?;

    // Clamp user-supplied paging before it reaches SQL.
    let limit = q.limit.unwrap_or(DEFAULT_LIMIT).clamp(1, MAX_LIMIT);
    let offset = q.offset.unwrap_or(0).max(0);

    let mut filter: oxllm_db::SpendFilter = (&q).into();
    filter.limit = Some(limit);
    filter.offset = Some(offset);
    // Expand a bare `YYYY-MM-DD` end date to the final millisecond of the
    // day so the bound is inclusive (otherwise the entire end day is
    // dropped because the bare date sorts before any same-day timestamp).
    filter.end_date = super::spend::inclusive_end_date(filter.end_date.as_deref());

    let logs = db.query_spend(&filter).await.map_err(db_err)?;
    // Use the same filter the row query uses so `total` reflects the
    // matching set (counting only on `key_id`+`start_date` overcounted
    // whenever `end_date` or `team_id` was supplied).
    let total = db.count_spend_filtered(&filter).await.map_err(db_err)?;

    let entries: Vec<serde_json::Value> = logs
        .iter()
        .map(|log| {
            json!({
                "id": log.id,
                "timestamp": log.timestamp,
                "action": log.status.as_deref().unwrap_or("spend_log"),
                "resource_type": "spend",
                "resource_id": log.request_id,
                "actor_key_id": log.key_id,
                "details": {
                    "model": log.model,
                    "provider": log.provider,
                    "cost": log.cost,
                    "prompt_tokens": log.prompt_tokens,
                    "completion_tokens": log.completion_tokens,
                    "total_tokens": log.total_tokens,
                    "team_id": log.team_id,
                    "user_id": log.user_id,
                },
            })
        })
        .collect();

    Ok(Json(json!({
        "data": entries,
        "paging": {
            "limit": limit,
            "offset": offset,
            "total": total,
        },
    }))
    .into_response())
}

/// `GET /request/logs/{request_id}` — single spend log by request_id.
pub async fn get_request_log(
    State(shared): State<SharedState>,
    Extension(auth): Extension<AuthUser>,
    Path(request_id): Path<String>,
) -> Result<Response, ApiError> {
    require_master(&auth)?;
    let (_state, db) = load_db(&shared, &auth).await?;

    // Direct indexed lookup. The previous implementation fetched the 10k
    // newest rows and scanned them in memory, so any request_id older than
    // that window returned a false 404.
    let log = db
        .query_spend_by_request_id(&request_id)
        .await
        .map_err(db_err)?;

    match log {
        Some(entry) => Ok(Json(json!({
            "id": entry.id,
            "request_id": entry.request_id,
            "timestamp": entry.timestamp,
            "model": entry.model,
            "provider": entry.provider,
            "prompt_tokens": entry.prompt_tokens,
            "completion_tokens": entry.completion_tokens,
            "total_tokens": entry.total_tokens,
            "cost": entry.cost,
            "key_id": entry.key_id,
            "team_id": entry.team_id,
            "user_id": entry.user_id,
            "status": entry.status,
            "latency_ms": entry.latency_ms,
        }))
        .into_response()),
        None => Err(ApiError::not_found(format!(
            "Request log with request_id={request_id} not found"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use arc_swap::ArcSwap;
    use axum::http::StatusCode;

    fn master_auth() -> AuthUser {
        AuthUser::master(crate::auth::hash_key(b"sk-master"))
    }

    fn audit_query(limit: Option<i64>, offset: Option<i64>) -> AuditLogsQuery {
        AuditLogsQuery {
            start_date: None,
            end_date: None,
            key_id: None,
            team_id: None,
            limit,
            offset,
        }
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

    fn spend_log(request_id: Option<&str>) -> oxllm_db::SpendLog {
        oxllm_db::SpendLog {
            id: None,
            timestamp: oxllm_db::now_iso(),
            request_id: request_id.map(str::to_string),
            model: "gpt-4o".to_string(),
            provider: "openai".to_string(),
            prompt_tokens: Some(10),
            completion_tokens: Some(5),
            total_tokens: Some(15),
            cost: Some(0.001),
            key_id: Some("k-1".to_string()),
            team_id: None,
            user_id: None,
            end_user_id: None,
            tag_name: None,
            tags: None,
            status: Some("success".to_string()),
            latency_ms: Some(42),
        }
    }

    async fn body_json(resp: Response) -> serde_json::Value {
        let bytes = axum::body::to_bytes(resp.into_body(), 1024 * 1024)
            .await
            .expect("body bytes");
        serde_json::from_slice(&bytes).expect("json body")
    }

    #[tokio::test]
    async fn get_request_log_returns_row_by_request_id() {
        let (shared, db) = shared_with_db().await;
        db.insert_spend_log(&spend_log(Some("req-abc")))
            .await
            .expect("insert");

        let resp = get_request_log(
            State(shared),
            Extension(master_auth()),
            Path("req-abc".to_string()),
        )
        .await
        .expect("handler");
        assert_eq!(resp.status(), StatusCode::OK);
        let body = body_json(resp).await;
        assert_eq!(body["request_id"], "req-abc");
        assert_eq!(body["model"], "gpt-4o");
    }

    #[tokio::test]
    async fn get_request_log_missing_request_id_is_404() {
        let (shared, db) = shared_with_db().await;
        // A different row exists; the missing id must still 404.
        db.insert_spend_log(&spend_log(Some("req-other")))
            .await
            .expect("insert");

        let err = get_request_log(
            State(shared),
            Extension(master_auth()),
            Path("req-missing".to_string()),
        )
        .await
        .expect_err("must 404");
        assert_eq!(err.status, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn get_request_log_rejects_non_master() {
        let (shared, _db) = shared_with_db().await;
        let err = get_request_log(
            State(shared),
            Extension(AuthUser::open()),
            Path("req-abc".to_string()),
        )
        .await
        .expect_err("must 403");
        assert_eq!(err.status, StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn list_audit_logs_rejects_non_master() {
        let (shared, _db) = shared_with_db().await;
        let err = list_audit_logs(
            State(shared),
            Extension(AuthUser::open()),
            Query(audit_query(None, None)),
        )
        .await
        .expect_err("must 403");
        assert_eq!(err.status, StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn list_audit_logs_clamps_limit_and_offset() {
        let (shared, db) = shared_with_db().await;
        db.insert_spend_log(&spend_log(Some("req-1")))
            .await
            .expect("insert");
        db.insert_spend_log(&spend_log(Some("req-2")))
            .await
            .expect("insert");

        // Negative limit/offset are clamped into range (not passed to SQL).
        let resp = list_audit_logs(
            State(shared.clone()),
            Extension(master_auth()),
            Query(audit_query(Some(-1), Some(-10))),
        )
        .await
        .expect("handler");
        assert_eq!(resp.status(), StatusCode::OK);
        let body = body_json(resp).await;
        assert_eq!(body["paging"]["limit"], 1);
        assert_eq!(body["paging"]["offset"], 0);
        assert_eq!(body["paging"]["total"], 2);
        assert_eq!(body["data"].as_array().expect("array").len(), 1);

        // A huge limit is clamped to MAX_LIMIT.
        let resp = list_audit_logs(
            State(shared),
            Extension(master_auth()),
            Query(audit_query(Some(1_000_000), None)),
        )
        .await
        .expect("handler");
        let body = body_json(resp).await;
        assert_eq!(body["paging"]["limit"], MAX_LIMIT);
        assert_eq!(body["data"].as_array().expect("array").len(), 2);
    }

    #[tokio::test]
    async fn list_audit_logs_response_has_data_and_paging() {
        let (shared, _db) = shared_with_db().await;
        let resp = list_audit_logs(
            State(shared),
            Extension(master_auth()),
            Query(audit_query(None, None)),
        )
        .await
        .expect("handler");
        let body = body_json(resp).await;
        assert!(body.get("data").is_some(), "house shape uses `data`");
        assert!(
            body.get("audit_logs").is_none(),
            "old `audit_logs` key is gone"
        );
        assert_eq!(body["paging"]["limit"], DEFAULT_LIMIT);
        assert_eq!(body["paging"]["offset"], 0);
        assert_eq!(body["paging"]["total"], 0);
    }

    fn audit_query_with_team(team_id: Option<&str>) -> AuditLogsQuery {
        AuditLogsQuery {
            start_date: None,
            end_date: None,
            key_id: None,
            team_id: team_id.map(str::to_string),
            limit: None,
            offset: None,
        }
    }

    fn spend_log_for_team(request_id: &str, team_id: Option<&str>) -> oxllm_db::SpendLog {
        oxllm_db::SpendLog {
            id: None,
            timestamp: oxllm_db::now_iso(),
            request_id: Some(request_id.to_string()),
            model: "gpt-4o".to_string(),
            provider: "openai".to_string(),
            prompt_tokens: Some(10),
            completion_tokens: Some(5),
            total_tokens: Some(15),
            cost: Some(0.001),
            key_id: Some("k-1".to_string()),
            team_id: team_id.map(str::to_string),
            user_id: None,
            end_user_id: None,
            tag_name: None,
            tags: None,
            status: Some("success".to_string()),
            latency_ms: Some(42),
        }
    }

    // MEDIUM: paging.total must respect the team_id filter (previously it
    // was computed only on key_id+start_date, so it overcounted whenever
    // team_id was supplied).
    #[tokio::test]
    async fn list_audit_logs_total_respects_team_filter() {
        let (shared, db) = shared_with_db().await;
        for i in 0..3 {
            db.insert_spend_log(&spend_log_for_team(&format!("team-a-{i}"), Some("team-a")))
                .await
                .expect("insert");
        }
        for i in 0..2 {
            db.insert_spend_log(&spend_log_for_team(&format!("team-b-{i}"), Some("team-b")))
                .await
                .expect("insert");
        }

        let resp = list_audit_logs(
            State(shared),
            Extension(master_auth()),
            Query(audit_query_with_team(Some("team-a"))),
        )
        .await
        .expect("handler");
        let body = body_json(resp).await;
        assert_eq!(
            body["paging"]["total"], 3,
            "total must reflect team_id filter"
        );
        assert_eq!(
            body["data"].as_array().expect("array").len(),
            3,
            "data already filtered correctly"
        );
    }

    // MEDIUM: end_date must be inclusive of the end day.
    #[tokio::test]
    async fn list_audit_logs_end_date_is_inclusive_of_end_day() {
        let (shared, db) = shared_with_db().await;
        db.insert_spend_log(&spend_log_for_team("early", None))
            .await
            .expect("insert");
        let mut end_day = spend_log_for_team("end-day", None);
        end_day.timestamp = "2026-07-31T12:00:00Z".to_string();
        db.insert_spend_log(&end_day).await.expect("insert");

        let q = AuditLogsQuery {
            start_date: Some("2026-07-01".to_string()),
            end_date: Some("2026-07-31".to_string()),
            ..audit_query(None, None)
        };
        let resp = list_audit_logs(State(shared), Extension(master_auth()), Query(q))
            .await
            .expect("handler");
        let body = body_json(resp).await;
        assert_eq!(
            body["data"].as_array().expect("array").len(),
            1,
            "end day must be included"
        );
    }

    // LOW: the per-row `details` object must surface `user_id` so the audit
    // trail carries user attribution (consistent with get_request_log, which
    // already exposes user_id).
    #[tokio::test]
    async fn list_audit_logs_details_includes_user_id() {
        let (shared, db) = shared_with_db().await;
        let mut log = spend_log_for_team("req-user", None);
        log.user_id = Some("user-42".to_string());
        db.insert_spend_log(&log).await.expect("insert");

        let resp = list_audit_logs(
            State(shared),
            Extension(master_auth()),
            Query(audit_query(None, None)),
        )
        .await
        .expect("handler");
        let body = body_json(resp).await;
        let entry = &body["data"][0];
        assert_eq!(entry["details"]["user_id"], "user-42");
        // team_id is still present alongside the new user_id field.
        assert!(entry["details"].get("team_id").is_some());
        // The actor identity field is named actor_key_id (the value is the
        // non-secret key id, not a hash digest).
        assert_eq!(entry["actor_key_id"], "k-1");
        assert!(
            entry.get("actor_key_hash").is_none(),
            "renamed field must not appear"
        );
    }
}
