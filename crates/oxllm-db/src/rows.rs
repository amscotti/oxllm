//! Mappers from `sqlx` rows to the persisted record types.

use anyhow::Result;
use sqlx::Row;
use sqlx::any::AnyRow;

use crate::types::{
    BudgetRecord, CustomerRecord, KeyRecord, ModelConfigRecord, OrganizationRecord, ProjectRecord,
    SkillRecord, SpendLog, TagRecord, TeamRecord, UserRecord,
};

/// Decode an integer column as `i64` across backends.
///
/// The `sqlx::Any` layer maps values strictly by their wire type: `i64`
/// decodes only from `BIGINT` (INT8) and `i32` only from `INTEGER` (INT4).
/// SQLite always stores integers as INT8, but Postgres `INTEGER` columns
/// (`is_active`, `rpm`, `tpm`, token counts, …) come back as INT4, so a
/// plain `try_get::<i64>` fails there. Try INT8 first (the SQLite path)
/// and fall back to INT4 + widening, so one mapper serves both backends.
pub(crate) fn get_int(row: &AnyRow, col: &str) -> Result<i64> {
    if let Ok(v) = row.try_get::<i64, _>(col) {
        return Ok(v);
    }
    Ok(i64::from(row.try_get::<i32, _>(col)?))
}

/// Nullable variant of [`get_int`].
pub(crate) fn get_opt_int(row: &AnyRow, col: &str) -> Result<Option<i64>> {
    if let Ok(v) = row.try_get::<Option<i64>, _>(col) {
        return Ok(v);
    }
    Ok(row.try_get::<Option<i32>, _>(col)?.map(i64::from))
}

pub(crate) fn row_to_key(row: &AnyRow) -> Result<KeyRecord> {
    Ok(KeyRecord {
        key_id: row.try_get("key_id")?,
        key_hash: row.try_get("key_hash")?,
        key_alias: row.try_get("key_alias")?,
        team_id: row.try_get("team_id")?,
        user_id: row.try_get("user_id")?,
        org_id: row.try_get("org_id")?,
        project_id: row.try_get("project_id")?,
        max_budget: row.try_get("max_budget")?,
        budget_duration: row.try_get("budget_duration")?,
        rpm: get_opt_int(row, "rpm")?,
        tpm: get_opt_int(row, "tpm")?,
        created_at: row.try_get("created_at")?,
        updated_at: row.try_get("updated_at")?,
        expires_at: row.try_get("expires_at")?,
        is_active: get_int(row, "is_active")? != 0,
        models: row.try_get("models")?,
        allowed_cache_controls: row.try_get("allowed_cache_controls")?,
    })
}

pub(crate) fn row_to_org(row: &AnyRow) -> Result<OrganizationRecord> {
    Ok(OrganizationRecord {
        org_id: row.try_get("org_id")?,
        org_name: row.try_get("org_name")?,
        max_budget: row.try_get("max_budget")?,
        budget_duration: row.try_get("budget_duration")?,
        rpm: get_opt_int(row, "rpm")?,
        tpm: get_opt_int(row, "tpm")?,
        created_at: row.try_get("created_at")?,
        updated_at: row.try_get("updated_at")?,
        models: row.try_get("models")?,
        metadata: row.try_get("metadata")?,
        soft_budget: row.try_get("soft_budget")?,
        max_parallel_requests: get_opt_int(row, "max_parallel_requests")?,
        model_max_budget: row.try_get("model_max_budget")?,
    })
}

pub(crate) fn row_to_project(row: &AnyRow) -> Result<ProjectRecord> {
    Ok(ProjectRecord {
        project_id: row.try_get("project_id")?,
        project_name: row.try_get("project_name")?,
        org_id: row.try_get("org_id")?,
        team_id: row.try_get("team_id")?,
        max_budget: row.try_get("max_budget")?,
        budget_duration: row.try_get("budget_duration")?,
        rpm: get_opt_int(row, "rpm")?,
        tpm: get_opt_int(row, "tpm")?,
        created_at: row.try_get("created_at")?,
        updated_at: row.try_get("updated_at")?,
    })
}

pub(crate) fn row_to_team(row: &AnyRow) -> Result<TeamRecord> {
    Ok(TeamRecord {
        team_id: row.try_get("team_id")?,
        team_name: row.try_get("team_name")?,
        max_budget: row.try_get("max_budget")?,
        budget_duration: row.try_get("budget_duration")?,
        rpm: get_opt_int(row, "rpm")?,
        tpm: get_opt_int(row, "tpm")?,
        created_at: row.try_get("created_at")?,
        updated_at: row.try_get("updated_at")?,
        models: row.try_get("models")?,
        blocked: get_int(row, "blocked")? != 0,
    })
}

pub(crate) fn row_to_user(row: &AnyRow) -> Result<UserRecord> {
    Ok(UserRecord {
        user_id: row.try_get("user_id")?,
        user_name: row.try_get("user_name")?,
        max_budget: row.try_get("max_budget")?,
        budget_duration: row.try_get("budget_duration")?,
        created_at: row.try_get("created_at")?,
        updated_at: row.try_get("updated_at")?,
        models: row.try_get("models")?,
    })
}

pub(crate) fn row_to_spend(row: &AnyRow) -> Result<SpendLog> {
    Ok(SpendLog {
        id: get_opt_int(row, "id")?,
        timestamp: row.try_get("timestamp")?,
        request_id: row.try_get("request_id")?,
        session_id: row.try_get("session_id")?,
        model: row.try_get("model")?,
        provider: row.try_get("provider")?,
        deployment_id: row.try_get("deployment_id")?,
        requested_model: row.try_get("requested_model")?,
        model_group: row.try_get("model_group")?,
        prompt_tokens: get_opt_int(row, "prompt_tokens")?,
        completion_tokens: get_opt_int(row, "completion_tokens")?,
        total_tokens: get_opt_int(row, "total_tokens")?,
        cost: row.try_get("cost")?,
        key_id: row.try_get("key_id")?,
        team_id: row.try_get("team_id")?,
        user_id: row.try_get("user_id")?,
        end_user_id: row.try_get("end_user_id")?,
        tag_name: row.try_get("tag_name")?,
        tags: row.try_get("tags")?,
        status: row.try_get("status")?,
        latency_ms: get_opt_int(row, "latency_ms")?,
    })
}

pub(crate) fn row_to_model(row: &AnyRow) -> Result<ModelConfigRecord> {
    let litellm_str: String = row.try_get("oxllm_params")?;
    let info_str: Option<String> = row.try_get("model_info")?;
    let oxllm_params: serde_json::Value = serde_json::from_str(&litellm_str)?;
    let model_info = match info_str {
        Some(s) if !s.is_empty() => Some(serde_json::from_str(&s)?),
        _ => None,
    };
    Ok(ModelConfigRecord {
        model_id: row.try_get("model_id")?,
        model_name: row.try_get("model_name")?,
        oxllm_params,
        model_info,
        created_at: row.try_get("created_at")?,
        updated_at: row.try_get("updated_at")?,
    })
}

pub(crate) fn row_to_customer(row: &AnyRow) -> Result<CustomerRecord> {
    Ok(CustomerRecord {
        end_user_id: row.try_get("end_user_id")?,
        alias: row.try_get("alias")?,
        email: row.try_get("email")?,
        max_budget: row.try_get("max_budget")?,
        spend: row.try_get("spend")?,
        budget_duration: row.try_get("budget_duration")?,
        blocked: get_int(row, "blocked")? != 0,
        model_blocklist: row.try_get("model_blocklist")?,
        created_at: row.try_get("created_at")?,
        updated_at: row.try_get("updated_at")?,
    })
}

pub(crate) fn row_to_budget(row: &AnyRow) -> Result<BudgetRecord> {
    Ok(BudgetRecord {
        budget_id: row.try_get("budget_id")?,
        max_budget: row.try_get("max_budget")?,
        soft_budget: row.try_get("soft_budget")?,
        max_parallel_requests: get_opt_int(row, "max_parallel_requests")?,
        tpm_limit: get_opt_int(row, "tpm_limit")?,
        rpm_limit: get_opt_int(row, "rpm_limit")?,
        budget_duration: row.try_get("budget_duration")?,
        model_max_budget: row.try_get("model_max_budget")?,
        created_at: row.try_get("created_at")?,
        updated_at: row.try_get("updated_at")?,
    })
}

pub(crate) fn row_to_tag(row: &AnyRow) -> Result<TagRecord> {
    Ok(TagRecord {
        name: row.try_get("name")?,
        description: row.try_get("description")?,
        models: row.try_get("models")?,
        model_info: row.try_get("model_info")?,
        spend: row.try_get("spend")?,
        max_budget: row.try_get("max_budget")?,
        created_at: row.try_get("created_at")?,
        updated_at: row.try_get("updated_at")?,
    })
}

pub(crate) fn row_to_skill(row: &AnyRow) -> Result<SkillRecord> {
    Ok(SkillRecord {
        id: row.try_get("id")?,
        name: row.try_get("name")?,
        description: row.try_get("description")?,
        content: row.try_get("content")?,
        version: row.try_get("version")?,
        model_pattern: row.try_get("model_pattern")?,
        team_id: row.try_get("team_id")?,
        created_at: row.try_get("created_at")?,
        updated_at: row.try_get("updated_at")?,
    })
}
