//! `models` table CRUD: model deployments persisted via the management API.

use anyhow::Result;

use crate::rows::row_to_model;
use crate::types::ModelConfigRecord;
use crate::{Database, ModelNameConflict};

fn model_write_error(error: sqlx::Error, model_name: &str) -> anyhow::Error {
    let alias_conflict = error.as_database_error().is_some_and(|error| {
        error.is_unique_violation()
            && (error.constraint() == Some("model_aliases_name")
                || error.message().contains("model_aliases.model_name"))
    });
    if alias_conflict {
        ModelNameConflict {
            model_name: model_name.to_string(),
        }
        .into()
    } else {
        error.into()
    }
}

impl Database {
    /// Atomically claim a previously unused alias. Legacy duplicates survive
    /// migration, but further creates/renames cannot add another deployment.
    pub async fn create_model(&self, model: &ModelConfigRecord) -> Result<()> {
        let q = format!(
            "INSERT INTO models (model_id, model_name, oxllm_params, model_info, created_at, \
             updated_at) VALUES ({}, {}, {}, {}, {}, {})",
            self.ph(1),
            self.ph(2),
            self.ph(3),
            self.ph(4),
            self.ph(5),
            self.ph(6)
        );
        let params = serde_json::to_string(&model.oxllm_params)?;
        let info = match &model.model_info {
            Some(v) => Some(serde_json::to_string(v)?),
            None => None,
        };
        sqlx::query(&q)
            .bind(&model.model_id)
            .bind(&model.model_name)
            .bind(&params)
            .bind(info)
            .bind(&model.created_at)
            .bind(&model.updated_at)
            .execute(self.pool())
            .await
            .map_err(|error| model_write_error(error, &model.model_name))?;
        Ok(())
    }

    pub async fn get_model(&self, model_id: &str) -> Result<Option<ModelConfigRecord>> {
        let q = format!(
            "SELECT model_id, model_name, oxllm_params, model_info, created_at, updated_at \
             FROM models WHERE model_id = {}",
            self.ph(1)
        );
        let row = sqlx::query(&q)
            .bind(model_id)
            .fetch_optional(self.pool())
            .await?;
        row.map(|r| row_to_model(&r)).transpose()
    }

    /// List model deployments newest-first, capped at `limit` rows (default
    /// 1 000 when `None`). The upper clamp is 1 001 so a handler can
    /// request one extra row as a sentinel to detect truncation without
    /// the result being silently clamped back down. See
    /// [`crate::Database::list_keys`] for why the cap lives in the repo.
    pub async fn list_models_db(&self, limit: Option<i64>) -> Result<Vec<ModelConfigRecord>> {
        let limit = limit.unwrap_or(1000).clamp(1, 1001);
        let q = format!(
            "SELECT model_id, model_name, oxllm_params, model_info, created_at, updated_at \
             FROM models ORDER BY created_at DESC, model_id DESC LIMIT {}",
            self.ph(1)
        );
        let rows = sqlx::query(&q).bind(limit).fetch_all(self.pool()).await?;
        rows.iter().map(row_to_model).collect()
    }

    /// List **all** model deployments newest-first with no row cap. Intended
    /// for internal full-scan merge paths (e.g. [`Database::list_models_db`]
    /// consumers that build the routable deployment list) that must see every
    /// row — the 1 000-row cap on [`Database::list_models_db`] is an
    /// API-pagination safeguard, not appropriate for a merge that would
    /// silently drop the oldest DB-stored models (making them unroutable with
    /// no warning). The management-API list endpoint keeps using the capped
    /// variant so human-facing pagination stays bounded.
    pub async fn list_all_models_db(&self) -> Result<Vec<ModelConfigRecord>> {
        let q = "SELECT model_id, model_name, oxllm_params, model_info, created_at, updated_at \
                 FROM models ORDER BY created_at DESC, model_id DESC";
        let rows = sqlx::query(q).fetch_all(self.pool()).await?;
        rows.iter().map(row_to_model).collect()
    }

    /// Update all mutable fields of a model row. `model_id` identifies the row;
    /// the remaining fields are overwritten from `model`. A rename claims its
    /// new alias atomically; an unchanged legacy duplicate name remains editable.
    pub async fn update_model(&self, model_id: &str, model: &ModelConfigRecord) -> Result<()> {
        let q = format!(
            "UPDATE models SET model_name = {}, oxllm_params = {}, model_info = {}, \
             updated_at = {} WHERE model_id = {}",
            self.ph(1),
            self.ph(2),
            self.ph(3),
            self.ph(4),
            self.ph(5)
        );
        let params = serde_json::to_string(&model.oxllm_params)?;
        let info = match &model.model_info {
            Some(v) => Some(serde_json::to_string(v)?),
            None => None,
        };
        sqlx::query(&q)
            .bind(&model.model_name)
            .bind(&params)
            .bind(info)
            .bind(&model.updated_at)
            .bind(model_id)
            .execute(self.pool())
            .await
            .map_err(|error| model_write_error(error, &model.model_name))?;
        Ok(())
    }

    /// Delete the row identified by `model_id`. Returns the number of rows
    /// actually removed (0 when no such row exists). Surfacing the count lets
    /// the management handler distinguish a genuine 404 from a successful
    /// delete without a separate existence pre-check, eliminating the
    /// check-then-delete TOCTOU window (two concurrent deletes of the same id
    /// both returning cleanly instead of one 500-ing).
    pub async fn delete_model(&self, model_id: &str) -> Result<u64> {
        let q = format!("DELETE FROM models WHERE model_id = {}", self.ph(1));
        let result = sqlx::query(&q).bind(model_id).execute(self.pool()).await?;
        Ok(result.rows_affected())
    }
}
