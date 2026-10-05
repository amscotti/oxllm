//! `tags` table CRUD.

use anyhow::Result;

use crate::Database;
use crate::binds::bind_optional_f64;
use crate::rows::row_to_tag;
use crate::types::TagRecord;

impl Database {
    pub async fn create_tag(&self, tag: &TagRecord) -> Result<()> {
        let q = format!(
            "INSERT INTO tags (name, description, models, model_info, spend, max_budget, \
 created_at, updated_at) VALUES ({}, {}, {}, {}, {}, {}, {}, {})",
            self.ph(1),
            self.ph(2),
            self.ph(3),
            self.ph(4),
            self.ph(5),
            self.ph(6),
            self.ph(7),
            self.ph(8)
        );
        let query = sqlx::query(&q)
            .bind(&tag.name)
            .bind(tag.description.clone())
            .bind(tag.models.clone())
            .bind(tag.model_info.clone());
        let query = bind_optional_f64(query, tag.spend);
        bind_optional_f64(query, tag.max_budget)
            .bind(&tag.created_at)
            .bind(&tag.updated_at)
            .execute(self.pool())
            .await?;
        Ok(())
    }

    pub async fn get_tag(&self, name: &str) -> Result<Option<TagRecord>> {
        let q = format!(
            "SELECT name, description, models, model_info, spend, max_budget, created_at, \
 updated_at FROM tags WHERE name = {}",
            self.ph(1)
        );
        let row = sqlx::query(&q)
            .bind(name)
            .fetch_optional(self.pool())
            .await?;
        row.map(|r| row_to_tag(&r)).transpose()
    }

    /// List tags newest-first, capped at `limit` rows (default 1 000 when
    /// `None`). The upper clamp is 1 001 so a handler can request one extra
    /// row as a sentinel to detect truncation without the result being
    /// silently clamped back down. See [`crate::Database::list_keys`] for
    /// why the cap lives in the repo.
    pub async fn list_tags(&self, limit: Option<i64>) -> Result<Vec<TagRecord>> {
        let limit = limit.unwrap_or(1000).clamp(1, 1001);
        let q = format!(
            "SELECT name, description, models, model_info, spend, max_budget, created_at, \
 updated_at FROM tags ORDER BY created_at DESC LIMIT {}",
            self.ph(1)
        );
        let rows = sqlx::query(&q).bind(limit).fetch_all(self.pool()).await?;
        rows.iter().map(row_to_tag).collect()
    }

    /// Overwrite the mutable fields of the tag identified by `tag.name`.
    /// Returns `true` when a row was matched and updated, `false` when no
    /// row matched `name` (the row was deleted between the caller's
    /// `get_tag` and this `UPDATE`). The caller should map `false` to a 404
    /// so a racy delete is not surfaced as a stale `200 OK`.
    pub async fn update_tag(&self, tag: &TagRecord) -> Result<bool> {
        // `spend` is owned by `increment_tag_spend` (the spend writer).
        // Writing the caller's snapshot here would clobber a concurrent increment.
        let q = format!(
            "UPDATE tags SET description = {}, models = {}, model_info = {}, \
max_budget = {}, updated_at = {} WHERE name = {}",
            self.ph(1),
            self.ph(2),
            self.ph(3),
            self.ph(4),
            self.ph(5),
            self.ph(6)
        );
        let query = sqlx::query(&q)
            .bind(tag.description.clone())
            .bind(tag.models.clone())
            .bind(tag.model_info.clone());
        let res = bind_optional_f64(query, tag.max_budget)
            .bind(&tag.updated_at)
            .bind(&tag.name)
            .execute(self.pool())
            .await?;
        Ok(res.rows_affected() > 0)
    }

    pub async fn delete_tag(&self, name: &str) -> Result<()> {
        let q = format!("DELETE FROM tags WHERE name = {}", self.ph(1));
        sqlx::query(&q).bind(name).execute(self.pool()).await?;
        Ok(())
    }

    /// Atomically add `delta` to the `spend` column of the tag identified
    /// by `name`. Returns `true` when a row was matched and updated, `false`
    /// when no such tag exists.
    ///
    /// For explicit adjustments; metering writes use `insert_spend_events`
    /// to update counters atomically with logs and exact tag memberships.
    /// Hot-path budget enforcement uses the cache counter `spend:tag:{name}`
    /// via `BudgetHook`. A negative `delta` (refund) is valid.
    pub async fn increment_tag_spend(&self, name: &str, delta: f64) -> Result<bool> {
        let q = format!(
            "UPDATE tags SET spend = COALESCE(spend, 0) + {} WHERE name = {}",
            self.ph(1),
            self.ph(2)
        );
        let res = sqlx::query(&q)
            .bind(delta)
            .bind(name)
            .execute(self.pool())
            .await?;
        Ok(res.rows_affected() > 0)
    }
}
