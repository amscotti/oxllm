//! `keys` table CRUD: virtual API key records.

use anyhow::Result;
use sqlx::{Any, Executor, Row};

use crate::binds::bind_optional_f64;
use crate::rows::row_to_key;
use crate::types::KeyRecord;
use crate::{Database, KeyTeamMutation};

/// Reject malformed key records before they reach SQL: an empty `key_id` /
/// `key_hash` would create an unaddressable row (lookups bind the exact id),
/// and a negative budget/rate silently disables enforcement downstream
/// (`None` already means unlimited, so negatives must never be stored).
fn validate_key_record(key: &KeyRecord) -> Result<()> {
    if key.key_id.trim().is_empty() {
        anyhow::bail!("key_id must not be empty");
    }
    if key.key_hash.trim().is_empty() {
        anyhow::bail!("key_hash must not be empty");
    }
    if let Some(b) = key.max_budget {
        if !b.is_finite() || b < 0.0 {
            anyhow::bail!("max_budget must be a finite value >= 0");
        }
    }
    if key.rpm.is_some_and(|v| v < 0) {
        anyhow::bail!("rpm must be >= 0");
    }
    if key.tpm.is_some_and(|v| v < 0) {
        anyhow::bail!("tpm must be >= 0");
    }
    Ok(())
}

impl Database {
    /// Unchecked scope links for imports and fixtures. Native management writes
    /// must use `create_key_scoped` so scope validation shares the commit.
    pub async fn create_key(&self, key: &KeyRecord) -> Result<()> {
        validate_key_record(key)?;
        self.create_key_on(key, self.pool()).await
    }

    /// Create a key only when every parent exists and an owned project's team
    /// matches the key. Validation and insertion hold the same graph lock.
    pub async fn create_key_scoped(&self, key: &KeyRecord) -> Result<()> {
        validate_key_record(key)?;
        let mut tx = self.begin_scope_links().await?;
        self.validate_key_scope_links_on(&mut tx, key).await?;
        self.create_key_on(key, &mut *tx).await?;
        tx.commit().await?;
        Ok(())
    }

    async fn create_key_on<'e>(
        &self,
        key: &KeyRecord,
        executor: impl Executor<'e, Database = Any>,
    ) -> Result<()> {
        let q = format!(
            "INSERT INTO keys (key_id, key_hash, key_alias, team_id, user_id, \
 org_id, project_id, max_budget, budget_duration, rpm, tpm, created_at, updated_at, expires_at, \
 is_active, models, allowed_cache_controls) \
 VALUES ({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {})",
            self.ph(1),
            self.ph(2),
            self.ph(3),
            self.ph(4),
            self.ph(5),
            self.ph(6),
            self.ph(7),
            self.ph(8),
            self.ph(9),
            self.ph(10),
            self.ph(11),
            self.ph(12),
            self.ph(13),
            self.ph(14),
            self.ph(15),
            self.ph(16),
            self.ph(17)
        );
        let query = sqlx::query(&q)
            .bind(&key.key_id)
            .bind(&key.key_hash)
            .bind(key.key_alias.clone())
            .bind(key.team_id.clone())
            .bind(key.user_id.clone())
            .bind(key.org_id.clone())
            .bind(key.project_id.clone());
        bind_optional_f64(query, key.max_budget)
            .bind(key.budget_duration.clone())
            .bind(key.rpm)
            .bind(key.tpm)
            .bind(&key.created_at)
            .bind(&key.updated_at)
            .bind(key.expires_at.clone())
            .bind(if key.is_active { 1_i64 } else { 0_i64 })
            .bind(key.models.clone())
            .bind(key.allowed_cache_controls.clone())
            .execute(executor)
            .await?;
        Ok(())
    }

    pub async fn get_key_by_hash(&self, hash: &str) -> Result<Option<KeyRecord>> {
        let q = format!(
            "SELECT key_id, key_hash, key_alias, team_id, user_id, \
 org_id, project_id, max_budget, budget_duration, rpm, tpm, created_at, updated_at, expires_at, \
 is_active, models, allowed_cache_controls \
 FROM keys WHERE key_hash = {}",
            self.ph(1)
        );
        let row = sqlx::query(&q)
            .bind(hash)
            .fetch_optional(self.pool())
            .await?;
        row.map(|r| row_to_key(&r)).transpose()
    }

    pub async fn get_key(&self, key_id: &str) -> Result<Option<KeyRecord>> {
        self.get_key_on(key_id, self.pool()).await
    }

    async fn get_key_on<'e>(
        &self,
        key_id: &str,
        executor: impl Executor<'e, Database = Any>,
    ) -> Result<Option<KeyRecord>> {
        let q = format!(
            "SELECT key_id, key_hash, key_alias, team_id, user_id, \
 org_id, project_id, max_budget, budget_duration, rpm, tpm, created_at, updated_at, expires_at, \
 is_active, models, allowed_cache_controls \
 FROM keys WHERE key_id = {}",
            self.ph(1)
        );
        let row = sqlx::query(&q)
            .bind(key_id)
            .fetch_optional(executor)
            .await?;
        row.map(|r| row_to_key(&r)).transpose()
    }

    /// List keys newest-first, capped at `limit` rows (default 1 000 when
    /// `None`). Pushing the cap into the SQL `LIMIT` avoids materialising the
    /// whole table on the proxy before the handler truncates — a single
    /// authenticated `GET /key/list` could otherwise OOM the gateway on a
    /// million-row `keys` table. The upper clamp is 1 001 so the handler can
    /// request one extra row as a sentinel to detect truncation without the
    /// result being silently clamped back down.
    pub async fn list_keys(&self, limit: Option<i64>) -> Result<Vec<KeyRecord>> {
        let limit = limit.unwrap_or(1000).clamp(1, 1001);
        let q = format!(
            "SELECT key_id, key_hash, key_alias, team_id, user_id, \
 org_id, project_id, max_budget, budget_duration, rpm, tpm, created_at, updated_at, expires_at, \
 is_active, models, allowed_cache_controls \
 FROM keys ORDER BY created_at DESC LIMIT {}",
            self.ph(1)
        );
        let rows = sqlx::query(&q).bind(limit).fetch_all(self.pool()).await?;
        rows.iter().map(row_to_key).collect()
    }

    /// Bulk flip `is_active` on every key attached to a team. Single
    /// statement → atomic and scoped (no unbounded table scan, no per-row
    /// UPDATE loop). Returns the number of rows touched.
    pub async fn set_keys_active_for_team(
        &self,
        team_id: &str,
        active: bool,
        updated_at: &str,
    ) -> Result<u64> {
        let q = format!(
            "UPDATE keys SET is_active = {}, updated_at = {} WHERE team_id = {}",
            self.ph(1),
            self.ph(2),
            self.ph(3)
        );
        let res = sqlx::query(&q)
            .bind(if active { 1_i64 } else { 0_i64 })
            .bind(updated_at)
            .bind(team_id)
            .execute(self.pool())
            .await?;
        Ok(res.rows_affected())
    }

    /// Clear `team_id` (set to NULL) on every key that currently references
    /// `team_id`. Used when a team is deleted so no key is left pointing at a
    /// removed scope (which would otherwise silently drop the team-level
    /// budget / rate enforcement). Returns rows affected.
    /// This import primitive does not check project ownership; native team
    /// removal uses `delete_team_scoped` to detach both sides in one commit.
    pub async fn clear_team_on_keys(&self, team_id: &str, updated_at: &str) -> Result<u64> {
        let q = format!(
            "UPDATE keys SET team_id = NULL, updated_at = {} WHERE team_id = {}",
            self.ph(1),
            self.ph(2)
        );
        let res = sqlx::query(&q)
            .bind(updated_at)
            .bind(team_id)
            .execute(self.pool())
            .await?;
        Ok(res.rows_affected())
    }

    /// Clear `user_id` (set to NULL) on every key that currently references
    /// `user_id`. Used when a user is deleted so no key is left pointing at a
    /// removed scope (which would otherwise silently drop the user-level
    /// budget cap on the next auth — `user_limits_for` misses →
    /// `user_max_budget = None`). Mirrors [`Self::clear_team_on_keys`].
    /// Returns rows affected.
    pub async fn clear_user_on_keys(&self, user_id: &str, updated_at: &str) -> Result<u64> {
        let q = format!(
            "UPDATE keys SET user_id = NULL, updated_at = {} WHERE user_id = {}",
            self.ph(1),
            self.ph(2)
        );
        let res = sqlx::query(&q)
            .bind(updated_at)
            .bind(user_id)
            .execute(self.pool())
            .await?;
        Ok(res.rows_affected())
    }

    /// Fetch just the key hashes for a team (used to invalidate the auth
    /// cache after a bulk mutation). Scoped to the team → no unbounded scan.
    pub async fn list_key_hashes_for_team(&self, team_id: &str) -> Result<Vec<String>> {
        let q = format!("SELECT key_hash FROM keys WHERE team_id = {}", self.ph(1));
        let rows = sqlx::query(&q).bind(team_id).fetch_all(self.pool()).await?;
        rows.iter()
            .map(|r| {
                r.try_get::<String, _>("key_hash")
                    .map_err(anyhow::Error::from)
            })
            .collect()
    }

    /// Bulk set or clear `team_id` on a user's keys. When `filter_team` is
    /// `Some(t)`, only keys currently in team `t` are touched (used by
    /// `team/member/delete`); otherwise every key owned by the user is
    /// touched (used by `team/member/add`). When `new_team` is `None` the
    /// key's `team_id` is cleared (set to NULL). Single statement → atomic
    /// and scoped. Returns rows affected.
    /// Scope links are unchecked for imports and fixtures. Native member
    /// endpoints use `set_team_for_user_keys_scoped` instead.
    pub async fn set_team_for_user_keys(
        &self,
        user_id: &str,
        filter_team: Option<&str>,
        new_team: Option<&str>,
        updated_at: &str,
    ) -> Result<u64> {
        let q = match (filter_team, new_team) {
            (Some(_), _) => format!(
                "UPDATE keys SET team_id = {}, updated_at = {} WHERE user_id = {} AND team_id = {}",
                self.ph(1),
                self.ph(2),
                self.ph(3),
                self.ph(4)
            ),
            (None, _) => format!(
                "UPDATE keys SET team_id = {}, updated_at = {} WHERE user_id = {}",
                self.ph(1),
                self.ph(2),
                self.ph(3)
            ),
        };
        let mut query = sqlx::query(&q).bind(new_team).bind(updated_at);
        query = query.bind(user_id);
        if filter_team.is_some() {
            query = query.bind(filter_team);
        }
        let res = query.execute(self.pool()).await?;
        Ok(res.rows_affected())
    }

    /// Atomically reassign all matching keys or reject the entire operation.
    /// Owned projects must agree with the resulting team, including a clear to
    /// NULL; ownerless projects allow any existing destination team.
    pub async fn set_team_for_user_keys_scoped(
        &self,
        user_id: &str,
        filter_team: Option<&str>,
        new_team: Option<&str>,
        updated_at: &str,
    ) -> Result<KeyTeamMutation> {
        let mut tx = self.begin_scope_links().await?;
        self.validate_user_key_team_links_on(&mut tx, user_id, filter_team, new_team)
            .await?;
        let filter = filter_team
            .map(|_| format!(" AND team_id = {}", self.ph(4)))
            .unwrap_or_default();
        let q = format!(
            "UPDATE keys SET team_id = {}, updated_at = {} WHERE user_id = {}{filter} \
             RETURNING key_hash",
            self.ph(1),
            self.ph(2),
            self.ph(3)
        );
        let mut query = sqlx::query(&q)
            .bind(new_team)
            .bind(updated_at)
            .bind(user_id);
        if let Some(team_id) = filter_team {
            query = query.bind(team_id);
        }
        let rows = query.fetch_all(&mut *tx).await?;
        let key_hashes = rows
            .iter()
            .map(|row| row.try_get("key_hash"))
            .collect::<Result<Vec<String>, _>>()?;
        tx.commit().await?;
        Ok(KeyTeamMutation {
            affected: key_hashes.len() as u64,
            key_hashes,
        })
    }

    /// Fetch just the key hashes for a user's keys, optionally restricted to
    /// a given team (used to invalidate the auth cache after a bulk
    /// reassignment). Scoped → no unbounded scan.
    pub async fn list_key_hashes_for_user(
        &self,
        user_id: &str,
        filter_team: Option<&str>,
    ) -> Result<Vec<String>> {
        let q = match filter_team {
            Some(_) => format!(
                "SELECT key_hash FROM keys WHERE user_id = {} AND team_id = {}",
                self.ph(1),
                self.ph(2)
            ),
            None => format!("SELECT key_hash FROM keys WHERE user_id = {}", self.ph(1)),
        };
        let mut query = sqlx::query(&q).bind(user_id);
        if filter_team.is_some() {
            query = query.bind(filter_team);
        }
        let rows = query.fetch_all(self.pool()).await?;
        rows.iter()
            .map(|r| {
                r.try_get::<String, _>("key_hash")
                    .map_err(anyhow::Error::from)
            })
            .collect()
    }

    /// Soft-delete: mark the key inactive. `updated_at` is bumped (matching
    /// every other mutation on the keys table) so consumers that read
    /// `updated_at` for audit can tell when the key was revoked.
    pub async fn delete_key(&self, key_id: &str, updated_at: &str) -> Result<()> {
        self.set_key_active(key_id, false, updated_at)
            .await
            .map(|_| ())
    }

    /// Change only the active flag, retaining scope links and model literals.
    /// Return the committed identity, or None when a concurrent purge won.
    pub async fn set_key_active(
        &self,
        key_id: &str,
        active: bool,
        updated_at: &str,
    ) -> Result<Option<String>> {
        let mut tx = self.begin_scope_links().await?;
        let q = format!(
            "UPDATE keys SET is_active = {}, updated_at = {} WHERE key_id = {} RETURNING key_hash",
            self.ph(1),
            self.ph(2),
            self.ph(3)
        );
        let row = sqlx::query(&q)
            .bind(if active { 1_i64 } else { 0_i64 })
            .bind(updated_at)
            .bind(key_id)
            .fetch_optional(&mut *tx)
            .await?;
        let hash = row.map(|row| row.try_get("key_hash")).transpose()?;
        tx.commit().await?;
        Ok(hash)
    }

    /// Hard-delete: remove the key row permanently. Spend logs are kept.
    ///
    /// **Attribution caveat:** the spend aggregations
    /// ([`crate::Database::query_spend_by_org`] / `query_spend_by_key`)
    /// resolve `org_id` / `key_alias` via `LEFT JOIN keys ON s.key_id =
    /// k.key_id`. After a purge, every historical `spend_logs` row for that
    /// `key_id` falls into the NULL group (org-level) or loses its alias
    /// (key-level), silently corrupting historical billing rollups. Prefer
    /// [`Self::delete_key`] (soft delete) when the key has historical spend;
    /// reserve `purge_key` for keys with no telemetry attached.
    pub async fn purge_key(&self, key_id: &str) -> Result<()> {
        self.purge_key_scoped(key_id).await.map(|_| ())
    }

    /// Remove a key under the graph lock and return its last committed hash.
    pub async fn purge_key_scoped(&self, key_id: &str) -> Result<Option<String>> {
        let mut tx = self.begin_scope_links().await?;
        let q = format!(
            "DELETE FROM keys WHERE key_id = {} RETURNING key_hash",
            self.ph(1)
        );
        let row = sqlx::query(&q)
            .bind(key_id)
            .fetch_optional(&mut *tx)
            .await?;
        let hash = row.map(|row| row.try_get("key_hash")).transpose()?;
        tx.commit().await?;
        Ok(hash)
    }

    /// Update key metadata (alias, team, user, org, project, budgets,
    /// limits, active flag). Does not change `key_hash` (use
    /// [`Self::rotate_key`] for that). Returns `true` when a row was
    /// matched and updated, `false` when no row matched `key_id` (the row
    /// was deleted between the caller's `get_key` and this `UPDATE`). The
    /// caller should map `false` to a 404 so a racy delete is not surfaced
    /// as a stale `200 OK` with the prior record.
    pub async fn update_key(&self, key: &KeyRecord) -> Result<bool> {
        validate_key_record(key)?;
        self.update_key_on(key, self.pool()).await
    }

    /// Apply a native metadata mutation to the current row inside the graph
    /// transaction. Omitted fields retain the latest committed values, not a
    /// handler's earlier snapshot. Immutable identity fields remain unchanged.
    /// Low-level `update_key` remains available for imports and fixtures.
    pub async fn update_key_scoped<F>(&self, key_id: &str, update: F) -> Result<Option<KeyRecord>>
    where
        F: FnOnce(&mut KeyRecord),
    {
        let mut tx = self.begin_scope_links().await?;
        let Some(mut key) = self.get_key_on(key_id, &mut *tx).await? else {
            tx.rollback().await?;
            return Ok(None);
        };
        let identity = (
            key.key_id.clone(),
            key.key_hash.clone(),
            key.created_at.clone(),
        );
        update(&mut key);
        (key.key_id, key.key_hash, key.created_at) = identity;
        validate_key_record(&key)?;
        self.validate_key_scope_links_on(&mut tx, &key).await?;
        self.update_key_on(&key, &mut *tx).await?;
        tx.commit().await?;
        Ok(Some(key))
    }

    async fn update_key_on<'e>(
        &self,
        key: &KeyRecord,
        executor: impl Executor<'e, Database = Any>,
    ) -> Result<bool> {
        let q = format!(
            "UPDATE keys SET key_alias = {}, team_id = {}, user_id = {}, org_id = {}, \
 project_id = {}, max_budget = {}, budget_duration = {}, rpm = {}, tpm = {}, updated_at = {}, \
 expires_at = {}, is_active = {}, models = {}, allowed_cache_controls = {} WHERE key_id = {}",
            self.ph(1),
            self.ph(2),
            self.ph(3),
            self.ph(4),
            self.ph(5),
            self.ph(6),
            self.ph(7),
            self.ph(8),
            self.ph(9),
            self.ph(10),
            self.ph(11),
            self.ph(12),
            self.ph(13),
            self.ph(14),
            self.ph(15)
        );
        let query = sqlx::query(&q)
            .bind(key.key_alias.clone())
            .bind(key.team_id.clone())
            .bind(key.user_id.clone())
            .bind(key.org_id.clone())
            .bind(key.project_id.clone());
        let res = bind_optional_f64(query, key.max_budget)
            .bind(key.budget_duration.clone())
            .bind(key.rpm)
            .bind(key.tpm)
            .bind(&key.updated_at)
            .bind(key.expires_at.clone())
            .bind(if key.is_active { 1_i64 } else { 0_i64 })
            .bind(key.models.clone())
            .bind(key.allowed_cache_controls.clone())
            .bind(&key.key_id)
            .execute(executor)
            .await?;
        Ok(res.rows_affected() > 0)
    }

    /// Replace the stored key hash (rotate secret). Returns previous hash.
    ///
    /// The graph lock serializes this read with native metadata writes, bulk
    /// moves and deletion. The returned identity is authoritative after commit
    /// and can be evicted from authentication caches without a stale pre-read.
    pub async fn rotate_key(
        &self,
        key_id: &str,
        new_hash: &str,
        updated_at: &str,
    ) -> Result<Option<String>> {
        let mut tx = self.begin_scope_links().await?;
        let select_q = format!("SELECT key_hash FROM keys WHERE key_id = {}", self.ph(1));
        let row = sqlx::query(&select_q)
            .bind(key_id)
            .fetch_optional(&mut *tx)
            .await?;
        let Some(row) = row else {
            tx.rollback().await.ok();
            return Ok(None);
        };
        let prev_hash: String = row.try_get("key_hash")?;
        let update_q = format!(
            "UPDATE keys SET key_hash = {}, updated_at = {} WHERE key_id = {}",
            self.ph(1),
            self.ph(2),
            self.ph(3)
        );
        sqlx::query(&update_q)
            .bind(new_hash)
            .bind(updated_at)
            .bind(key_id)
            .execute(&mut *tx)
            .await?;
        tx.commit().await?;
        Ok(Some(prev_hash))
    }
}
