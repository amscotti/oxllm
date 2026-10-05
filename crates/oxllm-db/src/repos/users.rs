//! `users` table CRUD.

use anyhow::Result;

use crate::Database;
use crate::binds::bind_optional_f64;
use crate::rows::row_to_user;
use crate::types::UserRecord;

impl Database {
    pub async fn create_user(&self, user: &UserRecord) -> Result<()> {
        let q = format!(
            "INSERT INTO users (user_id, user_name, max_budget, budget_duration, created_at, updated_at, models) \
 VALUES ({}, {}, {}, {}, {}, {}, {})",
            self.ph(1),
            self.ph(2),
            self.ph(3),
            self.ph(4),
            self.ph(5),
            self.ph(6),
            self.ph(7)
        );
        let query = sqlx::query(&q).bind(&user.user_id).bind(&user.user_name);
        bind_optional_f64(query, user.max_budget)
            .bind(user.budget_duration.clone())
            .bind(&user.created_at)
            .bind(&user.updated_at)
            .bind(user.models.clone())
            .execute(self.pool())
            .await?;
        Ok(())
    }

    pub async fn get_user(&self, user_id: &str) -> Result<Option<UserRecord>> {
        let q = format!(
            "SELECT user_id, user_name, max_budget, budget_duration, created_at, updated_at, models \
 FROM users WHERE user_id = {}",
            self.ph(1)
        );
        let row = sqlx::query(&q)
            .bind(user_id)
            .fetch_optional(self.pool())
            .await?;
        row.map(|r| row_to_user(&r)).transpose()
    }

    /// List users newest-first, capped at `limit` rows (default 1 000 when
    /// `None`). See [`crate::Database::list_keys`] for why the cap lives in
    /// the repo. The upper clamp is 1 001 so the handler can request one
    /// extra row as a truncation sentinel.
    pub async fn list_users(&self, limit: Option<i64>) -> Result<Vec<UserRecord>> {
        let limit = limit.unwrap_or(1000).clamp(1, 1001);
        let q = format!(
            "SELECT user_id, user_name, max_budget, budget_duration, created_at, updated_at, models \
 FROM users ORDER BY created_at DESC LIMIT {}",
            self.ph(1)
        );
        let rows = sqlx::query(&q).bind(limit).fetch_all(self.pool()).await?;
        rows.iter().map(row_to_user).collect()
    }

    /// Overwrite the mutable fields of the user identified by
    /// `user.user_id`. Returns `true` when a row was matched and updated,
    /// `false` when no row matched `user_id` (the row was deleted between
    /// the caller's `get_user` and this `UPDATE`). The caller should map
    /// `false` to a 404 so a racy delete is not surfaced as a stale
    /// `200 OK`.
    pub async fn update_user(&self, user: &UserRecord) -> Result<bool> {
        let q = format!(
            "UPDATE users SET user_name = {}, max_budget = {}, budget_duration = {}, \
 updated_at = {}, models = {} WHERE user_id = {}",
            self.ph(1),
            self.ph(2),
            self.ph(3),
            self.ph(4),
            self.ph(5),
            self.ph(6)
        );
        let query = sqlx::query(&q).bind(&user.user_name);
        let res = bind_optional_f64(query, user.max_budget)
            .bind(user.budget_duration.clone())
            .bind(&user.updated_at)
            .bind(user.models.clone())
            .bind(&user.user_id)
            .execute(self.pool())
            .await?;
        Ok(res.rows_affected() > 0)
    }

    /// Delete the user and detach their keys in one transaction.
    pub async fn delete_user(&self, user_id: &str) -> Result<()> {
        self.delete_user_and_detach(user_id).await.map(|_| ())
    }

    /// Return the detached identities after commit for auth-cache invalidation.
    pub async fn delete_user_and_detach(&self, user_id: &str) -> Result<crate::ScopeDeletion> {
        self.delete_scope(super::scopes::Scope::User, user_id).await
    }
}
