//! `teams` table CRUD.

use anyhow::Result;

use crate::Database;
use crate::binds::bind_optional_f64;
use crate::rows::row_to_team;
use crate::types::TeamRecord;

impl Database {
    pub async fn create_team(&self, team: &TeamRecord) -> Result<()> {
        let q = format!(
            "INSERT INTO teams (team_id, team_name, max_budget, budget_duration, rpm, tpm, \
 created_at, updated_at, models, blocked) VALUES ({}, {}, {}, {}, {}, {}, {}, {}, {}, {})",
            self.ph(1),
            self.ph(2),
            self.ph(3),
            self.ph(4),
            self.ph(5),
            self.ph(6),
            self.ph(7),
            self.ph(8),
            self.ph(9),
            self.ph(10)
        );
        let query = sqlx::query(&q).bind(&team.team_id).bind(&team.team_name);
        bind_optional_f64(query, team.max_budget)
            .bind(team.budget_duration.clone())
            .bind(team.rpm)
            .bind(team.tpm)
            .bind(&team.created_at)
            .bind(&team.updated_at)
            .bind(team.models.clone())
            .bind(if team.blocked { 1_i64 } else { 0_i64 })
            .execute(self.pool())
            .await?;
        Ok(())
    }

    pub async fn get_team(&self, team_id: &str) -> Result<Option<TeamRecord>> {
        let q = format!(
            "SELECT team_id, team_name, max_budget, budget_duration, rpm, tpm, created_at, \
 updated_at, models, blocked FROM teams WHERE team_id = {}",
            self.ph(1)
        );
        let row = sqlx::query(&q)
            .bind(team_id)
            .fetch_optional(self.pool())
            .await?;
        row.map(|r| row_to_team(&r)).transpose()
    }

    /// List teams newest-first, capped at `limit` rows (default 1 000 when
    /// `None`). See [`Self::list_keys`] for why the cap lives in the repo.
    /// The upper clamp is 1 001 so the handler can request one extra row as
    /// a truncation sentinel.
    pub async fn list_teams(&self, limit: Option<i64>) -> Result<Vec<TeamRecord>> {
        let limit = limit.unwrap_or(1000).clamp(1, 1001);
        let q = format!(
            "SELECT team_id, team_name, max_budget, budget_duration, rpm, tpm, created_at, \
 updated_at, models, blocked FROM teams ORDER BY created_at DESC LIMIT {}",
            self.ph(1)
        );
        let rows = sqlx::query(&q).bind(limit).fetch_all(self.pool()).await?;
        rows.iter().map(row_to_team).collect()
    }

    /// Delete the team and detach its keys/projects in one transaction.
    pub async fn delete_team(&self, team_id: &str) -> Result<()> {
        self.delete_team_and_detach(team_id).await.map(|_| ())
    }

    /// Return the detached identities after commit for auth-cache invalidation.
    pub async fn delete_team_and_detach(&self, team_id: &str) -> Result<crate::ScopeDeletion> {
        self.delete_scope(super::scopes::Scope::Team, team_id).await
    }

    /// Overwrite the mutable fields of the team identified by
    /// `team.team_id`. Returns `true` when a row was matched and updated,
    /// `false` when no row matched `team_id` (the row was deleted between
    /// the caller's `get_team` and this `UPDATE`). The caller should map
    /// `false` to a 404 so a racy delete is not surfaced as a stale
    /// `200 OK`.
    pub async fn update_team(&self, team: &TeamRecord) -> Result<bool> {
        let q = format!(
            "UPDATE teams SET team_name = {}, max_budget = {}, budget_duration = {}, \
 rpm = {}, tpm = {}, updated_at = {}, models = {}, blocked = {} WHERE team_id = {}",
            self.ph(1),
            self.ph(2),
            self.ph(3),
            self.ph(4),
            self.ph(5),
            self.ph(6),
            self.ph(7),
            self.ph(8),
            self.ph(9)
        );
        let query = sqlx::query(&q).bind(&team.team_name);
        let res = bind_optional_f64(query, team.max_budget)
            .bind(team.budget_duration.clone())
            .bind(team.rpm)
            .bind(team.tpm)
            .bind(&team.updated_at)
            .bind(team.models.clone())
            .bind(if team.blocked { 1_i64 } else { 0_i64 })
            .bind(&team.team_id)
            .execute(self.pool())
            .await?;
        Ok(res.rows_affected() > 0)
    }
}
