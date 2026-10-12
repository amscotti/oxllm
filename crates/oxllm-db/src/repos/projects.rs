//! `projects` table CRUD.

use anyhow::Result;
use sqlx::{Any, Executor};

use crate::Database;
use crate::binds::bind_optional_f64;
use crate::rows::row_to_project;
use crate::types::ProjectRecord;

impl Database {
    /// Unchecked parent links for imports and fixtures. Native management writes
    /// use `create_project_scoped` to validate the owning scopes atomically.
    pub async fn create_project(&self, project: &ProjectRecord) -> Result<()> {
        self.create_project_on(project, self.pool()).await
    }

    pub async fn create_project_scoped(&self, project: &ProjectRecord) -> Result<()> {
        let mut tx = self.begin_scope_links().await?;
        self.validate_project_scope_links_on(&mut tx, project)
            .await?;
        self.create_project_on(project, &mut *tx).await?;
        tx.commit().await?;
        Ok(())
    }

    async fn create_project_on<'e>(
        &self,
        project: &ProjectRecord,
        executor: impl Executor<'e, Database = Any>,
    ) -> Result<()> {
        let q = format!(
            "INSERT INTO projects (project_id, project_name, org_id, team_id, max_budget, \
 budget_duration, rpm, tpm, created_at, updated_at) \
 VALUES ({}, {}, {}, {}, {}, {}, {}, {}, {}, {})",
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
        let query = sqlx::query(&q)
            .bind(&project.project_id)
            .bind(&project.project_name)
            .bind(project.org_id.clone())
            .bind(project.team_id.clone());
        bind_optional_f64(query, project.max_budget)
            .bind(project.budget_duration.clone())
            .bind(project.rpm)
            .bind(project.tpm)
            .bind(&project.created_at)
            .bind(&project.updated_at)
            .execute(executor)
            .await?;
        Ok(())
    }

    pub async fn get_project(&self, project_id: &str) -> Result<Option<ProjectRecord>> {
        self.get_project_on(project_id, self.pool()).await
    }

    async fn get_project_on<'e>(
        &self,
        project_id: &str,
        executor: impl Executor<'e, Database = Any>,
    ) -> Result<Option<ProjectRecord>> {
        let q = format!(
            "SELECT project_id, project_name, org_id, team_id, max_budget, budget_duration, \
 rpm, tpm, created_at, updated_at FROM projects WHERE project_id = {}",
            self.ph(1)
        );
        let row = sqlx::query(&q)
            .bind(project_id)
            .fetch_optional(executor)
            .await?;
        row.map(|r| row_to_project(&r)).transpose()
    }

    /// List projects newest-first, capped at `limit` rows (default 1 000
    /// when `None`). The upper clamp is 1 001 so a handler can request one
    /// extra row as a sentinel to detect truncation without the result
    /// being silently clamped back down. See [`crate::Database::list_keys`]
    /// for why the cap lives in the repo.
    pub async fn list_projects(&self, limit: Option<i64>) -> Result<Vec<ProjectRecord>> {
        let limit = limit.unwrap_or(1000).clamp(1, 1001);
        let q = format!(
            "SELECT project_id, project_name, org_id, team_id, max_budget, \
 budget_duration, rpm, tpm, created_at, updated_at FROM projects ORDER BY created_at DESC LIMIT {}",
            self.ph(1)
        );
        let rows = sqlx::query(&q).bind(limit).fetch_all(self.pool()).await?;
        rows.iter().map(row_to_project).collect()
    }

    /// Overwrite the mutable fields of the project identified by
    /// `project.project_id`. Returns `true` when a row was matched and
    /// updated, `false` when no row matched `project_id` (the row was
    /// deleted between the caller's `get_project` and this `UPDATE`). The
    /// caller should map `false` to a 404 so a racy delete is not surfaced
    /// as a stale `200 OK`.
    pub async fn update_project(&self, project: &ProjectRecord) -> Result<bool> {
        self.update_project_on(project, self.pool()).await
    }

    /// Mutate the current project while holding the graph lock. An ownerless
    /// project can be shared; an owned project cannot acquire conflicting key
    /// teams. The low-level full-record writer remains available for imports.
    pub async fn update_project_scoped<F>(
        &self,
        project_id: &str,
        update: F,
    ) -> Result<Option<ProjectRecord>>
    where
        F: FnOnce(&mut ProjectRecord),
    {
        let mut tx = self.begin_scope_links().await?;
        let Some(mut project) = self.get_project_on(project_id, &mut *tx).await? else {
            tx.rollback().await?;
            return Ok(None);
        };
        let identity = (project.project_id.clone(), project.created_at.clone());
        update(&mut project);
        (project.project_id, project.created_at) = identity;
        self.validate_project_scope_links_on(&mut tx, &project)
            .await?;
        self.update_project_on(&project, &mut *tx).await?;
        tx.commit().await?;
        Ok(Some(project))
    }

    async fn update_project_on<'e>(
        &self,
        project: &ProjectRecord,
        executor: impl Executor<'e, Database = Any>,
    ) -> Result<bool> {
        let q = format!(
            "UPDATE projects SET project_name = {}, org_id = {}, team_id = {}, max_budget = \
{}, budget_duration = {}, rpm = {}, tpm = {}, updated_at = {} WHERE project_id = {}",
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
        let query = sqlx::query(&q)
            .bind(&project.project_name)
            .bind(project.org_id.clone())
            .bind(project.team_id.clone());
        let res = bind_optional_f64(query, project.max_budget)
            .bind(project.budget_duration.clone())
            .bind(project.rpm)
            .bind(project.tpm)
            .bind(&project.updated_at)
            .bind(&project.project_id)
            .execute(executor)
            .await?;
        Ok(res.rows_affected() > 0)
    }

    /// Delete the project and detach its keys in one transaction.
    pub async fn delete_project(&self, project_id: &str) -> Result<()> {
        self.delete_project_and_detach(project_id).await.map(|_| ())
    }

    /// Return the detached identities after commit for auth-cache invalidation.
    pub async fn delete_project_and_detach(
        &self,
        project_id: &str,
    ) -> Result<crate::ScopeDeletion> {
        self.delete_scope(super::scopes::Scope::Project, project_id)
            .await
    }
}
