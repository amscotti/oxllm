//! `organizations` table CRUD.

use anyhow::Result;

use crate::Database;
use crate::binds::bind_optional_f64;
use crate::rows::row_to_org;
use crate::types::OrganizationRecord;

impl Database {
    pub async fn create_organization(&self, org: &OrganizationRecord) -> Result<()> {
        let q = format!(
            "INSERT INTO organizations (org_id, org_name, max_budget, budget_duration, rpm, \
 tpm, created_at, updated_at, models, metadata, soft_budget, max_parallel_requests, \
 model_max_budget) VALUES ({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {})",
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
            self.ph(13)
        );
        let query = sqlx::query(&q).bind(&org.org_id).bind(&org.org_name);
        let query = bind_optional_f64(query, org.max_budget)
            .bind(org.budget_duration.clone())
            .bind(org.rpm)
            .bind(org.tpm)
            .bind(&org.created_at)
            .bind(&org.updated_at)
            .bind(org.models.clone())
            .bind(org.metadata.clone());
        bind_optional_f64(query, org.soft_budget)
            .bind(org.max_parallel_requests)
            .bind(org.model_max_budget.clone())
            .execute(self.pool())
            .await?;
        Ok(())
    }

    pub async fn get_organization(&self, org_id: &str) -> Result<Option<OrganizationRecord>> {
        let q = format!(
            "SELECT org_id, org_name, max_budget, budget_duration, rpm, tpm, created_at, \
 updated_at, models, metadata, soft_budget, max_parallel_requests, model_max_budget \
 FROM organizations WHERE org_id = {}",
            self.ph(1)
        );
        let row = sqlx::query(&q)
            .bind(org_id)
            .fetch_optional(self.pool())
            .await?;
        row.map(|r| row_to_org(&r)).transpose()
    }

    /// List organizations newest-first, capped at `limit` rows (default
    /// 1 000 when `None`). The upper clamp is 1 001 so a handler can
    /// request one extra row as a sentinel to detect truncation without
    /// the result being silently clamped back down. See
    /// [`crate::Database::list_keys`] for why the cap lives in the repo.
    pub async fn list_organizations(&self, limit: Option<i64>) -> Result<Vec<OrganizationRecord>> {
        let limit = limit.unwrap_or(1000).clamp(1, 1001);
        let q = format!(
            "SELECT org_id, org_name, max_budget, budget_duration, rpm, tpm, created_at, \
 updated_at, models, metadata, soft_budget, max_parallel_requests, model_max_budget \
 FROM organizations ORDER BY created_at DESC LIMIT {}",
            self.ph(1)
        );
        let rows = sqlx::query(&q).bind(limit).fetch_all(self.pool()).await?;
        rows.iter().map(row_to_org).collect()
    }

    /// Overwrite the mutable fields of the organization identified by
    /// `org.org_id`. Returns `true` when a row was matched and updated,
    /// `false` when no row matched `org_id` (the row was deleted between the
    /// caller's `get_organization` and this `UPDATE`). The caller should map
    /// `false` to a 404 so a racy delete is not surfaced as a stale `200 OK`.
    pub async fn update_organization(&self, org: &OrganizationRecord) -> Result<bool> {
        let q = format!(
            "UPDATE organizations SET org_name = {}, max_budget = {}, budget_duration = {}, \
rpm = {}, tpm = {}, models = {}, metadata = {}, soft_budget = {}, \
max_parallel_requests = {}, model_max_budget = {}, updated_at = {} WHERE org_id = {}",
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
            self.ph(12)
        );
        let query = sqlx::query(&q).bind(&org.org_name);
        let query = bind_optional_f64(query, org.max_budget)
            .bind(org.budget_duration.clone())
            .bind(org.rpm)
            .bind(org.tpm)
            .bind(org.models.clone())
            .bind(org.metadata.clone());
        let res = bind_optional_f64(query, org.soft_budget)
            .bind(org.max_parallel_requests)
            .bind(org.model_max_budget.clone())
            .bind(&org.updated_at)
            .bind(&org.org_id)
            .execute(self.pool())
            .await?;
        Ok(res.rows_affected() > 0)
    }

    /// Delete the organization and detach its keys/projects in one transaction.
    pub async fn delete_organization(&self, org_id: &str) -> Result<()> {
        self.delete_organization_and_detach(org_id)
            .await
            .map(|_| ())
    }

    /// Return the detached identities after commit for auth-cache invalidation.
    pub async fn delete_organization_and_detach(
        &self,
        org_id: &str,
    ) -> Result<crate::ScopeDeletion> {
        self.delete_scope(super::scopes::Scope::Organization, org_id)
            .await
    }
}
