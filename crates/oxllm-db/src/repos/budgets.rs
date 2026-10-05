//! `budgets` table CRUD.

use anyhow::Result;

use crate::Database;
use crate::binds::bind_optional_f64;
use crate::rows::row_to_budget;
use crate::types::BudgetRecord;

impl Database {
    pub async fn create_budget(&self, budget: &BudgetRecord) -> Result<()> {
        let q = format!(
            "INSERT INTO budgets (budget_id, max_budget, soft_budget, max_parallel_requests, \
 tpm_limit, rpm_limit, budget_duration, model_max_budget, created_at, updated_at) \
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
        let query = bind_optional_f64(sqlx::query(&q).bind(&budget.budget_id), budget.max_budget);
        bind_optional_f64(query, budget.soft_budget)
            .bind(budget.max_parallel_requests)
            .bind(budget.tpm_limit)
            .bind(budget.rpm_limit)
            .bind(budget.budget_duration.clone())
            .bind(budget.model_max_budget.clone())
            .bind(&budget.created_at)
            .bind(&budget.updated_at)
            .execute(self.pool())
            .await?;
        Ok(())
    }

    pub async fn get_budget(&self, budget_id: &str) -> Result<Option<BudgetRecord>> {
        let q = format!(
            "SELECT budget_id, max_budget, soft_budget, max_parallel_requests, tpm_limit, \
 rpm_limit, budget_duration, model_max_budget, created_at, updated_at \
 FROM budgets WHERE budget_id = {}",
            self.ph(1)
        );
        let row = sqlx::query(&q)
            .bind(budget_id)
            .fetch_optional(self.pool())
            .await?;
        row.map(|r| row_to_budget(&r)).transpose()
    }

    /// List budgets newest-first, capped at `limit` rows (default 1 000
    /// when `None`). The upper clamp is 1 001 so a handler can request one
    /// extra row as a sentinel to detect truncation without the result
    /// being silently clamped back down. See [`crate::Database::list_keys`]
    /// for why the cap lives in the repo.
    pub async fn list_budgets(&self, limit: Option<i64>) -> Result<Vec<BudgetRecord>> {
        let limit = limit.unwrap_or(1000).clamp(1, 1001);
        let q = format!(
            "SELECT budget_id, max_budget, soft_budget, max_parallel_requests, tpm_limit, \
 rpm_limit, budget_duration, model_max_budget, created_at, updated_at \
 FROM budgets ORDER BY created_at DESC LIMIT {}",
            self.ph(1)
        );
        let rows = sqlx::query(&q).bind(limit).fetch_all(self.pool()).await?;
        rows.iter().map(row_to_budget).collect()
    }

    /// Fetch many budgets by id in a single batched `WHERE budget_id IN (...)`
    /// query rather than one `SELECT` per id (the previous implementation was
    /// an N+1: `budget_info_bulk` issued `budgets.len()` round-trips). The id
    /// list is chunked so each statement stays under the backend's
    /// bind-parameter ceiling (Postgres 65 535, SQLite 999); each chunk is a
    /// separate statement (this is a read path, so all-or-nothing is not
    /// required). Rows are returned in the order their ids first appear in
    /// `budget_ids`; missing ids are silently omitted (matching the prior
    /// per-id `get_budget` behavior).
    pub async fn get_budgets_by_ids(&self, budget_ids: &[String]) -> Result<Vec<BudgetRecord>> {
        if budget_ids.is_empty() {
            return Ok(Vec::new());
        }
        let max_ids_per_chunk = self.param_limit();

        let mut by_id: std::collections::HashMap<String, BudgetRecord> =
            std::collections::HashMap::with_capacity(budget_ids.len());
        for chunk in budget_ids.chunks(max_ids_per_chunk) {
            let placeholders: Vec<String> = (0..chunk.len()).map(|i| self.ph(i + 1)).collect();
            let q = format!(
                "SELECT budget_id, max_budget, soft_budget, max_parallel_requests, tpm_limit, \
 rpm_limit, budget_duration, model_max_budget, created_at, updated_at \
 FROM budgets WHERE budget_id IN ({})",
                placeholders.join(", ")
            );
            let mut query = sqlx::query(&q);
            for id in chunk {
                query = query.bind(id);
            }
            let rows = query.fetch_all(self.pool()).await?;
            for row in &rows {
                let b = row_to_budget(row)?;
                by_id.insert(b.budget_id.clone(), b);
            }
        }
        let mut out = Vec::with_capacity(budget_ids.len());
        for id in budget_ids {
            if let Some(b) = by_id.get(id) {
                out.push(b.clone());
            }
        }
        Ok(out)
    }

    /// Overwrite the mutable fields of the budget identified by
    /// `budget.budget_id`. Returns `true` when a row was matched and
    /// updated, `false` when no row matched `budget_id` (the row was
    /// deleted between the caller's `get_budget` and this `UPDATE`). The
    /// caller should map `false` to a 404 so a racy delete is not surfaced
    /// as a stale `200 OK`.
    pub async fn update_budget(&self, budget: &BudgetRecord) -> Result<bool> {
        let q = format!(
            "UPDATE budgets SET max_budget = {}, soft_budget = {}, max_parallel_requests = {}, \
 tpm_limit = {}, rpm_limit = {}, budget_duration = {}, model_max_budget = {}, \
 updated_at = {} WHERE budget_id = {}",
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
        let query = bind_optional_f64(sqlx::query(&q), budget.max_budget);
        let res = bind_optional_f64(query, budget.soft_budget)
            .bind(budget.max_parallel_requests)
            .bind(budget.tpm_limit)
            .bind(budget.rpm_limit)
            .bind(budget.budget_duration.clone())
            .bind(budget.model_max_budget.clone())
            .bind(&budget.updated_at)
            .bind(&budget.budget_id)
            .execute(self.pool())
            .await?;
        Ok(res.rows_affected() > 0)
    }

    pub async fn delete_budget(&self, budget_id: &str) -> Result<()> {
        let q = format!("DELETE FROM budgets WHERE budget_id = {}", self.ph(1));
        sqlx::query(&q).bind(budget_id).execute(self.pool()).await?;
        Ok(())
    }

    /// Delete budgets in bind-limited, set-based chunks in one transaction.
    /// Any failing chunk rolls back every deletion. Missing/repeated IDs do
    /// not inflate the count returned after successful commit.
    pub async fn delete_budgets(&self, budget_ids: &[String]) -> Result<usize> {
        if budget_ids.is_empty() {
            return Ok(0);
        }
        let mut tx = self.pool().begin().await?;
        let mut n = 0usize;
        for chunk in budget_ids.chunks(self.param_limit()) {
            let ids = (0..chunk.len())
                .map(|i| self.ph(i + 1))
                .collect::<Vec<_>>()
                .join(", ");
            let q = format!("DELETE FROM budgets WHERE budget_id IN ({ids})");
            let mut query = sqlx::query(&q);
            for id in chunk {
                query = query.bind(id);
            }
            n += usize::try_from(query.execute(&mut *tx).await?.rows_affected())?;
        }
        tx.commit().await?;
        Ok(n)
    }
}
