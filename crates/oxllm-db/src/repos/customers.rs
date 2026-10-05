//! `customers` (end-user) table CRUD.

use anyhow::Result;

use crate::Database;
use crate::binds::bind_optional_f64;
use crate::rows::row_to_customer;
use crate::types::CustomerRecord;

impl Database {
    pub async fn create_customer(&self, customer: &CustomerRecord) -> Result<()> {
        let q = format!(
            "INSERT INTO customers (end_user_id, alias, email, max_budget, spend, budget_duration, \
 blocked, model_blocklist, created_at, updated_at) \
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
            .bind(&customer.end_user_id)
            .bind(customer.alias.clone())
            .bind(customer.email.clone());
        bind_optional_f64(query, customer.max_budget)
            .bind(customer.spend)
            .bind(customer.budget_duration.clone())
            .bind(i64::from(customer.blocked))
            .bind(customer.model_blocklist.clone())
            .bind(&customer.created_at)
            .bind(&customer.updated_at)
            .execute(self.pool())
            .await?;
        Ok(())
    }

    pub async fn get_customer(&self, end_user_id: &str) -> Result<Option<CustomerRecord>> {
        let q = format!(
            "SELECT end_user_id, alias, email, max_budget, spend, budget_duration, blocked, \
 model_blocklist, created_at, updated_at FROM customers WHERE end_user_id = {}",
            self.ph(1)
        );
        let row = sqlx::query(&q)
            .bind(end_user_id)
            .fetch_optional(self.pool())
            .await?;
        row.map(|r| row_to_customer(&r)).transpose()
    }

    /// List customers newest-first, capped at `limit` rows (default 1 000
    /// when `None`). The upper clamp is 1 001 so a handler can request one
    /// extra row as a sentinel to detect truncation without the result
    /// being silently clamped back down. See [`crate::Database::list_keys`]
    /// for why the cap lives in the repo.
    pub async fn list_customers(&self, limit: Option<i64>) -> Result<Vec<CustomerRecord>> {
        let limit = limit.unwrap_or(1000).clamp(1, 1001);
        let q = format!(
            "SELECT end_user_id, alias, email, max_budget, spend, budget_duration, blocked, \
 model_blocklist, created_at, updated_at FROM customers ORDER BY created_at DESC LIMIT {}",
            self.ph(1)
        );
        let rows = sqlx::query(&q).bind(limit).fetch_all(self.pool()).await?;
        rows.iter().map(row_to_customer).collect()
    }

    /// Overwrite the mutable fields of the customer identified by
    /// `customer.end_user_id`. Returns `true` when a row was matched and
    /// updated, `false` when no row matched `end_user_id` (the row was
    /// deleted between the caller's `get_customer` and this `UPDATE`). The
    /// caller should map `false` to a 404 so a racy delete is not surfaced
    /// as a stale `200 OK`.
    pub async fn update_customer(&self, customer: &CustomerRecord) -> Result<bool> {
        // `spend` is owned by `increment_customer_spend` (the spend writer).
        // Writing the caller's snapshot here would clobber a concurrent increment.
        let q = format!(
            "UPDATE customers SET alias = {}, email = {}, max_budget = {}, \
budget_duration = {}, blocked = {}, model_blocklist = {}, updated_at = {} \
WHERE end_user_id = {}",
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
            .bind(customer.alias.clone())
            .bind(customer.email.clone());
        let res = bind_optional_f64(query, customer.max_budget)
            .bind(customer.budget_duration.clone())
            .bind(i64::from(customer.blocked))
            .bind(customer.model_blocklist.clone())
            .bind(&customer.updated_at)
            .bind(&customer.end_user_id)
            .execute(self.pool())
            .await?;
        Ok(res.rows_affected() > 0)
    }

    /// Atomically add `delta` to the `spend` column of the customer
    /// identified by `end_user_id`. Returns `true` when a row was matched
    /// and updated, `false` when no such customer exists.
    ///
    /// For explicit adjustments; metering writes use `insert_spend_events`
    /// to update counters atomically with logs. Hot-path enforcement uses the cache counter
    /// `spend:customer:{id}` via `BudgetHook` (claim-based reserve/settle).
    ///
    /// Reads `spend` via the read-modify-write-free `spend = spend + $2`
    /// form so two concurrent increments cannot lose an update (the
    /// arithmetic runs under the statement's row lock). A negative `delta`
    /// (e.g. a refund) is valid; the column is `DOUBLE PRECISION`/`REAL`,
    /// so it is not clamped at zero.
    pub async fn increment_customer_spend(&self, end_user_id: &str, delta: f64) -> Result<bool> {
        let q = format!(
            "UPDATE customers SET spend = spend + {} WHERE end_user_id = {}",
            self.ph(1),
            self.ph(2)
        );
        let res = sqlx::query(&q)
            .bind(delta)
            .bind(end_user_id)
            .execute(self.pool())
            .await?;
        Ok(res.rows_affected() > 0)
    }

    pub async fn delete_customer(&self, end_user_id: &str) -> Result<()> {
        let q = format!("DELETE FROM customers WHERE end_user_id = {}", self.ph(1));
        sqlx::query(&q)
            .bind(end_user_id)
            .execute(self.pool())
            .await?;
        Ok(())
    }

    /// Bulk hard-delete every customer whose `end_user_id` is in
    /// `end_user_ids`, using chunked `DELETE … WHERE end_user_id IN (...)`
    /// statements inside a single transaction. The transaction makes the
    /// operation atomic: a failure in any chunk rolls back every prior
    /// chunk, so a mid-batch error can never leave a partial delete behind.
    /// Returns the total number of rows actually removed (ids that did not
    /// resolve are not counted), so the caller can report an accurate
    /// `deleted_customers` count derived from `rows_affected` rather than
    /// from a request-side counter that would over-report on a racy no-op.
    ///
    /// The id list is chunked so each statement stays under the backend's
    /// bind-parameter ceiling (Postgres 65 535, SQLite 999) — the same
    /// convention as [`Self::block_customers`].
    pub async fn delete_customers_bulk(&self, end_user_ids: &[String]) -> Result<u64> {
        if end_user_ids.is_empty() {
            return Ok(0);
        }
        let param_limit: usize = match self.placeholder_style() {
            crate::PlaceholderStyle::Dollar => 65_535,
            crate::PlaceholderStyle::Question => 999,
        };
        let mut tx = self.pool().begin().await?;
        let mut total: u64 = 0;
        for chunk in end_user_ids.chunks(param_limit.max(1)) {
            let placeholders: Vec<String> = (0..chunk.len()).map(|i| self.ph(i + 1)).collect();
            let q = format!(
                "DELETE FROM customers WHERE end_user_id IN ({})",
                placeholders.join(", ")
            );
            let mut query = sqlx::query(&q);
            for id in chunk {
                query = query.bind(id);
            }
            let res = query.execute(&mut *tx).await?;
            total += res.rows_affected();
        }
        tx.commit().await?;
        Ok(total)
    }

    /// Bulk set `blocked` on every customer whose `end_user_id` is in
    /// `end_user_ids`, using a single scoped `UPDATE … WHERE … IN (...)`
    /// statement. This is atomic (no partial state on mid-loop failure) and
    /// avoids an N+1 `get_customer`/`update_customer` round-trip per id.
    /// Returns the number of rows touched (ids that did not resolve are
    /// simply not counted).
    ///
    /// The id list is chunked so each statement stays under the backend's
    /// bind-parameter ceiling (Postgres 65 535, SQLite 999). Without
    /// chunking, a user-supplied list longer than ~996 ids would exceed the
    /// ceiling and the driver would reject the statement — surfacing as a
    /// 500 with a confusing "too many SQL variables" message rather than the
    /// atomic update the caller expects. The chunks run inside a single
    /// transaction so the operation is still all-or-nothing.
    pub async fn block_customers(
        &self,
        end_user_ids: &[String],
        blocked: bool,
        updated_at: &str,
    ) -> Result<u64> {
        if end_user_ids.is_empty() {
            return Ok(0);
        }
        let param_limit: usize = match self.placeholder_style() {
            crate::PlaceholderStyle::Dollar => 65_535,
            crate::PlaceholderStyle::Question => 999,
        };
        // The statement binds `blocked`, `updated_at`, then one placeholder
        // per id, so each chunk holds `2 + chunk_len` parameters.
        let max_ids_per_chunk = param_limit.saturating_sub(2).max(1);

        let mut tx = self.pool().begin().await?;
        let mut total: u64 = 0;
        for chunk in end_user_ids.chunks(max_ids_per_chunk) {
            let placeholders: Vec<String> = (0..chunk.len()).map(|i| self.ph(i + 3)).collect();
            let q = format!(
                "UPDATE customers SET blocked = {}, updated_at = {} WHERE end_user_id IN ({})",
                self.ph(1),
                self.ph(2),
                placeholders.join(", ")
            );
            let mut query = sqlx::query(&q)
                .bind(if blocked { 1_i64 } else { 0_i64 })
                .bind(updated_at);
            for id in chunk {
                query = query.bind(id);
            }
            let res = query.execute(&mut *tx).await?;
            total += res.rows_affected();
        }
        tx.commit().await?;
        Ok(total)
    }
}
