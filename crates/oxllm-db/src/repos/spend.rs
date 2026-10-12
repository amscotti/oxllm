//! `spend_logs` queries: single + batched inserts, filtered row fetches,
//! cursor pagination, and the aggregation rollups.
use std::collections::{BTreeMap, HashMap};

use crate::binds::bind_optional_f64;
use crate::rows::row_to_spend;
use crate::spend_metadata::SPEND_LOG_SELECT_COLS;
use crate::types::{
    DailySpend, SpendByKey, SpendByKeyPage, SpendByModel, SpendByModelPage, SpendByOrg,
    SpendByOrgPage, SpendByTag, SpendByTagPage, SpendByTeam, SpendByTeamPage, SpendByUser,
    SpendByUserPage, SpendFilter, SpendLog, SpendLogEvent, SpendPage, spend_tag_names,
};
use crate::{
    Database, PlaceholderStyle, SPEND_ROLLUP_MAX_LIMIT, SpendRollupLimitExceeded, placeholder,
};
use anyhow::{Result, anyhow};
use sqlx::Row;
use sqlx::any::AnyArguments;
use sqlx::query::Query;
/// One bind value accumulated by [`SpendWhereBuilder`], in placeholder order.
enum SpendBind {
    Str(String),
    Int(i64),
}
/// The canonical column list and bind order for inserting a
/// [`SpendLog`]. Both [`Database::insert_spend_log`] (single-row) and
/// [`Database::insert_spend_events`] (multi-row batch) source the
/// column list and the per-row bind sequence from here, so a future column
/// addition (e.g. `project_id`) updates one place rather than two — a
/// hand-synced pair could silently bind values to the wrong columns or
/// insert NULLs.
const SPEND_LOG_COLS: &str = "timestamp, request_id, model, provider, \
 prompt_tokens, completion_tokens, total_tokens, cost, key_id, team_id, user_id, \
 end_user_id, tag_name, tags, status, latency_ms, deployment_id, requested_model, model_group";
/// The number of columns in [`SPEND_LOG_COLS`] (and therefore the number of
/// binds applied per row by [`bind_spend_log`]). The single-row insert
/// ([`Database::insert_spend_log`]) and the chunked batch insert
/// ([`Database::insert_spend_logs_batch`] / [`Database::insert_spend_events`])
/// both derive their per-row placeholder count from this constant, so the
/// chunk-size arithmetic (`base = i * COLS`, `chunk_size = (limit-1)/COLS`)
/// cannot drift from the column list / bind order when a column is added.
/// Updating [`SPEND_LOG_COLS`] and [`bind_spend_log`] without bumping this
/// constant is caught by the batch-roundtrip tests.
const SPEND_LOG_COL_COUNT: usize = 19;
/// Bind every field of `log` onto `query`, in the same order as
/// [`SPEND_LOG_COLS`]. Both the single-row and batched insert paths route
/// through here so the column ↔ bind mapping has one source of truth.
fn bind_spend_log<'q>(
    mut query: Query<'q, sqlx::Any, AnyArguments<'q>>,
    log: &'q SpendLog,
) -> Query<'q, sqlx::Any, AnyArguments<'q>> {
    // Normalize on write: the read-side comparisons (see
    // [`SpendWhereBuilder::cmp_timestamp`]) assume fixed-precision TEXT.
    query = query
        .bind(crate::normalize_iso(&log.timestamp))
        .bind(log.request_id.clone())
        .bind(&log.model)
        .bind(&log.provider)
        .bind(log.prompt_tokens)
        .bind(log.completion_tokens)
        .bind(log.total_tokens);
    query = bind_optional_f64(query, log.cost)
        .bind(log.key_id.clone())
        .bind(log.team_id.clone())
        .bind(log.user_id.clone())
        .bind(log.end_user_id.clone())
        .bind(log.tag_name.clone())
        .bind(log.tags.clone())
        .bind(log.status.clone())
        .bind(log.latency_ms)
        .bind(log.deployment_id.clone())
        .bind(log.requested_model.clone())
        .bind(log.model_group.clone());
    query
}
/// Builds the `WHERE` clause shared by the spend-log queries and owns the
/// values to bind, in placeholder order.
///
/// [`Database::query_spend`], [`Database::query_spend_paginated`],
/// [`Database::total_spend`], [`Database::get_total_spend`],
/// [`Database::count_spend`], and the `query_spend_by_*` aggregations all
/// used to hand-roll the same "push one clause per `Some` filter, then bind
/// the values in the same order" pattern. Keeping the clause list and the
/// bind list in one builder guarantees the two cannot drift apart (a clause
/// without its bind — or vice versa — is a runtime error under Postgres).
struct SpendWhereBuilder {
    style: PlaceholderStyle,
    clauses: Vec<String>,
    binds: Vec<SpendBind>,
}
impl SpendWhereBuilder {
    fn new(style: PlaceholderStyle) -> Self {
        Self {
            style,
            clauses: Vec::new(),
            binds: Vec::new(),
        }
    }
    /// Add `{col} {op} <placeholder>` bound to `value` when `value` is
    /// `Some`; a `None` filter contributes no clause and no bind.
    fn cmp_str(mut self, col: &str, op: &str, value: Option<&str>) -> Self {
        if let Some(value) = value {
            self.clauses.push(format!(
                "{col} {op} {}",
                placeholder(self.style, self.binds.len() + 1)
            ));
            self.binds.push(SpendBind::Str(value.to_string()));
        }
        self
    }
    /// [`Self::cmp_str`], but the bound value is first normalized to a
    /// fixed-precision RFC 3339 form via [`crate::normalize_iso`]. The
    /// `timestamp` column is TEXT and jiff's default rendering varies the
    /// fractional width (`…59Z` vs `…59.5Z`), so a raw string compare can
    /// wrongly exclude sub-second boundary rows; comparing like-for-like
    /// fixed-width forms keeps TEXT ordering equal to chronological order.
    ///
    /// A bare `YYYY-MM-DD` upper bound (`<=`) is expanded to end-of-day, so
    /// `end_date: "2026-07-31"` includes rows from the whole of July 31st.
    /// Without this, the TEXT comparison `"2026-07-31" <=
    /// "2026-07-31T12:00:00…"` is false and the entire end day silently
    /// drops out of billing rollups. Lower bounds (`>=`) need no expansion:
    /// a bare date already sorts before every timestamp on that day.
    fn cmp_timestamp(mut self, col: &str, op: &str, value: Option<&str>) -> Self {
        if let Some(value) = value {
            let bound = if op == "<=" {
                crate::inclusive_spend_end_date(value)
            } else {
                value.to_string()
            };
            self.clauses.push(format!(
                "{col} {op} {}",
                placeholder(self.style, self.binds.len() + 1)
            ));
            self.binds
                .push(SpendBind::Str(crate::normalize_iso(&bound)));
        }
        self
    }
    /// Integer variant of [`Self::cmp_str`] (e.g. the `id < cursor` clause).
    fn cmp_i64(mut self, col: &str, op: &str, value: Option<i64>) -> Self {
        if let Some(value) = value {
            self.clauses.push(format!(
                "{col} {op} {}",
                placeholder(self.style, self.binds.len() + 1)
            ));
            self.binds.push(SpendBind::Int(value));
        }
        self
    }
    /// Literal membership, indexed when ready with an exact JSON fallback for
    /// old-replica/imported rows. Unlike LIKE against JSON text, this is
    /// case-sensitive on both backends and does not interpret %, _, quotes,
    /// backslashes or JSON escapes as matching syntax.
    fn matches_tag(mut self, value: Option<&str>) -> Self {
        if let Some(value) = value {
            self.clauses.push(format!(
                "EXISTS (SELECT 1 FROM spend_log_tags_read st \
                 WHERE st.spend_log_id = spend_logs.id AND st.tag_name = {})",
                placeholder(self.style, self.binds.len() + 1)
            ));
            self.binds.push(SpendBind::Str(value.to_string()));
        }
        self
    }

    /// The ` WHERE ...` fragment, or an empty string when no clause was added.
    fn where_sql(&self) -> String {
        if self.clauses.is_empty() {
            String::new()
        } else {
            format!(" WHERE {}", self.clauses.join(" AND "))
        }
    }
    /// 1-based index of the next placeholder after the accumulated binds.
    /// Callers that append their own trailing binds (e.g. LIMIT/OFFSET)
    /// number them from here.
    fn next_idx(&self) -> usize {
        self.binds.len() + 1
    }
    /// Bind every accumulated value onto `query`, in placeholder order.
    fn bind_all<'q>(
        &'q self,
        mut query: Query<'q, sqlx::Any, AnyArguments<'q>>,
    ) -> Query<'q, sqlx::Any, AnyArguments<'q>> {
        for bind in &self.binds {
            query = match bind {
                SpendBind::Str(s) => query.bind(s.as_str()),
                SpendBind::Int(i) => query.bind(i),
            };
        }
        query
    }
}

fn page_bounds(limit: i64, offset: i64) -> (i64, i64) {
    (limit.clamp(1, SPEND_ROLLUP_MAX_LIMIT), offset.max(0))
}

fn spend_page<T>(mut data: Vec<T>, limit: i64, offset: i64) -> SpendPage<T> {
    let has_more = data.len() > limit as usize;
    data.truncate(limit as usize);
    SpendPage {
        data,
        limit,
        offset,
        has_more,
        next_offset: if has_more {
            offset.checked_add(limit)
        } else {
            None
        },
    }
}

fn bounded_rollup<T>(page: SpendPage<T>, dimension: &'static str) -> Result<Vec<T>> {
    if page.has_more {
        return Err(SpendRollupLimitExceeded {
            dimension,
            limit: page.limit,
        }
        .into());
    }
    Ok(page.data)
}

fn text_order(style: PlaceholderStyle, column: &str) -> String {
    let collation = match style {
        PlaceholderStyle::Dollar => "\"C\"",
        PlaceholderStyle::Question => "BINARY",
    };
    format!("CASE WHEN {column} IS NULL THEN 0 ELSE 1 END, {column} COLLATE {collation}")
}

impl Database {
    /// Insert a new metering event and its durable customer/tag increments.
    /// Separate calls are separate events even if their request IDs coincide.
    pub async fn insert_spend_log(&self, log: &SpendLog) -> Result<()> {
        self.insert_spend_events(&[SpendLogEvent::new(log.clone())])
            .await?;
        Ok(())
    }
    /// Insert new events in bind-limited chunks, atomically with their durable
    /// counters. Call [`Self::insert_spend_events`] with retained identities
    /// when the caller needs to retry the same events.
    pub async fn insert_spend_logs_batch(&self, logs: &[SpendLog]) -> Result<()> {
        let events: Vec<_> = logs.iter().cloned().map(SpendLogEvent::new).collect();
        self.insert_spend_events(&events).await?;
        Ok(())
    }

    /// Atomically persist events, exact tag membership and aggregated durable
    /// customer/tag counters. Only rows returned by INSERT ... RETURNING are
    /// charged, so replaying an event (even split across fallback batches)
    /// never re-applies its counters. Distinct events with the same request ID
    /// remain distinct. Event identities live as long as their retained logs.
    ///
    /// The return value counts newly inserted events. An error rolls back all
    /// chunks and counter updates; a commit error is wrapped in CommitFailed.
    pub async fn insert_spend_events(&self, events: &[SpendLogEvent]) -> Result<usize> {
        if events.is_empty() {
            return Ok(0);
        }
        let mut by_event = HashMap::new();
        let mut unique = Vec::with_capacity(events.len());
        for event in events {
            if event.event_id.is_empty() {
                anyhow::bail!("spend event_id must not be empty");
            }
            if event.log.cost.is_some_and(|cost| !cost.is_finite()) {
                anyhow::bail!("spend cost must be finite");
            }
            if let std::collections::hash_map::Entry::Vacant(slot) =
                by_event.entry(event.event_id.as_str())
            {
                slot.insert(&event.log);
                unique.push(event);
            }
        }
        // event_id is an extra bind beyond the public SpendLog columns.
        let cols = SPEND_LOG_COL_COUNT + 1;
        let chunk_size = (self.param_limit() - 1) / cols;
        let mut tx = self.pool().begin().await?;
        let mut customers = BTreeMap::<String, f64>::new();
        let mut tags = BTreeMap::<String, f64>::new();
        let mut inserted = 0;
        for chunk in unique.chunks(chunk_size) {
            let values = (0..chunk.len())
                .map(|i| {
                    let binds = (0..cols)
                        .map(|j| self.ph(i * cols + j + 1))
                        .collect::<Vec<_>>()
                        .join(", ");
                    format!("({binds}, 1)")
                })
                .collect::<Vec<_>>()
                .join(", ");
            let q = format!(
                "INSERT INTO spend_logs ({SPEND_LOG_COLS}, event_id, metadata_version) VALUES {values} \
                 ON CONFLICT (event_id) DO NOTHING \
                 RETURNING CAST(id AS BIGINT) AS id, event_id"
            );
            let mut query = sqlx::query(&q);
            for event in chunk {
                query = bind_spend_log(query, &event.log).bind(&event.event_id);
            }
            let rows = query.fetch_all(&mut *tx).await?;
            inserted += rows.len();
            let mut memberships = Vec::new();
            for row in rows {
                let id: i64 = row.try_get("id")?;
                let event_id: String = row.try_get("event_id")?;
                let log = by_event
                    .get(event_id.as_str())
                    .ok_or_else(|| anyhow!("insert returned an unknown spend event"))?;
                let names = spend_tag_names(log.tags.as_deref(), log.tag_name.as_deref());
                for tag in &names {
                    memberships.push((id, tag.clone()));
                }
                if let Some(cost) = log.cost.filter(|cost| *cost != 0.0) {
                    if let Some(customer) = log
                        .end_user_id
                        .as_deref()
                        .map(str::trim)
                        .filter(|id| !id.is_empty())
                    {
                        *customers.entry(customer.to_string()).or_default() += cost;
                    }
                    for tag in names {
                        *tags.entry(tag).or_default() += cost;
                    }
                }
            }
            self.insert_spend_tag_memberships(&mut tx, &memberships)
                .await?;
        }
        self.increment_spend_totals(&mut tx, "customers", "end_user_id", &customers)
            .await?;
        self.increment_spend_totals(&mut tx, "tags", "name", &tags)
            .await?;
        tx.commit().await.map_err(crate::CommitFailed)?;
        Ok(inserted)
    }

    /// Shared with the v11 legacy backfill; no counter updates occur here.
    pub(crate) async fn insert_spend_tag_memberships(
        &self,
        conn: &mut sqlx::AnyConnection,
        memberships: &[(i64, String)],
    ) -> Result<()> {
        for chunk in memberships.chunks(self.param_limit() / 2) {
            let values = (0..chunk.len())
                .map(|i| format!("({}, {})", self.ph(i * 2 + 1), self.ph(i * 2 + 2)))
                .collect::<Vec<_>>()
                .join(", ");
            let q = format!(
                "INSERT INTO spend_log_tags (spend_log_id, tag_name) VALUES {values} \
                 ON CONFLICT (spend_log_id, tag_name) DO NOTHING"
            );
            let mut query = sqlx::query(&q);
            for (id, tag) in chunk {
                query = query.bind(id).bind(tag);
            }
            query.execute(&mut *conn).await?;
        }
        Ok(())
    }

    /// One set-based UPDATE per bind-limited group of distinct scopes, rather
    /// than N customer/tag UPDATE round trips per request in the writer.
    async fn increment_spend_totals(
        &self,
        conn: &mut sqlx::AnyConnection,
        table: &str,
        id_column: &str,
        totals: &BTreeMap<String, f64>,
    ) -> Result<()> {
        let totals: Vec<_> = totals.iter().collect();
        for chunk in totals.chunks(self.param_limit() / 2) {
            let values = (0..chunk.len())
                .map(|i| {
                    format!(
                        "(CAST({} AS TEXT), CAST({} AS DOUBLE PRECISION))",
                        self.ph(i * 2 + 1),
                        self.ph(i * 2 + 2)
                    )
                })
                .collect::<Vec<_>>()
                .join(", ");
            // Identifiers come only from the two call sites above.
            let q = format!(
                "WITH deltas(scope_id, delta) AS (VALUES {values}) \
                 UPDATE {table} SET spend = COALESCE({table}.spend, 0.0) + deltas.delta \
                 FROM deltas WHERE {table}.{id_column} = deltas.scope_id"
            );
            let mut query = sqlx::query(&q);
            for (id, delta) in chunk {
                if !delta.is_finite() {
                    anyhow::bail!("aggregated spend must be finite");
                }
                query = query.bind(id.as_str()).bind(**delta);
            }
            query.execute(&mut *conn).await?;
        }
        Ok(())
    }
    pub async fn query_spend(&self, filter: &SpendFilter) -> Result<Vec<SpendLog>> {
        let w = SpendWhereBuilder::new(self.placeholder_style())
            .cmp_timestamp("timestamp", ">=", filter.start_date.as_deref())
            .cmp_timestamp("timestamp", "<=", filter.end_date.as_deref())
            .cmp_str("key_id", "=", filter.key_id.as_deref())
            .cmp_str("team_id", "=", filter.team_id.as_deref())
            .cmp_str("model", "=", filter.model.as_deref())
            .cmp_str("deployment_id", "=", filter.deployment_id.as_deref())
            .cmp_str("requested_model", "=", filter.requested_model.as_deref())
            .cmp_str("model_group", "=", filter.model_group.as_deref())
            .cmp_str("request_id", "=", filter.request_id.as_deref())
            .matches_tag(filter.tag_name.as_deref());
        let where_sql = w.where_sql();
        let limit = filter.limit.unwrap_or(100).clamp(1, 1000);
        let offset = filter.offset.unwrap_or(0).max(0);
        // `id DESC` tiebreaker: rows sharing a timestamp would otherwise
        // have no stable order, so offset-based pages could duplicate or
        // skip rows between requests. `clamp` keeps a huge/negative `limit`
        // (SQLite treats a negative LIMIT as "no cap" → full-table dump)
        // bounded, matching [`Self::query_spend_paginated`].
        let q = format!(
            "SELECT {SPEND_LOG_SELECT_COLS} \
 FROM spend_logs_read AS spend_logs{where_sql} ORDER BY timestamp DESC, id DESC LIMIT {} OFFSET {}",
            self.ph(w.next_idx()),
            self.ph(w.next_idx() + 1)
        );
        let rows = w
            .bind_all(sqlx::query(&q))
            .bind(limit)
            .bind(offset)
            .fetch_all(self.pool())
            .await?;
        rows.iter().map(row_to_spend).collect()
    }
    /// Fetch a single spend log by its `request_id`. Returns `None` when
    /// no row matches. Used by `GET /request/logs/{request_id}`: a direct
    /// indexed lookup instead of scanning the newest N rows in memory
    /// (which false-404'd on any row older than the scan window).
    ///
    /// `request_id` carries no UNIQUE constraint (only the v6 non-unique
    /// index), so two rows can share one (e.g. a retry that logs a
    /// partial-success row plus a final row, or a colliding client-supplied
    /// id). The query orders by `id DESC` before `LIMIT 1` so the
    /// most-recent row wins deterministically, matching the spend-list
    /// convention — without the `ORDER BY` the planner would return an
    /// arbitrary row and silently hide the rest.
    pub async fn query_spend_by_request_id(&self, request_id: &str) -> Result<Option<SpendLog>> {
        let q = format!(
            "SELECT {SPEND_LOG_SELECT_COLS} \
 FROM spend_logs_read AS spend_logs WHERE request_id = {} ORDER BY id DESC LIMIT 1",
            self.ph(1)
        );
        let row = sqlx::query(&q)
            .bind(request_id)
            .fetch_optional(self.pool())
            .await?;
        row.map(|r| row_to_spend(&r)).transpose()
    }
    /// Cursor-paginated spend query. When sorting by `id` (the default),
    /// returns rows on the far side of `cursor` from the sort direction:
    /// `id < cursor` for `DESC`, `id > cursor` for `ASC`. When sorting by a
    /// different column the cursor is ignored and paging falls back to
    /// `OFFSET` only. Filters apply as in [`Self::query_spend`].
    ///
    /// The returned vector is ordered by `{order_col} {order_dir}, id DESC`.
    /// The next cursor value is the `id` of the last element (or `None` when
    /// the page is empty or there are no further rows).
    ///
    /// The keyset filter is only meaningful when sorting by `id` so the
    /// filter and the ORDER BY stay on the same column; applying both under a
    /// different sort would return duplicated or skipped rows across pages.
    pub async fn query_spend_paginated(
        &self,
        cursor: Option<i64>,
        limit: i64,
        filter: &SpendFilter,
    ) -> Result<Vec<SpendLog>> {
        // Whitelist sort columns (never interpolate untrusted identifiers).
        let order_col = match filter.sort_by.as_deref().unwrap_or("id") {
            "model" => "model",
            "provider" => "provider",
            "cost" => "cost",
            "key_id" => "key_id",
            "team_id" => "team_id",
            "prompt_tokens" => "prompt_tokens",
            "completion_tokens" => "completion_tokens",
            "timestamp" => "timestamp",
            _ => "id",
        };
        let safe_limit = limit.clamp(1, 1000);
        let order_dir = match filter.sort_order.as_deref().unwrap_or("desc") {
            "asc" | "ASC" => "ASC",
            _ => "DESC",
        };
        // The cursor is a keyset on `id`; only apply it when the sort is on
        // `id` so the filter and the ORDER BY stay on the same column. The
        // comparison operator must agree with the sort direction: a DESC walk
        // continues backward (`id < cursor`), an ASC walk continues forward
        // (`id > cursor`). Using `<` unconditionally made ASC cursor pages
        // re-filter to ids below the cursor while ordering them ascending,
        // returning duplicates of already-emitted rows or an empty set.
        let cursor = if order_col == "id" { cursor } else { None };
        let cursor_op = if order_dir == "DESC" { "<" } else { ">" };
        let w = SpendWhereBuilder::new(self.placeholder_style())
            .cmp_i64("id", cursor_op, cursor)
            .cmp_timestamp("timestamp", ">=", filter.start_date.as_deref())
            .cmp_timestamp("timestamp", "<=", filter.end_date.as_deref())
            .cmp_str("key_id", "=", filter.key_id.as_deref())
            .cmp_str("team_id", "=", filter.team_id.as_deref())
            .cmp_str("model", "=", filter.model.as_deref())
            .cmp_str("deployment_id", "=", filter.deployment_id.as_deref())
            .cmp_str("requested_model", "=", filter.requested_model.as_deref())
            .cmp_str("model_group", "=", filter.model_group.as_deref())
            .cmp_str("request_id", "=", filter.request_id.as_deref())
            .matches_tag(filter.tag_name.as_deref());
        let where_sql = w.where_sql();
        let offset = filter.offset.unwrap_or(0).max(0);
        let q = format!(
            "SELECT {SPEND_LOG_SELECT_COLS} \
 FROM spend_logs_read AS spend_logs{where_sql} ORDER BY {order_col} {order_dir}, id {order_dir} LIMIT {} OFFSET {}",
            self.ph(w.next_idx()),
            self.ph(w.next_idx() + 1)
        );
        let rows = w
            .bind_all(sqlx::query(&q))
            .bind(safe_limit)
            .bind(offset)
            .fetch_all(self.pool())
            .await?;
        rows.iter().map(row_to_spend).collect()
    }
    /// Total spend (SUM of `cost`) across all keys, optionally filtered to
    /// a date range. `None` bounds mean "no bound on that side".
    pub async fn total_spend(&self, start: Option<&str>, end: Option<&str>) -> Result<f64> {
        let w = SpendWhereBuilder::new(self.placeholder_style())
            .cmp_timestamp("timestamp", ">=", start)
            .cmp_timestamp("timestamp", "<=", end);
        let where_sql = w.where_sql();
        let q = format!(
            "SELECT COALESCE(SUM(cost), 0.0) AS total FROM spend_logs_read AS spend_logs{where_sql}"
        );
        let row = w.bind_all(sqlx::query(&q)).fetch_one(self.pool()).await?;
        // COALESCE guarantees a non-NULL value, so a decode failure is a
        // genuine type bug (e.g. a schema regression reintroducing REAL
        // FLOAT4 on Postgres) — propagate it rather than masking it as 0.0.
        let total: f64 = row
            .try_get("total")
            .map_err(|e| anyhow!("failed to read total_spend: {e}"))?;
        Ok(total)
    }
    /// Daily spend for the last `days` days, returned as
    /// `(YYYY-MM-DD, total_cost)` tuples ordered by date ascending.
    ///
    /// The date truncation is portable across SQLite (`substr`) and
    /// Postgres (`date_trunc`). We dispatch on the placeholder style as a
    /// proxy for the backend.
    ///
    /// On Postgres the `timestamp` column is double-quoted
    /// (`spend_logs."timestamp"`) to remove any ambiguity between the
    /// column name and the SQL `TIMESTAMP` type keyword. Postgres classifies
    /// `timestamp` as a `col_name_keyword` so the unquoted form resolves to
    /// the column today, but quoting makes the intent explicit and survives
    /// a future parser tightening or a column rename.
    pub async fn daily_activity(&self, days: i64) -> Result<Vec<DailySpend>> {
        let safe_days = days.max(1);
        let (ts_col, date_expr) = match self.placeholder_style() {
            PlaceholderStyle::Dollar => (
                r#"spend_logs."timestamp""#,
                r#"TO_CHAR(spend_logs."timestamp"::date, 'YYYY-MM-DD')"#,
            ),
            PlaceholderStyle::Question => ("timestamp", "substr(timestamp, 1, 10)"),
        };
        let q = format!(
            "SELECT {date_expr} AS day, COALESCE(SUM(cost), 0.0) AS total \
 FROM spend_logs_read AS spend_logs \
 WHERE {ts_col} >= {} \
 GROUP BY {date_expr} \
 ORDER BY day ASC",
            self.ph(1)
        );
        let since = crate::compute_since_iso(safe_days);
        let rows = sqlx::query(&q).bind(since).fetch_all(self.pool()).await?;
        let mut out: Vec<DailySpend> = Vec::with_capacity(rows.len());
        for row in &rows {
            let day: String = row
                .try_get("day")
                .map_err(|e| anyhow!("failed to read daily day: {e}"))?;
            let total: f64 = row
                .try_get("total")
                .map_err(|e| anyhow!("failed to read daily total: {e}"))?;
            out.push((day, total));
        }
        Ok(out)
    }
    pub async fn get_total_spend(&self, key_id: Option<&str>, since: Option<&str>) -> Result<f64> {
        let w = SpendWhereBuilder::new(self.placeholder_style())
            .cmp_str("key_id", "=", key_id)
            .cmp_timestamp("timestamp", ">=", since);
        let where_sql = w.where_sql();
        let q = format!(
            "SELECT COALESCE(SUM(cost), 0.0) AS total FROM spend_logs_read AS spend_logs{where_sql}"
        );
        let row = w.bind_all(sqlx::query(&q)).fetch_one(self.pool()).await?;
        let total: f64 = row
            .try_get("total")
            .map_err(|e| anyhow!("failed to read total_spend: {e}"))?;
        Ok(total)
    }
    pub async fn count_spend(&self, key_id: Option<&str>, since: Option<&str>) -> Result<i64> {
        let w = SpendWhereBuilder::new(self.placeholder_style())
            .cmp_str("key_id", "=", key_id)
            .cmp_timestamp("timestamp", ">=", since);
        let where_sql = w.where_sql();
        let q = format!("SELECT COUNT(*) AS total FROM spend_logs_read AS spend_logs{where_sql}");
        let row = w.bind_all(sqlx::query(&q)).fetch_one(self.pool()).await?;
        // Postgres names the unaliased column "count"; SQLite names it
        // "COUNT(*)". An explicit alias normalizes both to "total".
        let count: i64 = row
            .try_get::<i64, _>("total")
            .map_err(|e| anyhow!("failed to read count: {e}"))?;
        Ok(count)
    }
    /// Count `spend_logs` rows matching the given filter. Applies the same
    /// `WHERE` clauses as [`Self::query_spend`] (date range, key_id,
    /// team_id, model, request_id) and ignores paging/sort params. Use this
    /// — not [`Self::count_spend`] — whenever the caller already filters by
    /// more than `key_id` + `start_date`, so the reported total describes
    /// the same row set as the page being returned.
    pub async fn count_spend_filtered(&self, filter: &SpendFilter) -> Result<i64> {
        let w = SpendWhereBuilder::new(self.placeholder_style())
            .cmp_timestamp("timestamp", ">=", filter.start_date.as_deref())
            .cmp_timestamp("timestamp", "<=", filter.end_date.as_deref())
            .cmp_str("key_id", "=", filter.key_id.as_deref())
            .cmp_str("team_id", "=", filter.team_id.as_deref())
            .cmp_str("model", "=", filter.model.as_deref())
            .cmp_str("deployment_id", "=", filter.deployment_id.as_deref())
            .cmp_str("requested_model", "=", filter.requested_model.as_deref())
            .cmp_str("model_group", "=", filter.model_group.as_deref())
            .cmp_str("request_id", "=", filter.request_id.as_deref())
            .matches_tag(filter.tag_name.as_deref());
        let where_sql = w.where_sql();
        let q = format!("SELECT COUNT(*) AS total FROM spend_logs_read AS spend_logs{where_sql}");
        let row = w.bind_all(sqlx::query(&q)).fetch_one(self.pool()).await?;
        let count: i64 = row
            .try_get::<i64, _>("total")
            .map_err(|e| anyhow!("failed to read count: {e}"))?;
        Ok(count)
    }
    /// Sum of `cost` across `spend_logs` rows matching the given filter
    /// (same `WHERE` clauses as [`Self::query_spend`], ignoring paging/sort).
    /// Returns `0.0` when no rows match. Use this — not a per-page sum —
    /// when reporting total spend over the caller's full filter scope.
    pub async fn total_spend_filtered(&self, filter: &SpendFilter) -> Result<f64> {
        let w = SpendWhereBuilder::new(self.placeholder_style())
            .cmp_timestamp("timestamp", ">=", filter.start_date.as_deref())
            .cmp_timestamp("timestamp", "<=", filter.end_date.as_deref())
            .cmp_str("key_id", "=", filter.key_id.as_deref())
            .cmp_str("team_id", "=", filter.team_id.as_deref())
            .cmp_str("model", "=", filter.model.as_deref())
            .cmp_str("deployment_id", "=", filter.deployment_id.as_deref())
            .cmp_str("requested_model", "=", filter.requested_model.as_deref())
            .cmp_str("model_group", "=", filter.model_group.as_deref())
            .cmp_str("request_id", "=", filter.request_id.as_deref())
            .matches_tag(filter.tag_name.as_deref());
        let where_sql = w.where_sql();
        let q = format!(
            "SELECT COALESCE(SUM(cost), 0.0) AS total FROM spend_logs_read AS spend_logs{where_sql}"
        );
        let row = w.bind_all(sqlx::query(&q)).fetch_one(self.pool()).await?;
        let total: f64 = row
            .try_get("total")
            .map_err(|e| anyhow!("failed to read total_spend_filtered: {e}"))?;
        Ok(total)
    }

    /// Persisted spend totals for an explicit bounded set of key IDs. Chunked
    /// binds avoid backend parameter limits and one query per management row.
    /// Absent keys and keys without retained spend records return zero.
    pub async fn total_spend_for_keys(&self, key_ids: &[&str]) -> Result<HashMap<String, f64>> {
        let mut totals = key_ids
            .iter()
            .map(|id| ((*id).to_owned(), 0.0))
            .collect::<HashMap<_, _>>();
        for keys in key_ids.chunks(200) {
            let binds = (1..=keys.len())
                .map(|index| self.ph(index))
                .collect::<Vec<_>>()
                .join(", ");
            let statement = format!(
                "SELECT key_id, COALESCE(SUM(cost), 0.0) AS total FROM spend_logs_read \
                 WHERE key_id IN ({binds}) GROUP BY key_id"
            );
            let mut query = sqlx::query(&statement);
            for key in keys {
                query = query.bind(*key);
            }
            for row in query.fetch_all(self.pool()).await? {
                totals.insert(row.try_get("key_id")?, row.try_get("total")?);
            }
        }
        Ok(totals)
    }
    /// Delete at most [`crate::SPEND_PRUNE_BATCH_SIZE`] old spend-log rows in
    /// one transaction. The caller may repeat, yielding between batches so
    /// retention never holds SQLite's writer lock for an unbounded DELETE.
    /// Tag memberships cascade with the logs; durable lifetime counters are
    /// deliberately not decremented by telemetry retention.
    ///
    /// The comparison uses `<` (not `<=`) so a daily prune at midnight leaves
    /// today's rows untouched even if the cutoff timestamp happens to equal
    /// an existing row's `timestamp`.
    pub async fn delete_spend_logs_before(&self, timestamp: &str) -> Result<u64> {
        self.delete_spend_logs_before_batch(timestamp, crate::SPEND_PRUNE_BATCH_SIZE)
            .await
    }

    /// Bounded retention primitive, also useful for smaller maintenance
    /// slices. The limit is clamped before SQL on both backends.
    pub async fn delete_spend_logs_before_batch(&self, timestamp: &str, limit: i64) -> Result<u64> {
        let q = format!(
            "DELETE FROM spend_logs WHERE id IN \
             (SELECT id FROM spend_logs_read WHERE timestamp < {} \
              ORDER BY timestamp, id LIMIT {})",
            self.ph(1),
            self.ph(2)
        );
        // Normalize the cutoff to the fixed-precision form written by
        // `bind_spend_log`, so the TEXT `<` comparison tracks chronology.
        let res = sqlx::query(&q)
            .bind(crate::normalize_iso(timestamp))
            .bind(limit.clamp(1, crate::SPEND_PRUNE_BATCH_SIZE))
            .execute(self.pool())
            .await?;
        Ok(res.rows_affected())
    }
    /// Complete key rollup, bounded to 1,000 groups. Larger result sets return
    /// `SpendRollupLimitExceeded`; use the paginated variant to walk them.
    pub async fn query_spend_by_key(
        &self,
        start_date: Option<&str>,
        end_date: Option<&str>,
    ) -> Result<Vec<SpendByKey>> {
        bounded_rollup(
            self.query_spend_by_key_paginated(start_date, end_date, SPEND_ROLLUP_MAX_LIMIT, 0)
                .await?,
            "key",
        )
    }

    /// Page complete key aggregates, with newest normalized `last_used` and
    /// stable null-first/key-id tie-breaking after total spend descending.
    pub async fn query_spend_by_key_paginated(
        &self,
        start_date: Option<&str>,
        end_date: Option<&str>,
        limit: i64,
        offset: i64,
    ) -> Result<SpendByKeyPage> {
        let (limit, offset) = page_bounds(limit, offset);
        let w = SpendWhereBuilder::new(self.placeholder_style())
            .cmp_timestamp("s.timestamp", ">=", start_date)
            .cmp_timestamp("s.timestamp", "<=", end_date);
        let where_sql = w.where_sql();
        // LEFT JOIN with keys to get the alias, aggregate by key_id.
        // `k.key_alias` must appear in GROUP BY: Postgres rejects selecting a
        // non-aggregated column that is not grouped on (SQLite tolerates it).
        let q = format!(
            "SELECT s.key_id, k.key_alias, \
             COALESCE(SUM(s.cost), 0.0) AS total_spend, \
             COUNT(*) AS total_requests, \
             MAX(s.timestamp) AS last_used \
              FROM spend_logs_read s \
             LEFT JOIN keys k ON s.key_id = k.key_id \
             {where_sql} \
             GROUP BY s.key_id, k.key_alias \
              ORDER BY total_spend DESC, {} LIMIT {} OFFSET {}",
            text_order(self.placeholder_style(), "s.key_id"),
            self.ph(w.next_idx()),
            self.ph(w.next_idx() + 1)
        );
        let rows = w
            .bind_all(sqlx::query(&q))
            .bind(limit + 1)
            .bind(offset)
            .fetch_all(self.pool())
            .await?;
        let mut out: Vec<SpendByKey> = Vec::with_capacity(rows.len());
        for row in &rows {
            // `key_id` is nullable in spend_logs; keep NULL as `None` rather
            // than collapsing it to "" so clients can tell "no key" apart
            // from a key whose id happens to be empty.
            // Propagate decode failures (e.g. a schema regression changing the
            // column type) instead of silently attributing spend to NULL.
            let key_id: Option<String> = row
                .try_get("key_id")
                .map_err(|e| anyhow!("failed to read key_id: {e}"))?;
            let key_alias: Option<String> = row.try_get("key_alias")?;
            let total_spend: f64 = row
                .try_get("total_spend")
                .map_err(|e| anyhow!("failed to read total_spend: {e}"))?;
            let total_requests: i64 = row
                .try_get("total_requests")
                .map_err(|e| anyhow!("failed to read total_requests: {e}"))?;
            let last_used: Option<String> = row.try_get("last_used")?;
            out.push(SpendByKey {
                key_id,
                key_alias,
                total_spend,
                total_requests,
                last_used,
            });
        }
        Ok(spend_page(out, limit, offset))
    }
    /// Complete model/provider rollup, bounded to 1,000 groups.
    pub async fn query_spend_by_model(
        &self,
        start_date: Option<&str>,
        end_date: Option<&str>,
    ) -> Result<Vec<SpendByModel>> {
        bounded_rollup(
            self.query_spend_by_model_paginated(start_date, end_date, SPEND_ROLLUP_MAX_LIMIT, 0)
                .await?,
            "model",
        )
    }

    /// Page full model/provider aggregates, with stable bytewise tie-breaking.
    pub async fn query_spend_by_model_paginated(
        &self,
        start_date: Option<&str>,
        end_date: Option<&str>,
        limit: i64,
        offset: i64,
    ) -> Result<SpendByModelPage> {
        let (limit, offset) = page_bounds(limit, offset);
        let w = SpendWhereBuilder::new(self.placeholder_style())
            .cmp_timestamp("s.timestamp", ">=", start_date)
            .cmp_timestamp("s.timestamp", "<=", end_date);
        let where_sql = w.where_sql();
        let q = format!(
            "SELECT s.model, s.provider, \
             COALESCE(SUM(s.cost), 0.0) AS total_spend, \
             COUNT(*) AS total_requests \
              FROM spend_logs_read s \
             {where_sql} \
             GROUP BY s.model, s.provider \
              ORDER BY total_spend DESC, {}, {} LIMIT {} OFFSET {}",
            text_order(self.placeholder_style(), "s.model"),
            text_order(self.placeholder_style(), "s.provider"),
            self.ph(w.next_idx()),
            self.ph(w.next_idx() + 1)
        );
        let rows = w
            .bind_all(sqlx::query(&q))
            .bind(limit + 1)
            .bind(offset)
            .fetch_all(self.pool())
            .await?;
        let mut out: Vec<SpendByModel> = Vec::with_capacity(rows.len());
        for row in &rows {
            let model: Option<String> = row.try_get("model")?;
            let provider: Option<String> = row.try_get("provider")?;
            let total_spend: f64 = row
                .try_get("total_spend")
                .map_err(|e| anyhow!("failed to read total_spend: {e}"))?;
            let total_requests: i64 = row
                .try_get("total_requests")
                .map_err(|e| anyhow!("failed to read total_requests: {e}"))?;
            out.push(SpendByModel {
                model,
                provider,
                total_spend,
                total_requests,
            });
        }
        Ok(spend_page(out, limit, offset))
    }
    /// Complete tag rollup for small result sets. Errors rather than silently
    /// truncating when more than 1,000 tags match; use the paginated variant
    /// for larger sets.
    pub async fn query_spend_by_tag(
        &self,
        start_date: Option<&str>,
        end_date: Option<&str>,
    ) -> Result<Vec<SpendByTag>> {
        bounded_rollup(
            self.query_spend_by_tag_paginated(start_date, end_date, SPEND_ROLLUP_MAX_LIMIT, 0)
                .await?,
            "tag",
        )
    }

    /// Bounded tag rollup over the FULL date range. The indexed membership
    /// join groups by individual tags inside SQL, avoiding an unbounded Rust
    /// fetch of distinct JSON tag combinations. LIMIT is applied only AFTER
    /// aggregation: no input rows or per-tag billing totals are truncated.
    /// Each request counts once toward every distinct tag; untagged requests
    /// form a single NULL group. Tie-breaking and NULL ordering are stable on
    /// both backends. Pagination metadata advertises additional tag groups.
    pub async fn query_spend_by_tag_paginated(
        &self,
        start_date: Option<&str>,
        end_date: Option<&str>,
        limit: i64,
        offset: i64,
    ) -> Result<SpendByTagPage> {
        let (limit, offset) = page_bounds(limit, offset);
        let w = SpendWhereBuilder::new(self.placeholder_style())
            .cmp_timestamp("s.timestamp", ">=", start_date)
            .cmp_timestamp("s.timestamp", "<=", end_date);
        let where_sql = w.where_sql();
        let q = format!(
            "SELECT st.tag_name, COALESCE(SUM(s.cost), 0.0) AS total_spend, \
             COUNT(*) AS total_requests \
              FROM spend_logs_read s \
              LEFT JOIN spend_log_tags_read st ON st.spend_log_id = s.id \
             {where_sql} \
             GROUP BY st.tag_name \
              ORDER BY total_spend DESC, {} \
              LIMIT {} OFFSET {}",
            text_order(self.placeholder_style(), "st.tag_name"),
            self.ph(w.next_idx()),
            self.ph(w.next_idx() + 1)
        );
        let rows = w
            .bind_all(sqlx::query(&q))
            .bind(limit + 1)
            .bind(offset)
            .fetch_all(self.pool())
            .await?;
        let mut data = Vec::with_capacity(rows.len());
        for row in &rows {
            data.push(SpendByTag {
                tag_name: row.try_get("tag_name")?,
                total_spend: row.try_get("total_spend")?,
                total_requests: row.try_get("total_requests")?,
            });
        }
        Ok(spend_page(data, limit, offset))
    }
    /// Complete user rollup, bounded to 1,000 groups.
    pub async fn query_spend_by_user(
        &self,
        start_date: Option<&str>,
        end_date: Option<&str>,
    ) -> Result<Vec<SpendByUser>> {
        bounded_rollup(
            self.query_spend_by_user_paginated(start_date, end_date, SPEND_ROLLUP_MAX_LIMIT, 0)
                .await?,
            "user",
        )
    }

    /// Page full user aggregates with configured names and stable id ordering.
    pub async fn query_spend_by_user_paginated(
        &self,
        start_date: Option<&str>,
        end_date: Option<&str>,
        limit: i64,
        offset: i64,
    ) -> Result<SpendByUserPage> {
        let (limit, offset) = page_bounds(limit, offset);
        let w = SpendWhereBuilder::new(self.placeholder_style())
            .cmp_timestamp("s.timestamp", ">=", start_date)
            .cmp_timestamp("s.timestamp", "<=", end_date);
        let where_sql = w.where_sql();
        // `u.user_name` must appear in GROUP BY: Postgres rejects selecting a
        // non-aggregated column that is not grouped on (SQLite tolerates it).
        let q = format!(
            "SELECT s.user_id, u.user_name, \
             COALESCE(SUM(s.cost), 0.0) AS total_spend, \
             COUNT(*) AS total_requests \
              FROM spend_logs_read s \
             LEFT JOIN users u ON s.user_id = u.user_id \
             {where_sql} \
             GROUP BY s.user_id, u.user_name \
              ORDER BY total_spend DESC, {} LIMIT {} OFFSET {}",
            text_order(self.placeholder_style(), "s.user_id"),
            self.ph(w.next_idx()),
            self.ph(w.next_idx() + 1)
        );
        let rows = w
            .bind_all(sqlx::query(&q))
            .bind(limit + 1)
            .bind(offset)
            .fetch_all(self.pool())
            .await?;
        let mut out: Vec<SpendByUser> = Vec::with_capacity(rows.len());
        for row in &rows {
            // Propagate decode failures instead of silently attributing spend to NULL.
            let user_id: Option<String> = row
                .try_get("user_id")
                .map_err(|e| anyhow!("failed to read user_id: {e}"))?;
            let user_name: Option<String> = row.try_get("user_name")?;
            let total_spend: f64 = row
                .try_get("total_spend")
                .map_err(|e| anyhow!("failed to read total_spend: {e}"))?;
            let total_requests: i64 = row
                .try_get("total_requests")
                .map_err(|e| anyhow!("failed to read total_requests: {e}"))?;
            out.push(SpendByUser {
                user_id,
                user_name,
                total_spend,
                total_requests,
            });
        }
        Ok(spend_page(out, limit, offset))
    }
    /// Complete team rollup, bounded to 1,000 groups.
    pub async fn query_spend_by_team(
        &self,
        start_date: Option<&str>,
        end_date: Option<&str>,
    ) -> Result<Vec<SpendByTeam>> {
        bounded_rollup(
            self.query_spend_by_team_paginated(start_date, end_date, SPEND_ROLLUP_MAX_LIMIT, 0)
                .await?,
            "team",
        )
    }

    /// Page full team aggregates with configured names and stable id ordering.
    pub async fn query_spend_by_team_paginated(
        &self,
        start_date: Option<&str>,
        end_date: Option<&str>,
        limit: i64,
        offset: i64,
    ) -> Result<SpendByTeamPage> {
        let (limit, offset) = page_bounds(limit, offset);
        let w = SpendWhereBuilder::new(self.placeholder_style())
            .cmp_timestamp("s.timestamp", ">=", start_date)
            .cmp_timestamp("s.timestamp", "<=", end_date);
        let where_sql = w.where_sql();
        let q = format!(
            "SELECT s.team_id, t.team_name, \
             COALESCE(SUM(s.cost), 0.0) AS total_spend, \
             COUNT(*) AS total_requests \
              FROM spend_logs_read s \
             LEFT JOIN teams t ON s.team_id = t.team_id \
             {where_sql} \
             GROUP BY s.team_id, t.team_name \
              ORDER BY total_spend DESC, {} LIMIT {} OFFSET {}",
            text_order(self.placeholder_style(), "s.team_id"),
            self.ph(w.next_idx()),
            self.ph(w.next_idx() + 1)
        );
        let rows = w
            .bind_all(sqlx::query(&q))
            .bind(limit + 1)
            .bind(offset)
            .fetch_all(self.pool())
            .await?;
        let mut out: Vec<SpendByTeam> = Vec::with_capacity(rows.len());
        for row in &rows {
            // Propagate decode failures instead of silently attributing spend to NULL.
            let team_id: Option<String> = row
                .try_get("team_id")
                .map_err(|e| anyhow!("failed to read team_id: {e}"))?;
            let team_name: Option<String> = row.try_get("team_name")?;
            let total_spend: f64 = row
                .try_get("total_spend")
                .map_err(|e| anyhow!("failed to read total_spend: {e}"))?;
            let total_requests: i64 = row
                .try_get("total_requests")
                .map_err(|e| anyhow!("failed to read total_requests: {e}"))?;
            out.push(SpendByTeam {
                team_id,
                team_name,
                total_spend,
                total_requests,
            });
        }
        Ok(spend_page(out, limit, offset))
    }
    /// Aggregated spend by organization from spend_logs, with optional date
    /// range. The `org_id` is resolved through the key each row was logged
    /// against (the `keys` table carries `org_id`); rows without a key, or
    /// whose key has no org, aggregate into a `null` group.
    pub async fn query_spend_by_org(
        &self,
        start_date: Option<&str>,
        end_date: Option<&str>,
    ) -> Result<Vec<SpendByOrg>> {
        bounded_rollup(
            self.query_spend_by_org_paginated(start_date, end_date, SPEND_ROLLUP_MAX_LIMIT, 0)
                .await?,
            "org",
        )
    }

    /// Page full organization aggregates resolved through keys. Null-first/org
    /// id tie-breaking is stable; each returned total includes every input row.
    pub async fn query_spend_by_org_paginated(
        &self,
        start_date: Option<&str>,
        end_date: Option<&str>,
        limit: i64,
        offset: i64,
    ) -> Result<SpendByOrgPage> {
        let (limit, offset) = page_bounds(limit, offset);
        let w = SpendWhereBuilder::new(self.placeholder_style())
            .cmp_timestamp("s.timestamp", ">=", start_date)
            .cmp_timestamp("s.timestamp", "<=", end_date);
        let where_sql = w.where_sql();
        let q = format!(
            "SELECT k.org_id, \
             COALESCE(SUM(s.cost), 0.0) AS total_spend, \
             COUNT(*) AS total_requests \
              FROM spend_logs_read s \
             LEFT JOIN keys k ON s.key_id = k.key_id \
             {where_sql} \
             GROUP BY k.org_id \
              ORDER BY total_spend DESC, {} LIMIT {} OFFSET {}",
            text_order(self.placeholder_style(), "k.org_id"),
            self.ph(w.next_idx()),
            self.ph(w.next_idx() + 1)
        );
        let rows = w
            .bind_all(sqlx::query(&q))
            .bind(limit + 1)
            .bind(offset)
            .fetch_all(self.pool())
            .await?;
        let mut out: Vec<SpendByOrg> = Vec::with_capacity(rows.len());
        for row in &rows {
            // Propagate decode failures instead of silently attributing spend to NULL.
            let org_id: Option<String> = row
                .try_get("org_id")
                .map_err(|e| anyhow!("failed to read org_id: {e}"))?;
            let total_spend: f64 = row
                .try_get("total_spend")
                .map_err(|e| anyhow!("failed to read total_spend: {e}"))?;
            let total_requests: i64 = row
                .try_get("total_requests")
                .map_err(|e| anyhow!("failed to read total_requests: {e}"))?;
            out.push(SpendByOrg {
                org_id,
                total_spend,
                total_requests,
            });
        }
        Ok(spend_page(out, limit, offset))
    }
}

#[cfg(test)]
mod end_date_tests {
    use crate::inclusive_spend_end_date as expand_bare_end_date;

    #[test]
    fn bare_end_date_expands_to_end_of_day() {
        assert_eq!(
            expand_bare_end_date("2026-07-31"),
            "2026-07-31T23:59:59.999999999Z"
        );
    }

    #[test]
    fn full_timestamps_and_garbage_pass_through() {
        assert_eq!(
            expand_bare_end_date("2026-07-31T12:00:00Z"),
            "2026-07-31T12:00:00Z"
        );
        assert_eq!(expand_bare_end_date("not-a-date"), "not-a-date");
    }
}
