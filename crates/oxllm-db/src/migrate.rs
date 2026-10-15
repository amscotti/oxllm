//! Schema migrations: versioned DDL batches plus the `_schema_version`
//! bookkeeping that keeps them idempotent. Everything here runs at startup;
//! runtime query code lives in the domain modules under [`crate::repos`].

use std::time::Duration;

use anyhow::Result;
use sqlx::AnyConnection;
use sqlx::Executor;
use sqlx::Row;

use crate::{Database, PlaceholderStyle};

/// Bound short DDL/readiness lock acquisition. Online preparation has its own
/// longer bounded budget; historical metadata repair holds neither lock.
const MIGRATION_LOCK_TIMEOUT: Duration = Duration::from_secs(2);

/// Fixed advisory-lock key shared by every oxllm process so all replicas
/// contend on the same lock. Chosen as a stable non-trivial constant unlikely
/// to collide with application-level advisory keys ("OXLLMI" in hex).
pub(crate) const OXLLM_MIGRATE_LOCK_KEY: i64 = 0x4f58_4c4c_4d49;

impl Database {
    /// Run all pending migrations.
    ///
    /// H36: migrations are versioned. The `_schema_version` table is
    /// created unconditionally in v0; then each numbered batch runs only
    /// when the live schema version is below the batch's number. After a
    /// batch succeeds, the new version is recorded. This avoids
    /// re-running the full migration SQL on every startup once the
    /// schema is current.
    ///
    /// For databases that pre-date version tracking (tables/columns
    /// already exist, no `_schema_version` row yet), the batch
    /// statements are idempotent (`CREATE TABLE IF NOT EXISTS`,
    /// `CREATE INDEX IF NOT EXISTS`) or tolerated on re-application
    /// (`ADD COLUMN` of an existing column → "duplicate column" /
    /// "already exists" error is swallowed in
    /// `run_migration_batch`), so the first such migrate marks
    /// the current version without disturbing the schema.
    pub async fn migrate(&self) -> Result<()> {
        self.migrate_with_timeouts(
            MIGRATION_LOCK_TIMEOUT,
            crate::postgres_prepare::PREPARATION_TIMEOUT,
        )
        .await
    }

    #[cfg(test)]
    pub(crate) async fn migrate_with_lock_timeout(&self, lock_timeout: Duration) -> Result<()> {
        self.migrate_with_timeouts(lock_timeout, lock_timeout).await
    }

    pub(crate) async fn migrate_with_timeouts(
        &self,
        lock_timeout: Duration,
        preparation_timeout: Duration,
    ) -> Result<()> {
        if matches!(self.placeholder_style(), PlaceholderStyle::Dollar) {
            // Every PostgreSQL path, including fresh and already-ready schemas,
            // acquires the nonblocking session guard before opening a transaction.
            // Do not retain a preflight snapshot while awaiting another builder.
            self.migrate_postgres_online(lock_timeout, preparation_timeout)
                .await?;
        } else {
            if self.get_schema_version().await? >= crate::SCHEMA_VERSION {
                self.repair_spend_metadata_on_startup().await;
                return Ok(());
            }
            // SQLite: the file lock already serializes writers; no advisory
            // lock is needed. Wrap the DDL in a tracked transaction (`BEGIN IMMEDIATE`
            // upgrades to the reserved lock up front, so a concurrent writer
            // blocks instead of interleaving). Without this, a crash between
            // batches leaves a half-migrated schema with no version row —
            // startup then re-runs batches against unknown state. SQLite DDL
            // is transactional, so COMMIT/ROLLBACK here is atomic.
            // sqlx's transaction guard also rolls back on future cancellation.
            let mut tx = self.pool().begin_with("BEGIN IMMEDIATE").await?;
            let result = self.run_all_migrations(&mut tx, false).await;
            match result {
                Ok(()) => {
                    tx.commit().await?;
                }
                Err(e) => {
                    if let Err(rb) = tx.rollback().await {
                        tracing::error!(
                            error = %crate::redact_url_credentials(&rb.to_string()),
                            "sqlite migration rollback failed"
                        );
                    }
                    return Err(e);
                }
            }
        }
        // Separate, bounded commits release every DDL/table/advisory lock
        // before touching historical metadata. Readers already have a fallback.
        self.repair_spend_metadata_on_startup().await;
        Ok(())
    }

    /// Execute all versioned migration batches on the given connection.
    /// Called with either a borrowed transaction (Postgres) or a dedicated
    /// pooled connection (SQLite).
    pub(crate) async fn run_all_migrations(
        &self,
        conn: &mut AnyConnection,
        prepared_spend: bool,
    ) -> Result<()> {
        self.ensure_schema_version_table(&mut *conn).await?;

        let current = self.get_schema_version_on(&mut *conn).await?;
        tracing::debug!(
            current_version = current,
            "database schema version before migrate"
        );

        if current < 1 {
            self.run_prepared_migration_batch(
                v1_statements_for(self.placeholder_style()),
                conn,
                prepared_spend,
            )
            .await?;
            self.set_schema_version(1, &mut *conn).await?;
        }
        if current < 2 {
            self.run_migration_batch(v2_statements_for(self.placeholder_style()), conn)
                .await?;
            self.set_schema_version(2, &mut *conn).await?;
        }
        if current < 3 {
            self.run_migration_batch(v3_statements_for(self.placeholder_style()), conn)
                .await?;
            self.set_schema_version(3, &mut *conn).await?;
        }
        if current < 4 {
            self.run_migration_batch(v4_statements_for(self.placeholder_style()), conn)
                .await?;
            self.set_schema_version(4, &mut *conn).await?;
        }
        if current < 5 {
            self.run_migration_batch(v5_statements_for(), conn).await?;
            self.set_schema_version(5, &mut *conn).await?;
        }
        if current < 6 {
            self.run_prepared_migration_batch(v6_statements_for(), conn, prepared_spend)
                .await?;
            self.set_schema_version(6, &mut *conn).await?;
        }
        if current < 7 {
            self.run_migration_batch(v7_statements_for(self.placeholder_style()), conn)
                .await?;
            self.set_schema_version(7, &mut *conn).await?;
        }
        if current < 8 {
            self.run_prepared_migration_batch(
                v8_statements_for(self.placeholder_style()),
                conn,
                prepared_spend,
            )
            .await?;
            self.set_schema_version(8, &mut *conn).await?;
        }
        if current < 9 {
            self.run_prepared_migration_batch(
                v9_statements_for(self.placeholder_style()),
                conn,
                prepared_spend,
            )
            .await?;
            self.set_schema_version(9, &mut *conn).await?;
        }
        if current < 10 {
            self.run_prepared_migration_batch(
                v10_statements_for(self.placeholder_style()),
                conn,
                prepared_spend,
            )
            .await?;
            self.set_schema_version(10, &mut *conn).await?;
        }
        if current < 11 {
            self.run_prepared_migration_batch(
                v11_statements_for(self.placeholder_style()),
                conn,
                prepared_spend,
            )
            .await?;
            if matches!(self.placeholder_style(), PlaceholderStyle::Dollar) {
                self.ensure_spend_event_index_ready(conn).await?;
            }
            // The original models table used the legacy params column. The source-level
            // rename did not migrate that column on already-versioned DBs.
            if self.has_column_on(conn, "models", "litellm_params").await?
                && !self.has_column_on(conn, "models", "oxllm_params").await?
            {
                sqlx::query("ALTER TABLE models RENAME COLUMN litellm_params TO oxllm_params")
                    .execute(&mut *conn)
                    .await?;
            }
            self.set_schema_version(11, &mut *conn).await?;
        }
        if current < 12 {
            self.run_prepared_migration_batch(
                crate::spend_metadata::statements_for(self.placeholder_style()),
                conn,
                prepared_spend,
            )
            .await?;
            self.set_schema_version(12, &mut *conn).await?;
        }
        if current < 13 {
            self.run_migration_batch(v13_statements_for(self.placeholder_style()), conn)
                .await?;
            self.set_schema_version(13, &mut *conn).await?;
        }
        if current < 14 {
            self.run_migration_batch(v14_statements_for(self.placeholder_style()), conn)
                .await?;
            self.set_schema_version(14, &mut *conn).await?;
        }
        if current < 15 {
            self.run_prepared_migration_batch(
                crate::spend_metadata::deployment_attribution_statements(self.placeholder_style()),
                conn,
                prepared_spend,
            )
            .await?;
            self.set_schema_version(15, &mut *conn).await?;
        }
        if current < 16 {
            self.run_prepared_migration_batch(
                crate::spend_metadata::model_group_attribution_statements(self.placeholder_style()),
                conn,
                prepared_spend,
            )
            .await?;
            self.set_schema_version(16, &mut *conn).await?;
        }
        if current < 17 {
            self.run_prepared_migration_batch(
                crate::spend_metadata::session_attribution_statements(self.placeholder_style()),
                conn,
                prepared_spend,
            )
            .await?;
            self.set_schema_version(17, &mut *conn).await?;
        }
        Ok(())
    }

    /// Skip only spend ALTER/index statements already validated and committed
    /// by online preparation. Even IF NOT EXISTS can acquire a table lock, so
    /// do not re-acquire it around unrelated schema work or history scans.
    async fn run_prepared_migration_batch(
        &self,
        statements: Vec<String>,
        conn: &mut AnyConnection,
        prepared_spend: bool,
    ) -> Result<()> {
        let statements = statements
            .into_iter()
            .filter(|statement| {
                !prepared_spend || !crate::postgres_prepare::prepared_spend_statement(statement)
            })
            .collect();
        self.run_migration_batch(statements, conn).await
    }

    /// IF NOT EXISTS also skips invalid or incorrectly defined named indexes.
    /// A failed/in-progress concurrent preparation must not publish readiness
    /// for the event writer's ON CONFLICT contract. Validate catalog metadata
    /// without scanning spend history or deleting/replacing an operator index.
    pub(crate) async fn ensure_spend_event_index_ready(
        &self,
        conn: &mut AnyConnection,
    ) -> Result<()> {
        crate::postgres_prepare::ensure_event_index_ready(conn).await
    }

    /// Create the `_schema_version` bookkeeping table if it does not yet
    /// exist. Idempotent.
    async fn ensure_schema_version_table(&self, exec: &mut AnyConnection) -> Result<()> {
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS _schema_version (\
             version INTEGER PRIMARY KEY,\
             applied_at TEXT NOT NULL)",
        )
        .execute(&mut *exec)
        .await?;
        // A duplicate-column error aborts an entire PostgreSQL transaction,
        // even if Rust swallows the error. Use its idempotent DDL form;
        // SQLite's specific duplicate-column error is safe to tolerate.
        let add = match self.placeholder_style() {
            PlaceholderStyle::Dollar => "ADD COLUMN IF NOT EXISTS",
            PlaceholderStyle::Question => "ADD COLUMN",
        };
        self.run_migration_batch(
            vec![format!(
                "ALTER TABLE _schema_version {add} applied_at TEXT NOT NULL DEFAULT ''"
            )],
            exec,
        )
        .await?;
        Ok(())
    }

    pub(crate) async fn has_column_on(
        &self,
        conn: &mut AnyConnection,
        table: &str,
        column: &str,
    ) -> Result<bool> {
        let q = match self.placeholder_style() {
            PlaceholderStyle::Dollar => {
                // sqlx::Any supports TEXT, not PostgreSQL's NAME/catalog domains.
                "SELECT CAST(column_name AS TEXT) AS name FROM information_schema.columns \
                 WHERE table_schema = current_schema() AND table_name = $1"
            }
            PlaceholderStyle::Question => "SELECT name FROM pragma_table_info(?)",
        };
        let rows = sqlx::query(q).bind(table).fetch_all(&mut *conn).await?;
        for row in rows {
            if row.try_get::<String, _>("name")? == column {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Read the highest applied migration version from
    /// `_schema_version`. Returns `Ok(0)` when the table is missing or
    /// empty (a database that has not been migrated yet) — the caller
    /// (migrate) treats 0 as "run every batch". Any other failure is
    /// surfaced as `Err`: silently assuming version 0 after a transient
    /// connection/read error would cause migrate to re-run every DDL
    /// batch against a schema of unknown state.
    ///
    /// Exposed for migration tests and the `oxllm migrate` CLI, both of
    /// which read it after `migrate()` to assert/report the resulting
    /// version. Production code paths reach for it via [`Self::migrate`].
    pub async fn get_schema_version(&self) -> Result<i64> {
        let mut conn = self.pool().acquire().await.map_err(|e| {
            anyhow::anyhow!(
                "failed to acquire connection for schema-version read: {}",
                crate::redact_url_credentials(&e.to_string())
            )
        })?;
        match self.get_schema_version_on(&mut *conn).await {
            Ok(v) => Ok(v),
            // A not-yet-migrated database legitimately has no
            // `_schema_version` table; that is version 0, not an error.
            Err(e)
                if e.downcast_ref::<sqlx::Error>()
                    .is_some_and(is_missing_table_error) =>
            {
                Ok(0)
            }
            Err(e) => Err(e),
        }
    }

    /// Connection-scoped variant of [`Self::get_schema_version`] used
    /// internally by [`Self::run_all_migrations`] so the read runs on the
    /// same connection/transaction as the DDL it gates. The caller has
    /// already created `_schema_version`, so every error propagates.
    pub(crate) async fn get_schema_version_on<'e, E>(&self, exec: E) -> Result<i64>
    where
        E: Executor<'e, Database = sqlx::Any>,
    {
        // CAST ... AS BIGINT normalizes the aggregate's type across
        // backends: Postgres' MAX(INTEGER) yields INTEGER (INT4), which
        // sqlx's Any backend will not decode as i64, while SQLite yields
        // INT8. Both accept the standard CAST syntax.
        let q = "SELECT CAST(COALESCE(MAX(version), 0) AS BIGINT) AS v FROM _schema_version";
        let row = sqlx::query(q).fetch_one(exec).await?;
        crate::rows::get_int(&row, "v")
    }

    /// Record that migration batch `version` has been applied.
    ///
    /// The write is idempotent per backend so that two replicas booting cold
    /// against the same database at the same time cannot crash startup: both
    /// read `current < N`, both run batch N (the DDL is itself idempotent),
    /// and both attempt to record N. Postgres uses
    /// `ON CONFLICT (version) DO NOTHING`; SQLite uses `INSERT OR IGNORE`.
    /// The second writer's row is silently discarded and the version remains
    /// correct. PostgreSQL serializes online preparation and final DDL with
    /// a dedicated session advisory lock (see [`Self::migrate`]); replicas
    /// waiting outside SQL snapshots recheck readiness after acquiring it.
    /// Idempotent writes are a second line of defense.
    async fn set_schema_version<'e, E>(&self, version: i64, exec: E) -> Result<()>
    where
        E: Executor<'e, Database = sqlx::Any>,
    {
        let (verb, tail) = match self.placeholder_style() {
            PlaceholderStyle::Dollar => ("INSERT INTO", " ON CONFLICT (version) DO NOTHING"),
            PlaceholderStyle::Question => ("INSERT OR IGNORE INTO", ""),
        };
        let q = format!(
            "{verb} _schema_version (version, applied_at) VALUES ({}, {}){tail}",
            self.ph(1),
            self.ph(2)
        );
        sqlx::query(&q)
            .bind(version)
            .bind(crate::now_iso())
            .execute(exec)
            .await?;
        Ok(())
    }

    /// Execute a batch, tolerating only SQLite's duplicate-column error on
    /// reapplication. PostgreSQL ADD COLUMN statements use IF NOT EXISTS:
    /// swallowing its duplicate-column error would leave the enclosing
    /// transaction aborted. All other errors propagate. Classification uses the structured `sqlx::Error` shape
    /// and the driver's result code rather than a broad substring match, so a
    /// genuinely failing statement (e.g. a syntax error or a missing table)
    /// cannot be silently swallowed just because its text happens to contain
    /// "already exists".
    pub(crate) async fn run_migration_batch(
        &self,
        stmts: Vec<String>,
        conn: &mut AnyConnection,
    ) -> Result<()> {
        for stmt in stmts {
            if let Err(e) = sqlx::query(&stmt).execute(&mut *conn).await {
                if matches!(self.placeholder_style(), PlaceholderStyle::Question)
                    && stmt.starts_with("ALTER TABLE ")
                    && stmt.contains(" ADD COLUMN ")
                    && is_duplicate_column_error(&e)
                {
                    tracing::debug!(
                        error = %crate::redact_url_credentials(&e.to_string()),
                        "tolerated duplicate-column on migration statement"
                    );
                    continue;
                }
                // Name the failing statement: without it a migration failure
                // surfaces only the driver error, leaving the operator to
                // guess which of the ~dozen DDL statements in the batch broke.
                // Both halves are redacted so a DSN echoing through the driver
                // error cannot leak credentials into startup logs.
                tracing::error!(
                    statement = %crate::redact_url_credentials(&stmt),
                    error = %crate::redact_url_credentials(&e.to_string()),
                    "migration statement failed"
                );
                return Err(anyhow::anyhow!(
                    "migration failed on statement {}: {}",
                    crate::redact_url_credentials(&stmt),
                    crate::redact_url_credentials(&e.to_string())
                ));
            }
        }
        Ok(())
    }
}

/// Decide whether `e` is the idempotent "column already exists" error that
/// re-applying an `ADD COLUMN` produces, so [`Database::run_migration_batch`]
/// can tolerate it without masking unrelated failures. We only ever treat a
/// structured `sqlx::Error::Database` (never a connection / protocol error)
/// as tolerable, and within that we narrow by the driver's result code:
/// SQLite emits generic `SQLITE_ERROR` (`"1"`) whose message reads
/// `duplicate column name: …`, and Postgres emits SQLSTATE `42701`
/// (`duplicate_column`). A narrow message fallback covers a future driver
/// wording change without re-opening the broad "already exists" match.
fn is_duplicate_column_error(e: &sqlx::Error) -> bool {
    let db = match e {
        sqlx::Error::Database(d) => d.as_ref(),
        _ => return false,
    };
    let msg = db.message();
    match db.code().as_deref() {
        Some("1") => msg.contains("duplicate column name"),
        Some("42701") => true,
        _ => msg.contains("duplicate column name"),
    }
}

/// True when the error is Postgres SQLSTATE `42P01` (undefined_table) or
/// SQLite's "no such table" (result code 1), used by
/// [`Database::get_schema_version`] to treat a not-yet-migrated database as
/// version 0 instead of an error.
pub(crate) fn is_missing_table_error(e: &sqlx::Error) -> bool {
    let db = match e {
        sqlx::Error::Database(d) => d.as_ref(),
        _ => return false,
    };
    let msg = db.message();
    match db.code().as_deref() {
        Some("1") => msg.contains("no such table"),
        Some("42P01") => true,
        _ => false,
    }
}

/// v1 statements: the original idempotent schema bootstrap. CREATE TABLE
/// IF NOT EXISTS for every table, Postgres-specific `ALTER COLUMN TYPE`
/// repairs for legacy databases, and `CREATE INDEX IF NOT EXISTS`.
fn v1_statements_for(style: PlaceholderStyle) -> Vec<String> {
    let spend_logs_id = match style {
        // Postgres: standard identity column.
        PlaceholderStyle::Dollar => "id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY",
        // SQLite: autoincrement rowid.
        PlaceholderStyle::Question => "id INTEGER PRIMARY KEY AUTOINCREMENT",
    };
    // Postgres REAL is FLOAT4 (4-byte), which sqlx's Any backend will
    // not decode into Rust f64 (expects FLOAT8). Use DOUBLE PRECISION
    // there. SQLite REAL is 8-byte and maps fine to f64.
    let float_type = match style {
        PlaceholderStyle::Dollar => "DOUBLE PRECISION",
        PlaceholderStyle::Question => "REAL",
    };

    let mut stmts = vec![
        format!(
            "CREATE TABLE IF NOT EXISTS keys (\
 key_id TEXT PRIMARY KEY,\
 key_hash TEXT UNIQUE NOT NULL,\
 key_alias TEXT,\
 team_id TEXT,\
 user_id TEXT,\
 org_id TEXT,\
 project_id TEXT,\
 max_budget {float_type},\
 budget_duration TEXT,\
 rpm INTEGER,\
 tpm INTEGER,\
 created_at TEXT NOT NULL,\
 updated_at TEXT NOT NULL,\
 expires_at TEXT,\
 is_active INTEGER NOT NULL DEFAULT 1,\
 models TEXT)"
        ),
        format!(
            "CREATE TABLE IF NOT EXISTS teams (\
 team_id TEXT PRIMARY KEY,\
 team_name TEXT NOT NULL,\
 max_budget {float_type},\
 budget_duration TEXT,\
 rpm INTEGER,\
 tpm INTEGER,\
 created_at TEXT NOT NULL,\
 updated_at TEXT NOT NULL,\
 models TEXT,\
 blocked INTEGER NOT NULL DEFAULT 0)"
        ),
        format!(
            "CREATE TABLE IF NOT EXISTS users (\
 user_id TEXT PRIMARY KEY,\
 user_name TEXT NOT NULL,\
 max_budget {float_type},\
 budget_duration TEXT,\
 created_at TEXT NOT NULL,\
 updated_at TEXT NOT NULL,\
 models TEXT)"
        ),
        format!(
            "CREATE TABLE IF NOT EXISTS organizations (\
 org_id TEXT PRIMARY KEY,\
 org_name TEXT NOT NULL,\
 max_budget {float_type},\
 budget_duration TEXT,\
 rpm INTEGER,\
 tpm INTEGER,\
 created_at TEXT NOT NULL,\
 updated_at TEXT NOT NULL,\
 models TEXT,\
 metadata TEXT,\
 soft_budget {float_type},\
 max_parallel_requests INTEGER,\
 model_max_budget TEXT)"
        ),
        format!(
            "CREATE TABLE IF NOT EXISTS projects (\
 project_id TEXT PRIMARY KEY,\
 project_name TEXT NOT NULL,\
 org_id TEXT,\
 team_id TEXT,\
 max_budget {float_type},\
 budget_duration TEXT,\
 rpm INTEGER,\
 tpm INTEGER,\
 created_at TEXT NOT NULL,\
 updated_at TEXT NOT NULL)"
        ),
        format!(
            "CREATE TABLE IF NOT EXISTS spend_logs (\
 {spend_logs_id},\
 timestamp TEXT NOT NULL,\
 request_id TEXT,\
 model TEXT NOT NULL,\
 provider TEXT NOT NULL,\
 prompt_tokens INTEGER,\
 completion_tokens INTEGER,\
 total_tokens INTEGER,\
 cost {float_type},\
 key_id TEXT,\
 team_id TEXT,\
 user_id TEXT,\
 status TEXT,\
 latency_ms INTEGER)"
        ),
        "CREATE TABLE IF NOT EXISTS models (\
 model_id TEXT PRIMARY KEY,\
 model_name TEXT NOT NULL,\
 oxllm_params TEXT NOT NULL,\
 model_info TEXT,\
 created_at TEXT NOT NULL,\
 updated_at TEXT NOT NULL)"
            .to_string(),
    ];

    // Repair databases created by the buggy migration that used REAL
    // (FLOAT4) for float columns: `CREATE TABLE IF NOT EXISTS` is a
    // no-op for existing tables, so the fix above never reaches them.
    // `ALTER COLUMN ... TYPE DOUBLE PRECISION` is idempotent in Postgres
    // (re-setting the same type is allowed), so re-running the migration
    // upgrades the column type in place. SQLite's REAL is already an
    // 8-byte float, so no repair is needed there.
    if matches!(style, PlaceholderStyle::Dollar) {
        for (table, column) in [
            ("keys", "max_budget"),
            ("teams", "max_budget"),
            ("users", "max_budget"),
            ("spend_logs", "cost"),
        ] {
            stmts.push(format!(
                "ALTER TABLE {table} ALTER COLUMN {column} TYPE DOUBLE PRECISION"
            ));
        }
    }

    stmts.extend([
        "CREATE INDEX IF NOT EXISTS idx_spend_logs_timestamp ON spend_logs(timestamp)".to_string(),
        "CREATE INDEX IF NOT EXISTS idx_spend_logs_key_id ON spend_logs(key_id)".to_string(),
        "CREATE INDEX IF NOT EXISTS idx_spend_logs_model ON spend_logs(model)".to_string(),
        // H37: queries filtering by team_id (`SpendFilter::team_id`) were
        // doing full table scans because no index covered the column. The
        // `query_spend` / `query_spend_paginated` paths hit this for every
        // team-scoped dashboard request.
        "CREATE INDEX IF NOT EXISTS idx_spend_logs_team_id ON spend_logs(team_id)".to_string(),
        // H38: composite (key_id, timestamp) for the most common
        // dashboard shape: "give me this key's recent spend". Without a
        // composite, the planner picks one of the single-column indexes
        // and then filters the other column row-by-row.
        "CREATE INDEX IF NOT EXISTS idx_spend_logs_key_timestamp ON spend_logs(key_id, timestamp)"
            .to_string(),
    ]);
    stmts
}

/// v2 statements: additive `ALTER TABLE ... ADD COLUMN` statements that
/// post-date the original v1 schema. These are separated into their own
/// batch so that pre-existing databases (which already have the columns
/// via a prior idempotent migrate) can record v2 as applied without
/// touching the v1 schema. Tolerated as "duplicate column" /
/// "already exists" by [`Database::run_migration_batch`] when the column
/// is already present.
///
/// On a fresh database the v1 `CREATE TABLE` already includes these
/// columns, so the v2 ALTERs are no-ops there — but they are required
/// for upgrading an existing v1-only database.
fn v2_statements_for(style: PlaceholderStyle) -> Vec<String> {
    let mut stmts = Vec::new();

    // Add `models TEXT` to pre-existing databases that predate the column.
    // CREATE TABLE IF NOT EXISTS is a no-op for them, so the column must
    // be added separately. Postgres supports ADD COLUMN IF NOT EXISTS
    // (idempotent); SQLite's ADD COLUMN fails on a duplicate column, which
    // [`Database::run_migration_batch`] tolerates by matching the
    // resulting error message ("duplicate column name").
    for table in ["keys", "teams", "users"] {
        let stmt = match style {
            PlaceholderStyle::Dollar => {
                format!("ALTER TABLE {table} ADD COLUMN IF NOT EXISTS models TEXT")
            }
            PlaceholderStyle::Question => format!("ALTER TABLE {table} ADD COLUMN models TEXT"),
        };
        stmts.push(stmt);
    }

    // Add `org_id` and `project_id` to the `keys` table so a key can be
    // associated with an organization and/or project. Same idempotent ADD
    // COLUMN pattern as the `models` column above.
    for column in ["org_id", "project_id"] {
        let stmt = match style {
            PlaceholderStyle::Dollar => {
                format!("ALTER TABLE keys ADD COLUMN IF NOT EXISTS {column} TEXT")
            }
            PlaceholderStyle::Question => format!("ALTER TABLE keys ADD COLUMN {column} TEXT"),
        };
        stmts.push(stmt);
    }

    stmts
}

/// v3 statements: end-user (customer), budget plan, and tag tables used by
/// the management CRUD endpoints (`/customer/*`, `/budget/*`, `/tag/*`).
fn v3_statements_for(style: PlaceholderStyle) -> Vec<String> {
    let float_type = match style {
        PlaceholderStyle::Dollar => "DOUBLE PRECISION",
        PlaceholderStyle::Question => "REAL",
    };
    vec![
        format!(
            "CREATE TABLE IF NOT EXISTS customers (\
 end_user_id TEXT PRIMARY KEY,\
 alias TEXT,\
 email TEXT,\
 max_budget {float_type},\
 spend {float_type} NOT NULL DEFAULT 0,\
 budget_duration TEXT,\
 blocked INTEGER NOT NULL DEFAULT 0,\
 model_blocklist TEXT,\
 created_at TEXT NOT NULL,\
 updated_at TEXT NOT NULL)"
        ),
        format!(
            "CREATE TABLE IF NOT EXISTS budgets (\
 budget_id TEXT PRIMARY KEY,\
 max_budget {float_type},\
 soft_budget {float_type},\
 max_parallel_requests INTEGER,\
 tpm_limit INTEGER,\
 rpm_limit INTEGER,\
 budget_duration TEXT,\
 model_max_budget TEXT,\
 created_at TEXT NOT NULL,\
 updated_at TEXT NOT NULL)"
        ),
        format!(
            "CREATE TABLE IF NOT EXISTS tags (\
 name TEXT PRIMARY KEY,\
 description TEXT,\
 models TEXT,\
 model_info TEXT,\
 spend {float_type},\
 max_budget {float_type},\
 created_at TEXT NOT NULL,\
 updated_at TEXT NOT NULL)"
        ),
        "CREATE INDEX IF NOT EXISTS idx_customers_blocked ON customers(blocked)".to_string(),
    ]
}

/// v4 statements: additive `ALTER TABLE organizations ... ADD COLUMN`
/// statements for the organization-level fields that post-date v1/v3
/// (`models`, `metadata`, `soft_budget`, `max_parallel_requests`,
/// `model_max_budget`). Same idempotent ADD COLUMN pattern as v2: the
/// v1 `CREATE TABLE` for fresh databases already includes these columns
/// (so the ALTERs are no-ops there, tolerated as "duplicate column" /
/// "already exists"), while existing v3 databases get the columns added
/// in place.
fn v4_statements_for(style: PlaceholderStyle) -> Vec<String> {
    let float_type = match style {
        PlaceholderStyle::Dollar => "DOUBLE PRECISION",
        PlaceholderStyle::Question => "REAL",
    };

    let mut stmts = Vec::new();
    for (column, col_type) in [
        ("models", "TEXT"),
        ("metadata", "TEXT"),
        ("soft_budget", float_type),
        ("max_parallel_requests", "INTEGER"),
        ("model_max_budget", "TEXT"),
    ] {
        let stmt = match style {
            PlaceholderStyle::Dollar => {
                format!("ALTER TABLE organizations ADD COLUMN IF NOT EXISTS {column} {col_type}")
            }
            PlaceholderStyle::Question => {
                format!("ALTER TABLE organizations ADD COLUMN {column} {col_type}")
            }
        };
        stmts.push(stmt);
    }
    stmts
}

/// v5 statements: the `skills` table — versioned instruction files
/// (SKILL.md content) that extend a model's capabilities. Backend-agnostic:
/// only TEXT columns and an idempotent `CREATE TABLE IF NOT EXISTS` / index,
/// so a single statement list serves both Postgres and SQLite.
fn v5_statements_for() -> Vec<String> {
    vec![
        "CREATE TABLE IF NOT EXISTS skills (\
 id TEXT PRIMARY KEY,\
 name TEXT NOT NULL,\
 description TEXT,\
 content TEXT NOT NULL,\
 version TEXT NOT NULL DEFAULT '1.0',\
 model_pattern TEXT,\
 team_id TEXT,\
 created_at TEXT NOT NULL,\
 updated_at TEXT NOT NULL)"
            .to_string(),
        "CREATE INDEX IF NOT EXISTS idx_skills_name ON skills(name)".to_string(),
    ]
}

/// v6 statements: an index on `spend_logs.request_id`. The
/// `query_spend_by_request_id` lookup (the backing query for
/// `GET /request/logs/{request_id}`) does `WHERE request_id = ? LIMIT 1`;
/// without an index it full-scans an append-only telemetry table that the
/// retention pruner keeps large, turning a single request into a CPU/IO DoS
/// vector. The column is nullable but both Postgres and SQLite index NULLs,
/// so non-matching rows still aggregate cleanly.
fn v6_statements_for() -> Vec<String> {
    vec![
        "CREATE INDEX IF NOT EXISTS idx_spend_logs_request_id ON spend_logs(request_id)"
            .to_string(),
    ]
}

/// v7 statements: add a `blocked` boolean column to the `teams` table,
/// mirroring the reference `team.blocked` flag. Team block/unblock flips this
/// column instead of clobbering `is_active` on every member key, so an
/// `/team/unblock` no longer re-enables keys that were individually blocked
/// via `/key/block` for compromise/abuse/rotation. The column defaults to 0
/// (not blocked) so existing teams are unaffected. Same idempotent ADD COLUMN
/// pattern as v2/v4: fresh databases already include the column in the v1
/// CREATE TABLE, so the ALTER is a no-op there (tolerated as "duplicate
/// column" / "already exists"); existing v6 databases get the column added.
fn v7_statements_for(style: PlaceholderStyle) -> Vec<String> {
    let stmt = match style {
        PlaceholderStyle::Dollar => {
            "ALTER TABLE teams ADD COLUMN IF NOT EXISTS blocked INTEGER NOT NULL DEFAULT 0"
                .to_string()
        }
        PlaceholderStyle::Question => {
            "ALTER TABLE teams ADD COLUMN blocked INTEGER NOT NULL DEFAULT 0".to_string()
        }
    };
    vec![stmt]
}

/// v8 statements: attribute per-request cost to an end-user / customer via
/// `spend_logs.end_user_id` (OpenAI body `"user"` / legacy `end_user_id`).
/// Distinct from `user_id` (the virtual-key owner). Index supports
/// per-customer spend lookups without a full-table scan.
fn v8_statements_for(style: PlaceholderStyle) -> Vec<String> {
    let add_col = match style {
        PlaceholderStyle::Dollar => {
            "ALTER TABLE spend_logs ADD COLUMN IF NOT EXISTS end_user_id TEXT".to_string()
        }
        PlaceholderStyle::Question => {
            "ALTER TABLE spend_logs ADD COLUMN end_user_id TEXT".to_string()
        }
    };
    vec![
        add_col,
        "CREATE INDEX IF NOT EXISTS idx_spend_logs_end_user_id ON spend_logs(end_user_id)"
            .to_string(),
    ]
}

/// v9 statements: attribute per-request cost to a request tag via
/// `spend_logs.tag_name` (`x-oxllm-tag` header / request-body tags).
/// Index supports per-tag spend analytics without a full-table scan.
fn v9_statements_for(style: PlaceholderStyle) -> Vec<String> {
    let add_col = match style {
        PlaceholderStyle::Dollar => {
            "ALTER TABLE spend_logs ADD COLUMN IF NOT EXISTS tag_name TEXT".to_string()
        }
        PlaceholderStyle::Question => "ALTER TABLE spend_logs ADD COLUMN tag_name TEXT".to_string(),
    };
    vec![
        add_col,
        "CREATE INDEX IF NOT EXISTS idx_spend_logs_tag_name ON spend_logs(tag_name)".to_string(),
    ]
}

fn v10_statements_for(style: PlaceholderStyle) -> Vec<String> {
    // Multi-tag spend attribution: the full request tag list lands here as
    // a JSON array string; `tag_name` (v9) stays the primary/first tag for
    // single-tag compatibility. v11 indexes exact memberships separately.
    let add_col = match style {
        PlaceholderStyle::Dollar => {
            "ALTER TABLE spend_logs ADD COLUMN IF NOT EXISTS tags TEXT".to_string()
        }
        PlaceholderStyle::Question => "ALTER TABLE spend_logs ADD COLUMN tags TEXT".to_string(),
    };
    vec![add_col]
}

fn v11_statements_for(style: PlaceholderStyle) -> Vec<String> {
    let (add, tag_type) = match style {
        PlaceholderStyle::Dollar => ("ADD COLUMN IF NOT EXISTS", "TEXT COLLATE \"C\""),
        PlaceholderStyle::Question => ("ADD COLUMN", "TEXT COLLATE BINARY"),
    };
    let mut stmts = vec![
        format!("ALTER TABLE spend_logs {add} event_id TEXT"),
        // Existing nonempty PostgreSQL history is prepared concurrently first;
        // this inexpensive transactional create remains for fresh/empty DBs.
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_spend_logs_event_id ON spend_logs(event_id)"
            .to_string(),
        format!(
            "CREATE TABLE IF NOT EXISTS spend_log_tags (\
             spend_log_id BIGINT NOT NULL REFERENCES spend_logs(id) ON DELETE CASCADE,\
             tag_name {tag_type} NOT NULL,\
             PRIMARY KEY (spend_log_id, tag_name))"
        ),
        "CREATE INDEX IF NOT EXISTS idx_spend_log_tags_name ON spend_log_tags(tag_name, spend_log_id)"
            .to_string(),
    ];
    for (table, column) in [
        ("keys", "team_id"),
        ("keys", "user_id"),
        ("keys", "org_id"),
        ("keys", "project_id"),
        ("projects", "org_id"),
        ("projects", "team_id"),
    ] {
        stmts.push(format!(
            "CREATE INDEX IF NOT EXISTS idx_{table}_{column} ON {table}({column})"
        ));
    }
    stmts
}

/// Reserve aliases without rewriting or deleting legacy duplicate deployments.
/// Triggers make future inserts/renames atomic even for direct SQL writers;
/// unchanged-name edits remain available for every pre-existing duplicate.
fn v13_statements_for(style: PlaceholderStyle) -> Vec<String> {
    let collation = match style {
        PlaceholderStyle::Dollar => "\"C\"",
        PlaceholderStyle::Question => "BINARY",
    };
    let mut statements = vec![
        format!(
            "CREATE TABLE IF NOT EXISTS model_aliases (\
             model_name TEXT COLLATE {collation} NOT NULL \
             CONSTRAINT model_aliases_name PRIMARY KEY)"
        ),
        "CREATE INDEX IF NOT EXISTS idx_models_name ON models(model_name)".to_string(),
        format!(
            "INSERT INTO model_aliases (model_name) \
             SELECT DISTINCT model_name COLLATE {collation} FROM models WHERE TRUE \
             ON CONFLICT (model_name) DO NOTHING"
        ),
    ];
    match style {
        PlaceholderStyle::Dollar => {
            statements.extend([
                r#"CREATE OR REPLACE FUNCTION oxllm_claim_model_alias() RETURNS trigger
                   LANGUAGE plpgsql AS $$ BEGIN
                     IF TG_OP = 'INSERT' THEN
                       INSERT INTO model_aliases (model_name) VALUES (NEW.model_name);
                     ELSIF OLD.model_name COLLATE "C" IS DISTINCT FROM NEW.model_name COLLATE "C" THEN
                       INSERT INTO model_aliases (model_name) VALUES (NEW.model_name);
                     END IF;
                     RETURN NEW;
                   END; $$"#.to_string(),
                r#"CREATE OR REPLACE FUNCTION oxllm_release_model_alias() RETURNS trigger
                   LANGUAGE plpgsql AS $$ BEGIN
                     IF TG_OP = 'UPDATE' THEN
                       IF OLD.model_name COLLATE "C" IS NOT DISTINCT FROM NEW.model_name COLLATE "C" THEN
                         RETURN NEW;
                       END IF;
                     END IF;
                     -- Serialize release decisions for concurrent legacy-duplicate deletes.
                     PERFORM 1 FROM model_aliases WHERE model_name = OLD.model_name FOR UPDATE;
                     DELETE FROM model_aliases a WHERE a.model_name = OLD.model_name
                       AND NOT EXISTS (SELECT 1 FROM models m
                         WHERE m.model_name COLLATE "C" = OLD.model_name COLLATE "C");
                     IF TG_OP = 'DELETE' THEN RETURN OLD; END IF;
                     RETURN NEW;
                   END; $$"#.to_string(),
                "DROP TRIGGER IF EXISTS claim_model_alias_insert ON models".to_string(),
                "DROP TRIGGER IF EXISTS claim_model_alias_update ON models".to_string(),
                "DROP TRIGGER IF EXISTS release_model_alias_update ON models".to_string(),
                "DROP TRIGGER IF EXISTS release_model_alias_delete ON models".to_string(),
                "CREATE TRIGGER claim_model_alias_insert BEFORE INSERT ON models \
                 FOR EACH ROW EXECUTE FUNCTION oxllm_claim_model_alias()".to_string(),
                "CREATE TRIGGER claim_model_alias_update BEFORE UPDATE OF model_name ON models \
                 FOR EACH ROW EXECUTE FUNCTION oxllm_claim_model_alias()".to_string(),
                "CREATE TRIGGER release_model_alias_update AFTER UPDATE OF model_name ON models \
                 FOR EACH ROW EXECUTE FUNCTION oxllm_release_model_alias()".to_string(),
                "CREATE TRIGGER release_model_alias_delete AFTER DELETE ON models \
                 FOR EACH ROW EXECUTE FUNCTION oxllm_release_model_alias()".to_string(),
            ]);
        }
        PlaceholderStyle::Question => {
            statements.extend([
                "CREATE TRIGGER IF NOT EXISTS claim_model_alias_insert BEFORE INSERT ON models \
                 BEGIN INSERT INTO model_aliases (model_name) VALUES (NEW.model_name); END".to_string(),
                "CREATE TRIGGER IF NOT EXISTS claim_model_alias_update BEFORE UPDATE OF model_name ON models \
                 WHEN OLD.model_name COLLATE BINARY IS NOT NEW.model_name COLLATE BINARY \
                 BEGIN INSERT INTO model_aliases (model_name) VALUES (NEW.model_name); END".to_string(),
                "CREATE TRIGGER IF NOT EXISTS release_model_alias_update AFTER UPDATE OF model_name ON models \
                 WHEN OLD.model_name COLLATE BINARY IS NOT NEW.model_name COLLATE BINARY \
                 BEGIN DELETE FROM model_aliases WHERE model_name = OLD.model_name \
                   AND NOT EXISTS (SELECT 1 FROM models WHERE model_name COLLATE BINARY = OLD.model_name COLLATE BINARY); END".to_string(),
                "CREATE TRIGGER IF NOT EXISTS release_model_alias_delete AFTER DELETE ON models \
                 BEGIN DELETE FROM model_aliases WHERE model_name = OLD.model_name \
                   AND NOT EXISTS (SELECT 1 FROM models WHERE model_name COLLATE BINARY = OLD.model_name COLLATE BINARY); END".to_string(),
            ]);
        }
    }
    statements
}

/// Nullable JSON-TEXT key policy; older keys retain the deployment default.
fn v14_statements_for(style: PlaceholderStyle) -> Vec<String> {
    let add = match style {
        PlaceholderStyle::Dollar => "ADD COLUMN IF NOT EXISTS",
        PlaceholderStyle::Question => "ADD COLUMN",
    };
    vec![format!(
        "ALTER TABLE keys {add} allowed_cache_controls TEXT"
    )]
}

/// Build the full set of migration SQL statements for the given backend
/// style, concatenated across all version batches.
///
/// Runtime code paths no longer consume this flat list —
/// [`Database::migrate`] runs each batch via
/// [`v1_statements_for`] / [`v2_statements_for`] / [`v3_statements_for`] /
/// [`v4_statements_for`] / [`v5_statements_for`] gated by `_schema_version`.
/// This wrapper is preserved for the SQL-string regression tests, which
/// assert on the union of all batches, so it is gated to test builds.
#[cfg(test)]
pub(crate) fn migration_statements_for(style: PlaceholderStyle) -> Vec<String> {
    let mut stmts = vec![v0_schema_version_statement()];
    stmts.extend(v1_statements_for(style));
    stmts.extend(v2_statements_for(style));
    stmts.extend(v3_statements_for(style));
    stmts.extend(v4_statements_for(style));
    stmts.extend(v5_statements_for());
    stmts.extend(v6_statements_for());
    stmts.extend(v7_statements_for(style));
    stmts.extend(v8_statements_for(style));
    stmts.extend(v9_statements_for(style));
    stmts.extend(v10_statements_for(style));
    stmts.extend(v11_statements_for(style));
    stmts.extend(crate::spend_metadata::statements_for(style));
    stmts.extend(v13_statements_for(style));
    stmts.extend(v14_statements_for(style));
    stmts.extend(crate::spend_metadata::deployment_attribution_statements(
        style,
    ));
    stmts.extend(crate::spend_metadata::model_group_attribution_statements(
        style,
    ));
    stmts.extend(crate::spend_metadata::session_attribution_statements(style));
    stmts
}

/// v0 statement: the `_schema_version` bookkeeping table. Created
/// unconditionally on every startup so that `get_schema_version` has a
/// table to read from, even on a brand-new database. Test-gated because
/// the runtime path inlines the SQL inside
/// [`Database::ensure_schema_version_table`]; this helper exists only so
/// the SQL-string regression tests can assert on the full flattened
/// statement list.
#[cfg(test)]
fn v0_schema_version_statement() -> String {
    "CREATE TABLE IF NOT EXISTS _schema_version (\
     version INTEGER PRIMARY KEY,\
     applied_at TEXT NOT NULL)"
        .to_string()
}

#[cfg(test)]
mod sql_string_tests {
    //! Regression tests for Postgres compatibility bugs #4a / #4b.
    //!
    //! Pin both generated backend branches alongside the live migration
    //! contracts, including agreement with PostgreSQL online preparation.

    use super::migration_statements_for;
    use crate::PlaceholderStyle;
    use crate::pool::is_in_memory_sqlite;

    /// Session attribution must be in the complete migration statement union.
    #[test]
    fn session_attribution_migration_is_part_of_both_backend_contracts() {
        for style in [PlaceholderStyle::Dollar, PlaceholderStyle::Question] {
            let statements = migration_statements_for(style);
            assert!(statements.iter().any(|statement| {
                statement.starts_with("ALTER TABLE spend_logs ADD COLUMN")
                    && statement.ends_with("session_id TEXT")
            }));
            assert!(statements.iter().any(|statement| statement == "CREATE INDEX IF NOT EXISTS idx_spend_logs_session_timestamp ON spend_logs(session_id, timestamp)"));
        }
    }

    /// Bug #4a: Postgres REAL (FLOAT4) can't decode to f64 under sqlx's
    /// Any backend — the migration must emit DOUBLE PRECISION. SQLite must
    /// keep REAL (its 8-byte float type).
    #[test]
    fn float_column_type_branches_on_backend() {
        let pg = migration_statements_for(PlaceholderStyle::Dollar);
        let sqlite = migration_statements_for(PlaceholderStyle::Question);

        // Every CREATE TABLE that contains a float column should reference
        // the right type for its backend. Find the float-bearing tables by
        // their column name to stay robust against formatting changes.
        fn stmt_for_table<'a>(stmts: &'a [String], table: &str) -> &'a str {
            stmts
                .iter()
                .find(|s| s.starts_with("CREATE TABLE") && s.contains(&format!("{table} (")))
                .map(|s| s.as_str())
                .unwrap_or_else(|| panic!("CREATE TABLE for {table} not found"))
        }

        for table in ["keys", "teams", "users"] {
            let col = "max_budget";
            let pg_stmt = stmt_for_table(&pg, table);
            let sqlite_stmt = stmt_for_table(&sqlite, table);
            assert!(
                pg_stmt.contains(&format!("{col} DOUBLE PRECISION")),
                "pg {table} should use DOUBLE PRECISION for {col}: {pg_stmt}"
            );
            assert!(
                !pg_stmt.contains(&format!("{col} REAL")),
                "pg {table} must NOT use REAL for {col}: {pg_stmt}"
            );
            assert!(
                sqlite_stmt.contains(&format!("{col} REAL")),
                "sqlite {table} should use REAL for {col}: {sqlite_stmt}"
            );
            assert!(
                !sqlite_stmt.contains(&format!("{col} DOUBLE PRECISION")),
                "sqlite {table} must NOT use DOUBLE PRECISION for {col}: {sqlite_stmt}"
            );
        }

        // spend_logs uses `cost` as its float column.
        let pg_spend = stmt_for_table(&pg, "spend_logs");
        let sqlite_spend = stmt_for_table(&sqlite, "spend_logs");
        assert!(
            pg_spend.contains("cost DOUBLE PRECISION"),
            "pg spend_logs should use DOUBLE PRECISION for cost: {pg_spend}"
        );
        assert!(
            !pg_spend.contains("cost REAL"),
            "pg spend_logs must NOT use REAL for cost: {pg_spend}"
        );
        assert!(
            sqlite_spend.contains("cost REAL"),
            "sqlite spend_logs should use REAL for cost: {sqlite_spend}"
        );
        assert!(
            !sqlite_spend.contains("cost DOUBLE PRECISION"),
            "sqlite spend_logs must NOT use DOUBLE PRECISION for cost: {sqlite_spend}"
        );
    }

    /// Postgres additionally emits idempotent `ALTER TABLE ... TYPE
    /// DOUBLE PRECISION` repair statements for databases created by the
    /// buggy migration; SQLite needs none (its REAL is already 8-byte).
    /// Both branches still emit the same indices.
    #[test]
    fn migration_indices_match_between_backends() {
        let pg = migration_statements_for(PlaceholderStyle::Dollar);
        let sqlite = migration_statements_for(PlaceholderStyle::Question);
        // H36: the flat statement list is now ordered v0 → v1 → v2,
        // so the CREATE INDEX statements are no longer at the tail
        // (the v2 ADD COLUMN ALTERs are). Filter by prefix instead of
        // slicing the tail so the assertion keeps checking what it
        // cares about: both backends emit the same index statements.
        let pg_indices: Vec<&String> = pg
            .iter()
            .filter(|s| s.starts_with("CREATE INDEX"))
            .collect();
        let sl_indices: Vec<&String> = sqlite
            .iter()
            .filter(|s| s.starts_with("CREATE INDEX"))
            .collect();
        assert_eq!(
            pg_indices, sl_indices,
            "indices should be identical across backends"
        );
    }

    /// Regression: databases created by the buggy migration already have
    /// REAL columns, and `CREATE TABLE IF NOT EXISTS` is a no-op for
    /// them. The Postgres branch must include idempotent ALTER statements
    /// that repair the column type on re-migration; the SQLite branch
    /// must not emit any float-repair ALTERs (its REAL is already an
    /// 8-byte float, and SQLite has no `ALTER COLUMN TYPE`).
    #[test]
    fn postgres_migration_repairs_existing_real_columns() {
        let pg = migration_statements_for(PlaceholderStyle::Dollar);
        let sqlite = migration_statements_for(PlaceholderStyle::Question);

        for (table, column) in [
            ("keys", "max_budget"),
            ("teams", "max_budget"),
            ("users", "max_budget"),
            ("spend_logs", "cost"),
        ] {
            let expected =
                format!("ALTER TABLE {table} ALTER COLUMN {column} TYPE DOUBLE PRECISION");
            assert!(
                pg.iter().any(|s| s == &expected),
                "pg migration should repair {table}.{column}: missing `{expected}`"
            );
            assert!(
                !sqlite
                    .iter()
                    .any(|s| s.contains("ALTER COLUMN") && s.contains(column)),
                "sqlite migration must not emit float-repair ALTER COLUMN statements for {column}"
            );
        }

        // Float-type repair ALTERs live in the v1 batch and must run after
        // the v1 CREATE TABLE statements they target. Later batches (v3+)
        // may add more CREATE TABLE statements after v1, so only compare
        // against CREATE TABLEs that appear before the first float repair.
        let first_float_repair = pg
            .iter()
            .position(|s| s.contains("ALTER COLUMN") && s.contains("TYPE DOUBLE PRECISION"))
            .expect("float-repair ALTER TABLE statements present");
        let last_create_before_repair = pg[..first_float_repair]
            .iter()
            .rposition(|s| s.starts_with("CREATE TABLE"))
            .expect("CREATE TABLE statements present before float repairs");
        assert!(
            last_create_before_repair < first_float_repair,
            "float ALTER TABLE repairs must come after the CREATE TABLE statements they target"
        );
    }

    /// Both backends add the `models TEXT` column for ACLs: Postgres with
    /// idempotent `ADD COLUMN IF NOT EXISTS`, SQLite with plain
    /// `ADD COLUMN` (the duplicate-column error is tolerated by
    /// [`Database::migrate`]). CREATE TABLE statements on both backends
    /// also include the column so freshly-created databases don't need
    /// the ALTER.
    #[test]
    fn migration_adds_models_column_on_both_backends() {
        let pg = migration_statements_for(PlaceholderStyle::Dollar);
        let sqlite = migration_statements_for(PlaceholderStyle::Question);

        for table in ["keys", "teams", "users"] {
            let pg_expected = format!("ALTER TABLE {table} ADD COLUMN IF NOT EXISTS models");
            let sqlite_expected = format!("ALTER TABLE {table} ADD COLUMN models");
            assert!(
                pg.iter().any(|s| s.contains(&pg_expected)),
                "pg migration should add models column to {table}: missing `{pg_expected}`"
            );
            assert!(
                sqlite.iter().any(|s| s.contains(&sqlite_expected)),
                "sqlite migration should add models column to {table}: missing `{sqlite_expected}`"
            );
            // CREATE TABLE for fresh DBs also has the column.
            let create_pg = pg
                .iter()
                .find(|s| s.starts_with("CREATE TABLE") && s.contains(&format!("{table} (")))
                .unwrap_or_else(|| panic!("CREATE TABLE for {table} not found in pg"));
            let create_sqlite = sqlite
                .iter()
                .find(|s| s.starts_with("CREATE TABLE") && s.contains(&format!("{table} (")))
                .unwrap_or_else(|| panic!("CREATE TABLE for {table} not found in sqlite"));
            assert!(
                create_pg.contains("models TEXT"),
                "pg CREATE {table} missing models"
            );
            assert!(
                create_sqlite.contains("models TEXT"),
                "sqlite CREATE {table} missing models"
            );
        }
    }

    /// H37: a `team_id` index must exist on `spend_logs` so team-scoped
    /// dashboard queries don't full-scan.
    #[test]
    fn migration_creates_team_id_index() {
        let pg = migration_statements_for(PlaceholderStyle::Dollar);
        let sqlite = migration_statements_for(PlaceholderStyle::Question);
        let expected = "CREATE INDEX IF NOT EXISTS idx_spend_logs_team_id ON spend_logs(team_id)";
        assert!(
            pg.iter().any(|s| s == expected),
            "pg migration missing team_id index"
        );
        assert!(
            sqlite.iter().any(|s| s == expected),
            "sqlite migration missing team_id index"
        );
    }

    /// H38: composite (key_id, timestamp) index must exist for the common
    /// "this key's recent spend" dashboard shape.
    #[test]
    fn migration_creates_key_timestamp_composite_index() {
        let pg = migration_statements_for(PlaceholderStyle::Dollar);
        let sqlite = migration_statements_for(PlaceholderStyle::Question);
        let expected = "CREATE INDEX IF NOT EXISTS idx_spend_logs_key_timestamp ON spend_logs(key_id, timestamp)";
        assert!(
            pg.iter().any(|s| s == expected),
            "pg migration missing (key_id, timestamp) composite index"
        );
        assert!(
            sqlite.iter().any(|s| s == expected),
            "sqlite migration missing (key_id, timestamp) composite index"
        );
    }

    /// C57: a `request_id` index must exist so `query_spend_by_request_id`
    /// (`WHERE request_id = ? LIMIT 1`) doesn't full-scan the append-only
    /// telemetry table.
    #[test]
    fn migration_creates_request_id_index() {
        let pg = migration_statements_for(PlaceholderStyle::Dollar);
        let sqlite = migration_statements_for(PlaceholderStyle::Question);
        let expected =
            "CREATE INDEX IF NOT EXISTS idx_spend_logs_request_id ON spend_logs(request_id)";
        assert!(
            pg.iter().any(|s| s == expected),
            "pg migration missing request_id index"
        );
        assert!(
            sqlite.iter().any(|s| s == expected),
            "sqlite migration missing request_id index"
        );
    }

    /// v8: `spend_logs.end_user_id` + index for customer spend attribution.
    #[test]
    fn migration_adds_end_user_id_to_spend_logs() {
        let pg = migration_statements_for(PlaceholderStyle::Dollar);
        let sqlite = migration_statements_for(PlaceholderStyle::Question);
        for (label, stmts) in [("pg", &pg), ("sqlite", &sqlite)] {
            assert!(
                stmts.iter().any(|s| s.contains("spend_logs")
                    && s.contains("ADD COLUMN")
                    && s.contains("end_user_id")),
                "{label} migration should ADD COLUMN end_user_id to spend_logs: {stmts:?}"
            );
            assert!(
                stmts
                    .iter()
                    .any(|s| s.contains("idx_spend_logs_end_user_id")),
                "{label} migration should index spend_logs.end_user_id"
            );
        }
    }

    /// v9: `spend_logs.tag_name` + index for request-tag spend attribution.
    #[test]
    fn migration_adds_tag_name_to_spend_logs() {
        let pg = migration_statements_for(PlaceholderStyle::Dollar);
        let sqlite = migration_statements_for(PlaceholderStyle::Question);
        for (label, stmts) in [("pg", &pg), ("sqlite", &sqlite)] {
            assert!(
                stmts.iter().any(|s| s.contains("spend_logs")
                    && s.contains("ADD COLUMN")
                    && s.contains("tag_name")),
                "{label} migration should ADD COLUMN tag_name to spend_logs: {stmts:?}"
            );
            assert!(
                stmts.iter().any(|s| s.contains("idx_spend_logs_tag_name")),
                "{label} migration should index spend_logs.tag_name"
            );
        }
    }

    /// C27/H2: the `blocked` column must be added to the `teams` table on
    /// both backends (v7 ALTER for existing databases, included in v1 CREATE
    /// TABLE for fresh databases).
    #[test]
    fn migration_adds_blocked_column_to_teams() {
        let pg = migration_statements_for(PlaceholderStyle::Dollar);
        let sqlite = migration_statements_for(PlaceholderStyle::Question);

        // v7 ALTER for existing databases.
        assert!(
            pg.iter().any(|s| s.contains("ALTER TABLE teams")
                && s.contains("blocked")
                && s.contains("IF NOT EXISTS")),
            "pg migration should add blocked column to teams via v7 ALTER: {pg:?}"
        );
        assert!(
            sqlite
                .iter()
                .any(|s| s.contains("ALTER TABLE teams ADD COLUMN blocked")),
            "sqlite migration should add blocked column to teams via v7 ALTER"
        );

        // v1 CREATE TABLE for fresh databases also includes the column.
        for stmts in [&pg, &sqlite] {
            let create = stmts
                .iter()
                .find(|s| s.starts_with("CREATE TABLE") && s.contains("teams ("))
                .unwrap_or_else(|| panic!("CREATE TABLE teams not found"));
            assert!(
                create.contains("blocked INTEGER NOT NULL DEFAULT 0"),
                "fresh CREATE TABLE teams must include blocked column: {create}"
            );
        }
    }

    #[test]
    fn in_memory_sqlite_match_is_specific() {
        // Canonical in-memory DSN matches.
        assert!(is_in_memory_sqlite("sqlite::memory:"));
        assert!(is_in_memory_sqlite("sqlite::memory:?cache=shared"));
        // Common alias (`sqlite://path.db` style) must also match so it
        // gets the single-connection pool; sqlx itself rejects this
        // spelling, so `connect` rewrites it to `sqlite::memory:`.
        assert!(is_in_memory_sqlite("sqlite://:memory:"));
        assert!(is_in_memory_sqlite("sqlite://:memory:?cache=shared"));
        // Shared-cache DSN with `mode=memory` must NOT match — it can
        // safely use a larger pool because the backing file keeps rows
        // alive across connections.
        assert!(!is_in_memory_sqlite("sqlite:file.db?mode=memory"));
        assert!(!is_in_memory_sqlite("sqlite:data.db"));
        assert!(!is_in_memory_sqlite("postgres://localhost/db"));
    }
}
