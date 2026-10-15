//! Coordinated PostgreSQL preparation before atomic schema publication.
//!
//! Only short additive ALTERs run in a transaction. Existing spend-history
//! indexes are built concurrently after that commit, while older readers and
//! writers continue. A dedicated close-on-drop session owns coordination;
//! neither session locks nor session settings return to an ordinary borrower.

use std::time::{Duration, Instant};

use anyhow::Result;
use sqlx::{AnyConnection, Connection};

use crate::migrate::{OXLLM_MIGRATE_LOCK_KEY, is_missing_table_error};
use crate::{
    Database, PostgresPreparationTimedOut, PostgresSpendObjectNotReady, SpendEventIndexNotReady,
};

/// Public startup budget includes coordinator waits, concurrent builds and
/// schema publication. Explicit migration-test bounds use their supplied budget.
pub(crate) const PREPARATION_TIMEOUT: Duration = Duration::from_secs(30 * 60);
pub(crate) const PREPARATION_APPLICATION: &str = "oxllm online migration";

#[derive(Clone, Copy)]
struct SpendIndex {
    name: &'static str,
    columns: &'static [&'static str],
    unique: bool,
}

const EVENT_INDEX: SpendIndex = SpendIndex {
    name: "idx_spend_logs_event_id",
    columns: &["event_id"],
    unique: true,
};

// v1/v6/v8/v9 indexes can also scan old history. Prepare these alongside v11
// so an older upgrade does not move the same long scan to a different DDL batch.
const SPEND_INDEXES: &[SpendIndex] = &[
    EVENT_INDEX,
    SpendIndex {
        name: "idx_spend_logs_timestamp",
        columns: &["timestamp"],
        unique: false,
    },
    SpendIndex {
        name: "idx_spend_logs_key_id",
        columns: &["key_id"],
        unique: false,
    },
    SpendIndex {
        name: "idx_spend_logs_model",
        columns: &["model"],
        unique: false,
    },
    SpendIndex {
        name: "idx_spend_logs_team_id",
        columns: &["team_id"],
        unique: false,
    },
    SpendIndex {
        name: "idx_spend_logs_key_timestamp",
        columns: &["key_id", "timestamp"],
        unique: false,
    },
    SpendIndex {
        name: "idx_spend_logs_request_id",
        columns: &["request_id"],
        unique: false,
    },
    SpendIndex {
        name: "idx_spend_logs_session_timestamp",
        columns: &["session_id", "timestamp"],
        unique: false,
    },
    SpendIndex {
        name: "idx_spend_logs_end_user_id",
        columns: &["end_user_id"],
        unique: false,
    },
    SpendIndex {
        name: "idx_spend_logs_tag_name",
        columns: &["tag_name"],
        unique: false,
    },
];

const SPEND_COLUMNS: &[(&str, &str)] = &[
    ("session_id", "TEXT"),
    ("deployment_id", "TEXT"),
    ("requested_model", "TEXT"),
    ("model_group", "TEXT"),
    ("end_user_id", "TEXT"),
    ("tag_name", "TEXT"),
    ("tags", "TEXT"),
    ("event_id", "TEXT"),
    ("metadata_version", "INTEGER NOT NULL DEFAULT 0"),
];

#[derive(Clone, Copy, PartialEq, Eq)]
enum IndexState {
    Missing,
    Ready,
    NotReady,
}

fn index_not_ready(index: SpendIndex) -> anyhow::Error {
    if index.name == EVENT_INDEX.name {
        SpendEventIndexNotReady.into()
    } else {
        PostgresSpendObjectNotReady { object: index.name }.into()
    }
}

fn remaining(started: Instant, budget: Duration) -> Result<Duration> {
    budget
        .checked_sub(started.elapsed())
        .filter(|left| !left.is_zero())
        .ok_or_else(|| PostgresPreparationTimedOut { timeout: budget }.into())
}

fn milliseconds(timeout: Duration) -> String {
    format!("{}ms", timeout.as_millis().clamp(1, i32::MAX as u128))
}

async fn session_budget(
    conn: &mut AnyConnection,
    started: Instant,
    budget: Duration,
) -> Result<()> {
    let timeout = milliseconds(remaining(started, budget)?);
    sqlx::query(
        "SELECT set_config('statement_timeout', $1, false), set_config('lock_timeout', $1, false)",
    )
    .bind(timeout)
    .execute(&mut *conn)
    .await?;
    Ok(())
}

async fn index_state(conn: &mut AnyConnection, index: SpendIndex) -> Result<IndexState> {
    let query = format!(
        "SELECT CAST(CASE WHEN c.relkind = 'i' AND am.amname = 'btree' \
           AND i.indisvalid AND i.indisready AND i.indisunique = {} \
           AND i.indrelid = to_regclass('spend_logs') AND i.indnkeyatts = {} \
           AND i.indpred IS NULL AND i.indexprs IS NULL \
           AND i.indkey[0] = (SELECT a.attnum FROM pg_attribute a \
             WHERE a.attrelid = i.indrelid AND a.attname = $2 AND NOT a.attisdropped) \
           AND (i.indnkeyatts = 1 OR i.indkey[1] = (SELECT a.attnum FROM pg_attribute a \
             WHERE a.attrelid = i.indrelid AND a.attname = $3 AND NOT a.attisdropped)) \
           AND position('NULLS NOT DISTINCT' IN pg_get_indexdef(i.indexrelid)) = 0 \
           THEN 1 ELSE 0 END AS BIGINT) AS ready \
         FROM pg_class c LEFT JOIN pg_index i ON i.indexrelid = c.oid \
         LEFT JOIN pg_am am ON am.oid = c.relam WHERE c.oid = to_regclass($1)",
        if index.unique { "TRUE" } else { "FALSE" },
        index.columns.len()
    );
    let row = sqlx::query(&query)
        .bind(index.name)
        .bind(index.columns[0])
        .bind(index.columns.get(1).copied())
        .fetch_optional(&mut *conn)
        .await?;
    Ok(match row {
        None => IndexState::Missing,
        Some(row) if crate::rows::get_int(&row, "ready")? == 1 => IndexState::Ready,
        Some(_) => IndexState::NotReady,
    })
}

async fn column_ready(
    conn: &mut AnyConnection,
    column: &'static str,
    metadata: bool,
) -> Result<bool> {
    let query = if metadata {
        "SELECT CAST(CASE WHEN a.atttypid = 'integer'::regtype AND a.attnotnull \
           AND pg_get_expr(d.adbin, d.adrelid) = '0' THEN 1 ELSE 0 END AS BIGINT) AS ready \
         FROM pg_attribute a LEFT JOIN pg_attrdef d ON d.adrelid = a.attrelid AND d.adnum = a.attnum \
         WHERE a.attrelid = to_regclass('spend_logs') AND a.attname = $1 AND NOT a.attisdropped"
    } else {
        "SELECT CAST(CASE WHEN a.atttypid = 'text'::regtype AND NOT a.attnotnull \
           THEN 1 ELSE 0 END AS BIGINT) AS ready FROM pg_attribute a \
         WHERE a.attrelid = to_regclass('spend_logs') AND a.attname = $1 AND NOT a.attisdropped"
    };
    let row = sqlx::query(query)
        .bind(column)
        .fetch_optional(&mut *conn)
        .await?;
    match row {
        None => Ok(false),
        Some(row) if crate::rows::get_int(&row, "ready")? == 1 => Ok(true),
        Some(_) if column == "event_id" => Err(SpendEventIndexNotReady.into()),
        Some(_) => Err(PostgresSpendObjectNotReady { object: column }.into()),
    }
}

pub(crate) async fn ensure_event_index_ready(conn: &mut AnyConnection) -> Result<()> {
    if !column_ready(conn, "event_id", false).await?
        || index_state(conn, EVENT_INDEX).await? != IndexState::Ready
    {
        return Err(SpendEventIndexNotReady.into());
    }
    Ok(())
}

/// These statements are skipped only after every referenced column/index was
/// validated during preparation. Other tables and read-view DDL still execute.
pub(crate) fn prepared_spend_statement(statement: &str) -> bool {
    let compact = statement.split_whitespace().collect::<String>();
    SPEND_COLUMNS.iter().any(|(name, definition)| {
        compact
            == format!("ALTER TABLE spend_logs ADD COLUMN IF NOT EXISTS {name} {definition}")
                .split_whitespace()
                .collect::<String>()
    }) || compact
        == "ALTER TABLE spend_logs ALTER COLUMN cost TYPE DOUBLE PRECISION"
            .split_whitespace()
            .collect::<String>()
        || SPEND_INDEXES.iter().any(|index| {
            let unique = if index.unique { "UNIQUE " } else { "" };
            let expected = format!(
                "CREATE {unique}INDEX IF NOT EXISTS {} ON spend_logs({})",
                index.name,
                index.columns.join(", ")
            );
            compact == expected.split_whitespace().collect::<String>()
        })
}

impl Database {
    pub(crate) async fn migrate_postgres_online(
        &self,
        ddl_timeout: Duration,
        budget: Duration,
    ) -> Result<()> {
        // Bound wall time as well as each server statement. The connection is
        // marked before any persistent setting or session lock is acquired.
        let budget = budget.min(PREPARATION_TIMEOUT);
        let started = Instant::now();
        let operation = async {
            remaining(started, budget)?;
            let mut conn = self.pool().acquire().await?;
            conn.close_on_drop();
            session_budget(&mut conn, started, budget).await?;
            // A schema/database fingerprint distinguishes simultaneous builders
            // without exceeding PostgreSQL's 63-byte application_name limit.
            sqlx::query(
                "SELECT set_config('application_name', $1 || ':' || md5(current_database() || ':' || current_schema()), false)",
            )
                .bind(PREPARATION_APPLICATION)
                .execute(&mut *conn)
                .await?;
            // PostgreSQL 14+ detects a canceled owner's closed socket during a
            // long scan/lock wait, instead of retaining its session lock until
            // the statement finishes. Older servers remain server-timeout bound.
            let version: (String,) = sqlx::query_as("SELECT current_setting('server_version_num')")
                .fetch_one(&mut *conn)
                .await?;
            if version.0.parse::<u32>()? >= 140_000 {
                sqlx::query(
                    "SELECT set_config('client_connection_check_interval', '100ms', false)",
                )
                .execute(&mut *conn)
                .await?;
            }
            loop {
                session_budget(&mut conn, started, budget).await?;
                // A blocking SELECT pg_advisory_lock retains a transaction
                // snapshot: CREATE INDEX CONCURRENTLY can then wait on that
                // same peer, forming a lock/snapshot deadlock. Each try ends
                // its SQL statement before the bounded Rust-side backoff.
                let acquired: (i64,) = sqlx::query_as(
                    "SELECT CAST(CASE WHEN pg_try_advisory_lock($1) THEN 1 ELSE 0 END AS BIGINT)",
                )
                .bind(OXLLM_MIGRATE_LOCK_KEY)
                .fetch_one(&mut *conn)
                .await?;
                if acquired.0 == 1 {
                    break;
                }
                tokio::time::sleep(remaining(started, budget)?.min(Duration::from_millis(50)))
                    .await;
            }
            session_budget(&mut conn, started, budget).await?;
            // A waiting replica rechecks after acquiring the SAME coordination
            // lock; it cannot race an owner's in-progress concurrent index.
            let current = match self.get_schema_version_on(&mut *conn).await {
                Ok(version) => version,
                Err(error)
                    if error
                        .downcast_ref::<sqlx::Error>()
                        .is_some_and(is_missing_table_error) =>
                {
                    0
                }
                Err(error) => return Err(error),
            };
            if current < crate::SCHEMA_VERSION {
                let prepared_spend = self
                    .prepare_postgres_spend(&mut conn, ddl_timeout, started, budget)
                    .await?;
                session_budget(&mut conn, started, budget).await?;
                let mut tx = conn.begin().await?;
                sqlx::query("SELECT set_config('lock_timeout', $1, true)")
                    .bind(milliseconds(ddl_timeout))
                    .execute(&mut *tx)
                    .await?;
                self.run_all_migrations(&mut tx, prepared_spend).await?;
                self.ensure_spend_event_index_ready(&mut tx).await?;
                tx.commit().await?;
            } else {
                self.ensure_spend_event_index_ready(&mut conn).await?;
            }
            sqlx::query("SELECT pg_advisory_unlock($1)")
                .bind(OXLLM_MIGRATE_LOCK_KEY)
                .execute(&mut *conn)
                .await?;
            conn.close().await?;
            Ok(())
        };
        match tokio::time::timeout(budget, operation).await {
            Ok(result) => result,
            Err(_) => Err(PostgresPreparationTimedOut { timeout: budget }.into()),
        }
    }

    async fn prepare_postgres_spend(
        &self,
        conn: &mut AnyConnection,
        ddl_timeout: Duration,
        started: Instant,
        budget: Duration,
    ) -> Result<bool> {
        let table: Option<(i64,)> = sqlx::query_as(
            "SELECT CAST(CASE WHEN relkind = 'r' THEN 1 ELSE 0 END AS BIGINT) \
             FROM pg_class WHERE oid = to_regclass('spend_logs')",
        )
        .fetch_optional(&mut *conn)
        .await?;
        match table {
            None => return Ok(false),
            Some((1,)) => (),
            Some(_) => {
                return Err(PostgresSpendObjectNotReady {
                    object: "spend_logs",
                }
                .into());
            }
        }
        // Existing tables always use online preparation, including an empty
        // legacy table that an older writer could populate before final DDL.
        // This also keeps an already-valid index path entirely catalog-based.
        // Avoid silently rewriting a populated telemetry table when adopting a
        // malformed/unversioned FLOAT4 schema. Normal migrations use FLOAT8.
        let cost: Option<(i64,)> = sqlx::query_as(
            "SELECT CAST(CASE WHEN atttypid = 'double precision'::regtype THEN 1 ELSE 0 END AS BIGINT) \
             FROM pg_attribute WHERE attrelid = 'spend_logs'::regclass AND attname = 'cost' AND NOT attisdropped",
        )
        .fetch_optional(&mut *conn)
        .await?;
        if cost != Some((1,)) {
            let occupied: Option<(i64,)> =
                sqlx::query_as("SELECT CAST(1 AS BIGINT) FROM spend_logs LIMIT 1")
                    .fetch_optional(&mut *conn)
                    .await?;
            if occupied.is_none() {
                // Retain the original cheap repair for empty FLOAT4 schemas.
                return Ok(false);
            }
            return Err(PostgresSpendObjectNotReady { object: "cost" }.into());
        }
        for index in SPEND_INDEXES {
            if index_state(conn, *index).await? == IndexState::NotReady {
                return Err(index_not_ready(*index));
            }
        }
        self.prepare_spend_columns(conn, ddl_timeout, started, budget)
            .await?;
        for index in SPEND_INDEXES {
            match index_state(conn, *index).await? {
                IndexState::Ready => continue,
                IndexState::NotReady => return Err(index_not_ready(*index)),
                IndexState::Missing => (),
            }
            session_budget(conn, started, budget).await?;
            let unique = if index.unique { "UNIQUE " } else { "" };
            let statement = format!(
                "CREATE {unique}INDEX CONCURRENTLY {} ON spend_logs({})",
                index.name,
                index.columns.join(", ")
            );
            tracing::info!(index = index.name, "preparing spend index concurrently");
            if let Err(error) = sqlx::query(&statement).execute(&mut *conn).await {
                if error
                    .as_database_error()
                    .is_some_and(|error| error.code().as_deref() == Some("23505"))
                {
                    tracing::error!(index = index.name, error = %error, "spend index data is inconsistent");
                    return Err(index_not_ready(*index));
                }
                return Err(error.into());
            }
            if index_state(conn, *index).await? != IndexState::Ready {
                return Err(index_not_ready(*index));
            }
        }
        Ok(true)
    }

    pub(crate) async fn prepare_spend_columns(
        &self,
        conn: &mut AnyConnection,
        ddl_timeout: Duration,
        started: Instant,
        budget: Duration,
    ) -> Result<()> {
        let mut missing = Vec::new();
        for (column, definition) in SPEND_COLUMNS {
            if !column_ready(conn, column, *column == "metadata_version").await? {
                missing.push((*column, *definition));
            }
        }
        if missing.is_empty() {
            return Ok(());
        }
        session_budget(conn, started, budget).await?;
        let short = ddl_timeout.min(remaining(started, budget)?);
        let mut tx = conn.begin().await?;
        sqlx::query("SELECT set_config('lock_timeout', $1, true), set_config('statement_timeout', $1, true)")
            .bind(milliseconds(short)).execute(&mut *tx).await?;
        for (column, definition) in missing {
            // Identifiers/types come exclusively from SPEND_COLUMNS above.
            sqlx::query(&format!(
                "ALTER TABLE spend_logs ADD COLUMN IF NOT EXISTS {column} {definition}"
            ))
            .execute(&mut *tx)
            .await?;
        }
        tx.commit().await?;
        // No schema version is published here. Concurrent builds and the final
        // atomic DDL/read-view transaction must succeed before serving begins.
        Ok(())
    }
}

#[cfg(test)]
mod statement_tests {
    use super::*;
    use crate::PlaceholderStyle;

    #[test]
    fn preparation_only_skips_identical_known_spend_definitions() {
        assert!(prepared_spend_statement(
            "ALTER TABLE spend_logs\n ALTER COLUMN cost TYPE\t DOUBLE PRECISION"
        ));
        assert!(prepared_spend_statement(
            "ALTER TABLE spend_logs ADD COLUMN IF NOT EXISTS session_id TEXT"
        ));
        assert!(!prepared_spend_statement(
            "ALTER TABLE spend_logs ADD COLUMN IF NOT EXISTS future_column TEXT"
        ));
        assert!(!prepared_spend_statement(
            "ALTER TABLE spend_logs ADD COLUMN IF NOT EXISTS session_id BIGINT"
        ));
        assert!(!prepared_spend_statement(
            "CREATE INDEX IF NOT EXISTS idx_spend_logs_session_timestamp ON spend_logs(model, timestamp)"
        ));
        assert!(!prepared_spend_statement(
            "CREATE INDEX IF NOT EXISTS idx_spend_logs_session_timestamp ON other_table(session_id, timestamp)"
        ));
        assert!(!prepared_spend_statement(
            "CREATE INDEX IF NOT EXISTS idx_spend_logs_session_timestamp ON spend_logs(session_id, timestamp) WHERE session_id IS NOT NULL"
        ));
    }

    #[test]
    fn all_migrated_spend_columns_and_indexes_match_online_preparation() {
        // Only PostgreSQL skips online-prepared DDL. SQLite never skips it;
        // sql_string_tests separately pins equal index lists across backends.
        for statement in crate::migrate::migration_statements_for(PlaceholderStyle::Dollar) {
            if statement.starts_with("ALTER TABLE spend_logs ADD COLUMN IF NOT EXISTS ")
                || ((statement.starts_with("CREATE INDEX IF NOT EXISTS ")
                    || statement.starts_with("CREATE UNIQUE INDEX IF NOT EXISTS "))
                    && statement.contains(" ON spend_logs"))
            {
                assert!(
                    prepared_spend_statement(&statement),
                    "migration/preparation drift: {statement}"
                );
            }
        }
    }
}
