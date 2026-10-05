//! Connection setup: driver installation, pool tuning, and the liveness
//! probe.

use std::borrow::Cow;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use sqlx::pool::PoolOptions;

use crate::{Database, PlaceholderStyle};

impl Database {
    /// Connect to the database at `url` (e.g. `sqlite::memory:`,
    /// `postgres://...`). Installs the default sqlx drivers.
    pub async fn connect(url: &str) -> Result<Arc<Self>> {
        sqlx::any::install_default_drivers();
        // sqlx's Any driver rejects `sqlite://:memory:` ("empty host");
        // rewrite the common alias to the canonical `sqlite::memory:` form
        // before pool setup and the connect call.
        let url = canonicalize_sqlite_memory_url(url);
        let url: &str = url.as_ref();

        let placeholder_style = if url.starts_with("postgres") || url.starts_with("postgresql") {
            PlaceholderStyle::Dollar
        } else {
            PlaceholderStyle::Question
        };

        let mut opts: PoolOptions<sqlx::Any> = PoolOptions::<sqlx::Any>::new();
        // In-memory SQLite databases live only as long as a connection is
        // held; pin the pool to a single connection so the schema and rows
        // persist across queries within the pool. Match the canonical
        // `sqlite::memory:` DSN specifically — a bare `"memory"` substring
        // would also match `sqlite:file.db?mode=memory` shared-cache DSNs,
        // which can safely use a larger pool.
        if is_in_memory_sqlite(url) {
            opts = opts.max_connections(1);
        } else if url.starts_with("postgres") || url.starts_with("postgresql") {
            // H14: configure a production-grade pool for Postgres. sqlx's
            // default `PoolOptions` leave `max_connections` at 10 and the
            // timeouts at their defaults; for the proxy that would serialise
            // concurrent requests and starve upstream providers under load.
            // These values match common production guidance (sqlx docs /
            // tokio-postgres pools): enough headroom for parallelism, bounded
            // so an idle proxy doesn't hold dozens of server-side connections.
            opts = opts
                .max_connections(20)
                .min_connections(2)
                .acquire_timeout(Duration::from_secs(5))
                .max_lifetime(Some(Duration::from_secs(1800)))
                .idle_timeout(Some(Duration::from_secs(600)));
        } else if is_file_sqlite(url) {
            // File-backed SQLite allows only a single writer at a time. sqlx's
            // default pool caps `max_connections` at 10 with a 30s acquire
            // timeout; 10 connections fighting over one write lock surface
            // `SQLITE_BUSY` / "database is locked" under concurrent proxy load.
            // Cap the pool to a few connections so writers queue at the pool
            // layer. The sqlx sqlite driver already applies a 5s per-connection
            // `PRAGMA busy_timeout`, so a queued writer retries instead of
            // erroring immediately. WAL mode (set below) additionally lets
            // readers proceed concurrently with the single writer.
            opts = opts
                .max_connections(4)
                .acquire_timeout(Duration::from_secs(5));
        }

        let pool = opts.connect(url).await?;

        // File-backed SQLite: switch the database into WAL mode. WAL is a
        // persistent property of the database file, so a one-shot PRAGMA here
        // covers every subsequent connection (including across restarts). We
        // don't set it for in-memory DBs (no file to persist into, and they
        // run single-connection anyway). Errors are non-fatal: a read-only or
        // locked DB may reject the switch, and we'd rather boot than crash.
        if is_file_sqlite(url) {
            let _ = sqlx::query("PRAGMA journal_mode=WAL").execute(&pool).await;
        }

        Ok(Arc::new(Self::new(pool, placeholder_style)))
    }

    /// Liveness probe: run `SELECT 1`. Returns `true` on success.
    pub async fn ping(&self) -> bool {
        match sqlx::query("SELECT 1").execute(self.pool()).await {
            Ok(_) => true,
            Err(e) => {
                tracing::warn!(
                    error = %crate::redact_url_credentials(&e.to_string()),
                    "database ping failed"
                );
                false
            }
        }
    }
}

/// True when `url` is an in-memory SQLite DSN: the canonical
/// `sqlite::memory:` form, or the common `sqlite://:memory:` alias.
/// A bare `"memory"` substring check would also match
/// `sqlite:file.db?mode=memory` shared-cache DSNs, which can safely
/// use a larger pool.
pub(crate) fn is_in_memory_sqlite(url: &str) -> bool {
    let rest = url
        .strip_prefix("sqlite://")
        .or_else(|| url.strip_prefix("sqlite:"));
    match rest {
        Some(rest) => rest == ":memory:" || rest.starts_with(":memory:?"),
        None => false,
    }
}

/// Rewrite `sqlite://:memory:` (optionally with a query string) to the
/// sqlx-canonical `sqlite::memory:` form. Other URLs pass through.
fn canonicalize_sqlite_memory_url(url: &str) -> Cow<'_, str> {
    if !is_in_memory_sqlite(url) || url.starts_with("sqlite::memory:") {
        return Cow::Borrowed(url);
    }
    match url.split_once('?') {
        Some((_, q)) => Cow::Owned(format!("sqlite::memory:?{q}")),
        None => Cow::Borrowed("sqlite::memory:"),
    }
}

/// True when `url` selects a file-backed SQLite database — any `sqlite:`
/// DSN that is not the canonical in-memory one. This includes
/// `sqlite:data.db`, `sqlite:///abs/path.db`, and shared-cache
/// `mode=memory` DSNs (which are backed by a file-name handle and can
/// use a larger pool than the single-connection in-memory case).
pub(crate) fn is_file_sqlite(url: &str) -> bool {
    (url.starts_with("sqlite:") || url.starts_with("sqlite://")) && !is_in_memory_sqlite(url)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonicalize_rewrites_alt_memory_dsn() {
        assert_eq!(
            canonicalize_sqlite_memory_url("sqlite://:memory:").as_ref(),
            "sqlite::memory:"
        );
        assert_eq!(
            canonicalize_sqlite_memory_url("sqlite://:memory:?cache=shared").as_ref(),
            "sqlite::memory:?cache=shared"
        );
        assert_eq!(
            canonicalize_sqlite_memory_url("sqlite::memory:").as_ref(),
            "sqlite::memory:"
        );
        assert_eq!(
            canonicalize_sqlite_memory_url("sqlite:data.db").as_ref(),
            "sqlite:data.db"
        );
    }
}
