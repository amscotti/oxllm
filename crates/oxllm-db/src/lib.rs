//! Database persistence layer for oxllm.
//!
//! Behind the `postgres` and `sqlite` features. Uses `sqlx::Any` so a single
//! code path serves both Postgres (prod) and SQLite (zero-config local/dev).
//!
//! The implementation is split by domain: `pool` owns connection setup,
//! `migrate` owns schema versioning, `rows` maps `sqlx` rows into records,
//! and `repos` groups the per-entity query methods. All of them contribute
//! `impl Database` blocks in sibling modules, so the public API is unchanged:
//! every method is still reached as `db.create_key()`, `db.query_spend()`,
//! and so on.
#![forbid(unsafe_code)]

pub mod types;

pub use types::{
    BudgetRecord, CustomerRecord, DailySpend, KeyRecord, KeyTeamMutation, ModelConfigRecord,
    OrganizationRecord, ProjectRecord, ScopeDeletion, ScopeEntity, ScopeLinkError, SkillRecord,
    SpendByKey, SpendByKeyPage, SpendByModel, SpendByModelPage, SpendByOrg, SpendByOrgPage,
    SpendByTag, SpendByTagPage, SpendByTeam, SpendByTeamPage, SpendByUser, SpendByUserPage,
    SpendFilter, SpendLog, SpendLogEvent, SpendMetadataRepair, SpendPage, TagRecord, TeamRecord,
    UserRecord,
};

/// Latest schema version applied by `Database::migrate`.
pub const SCHEMA_VERSION: i64 = 16;

/// Maximum spend-log rows removed by one retention transaction.
pub const SPEND_PRUNE_BATCH_SIZE: i64 = 1_000;

/// Default and maximum group counts for spend-rollup pages.
pub const SPEND_ROLLUP_DEFAULT_LIMIT: i64 = 100;
pub const SPEND_ROLLUP_MAX_LIMIT: i64 = 1_000;

/// Maximum historical rows inspected in one metadata-repair transaction.
pub const SPEND_METADATA_REPAIR_BATCH_SIZE: i64 = 256;

#[cfg(any(feature = "postgres", feature = "sqlite"))]
mod binds;
#[cfg(any(feature = "postgres", feature = "sqlite"))]
mod migrate;
#[cfg(any(feature = "postgres", feature = "sqlite"))]
mod pool;
#[cfg(any(feature = "postgres", feature = "sqlite"))]
mod postgres_prepare;
#[cfg(any(feature = "postgres", feature = "sqlite"))]
mod repos;
#[cfg(any(feature = "postgres", feature = "sqlite"))]
mod rows;
#[cfg(any(feature = "postgres", feature = "sqlite"))]
mod scope_links;
#[cfg(any(feature = "postgres", feature = "sqlite"))]
mod spend_metadata;

#[cfg(any(feature = "postgres", feature = "sqlite"))]
use sqlx::Pool;

#[cfg(any(feature = "postgres", feature = "sqlite"))]
use crate::repos::skills::SkillsCache;

/// Which placeholder syntax the active backend expects.
#[cfg(any(feature = "postgres", feature = "sqlite"))]
#[derive(Clone, Copy)]
pub(crate) enum PlaceholderStyle {
    /// Postgres: `$1`, `$2`, ...
    Dollar,
    /// SQLite: `?` (positional).
    Question,
}

/// Build a placeholder token for the 1-based bind index `n`.
#[cfg(any(feature = "postgres", feature = "sqlite"))]
pub(crate) fn placeholder(style: PlaceholderStyle, n: usize) -> String {
    match style {
        PlaceholderStyle::Dollar => format!("${n}"),
        PlaceholderStyle::Question => "?".to_string(),
    }
}

/// A connection to the persistence layer.
#[cfg(any(feature = "postgres", feature = "sqlite"))]
#[derive(Clone)]
pub struct Database {
    pool: Pool<sqlx::Any>,
    placeholder_style: PlaceholderStyle,
    /// Short-TTL cache backing [`Database::list_skills_by_model`], the
    /// skills lookup on the per-chat-request hot path. Shared through an
    /// `Arc` so clones of this handle observe the same entries; invalidated
    /// by the skills write methods (`create`/`update`/`delete`) so
    /// single-process CRUD is always immediately visible. See
    /// `repos/skills.rs` for the staleness trade-off.
    skills_cache: std::sync::Arc<SkillsCache>,
}

#[cfg(any(feature = "postgres", feature = "sqlite"))]
impl Database {
    /// Wrap an established pool. Used by `pool`'s `connect`.
    pub(crate) fn new(pool: Pool<sqlx::Any>, placeholder_style: PlaceholderStyle) -> Self {
        Self {
            pool,
            placeholder_style,
            skills_cache: std::sync::Arc::new(SkillsCache::new()),
        }
    }

    /// The underlying pool (used by the batched spend writer).
    pub fn pool(&self) -> &Pool<sqlx::Any> {
        &self.pool
    }

    /// The placeholder style of the active backend.
    pub(crate) fn placeholder_style(&self) -> PlaceholderStyle {
        self.placeholder_style
    }

    /// Build a placeholder token for the 1-based bind index `n`.
    pub(crate) fn ph(&self, n: usize) -> String {
        placeholder(self.placeholder_style, n)
    }

    /// Conservative bind ceilings shared by set-based writes on both backends.
    pub(crate) fn param_limit(&self) -> usize {
        match self.placeholder_style {
            PlaceholderStyle::Dollar => 65_535,
            PlaceholderStyle::Question => 999,
        }
    }
}

/// A transaction-commit failure observed by the client.
///
/// Returned by [`Database::insert_spend_logs_batch`] when `tx.commit()`
/// itself fails. A commit-time error is *ambiguous*: the server may have
/// already committed the batch while the client observed a transport-level
/// failure (the classic two-generals window). Retrying the rows
/// with newly generated event IDs would then duplicate the batch. Callers that
/// retry must retain the original [`SpendLogEvent`] identities and use
/// [`Database::insert_spend_events`]. The spend writer conservatively avoids
/// per-row fallback on ambiguous commits to avoid N attempts during outages.
#[cfg(any(feature = "postgres", feature = "sqlite"))]
#[derive(Debug)]
pub struct CommitFailed(pub sqlx::Error);

#[cfg(any(feature = "postgres", feature = "sqlite"))]
impl std::fmt::Display for CommitFailed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "transaction commit failed: {}", self.0)
    }
}

#[cfg(any(feature = "postgres", feature = "sqlite"))]
impl std::error::Error for CommitFailed {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.0)
    }
}

/// An atomic model-alias claim conflicted with an existing deployment.
/// Management callers can downcast this error and map it to a bad request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelNameConflict {
    pub model_name: String,
}

impl std::fmt::Display for ModelNameConflict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "model_name '{}' already belongs to a deployment",
            self.model_name
        )
    }
}

impl std::error::Error for ModelNameConflict {}

/// Migration readiness cannot be published for an invalid event-id index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpendEventIndexNotReady;

impl std::fmt::Display for SpendEventIndexNotReady {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "spend event unique index is not ready; complete index preparation before migrating"
        )
    }
}

impl std::error::Error for SpendEventIndexNotReady {}

/// Coordinated online preparation exhausted its bounded startup budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PostgresPreparationTimedOut {
    pub timeout: std::time::Duration,
}

impl std::fmt::Display for PostgresPreparationTimedOut {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PostgreSQL online preparation exceeded {}ms",
            self.timeout.as_millis()
        )
    }
}

impl std::error::Error for PostgresPreparationTimedOut {}

/// A legacy spend schema cannot be adopted without changing an external object.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PostgresSpendObjectNotReady {
    pub object: &'static str,
}

impl std::fmt::Display for PostgresSpendObjectNotReady {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PostgreSQL spend object '{}' is not migration-ready",
            self.object
        )
    }
}

impl std::error::Error for PostgresSpendObjectNotReady {}

/// A complete, unpaged rollup exceeded its bounded result capacity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpendRollupLimitExceeded {
    pub dimension: &'static str,
    pub limit: i64,
}

impl std::fmt::Display for SpendRollupLimitExceeded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "more than {} spend {} groups; use query_spend_by_{}_paginated",
            self.limit, self.dimension, self.dimension
        )
    }
}

impl std::error::Error for SpendRollupLimitExceeded {}

/// ISO-8601 (UTC) timestamp string, suitable for the `*_at`/`timestamp`
/// columns. Kept here so callers don't depend on `jiff` directly.
///
/// Uses the fixed-precision form produced by [`normalize_iso`] so every
/// timestamp this crate writes (and compares) shares one lexicographic
/// ordering.
pub fn now_iso() -> String {
    jiff::Timestamp::now()
        .strftime("%Y-%m-%dT%H:%M:%S%.9fZ")
        .to_string()
}

/// Normalize an ISO-8601 timestamp string to a fixed-precision RFC 3339
/// form (`YYYY-MM-DDTHH:MM:SS.nnnnnnnnnZ`, always 9 fractional digits).
///
/// The `timestamp` (and `*_at`) columns are TEXT, and jiff's `Display`
/// impl *omits* fractional seconds when they are zero, so the same
/// instant can be rendered as `…59Z` or `…59.5Z`. Plain lexicographic
/// comparison — which every `timestamp >= / <=` filter and the retention
/// pruner relies on — then mis-orders boundary rows (`.` sorts before
/// `Z`, and `59.5Z` vs `59.500Z` compare on the fraction *width* rather
/// than the instant). Rendering both sides with a fixed 9-digit fraction
/// makes TEXT comparison agree with chronological order.
///
/// Values that do not parse as a timestamp (e.g. a bare `YYYY-MM-DD`
/// date filter, which still compares usefully as a prefix bound) are
/// returned unchanged so existing callers keep working.
pub fn normalize_iso(ts: &str) -> String {
    match ts.parse::<jiff::Timestamp>() {
        Ok(t) => t.strftime("%Y-%m-%dT%H:%M:%S%.9fZ").to_string(),
        Err(_) => ts.to_string(),
    }
}

/// Expand a valid bare end date through the final UTC nanosecond of that
/// day. Full timestamps and invalid input pass through unchanged. Shared by
/// DB queries and HTTP adapters so one layer cannot cut off submilliseconds.
pub fn inclusive_spend_end_date(value: &str) -> String {
    if value.len() == 10 && value.parse::<jiff::civil::Date>().is_ok() {
        format!("{value}T23:59:59.999999999Z")
    } else {
        value.to_string()
    }
}

/// ISO-8601 timestamp for `days` days ago. Used by `Database::daily_activity`
/// as the lower bound of the activity window.
pub fn compute_since_iso(days: i64) -> String {
    let now = jiff::Timestamp::now();
    // `Timestamp::checked_add` rejects spans with day/month units (they are
    // ambiguous without a timezone), so convert days to hours first. Use
    // checked arithmetic so an extreme input (e.g. `i64::MIN`, where the
    // product overflows and the subsequent unary negation of `i64::MIN`
    // would trap) degrades to a zero span — i.e. "now" — instead of
    // panicking.
    let hours = days.checked_mul(24).and_then(|h| h.checked_neg());
    let span = match hours {
        Some(h) => jiff::Span::new()
            .try_hours(h)
            .unwrap_or_else(|_| jiff::Span::new()),
        None => jiff::Span::new(),
    };
    let since = now.checked_add(span).unwrap_or(now);
    since.strftime("%Y-%m-%dT%H:%M:%S%.9fZ").to_string()
}

// Compile the same std-only redactor as core/server without adding a core
// dependency to this self-contained persistence crate (see workspace_layering).
#[cfg(any(feature = "postgres", feature = "sqlite"))]
#[path = "../../oxllm-core/src/redaction.rs"]
mod redaction;

/// Redact URL and keyword-DSN credentials throughout driver diagnostics,
/// including secret query parameters, malformed URLs, and multiple DSNs.
#[cfg(any(feature = "postgres", feature = "sqlite"))]
pub(crate) fn redact_url_credentials(s: &str) -> String {
    redaction::redact_connection_text(s)
}

#[cfg(all(test, any(feature = "postgres", feature = "sqlite")))]
mod log_redaction_tests {
    #[test]
    fn r20_db_diagnostic_helper_uses_shared_query_and_keyword_redaction() {
        let raw = "PostGreSQL://alice:first-secret@db/app?PaSS%77ord=query-secret; retry mysql://bob:second-secret@backup/app?api%5Fkey=key-secret; host=third password='quoted secret\\' with spaces' dbname=app";
        let safe = super::redact_url_credentials(raw);
        for secret in [
            "alice",
            "first-secret",
            "query-secret",
            "bob",
            "second-secret",
            "key-secret",
            "quoted secret",
            "with spaces",
        ] {
            assert!(
                !safe.contains(secret),
                "credential leaked in DB diagnostic: {secret}: {safe}"
            );
        }
        assert!(safe.contains("***@db/app"));
        assert!(safe.contains("***@backup/app"));
        assert!(safe.contains("host=third"));
        assert!(safe.contains("dbname=app"));
    }
}

#[cfg(all(test, any(feature = "postgres", feature = "sqlite")))]
mod db_regression_tests;
#[cfg(all(test, feature = "postgres"))]
mod postgres_prepare_tests;
#[cfg(all(test, any(feature = "postgres", feature = "sqlite")))]
mod release_tests;
#[cfg(all(test, any(feature = "postgres", feature = "sqlite")))]
mod scope_links_tests;
#[cfg(all(test, any(feature = "postgres", feature = "sqlite")))]
mod tests;
