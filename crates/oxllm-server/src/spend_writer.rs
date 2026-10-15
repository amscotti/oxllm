//! Batched spend writer.
//!
//! Request handlers never write to the DB on the hot path. Instead they queue
//! a [`SpendEntry`] onto an mpsc channel; a background task drains the channel
//! every `flush_interval_secs` and batch-inserts the accumulated logs.
//!
//! The `SpendEntry` type and the `SpendWriter` sender half are defined without
//! any dependency on `oxllm-db` so that the cost-tracking hook can always
//! reference them. The background drain task (which needs a `Database`) is only
//! available when the `db` feature is enabled.

#[cfg(feature = "db")]
use sqlx;

use tokio::sync::mpsc;

/// One accumulated request spend record, queued for batched DB insertion.
///
/// Mirrors `oxllm_db::SpendLog` but lives in the server crate so it has no
/// compile-time dependency on `sqlx`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SpendEntry {
    pub timestamp: String,
    pub request_id: Option<String>,
    #[serde(default)]
    pub session_id: Option<String>,
    pub model: String,
    pub provider: String,
    #[serde(default)]
    pub deployment_id: Option<String>,
    #[serde(default)]
    pub requested_model: Option<String>,
    #[serde(default)]
    pub model_group: Option<String>,
    pub prompt_tokens: Option<i64>,
    pub completion_tokens: Option<i64>,
    pub total_tokens: Option<i64>,
    pub cost: Option<f64>,
    pub key_id: Option<String>,
    pub team_id: Option<String>,
    pub user_id: Option<String>,
    /// OpenAI body end-user / customer id (distinct from `user_id`).
    pub end_user_id: Option<String>,
    /// Primary (first) normalized request tag across body and header carriers.
    pub request_tag: Option<String>,
    /// Every request tag, serialized as a JSON array string into the
    /// `spend_logs.tags` column. `None` when the request carried no tags.
    pub request_tags: Option<String>,
    pub status: Option<String>,
    pub latency_ms: Option<i64>,
}

/// Serialize the request tag list for the `spend_logs.tags` JSON column.
/// `None` when the request carried no tags (the column stays NULL).
pub fn tags_json(tags: &[String]) -> Option<String> {
    (!tags.is_empty()).then(|| serde_json::to_string(tags).unwrap_or_else(|_| "[]".to_string()))
}

/// Partial-session settlement runs after the response pump terminates. Give a
/// transiently full writer one bounded opportunity to accept its claimed row;
/// never manufacture a zero-cost failure row or enqueue the partial twice.
pub(crate) async fn queue_partial_spend(tx: &mpsc::Sender<SpendEntry>, entry: SpendEntry) {
    let entry = match tx.try_send(entry) {
        Ok(()) => return,
        Err(mpsc::error::TrySendError::Full(entry)) => entry,
        Err(mpsc::error::TrySendError::Closed(_)) => {
            tracing::warn!("spend writer closed; dropping partial spend entry");
            crate::metrics::record_spend_dropped("closed", 1);
            return;
        }
    };
    // Bound extra settlement latency and retain the bounded channel. Ordinary
    // SpendWriter::queue and request-path hooks remain immediate best-effort.
    match tokio::time::timeout(std::time::Duration::from_millis(250), tx.reserve()).await {
        Ok(Ok(permit)) => permit.send(entry),
        Ok(Err(_)) => {
            tracing::warn!("spend writer closed during partial settlement; dropping entry");
            crate::metrics::record_spend_dropped("closed", 1);
        }
        Err(_) => {
            tracing::warn!("spend writer remained full during partial settlement; dropping entry");
            crate::metrics::record_spend_dropped("full", 1);
        }
    }
}

/// A handle to the batched spend writer. Cloning is cheap (it shares the
/// channel sender).
#[derive(Clone)]
pub struct SpendWriter {
    tx: mpsc::Sender<SpendEntry>,
}

impl SpendWriter {
    /// Queue a spend entry for batched insertion. Never blocks: if the
    /// channel is full the entry is dropped (with a warning) so the request
    /// path is never stalled by DB writes. Drops are counted under
    /// `oxllm_spend_entries_dropped_total{reason=...}` so operators can
    /// alert on silent spend telemetry loss. The bounded channel +
    /// observable loss counter is the intended durability contract
    /// (the reference has no on-disk spill either — its in-memory queue is
    /// unbounded and silently lost on crash).
    pub async fn queue(&self, entry: SpendEntry) {
        match self.tx.try_send(entry) {
            Ok(()) => {}
            Err(mpsc::error::TrySendError::Full(_)) => {
                tracing::warn!("spend writer channel full; dropping spend entry");
                crate::metrics::record_spend_dropped("full", 1);
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                tracing::warn!("spend writer channel closed; dropping spend entry");
                crate::metrics::record_spend_dropped("closed", 1);
            }
        }
    }

    /// Clone out the channel sender so other components (e.g. the
    /// `CostTrackingHook`) can queue spend entries independently.
    pub fn sender(&self) -> mpsc::Sender<SpendEntry> {
        self.tx.clone()
    }
}

/// Default capacity of the spend-writer channel. Sized to absorb short bursts
/// without dropping under steady-state load; when exceeded entries are dropped
/// and counted under `oxllm_spend_entries_dropped_total{reason=...}`.
pub const DEFAULT_CHANNEL_CAPACITY: usize = 10_000;

/// Bound each write transaction even when producers continuously refill the
/// channel. The receiver returns to its shutdown/select loop between batches.
#[cfg(feature = "db")]
const MAX_FLUSH_BATCH_SIZE: usize = 1_000;

/// Three retries after the initial attempt, with at most 175 ms of added
/// backoff per batch. The final shutdown drain uses the same finite budget.
#[cfg(feature = "db")]
const BATCH_RETRY_BACKOFF: [std::time::Duration; 3] = [
    std::time::Duration::from_millis(25),
    std::time::Duration::from_millis(50),
    std::time::Duration::from_millis(100),
];

/// Create a [`SpendWriter`] and spawn its background drain task against `db`.
///
/// The task drains the channel every `flush_interval_secs` seconds and
/// batch-inserts all queued spend logs. It terminates when all senders are
/// dropped (i.e. when the `AppState` holding them is dropped on shutdown).
/// Uses [`DEFAULT_CHANNEL_CAPACITY`]; use [`spawn_with_capacity`] to override.
///
/// The [`SpendWriterHandle`] (which owns the task's shutdown trigger and
/// `JoinHandle`) is discarded, so the task cannot be explicitly flushed on
/// shutdown. Use [`spawn_with_shutdown`] when a bounded flush on exit is
/// needed (e.g. the server bootstrap).
#[cfg(feature = "db")]
pub fn spawn(db: std::sync::Arc<oxllm_db::Database>, flush_interval_secs: u64) -> SpendWriter {
    spawn_with_capacity(db, flush_interval_secs, DEFAULT_CHANNEL_CAPACITY)
}

/// Like [`spawn`] but with an explicit channel `capacity`, letting embedders
/// trade memory for drop-tolerance based on expected throughput. The returned
/// writer cannot be explicitly flushed on shutdown; see [`spawn_with_shutdown`].
#[cfg(feature = "db")]
pub fn spawn_with_capacity(
    db: std::sync::Arc<oxllm_db::Database>,
    flush_interval_secs: u64,
    capacity: usize,
) -> SpendWriter {
    spawn_with_shutdown(db, flush_interval_secs, capacity).0
}

/// Owned handle to a spawned spend-writer task. Calling [`shutdown`](Self::shutdown)
/// signals the background task to run a final `drain` of any buffered
/// entries and then exit, and awaits its completion. This is the deterministic
/// flush path used on graceful server shutdown so entries queued after the last
/// ticker tick are not lost when the process exits.
///
/// The signal is a `tokio::sync::Notify`, which is drop-safe: discarding this
/// handle (as `spawn` / `spawn_with_capacity` do) simply drops one `Arc`
/// reference and does NOT trigger a drain — the task keeps running until all
/// channel senders are dropped.
#[cfg(feature = "db")]
pub struct SpendWriterHandle {
    join: tokio::task::JoinHandle<()>,
    notify: std::sync::Arc<tokio::sync::Notify>,
}

#[cfg(feature = "db")]
impl SpendWriterHandle {
    /// Signal the writer to flush remaining buffered entries and exit, then
    /// await its completion. The returned future resolves once the final
    /// `drain` has run, so callers can rely on every queued entry being
    /// persisted (subject to the DB succeeding) before continuing.
    pub async fn shutdown(self) {
        self.notify.notify_one();
        let _ = self.join.await;
    }
}

/// Like [`spawn_with_capacity`] but also returns a [`SpendWriterHandle`] that
/// can trigger a bounded final flush on shutdown.
#[cfg(feature = "db")]
pub fn spawn_with_shutdown(
    db: std::sync::Arc<oxllm_db::Database>,
    flush_interval_secs: u64,
    capacity: usize,
) -> (SpendWriter, SpendWriterHandle) {
    use std::time::Duration;
    let cap = if capacity == 0 {
        DEFAULT_CHANNEL_CAPACITY
    } else {
        capacity
    };
    let (tx, mut rx) = mpsc::channel::<SpendEntry>(cap);
    let interval = if flush_interval_secs == 0 {
        Duration::from_secs(60)
    } else {
        Duration::from_secs(flush_interval_secs)
    };
    let notify = std::sync::Arc::new(tokio::sync::Notify::new());
    let task_notify = notify.clone();

    let join = tokio::spawn(async move {
        let mut ticker = tokio::time::interval(interval);
        ticker.tick().await;
        loop {
            tokio::select! {
            biased;
            _ = task_notify.notified() => {
            rx.close();
            while drain(&db, &mut rx).await {}
            tracing::info!("spend writer shutting down (shutdown signal)");
            return;
            }
            msg = rx.recv() => match msg {
            Some(entry) => {
            let mut batch = vec![entry];
            while batch.len() < MAX_FLUSH_BATCH_SIZE {
            let Ok(entry) = rx.try_recv() else { break };
            batch.push(entry);
            }
            if !batch.is_empty() {
            insert_batch(&db, batch).await;
            }
            }
            None => {
            drain(&db, &mut rx).await;
            tracing::info!("spend writer shutting down (channel closed)");
            return;
            }
            },
            _ = ticker.tick() => {
            drain(&db, &mut rx).await;
            }
            }
        }
    });

    (SpendWriter { tx }, SpendWriterHandle { join, notify })
}

/// Flush at most one bounded batch. Returns whether any entries were drained.
/// Shutdown closes the receiver first, then repeats until its finite backlog
/// is empty; regular ticks leave remaining entries to the recv/select loop.
#[cfg(feature = "db")]
async fn drain(
    db: &std::sync::Arc<oxllm_db::Database>,
    rx: &mut mpsc::Receiver<SpendEntry>,
) -> bool {
    let mut batch = Vec::new();
    while batch.len() < MAX_FLUSH_BATCH_SIZE {
        let Ok(entry) = rx.try_recv() else { break };
        batch.push(entry);
    }
    if batch.is_empty() {
        return false;
    }
    insert_batch(db, batch).await;
    true
}

/// Bound the indexed model label after settlement without merging long values.
#[cfg(feature = "db")]
fn indexed_model_identity(model: String) -> String {
    use sha2::{Digest, Sha256};

    const MAX_MODEL_BYTES: usize = 512;
    if model.len() <= MAX_MODEL_BYTES {
        return model;
    }
    // Provider model values and pre-routing failure selectors can be supplied
    // by an untrusted peer. Keep the indexed label bounded without merging
    // distinct long values. Pricing has already used the original identity.
    let suffix = format!("...#sha256:{:x}", Sha256::digest(model.as_bytes()));
    let mut end = MAX_MODEL_BYTES - suffix.len();
    while !model.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}{suffix}", &model[..end])
}

/// Give each queued entry a stable event identity before the first attempt.
/// The DB inserts logs, tag memberships and aggregated durable counters in
/// one transaction. Any fallback retains those identities: a replay can
/// neither duplicate a log nor increment its customer/tags a second time.
#[cfg(feature = "db")]
async fn insert_batch(db: &std::sync::Arc<oxllm_db::Database>, batch: Vec<SpendEntry>) {
    let n = batch.len();
    if n == 0 {
        return;
    }

    let events: Vec<oxllm_db::SpendLogEvent> = batch
        .into_iter()
        .map(|entry| oxllm_db::SpendLog {
            id: None,
            timestamp: entry.timestamp,
            request_id: entry.request_id,
            session_id: entry
                .session_id
                .as_deref()
                .and_then(crate::pipeline::normalize_session_id),
            model: indexed_model_identity(entry.model),
            provider: entry.provider,
            deployment_id: entry.deployment_id,
            requested_model: entry.requested_model,
            model_group: entry.model_group,
            prompt_tokens: entry.prompt_tokens,
            completion_tokens: entry.completion_tokens,
            total_tokens: entry.total_tokens,
            cost: entry.cost,
            key_id: entry.key_id,
            team_id: entry.team_id,
            user_id: entry.user_id,
            end_user_id: entry
                .end_user_id
                .as_deref()
                .and_then(crate::pipeline::normalize_end_user_id),
            tag_name: entry.request_tag,
            tags: entry.request_tags,
            status: entry.status,
            latency_ms: entry.latency_ms,
        })
        .map(oxllm_db::SpendLogEvent::new)
        .collect();

    match insert_events_with_retry(&events, |events| db.insert_spend_events(events)).await {
        Ok(inserted) => {
            tracing::debug!(
                count = n,
                inserted,
                "flushed spend logs and durable counters"
            );
        }
        Err(e) => {
            if should_retry_per_row(&e) {
                tracing::warn!(
                    error_class = ?classify_spend_insert_error(&e),
                    count = n,
                    "batch insert failed on transport error pre-commit; \
                     falling back to idempotent per-row inserts"
                );
                for event in &events {
                    if let Err(e) = db.insert_spend_events(std::slice::from_ref(event)).await {
                        tracing::warn!(
                            error_class = ?classify_spend_insert_error(&e),
                            "failed to insert spend log (per-row fallback)"
                        );
                        crate::metrics::record_spend_dropped("batch_error", 1);
                    }
                }
            } else {
                tracing::error!(
                    error_class = ?classify_spend_insert_error(&e),
                    count = n,
                    "batch insert failed; rollback retries exhausted or not eligible. \
                     No per-row retry for commit, pool, or database errors. \
                     The full batch is counted under reason=batch_error."
                );
                crate::metrics::record_spend_dropped("batch_error", n as u64);
            }
        }
    }
}

/// Only static error classes are logged: database messages can contain row
/// values, SQL or connection credentials.
#[cfg(feature = "db")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SpendInsertFailure {
    TransientDatabase(&'static str),
    PreCommitTransport,
    AmbiguousCommit,
    PoolUnavailable,
    PermanentDatabase,
    Other,
}

#[cfg(feature = "db")]
fn classify_spend_insert_error(err: &anyhow::Error) -> SpendInsertFailure {
    // Check the wrapper first, even if its source carries a retryable code.
    // The writer deliberately does not retry any commit-time failure.
    if err.downcast_ref::<oxllm_db::CommitFailed>().is_some() {
        return SpendInsertFailure::AmbiguousCommit;
    }
    match err.downcast_ref::<sqlx::Error>() {
        Some(sqlx::Error::Database(error)) => match error.code().as_deref() {
            Some("40001") => SpendInsertFailure::TransientDatabase("40001"),
            Some("40P01") => SpendInsertFailure::TransientDatabase("40P01"),
            Some("55P03") => SpendInsertFailure::TransientDatabase("55P03"),
            // SQLite reports numeric extended result codes, not SQLSTATEs.
            // BUSY includes RECOVERY, SNAPSHOT and TIMEOUT; LOCKED includes
            // SHAREDCACHE and VTAB. A fresh transaction can recover these.
            Some("5" | "261" | "517" | "773") => {
                SpendInsertFailure::TransientDatabase("sqlite_busy")
            }
            Some("6" | "262" | "518") => SpendInsertFailure::TransientDatabase("sqlite_locked"),
            _ => SpendInsertFailure::PermanentDatabase,
        },
        Some(sqlx::Error::Io(_)) => SpendInsertFailure::PreCommitTransport,
        Some(sqlx::Error::PoolClosed | sqlx::Error::PoolTimedOut) => {
            SpendInsertFailure::PoolUnavailable
        }
        _ => SpendInsertFailure::Other,
    }
}

/// Retry only known pre-commit database failures after the DB layer has
/// discarded the failed transaction. Every attempt borrows the same events;
/// neither UUID generation nor per-row fallback can reset this retry budget.
/// Execution time remains governed by the existing database/pool timeouts;
/// canceling a write here could introduce commit ambiguity.
#[cfg(feature = "db")]
async fn insert_events_with_retry<'a, F, Fut>(
    events: &'a [oxllm_db::SpendLogEvent],
    mut insert: F,
) -> anyhow::Result<usize>
where
    F: FnMut(&'a [oxllm_db::SpendLogEvent]) -> Fut,
    Fut: std::future::Future<Output = anyhow::Result<usize>>,
{
    let mut retries = 0;
    loop {
        let error = match insert(events).await {
            Ok(inserted) => return Ok(inserted),
            Err(error) => error,
        };
        let failure = classify_spend_insert_error(&error);
        if !matches!(failure, SpendInsertFailure::TransientDatabase(_)) {
            return Err(error);
        }
        let Some(backoff) = BATCH_RETRY_BACKOFF.get(retries) else {
            return Err(error);
        };
        tracing::warn!(
            error_class = ?failure,
            count = events.len(),
            attempt = retries + 1,
            max_attempts = BATCH_RETRY_BACKOFF.len() + 1,
            backoff_ms = backoff.as_millis() as u64,
            "retrying spend batch after transient pre-commit database failure"
        );
        retries += 1;
        tokio::time::sleep(*backoff).await;
    }
}

/// Whether the per-row fallback in [`insert_batch`] should run for `err`.
///
/// Returns `true` only when ALL of the following hold:
///
/// - The error is a transport-level failure (`sqlx::Error::Io`) raised
///   *before* `tx.commit()` — statements may have reached the DB, but the
///   transaction has not committed. Retries also retain event IDs.
///   [`oxllm_db::CommitFailed`] wraps commit-time errors and is treated as
///   non-retriable: a transport reset during the sub-second commit window
///   is the classic ambiguous-commit state where the server may have
///   committed while the client observed `Io`.
///
/// - The pool is still usable. `PoolClosed` / `PoolTimedOut` mean no
///   statement can succeed while the pool is down, so the per-row loop
///   would issue N doomed round-trips (each waiting on the acquire
///   timeout) and emit N identical warnings for zero recovered rows — a
///   self-inflicted latency and log-volume spike exactly when the system
///   is already unhealthy.
///
/// Other errors are not retried row-by-row: unavailable pools and persistent
/// constraint/schema failures would cause N futile attempts. Event IDs guard
/// against duplicate accounting independently of this retry policy.
#[cfg(feature = "db")]
fn should_retry_per_row(err: &anyhow::Error) -> bool {
    classify_spend_insert_error(err) == SpendInsertFailure::PreCommitTransport
}

#[cfg(all(test, feature = "db"))]
mod tests {
    use super::*;
    use sqlx::error::DatabaseError;
    use std::time::Duration;

    #[test]
    fn oversized_indexed_model_labels_are_utf8_safe_distinct_and_idempotent() {
        assert_eq!(indexed_model_identity("short-model".into()), "short-model");
        let long = "é".repeat(300);
        let label = indexed_model_identity(long.clone());
        assert!(label.len() <= 512);
        assert!(label.contains("...#sha256:"));
        assert_ne!(indexed_model_identity(format!("{long}x")), label);
        assert_eq!(indexed_model_identity(label.clone()), label);
    }

    #[tokio::test]
    async fn writer_defensively_bounds_session_and_rejects_invalid_end_user_attribution() {
        let db = oxllm_db::Database::connect("sqlite::memory:")
            .await
            .unwrap();
        db.migrate().await.unwrap();
        let entry: SpendEntry = serde_json::from_value(serde_json::json!({
            "timestamp":"2026-10-15T00:00:00Z", "request_id":"raw-attribution", "model":"short-model", "provider":"test",
            "session_id":"é".repeat(300), "end_user_id":"x".repeat(300), "cost":0.25
        })).unwrap();
        let (writer, handle) = spawn_with_shutdown(db.clone(), 3600, 8);
        writer.queue(entry).await;
        tokio::time::timeout(Duration::from_secs(5), handle.shutdown())
            .await
            .unwrap();
        let row = db
            .query_spend_by_request_id("raw-attribution")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(row.model, "short-model");
        assert_eq!(row.cost, Some(0.25));
        assert_eq!(row.session_id, Some("é".repeat(128)));
        assert!(row.end_user_id.is_none());
    }

    #[derive(Debug)]
    struct ReportedDatabaseError(Option<&'static str>);

    impl std::fmt::Display for ReportedDatabaseError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str(self.message())
        }
    }

    impl std::error::Error for ReportedDatabaseError {}

    impl sqlx::error::DatabaseError for ReportedDatabaseError {
        fn message(&self) -> &str {
            "row contains private-payload and postgres://user:private-password@localhost/db"
        }

        fn code(&self) -> Option<std::borrow::Cow<'_, str>> {
            self.0.map(std::borrow::Cow::Borrowed)
        }

        fn as_error(&self) -> &(dyn std::error::Error + Send + Sync + 'static) {
            self
        }

        fn as_error_mut(&mut self) -> &mut (dyn std::error::Error + Send + Sync + 'static) {
            self
        }

        fn into_error(self: Box<Self>) -> Box<dyn std::error::Error + Send + Sync + 'static> {
            self
        }

        fn kind(&self) -> sqlx::error::ErrorKind {
            sqlx::error::ErrorKind::Other
        }
    }

    fn reported_database_error(code: Option<&'static str>) -> anyhow::Error {
        sqlx::Error::Database(Box::new(ReportedDatabaseError(code))).into()
    }

    fn retry_test_events() -> Vec<oxllm_db::SpendLogEvent> {
        // Matching request IDs are distinct events, not an idempotency key.
        (0..2)
            .map(|_| {
                oxllm_db::SpendLogEvent::new(
                    serde_json::from_value(serde_json::json!({
                        "timestamp": "2026-01-01T00:00:00Z",
                        "request_id": "same-request-id",
                        "model": "test-model",
                        "provider": "test-provider",
                        "cost": 0.25,
                        "end_user_id": "retry-customer",
                        "tags": "[\"retry-tag\"]",
                    }))
                    .unwrap(),
                )
            })
            .collect()
    }

    #[test]
    fn batch_retry_classifies_only_known_rollback_codes() {
        for (code, class) in [
            ("40001", "40001"),
            ("40P01", "40P01"),
            ("55P03", "55P03"),
            ("5", "sqlite_busy"),
            ("261", "sqlite_busy"),
            ("517", "sqlite_busy"),
            ("773", "sqlite_busy"),
            ("6", "sqlite_locked"),
            ("262", "sqlite_locked"),
            ("518", "sqlite_locked"),
        ] {
            let error = reported_database_error(Some(code)).context("insert failed");
            assert_eq!(
                classify_spend_insert_error(&error),
                SpendInsertFailure::TransientDatabase(class),
                "code {code} must retry the whole rolled-back transaction"
            );
            assert!(
                !should_retry_per_row(&error),
                "code {code} is not transport I/O"
            );
        }
        for code in [
            None,
            Some("23505"),
            Some("23503"),
            Some("23514"),
            Some("42601"),
            Some("42P01"),
            Some("57014"),
            Some("08006"),
            Some("8"),
            Some("10"),
            Some("13"),
            Some("19"),
            Some("1811"),
            Some("40001extra"),
            Some("55p03"),
        ] {
            assert_eq!(
                classify_spend_insert_error(&reported_database_error(code)),
                SpendInsertFailure::PermanentDatabase,
                "unlisted code {code:?} must not retry"
            );
        }
    }

    #[test]
    fn batch_retry_distinguishes_transport_pool_and_ambiguous_commit() {
        let error = anyhow::Error::from(sqlx::Error::Io(std::io::Error::other("disconnected")));
        assert_eq!(
            classify_spend_insert_error(&error),
            SpendInsertFailure::PreCommitTransport
        );
        assert!(should_retry_per_row(&error));
        for error in [sqlx::Error::PoolClosed, sqlx::Error::PoolTimedOut] {
            assert_eq!(
                classify_spend_insert_error(&error.into()),
                SpendInsertFailure::PoolUnavailable
            );
        }
        // Neither an outer context nor a retryable source can bypass the
        // deliberate commit-ambiguity boundary.
        for source in [
            sqlx::Error::Io(std::io::Error::other("reset at commit")),
            sqlx::Error::Database(Box::new(ReportedDatabaseError(Some("40001")))),
            sqlx::Error::Database(Box::new(ReportedDatabaseError(Some("5")))),
        ] {
            let error =
                anyhow::Error::from(oxllm_db::CommitFailed(source)).context("commit failed");
            assert_eq!(
                classify_spend_insert_error(&error),
                SpendInsertFailure::AmbiguousCommit
            );
            assert!(!should_retry_per_row(&error));
        }
    }

    #[tokio::test(start_paused = true)]
    async fn batch_retry_recovers_with_same_event_ids_and_capped_backoff() {
        let events = retry_test_events();
        assert_ne!(events[0].event_id, events[1].event_id);
        let mut attempts = Vec::new();
        let mut failures = ["40001", "40P01", "55P03"].into_iter();
        let start = tokio::time::Instant::now();
        let inserted = insert_events_with_retry(&events, |batch| {
            assert!(std::ptr::eq(batch.as_ptr(), events.as_ptr()));
            attempts.push(
                batch
                    .iter()
                    .map(|event| event.event_id.clone())
                    .collect::<Vec<_>>(),
            );
            std::future::ready(match failures.next() {
                Some(code) => Err(reported_database_error(Some(code))),
                None => Ok(batch.len()),
            })
        })
        .await
        .unwrap();
        assert_eq!(inserted, 2);
        assert_eq!(attempts.len(), 4);
        assert!(attempts.iter().all(|ids| ids == &attempts[0]));
        assert_eq!(start.elapsed(), Duration::from_millis(175));
    }

    #[tokio::test(start_paused = true)]
    async fn batch_retry_exhaustion_is_bounded_without_resetting_the_budget() {
        let events = retry_test_events();
        let mut attempts = 0;
        let start = tokio::time::Instant::now();
        let error = insert_events_with_retry(&events, |_| {
            attempts += 1;
            // Changing transient classes must not restart the retry budget.
            let code = if attempts % 2 == 0 { "40001" } else { "55P03" };
            std::future::ready(Err(reported_database_error(Some(code))))
        })
        .await
        .unwrap_err();
        assert_eq!(attempts, 4);
        assert_eq!(start.elapsed(), Duration::from_millis(175));
        assert_eq!(
            classify_spend_insert_error(&error),
            SpendInsertFailure::TransientDatabase("40001")
        );
    }

    #[tokio::test(start_paused = true)]
    async fn batch_retry_stops_immediately_for_non_retryable_failures() {
        let events = retry_test_events();
        for error in [
            reported_database_error(Some("23514")),
            sqlx::Error::PoolClosed.into(),
            sqlx::Error::PoolTimedOut.into(),
            sqlx::Error::Io(std::io::Error::other("pre-commit transport failure")).into(),
            anyhow::Error::from(oxllm_db::CommitFailed(sqlx::Error::Database(Box::new(
                ReportedDatabaseError(Some("40P01")),
            )))),
        ] {
            let expected = classify_spend_insert_error(&error);
            let mut error = Some(error);
            let mut attempts = 0;
            let start = tokio::time::Instant::now();
            let outcome = insert_events_with_retry(&events, |_| {
                attempts += 1;
                std::future::ready(match error.take() {
                    Some(error) => Err(error),
                    None => Ok(2),
                })
            })
            .await;
            assert_eq!(classify_spend_insert_error(&outcome.unwrap_err()), expected);
            assert_eq!(attempts, 1);
            assert_eq!(start.elapsed(), Duration::ZERO);
        }
    }

    #[tokio::test(start_paused = true)]
    async fn batch_retry_stops_if_a_later_attempt_becomes_permanent_or_ambiguous() {
        let events = retry_test_events();
        for terminal in [
            reported_database_error(Some("23505")),
            anyhow::Error::from(oxllm_db::CommitFailed(sqlx::Error::Io(
                std::io::Error::other("ambiguous commit"),
            ))),
        ] {
            let expected = classify_spend_insert_error(&terminal);
            let mut failures = [reported_database_error(Some("40001")), terminal].into_iter();
            let mut attempts = 0;
            let start = tokio::time::Instant::now();
            let error = insert_events_with_retry(&events, |_| {
                attempts += 1;
                std::future::ready(Err(failures.next().expect("no third attempt")))
            })
            .await
            .unwrap_err();
            assert_eq!(classify_spend_insert_error(&error), expected);
            assert_eq!(attempts, 2);
            assert_eq!(start.elapsed(), Duration::from_millis(25));
        }
    }

    #[test]
    #[serial_test::serial]
    fn batch_retry_logs_safe_classes_without_database_payloads() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let events = retry_test_events();
        let mut attempts = 0;
        let (outcome, log) = crate::security::capture_test_logs(|| {
            runtime.block_on(insert_events_with_retry(&events, |_| {
                attempts += 1;
                std::future::ready(Err(reported_database_error(Some("55P03"))))
            }))
        });
        assert!(outcome.is_err());
        // The retry budget and backoff are pinned independently. Inspect the
        // captured output for static error classes rather than requiring an
        // exact log-line count from a concurrency-sensitive test capture.
        assert_eq!(attempts, 4);
        assert!(log.contains("retrying spend batch"));
        assert!(log.contains("55P03"));
        assert!(!log.contains("private-payload"));
        assert!(!log.contains("private-password"));
    }

    async fn sqlite_contention_fixture() -> (tempfile::TempDir, std::sync::Arc<oxllm_db::Database>)
    {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("spend.db");
        let url = format!("sqlite://{}?mode=rwc&cache=private", path.display());
        let db = oxllm_db::Database::connect(&url).await.unwrap();
        db.migrate().await.unwrap();
        let now = oxllm_db::now_iso();
        db.create_customer(&oxllm_db::CustomerRecord {
            end_user_id: "retry-customer".into(),
            alias: None,
            email: None,
            max_budget: None,
            spend: 0.0,
            budget_duration: None,
            blocked: false,
            model_blocklist: None,
            created_at: now.clone(),
            updated_at: now.clone(),
        })
        .await
        .unwrap();
        db.create_tag(&oxllm_db::TagRecord {
            name: "retry-tag".into(),
            description: None,
            models: None,
            model_info: None,
            spend: Some(0.0),
            max_budget: None,
            created_at: now.clone(),
            updated_at: now,
        })
        .await
        .unwrap();
        // Configure every connection in the four-connection file pool so
        // SQLite surfaces contention immediately instead of hiding retries
        // inside its default five-second busy handler.
        let mut connections = Vec::new();
        for _ in 0..4 {
            let mut connection = db.pool().acquire().await.unwrap();
            sqlx::query("PRAGMA busy_timeout = 0")
                .execute(&mut *connection)
                .await
                .unwrap();
            connections.push(connection);
        }
        drop(connections);
        (directory, db)
    }

    #[tokio::test]
    async fn sqlite_busy_batch_retry_persists_logs_and_counters_once() {
        let (_directory, db) = sqlite_contention_fixture().await;
        let events = retry_test_events();
        let mut holder = db.pool().begin().await.unwrap();
        sqlx::query("UPDATE customers SET spend = spend WHERE end_user_id = 'retry-customer'")
            .execute(&mut *holder)
            .await
            .unwrap();
        let mut holder = Some(holder);
        let mut attempts = 0;
        let inserted = insert_events_with_retry(&events, |batch| {
            attempts += 1;
            let holder = holder.take();
            let db = &db;
            async move {
                let result = db.insert_spend_events(batch).await;
                if let Some(holder) = holder {
                    let class = result.as_ref().err().map(classify_spend_insert_error);
                    holder.rollback().await.unwrap();
                    assert_eq!(
                        class,
                        Some(SpendInsertFailure::TransientDatabase("sqlite_busy")),
                        "the first write must fail with a real contention error"
                    );
                }
                result
            }
        })
        .await
        .unwrap();
        assert_eq!(attempts, 2);
        assert_eq!(inserted, 2);
        let mut stored: Vec<(String,)> = sqlx::query_as("SELECT event_id FROM spend_logs")
            .fetch_all(db.pool())
            .await
            .unwrap();
        stored.sort();
        let mut expected: Vec<_> = events
            .iter()
            .map(|event| (event.event_id.clone(),))
            .collect();
        expected.sort();
        assert_eq!(stored, expected, "durable UUIDs must survive the retry");
        // Replay the exact events: logs, tag memberships and counters must
        // stay single-charged, including calls with matching request IDs.
        assert_eq!(
            insert_events_with_retry(&events, |events| db.insert_spend_events(events))
                .await
                .unwrap(),
            0
        );
        assert_eq!(db.count_spend(None, None).await.unwrap(), 2);
        let memberships: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM spend_log_tags")
            .fetch_one(db.pool())
            .await
            .unwrap();
        assert_eq!(memberships.0, 2);
        assert_eq!(
            db.get_customer("retry-customer")
                .await
                .unwrap()
                .unwrap()
                .spend,
            0.5
        );
        assert_eq!(
            db.get_tag("retry-tag").await.unwrap().unwrap().spend,
            Some(0.5)
        );
        db.pool().close().await;
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn sqlite_busy_batch_exhaustion_counts_each_drop_once() {
        let _ = crate::metrics::install();
        let (_directory, db) = sqlite_contention_fixture().await;
        let mut holder = db.pool().begin().await.unwrap();
        sqlx::query("UPDATE customers SET spend = spend WHERE end_user_id = 'retry-customer'")
            .execute(&mut *holder)
            .await
            .unwrap();
        let entry: SpendEntry = serde_json::from_value(serde_json::json!({
            "timestamp": "2026-01-01T00:00:00Z",
            "model": "test-model",
            "provider": "test-provider",
            "cost": 0.25,
            "end_user_id": "retry-customer",
            "request_tags": "[\"retry-tag\"]",
        }))
        .unwrap();
        let before = dropped_count("batch_error");
        let outcome = tokio::time::timeout(
            Duration::from_secs(2),
            insert_batch(&db, vec![entry.clone(), entry]),
        )
        .await;
        holder.rollback().await.unwrap();
        outcome.expect("contention retries must be bounded");
        assert_eq!(dropped_count("batch_error") - before, 2);
        assert_eq!(db.count_spend(None, None).await.unwrap(), 0);
        assert_eq!(
            db.get_customer("retry-customer")
                .await
                .unwrap()
                .unwrap()
                .spend,
            0.0
        );
        assert_eq!(
            db.get_tag("retry-tag").await.unwrap().unwrap().spend,
            Some(0.0)
        );
        db.pool().close().await;
    }

    /// Parse the current value of `oxllm_spend_entries_dropped_total{reason=...}`
    /// out of the rendered exposition text (0 when the series is absent).
    fn dropped_count(reason: &str) -> u64 {
        let needle = format!("reason=\"{reason}\"");
        for line in crate::metrics::render().lines() {
            if line.starts_with("oxllm_spend_entries_dropped_total{") && line.contains(&needle) {
                if let Some(v) = line.rsplit(' ').next() {
                    if let Ok(n) = v.parse::<u64>() {
                        return n;
                    }
                }
            }
        }
        0
    }

    #[tokio::test]
    async fn queue_drops_gracefully_when_no_drain() {
        let (tx, mut rx) = mpsc::channel::<SpendEntry>(2);
        let writer = SpendWriter { tx };
        let entry = SpendEntry {
            deployment_id: None,
            requested_model: None,
            model_group: None,
            session_id: None,
            timestamp: "t".to_string(),
            request_id: None,
            model: "gpt-4o".to_string(),
            provider: "openai".to_string(),
            prompt_tokens: Some(1),
            completion_tokens: None,
            total_tokens: None,
            cost: Some(0.1),
            key_id: None,
            team_id: None,
            user_id: None,
            end_user_id: None,
            request_tag: None,
            request_tags: None,
            status: None,
            latency_ms: None,
        };
        writer.queue(entry.clone()).await;
        writer.queue(entry.clone()).await;
        let received = rx.recv().await;
        assert!(received.is_some());
    }

    /// MEDIUM: when the channel is full the overflow is dropped and counted
    /// under `reason=full` so the loss remains observable. There is no
    /// on-disk spill in production (the bounded channel + counter is the
    /// durability contract); the `reason=spilled` series must therefore
    /// never gain samples.
    #[tokio::test]
    #[serial_test::serial]
    async fn queue_overflow_increments_drop_counter() {
        let _ = crate::metrics::install();
        let before_full = dropped_count("full");
        let before_spilled = dropped_count("spilled");
        // Capacity 2, no draining receiver: the first two succeed, every
        // further send overflows.
        let (tx, _rx) = mpsc::channel::<SpendEntry>(2);
        let writer = SpendWriter { tx };
        let entry = SpendEntry {
            deployment_id: None,
            requested_model: None,
            model_group: None,
            session_id: None,
            timestamp: "t".to_string(),
            request_id: None,
            model: "overflow-probe".to_string(),
            provider: "openai".to_string(),
            prompt_tokens: Some(1),
            completion_tokens: None,
            total_tokens: None,
            cost: Some(0.1),
            key_id: None,
            team_id: None,
            user_id: None,
            end_user_id: None,
            request_tag: None,
            request_tags: None,
            status: None,
            latency_ms: None,
        };
        for _ in 0..5 {
            writer.queue(entry.clone()).await;
        }
        let full = dropped_count("full").saturating_sub(before_full);
        let spilled = dropped_count("spilled").saturating_sub(before_spilled);
        assert_eq!(
            full, 3,
            "expected 3 'full' drops (no spill path), got full={full} spilled={spilled}"
        );
        assert_eq!(
            spilled, 0,
            "spilled series must remain zero — spill was removed as dead in production"
        );
    }

    /// Coverage gap (C16): when the channel receiver has been dropped
    /// (`Closed`), every send must be counted under
    /// `oxllm_spend_entries_dropped_total{reason="closed"}` so the loss is
    /// observable. Mirrors `queue_overflow_increments_drop_counter` for the
    /// `full` branch.
    #[tokio::test]
    #[serial_test::serial]
    async fn queue_closed_increments_drop_counter() {
        let _ = crate::metrics::install();
        let before = dropped_count("closed");
        // Drop the receiver immediately so the channel is closed.
        let (tx, rx) = mpsc::channel::<SpendEntry>(2);
        drop(rx);
        let writer = SpendWriter { tx };
        let entry = SpendEntry {
            deployment_id: None,
            requested_model: None,
            model_group: None,
            session_id: None,
            timestamp: "t".to_string(),
            request_id: None,
            model: "closed-probe".to_string(),
            provider: "openai".to_string(),
            prompt_tokens: Some(1),
            completion_tokens: None,
            total_tokens: None,
            cost: Some(0.1),
            key_id: None,
            team_id: None,
            user_id: None,
            end_user_id: None,
            request_tag: None,
            request_tags: None,
            status: None,
            latency_ms: None,
        };
        for _ in 0..3 {
            writer.queue(entry.clone()).await;
        }
        let after = dropped_count("closed");
        assert_eq!(
            after.saturating_sub(before),
            3,
            "expected 3 'closed' drops, got before={before} after={after}"
        );
    }

    #[tokio::test]
    async fn spawned_writer_flushes_to_db() {
        let db = oxllm_db::Database::connect("sqlite::memory:")
            .await
            .expect("connect");
        db.migrate().await.expect("migrate");
        let (writer, handle) = spawn_with_shutdown(db.clone(), 60, DEFAULT_CHANNEL_CAPACITY);

        for i in 0..3 {
            writer
                .queue(SpendEntry {
                    deployment_id: None,
                    requested_model: None,
                    model_group: None,
                    session_id: None,
                    timestamp: oxllm_db::now_iso(),
                    request_id: Some(format!("r-{i}")),
                    model: "gpt-4o".to_string(),
                    provider: "openai".to_string(),
                    prompt_tokens: Some(1),
                    completion_tokens: None,
                    total_tokens: None,
                    cost: Some(0.01),
                    key_id: Some("k-1".to_string()),
                    team_id: None,
                    user_id: None,
                    end_user_id: None,
                    request_tag: None,
                    request_tags: None,
                    status: None,
                    latency_ms: None,
                })
                .await;
        }

        tokio::time::timeout(Duration::from_secs(5), handle.shutdown())
            .await
            .expect("writer flushed on shutdown");
        drop(writer);

        let logs = db
            .query_spend(&oxllm_db::SpendFilter {
                key_id: Some("k-1".to_string()),
                ..Default::default()
            })
            .await
            .expect("query");
        assert_eq!(
            logs.len(),
            3,
            "all queued logs must be flushed exactly once"
        );
    }

    #[tokio::test]
    async fn drain_bounds_a_transaction_and_leaves_the_next_batch_queued() {
        let db = oxllm_db::Database::connect("sqlite::memory:")
            .await
            .unwrap();
        db.migrate().await.unwrap();
        let (tx, mut rx) = mpsc::channel(MAX_FLUSH_BATCH_SIZE + 3);
        let entry: SpendEntry = serde_json::from_value(serde_json::json!({
            "timestamp": "2026-01-01T00:00:00Z",
            "model": "gpt-4o",
            "provider": "openai",
            "cost": 0.125,
        }))
        .unwrap();
        for _ in 0..MAX_FLUSH_BATCH_SIZE + 3 {
            tx.try_send(entry.clone()).unwrap();
        }
        assert!(drain(&db, &mut rx).await);
        assert_eq!(
            db.count_spend(None, None).await.unwrap(),
            MAX_FLUSH_BATCH_SIZE as i64
        );
        assert_eq!(
            rx.len(),
            3,
            "one drain must not monopolize the full backlog"
        );
        rx.close();
        assert!(drain(&db, &mut rx).await);
        assert!(!drain(&db, &mut rx).await);
        assert_eq!(
            db.count_spend(None, None).await.unwrap(),
            (MAX_FLUSH_BATCH_SIZE + 3) as i64
        );
    }

    /// MEDIUM regression: the spawned writer must flush buffered entries on an
    /// explicit `shutdown()` even while senders are still open (the structural
    /// state during graceful server shutdown, where `AppState`/the watcher keep
    /// a sender clone alive). Without the explicit drain, entries queued after
    /// the last ticker tick are lost when the process exits.
    #[tokio::test]
    async fn shutdown_flushes_buffered_entries_while_senders_alive() {
        let db = oxllm_db::Database::connect("sqlite::memory:")
            .await
            .expect("connect");
        db.migrate().await.expect("migrate");
        let (writer, handle) = spawn_with_shutdown(db.clone(), 3600, DEFAULT_CHANNEL_CAPACITY);

        // Queue a burst; each entry is inserted promptly on recv, but the
        // contract we verify is that shutdown drains deterministically and the
        // task exits even though a sender clone (`writer.sender()`) is kept
        // alive — mirroring how `AppState` pins the channel during `run()`.
        let keep_alive_sender = writer.sender();
        for i in 0..5 {
            writer
                .queue(SpendEntry {
                    deployment_id: None,
                    requested_model: None,
                    model_group: None,
                    session_id: None,
                    timestamp: oxllm_db::now_iso(),
                    request_id: Some(format!("r-shutdown-{i}")),
                    model: "gpt-4o".to_string(),
                    provider: "openai".to_string(),
                    prompt_tokens: Some(1),
                    completion_tokens: None,
                    total_tokens: None,
                    cost: Some(0.01),
                    key_id: Some("k-shutdown".to_string()),
                    team_id: None,
                    user_id: None,
                    end_user_id: None,
                    request_tag: None,
                    request_tags: None,
                    status: None,
                    latency_ms: None,
                })
                .await;
        }

        // Trigger the bounded flush. This must resolve (not hang) despite the
        // live sender, proving the explicit shutdown path works independently
        // of the all-senders-dropped condition.
        tokio::time::timeout(Duration::from_secs(5), handle.shutdown())
            .await
            .expect("shutdown must complete within 5s");

        drop(keep_alive_sender);
        drop(writer);

        let logs = db
            .query_spend(&oxllm_db::SpendFilter {
                key_id: Some("k-shutdown".to_string()),
                ..Default::default()
            })
            .await
            .expect("query");
        assert_eq!(
            logs.len(),
            5,
            "all 5 queued entries must be flushed by shutdown, got {}",
            logs.len()
        );
    }

    // ── U11: double-counting prevention on batch failure ──────────────

    /// MEDIUM (C16): `PoolClosed` must NOT trigger the per-row fallback.
    /// While the pool is down every per-row insert would fail identically,
    /// so the fallback would burn N round-trips and emit N identical
    /// warnings for zero recovered rows. The batch is dropped and counted
    /// under `reason=batch_error` instead.
    #[test]
    fn should_retry_per_row_rejects_pool_closed() {
        let err: anyhow::Error = sqlx::Error::PoolClosed.into();
        assert!(
            !should_retry_per_row(&err),
            "PoolClosed must NOT trigger per-row fallback (pool is unusable)"
        );
    }

    /// MEDIUM (C16): same reasoning as PoolClosed — the pool cannot serve
    /// any per-row attempt, so retrying is futile and noisy.
    #[test]
    fn should_retry_per_row_rejects_pool_timed_out() {
        let err: anyhow::Error = sqlx::Error::PoolTimedOut.into();
        assert!(
            !should_retry_per_row(&err),
            "PoolTimedOut must NOT trigger per-row fallback (pool is unavailable)"
        );
    }

    /// The only error that still qualifies for per-row retry: a
    /// transport-level failure raised before commit. The transaction did not
    /// commit, and per-row retries also retain stable event IDs.
    #[test]
    fn should_retry_per_row_accepts_io_error() {
        let io = std::io::Error::new(std::io::ErrorKind::ConnectionReset, "reset");
        let err: anyhow::Error = sqlx::Error::Io(io).into();
        assert!(
            should_retry_per_row(&err),
            "pre-commit Io must trigger the idempotent per-row fallback"
        );
    }

    /// MEDIUM (C16): a commit-time `Io` is ambiguous — the server may have
    /// committed while the client observed the transport reset. Wrapping it
    /// in `CommitFailed` marks it as non-retriable so the per-row fallback
    /// cannot duplicate the (possibly committed) batch.
    #[test]
    fn should_retry_per_row_rejects_commit_failure() {
        let io = std::io::Error::new(std::io::ErrorKind::ConnectionReset, "reset mid-commit");
        let commit_err = oxllm_db::CommitFailed(sqlx::Error::Io(io));
        let err: anyhow::Error = commit_err.into();
        assert!(
            !should_retry_per_row(&err),
            "commit-time Io is ambiguous and must NOT trigger per-row fallback"
        );
    }

    #[tokio::test]
    async fn should_retry_per_row_rejects_database_error() {
        // Syntax/constraint errors are permanent. Known rollback/transient
        // codes qualify for whole-batch retries, never per-row fallback.
        let db = oxllm_db::Database::connect("sqlite::memory:")
            .await
            .expect("connect");
        db.migrate().await.expect("migrate");
        let result = sqlx::query("SELECT * FROM nonexistent_table")
            .execute(db.pool())
            .await;
        let err: anyhow::Error = result.err().unwrap().into();
        assert!(
            !should_retry_per_row(&err),
            "database errors must NOT trigger per-row fallback"
        );
    }

    #[test]
    fn should_retry_per_row_rejects_row_not_found() {
        let err: anyhow::Error = sqlx::Error::RowNotFound.into();
        assert!(!should_retry_per_row(&err));
    }

    #[tokio::test]
    async fn insert_batch_increments_customer_and_tag_spend() {
        let db = oxllm_db::Database::connect("sqlite::memory:")
            .await
            .expect("connect");
        db.migrate().await.expect("migrate");
        let now = oxllm_db::now_iso();
        db.create_customer(&oxllm_db::CustomerRecord {
            end_user_id: "eu-1".to_string(),
            alias: None,
            email: None,
            max_budget: None,
            spend: 0.0,
            budget_duration: None,
            blocked: false,
            model_blocklist: None,
            created_at: now.clone(),
            updated_at: now.clone(),
        })
        .await
        .expect("customer");
        db.create_tag(&oxllm_db::TagRecord {
            name: "prod".to_string(),
            description: None,
            models: None,
            model_info: None,
            spend: Some(0.0),
            max_budget: None,
            created_at: now.clone(),
            updated_at: now,
        })
        .await
        .expect("tag");

        insert_batch(
            &db,
            vec![SpendEntry {
                deployment_id: None,
                requested_model: None,
                model_group: None,
                session_id: None,
                timestamp: oxllm_db::now_iso(),
                request_id: Some("r-1".to_string()),
                model: "gpt-4o".to_string(),
                provider: "openai".to_string(),
                prompt_tokens: Some(1),
                completion_tokens: None,
                total_tokens: None,
                cost: Some(0.25),
                key_id: Some("k-1".to_string()),
                team_id: None,
                user_id: None,
                end_user_id: Some("eu-1".to_string()),
                request_tag: Some("prod".to_string()),
                request_tags: Some(r#"["prod"]"#.to_string()),
                status: None,
                latency_ms: None,
            }],
        )
        .await;

        let customer = db
            .get_customer("eu-1")
            .await
            .expect("get")
            .expect("present");
        assert!(
            (customer.spend - 0.25).abs() < 1e-9,
            "customer spend after flush, got {}",
            customer.spend
        );
        let tag = db.get_tag("prod").await.expect("get").expect("present");
        assert!(
            (tag.spend.unwrap_or(-1.0) - 0.25).abs() < 1e-9,
            "tag spend after flush, got {:?}",
            tag.spend
        );
    }

    /// A non-transient batch failure must not attempt per-row recovery. Only
    /// one row is rejected by the trigger: a wrong fallback would persist its
    /// two valid siblings, making this a behavioral proof rather than checking
    /// an absent table that would also reject every fallback attempt.
    #[tokio::test]
    #[serial_test::serial]
    async fn batch_non_connection_error_does_not_retry_per_row() {
        let _ = crate::metrics::install();
        let db = oxllm_db::Database::connect("sqlite::memory:")
            .await
            .expect("connect");
        db.migrate().await.expect("migrate");

        // Insert one valid row first so we can distinguish "table has data"
        // from "table was just created".
        let valid = SpendEntry {
            deployment_id: None,
            requested_model: None,
            model_group: None,
            session_id: None,
            timestamp: oxllm_db::now_iso(),
            request_id: Some("r-valid".to_string()),
            model: "gpt-4o".to_string(),
            provider: "openai".to_string(),
            prompt_tokens: Some(1),
            completion_tokens: None,
            total_tokens: None,
            cost: Some(0.01),
            key_id: Some("k-1".to_string()),
            team_id: None,
            user_id: None,
            end_user_id: None,
            request_tag: None,
            request_tags: None,
            status: None,
            latency_ms: None,
        };
        insert_batch(&db, vec![valid]).await;
        let count_before = db
            .query_spend(&oxllm_db::SpendFilter::default())
            .await
            .expect("query")
            .len();
        assert_eq!(count_before, 1, "baseline: one row from successful batch");

        sqlx::query(
            "CREATE TRIGGER reject_one_spend_row BEFORE INSERT ON spend_logs \
            WHEN NEW.model = 'fail-sentinel' BEGIN SELECT RAISE(ABORT, 'rejected'); END",
        )
        .execute(db.pool())
        .await
        .expect("trigger");

        // The middle row fails; either sibling could succeed if retried alone.
        let entries: Vec<SpendEntry> = (0..3)
            .map(|i| SpendEntry {
                deployment_id: None,
                requested_model: None,
                model_group: None,
                session_id: None,
                timestamp: oxllm_db::now_iso(),
                request_id: Some(format!("r-retry-{i}")),
                model: if i == 1 { "fail-sentinel" } else { "gpt-4o" }.to_string(),
                provider: "openai".to_string(),
                prompt_tokens: Some(1),
                completion_tokens: None,
                total_tokens: None,
                cost: Some(0.01),
                key_id: Some("k-retry".to_string()),
                team_id: None,
                user_id: None,
                end_user_id: None,
                request_tag: None,
                request_tags: None,
                status: None,
                latency_ms: None,
            })
            .collect();
        let batch_error_before = dropped_count("batch_error");
        insert_batch(&db, entries).await;
        let batch_error_after = dropped_count("batch_error");
        assert_eq!(
            batch_error_after.saturating_sub(batch_error_before),
            3,
            "a 3-row batch dropped on a non-connection error must record 3 \
             batch_error drops, got before={batch_error_before} after={batch_error_after}"
        );

        let retry_logs = db
            .query_spend(&oxllm_db::SpendFilter {
                key_id: Some("k-retry".to_string()),
                ..Default::default()
            })
            .await
            .expect("query");
        assert!(
            retry_logs.is_empty(),
            "per-row fallback must NOT run on non-connection errors — \
             no 'k-retry' rows should exist, got {}",
            retry_logs.len()
        );
    }

    /// HIGH + MEDIUM (C16): on a multi-chunk batch whose second chunk fails
    /// with a non-connection error, the transaction wrapping
    /// `insert_spend_logs_batch` must roll back chunk 1 (so no duplicates
    /// can later be created by the per-row fallback), AND the
    /// `oxllm_spend_entries_dropped_total{reason="batch_error"}` counter
    /// must equal the **full** batch size — because the rollback means
    /// genuinely zero rows were committed, so reporting the full batch as
    /// dropped is now correct (it was previously over-counting when chunks
    /// 1..K-1 had autocommitted).
    #[tokio::test]
    #[serial_test::serial]
    async fn batch_mid_chunk_failure_rolls_back_and_counts_full_batch() {
        let _ = crate::metrics::install();
        let db = oxllm_db::Database::connect("sqlite::memory:")
            .await
            .expect("connect");
        db.migrate().await.expect("migrate");

        // Sentinel trigger: row with model 'fail-sentinel' aborts. Place it
        // on the 77th row (chunk 2, since SQLite chunk_size = 76).
        sqlx::query(
            "CREATE TRIGGER spend_logs_fail_sentinel BEFORE INSERT ON spend_logs \
             FOR EACH ROW WHEN NEW.model = 'fail-sentinel' \
             BEGIN SELECT RAISE(ABORT, 'sentinel model rejected'); END",
        )
        .execute(db.pool())
        .await
        .expect("create trigger");

        let mut entries: Vec<SpendEntry> = Vec::with_capacity(77);
        for i in 0..77 {
            let is_sentinel = i == 76;
            entries.push(SpendEntry {
                deployment_id: None,
                requested_model: None,
                model_group: None,
                session_id: None,
                timestamp: oxllm_db::now_iso(),
                request_id: Some(format!("r-mid-{i}")),
                model: if is_sentinel {
                    "fail-sentinel".to_string()
                } else {
                    "gpt-4o".to_string()
                },
                provider: "openai".to_string(),
                prompt_tokens: Some(1),
                completion_tokens: None,
                total_tokens: None,
                cost: Some(0.01),
                key_id: Some("k-mid".to_string()),
                team_id: None,
                user_id: None,
                end_user_id: None,
                request_tag: None,
                request_tags: None,
                status: None,
                latency_ms: None,
            });
        }

        let batch_error_before = dropped_count("batch_error");
        insert_batch(&db, entries).await;
        let batch_error_after = dropped_count("batch_error");

        // HIGH: zero rows committed (transaction rolled back chunk 1).
        let committed = db
            .query_spend(&oxllm_db::SpendFilter {
                key_id: Some("k-mid".to_string()),
                ..Default::default()
            })
            .await
            .expect("query");
        assert_eq!(
            committed.len(),
            0,
            "mid-batch failure must roll back the entire batch; got {} committed rows",
            committed.len()
        );

        // MEDIUM: the full 77 rows are accounted for under batch_error
        // (correct because the transaction rolled everything back — the
        // counter would have over-counted had chunk 1 autocommitted).
        assert_eq!(
            batch_error_after.saturating_sub(batch_error_before),
            77,
            "expected 77 batch_error drops (full batch, tx rolled back), \
             got before={batch_error_before} after={batch_error_after}"
        );
    }
}
