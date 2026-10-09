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
    /// Request tag from `x-oxllm-tag` (for durable tag spend). The primary
    /// (first) tag.
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
            model: entry.model,
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
            end_user_id: entry.end_user_id,
            tag_name: entry.request_tag,
            tags: entry.request_tags,
            status: entry.status,
            latency_ms: entry.latency_ms,
        })
        .map(oxllm_db::SpendLogEvent::new)
        .collect();

    match db.insert_spend_events(&events).await {
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
                    error = %e,
                    count = n,
                    "batch insert failed on transport error pre-commit; \
                     falling back to idempotent per-row inserts"
                );
                for event in &events {
                    if let Err(e) = db.insert_spend_events(std::slice::from_ref(event)).await {
                        tracing::warn!(
                            error = %e,
                            "failed to insert spend log (per-row fallback)"
                        );
                        crate::metrics::record_spend_dropped("batch_error", 1);
                    }
                }
            } else {
                tracing::error!(
                    error = %e,
                    count = n,
                    "batch insert failed; no per-row retry for a commit, pool, \
                     or non-transient database error. \
                     The full batch is counted under reason=batch_error."
                );
                crate::metrics::record_spend_dropped("batch_error", n as u64);
            }
        }
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
    if err.downcast_ref::<oxllm_db::CommitFailed>().is_some() {
        return false;
    }
    matches!(err.downcast_ref::<sqlx::Error>(), Some(sqlx::Error::Io(_)))
}

#[cfg(all(test, feature = "db"))]
mod tests {
    use super::*;
    use std::time::Duration;

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
        // A database error (syntax, constraint, etc.) proves the statement
        // reached the DB — retrying would risk double-counting.
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
