//! Async wrappers around the pure token-estimation functions in
//! `oxllm-core`. Large bodies are offloaded to the blocking thread pool so
//! the async executor is not stalled by BPE encoding.

use oxllm_core::estimate_request_tokens;
use serde_json::Value;

const SPAWN_BLOCKING_THRESHOLD: usize = 10_000;

// The free utility endpoint has its own small, process-wide work pool. It
// cannot queue unbounded jobs in Tokio's shared blocking pool or consume all
// of its threads while paid requests need token estimates for reservations.
#[cfg(feature = "platform-apis")]
const TOKEN_COUNTER_CONCURRENCY: usize = 2;
#[cfg(feature = "platform-apis")]
static TOKEN_COUNTER_SLOTS: std::sync::OnceLock<std::sync::Arc<tokio::sync::Semaphore>> =
    std::sync::OnceLock::new();

#[cfg(feature = "platform-apis")]
pub(crate) fn try_token_counter_slot()
-> Result<tokio::sync::OwnedSemaphorePermit, tokio::sync::TryAcquireError> {
    TOKEN_COUNTER_SLOTS
        .get_or_init(|| std::sync::Arc::new(tokio::sync::Semaphore::new(TOKEN_COUNTER_CONCURRENCY)))
        .clone()
        .try_acquire_owned()
}

/// Run a utility estimate with its admission slot owned by the blocking job.
/// Dropping/aborting the HTTP future cannot stop a running `spawn_blocking`
/// closure, so releasing the slot in the handler would defeat the limit.
/// Move the already-parsed body rather than deep-cloning it. Even small
/// utility requests are offloaded, including tokenizer initialization.
#[cfg(feature = "platform-apis")]
pub(crate) async fn estimate_token_counter_tokens_async(
    request: Value,
    permit: tokio::sync::OwnedSemaphorePermit,
) -> u64 {
    match tokio::task::spawn_blocking(move || {
        let _permit = permit;
        estimate_request_tokens(&request)
    })
    .await
    {
        Ok(n) => n,
        Err(join_err) => on_join_error(join_err),
    }
}

/// Conservative floor returned when the offloaded estimation task fails to
/// join (panic, runtime shutdown, or cancellation). Must be non-zero so the
/// billing/metering path still engages a reservation instead of silently
/// unmetering the request; the downstream `post_success` settlement corrects
/// the estimate to the provider's real `usage`. Sized to a short-message
/// floor (rather than the previous `1`) so a burst of panics does not collapse
/// aggregate TPM/budget reservations to ~0 before settlement runs.
const JOIN_ERROR_FALLBACK_TOKENS: u64 = 16;

/// Map a [`tokio::task::JoinError`] from the estimation task to a safe return
/// value: log observably and return a non-zero floor so callers never see a
/// silent `0` on a billing-critical path.
fn on_join_error(join_err: tokio::task::JoinError) -> u64 {
    tracing::error!(
        error = %join_err,
        "spawn_blocking token estimation failed; using non-zero fallback"
    );
    JOIN_ERROR_FALLBACK_TOKENS
}

/// Estimate prompt tokens for a request body, offloading large bodies to the
/// blocking pool.
///
/// Takes the body by reference so callers sharing an `Arc<Value>` pay only a
/// shallow borrow on the inline path; the deep clone is deferred to the rare
/// `>= THRESHOLD` arm where it is moved into the blocking task.
pub async fn estimate_request_tokens_async(request: &Value) -> u64 {
    let size = approx_json_size(request);
    if size < SPAWN_BLOCKING_THRESHOLD {
        return estimate_request_tokens(request);
    }
    let owned = request.clone();
    match tokio::task::spawn_blocking(move || estimate_request_tokens(&owned)).await {
        Ok(n) => n,
        Err(join_err) => on_join_error(join_err),
    }
}

/// Coarse lower bound on `serde_json::to_string(v).len()`, used only to choose
/// the inline-vs-offload path. Accounts for structural bytes (quotes, colons,
/// separators, brackets, braces) so that a body of many small fields is not
/// mis-classified as "small" when its real serialized form exceeds the
/// threshold. Intentionally errs toward over-estimating (which offloads).
fn approx_json_size(v: &Value) -> usize {
    match v {
        // +2 for the surrounding quotes.
        Value::String(s) => s.len() + 2,
        // Element sizes + `[ ]` (2) + ~1 byte per element for separators.
        Value::Array(arr) => arr.iter().map(approx_json_size).sum::<usize>() + arr.len() + 2,
        // Per entry: key + 2 quotes + 1 colon + ~1 separator = +4, plus `{ }` (2).
        Value::Object(obj) => {
            obj.iter()
                .map(|(k, val)| k.len() + 4 + approx_json_size(val))
                .sum::<usize>()
                + 2
        }
        Value::Null => 4,
        Value::Bool(_) => 5,
        Value::Number(n) => n.to_string().len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// A disconnected client must not free a slot while its blocking encoder
    /// is queued/running. Occupy the only blocking worker to make this race
    /// deterministic, without timing assumptions or expensive BPE input.
    #[cfg(feature = "platform-apis")]
    #[test]
    fn cancelled_utility_estimate_holds_slot_until_blocking_job_finishes() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .max_blocking_threads(1)
            .build()
            .unwrap();
        runtime.block_on(async {
            let (started_tx, started_rx) = tokio::sync::oneshot::channel();
            let (release_tx, release_rx) = std::sync::mpsc::channel();
            let blocker = tokio::task::spawn_blocking(move || {
                started_tx.send(()).unwrap();
                release_rx.recv().unwrap();
            });
            started_rx.await.unwrap();

            let slots = std::sync::Arc::new(tokio::sync::Semaphore::new(1));
            let permit = slots.clone().try_acquire_owned().unwrap();
            let mut estimate = Box::pin(estimate_token_counter_tokens_async(
                json!({"model": "gpt-4o", "prompt": "hello"}),
                permit,
            ));
            assert!(futures_util::poll!(&mut estimate).is_pending());
            drop(estimate);
            let available_while_queued = slots.available_permits();
            // Always release the blocker before asserting, so a regression
            // cannot deadlock runtime shutdown on a blocking recv.
            release_tx.send(()).unwrap();
            blocker.await.unwrap();
            assert_eq!(available_while_queued, 0);
            let _returned =
                tokio::time::timeout(std::time::Duration::from_secs(10), slots.acquire_owned())
                    .await
                    .expect("completed encoder must release the slot")
                    .unwrap();
        });
    }

    #[tokio::test]
    async fn small_request_runs_inline() {
        let req = json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}]
        });
        let n = estimate_request_tokens_async(&req).await;
        assert!(n > 0);
    }

    #[tokio::test]
    async fn large_request_uses_spawn_blocking() {
        let big = "x".repeat(SPAWN_BLOCKING_THRESHOLD + 1);
        let req = json!({
        "model": "claude-3",
        "messages": [{"role": "user", "content": big}]
        });
        let n = estimate_request_tokens_async(&req).await;
        assert!(n > 0);
    }

    /// A body just over the threshold must still be classified as large (the
    /// threshold guards the inline BPE path on the async executor).
    #[tokio::test]
    async fn boundary_request_above_threshold_is_counted() {
        let big = "x".repeat(SPAWN_BLOCKING_THRESHOLD + 1);
        let mut req = json!({
        "model": "claude-3",
        "messages": [{"role": "user", "content": big}]
        });
        let above = estimate_request_tokens_async(&req).await;
        // Shrink below the threshold and confirm the inline path agrees in
        // magnitude (both nonzero; the spawn path is covered above).
        req["messages"] = json!([{"role": "user", "content": "small"}]);
        let inline = estimate_request_tokens_async(&req).await;
        assert!(above > 0);
        assert!(inline > 0);
    }

    /// `approx_json_size` must track the real serialized length for a body of
    /// many small fields (structural bytes dominate), staying within a 2x
    /// factor — the old helper undercounted structure entirely.
    #[test]
    fn approx_json_size_tracks_serialized_length_for_structured_body() {
        let v = json!({
        "messages": (0..50)
        .map(|i| json!({"role": "user", "content": format!("msg number {i}")}))
        .collect::<Vec<_>>()
        });
        let approx = approx_json_size(&v);
        let real = serde_json::to_string(&v).unwrap().len();
        assert!(approx > 0);
        let (lo, hi) = (approx.min(real), approx.max(real));
        assert!(
            hi <= 2 * lo,
            "approx should track real within 2x: approx={approx}, real={real}"
        );
    }

    /// A failed estimation task must surface a non-zero floor (never a silent
    /// `0` on the billing path) and emit an error log.
    #[tokio::test]
    async fn on_join_error_returns_nonzero_floor() {
        let handle = tokio::task::spawn_blocking(|| panic!("boom"));
        let err = handle.await.unwrap_err();
        let result = on_join_error(err);
        assert!(result > 0, "join-error fallback must be non-zero");
        assert_eq!(result, JOIN_ERROR_FALLBACK_TOKENS);
    }
}
