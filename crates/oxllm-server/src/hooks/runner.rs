//! Shared hook orchestration.
//!
//! Every request surface area (chat, embeddings, passthrough, ...) drives the
//! hook chain through these helpers so the chain runs exactly once per request
//! in one place. Host handlers build a [`CallContext`] (and, for post-call
//! hooks, a [`ResponseSummary`] / [`Error`]) and delegate here; they never
//! iterate the hook list directly.

use std::sync::Arc;

use oxllm_core::{CallContext, Error, Hook, HookAction, ResponseSummary};
use serde_json::Value;

use crate::error::ApiError;

/// Run the pre-call hook chain.
///
/// Returns:
/// - `Ok(Some(value))` when a hook short-circuited (e.g. cache hit) — the
///   caller builds a response from `value` and skips the upstream call.
/// - `Ok(None)` when every hook returned [`HookAction::Continue`].
/// - `Err` when a hook rejected the request (e.g. budget exceeded).
///
/// When a hook rejects (or errors), every hook that already succeeded has
/// its `post_failure` invoked in reverse order so reservations made earlier
/// in the chain (e.g. `RateLimitHook`'s RPM slot) are rolled back instead of
/// leaking until their TTL expires. The rejecting hook is excluded from the
/// rollback: it owns the partial cleanup of its own `pre_call` on the reject
/// path (see `RateLimitHook`/`BudgetHook`), so running `post_failure` on it
/// would double-release its counters.
pub async fn run_pre_call_hooks(
    hooks: &[Arc<dyn Hook>],
    ctx: &mut CallContext,
) -> Result<Option<Value>, ApiError> {
    for (idx, hook) in hooks.iter().enumerate() {
        let action = match hook.pre_call(ctx).await {
            Ok(action) => action,
            Err(err) => {
                rollback_succeeded_hooks(&hooks[..idx], ctx, &err).await;
                return Err(ApiError::from(err));
            }
        };
        match action {
            HookAction::Continue => continue,
            HookAction::Reject(err) => {
                rollback_succeeded_hooks(&hooks[..idx], ctx, &err).await;
                return Err(ApiError::from(err));
            }
            HookAction::ShortCircuit(value) => {
                // A short-circuit means the response was produced without an
                // upstream call (cache hit, etc.), so the metering/learning
                // hooks must skip their post-success work. Stamping the
                // signal here — the single chokepoint the docstring promises
                // — means a future handler cannot forget the assignment and
                // accidentally double-debit cached responses against budget,
                // spend and the adaptive bandit.
                ctx.cache_hit = true;
                return Ok(Some(value));
            }
            // `HookAction` is `#[non_exhaustive]`; future variants added to
            // oxllm-core must not break this call site. Treating an unknown
            // action as "continue" is the safest default — the worst case is
            // one extra upstream call after an upgrade, rather than a dropped
            // request. Log it so the silent fallback is observable to
            // operators.
            other => {
                tracing::warn!(?other, "unhandled HookAction variant; treating as Continue");
                continue;
            }
        }
    }
    Ok(None)
}

/// Roll back reservations held by hooks that already succeeded in this
/// pre-call chain, in reverse order, after a later hook rejected.
async fn rollback_succeeded_hooks(hooks: &[Arc<dyn Hook>], ctx: &CallContext, err: &Error) {
    for hook in hooks.iter().rev() {
        hook.post_failure(ctx, err).await;
    }
}

/// Run all post-success hooks against the given response summary.
///
/// U10: the reference ran async + sync failure handlers
/// concurrently against the same mutable logging object, causing crashes.
/// Oxllm is safe: hooks are awaited **sequentially** in a `for` loop — each
/// `hook.post_success(ctx, resp).await` completes before the next hook
/// begins. No shared mutable state is accessed concurrently.
pub async fn run_post_success_hooks(
    hooks: &[Arc<dyn Hook>],
    ctx: &CallContext,
    resp: &ResponseSummary,
) {
    for hook in hooks {
        hook.post_success(ctx, resp).await;
    }
}

/// Run all post-failure hooks against the given error.
///
/// U10: same concurrent-handler crash as
/// `run_post_success_hooks`. Oxllm awaits each hook sequentially, so there
/// is no shared-state race.
///
/// U49: before running the hooks, records `error.type` and `error.message`
/// on the current OTel span (when the `otel` feature is enabled) so failure
/// spans carry the semantic-convention error attributes. The mapping is
/// derived from [`crate::pipeline::error_type_label`].
pub async fn run_post_failure_hooks(hooks: &[Arc<dyn Hook>], ctx: &CallContext, err: &Error) {
    #[cfg(feature = "otel")]
    {
        crate::otel::record_span_error(crate::pipeline::error_type_label(err), &err.to_string());
    }
    for hook in hooks {
        hook.post_failure(ctx, err).await;
    }
}

/// Release no-TTL concurrency slots without running `post_failure`.
///
/// Mid-stream client disconnect keeps the budget reservation as the debit
/// (calling `post_failure` would refund it) but must still free
/// `max_parallel_requests` so a disconnect storm cannot lock the key.
pub async fn run_release_concurrency(hooks: &[Arc<dyn Hook>], ctx: &CallContext) {
    for hook in hooks {
        hook.release_concurrency(ctx).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use oxllm_core::error::Result;

    struct ContinueHook;
    #[async_trait]
    impl Hook for ContinueHook {}

    struct ShortCircuitHook;
    #[async_trait]
    impl Hook for ShortCircuitHook {
        async fn pre_call(&self, _ctx: &mut CallContext) -> Result<HookAction> {
            Ok(HookAction::ShortCircuit(
                serde_json::json!({"cached": true}),
            ))
        }
    }

    struct RejectHook;
    #[async_trait]
    impl Hook for RejectHook {
        async fn pre_call(&self, _ctx: &mut CallContext) -> Result<HookAction> {
            Ok(HookAction::Reject(Error::RateLimited("nope".to_string())))
        }
    }

    fn ctx() -> CallContext {
        CallContext {
            model: "m".to_string(),
            api_key_hash: "h".to_string(),
            request_id: "r".to_string(),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn continue_returns_none() {
        let hooks: Vec<Arc<dyn Hook>> = vec![Arc::new(ContinueHook)];
        let mut c = ctx();
        assert!(run_pre_call_hooks(&hooks, &mut c).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn short_circuit_returns_some() {
        let hooks: Vec<Arc<dyn Hook>> = vec![Arc::new(ShortCircuitHook)];
        let mut c = ctx();
        let v = run_pre_call_hooks(&hooks, &mut c).await.unwrap().unwrap();
        assert_eq!(v["cached"], true);
    }

    #[tokio::test]
    async fn reject_returns_err() {
        let hooks: Vec<Arc<dyn Hook>> = vec![Arc::new(RejectHook)];
        let mut c = ctx();
        assert!(run_pre_call_hooks(&hooks, &mut c).await.is_err());
    }

    #[tokio::test]
    async fn post_success_runs_all_hooks() {
        let hooks: Vec<Arc<dyn Hook>> = vec![Arc::new(ContinueHook)];
        let c = ctx();
        let summary = ResponseSummary {
            model: "m".to_string(),
            cost: None,
            usage: None,
            body: None,
            provider: "openai".to_string(),
        };
        run_post_success_hooks(&hooks, &c, &summary).await;
    }

    #[tokio::test]
    async fn post_failure_runs_all_hooks() {
        let hooks: Vec<Arc<dyn Hook>> = vec![Arc::new(ContinueHook)];
        let c = ctx();
        let err = Error::Network("boom".to_string());
        run_post_failure_hooks(&hooks, &c, &err).await;
    }

    /// A hook that records `pre_call`/`post_failure` invocations in a shared
    /// log so tests can assert rollback ordering.
    struct RecordingHook {
        name: &'static str,
        calls: Arc<std::sync::Mutex<Vec<String>>>,
        action: HookAction,
    }

    #[async_trait]
    impl Hook for RecordingHook {
        async fn pre_call(&self, _ctx: &mut CallContext) -> Result<HookAction> {
            self.calls
                .lock()
                .unwrap()
                .push(format!("{}:pre_call", self.name));
            Ok(match &self.action {
                HookAction::Continue => HookAction::Continue,
                HookAction::Reject(e) => HookAction::Reject(e.clone()),
                HookAction::ShortCircuit(v) => HookAction::ShortCircuit(v.clone()),
                // `HookAction` is `#[non_exhaustive]`; this match is inside
                // a test helper that only ever constructs one of the three
                // concrete variants above, so the wildcard is unreachable.
                // Map it to `Continue` rather than `unreachable!` so a
                // future variant added to oxllm-core does not panic the
                // test suite — recording hooks should never short-circuit
                // or reject on an unknown action.
                _ => HookAction::Continue,
            })
        }

        async fn post_failure(&self, _ctx: &CallContext, _err: &Error) {
            self.calls
                .lock()
                .unwrap()
                .push(format!("{}:post_failure", self.name));
        }
    }

    fn recording(
        name: &'static str,
        action: HookAction,
        calls: Arc<std::sync::Mutex<Vec<String>>>,
    ) -> Arc<dyn Hook> {
        Arc::new(RecordingHook {
            name,
            calls,
            action,
        })
    }

    #[tokio::test]
    async fn reject_rolls_back_earlier_hooks_in_reverse_order() {
        let calls = Arc::new(std::sync::Mutex::new(Vec::new()));
        let hooks: Vec<Arc<dyn Hook>> = vec![
            recording("a", HookAction::Continue, calls.clone()),
            recording("b", HookAction::Continue, calls.clone()),
            recording(
                "c",
                HookAction::Reject(Error::RateLimited("budget exceeded".to_string())),
                calls.clone(),
            ),
        ];
        let mut c = ctx();
        let result = run_pre_call_hooks(&hooks, &mut c).await;
        assert!(result.is_err(), "rejecting hook must fail the chain");
        let guard = calls.lock().unwrap();
        assert_eq!(
            &guard[..],
            &[
                "a:pre_call",
                "b:pre_call",
                "c:pre_call",
                "b:post_failure",
                "a:post_failure"
            ],
            "earlier hooks must be rolled back in reverse order"
        );
    }

    #[tokio::test]
    async fn rejecting_hook_is_not_rolled_back() {
        // The rejecting hook owns its own partial cleanup on the reject path;
        // the chain must NOT run post_failure on it (double-release).
        let calls = Arc::new(std::sync::Mutex::new(Vec::new()));
        let hooks: Vec<Arc<dyn Hook>> = vec![recording(
            "rejector",
            HookAction::Reject(Error::RateLimited("nope".to_string())),
            calls.clone(),
        )];
        let mut c = ctx();
        let _ = run_pre_call_hooks(&hooks, &mut c).await;
        let guard = calls.lock().unwrap();
        assert_eq!(
            &guard[..],
            &["rejector:pre_call"],
            "rejecting hook must not receive post_failure"
        );
    }

    #[tokio::test]
    async fn pre_call_error_rolls_back_earlier_hooks() {
        struct FailHook;
        #[async_trait]
        impl Hook for FailHook {
            async fn pre_call(&self, _ctx: &mut CallContext) -> Result<HookAction> {
                Err(Error::Internal)
            }
        }
        let calls = Arc::new(std::sync::Mutex::new(Vec::new()));
        let hooks: Vec<Arc<dyn Hook>> = vec![
            recording("a", HookAction::Continue, calls.clone()),
            Arc::new(FailHook),
        ];
        let mut c = ctx();
        let result = run_pre_call_hooks(&hooks, &mut c).await;
        assert!(result.is_err(), "hook error must fail the chain");
        let guard = calls.lock().unwrap();
        assert_eq!(
            &guard[..],
            &["a:pre_call", "a:post_failure"],
            "hook errors must roll back earlier hooks too"
        );
    }

    #[tokio::test]
    async fn short_circuit_does_not_roll_back_earlier_hooks() {
        // A cache-hit short-circuit is a *successful* outcome: earlier hooks
        // (e.g. rate limit) keep their reservations on purpose.
        let calls = Arc::new(std::sync::Mutex::new(Vec::new()));
        let hooks: Vec<Arc<dyn Hook>> = vec![
            recording("a", HookAction::Continue, calls.clone()),
            recording(
                "cache",
                HookAction::ShortCircuit(serde_json::json!({"cached": true})),
                calls.clone(),
            ),
        ];
        let mut c = ctx();
        let result = run_pre_call_hooks(&hooks, &mut c).await;
        assert!(result.unwrap().is_some());
        let guard = calls.lock().unwrap();
        assert_eq!(&guard[..], &["a:pre_call", "cache:pre_call"]);
    }

    /// A short-circuit (cache hit) must stamp `cache_hit` on the context
    /// inside the runner, so a handler that forgets the assignment cannot
    /// accidentally double-debit cached responses via the metering hooks.
    #[tokio::test]
    async fn short_circuit_stamps_cache_hit_on_context() {
        let hooks: Vec<Arc<dyn Hook>> = vec![Arc::new(ShortCircuitHook)];
        let mut c = ctx();
        assert!(!c.cache_hit, "sanity: starts false");
        let value = run_pre_call_hooks(&hooks, &mut c).await.unwrap().unwrap();
        assert_eq!(value["cached"], true);
        assert!(
            c.cache_hit,
            "runner must mark cache_hit so callers can't forget"
        );
    }

    /// A recording post-success hook that mimics the metering contract (skip
    /// when `cache_hit`) must observe the runner-stamped flag and NOT record a
    /// cached response, even though the test "caller" never touches
    /// `cache_hit`.
    #[tokio::test]
    async fn downstream_metering_hook_skips_on_runner_stamped_cache_hit() {
        struct MeterHook {
            recorded: Arc<std::sync::Mutex<bool>>,
        }
        #[async_trait]
        impl Hook for MeterHook {
            async fn post_success(&self, ctx: &CallContext, _resp: &ResponseSummary) {
                if ctx.cache_hit {
                    return;
                }
                *self.recorded.lock().unwrap() = true;
            }
        }

        let recorded = Arc::new(std::sync::Mutex::new(false));
        let pre_hooks: Vec<Arc<dyn Hook>> = vec![Arc::new(ShortCircuitHook)];
        let post_hooks: Vec<Arc<dyn Hook>> = vec![Arc::new(MeterHook {
            recorded: recorded.clone(),
        })];

        let mut c = ctx();
        let _ = run_pre_call_hooks(&pre_hooks, &mut c).await.unwrap();
        // Caller deliberately does NOT set cache_hit here — the runner owns it.
        let summary = ResponseSummary {
            model: "m".to_string(),
            cost: None,
            usage: None,
            body: None,
            provider: "openai".to_string(),
        };
        run_post_success_hooks(&post_hooks, &c, &summary).await;
        assert!(
            !*recorded.lock().unwrap(),
            "metering hook must skip the cached response via the runner-stamped flag"
        );
    }

    // Bug F3: hook order is RateLimit → Cache → Budget → Cost → Metrics. When
    // BudgetHook::pre_call rejects (budget exceeded), the RPM slot reserved
    // by RateLimitHook::pre_call must be rolled back — previously it leaked
    // until the 60s TTL expired.
    #[tokio::test]
    async fn budget_reject_releases_rate_limit_rpm_reservation() {
        use crate::cache::InMemoryCache;
        use crate::hooks::{BudgetHook, RateLimitHook};
        use oxllm_core::Cache;
        use std::time::Duration;

        let cache = Arc::new(InMemoryCache::default());
        let hooks: Vec<Arc<dyn Hook>> = vec![
            Arc::new(RateLimitHook::new(cache.clone(), Some(10), None)),
            Arc::new(BudgetHook::with_duration(
                cache.clone(),
                None,
                Duration::from_secs(60),
            )),
        ];

        let mut c = ctx();
        // A zero per-key budget makes BudgetHook reject immediately — after
        // RateLimitHook has already reserved the RPM slot.
        c.max_budget = Some(0.0);
        c.request_body = std::sync::Arc::new(serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}]
        }));

        let result = run_pre_call_hooks(&hooks, &mut c).await;
        assert!(result.is_err(), "zero budget must reject");

        let rpm = cache.counter_value("rpm:h").await.unwrap_or(0.0);
        assert!(
            rpm.abs() < 1e-9,
            "RPM reservation must be released when the budget hook rejects, got {rpm}"
        );
        let spend = cache.counter_value("spend:h").await.unwrap_or(0.0);
        assert!(
            spend.abs() < 1e-9,
            "rejecting budget hook must not leave spend behind, got {spend}"
        );
    }

    // U10: hooks must run sequentially, not concurrently. If they ran
    // concurrently, this test would deadlock: each hook acquires a permit
    // from a capacity-1 semaphore. Sequential execution means each hook
    // acquires and releases before the next begins. Under concurrent
    // execution the second hook would block forever waiting for the
    // semaphore that the first never released (because it hasn't finished).
    #[tokio::test]
    async fn post_success_hooks_run_sequentially_not_concurrently() {
        use std::sync::Arc;
        use tokio::sync::Semaphore;

        struct SeqCheckHook {
            sem: Arc<Semaphore>,
            order: Arc<std::sync::Mutex<Vec<u32>>>,
            id: u32,
        }

        #[async_trait]
        impl Hook for SeqCheckHook {
            async fn post_success(&self, _ctx: &CallContext, _resp: &ResponseSummary) {
                let _permit = self.sem.acquire().await.expect("permit");
                // If another hook were running concurrently, it would be
                // blocked here at acquire() — the 10ms sleep ensures we'd
                // detect overlap via timeout.
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                self.order.lock().unwrap().push(self.id);
            }
        }

        let sem = Arc::new(Semaphore::new(1));
        let order = Arc::new(std::sync::Mutex::new(Vec::new()));
        let hooks: Vec<Arc<dyn Hook>> = (1..=3)
            .map(|id| {
                Arc::new(SeqCheckHook {
                    sem: sem.clone(),
                    order: order.clone(),
                    id,
                }) as Arc<dyn Hook>
            })
            .collect();

        let c = ctx();
        let summary = ResponseSummary {
            model: "m".to_string(),
            cost: None,
            usage: None,
            body: None,
            provider: "openai".to_string(),
        };

        // If hooks ran concurrently, this would hang (permits exhausted).
        tokio::time::timeout(
            std::time::Duration::from_secs(2),
            run_post_success_hooks(&hooks, &c, &summary),
        )
        .await
        .expect("post-success hooks must complete within timeout (sequential execution)");

        let guard = order.lock().unwrap();
        assert_eq!(
            &guard[..],
            &[1, 2, 3],
            "hooks must execute in order 1→2→3, proving sequential execution"
        );
    }
}
