//! The adaptive-router learning hook: updates the bandit's Beta cells from
//! post-call outcomes.
//!
//! The strategy (`oxllm_core::router::AdaptiveRouter`) stays pure — all
//! learning happens here:
//!
//! - `post_success` → `alpha += 1` for the stamped cell
//! - `post_failure` → `beta += 1` for the stamped cell
//!
//! The cell identity (`(request_type, deployment_key)`, where the key is
//! the router's cooldown key `{model_name}:{index}`) is stamped onto
//! [`CallContext::adaptive_cell`] by the orchestration at pick time —
//! re-stamped on every pick whose leg resolves to the adaptive strategy
//! and CLEARED by non-adaptive picks, so the post-call hooks always see
//! the LAST adaptive-sampled deployment (the serving one on success, the
//! last tried on failure). Failed deployments whose retries are exhausted
//! inside the retry loop are debited at the cooling point instead — the
//! orchestration clears the stamp right after that per-pick debit, so
//! each failed deployment is debited exactly once per request and a
//! rescued intermediate failure never reaches this hook. Requests with
//! no stamped cell (adaptive routing not in play) and cache hits (no
//! upstream observation, matching the metering hooks) are skipped.
//!
//! The hook holds a clone of the router's [`BanditStore`] `Arc`, so hook
//! updates are immediately visible to every adaptive strategy the router
//! built from the same store. Registered only when the adaptive strategy
//! is configured (globally or for any routing group) — see
//! `state::hook_builders`.

use std::sync::Arc;

use async_trait::async_trait;
use oxllm_core::router::{BanditStore, RequestType};
use oxllm_core::{CallContext, Error, Hook, ResponseSummary};

/// Post-call learning for the adaptive router, over the shared bandit
/// store.
#[derive(Debug)]
pub struct AdaptiveHook {
    bandit: Arc<BanditStore>,
}

/// Gauge of the bandit posterior's alpha per `(request_type, model)`,
/// refreshed after every learning update.
pub const ADAPTIVE_ALPHA: &str = "oxllm_adaptive_alpha";

/// Gauge of the bandit posterior's beta per `(request_type, model)`,
/// refreshed after every learning update.
pub const ADAPTIVE_BETA: &str = "oxllm_adaptive_beta";

/// Re-export a bandit cell's posterior to the alpha/beta gauges after a
/// learning update. Shared by this hook (`post_success` / `post_failure`)
/// and the orchestration's per-pick failure debit, which bypasses the hook
/// (it clears the stamp immediately after debiting) — without the shared
/// export, always-failing paths would leave the gauges permanently behind
/// the store. The deployment label uses the cell key's model part (the key
/// is `{model_name}:{index}`), keeping cardinality at model granularity
/// rather than per-deployment-index.
///
/// The model segment is recovered with [`str::rsplit_once`] on `:` so only
/// the trailing `:index` suffix is stripped. A left-to-right `split(':')`
/// would silently truncate a model name that itself contains a colon (e.g.
/// a provider-prefixed `openai:gpt-4o:0`) to just `openai`.
pub(crate) fn export_cell_gauges(bandit: &BanditStore, cell: &(RequestType, String)) {
    let snapshot = bandit.sample_snapshot(std::slice::from_ref(cell));
    let Some(posterior) = snapshot.first() else {
        return;
    };
    let model = cell
        .1
        .rsplit_once(':')
        .map(|(m, _)| m)
        .unwrap_or(cell.1.as_str())
        .to_string();
    ::metrics::gauge!(ADAPTIVE_ALPHA, "request_type" => cell.0.as_str(), "model" => model.clone())
        .set(posterior.alpha);
    ::metrics::gauge!(ADAPTIVE_BETA, "request_type" => cell.0.as_str(), "model" => model)
        .set(posterior.beta);
}

impl AdaptiveHook {
    /// Create the hook over the router's shared bandit store.
    pub fn new(bandit: Arc<BanditStore>) -> Self {
        Self { bandit }
    }
}

#[async_trait]
impl Hook for AdaptiveHook {
    async fn post_success(&self, ctx: &CallContext, _resp: &ResponseSummary) {
        // Cache hits made no upstream observation — there is nothing to
        // learn about the deployment (matches the metering hooks).
        if ctx.cache_hit {
            return;
        }
        let Some(cell) = &ctx.adaptive_cell else {
            return;
        };
        self.bandit.record_success(cell);
        export_cell_gauges(&self.bandit, cell);
        tracing::debug!(
            event = "adaptive_learned",
            outcome = "success",
            request_type = cell.0.as_str(),
            deployment = %cell.1,
            "bandit cell updated"
        );
    }

    async fn post_failure(&self, ctx: &CallContext, _err: &Error) {
        // Cache hits are a success outcome — the cache short-circuit always
        // returns a response, never an error — so `cache_hit == true` is
        // unreachable here today. The guard is kept for symmetry with
        // `post_success` and as defence against a future cache-hit path that
        // can fail; the early return is the documented contract (pinned by
        // `post_failure_on_cache_hit_context_is_a_noop`) — a future code path
        // setting `cache_hit = true` and still driving `post_failure` must not
        // debit the bandit cell.
        if ctx.cache_hit {
            return;
        }
        let Some(cell) = &ctx.adaptive_cell else {
            return;
        };
        self.bandit.record_failure(cell);
        export_cell_gauges(&self.bandit, cell);
        tracing::debug!(
            event = "adaptive_learned",
            outcome = "failure",
            request_type = cell.0.as_str(),
            deployment = %cell.1,
            "bandit cell updated"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oxllm_core::router::RequestType;

    fn cell() -> (RequestType, String) {
        (RequestType::Code, "gpt-4o:0".to_string())
    }

    fn ctx_with_cell() -> CallContext {
        CallContext {
            adaptive_cell: Some(cell()),
            ..CallContext::for_test("gpt-4o")
        }
    }

    fn summary() -> ResponseSummary {
        ResponseSummary {
            model: "gpt-4o".to_string(),
            cost: None,
            usage: None,
            body: None,
            provider: "openai".to_string(),
        }
    }

    #[tokio::test]
    async fn post_success_records_success_on_stamped_cell() {
        let bandit = Arc::new(BanditStore::new());
        let hook = AdaptiveHook::new(bandit.clone());
        hook.post_success(&ctx_with_cell(), &summary()).await;
        let snapshot = bandit.sample_snapshot(&[cell()]);
        assert_eq!(snapshot[0].alpha, 2.0, "success increments alpha");
        assert_eq!(snapshot[0].beta, 1.0);
    }

    #[tokio::test]
    async fn post_failure_records_failure_on_stamped_cell() {
        let bandit = Arc::new(BanditStore::new());
        let hook = AdaptiveHook::new(bandit.clone());
        hook.post_failure(&ctx_with_cell(), &Error::Network("boom".into()))
            .await;
        let snapshot = bandit.sample_snapshot(&[cell()]);
        assert_eq!(snapshot[0].alpha, 1.0);
        assert_eq!(snapshot[0].beta, 2.0, "failure increments beta");
    }

    /// The shared gauge export (hook post-paths AND the orchestration's
    /// per-pick debit) mirrors the cell's current posterior, with the
    /// deployment label reduced to the model part of the cell key.
    #[tokio::test]
    #[serial_test::serial]
    async fn export_cell_gauges_mirrors_current_posterior() {
        // Install the Prometheus recorder BEFORE writing — macro writes go
        // to a no-op recorder until the first `render()`/`install()`.
        let _ = crate::metrics::render();
        let bandit = Arc::new(BanditStore::new());
        let gauge_cell = (RequestType::Code, "gauge-export-model:0".to_string());
        bandit.record_success(&gauge_cell);
        bandit.record_failure(&gauge_cell);
        export_cell_gauges(&bandit, &gauge_cell);
        let rendered = crate::metrics::render();
        assert!(
            rendered.contains(
                "oxllm_adaptive_alpha{request_type=\"code\",model=\"gauge-export-model\"} 2"
            ),
            "alpha gauge must mirror the store (prior 1 + 1 success), got:\n{rendered}"
        );
        assert!(
            rendered.contains(
                "oxllm_adaptive_beta{request_type=\"code\",model=\"gauge-export-model\"} 2"
            ),
            "beta gauge must mirror the store (prior 1 + 1 failure), got:\n{rendered}"
        );
    }

    /// LOW: the model segment of the cell key (`{model_name}:{index}`) is
    /// recovered with `rsplit_once(':')` so only the trailing `:index` is
    /// stripped. A model name that itself contains a colon (e.g. a
    /// provider-prefixed `prefix:model:0`) must export the full
    /// `prefix:model` label, not the truncated `prefix`.
    #[tokio::test]
    #[serial_test::serial]
    async fn export_cell_gauges_handles_colon_in_model_name() {
        let _ = crate::metrics::render();
        let bandit = Arc::new(BanditStore::new());
        let gauge_cell = (RequestType::Code, "prefix:model:0".to_string());
        bandit.record_success(&gauge_cell);
        export_cell_gauges(&bandit, &gauge_cell);
        let rendered = crate::metrics::render();
        assert!(
            rendered.contains("oxllm_adaptive_alpha{request_type=\"code\",model=\"prefix:model\"}"),
            "colon-bearing model name must not be truncated, got:\n{rendered}"
        );
    }

    /// LOW: a cell key with no `:` (malformed/missing index) falls through to
    /// the whole string as the model label, never panicking.
    #[tokio::test]
    #[serial_test::serial]
    async fn export_cell_gauges_handles_key_without_colon() {
        let _ = crate::metrics::render();
        let bandit = Arc::new(BanditStore::new());
        let gauge_cell = (RequestType::Code, "gpt-4o".to_string());
        bandit.record_success(&gauge_cell);
        export_cell_gauges(&bandit, &gauge_cell);
        let rendered = crate::metrics::render();
        assert!(
            rendered.contains("oxllm_adaptive_alpha{request_type=\"code\",model=\"gpt-4o\"}"),
            "colon-less key must use the whole string as the model label, got:\n{rendered}"
        );
    }

    #[tokio::test]
    async fn cache_hit_skips_learning() {
        let bandit = Arc::new(BanditStore::new());
        let hook = AdaptiveHook::new(bandit.clone());
        let mut ctx = ctx_with_cell();
        ctx.cache_hit = true;
        hook.post_success(&ctx, &summary()).await;
        hook.post_failure(&ctx, &Error::Network("boom".into()))
            .await;
        assert_eq!(
            bandit.observations(&cell()),
            0.0,
            "cache hits carry no upstream observation"
        );
    }

    #[tokio::test]
    async fn unstamped_cell_is_a_noop() {
        let bandit = Arc::new(BanditStore::new());
        let hook = AdaptiveHook::new(bandit.clone());
        let ctx = CallContext::for_test("gpt-4o");
        hook.post_success(&ctx, &summary()).await;
        hook.post_failure(&ctx, &Error::Network("boom".into()))
            .await;
        assert_eq!(
            bandit.observations(&cell()),
            0.0,
            "no stamped cell → nothing to learn"
        );
    }

    /// Documents the contract: `post_failure` on a `cache_hit == true` context
    /// is a no-op (cache hits carry no upstream observation). Today this is
    /// unreachable because cache short-circuits produce a success, never a
    /// failure; the test pins that a future code path setting `cache_hit` and
    /// still driving `post_failure` does not debit the bandit cell. Runs under
    /// the `debug_assert!` in release without panicking.
    #[tokio::test]
    async fn post_failure_on_cache_hit_context_is_a_noop() {
        let bandit = Arc::new(BanditStore::new());
        let hook = AdaptiveHook::new(bandit.clone());
        let mut ctx = ctx_with_cell();
        ctx.cache_hit = true;
        hook.post_failure(&ctx, &Error::Network("post-short-circuit failure".into()))
            .await;
        assert_eq!(
            bandit.observations(&cell()),
            0.0,
            "post_failure on a cache-hit context must not debit the cell"
        );
    }

    #[tokio::test]
    async fn repeated_outcomes_accumulate() {
        let bandit = Arc::new(BanditStore::new());
        let hook = AdaptiveHook::new(bandit.clone());
        for _ in 0..3 {
            hook.post_success(&ctx_with_cell(), &summary()).await;
        }
        for _ in 0..2 {
            hook.post_failure(&ctx_with_cell(), &Error::Network("boom".into()))
                .await;
        }
        assert_eq!(bandit.observations(&cell()), 5.0);
    }
}
