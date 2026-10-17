//! Router-level budget reservation protocol: per-deployment and
//! per-provider spend caps enforced at pick time via atomic counter
//! reservations.
//!
//! Split out of `hooks::budget`: these functions do NOT participate in
//! the hook chain. They are consumed directly by the orchestration retry
//! loop ([`crate::orchestration::chat`], [`crate::orchestration::stream`]),
//! the embeddings handler, and the `budget_limiter` pre-call check. The
//! per-key/team/user [`crate::hooks::budget::BudgetHook`] remains in
//! `hooks::budget`.

use std::sync::Arc;
use std::time::Duration;

use futures_util::{StreamExt, stream};
use oxllm_core::router::deployment_key;
use oxllm_core::{Cache, CallContext, CounterWindow, WindowedIncrResult};

use crate::hooks::budget::{BUDGET_EXCEEDED_TOTAL, estimate_max_request_cost};
use crate::hooks::spend_keys::{deployment_spend_key, provider_spend_key};
use crate::scheduler::parse_budget_duration;

/// Default TTL for deployment/provider spend counters when no duration is
/// configured: one day (the reference default budget window).
const DEFAULT_ROUTER_BUDGET_TTL: Duration = Duration::from_secs(86400);

/// Bound per-request fan-out to a remote cache while avoiding one network
/// round trip per deployment in series on the routing hot path.
const SPEND_SNAPSHOT_CONCURRENCY: usize = 8;

/// Nominal cost recorded when a router budget is configured but the actual
/// cost is unknown (`None` / negative — e.g. the model is missing
/// from the pricing table). Without it the spend counter would never
/// advance and the configured cap would silently never engage.
/// A legitimate `Some(0.0)` cost (zero-priced model, fully cached) does NOT
/// use this — it settles as a real zero.
pub const NOMINAL_UNKNOWN_COST: f64 = 0.0001;

/// The scope of a router-level budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetScope {
    /// Per-deployment cap (`oxllm_params.max_budget`).
    Deployment,
    /// Per-provider cap (`router_settings.provider_budget_config`).
    Provider,
}

impl BudgetScope {
    pub fn as_str(self) -> &'static str {
        match self {
            BudgetScope::Deployment => "deployment",
            BudgetScope::Provider => "provider",
        }
    }
}

/// Outcome of evaluating router-budget counters against their limits.
#[derive(Debug, Clone, PartialEq)]
pub enum BudgetDecision {
    /// Every configured counter is under its limit.
    Allowed,
    /// A counter is at/over its limit.
    Exceeded {
        scope: BudgetScope,
        id: String,
        spend: f64,
        limit: f64,
    },
}

/// Pure router-budget evaluation shared by the pre-call `budget_limiter`
/// check (snapshot-based) and router-budget admission:
/// given the configured per-deployment and per-provider counters as
/// `(id, spend, limit)`, return the first scope whose spend is at/over its
/// cap (`spend >= limit` means the budget is exhausted).
///
/// Non-finite values (`NaN`, `±∞`) on either side are treated as exceeded.
/// A `NaN` spend counter (reachable only via a hand-edited Redis value or a
/// corrupt SQLite row, since the write path only ever `incr`s floats derived
/// from the price table) would otherwise make `spend >= limit` evaluate
/// `false` and grant an effectively-infinite budget. Treating any non-finite
/// value as exceeded makes corruption fail-closed. (A legitimately
/// "unlimited" deployment is represented by `max_budget: None`, which never
/// constructs a counter here — not by `Some(f64::INFINITY)`.)
pub fn evaluate_router_budget(
    deployment: Option<(&str, f64, f64)>,
    provider: Option<(&str, f64, f64)>,
) -> BudgetDecision {
    if let Some((id, spend, limit)) = deployment {
        if !spend.is_finite() || !limit.is_finite() || spend >= limit {
            return BudgetDecision::Exceeded {
                scope: BudgetScope::Deployment,
                id: id.to_string(),
                spend,
                limit,
            };
        }
    }
    if let Some((id, spend, limit)) = provider {
        if !spend.is_finite() || !limit.is_finite() || spend >= limit {
            return BudgetDecision::Exceeded {
                scope: BudgetScope::Provider,
                id: id.to_string(),
                spend,
                limit,
            };
        }
    }
    BudgetDecision::Allowed
}

/// Resolve the spend-counter TTL for a deployment: its `budget_duration`
/// parsed using the duration grammar, falling back to [`DEFAULT_ROUTER_BUDGET_TTL`].
fn deployment_budget_ttl(deployment: &oxllm_core::deployment::Deployment) -> Duration {
    deployment
        .oxllm_params
        .budget_duration
        .as_deref()
        .map(parse_budget_duration)
        .unwrap_or(DEFAULT_ROUTER_BUDGET_TTL)
}

/// One reserved router-budget counter.
#[derive(Debug, Clone)]
struct RouterBudgetReservationEntry {
    key: String,
    estimate: f64,
    // None is retained only for legacy manually constructed test reservations.
    // Every production reservation is issued atomically with Some(window).
    window: Option<CounterWindow>,
}

/// Reservations made by [`reserve_router_budgets`]: one entry per
/// configured scope (deployment and/or provider). [`settle_router_spend`]
/// settles each counter to the actual cost (`actual − reserved estimate`);
/// failure paths call [`RouterBudgetReservation::release`] to hand the
/// estimate back.
#[derive(Debug, Clone, Default)]
pub struct RouterBudgetReservation {
    entries: Vec<RouterBudgetReservationEntry>,
}

impl RouterBudgetReservation {
    /// The estimate reserved against `key`, when this reservation covers it.
    pub fn estimate_for(&self, key: &str) -> Option<f64> {
        self.entries
            .iter()
            .find(|e| e.key == key)
            .map(|e| e.estimate)
    }

    fn window_for(&self, key: &str) -> Option<&CounterWindow> {
        self.entries
            .iter()
            .find(|entry| entry.key == key)
            .and_then(|entry| entry.window.as_ref())
    }

    /// Release every reserved estimate (upstream-failure paths). Zero
    /// estimates were never incremented, so there is nothing to release.
    ///
    /// Production entries carry an atomic window receipt: an expired/replaced
    /// window is never refunded, even if another request recreated the key.
    pub async fn release(&self, cache: &Arc<dyn Cache>) {
        for entry in &self.entries {
            if entry.estimate > 0.0 {
                let result = if let Some(window) = &entry.window {
                    cache
                        .incr_in_window(&entry.key, -entry.estimate, window)
                        .await
                } else {
                    cache.decr_if_exists(&entry.key, entry.estimate).await
                };
                if let Err(e) = result {
                    tracing::warn!(
                        key = %entry.key,
                        error = %e,
                        "router budget release failed"
                    );
                }
            }
        }
    }
}

/// Refund an acquired reservation if its request is cancelled before ownership
/// reaches response settlement. Release runs independently of the awaiting task
/// so cancellation during a multi-scope refund cannot repeat an applied debit.
pub(super) struct RouterBudgetGuard {
    cache: Arc<dyn Cache>,
    reservation: RouterBudgetReservation,
    runtime: Option<tokio::runtime::Handle>,
}

impl RouterBudgetGuard {
    pub(super) fn new(cache: Arc<dyn Cache>, reservation: RouterBudgetReservation) -> Self {
        Self {
            cache,
            reservation,
            runtime: tokio::runtime::Handle::try_current().ok(),
        }
    }

    pub(super) fn disarm(&mut self) {
        self.reservation = RouterBudgetReservation::default();
    }

    fn into_reservation(mut self) -> RouterBudgetReservation {
        std::mem::take(&mut self.reservation)
    }

    pub(super) async fn release(&mut self) {
        if let Some(task) = self.dispatch() {
            if let Err(error) = task.await {
                tracing::warn!(%error, "router budget cleanup task failed");
            }
        }
    }

    fn dispatch(&mut self) -> Option<tokio::task::JoinHandle<()>> {
        let reservation = std::mem::take(&mut self.reservation);
        if reservation.entries.is_empty() {
            return None;
        }
        let Some(runtime) = &self.runtime else {
            tracing::warn!("no runtime available for router budget cleanup");
            return None;
        };
        let cache = self.cache.clone();
        Some(runtime.spawn(async move { reservation.release(&cache).await }))
    }
}

impl Drop for RouterBudgetGuard {
    fn drop(&mut self) {
        let _task = self.dispatch();
    }
}

/// Rejection from [`reserve_router_budgets`]: the 429 to surface plus the
/// exceeded budget's window, so the caller cools the deployment for the
/// rest of the budget window rather than the (much shorter) request-failure
/// cooldown.
#[derive(Debug)]
pub struct RouterBudgetError {
    pub api_error: crate::error::ApiError,
    pub window: Duration,
}

/// Build the budget-exceeded 429 for an exhausted router budget. The error
/// keeps HTTP 429 but carries a distinct `budget_exceeded` code so clients
/// can tell it apart from client rate limiting (`proxy_rate_limited`) and
/// provider 429s (`upstream_rate_limited`) — the same disambiguation
/// pattern `crate::error::ApiError` uses for those codes. Every router
/// budget rejection (reserve-time and read-only gate) funnels through here,
/// so this is also the single metering point for them.
fn router_budget_error(
    scope: BudgetScope,
    id: &str,
    spend: f64,
    limit: f64,
    ttl: Option<Duration>,
) -> RouterBudgetError {
    ::metrics::counter!(BUDGET_EXCEEDED_TOTAL, "scope" => scope.as_str()).increment(1);
    tracing::warn!(
        event = "budget_exceeded",
        scope = scope.as_str(),
        id = %id,
        spend = spend,
        limit = limit,
        "{} budget exceeded", scope.as_str()
    );
    let message = match scope {
        BudgetScope::Deployment => {
            format!("deployment budget ({limit} USD) exceeded for {id} (spend {spend})")
        }
        BudgetScope::Provider => {
            format!("provider budget ({limit} USD) exceeded for '{id}' (spend {spend})")
        }
    };
    RouterBudgetError {
        api_error: crate::error::ApiError::budget_exceeded(message),
        window: ttl.unwrap_or(DEFAULT_ROUTER_BUDGET_TTL),
    }
}

/// One router-budget scope to reserve against: its counter key, scope
/// label, id for logs/errors, configured limit, and counter TTL.
struct RouterScopeCheck<'a> {
    key: String,
    scope: BudgetScope,
    id: &'a str,
    limit: f64,
    ttl: Option<Duration>,
}

/// Atomically reserve `estimate` against one router-budget counter via
/// [`Cache::incr_if_below`], recording the reservation on success. Zero-cost
/// admission uses the same fail-closed atomic operation and still rejects an
/// exhausted counter, without debiting it.
async fn reserve_router_scope(
    state: &crate::state::AppState,
    reservation: &mut RouterBudgetReservation,
    check: RouterScopeCheck<'_>,
    estimate: f64,
) -> Result<(), RouterBudgetError> {
    let RouterScopeCheck {
        key,
        scope,
        id,
        limit,
        ttl,
    } = check;
    if !estimate.is_finite() {
        return Err(router_budget_error(scope, id, estimate, limit, ttl));
    }
    let estimate = estimate.max(0.0);
    match state
        .cache
        .incr_if_below_in_window(&key, estimate, limit, ttl, None)
        .await
    {
        Ok(WindowedIncrResult::Allowed {
            value: spend,
            window,
        }) => {
            if estimate == 0.0 && (!spend.is_finite() || spend >= limit) {
                return Err(router_budget_error(scope, id, spend, limit, ttl));
            }
            reservation.entries.push(RouterBudgetReservationEntry {
                key,
                estimate,
                window: Some(window),
            });
            Ok(())
        }
        Ok(WindowedIncrResult::Rejected(spend)) => {
            Err(router_budget_error(scope, id, spend, limit, ttl))
        }
        Ok(WindowedIncrResult::WindowChanged) => {
            Err(router_budget_error(scope, id, estimate, limit, ttl))
        }
        Err(e) => {
            // Fail closed: same posture as key-level BudgetHook. A cache
            // outage must not disable deployment/provider caps.
            tracing::warn!(
                error = %e,
                scope = scope.as_str(),
                id = %id,
                "router budget reservation failed; rejecting request (fail-closed)"
            );
            Err(RouterBudgetError {
                api_error: crate::error::ApiError::budget_exceeded(format!(
                    "router budget check unavailable for {}/{id}",
                    scope.as_str()
                )),
                window: ttl.unwrap_or(DEFAULT_ROUTER_BUDGET_TTL),
            })
        }
    }
}

/// Check the router-level budgets for a picked deployment and **atomically
/// reserve** an estimated cost against each configured cap: the
/// per-deployment cap (`oxllm_params.max_budget`, counter
/// `spend:dep:{deployment_key}`) and the per-provider cap
/// (`router_settings.provider_budget_config`, counter
/// `spend:provider:{slug}`).
///
/// Mirrors the per-key [`BudgetHook`] reservation pattern: the estimate is
/// the same one [`estimate_max_request_cost`] computes for the key budget
/// (reused from `call_ctx.budget_estimate` when the hook already stashed
/// it), reserved via [`Cache::incr_if_below`] so concurrent requests cannot
/// collectively exceed the ceiling. On success the caller settles the
/// reservation to the actual cost via [`settle_router_spend`]; on upstream
/// failure it releases the estimate via
/// [`RouterBudgetReservation::release`].
///
/// Returns `Err(429)` (code `budget_exceeded`) when either counter cannot
/// accommodate the estimate; the caller (the routing retry loop) then cools
/// the deployment — for the budget window carried on the error — and tries
/// the next one. When neither budget is configured this is a no-op
/// (existing behavior preserved).
///
/// Every metered call site, including native JSON/SSE relays, must settle or
/// release the returned reservation.
pub async fn reserve_router_budgets(
    state: &crate::state::AppState,
    deployment: &oxllm_core::deployment::Deployment,
    dep_key: &str,
    call_ctx: &CallContext,
) -> Result<RouterBudgetReservation, RouterBudgetError> {
    let dep_limit = deployment.oxllm_params.max_budget;
    let provider_slug = deployment.provider_slug();
    let provider_budget = state
        .router
        .config()
        .provider_budget_config
        .as_ref()
        .and_then(|cfg| cfg.get(provider_slug));
    if dep_limit.is_none() && provider_budget.is_none() {
        return Ok(RouterBudgetReservation::default());
    }

    // Same estimate the key-level BudgetHook reserves: reuse the value it
    // stashed on the context when it ran, otherwise compute it here.
    let estimate = match call_ctx.budget_estimate {
        Some(est) => est,
        None => estimate_max_request_cost(call_ctx).await,
    };

    let mut guard = RouterBudgetGuard::new(state.cache.clone(), RouterBudgetReservation::default());
    if let Some(limit) = dep_limit {
        let check = RouterScopeCheck {
            key: deployment_spend_key(dep_key),
            scope: BudgetScope::Deployment,
            id: dep_key,
            limit,
            ttl: Some(deployment_budget_ttl(deployment)),
        };
        reserve_router_scope(state, &mut guard.reservation, check, estimate).await?;
    }
    if let Some(provider_budget) = provider_budget {
        let check = RouterScopeCheck {
            key: provider_spend_key(provider_slug),
            scope: BudgetScope::Provider,
            id: provider_slug,
            limit: provider_budget.budget_limit,
            ttl: Some(parse_budget_duration(&provider_budget.time_period)),
        };
        if let Err(err) = reserve_router_scope(state, &mut guard.reservation, check, estimate).await
        {
            // Roll back the deployment reservation made above (same
            // rollback pattern BudgetHook::pre_call uses for its scopes).
            guard.release().await;
            return Err(err);
        }
    }
    Ok(guard.into_reservation())
}

/// Snapshot the current router-level spend counters: one entry per
/// deployment carrying a `max_budget` (`spend:dep:{deployment_key}`) and one
/// per provider with a configured budget (`spend:provider:{slug}`).
///
/// All deployments are scanned (not just `model`'s candidates) so the
/// snapshot stays correct when routing expands the candidate pool beyond
/// the requested model's group (e.g. the complexity router's tier pools).
///
/// The result feeds [`oxllm_core::router::PreCallData::spend_snapshot`] for
/// the `budget_limiter` pre-call check, so the check can filter
/// over-budget deployments synchronously. This is an advisory snapshot, not
/// a transaction: concurrent reservations/settlements can change counters
/// while it is read. Pick-time atomic reservations enforce the caps. Missing
/// counters read as zero spend. At most eight cache reads are in flight.
pub async fn snapshot_router_spends(
    state: &crate::state::AppState,
) -> std::sync::Arc<std::collections::HashMap<String, f64>> {
    let mut keys = std::collections::HashSet::new();
    for deployment in &state.deployments {
        if deployment.oxllm_params.max_budget.is_some() {
            keys.insert(deployment_spend_key(&deployment_key(deployment)));
        }
    }
    if let Some(provider_budgets) = &state.router.config().provider_budget_config {
        for slug in provider_budgets.keys() {
            keys.insert(provider_spend_key(slug));
        }
    }
    let snapshot = stream::iter(keys)
        .map(|key| async move {
            let spend = state.cache.counter_value(&key).await.unwrap_or(0.0);
            (key, spend)
        })
        .buffer_unordered(SPEND_SNAPSHOT_CONCURRENCY)
        .collect::<std::collections::HashMap<_, _>>()
        .await;
    std::sync::Arc::new(snapshot)
}

/// Record the router-level spend counters after a call with the computed
/// `cost`, for call sites that made **no** reservation: the cost is applied
/// as a plain increment. Callers that reserved via
/// [`reserve_router_budgets`] (the chat orchestration loop, the embeddings
/// handler) must use [`settle_router_spend`] instead. Currently only
/// exercised by tests — every production call site reserves.
///
/// See [`settle_router_spend`] for the counter/TTL/unknown-cost semantics.
#[allow(dead_code)]
pub async fn record_router_spend(
    state: &crate::state::AppState,
    deployment: &oxllm_core::deployment::Deployment,
    dep_key: &str,
    cost: Option<f64>,
) {
    settle_router_spend(
        state,
        deployment,
        dep_key,
        cost,
        &RouterBudgetReservation::default(),
    )
    .await;
}

/// Settle the router-level spend counters after a call with the computed
/// `cost`:
///
/// - `spend:dep:{deployment_key}` when the deployment carries a
///   `max_budget` (TTL: its `budget_duration`, default 1 day).
/// - `spend:provider:{provider_slug}` when the provider has a configured
///   budget (TTL: the budget's `time_period`).
///
/// Each counter is adjusted by `cost − reserved_estimate`, where the
/// reserved estimate comes from the [`RouterBudgetReservation`] made by
/// [`reserve_router_budgets`] at pick time. A call site that made no
/// reservation passes [`RouterBudgetReservation::default`], which degrades
/// gracefully to a plain increment of `cost`.
///
/// Counters only exist for scopes with a configured budget, so proxies
/// without router budgets pay no bookkeeping cost. When a budget IS
/// configured but the cost is unknown or negative (`None` / `< 0.0` — e.g.
/// the model is missing from the pricing table), a nominal
/// [`NOMINAL_UNKNOWN_COST`] increment is recorded instead and a warning is
/// logged: a configured cap that never accrues spend silently never
/// engages, which almost always means the pricing table needs attention.
pub async fn settle_router_spend(
    state: &crate::state::AppState,
    deployment: &oxllm_core::deployment::Deployment,
    dep_key: &str,
    cost: Option<f64>,
    reservation: &RouterBudgetReservation,
) {
    let provider_slug = deployment.provider_slug();
    let provider_budget = state
        .router
        .config()
        .provider_budget_config
        .as_ref()
        .and_then(|cfg| cfg.get(provider_slug));

    let cost = match cost {
        // Some(0.0) is a legitimate zero cost (zero-priced model, fully
        // cached response): settle as-is so any pick-time reservation is
        // refunded, but never accrue the nominal unknown-cost increment.
        Some(c) if c >= 0.0 => c,
        _ => {
            if deployment.oxllm_params.max_budget.is_some() || provider_budget.is_some() {
                tracing::warn!(
                    event = "budget_enforcement_skipped",
                    deployment = %dep_key,
                    provider = %provider_slug,
                    model = %deployment.oxllm_params.model,
                    nominal_cost = NOMINAL_UNKNOWN_COST,
                    "no cost data for model '{}' — recording nominal cost so the configured budget still advances; check pricing table",
                    deployment.oxllm_params.model
                );
                NOMINAL_UNKNOWN_COST
            } else {
                return;
            }
        }
    };

    if deployment.oxllm_params.max_budget.is_some() {
        let key = deployment_spend_key(dep_key);
        let reserved = reservation.estimate_for(&key).unwrap_or(0.0);
        let ttl = Some(deployment_budget_ttl(deployment));
        if let Err(e) = settle_counter(
            &state.cache,
            &key,
            cost - reserved,
            ttl,
            reservation.window_for(&key),
        )
        .await
        {
            tracing::warn!(error = %e, "deployment spend record failed");
        }
    }

    if let Some(provider_budget) = provider_budget {
        let key = provider_spend_key(provider_slug);
        let reserved = reservation.estimate_for(&key).unwrap_or(0.0);
        let ttl = Some(parse_budget_duration(&provider_budget.time_period));
        if let Err(e) = settle_counter(
            &state.cache,
            &key,
            cost - reserved,
            ttl,
            reservation.window_for(&key),
        )
        .await
        {
            tracing::warn!(error = %e, "provider spend record failed");
        }
    }
}

/// Apply a settle delta (`actual − reserved`) in its original admission window.
/// Both signs are skipped if that window expired/replaced; the durable actual
/// ledger is independent. None is only for unreserved/legacy synthetic calls.
async fn settle_counter(
    cache: &Arc<dyn Cache>,
    key: &str,
    delta: f64,
    ttl: Option<Duration>,
    window: Option<&CounterWindow>,
) -> Result<(), oxllm_core::Error> {
    if let Some(window) = window {
        return cache.incr_in_window(key, delta, window).await.map(|_| ());
    }
    if delta > 0.0 {
        cache.incr(key, delta, ttl).await.map(|_| ())
    } else if delta < 0.0 {
        cache.decr_if_exists(key, -delta).await.map(|_| ())
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::types::{GeneralSettings, OxllmSettings, ProxyConfig, RouterSettings};
    use crate::state::AppState;
    use oxllm_core::IncrResult;
    use oxllm_core::deployment::{Deployment, OxllmParams};

    fn make_ctx() -> CallContext {
        CallContext::for_test("gpt-4o")
    }

    fn router_budget_state(
        provider_budget_config: Option<
            std::collections::HashMap<String, oxllm_core::router::ProviderBudgetConfig>,
        >,
    ) -> AppState {
        let config = ProxyConfig {
            model_list: vec![],
            oxllm_settings: OxllmSettings::default(),
            general_settings: GeneralSettings::default(),
            router_settings: RouterSettings {
                provider_budget_config,
                ..Default::default()
            },
            extra: serde_json::Map::new(),
        };
        AppState::with_registry(config, oxllm_providers::default_registry()).expect("state built")
    }

    fn deployment_with_budget(
        max_budget: Option<f64>,
        budget_duration: Option<&str>,
    ) -> Deployment {
        Deployment {
            model_name: "gpt-4o".to_string(),
            oxllm_params: OxllmParams {
                model: "openai/gpt-4o".to_string(),
                max_budget,
                budget_duration: budget_duration.map(str::to_string),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    const DEP_KEY: &str = "gpt-4o:openai/gpt-4o";

    #[tokio::test]
    async fn cancelled_router_budget_refund_finishes_once_across_scopes() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use tokio::sync::Notify;

        #[derive(Default)]
        struct PausedRefundCache {
            inner: crate::cache::InMemoryCache,
            entered: Notify,
            resume: Notify,
            decrements: AtomicUsize,
        }

        #[async_trait::async_trait]
        impl Cache for PausedRefundCache {
            async fn get(&self, key: &str) -> Option<String> {
                self.inner.get(key).await
            }
            async fn set(&self, key: &str, value: String, ttl: Option<Duration>) {
                self.inner.set(key, value, ttl).await;
            }
            async fn delete(&self, key: &str) {
                self.inner.delete(key).await;
            }
            async fn incr(
                &self,
                key: &str,
                by: f64,
                ttl: Option<Duration>,
            ) -> oxllm_core::Result<f64> {
                self.inner.incr(key, by, ttl).await
            }
            async fn incr_if_below(
                &self,
                key: &str,
                by: f64,
                ceiling: f64,
                ttl: Option<Duration>,
            ) -> oxllm_core::Result<IncrResult> {
                self.inner.incr_if_below(key, by, ceiling, ttl).await
            }
            async fn counter_value(&self, key: &str) -> Option<f64> {
                self.inner.counter_value(key).await
            }
            async fn decr_if_exists(&self, key: &str, by: f64) -> oxllm_core::Result<bool> {
                let result = self.inner.decr_if_exists(key, by).await?;
                if self.decrements.fetch_add(1, Ordering::SeqCst) == 0 {
                    self.entered.notify_one();
                    self.resume.notified().await;
                }
                Ok(result)
            }

            async fn incr_if_below_in_window(
                &self,
                key: &str,
                delta: f64,
                ceiling: f64,
                ttl: Option<Duration>,
                expected: Option<&CounterWindow>,
            ) -> oxllm_core::Result<WindowedIncrResult> {
                self.inner
                    .incr_if_below_in_window(key, delta, ceiling, ttl, expected)
                    .await
            }

            async fn incr_in_window(
                &self,
                key: &str,
                delta: f64,
                window: &CounterWindow,
            ) -> oxllm_core::Result<bool> {
                let applied = self.inner.incr_in_window(key, delta, window).await?;
                if applied && delta < 0.0 && self.decrements.fetch_add(1, Ordering::SeqCst) == 0 {
                    self.entered.notify_one();
                    self.resume.notified().await;
                }
                Ok(applied)
            }
        }

        let cache = Arc::new(PausedRefundCache::default());
        let mut state = router_budget_state(Some(std::collections::HashMap::from([(
            "openai".into(),
            oxllm_core::router::ProviderBudgetConfig {
                budget_limit: 10.0,
                time_period: "1d".into(),
            },
        )])));
        state.cache = cache.clone();
        let deployment = deployment_with_budget(Some(10.0), None);
        let keys = [deployment_spend_key(DEP_KEY), provider_spend_key("openai")];
        for key in &keys {
            state.cache.incr(key, 0.25, None).await.unwrap();
        }
        let mut ctx = make_ctx();
        ctx.budget_estimate = Some(0.5);
        let reservation = reserve_router_budgets(&state, &deployment, DEP_KEY, &ctx)
            .await
            .unwrap();
        let mut guard = RouterBudgetGuard::new(state.cache.clone(), reservation);
        let task = tokio::spawn(async move { guard.release().await });
        tokio::time::timeout(Duration::from_secs(5), cache.entered.notified())
            .await
            .unwrap();
        assert_eq!(state.cache.counter_value(&keys[0]).await, Some(0.25));
        assert_eq!(state.cache.counter_value(&keys[1]).await, Some(0.75));
        task.abort();
        assert!(task.await.unwrap_err().is_cancelled());
        cache.resume.notify_one();
        tokio::time::timeout(Duration::from_secs(5), async {
            while state.cache.counter_value(&keys[1]).await != Some(0.25) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert_eq!(state.cache.counter_value(&keys[0]).await, Some(0.25));
        assert_eq!(
            cache.decrements.load(Ordering::SeqCst),
            2,
            "each acquired scope must be refunded once"
        );
    }

    #[tokio::test]
    async fn spend_snapshot_reads_all_scopes_with_bounded_concurrency() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct ObservedCache {
            values: std::collections::HashMap<String, f64>,
            reads: AtomicUsize,
            active: AtomicUsize,
            peak: AtomicUsize,
        }
        #[async_trait::async_trait]
        impl Cache for ObservedCache {
            async fn get(&self, _: &str) -> Option<String> {
                unreachable!("snapshot should use counter_value")
            }
            async fn set(&self, _: &str, _: String, _: Option<Duration>) {
                unreachable!("snapshot must be read-only")
            }
            async fn delete(&self, _: &str) {
                unreachable!("snapshot must be read-only")
            }
            async fn incr(
                &self,
                _: &str,
                _: f64,
                _: Option<Duration>,
            ) -> Result<f64, oxllm_core::Error> {
                unreachable!("snapshot must be read-only")
            }
            async fn counter_value(&self, key: &str) -> Option<f64> {
                self.reads.fetch_add(1, Ordering::SeqCst);
                let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
                self.peak.fetch_max(active, Ordering::SeqCst);
                // Yield once so overlapping reads are observable without
                // sleeps or assumptions about real cache/network latency.
                tokio::task::yield_now().await;
                self.active.fetch_sub(1, Ordering::SeqCst);
                self.values.get(key).copied()
            }
        }

        let mut state = router_budget_state(Some(std::collections::HashMap::from([(
            "openai".into(),
            oxllm_core::router::ProviderBudgetConfig {
                budget_limit: 100.0,
                time_period: "1d".into(),
            },
        )])));
        state.deployments = (0..20)
            .map(|i| {
                let mut dep = deployment_with_budget(Some(10.0), None);
                dep.model_name = format!("snapshot-{i}");
                dep
            })
            .collect();
        let mut expected = std::collections::HashMap::new();
        for (i, dep) in state.deployments.iter().enumerate() {
            expected.insert(deployment_spend_key(&deployment_key(dep)), i as f64);
        }
        expected.insert(provider_spend_key("openai"), 42.0);
        let mut values = expected.clone();
        // Genuine cache absence still contributes a zero entry.
        values.remove(&deployment_spend_key(&deployment_key(
            &state.deployments[0],
        )));
        let cache = Arc::new(ObservedCache {
            values,
            reads: AtomicUsize::new(0),
            active: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
        });
        state.cache = cache.clone();
        // Identical budget keys must not cause repeated reads.
        state.deployments.push(state.deployments[0].clone());
        state.deployments.push(deployment_with_budget(None, None));
        let snapshot = snapshot_router_spends(&state).await;
        assert_eq!(*snapshot, expected);
        assert_eq!(cache.reads.load(Ordering::SeqCst), expected.len());
        let peak = cache.peak.load(Ordering::SeqCst);
        assert!(peak > 1, "cache reads must overlap");
        assert!(
            peak <= SPEND_SNAPSHOT_CONCURRENCY,
            "cache fan-out must stay bounded"
        );
        assert_eq!(cache.active.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn router_budget_no_budgets_configured_passes() {
        let state = router_budget_state(None);
        let dep = deployment_with_budget(None, None);
        let reservation = reserve_router_budgets(&state, &dep, DEP_KEY, &make_ctx())
            .await
            .expect("no budgets configured must pass");
        assert!(
            reservation.entries.is_empty(),
            "no budgets configured → nothing reserved"
        );
    }

    #[tokio::test]
    async fn router_budget_zero_cost_admission_does_not_hide_counter_outages() {
        struct FailingCounterCache;
        #[async_trait::async_trait]
        impl Cache for FailingCounterCache {
            async fn get(&self, _key: &str) -> Option<String> {
                None
            }
            async fn set(&self, _key: &str, _value: String, _ttl: Option<Duration>) {}
            async fn delete(&self, _key: &str) {}
            async fn incr(
                &self,
                _key: &str,
                _by: f64,
                _ttl: Option<Duration>,
            ) -> Result<f64, oxllm_core::Error> {
                Err(oxllm_core::Error::Network("counter unavailable".into()))
            }
        }
        let mut state = router_budget_state(None);
        state.cache = Arc::new(FailingCounterCache);
        let deployment = deployment_with_budget(Some(10.0), None);
        for estimate in [0.0, 0.5] {
            let mut ctx = make_ctx();
            ctx.budget_estimate = Some(estimate);
            let error = reserve_router_budgets(&state, &deployment, DEP_KEY, &ctx)
                .await
                .unwrap_err();
            assert_eq!(error.api_error.code.as_deref(), Some("budget_exceeded"));
        }
    }

    #[tokio::test]
    async fn router_budget_deployment_under_limit_passes() {
        let state = router_budget_state(None);
        let dep = deployment_with_budget(Some(10.0), None);
        let _ = state
            .cache
            .incr("spend:dep:gpt-4o:openai/gpt-4o", 5.0, None)
            .await;
        assert!(
            reserve_router_budgets(&state, &dep, DEP_KEY, &make_ctx())
                .await
                .is_ok()
        );
    }

    #[tokio::test]
    async fn router_budget_deployment_at_limit_rejects_429() {
        let state = router_budget_state(None);
        let dep = deployment_with_budget(Some(10.0), None);
        let _ = state
            .cache
            .incr("spend:dep:gpt-4o:openai/gpt-4o", 10.0, None)
            .await;
        let err = reserve_router_budgets(&state, &dep, DEP_KEY, &make_ctx())
            .await
            .expect_err("at-cap deployment budget must reject");
        assert_eq!(
            err.api_error.status,
            axum::http::StatusCode::TOO_MANY_REQUESTS
        );
    }

    #[tokio::test]
    async fn router_budget_provider_at_limit_rejects() {
        let budgets = std::collections::HashMap::from([(
            "openai".to_string(),
            oxllm_core::router::ProviderBudgetConfig {
                budget_limit: 100.0,
                time_period: "1d".to_string(),
            },
        )]);
        let state = router_budget_state(Some(budgets));
        let dep = deployment_with_budget(None, None);
        let _ = state.cache.incr("spend:provider:openai", 100.0, None).await;
        let err = reserve_router_budgets(&state, &dep, DEP_KEY, &make_ctx())
            .await
            .expect_err("at-cap provider budget must reject");
        assert_eq!(
            err.api_error.status,
            axum::http::StatusCode::TOO_MANY_REQUESTS
        );
    }

    #[tokio::test]
    async fn router_budget_provider_under_limit_passes() {
        let budgets = std::collections::HashMap::from([(
            "openai".to_string(),
            oxllm_core::router::ProviderBudgetConfig {
                budget_limit: 100.0,
                time_period: "1d".to_string(),
            },
        )]);
        let state = router_budget_state(Some(budgets));
        let dep = deployment_with_budget(None, None);
        let _ = state.cache.incr("spend:provider:openai", 99.0, None).await;
        assert!(
            reserve_router_budgets(&state, &dep, DEP_KEY, &make_ctx())
                .await
                .is_ok()
        );
    }

    #[tokio::test]
    async fn router_budget_other_provider_budget_does_not_apply() {
        // A budget configured for "anthropic" must not gate an openai
        // deployment even when the anthropic counter is over the cap.
        let budgets = std::collections::HashMap::from([(
            "anthropic".to_string(),
            oxllm_core::router::ProviderBudgetConfig {
                budget_limit: 1.0,
                time_period: "1d".to_string(),
            },
        )]);
        let state = router_budget_state(Some(budgets));
        let dep = deployment_with_budget(None, None);
        let _ = state
            .cache
            .incr("spend:provider:anthropic", 500.0, None)
            .await;
        assert!(
            reserve_router_budgets(&state, &dep, DEP_KEY, &make_ctx())
                .await
                .is_ok()
        );
    }

    #[tokio::test]
    async fn router_spend_records_deployment_and_provider_counters() {
        let budgets = std::collections::HashMap::from([(
            "openai".to_string(),
            oxllm_core::router::ProviderBudgetConfig {
                budget_limit: 100.0,
                time_period: "1d".to_string(),
            },
        )]);
        let state = router_budget_state(Some(budgets));
        let dep = deployment_with_budget(Some(10.0), None);
        record_router_spend(&state, &dep, DEP_KEY, Some(0.5)).await;
        record_router_spend(&state, &dep, DEP_KEY, Some(0.25)).await;
        let dep_spend = state
            .cache
            .counter_value("spend:dep:gpt-4o:openai/gpt-4o")
            .await
            .unwrap_or(0.0);
        let provider_spend = state
            .cache
            .counter_value("spend:provider:openai")
            .await
            .unwrap_or(0.0);
        assert!((dep_spend - 0.75).abs() < 1e-9, "dep spend: {dep_spend}");
        assert!(
            (provider_spend - 0.75).abs() < 1e-9,
            "provider spend: {provider_spend}"
        );
    }

    #[tokio::test]
    async fn router_spend_no_budgets_creates_no_counters() {
        let state = router_budget_state(None);
        let dep = deployment_with_budget(None, None);
        record_router_spend(&state, &dep, DEP_KEY, Some(1.0)).await;
        assert!(
            state
                .cache
                .counter_value("spend:dep:gpt-4o:openai/gpt-4o")
                .await
                .is_none(),
            "without a configured budget no deployment counter may be created"
        );
        assert!(
            state
                .cache
                .counter_value("spend:provider:openai")
                .await
                .is_none()
        );
    }

    #[tokio::test]
    async fn router_spend_zero_cost_records_no_nominal_spend() {
        // A genuine zero cost (zero-priced model, fully cached response) is
        // NOT unknown cost data: it must settle as 0 so no phantom spend
        // accrues against a configured budget.
        let state = router_budget_state(None);
        let dep = deployment_with_budget(Some(10.0), None);
        record_router_spend(&state, &dep, DEP_KEY, Some(0.0)).await;
        let spend = state
            .cache
            .counter_value("spend:dep:gpt-4o:openai/gpt-4o")
            .await
            .unwrap_or(0.0);
        assert!(
            spend.abs() < 1e-12,
            "zero-cost call must record no phantom spend, got {spend}"
        );
    }

    #[tokio::test]
    async fn router_spend_zero_cost_refunds_reservation() {
        // A genuine zero cost must still settle against a pick-time
        // reservation, refunding the estimated delta.
        let state = router_budget_state(None);
        let dep = deployment_with_budget(Some(10.0), None);
        let key = deployment_spend_key(DEP_KEY);
        let _ = state.cache.incr(&key, 1.0, None).await; // prior reservation
        let reservation = RouterBudgetReservation {
            entries: vec![RouterBudgetReservationEntry {
                key: key.clone(),
                estimate: 1.0,
                window: None,
            }],
        };
        settle_router_spend(&state, &dep, DEP_KEY, Some(0.0), &reservation).await;
        let spend = state.cache.counter_value(&key).await.unwrap_or(0.0);
        assert!(
            spend.abs() < 1e-12,
            "zero-cost settle must refund the reservation, got {spend}"
        );
    }

    #[tokio::test]
    async fn router_spend_unknown_cost_records_nominal_when_budget_configured() {
        // An unknown cost (model missing from the pricing table) must
        // advance the counter by the nominal amount so the configured
        // budget still enforces; the warning log makes the missing pricing
        // observable.
        let state = router_budget_state(None);
        let dep = deployment_with_budget(Some(10.0), None);
        record_router_spend(&state, &dep, DEP_KEY, None).await;
        let spend = state
            .cache
            .counter_value("spend:dep:gpt-4o:openai/gpt-4o")
            .await
            .unwrap_or(0.0);
        assert!(
            (spend - NOMINAL_UNKNOWN_COST).abs() < 1e-12,
            "unknown-cost call with a configured budget must record the nominal cost, got {spend}"
        );
    }

    #[tokio::test]
    async fn router_spend_unknown_cost_without_budget_is_noop() {
        let state = router_budget_state(None);
        let dep = deployment_with_budget(None, None);
        record_router_spend(&state, &dep, DEP_KEY, None).await;
        assert!(
            state
                .cache
                .counter_value("spend:dep:gpt-4o:openai/gpt-4o")
                .await
                .is_none(),
            "unknown cost without a configured budget must not create counters"
        );
        assert!(
            state
                .cache
                .counter_value("spend:provider:openai")
                .await
                .is_none()
        );
    }

    #[tokio::test]
    async fn router_budget_window_resets_after_duration() {
        let state = router_budget_state(None);
        let dep = deployment_with_budget(Some(1.0), Some("1ms"));
        record_router_spend(&state, &dep, DEP_KEY, Some(1.5)).await;
        assert!(
            reserve_router_budgets(&state, &dep, DEP_KEY, &make_ctx())
                .await
                .is_err(),
            "over-cap deployment must be rejected within the window"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
        assert!(
            reserve_router_budgets(&state, &dep, DEP_KEY, &make_ctx())
                .await
                .is_ok(),
            "after the budget window expires the counter resets and the deployment passes"
        );
    }

    /// MEDIUM (C11): releasing (or negatively settling) a reservation after
    /// its budget window expired must be a no-op — the old behavior minted a
    /// fresh **negative, never-expiring** counter that permanently
    /// discounted every subsequent window's spend.
    #[tokio::test]
    async fn router_budget_release_after_window_expiry_mints_no_residue() {
        let state = router_budget_state(None);
        let dep = deployment_with_budget(Some(10.0), Some("5ms"));
        let mut ctx = make_ctx();
        ctx.budget_estimate = Some(0.5);
        let reservation = reserve_router_budgets(&state, &dep, DEP_KEY, &ctx)
            .await
            .expect("reserved");
        tokio::time::sleep(Duration::from_millis(10)).await;
        reservation.release(&state.cache).await;
        assert!(
            state
                .cache
                .counter_value("spend:dep:gpt-4o:openai/gpt-4o")
                .await
                .is_none(),
            "release after window expiry must not mint a negative residue counter"
        );
        // The fresh window starts clean: a new reservation lands at its
        // estimate, not at a negative discount.
        let reservation = reserve_router_budgets(&state, &dep, DEP_KEY, &ctx)
            .await
            .expect("fresh window reserves cleanly");
        let spend = state
            .cache
            .counter_value("spend:dep:gpt-4o:openai/gpt-4o")
            .await
            .unwrap_or(0.0);
        assert!(
            (spend - 0.5).abs() < 1e-9,
            "fresh window must start at the new reservation (0.5), got {spend}"
        );
        reservation.release(&state.cache).await;
    }

    /// Same hazard on the settle path: `cost − reserved < 0` landing on an
    /// expired counter must not create negative credit.
    #[tokio::test]
    async fn router_budget_negative_settle_after_window_expiry_mints_no_residue() {
        let state = router_budget_state(None);
        let dep = deployment_with_budget(Some(10.0), Some("5ms"));
        let mut ctx = make_ctx();
        ctx.budget_estimate = Some(0.5);
        let reservation = reserve_router_budgets(&state, &dep, DEP_KEY, &ctx)
            .await
            .expect("reserved");
        tokio::time::sleep(Duration::from_millis(10)).await;
        settle_router_spend(&state, &dep, DEP_KEY, Some(0.1), &reservation).await;
        assert!(
            state
                .cache
                .counter_value("spend:dep:gpt-4o:openai/gpt-4o")
                .await
                .is_none(),
            "negative settle after window expiry must not mint a negative residue counter"
        );
    }

    #[tokio::test]
    async fn router_budget_rejection_carries_budget_exceeded_code() {
        // Budget-exceeded 429s must be distinguishable from client rate
        // limiting (`proxy_rate_limited`) and provider 429s
        // (`upstream_rate_limited`).
        let state = router_budget_state(None);
        let dep = deployment_with_budget(Some(10.0), None);
        let _ = state
            .cache
            .incr("spend:dep:gpt-4o:openai/gpt-4o", 10.0, None)
            .await;
        let err = reserve_router_budgets(&state, &dep, DEP_KEY, &make_ctx())
            .await
            .expect_err("at-cap deployment budget must reject");
        assert_eq!(
            err.api_error.status,
            axum::http::StatusCode::TOO_MANY_REQUESTS
        );
        assert_eq!(err.api_error.code.as_deref(), Some("budget_exceeded"));
        assert_eq!(err.api_error.error_type, "rate_limit_error");
    }

    #[tokio::test]
    async fn router_budget_reservation_uses_stashed_key_estimate() {
        // When BudgetHook already stashed an estimate on the context, the
        // router check must reserve exactly that amount (not recompute).
        let state = router_budget_state(None);
        let dep = deployment_with_budget(Some(10.0), None);
        let mut ctx = make_ctx();
        ctx.budget_estimate = Some(0.5);
        let reservation = reserve_router_budgets(&state, &dep, DEP_KEY, &ctx)
            .await
            .expect("under cap");
        let spend = state
            .cache
            .counter_value("spend:dep:gpt-4o:openai/gpt-4o")
            .await
            .unwrap_or(0.0);
        assert!(
            (spend - 0.5).abs() < 1e-9,
            "the stashed estimate must be reserved, got {spend}"
        );
        assert_eq!(
            reservation.estimate_for("spend:dep:gpt-4o:openai/gpt-4o"),
            Some(0.5)
        );
    }

    #[tokio::test]
    async fn router_budget_reservation_released_on_failure() {
        // Upstream failure: releasing the reservation returns the counter
        // to its pre-reservation value so failed requests do not
        // permanently consume router budget.
        let state = router_budget_state(None);
        let dep = deployment_with_budget(Some(10.0), None);
        let _ = state
            .cache
            .incr("spend:dep:gpt-4o:openai/gpt-4o", 1.0, None)
            .await;
        let mut ctx = make_ctx();
        ctx.budget_estimate = Some(0.5);
        let reservation = reserve_router_budgets(&state, &dep, DEP_KEY, &ctx)
            .await
            .expect("under cap");
        let reserved = state
            .cache
            .counter_value("spend:dep:gpt-4o:openai/gpt-4o")
            .await
            .unwrap_or(0.0);
        assert!((reserved - 1.5).abs() < 1e-9, "reserved: {reserved}");
        reservation.release(&state.cache).await;
        let spend = state
            .cache
            .counter_value("spend:dep:gpt-4o:openai/gpt-4o")
            .await
            .unwrap_or(0.0);
        assert!(
            (spend - 1.0).abs() < 1e-9,
            "release must return the counter to its pre-reservation value, got {spend}"
        );
    }

    #[tokio::test]
    async fn router_budget_reservation_settles_to_actual_cost() {
        // Success: record_router_spend adjusts the counter by
        // (actual − estimate), leaving the actual cost behind.
        let budgets = std::collections::HashMap::from([(
            "openai".to_string(),
            oxllm_core::router::ProviderBudgetConfig {
                budget_limit: 100.0,
                time_period: "1d".to_string(),
            },
        )]);
        let state = router_budget_state(Some(budgets));
        let dep = deployment_with_budget(Some(10.0), None);
        let mut ctx = make_ctx();
        ctx.budget_estimate = Some(0.5);
        let reservation = reserve_router_budgets(&state, &dep, DEP_KEY, &ctx)
            .await
            .expect("under cap");
        let reserved = state
            .cache
            .counter_value("spend:dep:gpt-4o:openai/gpt-4o")
            .await
            .unwrap_or(0.0);
        assert!((reserved - 0.5).abs() < 1e-9, "reserved: {reserved}");
        settle_router_spend(&state, &dep, DEP_KEY, Some(0.125), &reservation).await;
        let dep_spend = state
            .cache
            .counter_value("spend:dep:gpt-4o:openai/gpt-4o")
            .await
            .unwrap_or(0.0);
        let provider_spend = state
            .cache
            .counter_value("spend:provider:openai")
            .await
            .unwrap_or(0.0);
        assert!(
            (dep_spend - 0.125).abs() < 1e-9,
            "deployment counter must settle to the actual cost, got {dep_spend}"
        );
        assert!(
            (provider_spend - 0.125).abs() < 1e-9,
            "provider counter must settle to the actual cost, got {provider_spend}"
        );
    }

    #[tokio::test]
    async fn router_budget_concurrent_reservations_never_exceed_cap() {
        // N concurrent checks must not collectively reserve past the cap:
        // the atomic incr_if_below bounds total in-flight spend.
        let state = std::sync::Arc::new(router_budget_state(None));
        let dep = deployment_with_budget(Some(1.0), None);
        let mut handles = Vec::new();
        for _ in 0..16 {
            let state = state.clone();
            let dep = dep.clone();
            handles.push(tokio::spawn(async move {
                let mut ctx = make_ctx();
                ctx.budget_estimate = Some(0.25);
                reserve_router_budgets(&state, &dep, DEP_KEY, &ctx).await
            }));
        }
        let mut allowed = 0u32;
        for h in handles {
            if h.await.expect("join").is_ok() {
                allowed += 1;
            }
        }
        let spend = state
            .cache
            .counter_value("spend:dep:gpt-4o:openai/gpt-4o")
            .await
            .unwrap_or(0.0);
        assert!(
            spend <= 1.0 + 1e-9,
            "atomic reservations must not exceed the ceiling: spend={spend}, allowed={allowed}"
        );
        assert_eq!(allowed, 4, "exactly four 0.25 reservations fit under 1.0");
    }

    #[tokio::test]
    async fn router_budget_provider_rejection_rolls_back_deployment_reservation() {
        // When the provider reservation fails after the deployment
        // reservation succeeded, the deployment reservation must be
        // released (BudgetHook reserved_keys rollback pattern).
        let budgets = std::collections::HashMap::from([(
            "openai".to_string(),
            oxllm_core::router::ProviderBudgetConfig {
                budget_limit: 1.0,
                time_period: "1d".to_string(),
            },
        )]);
        let state = router_budget_state(Some(budgets));
        let dep = deployment_with_budget(Some(10.0), None);
        // Provider is at cap; deployment has room.
        let _ = state.cache.incr("spend:provider:openai", 1.0, None).await;
        let mut ctx = make_ctx();
        ctx.budget_estimate = Some(0.5);
        let err = reserve_router_budgets(&state, &dep, DEP_KEY, &ctx)
            .await
            .expect_err("provider at cap must reject");
        assert_eq!(err.api_error.code.as_deref(), Some("budget_exceeded"));
        let dep_spend = state
            .cache
            .counter_value("spend:dep:gpt-4o:openai/gpt-4o")
            .await
            .unwrap_or(0.0);
        assert!(
            dep_spend.abs() < 1e-9,
            "deployment reservation must be rolled back when the provider rejects, got {dep_spend}"
        );
    }

    #[tokio::test]
    async fn router_budget_zero_estimate_falls_back_to_read_only_check() {
        // A zero estimate reserves nothing but must still reject an
        // already-exhausted cap.
        let state = router_budget_state(None);
        let dep = deployment_with_budget(Some(10.0), None);
        let mut ctx = make_ctx();
        ctx.budget_estimate = Some(0.0);
        let reservation = reserve_router_budgets(&state, &dep, DEP_KEY, &ctx)
            .await
            .expect("under cap passes with zero estimate");
        let spend = state
            .cache
            .counter_value("spend:dep:gpt-4o:openai/gpt-4o")
            .await
            .unwrap_or(0.0);
        assert!(spend.abs() < 1e-9, "zero estimate reserves nothing");
        // Settlement still accrues the actual cost (reserved = 0).
        settle_router_spend(&state, &dep, DEP_KEY, Some(0.25), &reservation).await;
        let spend = state
            .cache
            .counter_value("spend:dep:gpt-4o:openai/gpt-4o")
            .await
            .unwrap_or(0.0);
        assert!((spend - 0.25).abs() < 1e-9, "settled spend: {spend}");
        // Now exhaust the cap: the zero-estimate check must reject.
        let _ = state
            .cache
            .incr("spend:dep:gpt-4o:openai/gpt-4o", 10.0, None)
            .await;
        assert!(
            reserve_router_budgets(&state, &dep, DEP_KEY, &ctx)
                .await
                .is_err(),
            "zero estimate must still reject an exhausted cap"
        );
    }

    #[test]
    fn evaluate_router_budget_decisions() {
        // Nothing configured → allowed.
        assert_eq!(evaluate_router_budget(None, None), BudgetDecision::Allowed);
        // Under the cap → allowed.
        assert_eq!(
            evaluate_router_budget(Some(("dep", 9.99, 10.0)), None),
            BudgetDecision::Allowed
        );
        assert_eq!(
            evaluate_router_budget(None, Some(("openai", 99.0, 100.0))),
            BudgetDecision::Allowed
        );
        // At/over the cap → exceeded with the right scope.
        assert_eq!(
            evaluate_router_budget(Some(("dep", 10.0, 10.0)), None),
            BudgetDecision::Exceeded {
                scope: BudgetScope::Deployment,
                id: "dep".to_string(),
                spend: 10.0,
                limit: 10.0,
            }
        );
        assert_eq!(
            evaluate_router_budget(None, Some(("openai", 150.0, 100.0))),
            BudgetDecision::Exceeded {
                scope: BudgetScope::Provider,
                id: "openai".to_string(),
                spend: 150.0,
                limit: 100.0,
            }
        );
        // Deployment is evaluated before provider.
        assert_eq!(
            evaluate_router_budget(Some(("dep", 10.0, 10.0)), Some(("openai", 150.0, 100.0))),
            BudgetDecision::Exceeded {
                scope: BudgetScope::Deployment,
                id: "dep".to_string(),
                spend: 10.0,
                limit: 10.0,
            }
        );
    }

    /// C13/LOW: a non-finite spend or limit (NaN / ±∞ — reachable only via a
    /// corrupt cache row) must fail-closed to `Exceeded`. Without this guard
    /// `spend >= limit` evaluates `false` for NaN and grants an effectively
    /// infinite budget.
    #[test]
    fn evaluate_router_budget_fail_closed_on_non_finite() {
        // NaN spend on either scope → exceeded.
        assert!(matches!(
            evaluate_router_budget(Some(("dep", f64::NAN, 10.0)), None),
            BudgetDecision::Exceeded { .. }
        ));
        assert!(matches!(
            evaluate_router_budget(None, Some(("openai", f64::NAN, 100.0))),
            BudgetDecision::Exceeded { .. }
        ));
        // +∞ spend → exceeded.
        assert!(matches!(
            evaluate_router_budget(Some(("dep", f64::INFINITY, 10.0)), None),
            BudgetDecision::Exceeded { .. }
        ));
        // NaN limit → exceeded (corruption on the cap side too).
        assert!(matches!(
            evaluate_router_budget(Some(("dep", 1.0, f64::NAN)), None),
            BudgetDecision::Exceeded { .. }
        ));
        assert!(matches!(
            evaluate_router_budget(None, Some(("openai", 1.0, f64::NAN))),
            BudgetDecision::Exceeded { .. }
        ));
        // Finite spend/limit under the cap still allowed (regression guard).
        assert_eq!(
            evaluate_router_budget(Some(("dep", 1.0, 10.0)), None),
            BudgetDecision::Allowed
        );
    }

    // -----------------------------------------------------------------------
    // snapshot_router_spends (pre-call check spend snapshot)
    // -----------------------------------------------------------------------

    fn snapshot_state(
        provider_budget_config: Option<
            std::collections::HashMap<String, oxllm_core::router::ProviderBudgetConfig>,
        >,
        with_max_budget: bool,
    ) -> AppState {
        let model_list = if with_max_budget {
            vec![crate::config::types::ConfigModel {
                model_name: "gpt-4o".to_string(),
                oxllm_params: OxllmParams {
                    model: "openai/gpt-4o".to_string(),
                    max_budget: Some(10.0),
                    ..Default::default()
                },
                model_info: None,
                extra: serde_json::Map::new(),
            }]
        } else {
            vec![]
        };
        let config = ProxyConfig {
            model_list,
            oxllm_settings: OxllmSettings::default(),
            general_settings: GeneralSettings::default(),
            router_settings: RouterSettings {
                provider_budget_config,
                ..Default::default()
            },
            extra: serde_json::Map::new(),
        };
        AppState::with_registry(config, oxllm_providers::default_registry()).expect("state built")
    }

    #[tokio::test]
    async fn snapshot_router_spends_empty_when_no_budgets() {
        let state = snapshot_state(None, false);
        let snapshot = snapshot_router_spends(&state).await;
        assert!(
            snapshot.is_empty(),
            "no budgets configured → no counter reads, empty snapshot"
        );
    }

    #[tokio::test]
    async fn snapshot_router_spends_reads_deployment_and_provider_counters() {
        let budgets = std::collections::HashMap::from([(
            "openai".to_string(),
            oxllm_core::router::ProviderBudgetConfig {
                budget_limit: 100.0,
                time_period: "1d".to_string(),
            },
        )]);
        let state = snapshot_state(Some(budgets), true);
        let _ = state
            .cache
            .incr("spend:dep:gpt-4o:openai/gpt-4o", 7.5, None)
            .await;
        let _ = state.cache.incr("spend:provider:openai", 42.0, None).await;
        let snapshot = snapshot_router_spends(&state).await;
        assert_eq!(
            snapshot.get("spend:dep:gpt-4o:openai/gpt-4o"),
            Some(&7.5),
            "deployment spend must be snapshotted"
        );
        assert_eq!(
            snapshot.get("spend:provider:openai"),
            Some(&42.0),
            "provider spend must be snapshotted"
        );
    }

    #[tokio::test]
    async fn snapshot_router_spends_missing_counters_read_as_zero() {
        let budgets = std::collections::HashMap::from([(
            "openai".to_string(),
            oxllm_core::router::ProviderBudgetConfig {
                budget_limit: 100.0,
                time_period: "1d".to_string(),
            },
        )]);
        let state = snapshot_state(Some(budgets), true);
        let snapshot = snapshot_router_spends(&state).await;
        assert_eq!(
            snapshot.get("spend:dep:gpt-4o:openai/gpt-4o"),
            Some(&0.0),
            "missing deployment counter must read as zero spend (under any cap)"
        );
        assert_eq!(snapshot.get("spend:provider:openai"), Some(&0.0));
    }
}
