//! `BudgetHook` — enforces per-key/team/user `max_budget` over a rolling
//! `budget_duration` window.
//!
//! Exact cost cannot be known until the response arrives, so the hook
//! **atomically reserves** a conservative estimate via
//! [`Cache::incr_if_below`] in `pre_call`. Concurrent callers cannot
//! collectively exceed the ceiling. `post_success` adjusts the reservation
//! to the actual computed cost (`actual − estimate`); `post_failure` releases
//! an unsettled estimate. A partial session with known complete cost can use
//! `settle_budget` first, independently of its terminal failure notification.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use oxllm_core::{
    Cache, CallContext, CounterWindow, Error, Hook, HookAction, IncrResult, ResponseSummary,
    WindowedIncrResult,
};

use crate::hooks::spend_keys::{
    customer_spend_key, model_spend_key, org_spend_key, project_spend_key, spend_key,
    tag_spend_key, team_spend_key, user_spend_key,
};
use crate::scheduler::parse_budget_duration;

/// Budget rejections by scope (`key` / `team` / `org` / `project` / `user`
/// / `customer` / `tag` / `model` for the per-request hook, `deployment` /
/// `provider` for the router-level gates).
pub const BUDGET_EXCEEDED_TOTAL: &str = "oxllm_budget_exceeded_total";

/// Best-effort settlement (adjust/release) writes that the backing store
/// rejected (e.g. a Redis outage that began mid-request, after `pre_call`
/// reserved successfully while the store was up). Labeled by `op`
/// (`"adjust"` / `"release"`). The hook deliberately swallows such errors
/// (failing a completed request over a tracking write is worse), so this
/// counter is the only signal that spend tracking has degraded — an operator
/// cannot otherwise tell from logs or dashboards that the spend counter is
/// recording pre-call estimates instead of actual costs.
pub const BUDGET_SETTLE_FAILED_TOTAL: &str = "oxllm_budget_settle_failed_total";

/// Count a budget rejection and return the reject action. Centralized so
/// every `pre_call` rejection site is metered identically.
fn reject_over_budget(scope: &'static str, err: Error) -> HookAction {
    ::metrics::counter!(BUDGET_EXCEEDED_TOTAL, "scope" => scope).increment(1);
    HookAction::Reject(err)
}

/// Pick the most restrictive (lowest) of two optional budgets.
pub fn most_restrictive(a: Option<f64>, b: Option<f64>) -> Option<f64> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (Some(a), None) | (None, Some(a)) => Some(a),
        (None, None) => None,
    }
}

/// Estimate the worst-case cost of a single request for budget pre-checks.
///
/// # Estimate vs. settlement pricing
///
/// Key-level `pre_call` runs **before** the router picks a deployment, so
/// `ctx.deployment_pricing` is normally `None` at reservation time. The
/// estimate therefore uses the embedded price table
/// (`lookup_pricing(&ctx.model)`). After pick, orchestration stamps
/// `deployment_pricing` and calls
/// [`BudgetHook::top_up_for_deployment_pricing`] so marked-up deployments
/// reserve the delta against the same rate card that settlement uses.
///
/// When `ctx.deployment_pricing` is already present (e.g. a path that
/// re-runs estimate after pick), the reservation is sized on the override.
/// Unknown models (no override and no table entry) reserve a nominal `0.01`
/// floor plus any known Model Router infrastructure fee.
pub(crate) async fn estimate_max_request_cost(ctx: &CallContext) -> f64 {
    if ctx.pre_call_pricing_is_free
        && ctx.deployment_pricing.is_none()
        && ctx.pricing_model.is_none()
    {
        return 0.0;
    }
    let pricing_model = ctx.pricing_model.as_deref().unwrap_or(&ctx.model);
    let table = oxllm_core::lookup_pricing(pricing_model);
    let duration_ms = ctx
        .upstream_duration_estimate
        .unwrap_or(Duration::from_secs(600))
        .as_millis() as f64;
    let duration_cost = match ctx.deployment_pricing.as_ref() {
        Some(pricing) => pricing.cost_per_second(duration_ms),
        None => table.and_then(|pricing| pricing.cost_per_second(duration_ms)),
    };
    if let Some(cost) = duration_cost {
        return cost;
    }
    // H17: reuse the cached prompt-token estimate when a previous hook
    // (e.g. RateLimitHook) already computed it. Only compute it here
    // when no other hook has populated `token_estimate`.
    let prompt_estimate = match ctx.token_estimate.get() {
        Some(cached) => *cached,
        None => {
            let est = crate::tokens::estimate_request_tokens_async(ctx.request_body.as_ref()).await;
            let _ = ctx.token_estimate.set(est);
            est
        }
    };
    let model_router_fee = ctx
        .model_router_input_rate
        .filter(|_| ctx.deployment_pricing.is_none())
        .map(|rate| prompt_estimate as f64 * rate);
    if crate::pipeline::is_model_router_fee_entry(pricing_model)
        && let Some(fee) = model_router_fee
    {
        return fee;
    }
    const DEFAULT_MAX_OUTPUT_TOKENS: u64 = 16_384;
    let requested_output = ["max_completion_tokens", "max_tokens", "max_output_tokens"]
        .iter()
        .find_map(|field| ctx.request_body.get(field).and_then(|v| v.as_u64()))
        .unwrap_or(DEFAULT_MAX_OUTPUT_TOKENS);
    // The override carries no `max_output_tokens`, so the model ceiling still
    // comes from the table (or the default) when present.
    let model_ceiling = table
        .and_then(|p| p.max_output_tokens.map(u64::from))
        .unwrap_or(requested_output);
    let output_multiplier = ["n", "best_of"]
        .iter()
        .filter_map(|field| ctx.request_body.get(field).and_then(|v| v.as_u64()))
        .max()
        .unwrap_or(1)
        .max(1);
    let output_estimate = requested_output
        .min(model_ceiling)
        .saturating_mul(output_multiplier);
    // Prefer the deployment pricing override so the reservation is sized on
    // the same rate card as post-call settlement; fall back to the table.
    // For tiered models (e.g. gpt-5.6, whose >272k prompt rates are ~2x the
    // base), size the reservation on the rates that apply for the ESTIMATED
    // prompt length — a base-rate reservation under-reserves requests that
    // cross a tier threshold, letting a concurrent burst overshoot the
    // ceiling by the un-reserved delta. The override path has no tiers.
    //
    // The reservation is a WORST-CASE bound so settlement (which can add
    // cache-creation tokens beyond the prompt estimate, or bill audio /
    // reasoning buckets) never exceeds it: every estimated prompt token is
    // reserved at the highest non-additive input-side rate (input, cache-read,
    // audio), plus a full-prompt cache write at the higher of the cache-
    // creation rates, and every estimated output token at the higher of the
    // output / reasoning rates. For flat models with free cache buckets this
    // collapses to the legacy `prompt * input + output * output`.
    let (prompt_estimate_cost, output_estimate_cost, search_estimate_cost) =
        match ctx.deployment_pricing.as_ref() {
            Some(dp) => {
                let input = dp.input_cost_per_token;
                let output = dp.output_cost_per_token;
                let cache_read = dp.cache_read_input_token_cost.unwrap_or(input);
                let cache_creation = dp.cache_creation_input_token_cost.unwrap_or(input);
                let reasoning = dp.output_cost_per_reasoning_token.unwrap_or(output);
                let cache_creation_1h = dp
                    .cache_creation_input_token_cost_above_1hr
                    .unwrap_or(cache_creation);
                let prompt_side = input.max(cache_read) + cache_creation.max(cache_creation_1h);
                let output_side = output.max(reasoning);
                (prompt_side, output_side, 0.0)
            }
            None => match table {
                Some(p) => {
                    // Cache-creation counts are unknown before the response, so
                    // the threshold uses 0 (the estimate is input tokens only).
                    //
                    // LOW (known trade-off, documented): this means a tiered
                    // model (e.g. `claude-sonnet-4-20250514`, 200k tier) whose
                    // context crosses the threshold ONLY via cache-creation
                    // tokens (e.g. 180k fresh prompt + 30k cache-write = 210k)
                    // reserves on the cheaper BASE rate, since the threshold
                    // fires on `prompt_estimate` alone. The post-success settle
                    // (`actual − estimate`) debits the difference, so lifetime
                    // spend is correct, but the IN-FLIGHT reservation can be
                    // ~2x too low for such requests — allowing budget overshoot
                    // proportional to concurrency on tiered cache-heavy traffic.
                    // This is consistent with the architecture doc's "router/
                    // budget estimates are approximate soft caps" stance. A
                    // tighter reservation could add the configured
                    // `prompt_cache_min_tokens` (or a deployment-level cache-
                    // write estimate) to `prompt_estimate` here before sizing
                    // the tier.
                    let rates =
                        p.rates_for_service_tier(prompt_estimate, 0, ctx.requested_service_tier());
                    // Cache-creation is ADDITIVE (it bills on top of the prompt
                    // buckets), so its rate is reserved for a full-prompt cache
                    // write in addition to the highest non-additive prompt rate.
                    let prompt_side = rates.input.max(rates.cache_read).max(rates.audio)
                        + rates.cache_creation.max(rates.cache_creation_1h);
                    let output_side = rates.output.max(rates.reasoning) + rates.citation;
                    // Search/citation buckets are additive to ordinary token spend.
                    // Use an explicit hosted-tool maximum when supplied; otherwise
                    // reserve at least one query for a search-priced model.
                    let search_uses = ctx
                        .request_body
                        .get("tools")
                        .and_then(serde_json::Value::as_array)
                        .into_iter()
                        .flatten()
                        .filter(|tool| {
                            tool.get("type")
                                .and_then(serde_json::Value::as_str)
                                .is_some_and(|kind| kind.starts_with("web_search"))
                        })
                        .filter_map(|tool| tool.get("max_uses").and_then(serde_json::Value::as_u64))
                        .max()
                        .unwrap_or(1)
                        .max(1);
                    (
                        prompt_side,
                        output_side,
                        rates.search_query * search_uses as f64,
                    )
                }
                None => return 0.01 + model_router_fee.unwrap_or(0.0),
            },
        };
    prompt_estimate as f64 * prompt_estimate_cost
        + output_estimate as f64 * output_estimate_cost
        + search_estimate_cost
        + model_router_fee.unwrap_or(0.0)
}

/// A hook that enforces budget ceilings per API key, team, and user.
pub struct BudgetHook {
    pub cache: Arc<dyn Cache>,
    pub max_budget: Option<f64>,
    pub budget_duration: Duration,
    /// Per-model global spend caps (model name → USD limit). When a request's
    /// model is present here, the hook additionally enforces a global ceiling
    /// against `spend:model:{model}` independent of the calling key/team/user.
    pub model_max_budget: Option<HashMap<String, f64>>,
}

impl BudgetHook {
    pub fn new(cache: Arc<dyn Cache>, max_budget: Option<f64>, budget_duration: &str) -> Self {
        Self {
            cache,
            max_budget,
            budget_duration: parse_budget_duration(budget_duration),
            model_max_budget: None,
        }
    }

    pub fn with_duration(
        cache: Arc<dyn Cache>,
        max_budget: Option<f64>,
        budget_duration: Duration,
    ) -> Self {
        Self {
            cache,
            max_budget,
            budget_duration,
            model_max_budget: None,
        }
    }

    /// Builder-style setter for the per-model global spend caps.
    pub fn with_model_budget(mut self, model_max_budget: HashMap<String, f64>) -> Self {
        self.model_max_budget = Some(model_max_budget);
        self
    }

    fn budget_ttl(&self, ctx: &CallContext) -> Duration {
        ctx.budget_duration
            .as_deref()
            .map(parse_budget_duration)
            .unwrap_or(self.budget_duration)
    }

    fn team_budget_ttl(&self, ctx: &CallContext) -> Duration {
        ctx.team_budget_duration
            .as_deref()
            .map(parse_budget_duration)
            .unwrap_or_else(|| self.budget_ttl(ctx))
    }

    fn org_budget_ttl(&self, ctx: &CallContext) -> Duration {
        ctx.org_budget_duration
            .as_deref()
            .map(parse_budget_duration)
            .unwrap_or_else(|| self.budget_ttl(ctx))
    }

    fn project_budget_ttl(&self, ctx: &CallContext) -> Duration {
        ctx.project_budget_duration
            .as_deref()
            .map(parse_budget_duration)
            .unwrap_or_else(|| self.budget_ttl(ctx))
    }

    fn user_budget_ttl(&self, ctx: &CallContext) -> Duration {
        ctx.user_budget_duration
            .as_deref()
            .map(parse_budget_duration)
            .unwrap_or_else(|| self.budget_ttl(ctx))
    }

    fn customer_budget_ttl(&self, ctx: &CallContext) -> Duration {
        ctx.customer_budget_duration
            .as_deref()
            .map(parse_budget_duration)
            .unwrap_or_else(|| self.budget_ttl(ctx))
    }

    fn model_budget(&self, model: &str) -> Option<f64> {
        self.model_max_budget.as_ref()?.get(model).copied()
    }

    /// Legacy estimate reservation for synthetic contexts that did not run
    /// admission. Production pre_call uses reserve_and_record/window receipts.
    ///
    /// U2: oxllm is already fail-closed. When the
    /// backing store (e.g. Redis) returns an `Err` from `incr_if_below`
    /// (Redis outage), the `?` operator propagates it as `Err(Error)`.
    /// In `pre_call`, every `reserve` call site catches `Err` and returns
    /// `reject_over_budget(...)` — so a cache outage REJECTS the request
    /// rather than letting it through. The reference bug was swallowing the
    /// error and continuing (fail-open); oxllm propagates it (fail-closed).
    async fn reserve(
        &self,
        key: &str,
        estimate: f64,
        limit: f64,
        ttl: Option<Duration>,
        reason: &str,
    ) -> Result<(), Error> {
        if estimate > 0.0 {
            match self.cache.incr_if_below(key, estimate, limit, ttl).await? {
                IncrResult::Allowed(_) => Ok(()),
                IncrResult::Rejected(_) => Err(Error::BudgetExceeded(format!(
                    "{reason} ({limit}) exceeded for this key"
                ))),
            }
        } else {
            // Zero/negative estimate: there is no spend to reserve, but the
            // request must still be gated on the ceiling through the SAME
            // fail-closed atomic path as a non-zero reservation. The prior
            // implementation read `counter_value(...).unwrap_or(0.0)`, which
            // silently swallowed a backing-store outage (e.g. Redis down) as
            // `0.0` and let the request through — a narrow exception to the
            // fail-closed guarantee. Routing through `incr_if_below` (with a
            // zero delta, so the counter is unchanged) surfaces such
            // outages as `Err` via `?`. A zero delta never exceeds the
            // ceiling on its own, so we additionally reject when the
            // counter has already reached the ceiling, preserving the
            // prior "reject once the counter is at the limit" semantics.
            match self.cache.incr_if_below(key, 0.0, limit, ttl).await? {
                IncrResult::Allowed(current) if current >= limit => Err(Error::BudgetExceeded(
                    format!("{reason} ({limit}) exceeded for this key"),
                )),
                IncrResult::Allowed(_) => Ok(()),
                IncrResult::Rejected(_) => Err(Error::BudgetExceeded(format!(
                    "{reason} ({limit}) exceeded for this key"
                ))),
            }
        }
    }

    /// Reserve `estimate` in one counter window, returning its receipt.
    ///
    /// `WindowChanged` maps to [`Error::RateLimited`], not `BudgetExceeded`:
    /// a stale receipt at top-up is a transient admission race (the window
    /// rolled over mid-flight and the replacement has fresh capacity), so the
    /// client should retry — a terminal-looking `budget_exceeded` code would
    /// make SDKs give up on a request that would succeed immediately.
    async fn reserve_window(
        &self,
        key: &str,
        estimate: f64,
        limit: f64,
        ttl: Option<Duration>,
        reason: &str,
        expected: Option<&CounterWindow>,
    ) -> Result<CounterWindow, Error> {
        if !estimate.is_finite() {
            return Err(Error::Config(
                "non-finite budget reservation estimate".into(),
            ));
        }
        match self
            .cache
            .incr_if_below_in_window(key, estimate.max(0.0), limit, ttl, expected)
            .await?
        {
            WindowedIncrResult::Allowed { value, window } if estimate > 0.0 || value < limit => {
                Ok(window)
            }
            WindowedIncrResult::Allowed { .. } | WindowedIncrResult::Rejected(_) => Err(
                Error::BudgetExceeded(format!("{reason} ({limit}) exceeded for this key")),
            ),
            WindowedIncrResult::WindowChanged => Err(Error::RateLimited(
                "budget window changed during admission; retry the request".into(),
            )),
        }
    }

    async fn reserve_and_record(
        &self,
        key: &str,
        estimate: f64,
        limit: f64,
        ttl: Option<Duration>,
        reason: &str,
        reserved: &mut Vec<(String, CounterWindow)>,
    ) -> Result<(), Error> {
        let window = self
            .reserve_window(key, estimate, limit, ttl, reason, None)
            .await?;
        reserved.push((key.to_string(), window));
        Ok(())
    }

    async fn release_window(&self, key: &str, estimate: f64, window: &CounterWindow) {
        if estimate > 0.0 {
            if let Err(error) = self.cache.incr_in_window(key, -estimate, window).await {
                tracing::warn!(%error, "budget window release failed");
                ::metrics::counter!(BUDGET_SETTLE_FAILED_TOTAL, "op" => "release").increment(1);
            }
        }
    }

    async fn release(&self, key: &str, estimate: f64) {
        // Compatibility only for manually constructed contexts without
        // admission receipts. Existence prevents recreating an absent counter,
        // but cannot prove ownership of a replacement window. Production
        // reservations/refunds always use release_window instead.
        if estimate > 0.0 {
            // Best-effort: a mid-flight cache outage (Redis goes down after
            // `pre_call` reserved successfully) means the reservation is
            // never released and hangs around until the window TTL. Failing
            // the completed request over a tracking write is worse, so the
            // error is swallowed — but it is logged and counted so the
            // billing inaccuracy is observable (mirrors the rate-limit
            // hook's `post_success` adjust logging).
            if let Err(e) = self.cache.decr_if_exists(key, estimate).await {
                tracing::warn!(error = %e, "budget release failed");
                ::metrics::counter!(BUDGET_SETTLE_FAILED_TOTAL, "op" => "release").increment(1);
            }
        }
    }

    /// Release the `estimate` reservation from every configured scope.
    /// Shared by `post_failure` (upstream error) and the cache-hit path of
    /// `post_success` (no upstream cost incurred).
    async fn release_reservations(&self, ctx: &CallContext, estimate: f64) {
        if let Some(windows) = &ctx.budget_windows {
            for (key, window) in windows {
                self.release_window(key, estimate, window).await;
            }
            return;
        }
        // Legacy synthetic contexts may supply an estimate without admission
        // receipts. Production pre_call always installs the window map below.
        if most_restrictive(ctx.max_budget, self.max_budget).is_some() {
            self.release(&spend_key(&ctx.api_key_hash), estimate).await;
        }
        if ctx.team_max_budget.is_some() {
            if let Some(team_id) = ctx.team_id.as_deref() {
                self.release(&team_spend_key(team_id), estimate).await;
            }
        }
        if ctx.org_max_budget.is_some() {
            if let Some(org_id) = ctx.org_id.as_deref() {
                self.release(&org_spend_key(org_id), estimate).await;
            }
        }
        if ctx.project_max_budget.is_some() {
            if let Some(project_id) = ctx.project_id.as_deref() {
                self.release(&project_spend_key(project_id), estimate).await;
            }
        }
        if ctx.user_max_budget.is_some() {
            if let Some(user_id) = ctx.user_id.as_deref() {
                self.release(&user_spend_key(user_id), estimate).await;
            }
        }
        if ctx.customer_max_budget.is_some() {
            if let Some(end_user_id) = ctx.end_user_id.as_deref() {
                self.release(&customer_spend_key(end_user_id), estimate)
                    .await;
            }
        }
        if !ctx.tag_budgets.is_empty() {
            for (tag, _) in &ctx.tag_budgets {
                self.release(&tag_spend_key(tag), estimate).await;
            }
        } else if ctx.tag_max_budget.is_some() {
            if let Some(tag) = ctx.request_tag.as_deref() {
                self.release(&tag_spend_key(tag), estimate).await;
            }
        }
        if self.model_budget(&ctx.model).is_some() {
            self.release(&model_spend_key(&ctx.model), estimate).await;
        }
    }

    async fn adjust(&self, key: &str, estimate: f64, actual: f64, ttl: Option<Duration>) {
        let delta = actual - estimate;
        // Exact-zero short-circuit (NOT a tolerance): real cost deltas live
        // in `[~1e-9, ~10]` USD, so this only fires when `delta` is exactly
        // `0.0` (estimate == actual). An `incr` by exactly `0.0` is a no-op
        // for both in-memory and Redis, so skipping it just avoids the
        // round-trip.
        if delta == 0.0 {
            return;
        }
        // Best-effort, same swallowing semantics as `release`: a mid-flight
        // outage drops the actual-cost delta, so the counter records the
        // pre-call estimate for the rest of the window. Logged + counted so
        // the degradation is observable. Negative deltas are
        // existence-gated so settling below an expired window's estimate
        // cannot mint a persistent negative counter.
        let outcome = if delta > 0.0 {
            self.cache.incr(key, delta, ttl).await.map(|_| ())
        } else if delta < 0.0 {
            self.cache.decr_if_exists(key, -delta).await.map(|_| ())
        } else {
            Ok(())
        };
        if let Err(e) = outcome {
            tracing::warn!(error = %e, "budget adjust failed");
            ::metrics::counter!(BUDGET_SETTLE_FAILED_TOTAL, "op" => "adjust").increment(1);
        }
    }

    /// The same monetary operation for success and known-cost partial work.
    /// Terminal status must not affect which reserved scopes/TTLs are settled.
    async fn settle_reservations(&self, ctx: &CallContext, actual: f64) {
        if ctx.is_master {
            return;
        }
        let Some(estimate) = ctx.budget_estimate else {
            return;
        };
        if ctx.cache_hit {
            self.release_reservations(ctx, estimate).await;
            return;
        }
        if let Some(windows) = &ctx.budget_windows {
            let delta = actual - estimate;
            if delta != 0.0 {
                for (key, window) in windows {
                    if let Err(error) = self.cache.incr_in_window(key, delta, window).await {
                        tracing::warn!(%error, "budget window adjust failed");
                        ::metrics::counter!(BUDGET_SETTLE_FAILED_TOTAL, "op" => "adjust")
                            .increment(1);
                    }
                }
            }
            return;
        }
        let ttl = Some(self.budget_ttl(ctx));
        if most_restrictive(ctx.max_budget, self.max_budget).is_some() {
            self.adjust(&spend_key(&ctx.api_key_hash), estimate, actual, ttl)
                .await;
        }
        if ctx.team_max_budget.is_some() {
            if let Some(team_id) = ctx.team_id.as_deref() {
                self.adjust(
                    &team_spend_key(team_id),
                    estimate,
                    actual,
                    Some(self.team_budget_ttl(ctx)),
                )
                .await;
            }
        }
        if ctx.org_max_budget.is_some() {
            if let Some(org_id) = ctx.org_id.as_deref() {
                self.adjust(
                    &org_spend_key(org_id),
                    estimate,
                    actual,
                    Some(self.org_budget_ttl(ctx)),
                )
                .await;
            }
        }
        if ctx.project_max_budget.is_some() {
            if let Some(project_id) = ctx.project_id.as_deref() {
                self.adjust(
                    &project_spend_key(project_id),
                    estimate,
                    actual,
                    Some(self.project_budget_ttl(ctx)),
                )
                .await;
            }
        }
        if ctx.user_max_budget.is_some() {
            if let Some(user_id) = ctx.user_id.as_deref() {
                self.adjust(
                    &user_spend_key(user_id),
                    estimate,
                    actual,
                    Some(self.user_budget_ttl(ctx)),
                )
                .await;
            }
        }
        if ctx.customer_max_budget.is_some() {
            if let Some(end_user_id) = ctx.end_user_id.as_deref() {
                self.adjust(
                    &customer_spend_key(end_user_id),
                    estimate,
                    actual,
                    Some(self.customer_budget_ttl(ctx)),
                )
                .await;
            }
        }
        if !ctx.tag_budgets.is_empty() {
            for (tag, _) in &ctx.tag_budgets {
                self.adjust(&tag_spend_key(tag), estimate, actual, ttl)
                    .await;
            }
        } else if ctx.tag_max_budget.is_some() {
            if let Some(tag) = ctx.request_tag.as_deref() {
                self.adjust(&tag_spend_key(tag), estimate, actual, ttl)
                    .await;
            }
        }
        if self.model_budget(&ctx.model).is_some() {
            self.adjust(
                &model_spend_key(&ctx.model),
                estimate,
                actual,
                Some(self.budget_duration),
            )
            .await;
        }
    }
}

#[async_trait]
impl Hook for BudgetHook {
    async fn pre_call(&self, ctx: &mut CallContext) -> oxllm_core::Result<HookAction> {
        if ctx.is_master {
            return Ok(HookAction::Continue);
        }

        let key_limit = most_restrictive(ctx.max_budget, self.max_budget);
        let team_limit = ctx.team_max_budget;
        let org_limit = ctx.org_max_budget;
        let project_limit = ctx.project_max_budget;
        let user_limit = ctx.user_max_budget;
        let customer_limit = ctx.customer_max_budget;
        let tag_limit = ctx.tag_max_budget;
        let model_limit = self.model_budget(&ctx.model);

        if key_limit.is_none()
            && team_limit.is_none()
            && org_limit.is_none()
            && project_limit.is_none()
            && user_limit.is_none()
            && customer_limit.is_none()
            && tag_limit.is_none()
            && ctx.tag_budgets.is_empty()
            && model_limit.is_none()
        {
            return Ok(HookAction::Continue);
        }

        let estimate = estimate_max_request_cost(ctx).await;
        let ttl = Some(self.budget_ttl(ctx));
        let team_ttl = Some(self.team_budget_ttl(ctx));
        let org_ttl = Some(self.org_budget_ttl(ctx));
        let project_ttl = Some(self.project_budget_ttl(ctx));
        let user_ttl = Some(self.user_budget_ttl(ctx));
        let customer_ttl = Some(self.customer_budget_ttl(ctx));
        let mut reserved_keys: Vec<(String, CounterWindow)> = Vec::new();

        if let Some(limit) = key_limit {
            // DELIBERATE DIVERGENCE FROM THE REFERENCE (C31): oxllm treats a
            // `max_budget` of `0` (or negative) as "reject everything", while
            // The reference (`budget_reservation.py`) treats `0` as "no limit" and
            // only enforces when `max_budget > 0`. A config ported verbatim
            // from an existing proxy with `max_budget: 0` (intended as unlimited there)
            // would 403 every request under oxllm. The block-all semantics
            // are pinned by `zero_max_budget_rejects`; do not flip them
            // without intent and without surfacing the change to operators.
            if limit <= 0.0 {
                return Ok(reject_over_budget(
                    "key",
                    Error::BudgetExceeded(format!("max_budget ({limit}) exceeded for this key")),
                ));
            }
            let key = spend_key(&ctx.api_key_hash);
            if let Err(err) = self
                .reserve_and_record(&key, estimate, limit, ttl, "max_budget", &mut reserved_keys)
                .await
            {
                return Ok(reject_over_budget("key", err));
            }
        }

        if let (Some(limit), Some(team_id)) = (team_limit, ctx.team_id.as_deref()) {
            if limit <= 0.0 {
                for (k, window) in &reserved_keys {
                    self.release_window(k, estimate, window).await;
                }
                return Ok(reject_over_budget(
                    "team",
                    Error::BudgetExceeded(format!("team max_budget ({limit}) exceeded")),
                ));
            }
            let tkey = team_spend_key(team_id);
            if let Err(err) = self
                .reserve_and_record(
                    &tkey,
                    estimate,
                    limit,
                    team_ttl,
                    "team max_budget",
                    &mut reserved_keys,
                )
                .await
            {
                for (k, window) in &reserved_keys {
                    self.release_window(k, estimate, window).await;
                }
                return Ok(reject_over_budget("team", err));
            }
        }

        if let (Some(limit), Some(org_id)) = (org_limit, ctx.org_id.as_deref()) {
            if limit <= 0.0 {
                for (k, window) in &reserved_keys {
                    self.release_window(k, estimate, window).await;
                }
                return Ok(reject_over_budget(
                    "org",
                    Error::BudgetExceeded(format!("org max_budget ({limit}) exceeded")),
                ));
            }
            let okey = org_spend_key(org_id);
            if let Err(err) = self
                .reserve_and_record(
                    &okey,
                    estimate,
                    limit,
                    org_ttl,
                    "org max_budget",
                    &mut reserved_keys,
                )
                .await
            {
                for (k, window) in &reserved_keys {
                    self.release_window(k, estimate, window).await;
                }
                return Ok(reject_over_budget("org", err));
            }
        }

        if let (Some(limit), Some(project_id)) = (project_limit, ctx.project_id.as_deref()) {
            if limit <= 0.0 {
                for (k, window) in &reserved_keys {
                    self.release_window(k, estimate, window).await;
                }
                return Ok(reject_over_budget(
                    "project",
                    Error::BudgetExceeded(format!("project max_budget ({limit}) exceeded")),
                ));
            }
            let pkey = project_spend_key(project_id);
            if let Err(err) = self
                .reserve_and_record(
                    &pkey,
                    estimate,
                    limit,
                    project_ttl,
                    "project max_budget",
                    &mut reserved_keys,
                )
                .await
            {
                for (k, window) in &reserved_keys {
                    self.release_window(k, estimate, window).await;
                }
                return Ok(reject_over_budget("project", err));
            }
        }

        if let (Some(limit), Some(user_id)) = (user_limit, ctx.user_id.as_deref()) {
            if limit <= 0.0 {
                for (k, window) in &reserved_keys {
                    self.release_window(k, estimate, window).await;
                }
                return Ok(reject_over_budget(
                    "user",
                    Error::BudgetExceeded(format!("user max_budget ({limit}) exceeded")),
                ));
            }
            let ukey = user_spend_key(user_id);
            if let Err(err) = self
                .reserve_and_record(
                    &ukey,
                    estimate,
                    limit,
                    user_ttl,
                    "user max_budget",
                    &mut reserved_keys,
                )
                .await
            {
                for (k, window) in &reserved_keys {
                    self.release_window(k, estimate, window).await;
                }
                return Ok(reject_over_budget("user", err));
            }
        }

        if let (Some(limit), Some(end_user_id)) = (customer_limit, ctx.end_user_id.as_deref()) {
            if limit <= 0.0 {
                for (k, window) in &reserved_keys {
                    self.release_window(k, estimate, window).await;
                }
                return Ok(reject_over_budget(
                    "customer",
                    Error::BudgetExceeded(format!("customer max_budget ({limit}) exceeded")),
                ));
            }
            let ckey = customer_spend_key(end_user_id);
            if let Err(err) = self
                .reserve_and_record(
                    &ckey,
                    estimate,
                    limit,
                    customer_ttl,
                    "customer max_budget",
                    &mut reserved_keys,
                )
                .await
            {
                for (k, window) in &reserved_keys {
                    self.release_window(k, estimate, window).await;
                }
                return Ok(reject_over_budget("customer", err));
            }
        }

        if !ctx.tag_budgets.is_empty() {
            // Per-tag reservation: each carried tag with a budget row gets
            // its own `spend:tag:{name}` counter reserved, so one request
            // counts toward EVERY application budget it is tagged with (the
            // shared-key / per-application cap pattern).
            for (tag, limit) in &ctx.tag_budgets {
                if *limit <= 0.0 {
                    for (k, window) in &reserved_keys {
                        self.release_window(k, estimate, window).await;
                    }
                    return Ok(reject_over_budget(
                        "tag",
                        Error::BudgetExceeded(format!(
                            "tag max_budget ({limit}) exceeded for tag '{tag}'"
                        )),
                    ));
                }
                let tkey = tag_spend_key(tag);
                if self
                    .reserve_and_record(
                        &tkey,
                        estimate,
                        *limit,
                        ttl,
                        "tag max_budget",
                        &mut reserved_keys,
                    )
                    .await
                    .is_err()
                {
                    // Releases every earlier reservation, including the tag
                    // counters pushed by previous iterations of this loop.
                    for (k, window) in &reserved_keys {
                        self.release_window(k, estimate, window).await;
                    }
                    return Ok(reject_over_budget(
                        "tag",
                        Error::BudgetExceeded(format!(
                            "tag max_budget ({limit}) exceeded for tag '{tag}'"
                        )),
                    ));
                }
            }
        } else if let (Some(limit), Some(tag)) = (tag_limit, ctx.request_tag.as_deref()) {
            if limit <= 0.0 {
                for (k, window) in &reserved_keys {
                    self.release_window(k, estimate, window).await;
                }
                return Ok(reject_over_budget(
                    "tag",
                    Error::BudgetExceeded(format!("tag max_budget ({limit}) exceeded")),
                ));
            }
            let tkey = tag_spend_key(tag);
            if let Err(err) = self
                .reserve_and_record(
                    &tkey,
                    estimate,
                    limit,
                    ttl,
                    "tag max_budget",
                    &mut reserved_keys,
                )
                .await
            {
                for (k, window) in &reserved_keys {
                    self.release_window(k, estimate, window).await;
                }
                return Ok(reject_over_budget("tag", err));
            }
        }

        if let Some(limit) = model_limit {
            if limit <= 0.0 {
                for (k, window) in &reserved_keys {
                    self.release_window(k, estimate, window).await;
                }
                return Ok(reject_over_budget(
                    "model",
                    Error::BudgetExceeded(format!(
                        "model budget ({limit}) exceeded for model '{}'",
                        ctx.model
                    )),
                ));
            }
            let mkey = model_spend_key(&ctx.model);
            if self
                .reserve_and_record(
                    &mkey,
                    estimate,
                    limit,
                    Some(self.budget_duration),
                    "model budget",
                    &mut reserved_keys,
                )
                .await
                .is_err()
            {
                for (k, window) in &reserved_keys {
                    self.release_window(k, estimate, window).await;
                }
                return Ok(reject_over_budget(
                    "model",
                    Error::BudgetExceeded(format!(
                        "model budget ({limit}) exceeded for model '{}'",
                        ctx.model
                    )),
                ));
            }
        }

        // Stash so post_success / post_failure can adjust or release without
        // re-running token estimation on the same request body.
        ctx.budget_estimate = Some(estimate);
        ctx.budget_windows = Some(reserved_keys.into_iter().collect());
        Ok(HookAction::Continue)
    }

    async fn post_success(&self, ctx: &CallContext, resp: &ResponseSummary) {
        if ctx.is_master || ctx.budget_estimate.is_none() {
            return;
        }

        // Prefer the cost the pipeline already resolved (`ResponseSummary.cost`)
        // — it is authoritative and covers every code path the spend log sees:
        // `ctx.cost_override` (passthrough-tier `x-oxllm-response-cost`), the
        // token price table, per-second pricing (`cost_per_second`), and
        // aggregator body-reported cost (`usage.cost`, OpenRouter). Reading it
        // first keeps the budget counter byte-for-byte aligned with the spend
        // log for all of them. Only when the pipeline did not populate it do we
        // fall back to a local token-table recompute (provider-aware, with a
        // `ctx.model` fallback mirroring `CostTrackingHook::post_success`).
        // Unknown pricing → treat actual as 0 so the reservation is released
        // rather than left as a permanent over-charge of the estimate.
        let actual = crate::hooks::cost_fallback::fallback_cost(ctx, resp).unwrap_or(0.0);
        self.settle_reservations(ctx, actual).await;
    }

    async fn settle_budget(&self, ctx: &CallContext, cost: f64) {
        self.settle_reservations(ctx, cost).await;
    }

    async fn post_failure(&self, ctx: &CallContext, _err: &Error) {
        // `is_master` bypasses budget entirely. The `cache_hit` guard that
        // used to live here is intentionally removed: if `post_failure` is
        // reached at all (even on a cache-hit request whose response later
        // failed downstream — e.g. serialization/transport after the
        // short-circuit), the reservation must be released rather than
        // leaking until the window TTL. The happy cache-hit path is handled
        // by `post_success`; `post_failure` handles the unhappy path. The
        // `budget_estimate` None check below short-circuits when nothing was
        // reserved.
        if ctx.is_master {
            return;
        }
        let Some(estimate) = ctx.budget_estimate else {
            return;
        };
        let key_limit = most_restrictive(ctx.max_budget, self.max_budget);
        let team_limit = ctx.team_max_budget;
        let org_limit = ctx.org_max_budget;
        let project_limit = ctx.project_max_budget;
        let user_limit = ctx.user_max_budget;
        let customer_limit = ctx.customer_max_budget;
        let tag_limit = ctx.tag_max_budget;
        let model_limit = self.model_budget(&ctx.model);
        if key_limit.is_none()
            && team_limit.is_none()
            && org_limit.is_none()
            && project_limit.is_none()
            && user_limit.is_none()
            && customer_limit.is_none()
            && tag_limit.is_none()
            && ctx.tag_budgets.is_empty()
            && model_limit.is_none()
        {
            return;
        }
        self.release_reservations(ctx, estimate).await;
    }
}

impl BudgetHook {
    /// After the router stamps `deployment_pricing`, re-estimate cost and top
    /// up any already-reserved spend counters by the positive delta. Marked-up
    /// deployments would otherwise under-reserve at `pre_call` (table rates)
    /// and overshoot the hard cap at settle.
    ///
    /// No-op when no prior reservation exists or the new estimate is not higher.
    /// A top-up must still own every original window; expiry/replacement rejects
    /// admission without reserving a partial delta in another request's window.
    /// Fail-closed on cache errors and stale receipts — the stale-receipt
    /// rejection surfaces as retryable [`Error::RateLimited`], since the
    /// replacement window typically has fresh capacity.
    pub async fn top_up_for_deployment_pricing(&self, ctx: &mut CallContext) -> Result<(), Error> {
        if ctx.is_master {
            return Ok(());
        }
        let Some(old_estimate) = ctx.budget_estimate else {
            return Ok(());
        };
        let new_estimate = estimate_max_request_cost(ctx).await;
        if new_estimate.partial_cmp(&(old_estimate + 1e-12)) != Some(std::cmp::Ordering::Greater) {
            return Ok(());
        }
        let delta = new_estimate - old_estimate;
        let key_limit = most_restrictive(ctx.max_budget, self.max_budget);
        let team_limit = ctx.team_max_budget;
        let org_limit = ctx.org_max_budget;
        let project_limit = ctx.project_max_budget;
        let user_limit = ctx.user_max_budget;
        let customer_limit = ctx.customer_max_budget;
        let tag_limit = ctx.tag_max_budget;
        let model_limit = self.model_budget(&ctx.model);
        if key_limit.is_none()
            && team_limit.is_none()
            && org_limit.is_none()
            && project_limit.is_none()
            && user_limit.is_none()
            && customer_limit.is_none()
            && tag_limit.is_none()
            && ctx.tag_budgets.is_empty()
            && model_limit.is_none()
        {
            return Ok(());
        }

        let key_ttl = Some(self.budget_ttl(ctx));
        let team_ttl = Some(self.team_budget_ttl(ctx));
        let org_ttl = Some(self.org_budget_ttl(ctx));
        let project_ttl = Some(self.project_budget_ttl(ctx));
        let user_ttl = Some(self.user_budget_ttl(ctx));
        let customer_ttl = Some(self.customer_budget_ttl(ctx));
        let mut topped: Vec<(String, Option<CounterWindow>)> = Vec::new();

        // Helper that rolls back prior tops on failure.
        #[expect(clippy::too_many_arguments)]
        async fn top(
            this: &BudgetHook,
            key: String,
            delta: f64,
            limit: f64,
            ttl: Option<Duration>,
            reason: &str,
            topped: &mut Vec<(String, Option<CounterWindow>)>,
            windows: Option<&HashMap<String, CounterWindow>>,
        ) -> Result<(), Error> {
            let result = if let Some(windows) = windows {
                match windows.get(&key) {
                    Some(window) => this
                        .reserve_window(&key, delta, limit, ttl, reason, Some(window))
                        .await
                        .map(Some),
                    None => Err(Error::Config(
                        "missing budget window receipt during top-up".into(),
                    )),
                }
            } else {
                // Compatibility for manually reserved, synthetic contexts.
                // Production admission always supplies per-scope receipts.
                this.reserve(&key, delta, limit, ttl, reason)
                    .await
                    .map(|()| None)
            };
            match result {
                Ok(window) => topped.push((key, window)),
                Err(error) => {
                    for (key, window) in topped.iter() {
                        if let Some(window) = window {
                            this.release_window(key, delta, window).await;
                        } else {
                            this.release(key, delta).await;
                        }
                    }
                    return Err(error);
                }
            }
            Ok(())
        }

        if let Some(limit) = key_limit {
            top(
                self,
                spend_key(&ctx.api_key_hash),
                delta,
                limit,
                key_ttl,
                "key max_budget",
                &mut topped,
                ctx.budget_windows.as_ref(),
            )
            .await?;
        }
        if let (Some(limit), Some(team_id)) = (team_limit, ctx.team_id.as_deref()) {
            top(
                self,
                team_spend_key(team_id),
                delta,
                limit,
                team_ttl,
                "team max_budget",
                &mut topped,
                ctx.budget_windows.as_ref(),
            )
            .await?;
        }
        if let (Some(limit), Some(org_id)) = (org_limit, ctx.org_id.as_deref()) {
            top(
                self,
                org_spend_key(org_id),
                delta,
                limit,
                org_ttl,
                "org max_budget",
                &mut topped,
                ctx.budget_windows.as_ref(),
            )
            .await?;
        }
        if let (Some(limit), Some(project_id)) = (project_limit, ctx.project_id.as_deref()) {
            top(
                self,
                project_spend_key(project_id),
                delta,
                limit,
                project_ttl,
                "project max_budget",
                &mut topped,
                ctx.budget_windows.as_ref(),
            )
            .await?;
        }
        if let (Some(limit), Some(user_id)) = (user_limit, ctx.user_id.as_deref()) {
            top(
                self,
                user_spend_key(user_id),
                delta,
                limit,
                user_ttl,
                "user max_budget",
                &mut topped,
                ctx.budget_windows.as_ref(),
            )
            .await?;
        }
        if let (Some(limit), Some(end_user_id)) = (customer_limit, ctx.end_user_id.as_deref()) {
            top(
                self,
                customer_spend_key(end_user_id),
                delta,
                limit,
                customer_ttl,
                "customer max_budget",
                &mut topped,
                ctx.budget_windows.as_ref(),
            )
            .await?;
        }
        if !ctx.tag_budgets.is_empty() {
            for (tag, limit) in &ctx.tag_budgets {
                top(
                    self,
                    tag_spend_key(tag),
                    delta,
                    *limit,
                    key_ttl,
                    "tag max_budget",
                    &mut topped,
                    ctx.budget_windows.as_ref(),
                )
                .await?;
            }
        } else if let (Some(limit), Some(tag)) = (tag_limit, ctx.request_tag.as_deref()) {
            top(
                self,
                tag_spend_key(tag),
                delta,
                limit,
                key_ttl,
                "tag max_budget",
                &mut topped,
                ctx.budget_windows.as_ref(),
            )
            .await?;
        }
        if let Some(limit) = model_limit {
            top(
                self,
                model_spend_key(&ctx.model),
                delta,
                limit,
                Some(self.budget_duration),
                "model budget",
                &mut topped,
                ctx.budget_windows.as_ref(),
            )
            .await?;
        }

        ctx.budget_estimate = Some(new_estimate);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::InMemoryCache;
    use oxllm_core::UsageSummary;

    fn make_ctx() -> CallContext {
        CallContext::for_test("gpt-4o")
    }

    #[tokio::test]
    async fn free_group_zero_estimates_still_reject_exhausted_positive_budgets() {
        for current in [1.0, 1.5] {
            let cache = Arc::new(InMemoryCache::default());
            cache.incr("spend:abc", current, None).await.unwrap();
            let hook = BudgetHook::new(cache.clone(), Some(1.0), "1h");
            let mut ctx = CallContext {
                pre_call_pricing_is_free: true,
                ..CallContext::for_test("free-group")
            };
            assert!(matches!(
                hook.pre_call(&mut ctx).await.unwrap(),
                HookAction::Reject(Error::BudgetExceeded(_))
            ));
            assert_eq!(cache.counter_value("spend:abc").await, Some(current));
            assert!(ctx.budget_estimate.is_none());
        }
    }

    #[tokio::test]
    async fn duration_reservation_rejects_a_request_the_budget_cannot_cover() {
        let cache = Arc::new(InMemoryCache::default());
        let hook = BudgetHook::with_duration(cache.clone(), Some(5.0), Duration::from_secs(60));
        let mut ctx = CallContext::for_test("private-duration-model");
        ctx.deployment_pricing = Some(oxllm_core::DeploymentPricing {
            cost_per_second_rate: Some(2.0),
            ..Default::default()
        });
        ctx.upstream_duration_estimate = Some(Duration::from_secs(5));
        assert!(matches!(
            hook.pre_call(&mut ctx).await.unwrap(),
            HookAction::Reject(_)
        ));
        assert_eq!(cache.counter_value("spend:abc").await.unwrap_or(0.0), 0.0);
    }

    #[tokio::test]
    async fn search_and_citation_charges_are_reserved_before_a_search_request() {
        let mut ctx = CallContext::for_test("perplexity/sonar-deep-research");
        let _ = ctx.token_estimate.set(10);
        ctx.request_body = Arc::new(serde_json::json!({"max_tokens":10}));
        let estimate = estimate_max_request_cost(&ctx).await;
        let actual = oxllm_core::calculate_cost(
            &ctx.model,
            &UsageSummary {
                prompt_tokens: 10,
                completion_tokens: 10,
                citation_tokens: Some(10),
                web_search_requests: Some(1),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(
            estimate >= actual,
            "reservation {estimate} must cover {actual}"
        );
        let hook = BudgetHook::new(Arc::new(InMemoryCache::default()), Some(0.001), "1h");
        assert!(matches!(
            hook.pre_call(&mut ctx).await.unwrap(),
            HookAction::Reject(_)
        ));
    }

    #[tokio::test]
    async fn reservations_honor_explicit_output_limits_and_one_hour_write_rates() {
        let mut ctx = CallContext::for_test("private-token-model");
        let _ = ctx.token_estimate.set(100);
        ctx.request_body = Arc::new(serde_json::json!({"max_tokens":100_000}));
        ctx.deployment_pricing = Some(oxllm_core::DeploymentPricing {
            input_cost_per_token: 1.0,
            output_cost_per_token: 0.01,
            cache_creation_input_token_cost: Some(0.5),
            cache_creation_input_token_cost_above_1hr: Some(10.0),
            ..Default::default()
        });
        assert_eq!(estimate_max_request_cost(&ctx).await, 2100.0);
    }

    #[tokio::test]
    async fn global_model_budget_does_not_expire_with_the_callers_key_window() {
        let cache = Arc::new(InMemoryCache::default());
        let hook = BudgetHook::with_duration(cache.clone(), None, Duration::from_secs(60))
            .with_model_budget(HashMap::from([("private-model".into(), 100.0)]));
        let mut ctx = CallContext::for_test("private-model");
        ctx.budget_duration = Some("10ms".into());
        assert!(matches!(
            hook.pre_call(&mut ctx).await.unwrap(),
            HookAction::Continue
        ));
        tokio::time::sleep(Duration::from_millis(30)).await;
        assert_eq!(
            cache.counter_value("spend:model:private-model").await,
            ctx.budget_estimate
        );
    }

    // U2: A cache whose `incr_if_below` always returns `Err`, simulating a
    // Redis outage. This lets us verify the BudgetHook is fail-closed.
    struct FailingCache;

    #[async_trait]
    impl Cache for FailingCache {
        async fn get(&self, _key: &str) -> Option<String> {
            None
        }
        async fn set(&self, _key: &str, _value: String, _ttl: Option<Duration>) {}
        async fn delete(&self, _key: &str) {}
        async fn incr(&self, _key: &str, _by: f64, _ttl: Option<Duration>) -> Result<f64, Error> {
            Err(Error::Network("cache backend unavailable".into()))
        }
        // Default `incr_if_below` calls `incr` and propagates the error via `?`.
    }

    fn make_ctx_with_budget(max_budget: Option<f64>) -> CallContext {
        CallContext {
            max_budget,
            ..make_ctx()
        }
    }

    fn make_response(model: &str, prompt: u64, completion: u64) -> ResponseSummary {
        ResponseSummary {
            model: model.to_string(),
            cost: None,
            usage: Some(UsageSummary {
                prompt_tokens: prompt,
                completion_tokens: completion,
                total_tokens: prompt + completion,
                cache_read_input_tokens: None,
                cache_creation_input_tokens: None,
                audio_input_tokens: None,
                reasoning_tokens: None,
                citation_tokens: None,
                web_search_requests: None,
                browser_open_requests: None,
                cache_creation_1h_input_tokens: None,
            }),
            body: None,
            provider: "openai".to_string(),
        }
    }

    #[tokio::test]
    async fn model_router_unpriced_deployment_tops_up_nominal_budget_and_settles_fee() {
        let config = serde_json::from_value(serde_json::json!({"model_list":[]})).unwrap();
        let state =
            crate::state::AppState::with_registry(config, oxllm_providers::default_registry())
                .unwrap();
        let hook = BudgetHook::new(state.cache.clone(), None, "1h");
        let mut ctx = CallContext::for_test("public-router");
        ctx.max_budget = Some(1.0);
        ctx.token_estimate.set(100_000).unwrap();
        ctx.request_body = Arc::new(serde_json::json!({"model":"public-router","max_tokens":20}));
        assert!(matches!(
            hook.pre_call(&mut ctx).await.unwrap(),
            HookAction::Continue
        ));
        assert_eq!(ctx.budget_estimate, Some(0.01));

        let deployment = serde_json::from_value(serde_json::json!({
            "model_name":"public-router",
            "oxllm_params":{"model":"azure_ai/model_router/private-deployment"}
        }))
        .unwrap();
        let request = ctx.request_body.clone();
        crate::pipeline::prepare_deployment_pricing(&state, &mut ctx, &deployment, &request)
            .await
            .unwrap();
        assert!(oxllm_core::lookup_pricing(ctx.pricing_model.as_deref().unwrap()).is_none());
        let rate = oxllm_core::cost::price_table()["azure_ai/model-router"].input_cost_per_token;
        let expected_reservation = 0.01 + 100_000.0 * rate;
        assert!(
            (ctx.budget_estimate.unwrap() - expected_reservation).abs() < 1e-12,
            "an unpriced router deployment must reserve its fee on top of the nominal floor"
        );
        assert!(
            (state.cache.counter_value("spend:abc").await.unwrap() - expected_reservation).abs()
                < 1e-12
        );

        let mut response = make_response("unpriced-selected-model", 60_000, 20);
        response.provider = "azure_ai".into();
        response.usage.as_mut().unwrap().cache_read_input_tokens = Some(30_000);
        hook.post_success(&ctx, &response).await;
        let expected_actual = 60_000.0 * rate;
        assert!(
            (state.cache.counter_value("spend:abc").await.unwrap() - expected_actual).abs() < 1e-12
        );
    }

    #[tokio::test]
    async fn model_router_fee_can_exceed_the_nominal_unknown_model_budget() {
        let cache = Arc::new(InMemoryCache::default());
        let rate = oxllm_core::cost::price_table()["azure_ai/model-router"].input_cost_per_token;
        let hook = BudgetHook::new(cache.clone(), Some(0.01 + 50_000.0 * rate), "1h");
        let mut ctx = CallContext::for_test("unpriced-router-deployment");
        ctx.model_router_input_rate = Some(rate);
        ctx.token_estimate.set(100_000).unwrap();
        assert!(matches!(
            hook.pre_call(&mut ctx).await.unwrap(),
            HookAction::Reject(Error::BudgetExceeded(_))
        ));
        assert_eq!(cache.counter_value("spend:abc").await.unwrap_or(0.0), 0.0);
    }

    #[tokio::test]
    async fn model_router_fee_only_estimate_and_settlement_honor_rate_overrides_once() {
        let catalog_rate =
            oxllm_core::cost::price_table()["azure_ai/model-router"].input_cost_per_token;
        for (provider, model) in [
            ("azure", "azure/model-router"),
            ("azure_ai", "azure_ai/model-router"),
            ("azure_ai", "azure_ai/model_router"),
        ] {
            for rate in [catalog_rate, catalog_rate * 2.0, 0.0] {
                let cache = Arc::new(InMemoryCache::default());
                let hook = BudgetHook::new(cache.clone(), Some(1.0), "1h");
                let mut ctx = CallContext::for_test("public-router");
                ctx.pricing_model = Some(model.into());
                ctx.model_router_input_rate = Some(rate);
                ctx.token_estimate.set(1_000).unwrap();
                assert!(matches!(
                    hook.pre_call(&mut ctx).await.unwrap(),
                    HookAction::Continue
                ));
                let expected_estimate = 1_000.0 * rate;
                assert!(
                    (ctx.budget_estimate.unwrap() - expected_estimate).abs() < 1e-12,
                    "{model}/{rate}: the canonical row must reserve the effective fee once"
                );
                let mut response = make_response("model-router", 800, 20);
                response.provider = provider.into();
                response.usage.as_mut().unwrap().cache_read_input_tokens = Some(400);
                hook.post_success(&ctx, &response).await;
                assert!(
                    (cache.counter_value("spend:abc").await.unwrap() - 800.0 * rate).abs() < 1e-12,
                    "{model}/{rate}: cached-read tokens must still incur exactly one fee"
                );
            }
        }
    }

    #[tokio::test]
    async fn model_router_cache_hit_releases_nominal_and_fee_reservations() {
        let cache = Arc::new(InMemoryCache::default());
        let hook = BudgetHook::new(cache.clone(), Some(1.0), "1h");
        let mut ctx = CallContext::for_test("unpriced-router-deployment");
        let rate = oxllm_core::cost::price_table()["azure_ai/model-router"].input_cost_per_token;
        ctx.model_router_input_rate = Some(rate);
        ctx.token_estimate.set(100_000).unwrap();
        assert!(matches!(
            hook.pre_call(&mut ctx).await.unwrap(),
            HookAction::Continue
        ));
        assert!((ctx.budget_estimate.unwrap() - (0.01 + 100_000.0 * rate)).abs() < 1e-12);
        ctx.cache_hit = true;
        let mut response = make_response("gpt-6.1-sol", 100_000, 20);
        response.cost = Some(1.0);
        hook.post_success(&ctx, &response).await;
        assert_eq!(cache.counter_value("spend:abc").await.unwrap_or(0.0), 0.0);
    }

    #[tokio::test]
    async fn multi_tag_success_settles_every_reserved_scope_not_the_primary_label() {
        // Exercise both a refund and an upward adjustment against the unknown
        // model's nominal reservation. The primary label has no budget.
        for actual in [0.002, 0.025] {
            let cache = Arc::new(InMemoryCache::default());
            cache.incr("spend:tag:app", 0.125, None).await.unwrap();
            cache.incr("spend:tag:prod", 0.25, None).await.unwrap();
            let hook = BudgetHook::with_duration(cache.clone(), None, Duration::from_secs(60));
            let mut ctx = CallContext::for_test("unknown-metered-model");
            ctx.request_tag = Some("label-only".into());
            ctx.request_tags = vec!["label-only".into(), "app".into(), "prod".into()];
            ctx.tag_budgets = vec![("app".into(), 1.0), ("prod".into(), 1.0)];
            // tag_budgets is authoritative even without the legacy summary.
            ctx.tag_max_budget = None;
            assert!(matches!(
                hook.pre_call(&mut ctx).await.unwrap(),
                HookAction::Continue
            ));
            assert!(ctx.budget_estimate.is_some());
            let resp = ResponseSummary {
                model: ctx.model.clone(),
                provider: "openai".into(),
                cost: Some(actual),
                usage: None,
                body: None,
            };
            hook.post_success(&ctx, &resp).await;
            for (tag, prior) in [("app", 0.125), ("prod", 0.25)] {
                let spent = cache
                    .counter_value(&tag_spend_key(tag))
                    .await
                    .expect("reserved scope");
                assert!((spent - (prior + actual)).abs() < 1e-12, "{tag}: {spent}");
            }
            assert_eq!(cache.counter_value("spend:tag:label-only").await, None);
        }
    }

    #[tokio::test]
    async fn multi_tag_refund_restores_room_for_the_next_request() {
        let cache = Arc::new(InMemoryCache::default());
        let hook = BudgetHook::with_duration(cache.clone(), None, Duration::from_secs(60));
        let context = || {
            let mut ctx = CallContext::for_test("unknown-metered-model");
            ctx.request_tag = Some("app".into());
            ctx.tag_max_budget = Some(0.015);
            ctx.tag_budgets = vec![("app".into(), 0.015), ("prod".into(), 0.015)];
            ctx
        };
        let mut first = context();
        assert!(matches!(
            hook.pre_call(&mut first).await.unwrap(),
            HookAction::Continue
        ));
        hook.post_success(
            &first,
            &ResponseSummary {
                model: first.model.clone(),
                provider: "openai".into(),
                cost: Some(0.001),
                usage: None,
                body: None,
            },
        )
        .await;
        let mut second = context();
        assert!(
            matches!(
                hook.pre_call(&mut second).await.unwrap(),
                HookAction::Continue
            ),
            "secondary tag must regain unused reservation, allowing another request"
        );
        hook.post_failure(&second, &Error::Network("upstream unavailable".into()))
            .await;
        for tag in ["app", "prod"] {
            let spent = cache.counter_value(&tag_spend_key(tag)).await.unwrap();
            assert!((spent - 0.001).abs() < 1e-12);
        }
    }

    #[tokio::test]
    async fn tag_reservations_release_on_cache_hits_and_failures_including_legacy_contexts() {
        for legacy in [false, true] {
            for cache_hit in [false, true] {
                let cache = Arc::new(InMemoryCache::default());
                let hook = BudgetHook::with_duration(cache.clone(), None, Duration::from_secs(60));
                let mut ctx = CallContext::for_test("unknown-metered-model");
                ctx.request_tag = Some("app".into());
                ctx.tag_max_budget = Some(1.0);
                if !legacy {
                    ctx.tag_budgets = vec![("app".into(), 1.0), ("prod".into(), 1.0)];
                }
                assert!(matches!(
                    hook.pre_call(&mut ctx).await.unwrap(),
                    HookAction::Continue
                ));
                if cache_hit {
                    ctx.cache_hit = true;
                    hook.post_success(&ctx, &make_response("gpt-4o", 10, 5))
                        .await;
                } else {
                    hook.post_failure(&ctx, &Error::Network("failed".into()))
                        .await;
                }
                assert!(cache.counter_value("spend:tag:app").await.unwrap().abs() < 1e-12);
                if !legacy {
                    assert!(cache.counter_value("spend:tag:prod").await.unwrap().abs() < 1e-12);
                }
            }
        }
    }

    #[tokio::test]
    async fn no_max_budget_always_continues() {
        let cache = Arc::new(InMemoryCache::default());
        let hook = BudgetHook::with_duration(cache, None, Duration::from_secs(60));
        let mut ctx = make_ctx();
        let action = hook.pre_call(&mut ctx).await.expect("continue");
        assert!(matches!(action, HookAction::Continue));
    }

    /// LOW (C31): pins the DELIBERATE divergence from the reference. oxllm treats
    /// `max_budget: 0` as block-all; the reference treats `0` as "no limit" (only
    /// `max_budget > 0` is enforced). A config ported verbatim from an existing proxy
    /// with `max_budget: 0` would 403 every request under oxllm — see the
    /// divergence note at the `<= 0.0` check in `pre_call`. Do not flip this
    /// without intent.
    #[tokio::test]
    async fn zero_max_budget_rejects() {
        let cache = Arc::new(InMemoryCache::default());
        let hook = BudgetHook::with_duration(cache, Some(0.0), Duration::from_secs(60));
        let mut ctx = make_ctx();
        let action = hook.pre_call(&mut ctx).await.expect("ok");
        assert!(matches!(action, HookAction::Reject(_)));
    }

    #[tokio::test]
    async fn under_budget_continues_and_reserves() {
        let cache = Arc::new(InMemoryCache::default());
        let hook = BudgetHook::with_duration(cache.clone(), Some(1.0), Duration::from_secs(60));
        let mut ctx = make_ctx();
        ctx.request_body = std::sync::Arc::new(serde_json::json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}]
        }));
        let action = hook.pre_call(&mut ctx).await.expect("continue");
        assert!(matches!(action, HookAction::Continue));
        let reserved = cache.counter_value("spend:abc").await.unwrap_or(0.0);
        assert!(reserved > 0.0, "pre_call must reserve an estimate");
        assert_eq!(
            ctx.budget_estimate,
            Some(reserved),
            "pre_call must stash the reservation for post_* reuse"
        );
    }

    #[tokio::test]
    async fn post_success_reuses_stashed_estimate_without_recompute() {
        let cache = Arc::new(InMemoryCache::default());
        let hook = BudgetHook::with_duration(cache.clone(), Some(10.0), Duration::from_secs(60));
        let mut ctx = make_ctx();
        ctx.request_body = std::sync::Arc::new(serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}]
        }));
        hook.pre_call(&mut ctx).await.expect("continue");
        let stashed = ctx.budget_estimate.expect("estimate stashed");
        // If post_success re-ran estimation it would still match, but using a
        // deliberately different stash proves the cached value is what adjusts.
        ctx.budget_estimate = Some(0.001);
        let resp = make_response("gpt-4o", 10, 5);
        hook.post_success(&ctx, &resp).await;
        let spend = cache.counter_value("spend:abc").await.unwrap_or(0.0);
        // Spent amount should be actual cost adjusted from 0.001, not from stashed.
        // After adjust: spend = reserved_amount_in_cache + (actual - 0.001).
        // Cache still has the original pre_call reservation (stashed), not 0.001.
        // Wait: pre_call reserved `stashed` into cache. We only changed ctx field
        // to 0.001. adjust does incr(actual - 0.001), so final =
        // stashed + actual - 0.001.
        let actual = oxllm_core::calculate_cost("gpt-4o", resp.usage.as_ref().expect("usage"))
            .unwrap_or(0.0);
        let expected = stashed + actual - 0.001;
        assert!(
            (spend - expected).abs() < 1e-9,
            "post_success must use ctx.budget_estimate (spend={spend}, expected={expected})"
        );
    }

    #[tokio::test]
    async fn over_budget_rejects() {
        let cache = Arc::new(InMemoryCache::default());
        let _ = cache.incr("spend:abc", 5.0, None).await;
        let hook = BudgetHook::with_duration(cache, Some(1.0), Duration::from_secs(60));
        let mut ctx = make_ctx();
        let action = hook.pre_call(&mut ctx).await.expect("ok");
        match action {
            HookAction::Reject(Error::BudgetExceeded(_)) => {}
            other => panic!("expected Reject, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn post_success_adjusts_to_actual_cost() {
        let cache = Arc::new(InMemoryCache::default());
        let hook = BudgetHook::with_duration(cache.clone(), Some(100.0), Duration::from_secs(60));
        let mut ctx = make_ctx();
        ctx.request_body = std::sync::Arc::new(serde_json::json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}]
        }));
        hook.pre_call(&mut ctx).await.expect("continue");
        let reserved = cache.counter_value("spend:abc").await.unwrap_or(0.0);
        let resp = make_response("gpt-4o", 1_000_000, 500_000);
        hook.post_success(&ctx, &resp).await;
        let spent = cache.counter_value("spend:abc").await.unwrap_or(0.0);
        let expected = 1_000_000.0 * 0.0000025 + 500_000.0 * 0.00001;
        assert!(
            (spent - expected).abs() < 1e-6,
            "final spend ({spent}) should be actual cost ({expected}), reserved was {reserved}"
        );
    }

    /// LOW: when `resp.model` is unpriced but `ctx.model` (the client-requested
    /// model) resolves, the budget hook must fall back to `ctx.model` so the
    /// budget counter agrees with `CostTrackingHook::post_success` (which has
    /// the same fallback). Without this, a model-name mismatch where the
    /// serving name is unpriced would debit 0 in budget but a positive cost
    /// in the spend log.
    #[tokio::test]
    async fn post_success_falls_back_to_ctx_model_when_resp_model_unpriced() {
        let cache = Arc::new(InMemoryCache::default());
        let hook = BudgetHook::with_duration(cache.clone(), Some(100.0), Duration::from_secs(60));
        let mut ctx = make_ctx();
        ctx.model = "gpt-4o".to_string();
        ctx.request_body = std::sync::Arc::new(serde_json::json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}]
        }));
        hook.pre_call(&mut ctx).await.expect("continue");
        // resp.model is an unpriced alias; ctx.model is the priced canonical name.
        let resp = make_response("totally-unknown-serving-alias", 100, 50);
        hook.post_success(&ctx, &resp).await;
        let spent = cache.counter_value("spend:abc").await.unwrap_or(0.0);

        let usage = resp.usage.as_ref().expect("usage");
        let expected =
            oxllm_core::calculate_cost_with_override_for_provider("gpt-4o", usage, None, "openai")
                .expect("ctx.model resolves");
        assert!(
            (spent - expected).abs() < 1e-9,
            "budget must debit the ctx.model fallback cost ({expected}), got {spent}"
        );
        assert!(expected > 0.0, "test premise: gpt-4o must be priced");
    }

    /// HIGH (C31): `post_success` must debit the authoritative
    /// `ResponseSummary.cost` (already resolved by the pipeline) rather than
    /// recomputing from the token table. Without this, the budget counter and
    /// the spend log permanently disagree for every code path the pipeline
    /// resolves beyond the token table (cost_override, per-second, body-cost).
    #[tokio::test]
    async fn post_success_uses_response_summary_cost_when_present() {
        let cache = Arc::new(InMemoryCache::default());
        let hook = BudgetHook::with_duration(cache.clone(), Some(100.0), Duration::from_secs(60));
        let mut ctx = make_ctx();
        ctx.request_body = std::sync::Arc::new(serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}]
        }));
        hook.pre_call(&mut ctx).await.expect("continue");
        // The pipeline would have debited a token-table cost for this usage;
        // surface a deliberately different authoritative cost and assert the
        // counter ends at it, not the recomputed value.
        let mut resp = make_response("gpt-4o", 1_000_000, 500_000);
        resp.cost = Some(0.99);
        hook.post_success(&ctx, &resp).await;
        let spent = cache.counter_value("spend:abc").await.unwrap_or(0.0);
        let recomputed = oxllm_core::calculate_cost_with_override_for_provider(
            "gpt-4o",
            resp.usage.as_ref().expect("usage"),
            None,
            "openai",
        )
        .unwrap_or(0.0);
        assert!(
            (recomputed - 0.99).abs() > 1e-6,
            "test premise: recomputed cost ({recomputed}) must differ from authoritative 0.99"
        );
        assert!(
            (spent - 0.99).abs() < 1e-9,
            "budget must debit ResponseSummary.cost (0.99), not the recomputed token-table cost ({recomputed}); got {spent}"
        );
    }

    /// HIGH (C31): the passthrough tier sets `ctx.cost_override` (upstream
    /// `x-oxllm-response-cost`), which the pipeline stamps onto
    /// `ResponseSummary.cost`. The budget counter must debit that override so
    // the passthrough tier — whose whole point is forwarding provider-reported
    // spend — is actually enforced.
    #[tokio::test]
    async fn post_success_uses_cost_override_for_passthrough() {
        let cache = Arc::new(InMemoryCache::default());
        let hook = BudgetHook::with_duration(cache.clone(), Some(100.0), Duration::from_secs(60));
        let mut ctx = make_ctx();
        ctx.cost_override = Some(5.0);
        ctx.request_body = std::sync::Arc::new(serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}]
        }));
        hook.pre_call(&mut ctx).await.expect("continue");
        // Pipeline resolves cost_override → ResponseSummary.cost; resp.cost is
        // Some(5.0) while the token table would compute a different number.
        let mut resp = make_response("gpt-4o", 1_000, 500);
        resp.cost = Some(5.0);
        hook.post_success(&ctx, &resp).await;
        let spent = cache.counter_value("spend:abc").await.unwrap_or(0.0);
        assert!(
            (spent - 5.0).abs() < 1e-9,
            "budget must debit the override (5.0), not the token-table recompute; got {spent}"
        );
    }

    /// HIGH (C31): models the token cost path returns `None` for — notably
    /// per-second-priced rows (SageMaker/Replicate) and aggregator
    /// body-reported cost (OpenRouter) — were debited `0.0` by the prior
    /// recompute, so the reservation was released and these model classes were
    /// effectively unenforced by budget even though the spend log recorded a
    /// positive cost. The pipeline resolves such costs onto
    /// `ResponseSummary.cost`; reading it first debits correctly.
    #[tokio::test]
    async fn post_success_debits_per_second_model() {
        let cache = Arc::new(InMemoryCache::default());
        let hook = BudgetHook::with_duration(cache.clone(), Some(100.0), Duration::from_secs(60));
        let mut ctx = make_ctx();
        ctx.model = "sagemaker/jumpstart-dummy".to_string();
        ctx.request_body = std::sync::Arc::new(serde_json::json!({
            "model": "sagemaker/jumpstart-dummy",
            "messages": [{"role": "user", "content": "hi"}]
        }));
        hook.pre_call(&mut ctx).await.expect("continue");
        let reserved = cache.counter_value("spend:abc").await.unwrap_or(0.0);
        // The token path returns None for this usage (no per-token rates); the
        // pipeline-resolved per-second cost lands on resp.cost.
        let usage = UsageSummary {
            prompt_tokens: 100,
            completion_tokens: 50,
            total_tokens: 150,
            ..Default::default()
        };
        let token_cost = oxllm_core::calculate_cost_with_override_for_provider(
            "sagemaker/jumpstart-dummy",
            &usage,
            None,
            "sagemaker",
        );
        let resp = ResponseSummary {
            model: "sagemaker/jumpstart-dummy".to_string(),
            cost: Some(0.0073),
            usage: Some(usage),
            body: None,
            provider: "sagemaker".to_string(),
        };
        assert!(
            token_cost.is_none(),
            "test premise: token path must return None for a per-second model"
        );
        hook.post_success(&ctx, &resp).await;
        let spent = cache.counter_value("spend:abc").await.unwrap_or(0.0);
        assert!(
            (spent - 0.0073).abs() < 1e-9,
            "budget must debit the per-second cost (0.0073) instead of releasing the reservation; reserved was {reserved}, got {spent}"
        );
    }

    #[tokio::test]
    async fn post_failure_releases_reservation() {
        let cache = Arc::new(InMemoryCache::default());
        let hook = BudgetHook::with_duration(cache.clone(), Some(100.0), Duration::from_secs(60));
        let mut ctx = make_ctx();
        ctx.request_body = std::sync::Arc::new(serde_json::json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}]
        }));
        hook.pre_call(&mut ctx).await.expect("continue");
        assert!(cache.counter_value("spend:abc").await.unwrap_or(0.0) > 0.0);
        hook.post_failure(&ctx, &Error::Network("boom".into()))
            .await;
        let spent = cache.counter_value("spend:abc").await.unwrap_or(0.0);
        assert!(
            spent.abs() < 1e-9,
            "failure must release reservation, got {spent}"
        );
    }

    #[tokio::test]
    async fn post_success_unknown_model_releases_to_zero() {
        let cache = Arc::new(InMemoryCache::default());
        let hook = BudgetHook::with_duration(cache.clone(), Some(100.0), Duration::from_secs(60));
        let mut ctx = make_ctx();
        ctx.model = "made-up-model".to_string();
        ctx.request_body = std::sync::Arc::new(serde_json::json!({
        "model": "made-up-model",
        "messages": [{"role": "user", "content": "hi"}]
        }));
        hook.pre_call(&mut ctx).await.expect("continue");
        let resp = make_response("made-up-model", 100, 100);
        hook.post_success(&ctx, &resp).await;
        let spent = cache.counter_value("spend:abc").await.unwrap_or(0.0);
        assert!(
            spent.abs() < 1e-9,
            "unknown model should not leave residual reservation, got {spent}"
        );
    }

    #[tokio::test]
    async fn parse_budget_duration_from_str() {
        let cache: Arc<dyn Cache> = Arc::new(InMemoryCache::default());
        let hook = BudgetHook::new(cache, Some(10.0), "1h");
        assert_eq!(hook.budget_duration, Duration::from_secs(3600));
    }

    #[tokio::test]
    async fn per_key_budget_enforced_without_global() {
        let cache = Arc::new(InMemoryCache::default());
        let _ = cache.incr("spend:abc", 5.0, None).await;
        let hook = BudgetHook::with_duration(cache, None, Duration::from_secs(60));
        let mut ctx = make_ctx_with_budget(Some(5.0));
        let action = hook.pre_call(&mut ctx).await.expect("ok");
        assert!(matches!(action, HookAction::Reject(_)));
    }

    #[tokio::test]
    async fn master_key_bypasses_pre_call_budget_check() {
        let cache = Arc::new(InMemoryCache::default());
        let _ = cache.incr("spend:abc", 1_000_000.0, None).await;
        let hook = BudgetHook::with_duration(cache.clone(), Some(1.0), Duration::from_secs(60));
        let mut ctx = make_ctx_with_budget(Some(1.0));
        ctx.is_master = true;
        let action = hook.pre_call(&mut ctx).await.expect("ok");
        assert!(matches!(action, HookAction::Continue));
    }

    #[tokio::test]
    async fn master_key_skips_post_success_budget_increment() {
        let cache = Arc::new(InMemoryCache::default());
        let hook = BudgetHook::with_duration(cache.clone(), Some(100.0), Duration::from_secs(60));
        let mut ctx = make_ctx();
        ctx.is_master = true;
        let resp = make_response("gpt-4o", 1_000_000, 500_000);
        hook.post_success(&ctx, &resp).await;
        assert_eq!(cache.counter_value("spend:abc").await, None);
    }

    #[tokio::test]
    async fn cache_hit_skips_post_success_budget_increment() {
        let cache = Arc::new(InMemoryCache::default());
        let hook = BudgetHook::with_duration(cache.clone(), Some(100.0), Duration::from_secs(60));
        let mut ctx = make_ctx();
        ctx.cache_hit = true;
        let resp = make_response("gpt-4o", 1_000_000, 500_000);
        hook.post_success(&ctx, &resp).await;
        assert_eq!(cache.counter_value("spend:abc").await, None);
    }

    #[tokio::test]
    async fn cache_hit_releases_reservation() {
        // A reservation made in pre_call must not leak when the request is
        // later served from the response cache: post_success on a cache hit
        // releases the estimate so the counter returns to its
        // pre-reservation value.
        let cache = Arc::new(InMemoryCache::default());
        let hook = BudgetHook::with_duration(cache.clone(), Some(100.0), Duration::from_secs(60));
        let mut ctx = make_ctx();
        ctx.request_body = std::sync::Arc::new(serde_json::json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}]
        }));
        hook.pre_call(&mut ctx).await.expect("continue");
        assert!(
            cache.counter_value("spend:abc").await.unwrap_or(0.0) > 0.0,
            "pre_call must have reserved an estimate"
        );
        ctx.cache_hit = true;
        hook.post_success(&ctx, &make_response("gpt-4o", 1_000_000, 500_000))
            .await;
        let spent = cache.counter_value("spend:abc").await.unwrap_or(0.0);
        assert!(
            spent.abs() < 1e-9,
            "cache hit must release the reservation, got {spent}"
        );
    }

    #[test]
    fn most_restrictive_picks_lower() {
        assert_eq!(most_restrictive(Some(1.0), Some(2.0)), Some(1.0));
        assert_eq!(most_restrictive(Some(2.0), Some(1.0)), Some(1.0));
        assert_eq!(most_restrictive(Some(1.0), None), Some(1.0));
        assert_eq!(most_restrictive(None, Some(2.0)), Some(2.0));
        assert_eq!(most_restrictive(None, None), None);
    }

    #[tokio::test]
    async fn user_budget_duration_uses_user_window_not_key() {
        let cache = Arc::new(InMemoryCache::default());
        let hook = BudgetHook::with_duration(cache.clone(), None, Duration::from_secs(3600));
        let mut ctx = make_ctx();
        ctx.user_id = Some("u1".to_string());
        ctx.user_max_budget = Some(1.0);
        ctx.user_budget_duration = Some("30s".to_string());
        ctx.request_body = std::sync::Arc::new(serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 10
        }));
        hook.pre_call(&mut ctx).await.expect("ok");
        // Counter must exist with a short TTL window; a second reserve after
        // sleeping would be flaky — just assert the user key was written.
        let spend = cache.counter_value("spend:user:u1").await.unwrap_or(0.0);
        assert!(
            spend > 0.0,
            "user budget reserve must debit spend:user:u1, got {spend}"
        );
        assert!(ctx.budget_estimate.is_some());
    }

    /// After pick stamps a markup override, top-up must increase the
    /// reservation (and fail closed if the ceiling cannot cover the delta).
    #[tokio::test]
    async fn top_up_for_deployment_pricing_increases_reservation() {
        let cache = Arc::new(InMemoryCache::default());
        let hook = BudgetHook::with_duration(cache.clone(), Some(100.0), Duration::from_secs(60));
        let mut ctx = make_ctx();
        ctx.request_body = std::sync::Arc::new(serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100
        }));
        // Reserve at table rates (no override yet).
        let action = hook.pre_call(&mut ctx).await.expect("ok");
        assert!(matches!(action, HookAction::Continue));
        let base = ctx.budget_estimate.expect("estimate");
        let spent_before = cache.counter_value("spend:abc").await.unwrap_or(0.0);

        // Stamp a 10x markup override and top up.
        ctx.deployment_pricing = Some(oxllm_core::DeploymentPricing {
            input_cost_per_token: 0.0001,
            output_cost_per_token: 0.0003,
            cache_read_input_token_cost: None,
            cache_creation_input_token_cost: None,
            output_cost_per_reasoning_token: None,
            ..Default::default()
        });
        hook.top_up_for_deployment_pricing(&mut ctx)
            .await
            .expect("top-up ok");
        let after = ctx.budget_estimate.expect("estimate after");
        assert!(
            after > base + 1e-12,
            "top-up must raise estimate: base={base} after={after}"
        );
        let spent_after = cache.counter_value("spend:abc").await.unwrap_or(0.0);
        assert!(
            spent_after > spent_before + 1e-12,
            "top-up must debit the delta: before={spent_before} after={spent_after}"
        );
    }

    /// A stale receipt at top-up is a transient admission race, not an
    /// exhausted cap: the failure must classify as retryable rate limiting so
    /// SDKs do not treat it like a terminal `budget_exceeded` verdict.
    #[tokio::test]
    async fn top_up_after_window_replacement_fails_with_retryable_rate_limit() {
        let cache = Arc::new(InMemoryCache::default());
        let hook = BudgetHook::with_duration(cache.clone(), Some(100.0), Duration::from_secs(60));
        let mut ctx = make_ctx();
        ctx.request_body = std::sync::Arc::new(serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100
        }));
        hook.pre_call(&mut ctx).await.expect("ok");
        let base = ctx.budget_estimate.expect("estimate");

        // The budget window rolls over mid-flight and another request recreates
        // the counter in a fresh incarnation. The held receipt is now stale.
        cache.expire_counter_for_test("spend:abc").await;
        cache
            .incr_if_below("spend:abc", 0.01, 100.0, Some(Duration::from_secs(60)))
            .await
            .unwrap();

        ctx.deployment_pricing = Some(oxllm_core::DeploymentPricing {
            input_cost_per_token: 0.0001,
            output_cost_per_token: 0.0003,
            ..Default::default()
        });
        let error = hook
            .top_up_for_deployment_pricing(&mut ctx)
            .await
            .expect_err("stale receipt must reject");
        assert!(
            matches!(error, Error::RateLimited(ref message) if message.contains("window changed")),
            "expected retryable RateLimited, got {error:?}"
        );
        // Fail-closed without side effects: the estimate is untouched and the
        // replacement request's counter was not debited or rolled back.
        assert_eq!(ctx.budget_estimate, Some(base));
        assert_eq!(cache.counter_value("spend:abc").await, Some(0.01));
    }

    /// LOW (C31): `top_up_for_deployment_pricing` must pass each scope its OWN
    /// resolved TTL (matching `pre_call`), not the key's window. With a store
    /// that resets TTL on every increment, threading the key TTL into the team
    /// counter would re-stamp the key's (longer) window onto it. This records
    /// the TTL `incr_if_below` receives for the team key and asserts it is the
    /// team window.
    #[tokio::test]
    async fn top_up_uses_team_window_for_team_counter() {
        // A recording cache that delegates to an in-memory store for
        // correctness but captures the TTL handed to `incr_if_below` per key.
        // Unlike InMemoryCache (which preserves an existing key's TTL on
        // incr), this cache records exactly what the hook passed so the
        // per-scope TTL wiring is observable.
        #[derive(Default)]
        struct TtlRecordingCache {
            inner: InMemoryCache,
            seen: std::sync::Mutex<HashMap<String, Duration>>,
        }

        #[async_trait]
        impl Cache for TtlRecordingCache {
            async fn get(&self, key: &str) -> Option<String> {
                self.inner.get(key).await
            }
            async fn set(&self, key: &str, value: String, ttl: Option<Duration>) {
                self.inner.set(key, value, ttl).await;
            }
            async fn delete(&self, key: &str) {
                self.inner.delete(key).await;
            }
            async fn incr(&self, key: &str, by: f64, ttl: Option<Duration>) -> Result<f64, Error> {
                self.inner.incr(key, by, ttl).await
            }
            async fn incr_if_below(
                &self,
                key: &str,
                delta: f64,
                ceiling: f64,
                ttl: Option<Duration>,
            ) -> Result<IncrResult, Error> {
                if let Some(d) = ttl {
                    self.seen
                        .lock()
                        .expect("seen lock")
                        .insert(key.to_string(), d);
                }
                self.inner.incr_if_below(key, delta, ceiling, ttl).await
            }
            async fn counter_value(&self, key: &str) -> Option<f64> {
                self.inner.counter_value(key).await
            }

            async fn incr_if_below_in_window(
                &self,
                key: &str,
                delta: f64,
                ceiling: f64,
                ttl: Option<Duration>,
                expected: Option<&CounterWindow>,
            ) -> Result<WindowedIncrResult, Error> {
                if let Some(ttl) = ttl {
                    self.seen.lock().unwrap().insert(key.to_string(), ttl);
                }
                self.inner
                    .incr_if_below_in_window(key, delta, ceiling, ttl, expected)
                    .await
            }

            async fn incr_in_window(
                &self,
                key: &str,
                delta: f64,
                window: &CounterWindow,
            ) -> Result<bool, Error> {
                self.inner.incr_in_window(key, delta, window).await
            }
        }

        let cache = Arc::new(TtlRecordingCache::default());
        let hook = BudgetHook::with_duration(cache.clone(), Some(100.0), Duration::from_secs(60));
        let mut ctx = make_ctx();
        ctx.team_id = Some("t-win".to_string());
        ctx.team_max_budget = Some(100.0);
        ctx.budget_duration = Some("30d".to_string());
        ctx.team_budget_duration = Some("1d".to_string());
        ctx.request_body = std::sync::Arc::new(serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100
        }));
        hook.pre_call(&mut ctx).await.expect("continue");
        // Stamp a markup override so top-up reserves a positive delta.
        ctx.deployment_pricing = Some(oxllm_core::DeploymentPricing {
            input_cost_per_token: 0.0001,
            output_cost_per_token: 0.0003,
            cache_read_input_token_cost: None,
            cache_creation_input_token_cost: None,
            output_cost_per_reasoning_token: None,
            ..Default::default()
        });
        hook.top_up_for_deployment_pricing(&mut ctx)
            .await
            .expect("top-up ok");

        let key_window = crate::scheduler::parse_budget_duration("30d");
        let team_window = crate::scheduler::parse_budget_duration("1d");
        assert_ne!(key_window, team_window, "test premise: windows must differ");
        let team_ttl_seen = *cache
            .seen
            .lock()
            .expect("seen lock")
            .get("spend:team:t-win")
            .expect("top-up must reserve against the team key");
        assert_eq!(
            team_ttl_seen, team_window,
            "top-up must pass the team window ({team_window:?}) to the team counter, not the key window ({key_window:?})"
        );
    }

    #[tokio::test]
    async fn concurrent_pre_call_never_exceeds_budget() {
        let cache = Arc::new(InMemoryCache::default());
        // Tiny budget so only a few estimates fit.
        let hook = Arc::new(BudgetHook::with_duration(
            cache.clone(),
            Some(0.5),
            Duration::from_secs(60),
        ));
        let mut handles = Vec::new();
        for _ in 0..16 {
            let hook = hook.clone();
            handles.push(tokio::spawn(async move {
                let mut ctx = make_ctx();
                ctx.request_body = std::sync::Arc::new(serde_json::json!({
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hi there friend"}]
                }));
                hook.pre_call(&mut ctx).await
            }));
        }
        let mut allowed = 0u32;
        for h in handles {
            if matches!(h.await.expect("join").expect("ok"), HookAction::Continue) {
                allowed += 1;
            }
        }
        let spent = cache.counter_value("spend:abc").await.unwrap_or(0.0);
        assert!(
            spent <= 0.5 + 1e-9,
            "atomic reserve must not exceed ceiling: spent={spent}, allowed={allowed}"
        );
        assert!(allowed >= 1, "at least one request should fit");
    }

    #[tokio::test]
    async fn team_budget_independent_of_key_budget() {
        let cache = Arc::new(InMemoryCache::default());
        // Key has plenty of room; team is exhausted.
        let hook = BudgetHook::with_duration(cache.clone(), Some(100.0), Duration::from_secs(60));
        let _ = cache.incr("spend:team:t1", 10.0, None).await;
        let mut ctx = make_ctx();
        ctx.team_id = Some("t1".to_string());
        ctx.max_budget = Some(100.0);
        ctx.team_max_budget = Some(10.0);
        let action = hook.pre_call(&mut ctx).await.expect("ok");
        assert!(
            matches!(action, HookAction::Reject(_)),
            "team ceiling alone must block"
        );
    }

    #[tokio::test]
    async fn team_spend_not_checked_without_team_limit() {
        let cache = Arc::new(InMemoryCache::default());
        // Team counter high but no team_max_budget → only key limit applies.
        let _ = cache.incr("spend:team:t1", 1_000.0, None).await;
        let hook = BudgetHook::with_duration(cache.clone(), Some(100.0), Duration::from_secs(60));
        let mut ctx = make_ctx();
        ctx.team_id = Some("t1".to_string());
        ctx.max_budget = Some(100.0);
        // team_max_budget is None
        let action = hook.pre_call(&mut ctx).await.expect("ok");
        assert!(
            matches!(action, HookAction::Continue),
            "without team_max_budget, team counter must not block"
        );
    }

    #[tokio::test]
    async fn two_keys_share_team_spend_bucket() {
        let cache = Arc::new(InMemoryCache::default());
        let hook = BudgetHook::with_duration(cache.clone(), None, Duration::from_secs(60));

        let mut ctx_a = make_ctx();
        ctx_a.team_id = Some("t1".to_string());
        ctx_a.api_key_hash = "keyA".to_string();
        ctx_a.team_max_budget = Some(100.0);
        ctx_a.request_body = std::sync::Arc::new(serde_json::json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "a"}]
        }));
        hook.pre_call(&mut ctx_a).await.expect("ok");
        hook.post_success(&ctx_a, &make_response("gpt-4o", 1_000_000, 500_000))
            .await;

        let mut ctx_b = make_ctx();
        ctx_b.team_id = Some("t1".to_string());
        ctx_b.api_key_hash = "keyB".to_string();
        ctx_b.team_max_budget = Some(100.0);
        ctx_b.request_body = std::sync::Arc::new(serde_json::json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "b"}]
        }));
        hook.pre_call(&mut ctx_b).await.expect("ok");
        hook.post_success(&ctx_b, &make_response("gpt-4o", 1_000_000, 500_000))
            .await;

        let team = cache.counter_value("spend:team:t1").await.unwrap_or(0.0);
        assert!(team > 0.0);
        // Exhaust team budget.
        let _ = cache.incr("spend:team:t1", 100.0, None).await;
        let action = hook.pre_call(&mut ctx_b).await.expect("ok");
        assert!(matches!(action, HookAction::Reject(_)));
    }

    #[tokio::test]
    async fn user_budget_shared_across_keys() {
        let cache = Arc::new(InMemoryCache::default());
        let hook = BudgetHook::with_duration(cache.clone(), None, Duration::from_secs(60));
        let _ = cache.incr("spend:user:u1", 5.0, None).await;
        let mut ctx = make_ctx();
        ctx.user_id = Some("u1".to_string());
        ctx.user_max_budget = Some(5.0);
        let action = hook.pre_call(&mut ctx).await.expect("ok");
        assert!(matches!(action, HookAction::Reject(_)));
    }

    /// Customer / end-user budget uses `spend:customer:{end_user_id}` and is
    /// independent of the virtual-key owner (`user_id`).
    #[tokio::test]
    async fn customer_budget_reserves_and_rejects_when_exhausted() {
        let cache = Arc::new(InMemoryCache::default());
        let hook = BudgetHook::with_duration(cache.clone(), None, Duration::from_secs(60));
        let mut ctx = make_ctx();
        ctx.end_user_id = Some("eu-alice".to_string());
        ctx.customer_max_budget = Some(1.0);
        ctx.request_body = std::sync::Arc::new(serde_json::json!({
            "model": "gpt-4o",
            "user": "eu-alice",
            "messages": [{"role": "user", "content": "hi"}]
        }));
        let action = hook.pre_call(&mut ctx).await.expect("ok");
        assert!(matches!(action, HookAction::Continue));
        let reserved = cache
            .counter_value("spend:customer:eu-alice")
            .await
            .unwrap_or(0.0);
        assert!(
            reserved > 0.0,
            "customer budget must reserve on spend:customer:{{id}}, got {reserved}"
        );

        // Exhaust the customer counter and ensure a subsequent request rejects.
        let _ = cache.incr("spend:customer:eu-alice", 10.0, None).await;
        let mut ctx2 = make_ctx();
        ctx2.end_user_id = Some("eu-alice".to_string());
        ctx2.customer_max_budget = Some(1.0);
        ctx2.request_body = ctx.request_body.clone();
        let action = hook.pre_call(&mut ctx2).await.expect("ok");
        assert!(
            matches!(action, HookAction::Reject(_)),
            "exhausted customer budget must reject"
        );
    }

    #[tokio::test]
    async fn customer_budget_released_on_failure() {
        let cache = Arc::new(InMemoryCache::default());
        let hook = BudgetHook::with_duration(cache.clone(), None, Duration::from_secs(60));
        let mut ctx = make_ctx();
        ctx.end_user_id = Some("eu-bob".to_string());
        ctx.customer_max_budget = Some(10.0);
        ctx.request_body = std::sync::Arc::new(serde_json::json!({
            "model": "gpt-4o",
            "user": "eu-bob",
            "messages": [{"role": "user", "content": "hi"}]
        }));
        hook.pre_call(&mut ctx).await.expect("continue");
        let reserved = cache
            .counter_value("spend:customer:eu-bob")
            .await
            .unwrap_or(0.0);
        assert!(reserved > 0.0);
        hook.post_failure(
            &ctx,
            &Error::Upstream {
                status: 500,
                message: "boom".into(),
            },
        )
        .await;
        let after = cache
            .counter_value("spend:customer:eu-bob")
            .await
            .unwrap_or(0.0);
        assert!(
            after < 1e-12,
            "customer reservation must release on failure, got {after}"
        );
    }

    /// Multi-tag requests reserve against EVERY carried tag's own budget
    /// counter: one shared key tagging two applications accrues in both
    /// `spend:tag:{app}` ledgers, and exhausting either rejects.
    #[tokio::test]
    async fn multi_tag_budgets_reserve_every_carried_tag() {
        let cache = Arc::new(InMemoryCache::default());
        let hook = BudgetHook::with_duration(cache.clone(), None, Duration::from_secs(60));
        let mut ctx = make_ctx();
        ctx.request_tags = vec!["app-a".to_string(), "env-prod".to_string()];
        ctx.request_tag = Some("app-a".to_string());
        ctx.tag_budgets = vec![("app-a".to_string(), 1.0), ("env-prod".to_string(), 5.0)];
        ctx.tag_max_budget = Some(1.0);
        ctx.request_body = std::sync::Arc::new(serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}]
        }));
        let action = hook.pre_call(&mut ctx).await.expect("ok");
        assert!(matches!(action, HookAction::Continue));
        for tag in ["app-a", "env-prod"] {
            let reserved = cache
                .counter_value(&format!("spend:tag:{tag}"))
                .await
                .unwrap_or(0.0);
            assert!(reserved > 0.0, "tag {tag} must have its own reservation");
        }

        // Exhaust ONLY app-a: the next tagged request must be rejected with
        // the tag named, and env-prod's counter untouched by the release.
        let _ = cache.incr("spend:tag:app-a", 10.0, None).await;
        let mut ctx2 = make_ctx();
        ctx2.request_tags = ctx.request_tags.clone();
        ctx2.request_tag = ctx.request_tag.clone();
        ctx2.tag_budgets = ctx.tag_budgets.clone();
        ctx2.tag_max_budget = Some(1.0);
        ctx2.request_body = ctx.request_body.clone();
        let action = hook.pre_call(&mut ctx2).await.expect("ok");
        match action {
            HookAction::Reject(err) => {
                let msg = format!("{err}");
                assert!(msg.contains("app-a"), "rejection must name the tag: {msg}");
            }
            other => panic!("expected rejection, got {other:?}"),
        }
    }

    /// Tag budgets use `spend:tag:{name}` for normalized request tags.
    #[tokio::test]
    async fn tag_budget_reserves_and_rejects_when_exhausted() {
        let cache = Arc::new(InMemoryCache::default());
        let hook = BudgetHook::with_duration(cache.clone(), None, Duration::from_secs(60));
        let mut ctx = make_ctx();
        ctx.request_tag = Some("canary".to_string());
        ctx.tag_max_budget = Some(1.0);
        ctx.request_body = std::sync::Arc::new(serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}]
        }));
        let action = hook.pre_call(&mut ctx).await.expect("ok");
        assert!(matches!(action, HookAction::Continue));
        let reserved = cache.counter_value("spend:tag:canary").await.unwrap_or(0.0);
        assert!(
            reserved > 0.0,
            "tag budget must reserve on spend:tag:{{name}}, got {reserved}"
        );

        let _ = cache.incr("spend:tag:canary", 10.0, None).await;
        let mut ctx2 = make_ctx();
        ctx2.request_tag = Some("canary".to_string());
        ctx2.tag_max_budget = Some(1.0);
        ctx2.request_body = ctx.request_body.clone();
        let action = hook.pre_call(&mut ctx2).await.expect("ok");
        assert!(
            matches!(action, HookAction::Reject(_)),
            "exhausted tag budget must reject"
        );
    }

    #[tokio::test]
    async fn estimate_max_request_cost_uses_output_rate_for_known_model() {
        let mut ctx = make_ctx();
        ctx.request_body = std::sync::Arc::new(serde_json::json!({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 100
        }));
        let est = estimate_max_request_cost(&ctx).await;
        assert!(
            est >= 100.0 * 0.00001,
            "estimate must include requested output cost, got {est}"
        );
    }

    #[tokio::test]
    async fn estimate_max_request_cost_falls_back_for_unknown_model() {
        let mut ctx = make_ctx();
        ctx.model = "made-up-model".to_string();
        ctx.request_body = std::sync::Arc::new(serde_json::json!({
        "model": "made-up-model",
        "messages": [{"role": "user", "content": "hi"}]
        }));
        let est = estimate_max_request_cost(&ctx).await;
        assert_eq!(est, 0.01);
    }

    /// When a deployment pricing override is stashed on the context, the
    /// pre-call reservation must be sized from the OVERRIDE rates (matching
    /// post-success settlement) rather than the cheaper embedded table, so a
    /// marked-up deployment cannot slip concurrent requests under the ceiling.
    #[tokio::test]
    async fn estimate_max_request_cost_uses_deployment_pricing_override() {
        let body = std::sync::Arc::new(serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 100
        }));

        let mut ctx_table = make_ctx();
        ctx_table.request_body = body.clone();
        let table_est = estimate_max_request_cost(&ctx_table).await;

        let mut ctx_override = make_ctx();
        ctx_override.request_body = body;
        ctx_override.deployment_pricing = Some(oxllm_core::DeploymentPricing {
            input_cost_per_token: 0.5,
            output_cost_per_token: 0.5,
            ..Default::default()
        });
        let override_est = estimate_max_request_cost(&ctx_override).await;

        assert!(
            override_est > table_est,
            "override estimate ({override_est}) must exceed table estimate ({table_est})"
        );
        // Sanity: the override must actually reflect the override rates, not
        // the table rates (0.5/token dwarfs gpt-4o's 2.5e-6).
        assert!(
            override_est > 100.0 * 0.5,
            "override estimate must be computed from the override output rate, got {override_est}"
        );
    }

    /// Issue: the pre-call reservation must be tier-aware. gpt-5.6's >272k
    /// prompt rates are ~2x the base rates; a request whose estimated prompt
    /// exceeds the threshold must reserve on the tier rates so a concurrent
    /// burst cannot collectively overshoot the ceiling by the un-reserved
    /// delta.
    #[tokio::test]
    async fn estimate_max_request_cost_sizes_tiered_models_on_estimated_prompt() {
        let mut ctx = make_ctx();
        ctx.model = "gpt-5.6".to_string();
        // Pre-seed the prompt-token estimate above the 272k tier threshold
        // (as RateLimitHook would have done).
        let _ = ctx.token_estimate.set(300_000);
        ctx.request_body = std::sync::Arc::new(serde_json::json!({
            "model": "gpt-5.6",
            "messages": [{"role": "user", "content": "x"}],
        }));
        let est = estimate_max_request_cost(&ctx).await;

        // Base rates would reserve 300_000 * 4e-6 == 1.2 for input; the
        // tier-aware reservation uses 300_000 * 8e-6 == 2.4.
        assert!(
            est > 300_000.0 * 4e-6,
            "estimate must use the >272k tier input rate: got {est}"
        );
        // The reservation must cover the actual cost of a 272k-prompt response
        // (worst case: a prompt that just crosses the tier boundary), plus the
        // default output estimate at the tier output rate.
        let actual_272k_prompt = 272_001.0 * 8e-6 + 16_384.0 * 3e-5;
        assert!(
            est >= actual_272k_prompt,
            "tier-aware reservation ({est}) must cover a 272k-prompt response ({actual_272k_prompt})"
        );
    }

    /// LOW: the reservation must also cover the cache buckets, which settle
    /// on top of the prompt estimate. For a tiered model whose tiered
    /// cache-creation rate exceeds its input rate (gpt-5.6 >272k: 1e-5 vs
    /// 8e-6), a cache-write-heavy request would settle above a
    /// `prompt * input + output * output` reservation. The worst-case
    /// reservation must cover a full-prompt cache write.
    #[tokio::test]
    async fn estimate_reserves_tiered_cache_creation_worst_case() {
        let mut ctx = make_ctx();
        ctx.model = "gpt-5.6".to_string();
        let _ = ctx.token_estimate.set(300_000);
        ctx.request_body = std::sync::Arc::new(serde_json::json!({
            "model": "gpt-5.6",
            "messages": [{"role": "user", "content": "x"}],
        }));
        let est = estimate_max_request_cost(&ctx).await;

        // Worst-case settled cost for the estimated sizes: the full 300k
        // prompt billed at the >272k tier input rate (8e-6) PLUS a full-prompt
        // cache write at the tier cache-creation rate (1e-5), and the
        // default output estimate at the tier output rate (3e-5).
        let settled = 300_000.0 * 8e-6 + 300_000.0 * 1e-5 + 16_384.0 * 3e-5;
        assert!(
            est + 1e-6 >= settled,
            "reservation ({est}) must cover a cache-write-heavy tiered request ({settled})"
        );
        // Sanity: the plain input+output reservation would have undershot.
        let plain = 300_000.0 * 8e-6 + 16_384.0 * 3e-5;
        assert!(
            est > plain,
            "reservation must include the additive cache-creation bucket, got {est}"
        );
    }

    /// LOW (documented trade-off): the pre-call reservation sizes the tier on
    /// `prompt_estimate` ALONE because cache-creation counts are unknowable
    /// before the upstream responds (`rates_for_prompt(prompt_estimate, 0)`).
    /// For a tiered model whose context crosses the threshold ONLY via
    /// cache-creation tokens, the reservation uses the cheaper BASE rate
    /// while settle bills the TIER rate. This is the cleanest manifestation
    /// on a shipped model: `gemini/gemini-2.5-pro` declares NO base
    /// cache-creation rate (so the reservation's cache overlay is 0.0), and
    /// a 180k fresh prompt + 30k cache-write pushes the true context to 210k
    /// (>200k tier). The reservation under-reserves by ~2x; settle recovers
    /// the full tier cost so lifetime spend is correct. See
    /// `estimate_max_request_cost` doc.
    #[tokio::test]
    async fn budget_reservation_uses_base_rate_when_cache_creation_unknown() {
        let cache = Arc::new(InMemoryCache::default());
        let hook = BudgetHook::with_duration(cache.clone(), Some(1_000.0), Duration::from_secs(60));

        let pricing = oxllm_core::lookup_pricing("gemini/gemini-2.5-pro")
            .expect("gemini/gemini-2.5-pro priced");
        assert!(
            pricing.cache_creation_input_token_cost.is_none(),
            "test premise: gemini-2.5-pro must declare NO base cache-creation rate"
        );
        assert!(
            pricing.tiers.iter().any(|t| t.threshold_tokens == 200_000),
            "test premise: gemini-2.5-pro must have a 200k tier"
        );

        // 180k fresh prompt: BELOW the 200k tier on its own, so the
        // reservation sizes on the BASE input rate (1.25e-6). `max_tokens: 0`
        // zeroes the output estimate so the under-reserve is observable on
        // the prompt-side bucket alone.
        let fresh_prompt = 180_000u64;
        let cache_creation = 30_000u64;
        let mut ctx = make_ctx();
        ctx.model = "gemini/gemini-2.5-pro".to_string();
        let _ = ctx.token_estimate.set(fresh_prompt);
        ctx.request_body = std::sync::Arc::new(serde_json::json!({
            "model": "gemini/gemini-2.5-pro",
            "messages": [{"role": "user", "content": "x"}],
            "max_tokens": 0,
        }));
        hook.pre_call(&mut ctx).await.expect("continue");
        let reserved = cache
            .counter_value("spend:abc")
            .await
            .expect("reservation stashed");

        // The reservation is sized on the BASE rates (no base cache-creation
        // rate → cache overlay is 0.0, so prompt_side = base input only).
        let base_rates = pricing.rates_for_prompt(fresh_prompt, 0);
        let expected_reservation = fresh_prompt as f64
            * (base_rates
                .input
                .max(base_rates.cache_read)
                .max(base_rates.audio)
                + base_rates.cache_creation.max(base_rates.cache_creation_1h));
        assert!(
            (reserved - expected_reservation).abs() < 1e-6,
            "reservation must be sized on the BASE rate (cache-creation unknown): \
             got {reserved}, expected ~{expected_reservation}"
        );

        // The response arrives with 30k cache-creation tokens, making the
        // true context 210k (>200k tier). Settle must debit the FULL tier
        // cost so the final counter equals the actual spend.
        let resp = ResponseSummary {
            model: "gemini-2.5-pro".to_string(),
            cost: None,
            usage: Some(UsageSummary {
                prompt_tokens: fresh_prompt,
                completion_tokens: 0,
                total_tokens: fresh_prompt,
                cache_read_input_tokens: None,
                cache_creation_input_tokens: Some(cache_creation),
                audio_input_tokens: None,
                reasoning_tokens: None,
                citation_tokens: None,
                web_search_requests: None,
                browser_open_requests: None,
                cache_creation_1h_input_tokens: None,
            }),
            body: None,
            provider: "gemini".to_string(),
        };
        hook.post_success(&ctx, &resp).await;
        let spent = cache.counter_value("spend:abc").await.unwrap_or(0.0);

        // The actual cost uses the TIER rates (context = 180k + 30k = 210k
        // > 200k): fresh input at 2.5e-6, cache-write at the tier's 2.5e-7
        // cache-creation rate.
        let tier_rates = pricing.rates_for_prompt(fresh_prompt, cache_creation);
        let actual = fresh_prompt as f64 * tier_rates.input
            + cache_creation as f64 * tier_rates.cache_creation;
        assert!(
            (spent - actual).abs() < 1e-6,
            "settle must recover the full tier cost: spent={spent}, actual={actual}, \
             reserved was {reserved}"
        );

        // The documented trade-off: because cache-creation is unknown
        // pre-call, the reservation sizes on the base input rate (1.25e-6)
        // while the actual bills the tier input rate (2.5e-6) — the
        // reservation under-reserves by ~2x for such requests. If they ever
        // match, the estimator learned about cache-creation pre-call (good!)
        // and this assertion should be replaced with one asserting parity.
        assert!(
            actual > reserved,
            "the documented trade-off: a tiered cache-heavy request under-reserves \
             (actual {actual} should exceed reservation {reserved})"
        );
    }

    fn hook_with_model_budget(
        cache: Arc<InMemoryCache>,
        model_max_budget: HashMap<String, f64>,
    ) -> BudgetHook {
        BudgetHook::with_duration(cache, None, Duration::from_secs(60))
            .with_model_budget(model_max_budget)
    }

    #[tokio::test]
    async fn model_budget_enforced_across_different_keys() {
        let cache = Arc::new(InMemoryCache::default());
        let mut budgets = HashMap::new();
        budgets.insert("gpt-4o".to_string(), 10.0);
        let hook = hook_with_model_budget(cache.clone(), budgets);

        // First key reserves successfully against `spend:model:gpt-4o`.
        let mut ctx_a = make_ctx();
        ctx_a.api_key_hash = "keyA".to_string();
        ctx_a.request_body = std::sync::Arc::new(serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 10
        }));
        let action = hook.pre_call(&mut ctx_a).await.expect("ok");
        assert!(matches!(action, HookAction::Continue));
        let model_a = cache
            .counter_value("spend:model:gpt-4o")
            .await
            .unwrap_or(0.0);
        assert!(model_a > 0.0, "model counter must be reserved");

        // A different key calling the same model must see the same global
        // counter and be rejected once the model ceiling is exhausted.
        let mut ctx_b = make_ctx();
        ctx_b.api_key_hash = "keyB".to_string();
        ctx_b.request_body = std::sync::Arc::new(serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 10
        }));
        // Push the counter past the ceiling so any new reservation fails.
        let _ = cache.incr("spend:model:gpt-4o", 100.0, None).await;
        let action = hook.pre_call(&mut ctx_b).await.expect("ok");
        match action {
            HookAction::Reject(Error::BudgetExceeded(msg)) => {
                assert!(
                    msg.contains("model budget") && msg.contains("gpt-4o"),
                    "rejection message must name the model budget, got: {msg}"
                );
            }
            other => panic!("expected Reject, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn model_budget_independent_of_key_budget() {
        let cache = Arc::new(InMemoryCache::default());
        let mut budgets = HashMap::new();
        budgets.insert("gpt-4o".to_string(), 0.01);
        let hook = BudgetHook::with_duration(cache.clone(), Some(1_000.0), Duration::from_secs(60))
            .with_model_budget(budgets);

        // Key budget is huge, model budget is tiny → model ceiling blocks.
        let mut ctx = make_ctx_with_budget(Some(1_000.0));
        ctx.request_body = std::sync::Arc::new(serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi there friend"}]
        }));
        let action = hook.pre_call(&mut ctx).await.expect("ok");
        assert!(
            matches!(action, HookAction::Reject(_)),
            "model budget must block even when key budget has room"
        );

        // key spend must be rolled back when the model reservation fails.
        let key_spend = cache.counter_value("spend:abc").await.unwrap_or(0.0);
        assert!(
            key_spend.abs() < 1e-9,
            "failed model reservation must roll back key reservation, got {key_spend}"
        );
    }

    #[tokio::test]
    async fn model_budget_reservation_released_on_failure() {
        let cache = Arc::new(InMemoryCache::default());
        let mut budgets = HashMap::new();
        budgets.insert("gpt-4o".to_string(), 100.0);
        let hook = hook_with_model_budget(cache.clone(), budgets);

        let mut ctx = make_ctx();
        ctx.request_body = std::sync::Arc::new(serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}]
        }));
        hook.pre_call(&mut ctx).await.expect("continue");
        assert!(
            cache
                .counter_value("spend:model:gpt-4o")
                .await
                .unwrap_or(0.0)
                > 0.0
        );
        hook.post_failure(&ctx, &Error::Network("boom".into()))
            .await;
        let spent = cache
            .counter_value("spend:model:gpt-4o")
            .await
            .unwrap_or(0.0);
        assert!(
            spent.abs() < 1e-9,
            "failure must release model reservation, got {spent}"
        );
    }

    #[tokio::test]
    async fn model_budget_not_checked_when_not_configured() {
        let cache = Arc::new(InMemoryCache::default());
        // No model_max_budget configured → backward-compat behavior.
        let hook = BudgetHook::with_duration(cache.clone(), None, Duration::from_secs(60));
        let mut ctx = make_ctx();
        ctx.request_body = std::sync::Arc::new(serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}]
        }));
        let action = hook.pre_call(&mut ctx).await.expect("ok");
        assert!(matches!(action, HookAction::Continue));
        assert_eq!(cache.counter_value("spend:model:gpt-4o").await, None);
    }

    #[tokio::test]
    async fn multiple_models_have_independent_budgets() {
        let cache = Arc::new(InMemoryCache::default());
        let mut budgets = HashMap::new();
        budgets.insert("gpt-4o".to_string(), 0.01);
        budgets.insert("claude-3".to_string(), 100.0);
        let hook = hook_with_model_budget(cache.clone(), budgets);

        // gpt-4o ceiling is tiny → first request with substantial estimate
        // should fail.
        let mut ctx_gpt = make_ctx();
        ctx_gpt.model = "gpt-4o".to_string();
        ctx_gpt.request_body = std::sync::Arc::new(serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi there friend"}]
        }));
        let action_gpt = hook.pre_call(&mut ctx_gpt).await.expect("ok");
        assert!(matches!(action_gpt, HookAction::Reject(_)));

        // claude-3 has a generous budget → must succeed independently.
        let mut ctx_claude = make_ctx();
        ctx_claude.model = "claude-3".to_string();
        ctx_claude.request_body = std::sync::Arc::new(serde_json::json!({
            "model": "claude-3",
            "messages": [{"role": "user", "content": "hi"}]
        }));
        let action_claude = hook.pre_call(&mut ctx_claude).await.expect("ok");
        assert!(
            matches!(action_claude, HookAction::Continue),
            "claude-3 budget is independent of gpt-4o and must allow the call"
        );
    }

    #[tokio::test]
    async fn model_budget_master_key_bypasses() {
        let cache = Arc::new(InMemoryCache::default());
        let _ = cache.incr("spend:model:gpt-4o", 1_000_000.0, None).await;
        let mut budgets = HashMap::new();
        budgets.insert("gpt-4o".to_string(), 1.0);
        let hook = hook_with_model_budget(cache, budgets);

        let mut ctx = make_ctx();
        ctx.is_master = true;
        ctx.request_body = std::sync::Arc::new(serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}]
        }));
        let action = hook.pre_call(&mut ctx).await.expect("ok");
        assert!(
            matches!(action, HookAction::Continue),
            "master key must bypass per-model budget"
        );
    }

    #[tokio::test]
    async fn model_budget_post_success_adjusts_to_actual() {
        let cache = Arc::new(InMemoryCache::default());
        let mut budgets = HashMap::new();
        budgets.insert("gpt-4o".to_string(), 100.0);
        let hook = hook_with_model_budget(cache.clone(), budgets);

        let mut ctx = make_ctx();
        ctx.request_body = std::sync::Arc::new(serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}]
        }));
        hook.pre_call(&mut ctx).await.expect("continue");
        let resp = make_response("gpt-4o", 1_000_000, 500_000);
        hook.post_success(&ctx, &resp).await;

        let spent = cache
            .counter_value("spend:model:gpt-4o")
            .await
            .unwrap_or(0.0);
        let expected = 1_000_000.0 * 0.0000025 + 500_000.0 * 0.00001;
        assert!(
            (spent - expected).abs() < 1e-6,
            "model spend must reflect actual cost ({expected}), got {spent}"
        );
    }

    #[tokio::test]
    async fn model_budget_rolls_back_key_reservation_on_failure_path() {
        // When the per-model reservation fails AFTER a key reservation was
        // made, the key reservation must be released (same rollback pattern
        // used for team/user).
        let cache = Arc::new(InMemoryCache::default());
        let mut budgets = HashMap::new();
        budgets.insert("gpt-4o".to_string(), 0.001);
        let hook = BudgetHook::with_duration(cache.clone(), Some(1_000.0), Duration::from_secs(60))
            .with_model_budget(budgets);

        let mut ctx = make_ctx_with_budget(Some(1_000.0));
        ctx.request_body = std::sync::Arc::new(serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi there friend"}]
        }));
        let action = hook.pre_call(&mut ctx).await.expect("ok");
        assert!(matches!(action, HookAction::Reject(_)));
        let key_spend = cache.counter_value("spend:abc").await.unwrap_or(0.0);
        assert!(
            key_spend.abs() < 1e-9,
            "key reservation must be released when model budget rejects, got {key_spend}"
        );
    }

    #[tokio::test]
    async fn org_budget_independent_of_team_and_key_budget() {
        let cache = Arc::new(InMemoryCache::default());
        // Key and team have plenty of room; org is exhausted.
        let hook = BudgetHook::with_duration(cache.clone(), Some(100.0), Duration::from_secs(60));
        let _ = cache.incr("spend:org:o1", 10.0, None).await;
        let mut ctx = make_ctx();
        ctx.team_id = Some("t1".to_string());
        ctx.org_id = Some("o1".to_string());
        ctx.max_budget = Some(100.0);
        ctx.team_max_budget = Some(100.0);
        ctx.org_max_budget = Some(10.0);
        let action = hook.pre_call(&mut ctx).await.expect("ok");
        assert!(
            matches!(action, HookAction::Reject(_)),
            "org ceiling alone must block"
        );
    }

    #[tokio::test]
    async fn project_budget_blocks_when_exhausted() {
        let cache = Arc::new(InMemoryCache::default());
        let hook = BudgetHook::with_duration(cache.clone(), Some(100.0), Duration::from_secs(60));
        let _ = cache.incr("spend:project:p1", 10.0, None).await;
        let mut ctx = make_ctx();
        ctx.project_id = Some("p1".to_string());
        ctx.max_budget = Some(100.0);
        ctx.project_max_budget = Some(10.0);
        let action = hook.pre_call(&mut ctx).await.expect("ok");
        assert!(matches!(action, HookAction::Reject(_)));
    }

    #[tokio::test]
    async fn org_budget_reservation_released_on_failure() {
        let cache = Arc::new(InMemoryCache::default());
        let hook = BudgetHook::with_duration(cache.clone(), Some(100.0), Duration::from_secs(60));
        let mut ctx = make_ctx();
        ctx.org_id = Some("o1".to_string());
        ctx.org_max_budget = Some(100.0);
        ctx.request_body = std::sync::Arc::new(serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}]
        }));
        hook.pre_call(&mut ctx).await.expect("continue");
        assert!(cache.counter_value("spend:org:o1").await.unwrap_or(0.0) > 0.0);
        hook.post_failure(&ctx, &Error::Network("boom".into()))
            .await;
        let spent = cache.counter_value("spend:org:o1").await.unwrap_or(0.0);
        assert!(
            spent.abs() < 1e-9,
            "failure must release org reservation, got {spent}"
        );
    }

    #[tokio::test]
    async fn project_budget_reservation_released_on_failure() {
        let cache = Arc::new(InMemoryCache::default());
        let hook = BudgetHook::with_duration(cache.clone(), Some(100.0), Duration::from_secs(60));
        let mut ctx = make_ctx();
        ctx.project_id = Some("p1".to_string());
        ctx.project_max_budget = Some(100.0);
        ctx.request_body = std::sync::Arc::new(serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}]
        }));
        hook.pre_call(&mut ctx).await.expect("continue");
        assert!(cache.counter_value("spend:project:p1").await.unwrap_or(0.0) > 0.0);
        hook.post_failure(&ctx, &Error::Network("boom".into()))
            .await;
        let spent = cache.counter_value("spend:project:p1").await.unwrap_or(0.0);
        assert!(
            spent.abs() < 1e-9,
            "failure must release project reservation, got {spent}"
        );
    }

    #[tokio::test]
    async fn full_hierarchy_all_scopes_checked_independently() {
        // org > project > team > user > key — all ceilings set; all
        // reservations must succeed when each has room.
        let cache = Arc::new(InMemoryCache::default());
        let hook = BudgetHook::with_duration(cache.clone(), Some(100.0), Duration::from_secs(60));
        let mut ctx = make_ctx();
        ctx.team_id = Some("t1".to_string());
        ctx.org_id = Some("o1".to_string());
        ctx.project_id = Some("p1".to_string());
        ctx.user_id = Some("u1".to_string());
        ctx.max_budget = Some(100.0);
        ctx.team_max_budget = Some(100.0);
        ctx.org_max_budget = Some(100.0);
        ctx.project_max_budget = Some(100.0);
        ctx.user_max_budget = Some(100.0);
        ctx.request_body = std::sync::Arc::new(serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}]
        }));
        let action = hook.pre_call(&mut ctx).await.expect("ok");
        assert!(matches!(action, HookAction::Continue));
        // All five scopes should have a positive reservation.
        assert!(cache.counter_value("spend:abc").await.unwrap_or(0.0) > 0.0);
        assert!(cache.counter_value("spend:team:t1").await.unwrap_or(0.0) > 0.0);
        assert!(cache.counter_value("spend:org:o1").await.unwrap_or(0.0) > 0.0);
        assert!(cache.counter_value("spend:project:p1").await.unwrap_or(0.0) > 0.0);
        assert!(cache.counter_value("spend:user:u1").await.unwrap_or(0.0) > 0.0);

        // Now exhaust the org ceiling; a second call must reject and roll
        // back all earlier reservations.
        let _ = cache.incr("spend:org:o1", 100.0, None).await;
        let action = hook.pre_call(&mut ctx).await.expect("ok");
        assert!(matches!(action, HookAction::Reject(_)));
    }

    #[tokio::test]
    async fn org_budget_post_success_adjusts_to_actual() {
        let cache = Arc::new(InMemoryCache::default());
        let hook = BudgetHook::with_duration(cache.clone(), Some(100.0), Duration::from_secs(60));
        let mut ctx = make_ctx();
        ctx.org_id = Some("o1".to_string());
        ctx.org_max_budget = Some(100.0);
        ctx.request_body = std::sync::Arc::new(serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}]
        }));
        hook.pre_call(&mut ctx).await.expect("continue");
        let resp = make_response("gpt-4o", 1_000_000, 500_000);
        hook.post_success(&ctx, &resp).await;
        let spent = cache.counter_value("spend:org:o1").await.unwrap_or(0.0);
        let expected = 1_000_000.0 * 0.0000025 + 500_000.0 * 0.00001;
        assert!(
            (spent - expected).abs() < 1e-6,
            "org spend must reflect actual cost ({expected}), got {spent}"
        );
    }

    #[tokio::test]
    async fn org_budget_rolls_back_team_reservation_when_org_rejects() {
        let cache = Arc::new(InMemoryCache::default());
        let hook = BudgetHook::with_duration(cache.clone(), Some(100.0), Duration::from_secs(60));
        let _ = cache.incr("spend:org:o1", 100.0, None).await;
        let mut ctx = make_ctx();
        ctx.team_id = Some("t1".to_string());
        ctx.org_id = Some("o1".to_string());
        ctx.max_budget = Some(100.0);
        ctx.team_max_budget = Some(100.0);
        ctx.org_max_budget = Some(1.0);
        let action = hook.pre_call(&mut ctx).await.expect("ok");
        assert!(matches!(action, HookAction::Reject(_)));
        // Team reservation must have been rolled back.
        let team_spend = cache.counter_value("spend:team:t1").await.unwrap_or(0.0);
        assert!(
            team_spend.abs() < 1e-9,
            "team reservation must release when org budget rejects, got {team_spend}"
        );
    }

    // U2: When the cache backend (e.g. Redis) is unavailable, `incr_if_below`
    // returns `Err`. The BudgetHook must fail CLOSED — i.e. reject the
    // request — not silently let it through (fail-open).
    #[tokio::test]
    async fn cache_outage_rejects_request_fail_closed() {
        let cache: Arc<dyn Cache> = Arc::new(FailingCache);
        let hook = BudgetHook::with_duration(cache, Some(100.0), Duration::from_secs(60));
        let mut ctx = make_ctx();
        ctx.max_budget = Some(100.0);
        ctx.request_body = std::sync::Arc::new(serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}]
        }));
        let action = hook.pre_call(&mut ctx).await.expect("ok");
        assert!(
            matches!(action, HookAction::Reject(_)),
            "cache outage must reject the request (fail-closed), not continue"
        );
    }

    /// A zero estimate (zero-cost model) must still gate on the ceiling
    /// through the fail-closed atomic path. Previously the zero-estimate
    /// branch read `counter_value(...).unwrap_or(0.0)`, silently swallowing
    /// a backing-store outage as `0.0` and letting the request through. Now
    /// it routes through `incr_if_below`, so a cache outage surfaces as
    /// `Err` (fail-closed) even when there is nothing to reserve.
    #[tokio::test]
    async fn zero_estimate_reserve_fails_closed_on_cache_outage() {
        let cache: Arc<dyn Cache> = Arc::new(FailingCache);
        let hook = BudgetHook::with_duration(cache, Some(100.0), Duration::from_secs(60));
        let result = hook
            .reserve(
                &spend_key("abc"),
                0.0,
                100.0,
                Some(Duration::from_secs(60)),
                "max_budget",
            )
            .await;
        assert!(
            result.is_err(),
            "zero-estimate reserve must fail closed on cache outage, got {result:?}"
        );
    }

    /// HIGH (C03): the team spend counter must use the team's own
    /// `budget_duration` window, NOT the key's window. Previously the team
    /// reservation was passed the key's `ttl`, so a team configured with a
    /// shorter window than the key never reset on its intended cadence.
    /// `team_budget_ttl()` resolves independently, falling back to the key
    /// window only when the team has none.
    #[tokio::test]
    async fn team_budget_uses_team_window_not_key_window() {
        let cache = Arc::new(InMemoryCache::default());
        // Hook default window (key fallback) is 60s.
        let hook = BudgetHook::with_duration(cache.clone(), Some(100.0), Duration::from_secs(60));
        // Key window = 30d, team window = 1d. The team counter must pick the
        // team's window.
        let mut ctx = make_ctx();
        ctx.team_id = Some("t-win".to_string());
        ctx.max_budget = Some(100.0);
        ctx.budget_duration = Some("30d".to_string());
        ctx.team_max_budget = Some(100.0);
        ctx.team_budget_duration = Some("1d".to_string());
        ctx.request_body = std::sync::Arc::new(serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}]
        }));
        let key_ttl = hook.budget_ttl(&ctx);
        let team_ttl = hook.team_budget_ttl(&ctx);
        assert_eq!(
            key_ttl,
            crate::scheduler::parse_budget_duration("30d"),
            "key window must resolve from ctx.budget_duration"
        );
        assert_eq!(
            team_ttl,
            crate::scheduler::parse_budget_duration("1d"),
            "team window must resolve from ctx.team_budget_duration, not the key's 30d"
        );
        assert_ne!(
            key_ttl, team_ttl,
            "team TTL must differ from key TTL when configured independently"
        );
    }

    /// HIGH (C03): when the team has no `budget_duration` of its own, the
    /// team counter falls back to the key's window (preserving the prior
    /// behavior for teams without an explicit window).
    #[tokio::test]
    async fn team_budget_falls_back_to_key_window_when_unset() {
        let cache = Arc::new(InMemoryCache::default());
        let hook = BudgetHook::with_duration(cache, Some(100.0), Duration::from_secs(60));
        let mut ctx = make_ctx();
        ctx.budget_duration = Some("7d".to_string());
        // team_budget_duration intentionally left None.
        assert_eq!(
            hook.team_budget_ttl(&ctx),
            crate::scheduler::parse_budget_duration("7d"),
            "team TTL must fall back to the key's window when the team has none"
        );
    }

    // C31: settlement observability + post_failure cache-hit edge cases.

    // Simulates a cache that is healthy for atomic reservations (`pre_call`)
    // but fails plain `incr` writes (the `adjust`/`release` paths) — i.e. a
    // Redis outage that begins mid-request, AFTER the reservation succeeded.
    // `incr_if_below` delegates to a healthy in-memory cache so `pre_call`
    // can reserve; only the plain `incr` (adjust/release) is broken.
    struct MidFlightOutageCache {
        inner: Arc<InMemoryCache>,
    }

    #[async_trait]
    impl Cache for MidFlightOutageCache {
        async fn get(&self, key: &str) -> Option<String> {
            self.inner.get(key).await
        }
        async fn set(&self, key: &str, value: String, ttl: Option<Duration>) {
            self.inner.set(key, value, ttl).await;
        }
        async fn delete(&self, key: &str) {
            self.inner.delete(key).await;
        }
        async fn incr(&self, _key: &str, _by: f64, _ttl: Option<Duration>) -> Result<f64, Error> {
            Err(Error::Network(
                "cache backend unavailable mid-flight".into(),
            ))
        }
        async fn incr_if_below(
            &self,
            key: &str,
            delta: f64,
            ceiling: f64,
            ttl: Option<Duration>,
        ) -> Result<IncrResult, Error> {
            self.inner.incr_if_below(key, delta, ceiling, ttl).await
        }
        async fn counter_value(&self, key: &str) -> Option<f64> {
            self.inner.counter_value(key).await
        }

        async fn incr_if_below_in_window(
            &self,
            key: &str,
            delta: f64,
            ceiling: f64,
            ttl: Option<Duration>,
            expected: Option<&CounterWindow>,
        ) -> Result<WindowedIncrResult, Error> {
            self.inner
                .incr_if_below_in_window(key, delta, ceiling, ttl, expected)
                .await
        }

        async fn incr_in_window(&self, _: &str, _: f64, _: &CounterWindow) -> Result<bool, Error> {
            Err(Error::Network(
                "cache backend unavailable mid-flight".into(),
            ))
        }
    }

    /// MEDIUM (C31): when the cache backend goes down mid-request (after
    /// `pre_call` reserved successfully), the `post_success` actual-cost
    /// adjust write fails. The hook must SWALLOW the error (best-effort:
    /// failing a completed request over a tracking write is worse) and
    /// continue without panicking — the counter stays at the reservation
    /// value. The fix adds a `tracing::warn!` + bumps
    /// `oxllm_budget_settle_failed_total{op="adjust"}` so the billing
    /// inaccuracy is observable. (The metric/log are best-effort and not
    /// asserted here as there is no test recorder wired in; the behavioral
    /// contract — swallow + continue — is what is pinned.)
    #[tokio::test]
    async fn budget_adjust_logs_and_continues_on_cache_err() {
        let cache: Arc<dyn Cache> = Arc::new(MidFlightOutageCache {
            inner: Arc::new(InMemoryCache::default()),
        });
        let hook = BudgetHook::with_duration(cache.clone(), Some(100.0), Duration::from_secs(60));
        let mut ctx = make_ctx();
        ctx.request_body = std::sync::Arc::new(serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}]
        }));
        // pre_call reserves successfully (incr_if_below delegates to the
        // healthy inner cache).
        hook.pre_call(&mut ctx).await.expect("continue");
        let reserved = cache.counter_value("spend:abc").await.unwrap_or(0.0);
        assert!(reserved > 0.0, "pre_call must have reserved an estimate");

        // post_success runs the adjust; the mid-flight outage rejects the
        // delta write. Must not panic.
        let resp = make_response("gpt-4o", 1_000_000, 500_000);
        hook.post_success(&ctx, &resp).await;

        // The adjust incr was rejected by the outage, so the counter retains
        // the pre_call reservation unchanged.
        let spend = cache.counter_value("spend:abc").await.unwrap_or(0.0);
        assert!(
            (spend - reserved).abs() < 1e-9,
            "adjust failure must be swallowed (counter unchanged), got {spend} vs reserved {reserved}"
        );
    }

    /// MEDIUM (C31): the same swallowing contract covers `release` (used by
    /// `post_failure` and the cache-hit `post_success` path). A mid-flight
    /// outage must not panic and must not propagate.
    #[tokio::test]
    async fn budget_release_swallows_cache_err() {
        let cache: Arc<dyn Cache> = Arc::new(MidFlightOutageCache {
            inner: Arc::new(InMemoryCache::default()),
        });
        let hook = BudgetHook::with_duration(cache.clone(), Some(100.0), Duration::from_secs(60));
        let mut ctx = make_ctx();
        ctx.request_body = std::sync::Arc::new(serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}]
        }));
        hook.pre_call(&mut ctx).await.expect("continue");
        let reserved = cache.counter_value("spend:abc").await.unwrap_or(0.0);
        assert!(reserved > 0.0);
        // post_failure routes through release; the outage rejects the
        // decrement. Must not panic; counter is left at the reservation.
        hook.post_failure(&ctx, &Error::Network("boom".into()))
            .await;
        let spend = cache.counter_value("spend:abc").await.unwrap_or(0.0);
        assert!(
            (spend - reserved).abs() < 1e-9,
            "release failure must be swallowed (counter unchanged), got {spend} vs reserved {reserved}"
        );
    }

    /// LOW (C31): an exact-zero delta (estimate == actual) is a no-op — the
    /// short-circuit returns without touching the cache. The prior
    /// `delta.abs() < f64::EPSILON` check was a misleadingly-spelled exact-
    /// zero test; `delta == 0.0` makes the intent clear.
    #[tokio::test]
    async fn adjust_with_zero_delta_is_noop() {
        let cache = Arc::new(InMemoryCache::default());
        let hook = BudgetHook::with_duration(cache.clone(), Some(100.0), Duration::from_secs(60));
        // Seed the counter with a known value.
        let _ = cache.incr("spend:abc", 5.0, None).await;
        // estimate == actual → delta is exactly 0.0 → no incr issued.
        hook.adjust(&spend_key("abc"), 2.5, 2.5, Some(Duration::from_secs(60)))
            .await;
        assert_eq!(
            cache.counter_value("spend:abc").await,
            Some(5.0),
            "exact-zero delta must not touch the counter"
        );
    }

    /// LOW (C31): `post_failure` must release the reservation even when
    /// `ctx.cache_hit = true`. The guard that used to skip release on a
    /// cache hit is removed: if `post_failure` is reached at all (e.g. the
    /// cache-hit response later failed downstream), the reservation should
    /// be freed rather than leaking until the window TTL. The happy
    /// cache-hit path is handled by `post_success`.
    #[tokio::test]
    async fn post_failure_releases_reservation_even_when_cache_hit_set() {
        let cache = Arc::new(InMemoryCache::default());
        let hook = BudgetHook::with_duration(cache.clone(), Some(100.0), Duration::from_secs(60));
        let mut ctx = make_ctx();
        ctx.request_body = std::sync::Arc::new(serde_json::json!({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}]
        }));
        hook.pre_call(&mut ctx).await.expect("continue");
        let reserved = cache.counter_value("spend:abc").await.unwrap_or(0.0);
        assert!(reserved > 0.0, "pre_call must reserve");
        // Simulate the cache-hit response later failing downstream.
        ctx.cache_hit = true;
        hook.post_failure(&ctx, &Error::Network("downstream serialize failed".into()))
            .await;
        let spent = cache.counter_value("spend:abc").await.unwrap_or(0.0);
        assert!(
            spent.abs() < 1e-9,
            "post_failure must release the reservation even when cache_hit is set, got {spent}"
        );
    }
}
